"""E3: controllability test — can structured absorber profiles reach a RANGE of
longitudinal-shape (front_fraction) targets better than a uniform fixed-budget
absorber?

This is a CONTROLLABILITY / reachability test, NOT a final physics-performance
claim. We hold the total absorber budget A_tot = N*a0 and the layer count N fixed,
hold the GAP profile uniform (never scaled), and vary only the absorber
DISTRIBUTION across regions:
  * uniform      : a_l = a0 for all layers            (the fixed reference)
  * two-region   : front block a_front, rear solves budget            (1 free DOF)
  * three-region : blocks a1,a2, third solves budget                  (2 free DOF)

Objective per target: L = (front_fraction - target)^2, evaluated over 2-3 targets
spanning below / near / above the uniform front_fraction. For each parameterization
and target we report the best-reaching profile, its loss, params, empirical
across-seed SE, and whether it beats uniform by > 3x SE (per DECISIONS 2026-07-06:
use the empirical across-seed scatter, not the sim's var_dE column).

Forward-only, gradient-free grid scan. No AD, no SLURM, local dev stats only.
Scope guard: front_fraction ONLY; no proxy-D, no e/gamma PID, no shower_max_depth.
"""
from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tools.sim as _simmod
from tools import observables as _obs
from tools import sim as _sim

# Forward N=1000 ~185s uncontended but exceeds the shared 300s cap under node
# load; raise the cap in-process ONLY (no config.yaml edit). Same precedent as
# experiments/_proxy_*.py and experiments/e0b_ffront_slurm.py.
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

OBS = "front_fraction"
ABS_FLOOR_MM = 0.3   # physical floor: skip profiles with any region thinner than this


def _dp_with_abs(cfg, abs_profile, gap_mm, energy, n_layers):
    """DesignPoint carrying an explicit per-layer absorber profile; uniform gap."""
    base = _sim.default_design_point(cfg)
    gap_profile = tuple(float(gap_mm) for _ in range(n_layers))
    return base.__class__(a=base.a, g=float(gap_mm), energy=float(energy),
                          n_layers=int(n_layers), transverse=base.transverse,
                          particle=base.particle,
                          abs_profile=tuple(float(x) for x in abs_profile),
                          gap_profile=gap_profile)


# --------------------------------------------------------------------------- #
# Budget-conserving region maps (absorber only; Sigma a_l == N*a0 by construction)
# --------------------------------------------------------------------------- #
def uniform_profile(N, a0):
    return tuple(float(a0) for _ in range(N))


def two_region_profile(a_front, N, a0):
    k = N // 2
    a_tot = N * a0
    a_rear = (a_tot - k * a_front) / (N - k)
    if a_front < ABS_FLOOR_MM or a_rear < ABS_FLOOR_MM:
        return None
    prof = [a_front] * k + [a_rear] * (N - k)
    assert abs(sum(prof) - a_tot) < 1e-6
    return tuple(float(x) for x in prof), {"a_front": float(a_front),
                                           "a_rear": float(a_rear), "k": k}


def three_region_profile(a1, a2, N, a0):
    s1, s2 = N // 3, 2 * (N // 3)
    n1, n2, n3 = s1, s2 - s1, N - s2
    a_tot = N * a0
    a3 = (a_tot - n1 * a1 - n2 * a2) / n3
    if min(a1, a2, a3) < ABS_FLOOR_MM:
        return None
    prof = [a1] * n1 + [a2] * n2 + [a3] * n3
    assert abs(sum(prof) - a_tot) < 1e-6
    return tuple(float(x) for x in prof), {"a1": float(a1), "a2": float(a2),
                                           "a3": float(a3), "splits": (s1, s2)}


# --------------------------------------------------------------------------- #
# Measurement: per-seed front_fraction -> empirical across-seed mean + SE
# --------------------------------------------------------------------------- #
def _ff_per_seed(cfg, ctrl, abs_profile, gap_mm, energy, n_layers, n_events, seeds):
    vals = []
    for s in seeds:
        dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, n_layers)
        o = _obs.observe(OBS, dp, n_events=n_events, seeds=[s], ctrl=ctrl)
        vals.append(float(o.value))
    vals = np.array(vals)
    n_bad = int(np.sum(~np.isfinite(vals)))
    if n_bad:
        # a failed/timed-out run poisons the mean; surface it, don't hide it.
        print(f"  [WARN] {n_bad}/{len(seeds)} seeds NaN for profile "
              f"{tuple(round(x, 3) for x in abs_profile[:3])}...")
    mean = float(np.nanmean(vals)) if n_bad < len(vals) else float("nan")
    good = vals[np.isfinite(vals)]
    se = (float(np.std(good, ddof=1) / np.sqrt(len(good)))
          if len(good) > 1 else float("nan"))
    return mean, se, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=50)
    ap.add_argument("--a0", type=float, default=2.3)
    ap.add_argument("--gap", type=float, default=5.7)
    ap.add_argument("--energy", type=float, default=10000.0)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--grid", type=int, default=3,
                    help="grid points per free DOF (fractions of a0 in [--fmin,--fmax])")
    ap.add_argument("--fmin", type=float, default=0.6)
    ap.add_argument("--fmax", type=float, default=1.4)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="target offset from uniform front_fraction (below/near/above)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0
    fracs = np.linspace(args.fmin, args.fmax, args.grid)

    # ---- enumerate candidate profiles per parameterization -------------------
    # entry: (param_name, abs_profile_tuple, params_dict)
    candidates = [("uniform", uniform_profile(N, a0), {"a": float(a0)})]
    for f in fracs:
        r = two_region_profile(f * a0, N, a0)
        if r is not None:
            candidates.append(("two_region", r[0], r[1]))
    for f1, f2 in itertools.product(fracs, fracs):
        r = three_region_profile(f1 * a0, f2 * a0, N, a0)
        if r is not None:
            candidates.append(("three_region", r[0], r[1]))

    print(f"# E3 controllability scan  N={N} a0={a0} A_tot={N*a0:.1f}mm  "
          f"gap={args.gap}(uniform)  E={args.energy}  N_ev={args.n_events} "
          f"seeds={args.seeds}")
    print(f"# candidates: {sum(1 for c in candidates if c[0]=='uniform')} uniform, "
          f"{sum(1 for c in candidates if c[0]=='two_region')} two-region, "
          f"{sum(1 for c in candidates if c[0]=='three_region')} three-region")

    # ---- measure every candidate (thread-pooled; each obs is its own subprocess)
    def _measure(item):
        name, prof, params = item
        mean, se, _ = _ff_per_seed(cfg, ctrl, prof, args.gap, args.energy, N,
                                   args.n_events, args.seeds)
        return name, params, mean, se

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, params, mean, se in ex.map(_measure, candidates):
            results.append({"param": name, "params": params, "ff": mean, "se": se})

    # ---- uniform reference + targets ----------------------------------------
    uni = next(r for r in results if r["param"] == "uniform")
    ff_u, se_u = uni["ff"], uni["se"]
    targets = {"below": ff_u - args.delta, "near": ff_u, "above": ff_u + args.delta}
    print(f"\n# uniform front_fraction = {ff_u:.4f} (empirical SE {se_u:.4f})")
    print(f"# targets: below={targets['below']:.4f} near={targets['near']:.4f} "
          f"above={targets['above']:.4f}  (delta={args.delta})")

    # ---- ff range reachable per parameterization ----------------------------
    print("\n# reachable front_fraction range per parameterization:")
    for pname in ("uniform", "two_region", "three_region"):
        ffs = [r["ff"] for r in results if r["param"] == pname]
        if ffs:
            print(f"  {pname:12s}: [{min(ffs):.4f}, {max(ffs):.4f}]  "
                  f"({len(ffs)} feasible profiles)")

    # ---- best profile per (parameterization, target) ------------------------
    print("\n# best target-matching per parameterization "
          "(loss = (ff-target)^2):")
    hdr = ("target | param | best_ff | best_loss | dist | SE | "
           "improve_vs_uniform | >3SE?")
    print(hdr)
    for tname, tval in targets.items():
        # uniform distance to this target (the reference to beat)
        dist_u = abs(ff_u - tval)
        for pname in ("uniform", "two_region", "three_region"):
            rows = [r for r in results if r["param"] == pname]
            if not rows:
                continue
            best = min(rows, key=lambda r: (r["ff"] - tval) ** 2)
            dist = abs(best["ff"] - tval)
            loss = (best["ff"] - tval) ** 2
            improve = dist_u - dist              # >0 means closer to target than uniform
            se_b = best["se"]
            sig = (np.isfinite(se_b) and improve > 3 * se_b)
            pstr = ",".join(f"{k}={v:.3f}" for k, v in best["params"].items()
                            if isinstance(v, float))
            print(f"{tname:5s} | {pname:12s} | {best['ff']:.4f} | {loss:.2e} | "
                  f"{dist:.4f} | {se_b:.4f} | {improve:+.4f} | "
                  f"{'YES' if sig else 'no'}   [{pstr}]")


if __name__ == "__main__":
    main()

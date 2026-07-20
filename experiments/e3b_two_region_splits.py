"""E3b: is E3's two-region collapse a grid/split artifact?

E3 found the centered (k=N/2), grid=5 two-region basis could not move
front_fraction off the uniform value, while three-region could. E3b isolates the
cause by sweeping the split point k in {N/3, N/2, 2N/3} with a FINER ratio grid
(default 11) over the same [0.6, 1.4] window. Everything else is identical to E3
(targets, seeds, N, budget, uniform gap, 1400s cap, budget assertion, loud NaN).

Reuses E3's building blocks (measurement, DesignPoint, floors, uniform profile)
so numbers are directly comparable. Controllability question ONLY — not a
physics-performance claim. front_fraction ONLY; no proxy-D / e-gamma / shower_max.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Reuse the validated E3 harness (imports also set the 1400s subprocess cap).
from experiments.e3_region_scan import (
    OBS, ABS_FLOOR_MM, uniform_profile, _ff_per_seed,
)
from tools import sim as _sim


def two_region_profile_k(a_front, k, N, a0):
    """Budget-conserving two-region absorber profile with an explicit split k.

    Front block [0,k) at a_front; rear block [k,N) solves Sigma a_l = N*a0.
    Returns None if either region falls below the physical floor.
    """
    a_tot = N * a0
    a_rear = (a_tot - k * a_front) / (N - k)
    if a_front < ABS_FLOOR_MM or a_rear < ABS_FLOOR_MM:
        return None
    prof = [a_front] * k + [a_rear] * (N - k)
    assert abs(sum(prof) - a_tot) < 1e-6, "budget not conserved"
    return tuple(float(x) for x in prof), {"a_front": float(a_front),
                                           "a_rear": float(a_rear), "k": int(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=50)
    ap.add_argument("--a0", type=float, default=2.3)
    ap.add_argument("--gap", type=float, default=5.7)
    ap.add_argument("--energy", type=float, default=10000.0)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--fmin", type=float, default=0.6)
    ap.add_argument("--fmax", type=float, default=1.4)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0
    a_tot = N * a0
    fracs = np.linspace(args.fmin, args.fmax, args.grid)
    splits = [N // 3, N // 2, 2 * N // 3]

    # ---- enumerate candidates: uniform once + two-region per (k, frac) -------
    candidates = [("uniform", None, uniform_profile(N, a0), {"a": float(a0)})]
    per_split_feasible = {k: 0 for k in splits}
    for k in splits:
        for f in fracs:
            r = two_region_profile_k(f * a0, k, N, a0)
            if r is not None:
                candidates.append(("two_region", k, r[0], r[1]))
                per_split_feasible[k] += 1

    print(f"# E3b two-region split scan  N={N} a0={a0} A_tot={a_tot:.1f}mm "
          f"gap={args.gap}(uniform) E={args.energy} N_ev={args.n_events} "
          f"seeds={args.seeds}")
    print(f"# splits k={splits}  grid={args.grid} over [{args.fmin},{args.fmax}]  "
          f"feasible/split={per_split_feasible}")
    print(f"# total candidates (incl uniform): {len(candidates)}")

    # ---- measure all (thread-pooled; each obs is its own subprocess) ---------
    def _measure(item):
        name, k, prof, params = item
        mean, se, vals = _ff_per_seed(cfg, ctrl, prof, args.gap, args.energy, N,
                                      args.n_events, args.seeds)
        n_bad = int(np.sum(~np.isfinite(vals)))
        return {"param": name, "k": k, "params": params, "ff": mean,
                "se": se, "n_bad": n_bad}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_measure, candidates):
            results.append(r)

    # ---- diagnostics ---------------------------------------------------------
    total_runs = len(candidates) * len(args.seeds)
    total_bad = sum(r["n_bad"] for r in results)
    uni = next(r for r in results if r["param"] == "uniform")
    ff_u, se_u = uni["ff"], uni["se"]
    targets = {"below": ff_u - args.delta, "near": ff_u, "above": ff_u + args.delta}

    print("\n# ---- DIAGNOSTICS ----")
    print(f"# 1. total forward runs attempted: {total_runs} "
          f"({len(candidates)} profiles x {len(args.seeds)} seeds)")
    print(f"# 2. timeouts/NaNs: {total_bad}")
    print(f"# 3. budget-conservation violations: 0 (asserted per profile)")
    print(f"# 4. uniform front_fraction = {ff_u:.4f} (empirical SE {se_u:.4f})")
    print(f"# 5. reachable ff range per split:")
    for k in splits:
        ffs = [r["ff"] for r in results if r["param"] == "two_region" and r["k"] == k]
        if ffs:
            print(f"#      k={k:2d}: [{min(ffs):.4f}, {max(ffs):.4f}]  "
                  f"({len(ffs)} profiles)")
    print(f"# targets: below={targets['below']:.4f} near={targets['near']:.4f} "
          f"above={targets['above']:.4f} (delta={args.delta})")

    # ---- best target-matching per split -------------------------------------
    print("\nsplit k | grid | reachable ff range | best below loss | "
          "best near loss | best above loss | best params | verdict")
    any_artifact = False
    for k in splits:
        rows = [r for r in results if r["param"] == "two_region" and r["k"] == k]
        if not rows:
            continue
        ffs = [r["ff"] for r in rows]
        rng = f"[{min(ffs):.4f},{max(ffs):.4f}]"
        cells, verdicts, bestparams = [], [], None
        for tname, tval in targets.items():
            best = min(rows, key=lambda r: (r["ff"] - tval) ** 2)
            loss = (best["ff"] - tval) ** 2
            dist = abs(best["ff"] - tval)
            dist_u = abs(ff_u - tval)
            improve = dist_u - dist
            se_b = best["se"]
            sig = np.isfinite(se_b) and improve > 3 * se_b
            if tname in ("below", "above") and sig:
                any_artifact = True
                verdicts.append(f"{tname}:>3SE")
                bestparams = best["params"]
            cells.append(f"{loss:.2e}")
        vtxt = ("ARTIFACT (" + ",".join(verdicts) + ")") if verdicts else "no off-uniform gain"
        pstr = (",".join(f"{kk}={vv:.3f}" for kk, vv in bestparams.items()
                         if isinstance(vv, float)) if bestparams else "-")
        print(f"k={k:2d} | {args.grid} | {rng} | {cells[0]} | {cells[1]} | "
              f"{cells[2]} | {pstr} | {vtxt}")

    print(f"\n# OVERALL: {'ARTIFACT — a two-region split reaches an off-uniform target >3xSE better than uniform' if any_artifact else 'REAL — no two-region split moves ff off uniform beyond noise (2-DOF/three-region required)'}")


if __name__ == "__main__":
    main()

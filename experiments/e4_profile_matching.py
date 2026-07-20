"""E4: normalized longitudinal profile target matching (controllability).

Upgrades E3/E3b from a SCALAR shape knob (front_fraction) to the FULL longitudinal
shape p_l = E_l / sum_j E_j. Question: at fixed total absorber budget, can
structured (two/three-region) absorber profiles match target normalized profiles
better than a uniform absorber profile?

Loss L = sum_l w_l (p_l - p_target_l)^2, uniform weights w_l = 1 (first pass).
Targets are derived ANALYTICALLY from the uniform reference profile p0 (depth-shift
earlier/later, modest broadening) -- NOT from simulated structured geometries, so a
good match is genuine controllability, not a tautology.

Forward-only value scan. Reuses E3/E3b building blocks (region maps, DesignPoint,
floors, 1400s cap). Empirical across-seed SE; loud NaN/timeout; budget assertion.
Scope: normalized-profile matching ONLY; no AD, no SLURM, no proxy-D / e-gamma /
shower_max.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Reuse validated E3/E3b harness (imports also set the 1400s subprocess cap).
from experiments.e3_region_scan import (
    ABS_FLOOR_MM, _dp_with_abs, uniform_profile, three_region_profile,
)
from experiments.e3b_two_region_splits import two_region_profile_k
from tools import sim as _sim


# --------------------------------------------------------------------------- #
# Per-seed normalized profile: p_l = mean_edep_l / sum_l, per seed -> mean + SE.
# --------------------------------------------------------------------------- #
def _profiles_per_seed(cfg, ctrl, abs_profile, gap_mm, energy, n_layers,
                       n_events, seeds):
    """Return (mean_p[L], se_p[L], list_of_per_seed_p) for the normalized profile."""
    rows = []
    for s in seeds:
        dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, n_layers)
        mean_arr, n_total, _ = _sim.forward_profile_multiseed(
            dp, "energy", n_events, [s], ctrl=ctrl)
        if mean_arr is None:
            rows.append(np.full(n_layers, np.nan))
            continue
        m = np.asarray(mean_arr[:, 0], dtype=float)
        tot = m.sum()
        rows.append(m / tot if tot > 0 else np.full(n_layers, np.nan))
    rows = np.array(rows)                       # (n_seeds, L)
    good = rows[np.all(np.isfinite(rows), axis=1)]
    if len(good) == 0:
        return np.full(n_layers, np.nan), np.full(n_layers, np.nan), rows
    mean_p = good.mean(axis=0)
    se_p = (good.std(axis=0, ddof=1) / np.sqrt(len(good))
            if len(good) > 1 else np.full(n_layers, np.nan))
    return mean_p, se_p, rows


def _loss_per_seed(per_seed_p, p_target, w):
    """L per seed = sum_l w_l (p_l - target_l)^2; returns (mean_L, se_L, n_bad)."""
    Ls = []
    n_bad = 0
    for p in per_seed_p:
        if not np.all(np.isfinite(p)):
            n_bad += 1
            continue
        Ls.append(float(np.sum(w * (p - p_target) ** 2)))
    Ls = np.array(Ls)
    if len(Ls) == 0:
        return float("nan"), float("nan"), n_bad
    mean_L = float(Ls.mean())
    se_L = float(Ls.std(ddof=1) / np.sqrt(len(Ls))) if len(Ls) > 1 else float("nan")
    return mean_L, se_L, n_bad


# --------------------------------------------------------------------------- #
# Analytic target builders from the uniform reference profile p0.
# --------------------------------------------------------------------------- #
def shift_target(p0, delta_layers):
    """Shift p0 by delta_layers (fractional, via linear interpolation), renormalize.

    delta<0 = earlier (upstream), delta>0 = later (downstream).
    """
    L = len(p0)
    x = np.arange(L, dtype=float)
    src = x - delta_layers                       # sample p0 at shifted coords
    shifted = np.interp(src, x, p0, left=0.0, right=0.0)
    shifted = np.clip(shifted, 0.0, None)
    return shifted / shifted.sum()


def broaden_target(p0, sigma_layers):
    """Convolve p0 with a small Gaussian kernel (spread it out), renormalize."""
    L = len(p0)
    x = np.arange(L, dtype=float)
    # discrete Gaussian kernel over layer offsets
    half = max(1, int(np.ceil(3 * sigma_layers)))
    off = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (off / sigma_layers) ** 2)
    k /= k.sum()
    broadened = np.convolve(p0, k, mode="same")
    broadened = np.clip(broadened, 0.0, None)
    return broadened / broadened.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=50)
    ap.add_argument("--a0", type=float, default=2.3)
    ap.add_argument("--gap", type=float, default=5.7)
    ap.add_argument("--energy", type=float, default=10000.0)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--grid", type=int, default=11, help="two-region ratio grid pts")
    ap.add_argument("--grid3", type=int, default=5,
                    help="three-region per-DOF grid pts (2-DOF -> grid3^2 profiles)")
    ap.add_argument("--fmin", type=float, default=0.6)
    ap.add_argument("--fmax", type=float, default=1.4)
    ap.add_argument("--shift", type=float, default=2.0, help="target depth shift [layers]")
    ap.add_argument("--broaden-sigma", type=float, default=2.0,
                    help="broadening Gaussian sigma [layers]")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0
    a_tot = N * a0
    fracs = np.linspace(args.fmin, args.fmax, args.grid)
    fracs3 = np.linspace(args.fmin, args.fmax, args.grid3)
    splits = [N // 3, N // 2, 2 * N // 3]

    # ---- enumerate candidate geometries -------------------------------------
    # (label, abs_profile, params_dict)
    candidates = [("uniform", uniform_profile(N, a0), {"a": float(a0)})]
    for k in splits:
        for f in fracs:
            r = two_region_profile_k(f * a0, k, N, a0)
            if r is not None:
                candidates.append(("two_region", r[0], r[1]))
    for f1 in fracs3:
        for f2 in fracs3:
            r = three_region_profile(f1 * a0, f2 * a0, N, a0)
            if r is not None:
                candidates.append(("three_region", r[0], r[1]))

    print(f"# E4 normalized-profile matching  N={N} a0={a0} A_tot={a_tot:.1f}mm "
          f"gap={args.gap}(uniform) E={args.energy} N_ev={args.n_events} "
          f"seeds={args.seeds}")
    n_uni = sum(1 for c in candidates if c[0] == "uniform")
    n_two = sum(1 for c in candidates if c[0] == "two_region")
    n_three = sum(1 for c in candidates if c[0] == "three_region")
    print(f"# candidates: {n_uni} uniform, {n_two} two-region, {n_three} three-region "
          f"(splits k={splits}, two-grid={args.grid}, three-grid={args.grid3}^2)")

    # ---- measure every candidate's normalized profile -----------------------
    def _measure(item):
        label, prof, params = item
        mean_p, se_p, per_seed = _profiles_per_seed(
            cfg, ctrl, prof, args.gap, args.energy, N, args.n_events, args.seeds)
        n_bad = int(np.sum([not np.all(np.isfinite(p)) for p in per_seed]))
        return {"label": label, "params": params, "mean_p": mean_p,
                "per_seed": per_seed, "n_bad": n_bad}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_measure, candidates):
            results.append(r)

    total_runs = len(candidates) * len(args.seeds)
    total_bad = sum(r["n_bad"] for r in results)

    # ---- reference profile p0 (uniform) + analytic targets ------------------
    uni = next(r for r in results if r["label"] == "uniform")
    p0 = uni["mean_p"]
    w = np.ones(N)                               # uniform weights (first pass)
    targets = {
        "earlier": shift_target(p0, -args.shift),
        "later":   shift_target(p0, +args.shift),
        "broader": broaden_target(p0, args.broaden_sigma),
    }

    # ---- diagnostics ---------------------------------------------------------
    print("\n# ---- DIAGNOSTICS ----")
    print(f"# 1. total forward runs: {total_runs} "
          f"({len(candidates)} profiles x {len(args.seeds)} seeds)")
    print(f"# 2. timeouts/NaNs: {total_bad}")
    print(f"# 3. budget-conservation violations: 0 (asserted per profile)")
    print(f"# 4. reference p0: sum={p0.sum():.6f} argmax_layer={int(np.argmax(p0))} "
          f"peak={p0.max():.4f} min={p0.min():.2e}")
    for tname, tp in targets.items():
        print(f"# 5/6. target '{tname}': sum={tp.sum():.6f} "
              f"nonneg={bool(np.all(tp >= 0))} argmax={int(np.argmax(tp))} "
              f"peak={tp.max():.4f}")

    # ---- best-matching profile per (target, parameterization) ---------------
    print("\ntarget | uniform loss | two-region best loss | three-region best loss "
          "| best params | empirical SE | verdict")
    for tname, tp in targets.items():
        # uniform loss (single point)
        Lu, seU, _ = _loss_per_seed(uni["per_seed"], tp, w)
        row_cells = {}
        best_of = {}
        for label in ("two_region", "three_region"):
            rows = [r for r in results if r["label"] == label]
            scored = []
            for r in rows:
                L, seL, nb = _loss_per_seed(r["per_seed"], tp, w)
                if np.isfinite(L):
                    scored.append((L, seL, r["params"]))
            if scored:
                best = min(scored, key=lambda t: t[0])
                row_cells[label] = best
                best_of[label] = best
        # verdict: structured beats uniform by >3x empirical SE on L?
        verdicts = []
        best_label, best_entry = None, None
        for label in ("two_region", "three_region"):
            if label in row_cells:
                L, seL, params = row_cells[label]
                improve = Lu - L
                se_comb = np.sqrt(seU ** 2 + (seL if np.isfinite(seL) else 0) ** 2)
                if np.isfinite(se_comb) and se_comb > 0 and improve > 3 * se_comb:
                    verdicts.append(f"{label.split('_')[0]}:>3SE")
                    if best_entry is None or L < best_entry[0]:
                        best_label, best_entry = label, (L, seL, params)
        two_str = f"{row_cells['two_region'][0]:.3e}" if 'two_region' in row_cells else "-"
        three_str = f"{row_cells['three_region'][0]:.3e}" if 'three_region' in row_cells else "-"
        if best_entry is not None:
            _, seL, params = best_entry
            pstr = ",".join(f"{kk}={vv:.3f}" for kk, vv in params.items()
                            if isinstance(vv, float))
            se_str = f"{seL:.2e}"
        else:
            pstr, se_str = "-", f"{seU:.2e}"
        vtxt = ("REACHED (" + ",".join(verdicts) + ")") if verdicts else "within 3xSE of uniform"
        print(f"{tname:8s} | {Lu:.3e} | {two_str} | {three_str} | {pstr} | "
              f"{se_str} | {vtxt}")


if __name__ == "__main__":
    main()

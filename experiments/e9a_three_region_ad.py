"""E9a: Validate the multi-region AD budget chain for three-region absorbers.

Three regions with splits (16, 33), two free thickness parameters (a1, a2),
third solved from A_tot = 115 mm.  Objective: normalized-profile matching,
target = earlier_2 (shift -2 layers), from the same construction used in E4/E5.

Validation plan
---------------
1. Derive and implement the chain: per-layer reverse-AD gradients -> dL/da1, dL/da2.
2. At the initial (uniform) point: compare both AD components with central FD
   using common random numbers.
3. If both components pass sign check: run <= 8 GD steps.
4. Assert exact budget conservation at every step.
5. Compare final loss to: (a) uniform, (b) E4 three-region grid best for earlier target.

Budget chain
------------
  regions: n1 = s1 = 16, n2 = s2 - s1 = 17, n3 = N - s2 = 17
  a_tot = N * a0 = 115 mm
  a3 = (a_tot - n1*a1 - n2*a2) / n3      <- solved exactly, no rounding

  Jacobian of (a1, a2, a3) wrt (a1, a2):
    da3/da1 = -n1 / n3
    da3/da2 = -n2 / n3

  Reverse-AD gives dL/d(abs_thick[l]) for each layer l.
  Chain rule:
    dL/da1 = sum_{l in R1} dL/d(abs_thick[l])
             + (da3/da1) * sum_{l in R3} dL/d(abs_thick[l])
           = sum_{R1} bar_l - (n1/n3) * sum_{R3} bar_l

    dL/da2 = sum_{l in R2} dL/d(abs_thick[l])
             + (da3/da2) * sum_{l in R3} dL/d(abs_thick[l])
           = sum_{R2} bar_l - (n2/n3) * sum_{R3} bar_l

Output adjoint for the normalized profile (same as E5):
    dL/dE_l = (2/S) [ (p_l - p*_l) - sum_j (p_j - p*_j) p_j ]   S = sum E
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import tools.sim as _simmod
from experiments.e3_region_scan import ABS_FLOOR_MM, _dp_with_abs, uniform_profile
from experiments.e4_profile_matching import shift_target
from tools import sim as _sim

# same 1400s cap used in E3/E4/E5
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_DEFAULT  = 50
A0_DEFAULT = 2.3
GAP_DEFAULT = 5.7
E_DEFAULT   = 10000.0
SPLITS      = (16, 33)   # s1, s2 — three regions: [0,s1), [s1,s2), [s2,N)


# ---------------------------------------------------------------------------
# Region-profile builders
# ---------------------------------------------------------------------------

def three_region_budgeted(a1, a2, N, a0, splits=SPLITS):
    """Budget-conserving three-region absorber profile from two free params.

    Returns (abs_profile tuple, params dict) or None if floor violated."""
    s1, s2 = splits
    n1, n2, n3 = s1, s2 - s1, N - s2
    a_tot = N * a0
    if not (np.isfinite(a1) and np.isfinite(a2)):
        return None
    a3 = (a_tot - n1 * a1 - n2 * a2) / n3
    if not np.isfinite(a3) or min(a1, a2, a3) < ABS_FLOOR_MM:
        return None
    prof = [a1] * n1 + [a2] * n2 + [a3] * n3
    assert abs(sum(prof) - a_tot) < 1e-6, f"budget violated: {sum(prof):.4f} != {a_tot:.4f}"
    return (tuple(float(x) for x in prof),
            {"a1": float(a1), "a2": float(a2), "a3": float(a3),
             "n1": n1, "n2": n2, "n3": n3, "s1": s1, "s2": s2})


def feasible_bounds(N, a0, splits=SPLITS):
    """Conservative rectangular bounds for (a1, a2) s.t. all regions above floor."""
    s1, s2 = splits
    n1, n2, n3 = s1, s2 - s1, N - s2
    a_tot = N * a0
    # a3 = (a_tot - n1*a1 - n2*a2) / n3 >= ABS_FLOOR
    # worst case for a3: both a1, a2 at their max -> maximally consumed
    # For a simple rectangular box: a1 in [floor, hi1], a2 in [floor, hi2]
    # where hi = (a_tot - floor*(n2+n3)) / n1 (and symmetric for a2)
    hi1 = (a_tot - ABS_FLOOR_MM * (n2 + n3)) / n1
    hi2 = (a_tot - ABS_FLOOR_MM * (n1 + n3)) / n2
    return (ABS_FLOOR_MM, hi1), (ABS_FLOOR_MM, hi2)


# ---------------------------------------------------------------------------
# Simulation helpers (per-seed; each in its own temp dir via _dp_with_abs)
# ---------------------------------------------------------------------------

def _forward_one(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seed):
    dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, N)
    mean_arr, _, _ = _sim.forward_profile_multiseed(dp, "energy", n_events, [seed], ctrl=ctrl)
    if mean_arr is None:
        return None, None
    E = np.asarray(mean_arr[:, 0], dtype=float)
    S = E.sum()
    if not (S > 0):
        return None, None
    return E, S


def per_layer_profiles(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seeds, workers=6):
    """List of per-seed normalized profiles + raw E arrays."""
    def _one(s):
        E, S = _forward_one(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, s)
        if E is None:
            return np.full(N, np.nan), np.full(N, np.nan)
        return E / S, E
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, seeds))
    return [o[0] for o in out], [o[1] for o in out]


def loss_stats(ps, p_target):
    """Mean loss + across-seed SE; count bad seeds."""
    Ls = [float(np.sum((p - p_target) ** 2)) for p in ps if np.all(np.isfinite(p))]
    n_bad = len(ps) - len(Ls)
    if not Ls:
        return float("nan"), float("nan"), n_bad
    arr = np.array(Ls)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return float(arr.mean()), se, n_bad


# ---------------------------------------------------------------------------
# AD gradient chain: per-layer reverse AD -> dL/da1, dL/da2
# ---------------------------------------------------------------------------

def _output_adjoint(E, S, p_target):
    p = E / S
    d = p - p_target
    return (2.0 / S) * (d - float(np.dot(d, p)))


def ad_grad_three_region(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seeds,
                         p_target, splits=SPLITS):
    """AD gradients dL/da1 and dL/da2 via reverse per-layer, chained through budget.

    Returns ((g1, g1_se), (g2, g2_se), n_bad).
    """
    s1, s2 = splits
    n1, n2, n3 = s1, s2 - s1, N - s2

    def _one(seed):
        E, S = _forward_one(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seed)
        if E is None:
            return None
        adj = _output_adjoint(E, S, p_target)
        dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, N)
        per = _sim.run_reverse_per_layer(dp, adj, n_events=n_events, seed=seed, ctrl=ctrl)
        if per is None:
            return None
        bar = np.asarray(per, dtype=float)[:N, 0]   # dL/d(abs_thick[l])
        # budget chain
        g1 = float(np.sum(bar[:s1]))     - (n1 / n3) * float(np.sum(bar[s2:]))
        g2 = float(np.sum(bar[s1:s2]))   - (n2 / n3) * float(np.sum(bar[s2:]))
        return g1, g2

    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        raw = list(ex.map(_one, seeds))
    valid = [r for r in raw if r is not None]
    n_bad = len(seeds) - len(valid)
    if not valid:
        return (float("nan"), float("nan")), (float("nan"), float("nan")), n_bad
    g1s = np.array([v[0] for v in valid])
    g2s = np.array([v[1] for v in valid])
    def _stats(arr):
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        return float(arr.mean()), se
    return _stats(g1s), _stats(g2s), n_bad


# ---------------------------------------------------------------------------
# FD gradient (CRN): central differences on the scalar loss
# ---------------------------------------------------------------------------

def fd_grad_three_region(cfg, ctrl, a1, a2, gap_mm, energy, N, a0, n_events, seeds,
                         p_target, splits=SPLITS, h=0.05):
    """Central finite differences for dL/da1 and dL/da2 using common random numbers."""
    def L_at(aa1, aa2):
        r = three_region_budgeted(aa1, aa2, N, a0, splits)
        if r is None:
            return float("nan")
        prof = r[0]
        ps, _ = per_layer_profiles(cfg, ctrl, prof, gap_mm, energy, N, n_events, seeds)
        L, _, _ = loss_stats(ps, p_target)
        return L

    Lp1 = L_at(a1 + h, a2);  Lm1 = L_at(a1 - h, a2)
    Lp2 = L_at(a1, a2 + h);  Lm2 = L_at(a1, a2 - h)

    fd1 = (Lp1 - Lm1) / (2 * h) if (np.isfinite(Lp1) and np.isfinite(Lm1)) else float("nan")
    fd2 = (Lp2 - Lm2) / (2 * h) if (np.isfinite(Lp2) and np.isfinite(Lm2)) else float("nan")
    return fd1, fd2


def fd_se_three_region(cfg, ctrl, a1, a2, gap_mm, energy, N, a0, n_events, seeds,
                       p_target, splits=SPLITS, h=0.05):
    """Estimate FD SE from per-seed losses at the +h and -h points."""
    def per_seed_losses(aa1, aa2):
        r = three_region_budgeted(aa1, aa2, N, a0, splits)
        if r is None:
            return None
        prof = r[0]
        ps, _ = per_layer_profiles(cfg, ctrl, prof, gap_mm, energy, N, n_events, seeds)
        return [float(np.sum((p - p_target) ** 2)) if np.all(np.isfinite(p))
                else float("nan") for p in ps]

    ls_p1 = per_seed_losses(a1 + h, a2); ls_m1 = per_seed_losses(a1 - h, a2)
    ls_p2 = per_seed_losses(a1, a2 + h); ls_m2 = per_seed_losses(a1, a2 - h)

    def _fd_se(lsp, lsm):
        if lsp is None or lsm is None:
            return float("nan")
        diffs = [(p - m) / (2 * h) for p, m in zip(lsp, lsm)
                 if np.isfinite(p) and np.isfinite(m)]
        if len(diffs) < 2:
            return float("nan")
        arr = np.array(diffs)
        return float(arr.std(ddof=1) / np.sqrt(len(arr)))

    return _fd_se(ls_p1, ls_m1), _fd_se(ls_p2, ls_m2)


# ---------------------------------------------------------------------------
# Gradient-descent optimizer (budget-conserving, <= 8 steps)
# ---------------------------------------------------------------------------

def optimize(cfg, ctrl, p_target, a1_init, a2_init, N, a0, gap_mm, energy,
             n_events, seeds, max_steps=8, da_target=0.05, max_step_mm=0.1,
             splits=SPLITS):
    s1, s2 = splits
    n1, n2, n3 = s1, s2 - s1, N - s2
    (lo1, hi1), (lo2, hi2) = feasible_bounds(N, a0, splits)

    a1, a2 = float(a1_init), float(a2_init)
    eta1, eta2 = None, None
    hist = []
    stall = 0
    prev_L = None

    print(f"\nstep | a1     | a2     | a3     | loss    | loss_SE | g1      | g2      | notes")

    for step in range(max_steps):
        r = three_region_budgeted(a1, a2, N, a0, splits)
        assert r is not None, f"infeasible at step {step}: a1={a1:.4f} a2={a2:.4f}"
        prof, pdict = r
        a3 = pdict["a3"]
        assert abs(sum(prof) - N * a0) < 1e-6, "budget not conserved"

        ps, _ = per_layer_profiles(cfg, ctrl, prof, gap_mm, energy, N, n_events, seeds)
        L, L_se, nbadL = loss_stats(ps, p_target)

        (g1, g1_se), (g2, g2_se), nbadG = ad_grad_three_region(
            cfg, ctrl, prof, gap_mm, energy, N, n_events, seeds, p_target, splits)

        note = ""
        if nbadL or nbadG:
            note += f"NaN(L={nbadL},G={nbadG}) "

        if step == 0:
            # FD validation (CRN, h=0.05 mm — in the E0 resolved window)
            fd1, fd2 = fd_grad_three_region(cfg, ctrl, a1, a2, gap_mm, energy, N, a0,
                                            n_events, seeds, p_target, splits)
            fd1_se, fd2_se = fd_se_three_region(cfg, ctrl, a1, a2, gap_mm, energy, N, a0,
                                                n_events, seeds, p_target, splits)
            r1 = g1 / fd1 if (np.isfinite(fd1) and fd1 != 0) else float("nan")
            r2 = g2 / fd2 if (np.isfinite(fd2) and fd2 != 0) else float("nan")
            # SNR thresholds: a component must be resolved (|val|/SE >= 2) to apply
            # the sign gate.  An unresolved component is flagged "UNRESOLVED" but does
            # NOT stop the run — sign comparison on noise is meaningless.
            ad1_resolved = np.isfinite(g1_se) and g1_se > 0 and abs(g1) / g1_se >= 2
            ad2_resolved = np.isfinite(g2_se) and g2_se > 0 and abs(g2) / g2_se >= 2
            fd1_resolved = np.isfinite(fd1_se) and fd1_se > 0 and abs(fd1) / fd1_se >= 2
            fd2_resolved = np.isfinite(fd2_se) and fd2_se > 0 and abs(fd2) / fd2_se >= 2
            # Sign gate fires only when BOTH AD and FD are resolved for that parameter
            s1_resolved = ad1_resolved and fd1_resolved
            s2_resolved = ad2_resolved and fd2_resolved
            s1_ok = (not s1_resolved) or (np.isfinite(r1) and r1 > 0)
            s2_ok = (not s2_resolved) or (np.isfinite(r2) and r2 > 0)
            note += f"[FD-check]"

        print(f"{step:4d} | {a1:.4f} | {a2:.4f} | {a3:.4f} | {L:.4e} | "
              f"{L_se:.2e} | {g1:+.3e} | {g2:+.3e} | {note}")

        if step == 0:
            # Surface FD table
            def _sign_label(ok, resolved, ratio):
                if not resolved:
                    return "UNRESOLVED"
                return "OK" if ok else "WRONG-SIGN"

            print(f"\n### Step-0 AD vs FD validation ###")
            print(f"param | AD        | AD_SE     | FD        | FD_SE     | AD/FD  | sign")
            print(f"a1    | {g1:+.4e} | {g1_se:.3e} | {fd1:+.4e} | {fd1_se:.3e} | "
                  f"{r1:+.4f} | {_sign_label(s1_ok, s1_resolved, r1)}"
                  f"  (AD_SNR={abs(g1)/g1_se:.1f} FD_SNR={abs(fd1)/fd1_se:.1f})")
            print(f"a2    | {g2:+.4e} | {g2_se:.3e} | {fd2:+.4e} | {fd2_se:.3e} | "
                  f"{r2:+.4f} | {_sign_label(s2_ok, s2_resolved, r2)}"
                  f"  (AD_SNR={abs(g2)/g2_se:.1f} FD_SNR={abs(fd2)/fd2_se:.1f})")
            print()
            if not s1_ok or not s2_ok:
                print("# STOP: wrong sign on at least one resolved gradient component. "
                      "Do not trust descent direction.")
                return None, s1_ok, s2_ok

            # Trust-region initial step sizes
            eta1 = da_target / (abs(g1) + 1e-30)
            eta2 = da_target / (abs(g2) + 1e-30)

        hist.append({"step": step, "a1": a1, "a2": a2, "a3": a3,
                     "loss": L, "loss_se": L_se, "g1": g1, "g2": g2})

        if prev_L is not None and L >= prev_L - 1e-12:
            stall += 1
        else:
            stall = 0
        prev_L = L
        if stall >= 3:
            print(f"# early-stop: loss stalled 3 steps")
            break

        if eta1 is None:
            break

        if not (np.isfinite(g1) and np.isfinite(g2)):
            print(f"# early-stop: NaN gradient at step {step} — all reverse seeds failed/timed out")
            break

        raw1 = eta1 * g1
        raw2 = eta2 * g2
        d1 = float(np.clip(raw1, -max_step_mm, max_step_mm))
        d2 = float(np.clip(raw2, -max_step_mm, max_step_mm))
        a1_new = float(np.clip(a1 - d1, lo1, hi1))
        a2_new = float(np.clip(a2 - d2, lo2, hi2))

        # Check the new point is feasible (a3 >= floor)
        r_new = three_region_budgeted(a1_new, a2_new, N, a0, splits)
        if r_new is None:
            # shrink step until feasible or give up
            for shrink in [0.5, 0.25, 0.1]:
                a1_try = float(np.clip(a1 - shrink * d1, lo1, hi1))
                a2_try = float(np.clip(a2 - shrink * d2, lo2, hi2))
                if three_region_budgeted(a1_try, a2_try, N, a0, splits) is not None:
                    a1_new, a2_new = a1_try, a2_try
                    break
            else:
                print(f"# early-stop: cannot find feasible step from a1={a1:.4f} a2={a2:.4f}")
                break

        if abs(a1_new - a1) < 0.005 and abs(a2_new - a2) < 0.005:
            a1, a2 = a1_new, a2_new
            print(f"# early-stop: |da|<0.005mm for both params")
            break
        a1, a2 = a1_new, a2_new

    return hist, True, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("--energy", type=float, default=E_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=2.0)
    ap.add_argument("--da-target", type=float, default=0.05)
    ap.add_argument("--max-step-mm", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=None,
                    help="write JSON result summary to this file")
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0
    splits = SPLITS

    print(f"# E9a multi-region AD budget chain validation")
    print(f"# N={N} a0={a0} A_tot={N*a0:.1f}mm gap={args.gap} E={args.energy}")
    print(f"# splits={splits} n1={splits[0]}, n2={splits[1]-splits[0]}, n3={N-splits[1]}")
    print(f"# n_events={args.n_events} seeds={args.seeds}")
    print(f"# target: earlier_2 (shift={args.shift} layers earlier)")
    print()

    # Build the target from uniform profile (same construction as E4/E5)
    uni_prof = uniform_profile(N, a0)
    ps0, _ = per_layer_profiles(cfg, ctrl, uni_prof, args.gap, args.energy, N,
                                args.n_events, args.seeds)
    p0 = np.nanmean(np.array(ps0), axis=0)
    p_target = shift_target(p0, -args.shift)

    print(f"# p0: sum={p0.sum():.6f} peak_layer={int(np.argmax(p0))}")
    print(f"# p_target: sum={p_target.sum():.6f} peak_layer={int(np.argmax(p_target))}")

    # Uniform baseline loss
    Lu, Lu_se, _ = loss_stats(ps0, p_target)
    print(f"# uniform loss = {Lu:.4e} (SE {Lu_se:.2e})")

    # Start from uniform: a1=a2=a0 (a3 solved = a0, budget conserved)
    a1_init, a2_init = a0, a0
    r = three_region_budgeted(a1_init, a2_init, N, a0, splits)
    assert r is not None
    assert abs(sum(r[0]) - N * a0) < 1e-6
    print(f"# initial point: a1={a1_init} a2={a2_init} a3={r[1]['a3']:.4f} "
          f"(uniform = {a0})")

    print()
    hist, s1_ok, s2_ok = optimize(
        cfg, ctrl, p_target, a1_init, a2_init, N, a0, args.gap, args.energy,
        args.n_events, args.seeds, args.max_steps, args.da_target,
        args.max_step_mm, splits)

    result = {
        "experiment": "E9a",
        "status": "done" if hist else "STOP_wrong_sign",
        "a0": a0, "N": N, "A_tot": N * a0, "splits": list(splits),
        "n_events": args.n_events, "seeds": args.seeds,
        "uniform_loss": Lu, "uniform_loss_se": Lu_se,
        "sign_ok_a1": s1_ok, "sign_ok_a2": s2_ok,
    }

    if hist:
        best = min(hist, key=lambda r: r["loss"])
        result["best"] = best
        result["n_steps"] = len(hist)

        print(f"\n### SUMMARY ###")
        print(f"uniform loss:           {Lu:.4e} (SE {Lu_se:.2e})")
        print(f"AD best loss:           {best['loss']:.4e} (SE {best['loss_se']:.2e}) "
              f"at a1={best['a1']:.4f} a2={best['a2']:.4f} a3={best['a3']:.4f}")
        improve = Lu - best["loss"]
        se_comb = np.sqrt(Lu_se**2 + (best["loss_se"] if np.isfinite(best["loss_se"]) else 0)**2)
        beats = np.isfinite(se_comb) and se_comb > 0 and improve > 3 * se_comb
        print(f"improvement:            {improve:+.3e} (3xSE threshold: {3*se_comb:.3e})")
        print(f"AD beats uniform >3xSE: {'YES' if beats else 'no'}")
        # E4 grid best for earlier target, three-region (from e4_profile_grid11_n1000.txt)
        e4_three_best = 3.231e-04
        print(f"E4 grid best (three-region, earlier): {e4_three_best:.3e}")
        print(f"AD final vs E4 grid best: {best['loss'] / e4_three_best:.3f}x")
        result["beats_uniform_3xSE"] = beats
        result["e4_grid_best_three_region_earlier"] = e4_three_best
    else:
        print(f"\n# STOP: sign check failed — a1 OK={s1_ok}, a2 OK={s2_ok}")
        print(f"# Do not proceed with optimization.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n# Result written to {args.out}")

    return 0 if (hist is not None) else 1


if __name__ == "__main__":
    sys.exit(main())

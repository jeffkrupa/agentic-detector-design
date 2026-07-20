"""E5: AD optimization of two-region absorber CONTRAST at a fixed split k.

Outer discrete choice: split k (fixed per run). Inner continuous variable:
a_front (a_rear solved exactly from the fixed total absorber budget A_tot=N*a0,
so budget is conserved BY CONSTRUCTION). Objective: match a target normalized
longitudinal profile p* via L = sum_l (p_l - p*_l)^2, p_l = E_l / sum_j E_j.

Gradient chain (validated pieces from E0b/E4):
  forward -> per-layer mean E_l -> p_l, L
  output adjoints  dL/dE_l = (2/S)[ (p_l - p*_l) - sum_j (p_j-p*_j) p_j ]   (S=sum E)
  reverse (run_reverse_per_layer) -> dL/da_i  (rows 0..N-1)
  budget chain: a_rear=(A_tot-k*a_front)/(N-k), da_rear/da_front=-k/(N-k)
     dL/da_front = sum_{i<k} dL/da_i - k/(N-k) * sum_{i>=k} dL/da_i

Plain GD on the single scalar a_front, parameter-trust-region step + clipping.
Step-0 AD-vs-FD sanity check (expect same sign, ~E0b 0.8 scale). Exact budget,
6 matched CRN seeds, empirical across-seed SE. No SLURM, no agentic loop.

Stage 1 (pilot): run only (earlier,k=16) and (later,k=25) via --combo.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Reuse validated E3/E4 pieces (imports also set the 1400s subprocess cap).
from experiments.e3_region_scan import ABS_FLOOR_MM, _dp_with_abs, uniform_profile
from experiments.e3b_two_region_splits import two_region_profile_k
from experiments.e4_profile_matching import shift_target, broaden_target
from tools import sim as _sim

N_DEFAULT, A0_DEFAULT, GAP_DEFAULT, E_DEFAULT = 50, 2.3, 5.7, 10000.0


def per_layer_profiles(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seeds,
                       workers=6):
    """Per-seed normalized profiles p_l (list of arrays) + per-seed raw E_l.

    Seeds run concurrently (each is its own sim subprocess in a temp dir)."""
    def _one(s):
        dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, N)
        mean_arr, _, _ = _sim.forward_profile_multiseed(dp, "energy", n_events, [s], ctrl=ctrl)
        if mean_arr is None:
            return np.full(N, np.nan), np.full(N, np.nan)
        E = np.asarray(mean_arr[:, 0], dtype=float)
        tot = E.sum()
        return (E / tot if tot > 0 else np.full(N, np.nan)), E
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, seeds))
    ps = [o[0] for o in out]
    Es = [o[1] for o in out]
    return ps, Es


def loss_stats(ps, p_target):
    """Mean loss + empirical across-seed SE over good seeds; count bad."""
    Ls = [float(np.sum((p - p_target) ** 2)) for p in ps if np.all(np.isfinite(p))]
    n_bad = len(ps) - len(Ls)
    if not Ls:
        return float("nan"), float("nan"), n_bad
    Ls = np.array(Ls)
    se = float(Ls.std(ddof=1) / np.sqrt(len(Ls))) if len(Ls) > 1 else float("nan")
    return float(Ls.mean()), se, n_bad


def ad_grad_afront(cfg, ctrl, abs_profile, gap_mm, energy, N, k, n_events, seeds,
                   p_target):
    """Reverse-AD dL/da_front (chained through the budget), averaged over seeds.

    Returns (grad_mean, grad_se, n_bad). Uses per-seed output adjoints computed
    from that seed's own profile, then reverse per-layer, then the budget chain.
    """
    coeff = k / (N - k)                     # -da_rear/da_front
    def _one(s):
        dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, N)
        mean_arr, _, _ = _sim.forward_profile_multiseed(dp, "energy", n_events, [s], ctrl=ctrl)
        if mean_arr is None:
            return None
        E = np.asarray(mean_arr[:, 0], dtype=float)
        S = E.sum()
        if not (S > 0):
            return None
        p = E / S
        d = p - p_target
        # dL/dE_l = (2/S)[ (p_l - p*_l) - sum_j (p_j-p*_j) p_j ]
        adj = (2.0 / S) * (d - float(np.sum(d * p)))
        per = _sim.run_reverse_per_layer(dp, adj, n_events=n_events, seed=s, ctrl=ctrl)
        if per is None:
            return None
        dLda = np.asarray(per, dtype=float)[:N, 0]        # rows 0..N-1 = d/d abs_i
        return float(np.sum(dLda[:k]) - coeff * np.sum(dLda[k:]))
    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        grads = [g for g in ex.map(_one, seeds) if g is not None]
    if not grads:
        return float("nan"), float("nan"), len(seeds)
    grads = np.array(grads)
    se = float(grads.std(ddof=1) / np.sqrt(len(grads))) if len(grads) > 1 else float("nan")
    return float(grads.mean()), se, len(seeds) - len(grads)


def fd_grad_afront(cfg, ctrl, a_front, gap_mm, energy, N, a0, k, n_events, seeds,
                   p_target, h=0.05):
    """Central-difference dL/da_front with common random numbers (matched seeds)."""
    def L_at(af):
        prof = two_region_profile_k(af, k, N, a0)
        if prof is None:
            return float("nan")
        ps, _ = per_layer_profiles(cfg, ctrl, prof[0], gap_mm, energy, N, n_events, seeds)
        L, _, _ = loss_stats(ps, p_target)
        return L
    Lp, Lm = L_at(a_front + h), L_at(a_front - h)
    return (Lp - Lm) / (2 * h)


def a_front_bounds(N, a0, k):
    a_tot = N * a0
    hi = (a_tot - (N - k) * ABS_FLOOR_MM) / k
    return ABS_FLOOR_MM, hi


def optimize(cfg, ctrl, target_name, p_target, k, N, a0, gap_mm, energy,
             n_events, seeds, max_steps, da_target, max_step_mm):
    lo, hi = a_front_bounds(N, a0, k)
    a_front = float(a0)                     # start from uniform (a_rear=a0 too)
    print(f"\n### RUN target={target_name} k={k}  (start uniform a_front={a0})")
    print("step | a_front | a_rear | loss | loss_SE | grad | step_size | notes")

    eta = None
    hist = []
    prev_L = None
    stall = 0
    for step in range(max_steps):
        prof = two_region_profile_k(a_front, k, N, a0)
        assert prof is not None, f"infeasible a_front={a_front}"
        abs_profile, params = prof
        a_rear = params["a_rear"]
        assert abs(sum(abs_profile) - N * a0) < 1e-6, "budget not conserved"

        ps, _ = per_layer_profiles(cfg, ctrl, abs_profile, gap_mm, energy, N, n_events, seeds)
        L, L_se, nbadL = loss_stats(ps, p_target)
        g, g_se, nbadG = ad_grad_afront(cfg, ctrl, abs_profile, gap_mm, energy, N,
                                        k, n_events, seeds, p_target)
        note = ""
        if nbadL or nbadG:
            note += f"NaN(L={nbadL},G={nbadG}) "

        # step-0 AD-vs-FD sanity gate
        if step == 0:
            fd = fd_grad_afront(cfg, ctrl, a_front, gap_mm, energy, N, a0, k,
                                n_events, seeds, p_target)
            ratio = g / fd if fd != 0 else float("nan")
            same_sign = np.isfinite(ratio) and ratio > 0
            note += f"[AD/FD check: AD={g:+.3e} FD={fd:+.3e} ratio={ratio:+.2f} " \
                    f"{'OK' if same_sign else 'WRONG-SIGN'}] "
            if not same_sign:
                print(f"{step:4d} | {a_front:.4f} | {a_rear:.4f} | {L:.4e} | "
                      f"{L_se:.2e} | {g:+.3e} | -- | {note}")
                print(f"# STOP: step-0 AD/FD wrong sign at target={target_name} k={k}; "
                      f"do not trust descent. Diagnose chain.")
                return {"target": target_name, "k": k, "status": "STOP_wrong_sign",
                        "ad": g, "fd": fd, "ratio": ratio}
            # trust-region: set eta so first move ~ da_target mm
            eta = da_target / (abs(g) + 1e-30)

        # GD step with clipping (skip update on the final iteration's readout)
        raw = eta * g if eta is not None else 0.0
        dstep = float(np.clip(raw, -max_step_mm, max_step_mm))
        step_size = dstep
        hist.append({"step": step, "a_front": a_front, "a_rear": a_rear,
                     "loss": L, "loss_se": L_se, "grad": g, "step_size": step_size})
        print(f"{step:4d} | {a_front:.4f} | {a_rear:.4f} | {L:.4e} | {L_se:.2e} | "
              f"{g:+.3e} | {step_size:+.4f} | {note}")

        # early stop
        if prev_L is not None and L >= prev_L - 1e-12:
            stall += 1
        else:
            stall = 0
        prev_L = L
        if stall >= 3:
            print(f"# early-stop: loss stalled 3 steps"); break
        a_new = float(np.clip(a_front - dstep, lo, hi))
        if abs(a_new - a_front) < 0.01:
            a_front = a_new
            print(f"# early-stop: |da|<0.01mm"); break
        a_front = a_new

    best = min(hist, key=lambda r: r["loss"])
    return {"target": target_name, "k": k, "status": "done", "hist": hist,
            "best": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True,
                    choices=["earlier_k16", "earlier_k25", "earlier_k33",
                             "later_k16", "later_k25", "later_k33"],
                    help="combo to run: {earlier|later}_k{16|25|33}")
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("--energy", type=float, default=E_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--shift", type=float, default=2.0)
    ap.add_argument("--da-target", type=float, default=0.08, help="first-step |da| [mm]")
    ap.add_argument("--max-step-mm", type=float, default=0.1)
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0

    # reference uniform profile p0 -> analytic targets (same construction as E4)
    uni_prof = uniform_profile(N, a0)
    ps0, _ = per_layer_profiles(cfg, ctrl, uni_prof, args.gap, args.energy, N,
                                args.n_events, args.seeds)
    p0 = np.nanmean(np.array(ps0), axis=0)
    targets = {"earlier": shift_target(p0, -args.shift),
               "later": shift_target(p0, +args.shift)}

    # combo "{target}_k{split}" -> (target_name, split_index)
    tname, kstr = args.combo.split("_k")
    k = int(kstr)
    assert k in (N // 3, N // 2, 2 * N // 3), f"k={k} not in {{N/3,N/2,2N/3}}"
    p_target = targets[tname]

    # uniform-loss baseline for this target
    Lu, Lu_se, _ = loss_stats(ps0, p_target)
    print(f"# E5 combo={args.combo}  N={N} a0={a0} A_tot={N*a0:.1f}mm "
          f"gap={args.gap} E={args.energy} N_ev={args.n_events} seeds={args.seeds}")
    print(f"# p0 sum={p0.sum():.4f} peak@{int(np.argmax(p0))}; "
          f"target='{tname}' sum={p_target.sum():.4f} peak@{int(np.argmax(p_target))}")
    print(f"# uniform loss (this target) = {Lu:.4e}  SE={Lu_se:.2e}")

    res = optimize(cfg, ctrl, tname, p_target, k, N, a0, args.gap, args.energy,
                   args.n_events, args.seeds, args.max_steps, args.da_target,
                   args.max_step_mm)

    if res["status"].startswith("STOP"):
        return
    best = res["best"]
    print(f"\n# SUMMARY combo={args.combo}")
    print(f"#   uniform loss          = {Lu:.4e} (SE {Lu_se:.2e})")
    print(f"#   final/best AD loss    = {best['loss']:.4e} (SE {best['loss_se']:.2e}) "
          f"at a_front={best['a_front']:.4f} a_rear={best['a_rear']:.4f}")
    improve = Lu - best["loss"]
    se_comb = np.sqrt(Lu_se ** 2 + (best["loss_se"] if np.isfinite(best["loss_se"]) else 0) ** 2)
    beats = np.isfinite(se_comb) and se_comb > 0 and improve > 3 * se_comb
    print(f"#   AD beats uniform by >3xSE? {'YES' if beats else 'no'} "
          f"(improve={improve:+.3e}, 3xSE={3*se_comb:.3e})")
    print(f"#   [same-k grid best: fill from E4/E3b logs for k={k}, target={tname}]")


if __name__ == "__main__":
    main()

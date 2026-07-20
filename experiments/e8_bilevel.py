"""E8: controlled agentic/discrete outer loop + AD inner optimization for the
energy-reconstruction FOM. DRY-RUN scope = candidates 1-2 only.

Pipeline (the bilevel plumbing this dry-run validates):
  proposer table  ->  per-candidate AD inner optimization  ->  held-out E6b eval.

DRY-RUN candidates (declared, frozen):
  1. uniform (1 region)                          -- no inner loop (0 free DOF)
  2. two-region @ k=16, front-thickened init     -- AD-optimize a_front (1 DOF)

Inner loop: E7 mean-reconstruction surrogate L=Σ_E((Ê_mean−E)/E)² with a
frozen-from-uniform ridge estimator; per-layer reverse-AD -> budget chain; step-0
AD/FD gate at an off-minimum probe; plain GD, trust-region step, ≤max_steps.
Final judge: E6b held-out per-event rel_RMSE (train[1-4] fit, test[5-8] score),
ridge on raw layer energies, energies {1,3,10,30} GeV. Reports frozen AND
refit-at-final held-out numbers. Empirical across-seed SE throughout.

Scope: candidates 1-2 ONLY; NO three-region/2-DOF; NO free-form proposer; SLURM-only.
Reuses validated E7 functions verbatim (no new estimator, no new objective).
"""
from __future__ import annotations

import argparse

import numpy as np

import tools.sim as _simmod
from experiments.e3_region_scan import uniform_profile
from experiments.e3b_two_region_splits import two_region_profile_k
from experiments.e7_ad_fom_optimize import (
    fit_frozen_estimator, _wb, surrogate_loss_and_grad, fd_grad, heldout_fom,
    fit_frozen_estimator_geom)
from tools import sim as _sim

_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_DEFAULT, A0_DEFAULT, GAP_DEFAULT = 50, 2.3, 5.7

# Proposer table (DRY-RUN: candidates 1-2 only). Each entry declares the discrete
# structure + AD-inner-loop config. Frozen before any evaluation.
PROPOSER_TABLE = [
    {"name": "uniform", "regions": 1, "k": None, "init_a_front": None},
    {"name": "two_region_k16", "regions": 2, "k": 16, "init_a_front": 2.8},  # front-thickened init
]


def _fd_with_se(cfg, ctrl, a_front, N, a0, gap, k, n_events, seeds, energies, w, b, h=0.05):
    """Central-diff dL/da_front + propagated SE (from the L_SE at each side)."""
    Lp, Lpse, *_ = surrogate_loss_and_grad(cfg, ctrl, a_front + h, N, a0, gap, k,
                                           n_events, seeds, energies, w, b)
    Lm, Lmse, *_ = surrogate_loss_and_grad(cfg, ctrl, a_front - h, N, a0, gap, k,
                                           n_events, seeds, energies, w, b)
    fd = (Lp - Lm) / (2 * h)
    se = float(np.sqrt(Lpse ** 2 + Lmse ** 2) / (2 * h)) if np.isfinite(Lpse) and np.isfinite(Lmse) else float("nan")
    return fd, se


def gate_ad_fd(cfg, ctrl, a_probe, N, a0, gap, k, n_events, seeds, energies, w, b):
    """SE-aware three-outcome AD/FD gate at one probe point.

    Returns dict {status: PASS|FLAT|FAIL, ad, ad_se, fd, fd_se, ratio}.
      * PASS  : AD & FD both RESOLVED (|x|>3·SE) AND same sign.
      * FLAT  : AD or FD unresolved (|x|<=3·SE) -> gradient too small vs noise;
                do NOT treat a wrong-sign here as a chain failure.
      * FAIL  : AD & FD both resolved but OPPOSITE sign (a genuine chain bug).
    """
    _, _, ad, ad_se, _ = surrogate_loss_and_grad(cfg, ctrl, a_probe, N, a0, gap, k,
                                                 n_events, seeds, energies, w, b)
    fd, fd_se = _fd_with_se(cfg, ctrl, a_probe, N, a0, gap, k, n_events, seeds, energies, w, b)
    ad_res = np.isfinite(ad_se) and ad_se > 0 and abs(ad) > 3 * ad_se
    fd_res = np.isfinite(fd_se) and fd_se > 0 and abs(fd) > 3 * fd_se
    ratio = ad / fd if fd else float("nan")
    if ad_res and fd_res:
        status = "PASS" if (np.isfinite(ratio) and ratio > 0) else "FAIL"
    else:
        status = "FLAT"
    return {"status": status, "ad": ad, "ad_se": ad_se, "fd": fd, "fd_se": fd_se,
            "ratio": ratio, "a_probe": a_probe}


def ad_inner_optimize(cfg, ctrl, k, N, a0, gap, n_events, seeds, energies, w, b,
                      init_a_front, max_steps, da_target, max_step_mm):
    """AD-optimize a_front at fixed split k on the E7 surrogate. Returns dict."""
    lo, hi = 0.3, (N * a0 - (N - k) * 0.3) / k

    # SE-aware AD/FD gate. Probe at RESOLVED-gradient points (away from the flat
    # minimum near a0): try a_front=2.0 then 2.6. Take the first PASS; if none pass
    # but at least one is FLAT (not FAIL), proceed cautiously (do NOT abort — the
    # surrogate is genuinely shallow, per E7). Abort ONLY on a resolved FAIL.
    probes = [a0 - 0.3, a0 + 0.3]        # 2.0 and 2.6 for a0=2.3
    gates = [gate_ad_fd(cfg, ctrl, ap, N, a0, gap, k, n_events, seeds, energies, w, b)
             for ap in probes]
    for gt in gates:
        print(f"  [gate @a_front={gt['a_probe']:.2f}] AD={gt['ad']:+.3e}"
              f"(SE{gt['ad_se']:.1e}) FD={gt['fd']:+.3e}(SE{gt['fd_se']:.1e}) "
              f"ratio={gt['ratio']:+.2f} -> {gt['status']}")
    statuses = [gt["status"] for gt in gates]
    if "FAIL" in statuses and "PASS" not in statuses:
        # a resolved opposite-sign gate = genuine chain bug -> abort
        bad = next(gt for gt in gates if gt["status"] == "FAIL")
        print(f"  !! GATE FAIL (resolved opposite-sign @a_front={bad['a_probe']:.2f}) "
              f"— chain bug, abort")
        return {"status": "GATE_FAIL", "gates": gates}
    gate_summary = "PASS" if "PASS" in statuses else "FLAT"
    print(f"  gate verdict: {gate_summary}"
          f"{' (surrogate shallow/unresolved — proceeding cautiously)' if gate_summary == 'FLAT' else ''}")

    a = float(init_a_front); eta = None; prev = None; stall = 0; traj = []
    print("  step | a_front | a_rear | surrogate_L | L_SE | grad | dstep")
    for step in range(max_steps):
        L, Lse, g, gse, nb = surrogate_loss_and_grad(
            cfg, ctrl, a, N, a0, gap, k, n_events, seeds, energies, w, b)
        a_rear = (N * a0 - k * a) / (N - k)
        if eta is None:
            eta = da_target / (abs(g) + 1e-30)
        dstep = float(np.clip(eta * g, -max_step_mm, max_step_mm))
        traj.append({"step": step, "a_front": a, "L": L, "grad": g})
        print(f"  {step:4d} | {a:.4f} | {a_rear:.4f} | {L:.4e} | {Lse:.1e} | "
              f"{g:+.3e} | {-dstep:+.4f}{' NaN'+str(nb) if nb else ''}")
        if prev is not None and L >= prev - 1e-12:
            stall += 1
        else:
            stall = 0
        prev = L
        if stall >= 3:
            print("  early-stop: surrogate stalled"); break
        a_new = float(np.clip(a - dstep, lo, hi))
        if abs(a_new - a) < 0.01:
            a = a_new; print("  early-stop: |da|<0.01mm"); break
        a = a_new
    a_star = min(traj, key=lambda r: r["L"])["a_front"]
    return {"status": "ok", "gate": gate_summary, "a_front": a_star,
            "a_rear": (N * a0 - k * a_star) / (N - k), "traj": traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--energies", type=float, nargs="+", default=[1000., 3000., 10000., 30000.])
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--da-target", type=float, default=0.08)
    ap.add_argument("--max-step-mm", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = _sim.load_config(); ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0, gap = args.n_layers, args.a0, args.gap

    print(f"# E8 DRY-RUN (candidates 1-2)  N={N} a0={a0} A_tot={N*a0:.1f} gap={gap} "
          f"N_ev={args.n_events} energies={[int(e/1000) for e in args.energies]}G")
    print(f"# train={args.train_seeds} test={args.test_seeds}")
    print(f"# proposer table: {[c['name'] for c in PROPOSER_TABLE]}")

    # frozen-from-uniform estimator (shared inner-loop readout)
    model = fit_frozen_estimator(cfg, ctrl, N, a0, gap, args.n_events,
                                 args.train_seeds, args.energies, args.ridge_lam, args.workers)
    w, b = _wb(model)
    print(f"# frozen estimator |w|={np.linalg.norm(w):.4g} b={b:+.4g}\n")

    # ---- OUTER LOOP over proposer table -------------------------------------
    geoms = {}
    for cand in PROPOSER_TABLE:
        name = cand["name"]
        print(f"## CANDIDATE: {name} (regions={cand['regions']}, k={cand['k']})")
        if cand["regions"] == 1:
            geoms[name] = {"abs_profile": uniform_profile(N, a0), "a_front": a0, "a_rear": a0}
            print("  (uniform: no inner loop, 0 free DOF)\n")
            continue
        res = ad_inner_optimize(cfg, ctrl, cand["k"], N, a0, gap, args.n_events,
                                args.train_seeds, args.energies, w, b,
                                cand["init_a_front"], args.max_steps,
                                args.da_target, args.max_step_mm)
        if res["status"] != "ok":
            print(f"  !! inner loop {res['status']} — skipping candidate\n")
            continue
        prof = two_region_profile_k(res["a_front"], cand["k"], N, a0)[0]
        assert abs(sum(prof) - N * a0) < 1e-6, "budget not conserved"
        geoms[name] = {"abs_profile": prof, "a_front": res["a_front"],
                       "a_rear": res["a_rear"], "gate": res.get("gate")}
        print(f"  -> AD optimum a_front={res['a_front']:.4f} a_rear={res['a_rear']:.4f} "
              f"(gate: {res.get('gate')})\n")

    # earlier_k16 reference geometry (known-good, a_front=2.516)
    geoms["earlier_k16_ref"] = {
        "abs_profile": two_region_profile_k(2.516, 16, N, a0)[0],
        "a_front": 2.516, "a_rear": (N * a0 - 16 * 2.516) / (N - 16)}

    # ---- FINAL HELD-OUT EVAL (E6b protocol) ---------------------------------
    print("# HELD-OUT per-event rel_RMSE (E6b protocol)")
    print("geometry | frozen rel_RMSE | SE | refit rel_RMSE | SE | worst-E(frozen)")
    ho = {}
    for name, g in geoms.items():
        prof = g["abs_profile"]
        rr_f, se_f = heldout_fom(cfg, ctrl, prof, N, gap, args.n_events,
                                 args.test_seeds, args.energies, w, b, args.workers)
        # refit-at-final: estimator refit on THIS geometry (fair geometry comparison)
        m2 = fit_frozen_estimator_geom(cfg, ctrl, prof, N, gap, args.n_events,
                                       args.train_seeds, args.energies, args.ridge_lam, args.workers)
        w2, b2 = _wb(m2)
        rr_r, se_r = heldout_fom(cfg, ctrl, prof, N, gap, args.n_events,
                                 args.test_seeds, args.energies, w2, b2, args.workers)
        # worst-energy (frozen) via per-energy eval
        we = max(heldout_fom(cfg, ctrl, prof, N, gap, args.n_events, args.test_seeds,
                             [E], w, b, args.workers)[0] for E in args.energies)
        ho[name] = {"frozen": (rr_f, se_f), "refit": (rr_r, se_r), "worstE": we,
                    "a_front": g["a_front"]}
        print(f"{name:18s} | {rr_f:.5f} | {se_f:.1e} | {rr_r:.5f} | {se_r:.1e} | {we:.5f}")

    # ---- dry-run verdict ----------------------------------------------------
    print("\n# DRY-RUN VERDICT")
    if "uniform" in ho and "two_region_k16" in ho:
        u = ho["uniform"]; k16 = ho["two_region_k16"]; ref = ho.get("earlier_k16_ref")
        for tag in ("frozen", "refit"):
            imp = u[tag][0] - k16[tag][0]; se = np.sqrt(u[tag][1]**2 + k16[tag][1]**2)
            beats = np.isfinite(se) and se > 0 and imp > 3*se
            print(f"#   AD-k16 vs uniform ({tag}): improve={imp:+.2e} (3xSE={3*se:.2e}) "
                  f"{'BEATS>3SE' if beats else 'no'}")
        print(f"#   AD-recovered a_front={k16['a_front']:.4f} (target earlier_k16≈2.516; "
              f"|Δ|={abs(k16['a_front']-2.516):.4f})")
        if ref:
            d = abs(k16['refit'][0] - ref['refit'][0])
            se = np.sqrt(k16['refit'][1]**2 + ref['refit'][1]**2)
            print(f"#   AD-k16 vs earlier_k16_ref (refit): |Δrel_RMSE|={d:.2e} "
                  f"(1xSE={se:.2e}) {'MATCH within 1xSE' if d <= se else 'differ'}")
    else:
        print("#   INCOMPLETE — a candidate failed; pipeline needs inspection.")


if __name__ == "__main__":
    main()

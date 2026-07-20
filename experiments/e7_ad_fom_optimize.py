"""E7: direct AD optimization of a two-region absorber for the energy-reco FOM.

Closes the E5/E6 gap: E5 AD-optimized a SHAPE loss; E6/E6b only EVALUATED fixed
geometries on energy reconstruction. E7 uses AD to optimize the energy-reco
objective itself (fixed split k=16, free a_front, a_rear budget-solved).

KEY LIMIT (see EXPERIMENTS.md E7): the sim's reverse-AD propagates through the MEAN
per-layer profile Ē_l(θ), not per-event spread. So E7 optimizes a differentiable
MEAN-RECONSTRUCTION surrogate (bias/calibration-like); the TRUE per-event held-out
RMSE (E6b protocol) is the honest judge. Surrogate loss and held-out RMSE are
reported SEPARATELY — surrogate improvement alone is NOT success.

Frozen-from-uniform ridge estimator (weights fit once on uniform, then frozen for
the descent). Reuses the E5 gradient chain and E6 estimator/FOM helpers.

Surrogate loss over training energies:  L(θ) = Σ_E ((Ê_mean(E,θ) − E)/E)²
  Ê_mean = w·Ē_vec + b ;  dL/da_front via per-layer reverse-AD with output adjoints
  a_l = 2((Ê_mean−E)/E²)·w_l, chained through the budget (da_rear/da_front=−k/(N−k)),
  summed over energies. Multi-seed CRN, empirical across-seed SE.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tools.sim as _simmod
from experiments.e3_region_scan import _dp_with_abs, uniform_profile
from experiments.e3b_two_region_splits import two_region_profile_k
from experiments.e6_energy_fom import (fit_ridge, pred_ridge, collect, fom,
                                       build_geometry)
from tools import sim as _sim

_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_DEFAULT, A0_DEFAULT, GAP_DEFAULT, K_DEFAULT = 50, 2.3, 5.7, 16


# --------------------------------------------------------------------------- #
# Frozen estimator: fit ridge on UNIFORM geometry (train seeds, all energies).
# --------------------------------------------------------------------------- #
def fit_frozen_estimator(cfg, ctrl, N, a0, gap, n_events, train_seeds, energies,
                         ridge_lam, workers):
    uni = uniform_profile(N, a0)
    X, y = [], []
    def _one(job):
        e, s = job
        return e, collect(cfg, ctrl, uni, gap, N, n_events, s, e)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for e, P in ex.map(_one, [(e, s) for e in energies for s in train_seeds]):
            X.append(P); y.append(np.full(P.shape[0], e))
    X = np.vstack(X); y = np.concatenate(y)
    return fit_ridge(X, y, ridge_lam)          # {"w": w_with_bias}


def _wb(model):
    """Split ridge model {'w':[w_layers.., b]} into (w_layers, b)."""
    wfull = model["w"]
    return wfull[:-1], float(wfull[-1])


# --------------------------------------------------------------------------- #
# Surrogate loss + AD gradient at a_front (mean-reconstruction, per-seed CRN).
# --------------------------------------------------------------------------- #
def surrogate_loss_and_grad(cfg, ctrl, a_front, N, a0, gap, k, n_events, seeds,
                            energies, w, b):
    """Return (L_mean, L_se, dL/da_front_mean, grad_se, n_bad) over seeds.

    Per seed: for each energy get mean profile Ē (forward), Ê_mean=w·Ē+b,
    accumulate ((Ê_mean-E)/E)^2; and the reverse-AD gradient summed over energies.
    """
    prof = two_region_profile_k(a_front, k, N, a0)
    assert prof is not None and abs(sum(prof[0]) - N * a0) < 1e-6, "budget/feasible"
    abs_profile = prof[0]
    coeff = k / (N - k)                        # -da_rear/da_front

    def _one_seed(s):
        L_s = 0.0
        g_s = 0.0
        for E in energies:
            dp = _dp_with_abs(cfg, abs_profile, gap, E, N)
            mean_arr, _, _ = _sim.forward_profile_multiseed(dp, "energy", n_events, [s], ctrl=ctrl)
            if mean_arr is None:
                return None
            Ebar = np.asarray(mean_arr[:, 0], dtype=float)      # mean per-layer E
            Ehat = float(np.dot(w, Ebar) + b)                   # Ê_mean
            resid = (Ehat - E) / E
            L_s += resid ** 2
            # dL/dEbar_l = 2 resid * (1/E) * w_l ; reverse gives d(Σ a_l Ebar_l)/da_i
            adj = 2.0 * resid * (1.0 / E) * w
            per = _sim.run_reverse_per_layer(dp, adj, n_events=n_events, seed=s, ctrl=ctrl)
            if per is None:
                return None
            dLda = np.asarray(per, dtype=float)[:N, 0]
            g_s += float(np.sum(dLda[:k]) - coeff * np.sum(dLda[k:]))
        return L_s, g_s

    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        out = [r for r in ex.map(_one_seed, seeds) if r is not None]
    n_bad = len(seeds) - len(out)
    if not out:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_bad
    Ls = np.array([o[0] for o in out]); gs = np.array([o[1] for o in out])
    se_L = float(Ls.std(ddof=1) / np.sqrt(len(Ls))) if len(Ls) > 1 else float("nan")
    se_g = float(gs.std(ddof=1) / np.sqrt(len(gs))) if len(gs) > 1 else float("nan")
    return float(Ls.mean()), se_L, float(gs.mean()), se_g, n_bad


def fd_grad(cfg, ctrl, a_front, N, a0, gap, k, n_events, seeds, energies, w, b, h=0.05):
    Lp, *_ = surrogate_loss_and_grad(cfg, ctrl, a_front + h, N, a0, gap, k, n_events, seeds, energies, w, b)
    Lm, *_ = surrogate_loss_and_grad(cfg, ctrl, a_front - h, N, a0, gap, k, n_events, seeds, energies, w, b)
    return (Lp - Lm) / (2 * h)


# --------------------------------------------------------------------------- #
# Held-out per-event FOM (E6b protocol) at a fixed geometry.
# --------------------------------------------------------------------------- #
def heldout_fom(cfg, ctrl, abs_profile, N, gap, n_events, test_seeds, energies,
                w, b, workers):
    """Per-test-seed relative RMSE (pooled over energies) -> mean + empirical SE."""
    def _one(job):
        s, = job
        yp_all, yt_all = [], []
        for E in energies:
            P = collect(cfg, ctrl, abs_profile, gap, N, n_events, s, E)
            yp = P @ w + b; yt = np.full(P.shape[0], E)
            yp_all.append(yp); yt_all.append(yt)
        m = fom(np.concatenate(yt_all), np.concatenate(yp_all))
        return m["rel_rmse"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rr = np.array(list(ex.map(_one, [(s,) for s in test_seeds])))
    se = float(rr.std(ddof=1) / np.sqrt(len(rr))) if len(rr) > 1 else float("nan")
    return float(rr.mean()), se


def _short_descent(cfg, ctrl, a_start, N, a0, gap, k, n_events, seeds, energies,
                   w, b, max_steps, da_target, max_step_mm):
    """<=max_steps GD steps on the surrogate from a_start; stop if it moves to uniform."""
    lo, hi = 0.3, (N * a0 - (N - k) * 0.3) / k
    a = float(a_start); eta = None; traj = []
    for step in range(max_steps):
        L, Lse, g, gse, nb = surrogate_loss_and_grad(
            cfg, ctrl, a, N, a0, gap, k, n_events, seeds, energies, w, b)
        traj.append({"step": step, "a_front": a, "L": L, "L_se": Lse, "grad": g})
        if eta is None:
            eta = da_target / (abs(g) + 1e-30)
        dstep = float(np.clip(eta * g, -max_step_mm, max_step_mm))
        a_new = float(np.clip(a - dstep, lo, hi))
        moving_to_uniform = (a_start < a0 and a_new > a) or (a_start > a0 and a_new < a)
        print(f"    step {step}: a_front={a:.4f} L={L:.4e}(SE{Lse:.1e}) grad={g:+.3e} "
              f"-> da={-dstep:+.4f} {'(toward uniform)' if moving_to_uniform else ''}"
              f"{' NaN'+str(nb) if nb else ''}")
        if abs(a_new - a) < 0.01:
            a = a_new; print("    early-stop |da|<0.01mm"); break
        a = a_new
    return traj, a


def run_diag(cfg, ctrl, N, a0, gap, k, args):
    seeds, test_seeds, energies = args.train_seeds, args.test_seeds, args.energies
    nev = args.n_events
    print(f"# E7 DIAGNOSTIC (bounded)  N={N} a0={a0} A_tot={N*a0:.1f} gap={gap} k={k} "
          f"N_ev={nev} energies={[int(e/1000) for e in energies]}G")
    print(f"# train={seeds} test={test_seeds} frozen-from-uniform ridge")

    model = fit_frozen_estimator(cfg, ctrl, N, a0, gap, nev, seeds, energies,
                                 args.ridge_lam, args.workers)
    w, b = _wb(model)
    print(f"# estimator |w|={np.linalg.norm(w):.4g} b={b:+.4g}")

    # (1) surrogate curve near a_front in {2.0, 2.3, 2.516, 2.6}
    print("\n# (1) SURROGATE CURVE (train seeds)")
    print("a_front | surrogate_L | L_SE | AD_grad | grad_SE")
    curve = {}
    for af in [2.0, 2.3, 2.516, 2.6]:
        L, Lse, g, gse, nb = surrogate_loss_and_grad(
            cfg, ctrl, af, N, a0, gap, k, nev, seeds, energies, w, b)
        curve[af] = {"L": L, "grad": g}
        print(f"{af:.3f}   | {L:.4e} | {Lse:.1e} | {g:+.3e} | {gse:.1e}")

    # (2) short AD descents (<=max_steps, default 5) from 2.0, 2.6, and 2.516
    ms = min(args.max_steps, 5)
    print(f"\n# (2) SHORT AD DESCENTS (<={ms} steps)")
    endpoints = {}
    for a_start in [2.0, 2.6, 2.516]:
        print(f"  descent from a_front={a_start}:")
        traj, a_end = _short_descent(cfg, ctrl, a_start, N, a0, gap, k, nev, seeds,
                                     energies, w, b, ms, args.da_target, args.max_step_mm)
        endpoints[a_start] = a_end

    # surrogate optimum from the curve (min L)
    a_surr_opt = min(curve, key=lambda af: curve[af]["L"])
    print(f"\n# surrogate optimum (of curve points) = a_front {a_surr_opt} "
          f"(L={curve[a_surr_opt]['L']:.4e})")

    # (4) held-out E6b-style per-event RMSE: uniform / earlier_k16 / E7 point
    a_e7 = a_surr_opt   # E7 mean-surrogate optimum
    geoms = {
        "uniform": uniform_profile(N, a0),
        "earlier_k16": build_geometry("earlier_k16", N, a0)[0],
        f"E7_surr_opt(a={a_e7})": two_region_profile_k(a_e7, k, N, a0)[0],
    }
    print("\n# (4) HELD-OUT per-event rel_RMSE (E6b protocol, frozen-from-uniform estimator)")
    print("geometry | held-out rel_RMSE | SE")
    ho = {}
    for name, prof in geoms.items():
        rr, se = heldout_fom(cfg, ctrl, prof, N, gap, nev, test_seeds, energies, w, b, args.workers)
        ho[name] = (rr, se); print(f"{name:22s} | {rr:.5f} | {se:.2e}")

    # (5)+(6) interpretation
    uni = ho["uniform"]; e16 = ho["earlier_k16"]
    print("\n# (5)/(6) INTERPRETATION")
    downhill = all(  # AD grad sign points toward uniform on both sides
        (curve[2.0]["grad"] < 0) and (curve[2.6]["grad"] > 0))
    print(f"#   machinery: AD moves downhill on surrogate (grad<0 below uniform, "
          f">0 above)? {'PASS' if downhill else 'CHECK'}")
    print(f"#   surrogate optimum near uniform? {'YES' if abs(a_surr_opt-a0)<=0.25 else 'no'} "
          f"(opt a_front={a_surr_opt})")
    imp16 = uni[0] - e16[0]; se16 = np.sqrt(uni[1]**2 + e16[1]**2)
    e16_beats = np.isfinite(se16) and se16 > 0 and imp16 > 3*se16
    print(f"#   earlier_k16 beats uniform on held-out RMSE >3xSE? "
          f"{'YES' if e16_beats else 'no'} (improve={imp16:+.2e}, 3xSE={3*se16:.2e})")
    print(f"#   => if surrogate opt=uniform BUT earlier_k16 wins held-out RMSE: "
          f"KILL/BOUND for direct mean-surrogate FOM opt; gain lives in "
          f"resolution/variance (AD-invisible via mean profile).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--energies", type=float, nargs="+", default=[1000., 3000., 10000., 30000.])
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--da-target", type=float, default=0.08)
    ap.add_argument("--max-step-mm", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="bounded diagnostic: surrogate curve + short (<=5-step) descents")
    args = ap.parse_args()

    cfg = _sim.load_config(); ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0, k, gap = args.n_layers, args.a0, args.k, args.gap

    if args.diag:
        run_diag(cfg, ctrl, N, a0, gap, k, args)
        return

    if args.smoke:
        e2 = [3000., 10000.]; ts = [1]; hs = [5]; nev = 200
        print(f"# E7 SMOKE k={k} energies={ [int(e/1000) for e in e2] }G N={nev} train={ts} test={hs}")
        model = fit_frozen_estimator(cfg, ctrl, N, a0, gap, nev, ts, e2, args.ridge_lam, args.workers)
        w, b = _wb(model)
        print(f"# estimator: |w|={np.linalg.norm(w):.4g} b={b:+.4g} (len w={len(w)})")
        L, Lse, g, gse, nb = surrogate_loss_and_grad(cfg, ctrl, a0, N, a0, gap, k, nev, ts, e2, w, b)
        fd = fd_grad(cfg, ctrl, a0, N, a0, gap, k, nev, ts, e2, w, b)
        ratio = g / fd if fd else float("nan")
        print(f"# surrogate L(uniform)={L:.4e} dL/da_front AD={g:+.3e} FD={fd:+.3e} "
              f"ratio={ratio:+.2f} {'SIGN-OK' if (np.isfinite(ratio) and ratio>0) else 'SIGN-BAD'}")
        uni = uniform_profile(N, a0)
        rr, rse = heldout_fom(cfg, ctrl, uni, N, gap, nev, hs, e2, w, b, args.workers)
        print(f"# held-out per-event rel_RMSE(uniform)={rr:.4f} (shapes/metric OK)")
        return

    print(f"# E7 AD-FOM optimize  N={N} a0={a0} A_tot={N*a0:.1f} gap={gap} k={k} "
          f"N_ev={args.n_events} energies={[int(e/1000) for e in args.energies]}G")
    print(f"# train={args.train_seeds} test={args.test_seeds} (frozen-from-uniform ridge)")

    # frozen estimator from uniform
    model = fit_frozen_estimator(cfg, ctrl, N, a0, gap, args.n_events,
                                 args.train_seeds, args.energies, args.ridge_lam, args.workers)
    w, b = _wb(model)
    lo, hi = 0.3, (N * a0 - (N - k) * 0.3) / k

    # ---- AD/FD machinery gate at an OFF-MINIMUM probe point -----------------
    # The surrogate is ~convex with min near uniform (a_front≈a0), so the gradient
    # at the start is ~0 and a start-point AD/FD ratio is noise-meaningless. Gate
    # the CHAIN at a_front = a0 - 0.3 (resolved gradient) instead. Expect same sign.
    a_probe = a0 - 0.3
    _, _, g_p, _, _ = surrogate_loss_and_grad(cfg, ctrl, a_probe, N, a0, gap, k,
                                              args.n_events, args.train_seeds, args.energies, w, b)
    fd_p = fd_grad(cfg, ctrl, a_probe, N, a0, gap, k, args.n_events,
                   args.train_seeds, args.energies, w, b)
    ratio_p = g_p / fd_p if fd_p else float("nan")
    ok_p = np.isfinite(ratio_p) and ratio_p > 0
    print(f"\n# AD/FD gate @a_front={a_probe:.2f} (off-minimum): AD={g_p:+.3e} "
          f"FD={fd_p:+.3e} ratio={ratio_p:+.2f} {'OK' if ok_p else 'BAD'}")
    if not ok_p:
        print("# STOP: AD/FD wrong sign at resolved probe point — chain bug.")
        return

    # ---- AD/GD descent on the surrogate (train seeds), start from uniform ----
    a_front = float(a0); eta = None; prev = None; stall = 0; hist = []
    print("\n# DESCENT (surrogate loss, train seeds, start=uniform)")
    print("step | a_front | a_rear | surrogate_L | L_SE | grad | step | notes")
    for step in range(args.max_steps):
        L, Lse, g, gse, nb = surrogate_loss_and_grad(
            cfg, ctrl, a_front, N, a0, gap, k, args.n_events, args.train_seeds,
            args.energies, w, b)
        a_rear = (N * a0 - k * a_front) / (N - k)
        note = f"NaN({nb})" if nb else ""
        if step == 0:
            eta = args.da_target / (abs(g) + 1e-30)
        dstep = float(np.clip(eta * g, -args.max_step_mm, args.max_step_mm))
        hist.append({"step": step, "a_front": a_front, "a_rear": a_rear, "L": L, "L_se": Lse})
        print(f"{step:4d} | {a_front:.4f} | {a_rear:.4f} | {L:.4e} | {Lse:.1e} | {g:+.3e} | {dstep:+.4f} | {note}")
        if prev is not None and L >= prev - 1e-12:
            stall += 1
        else:
            stall = 0
        prev = L
        if stall >= 3:
            print("# early-stop: surrogate stalled 3 steps"); break
        a_new = float(np.clip(a_front - dstep, lo, hi))
        if abs(a_new - a_front) < 0.01:
            a_front = a_new; print("# early-stop: |da|<0.01mm"); break
        a_front = a_new

    a_star = min(hist, key=lambda r: r["L"])["a_front"]
    print(f"\n# converged a_front*={a_star:.4f} (a_rear*={(N*a0-k*a_star)/(N-k):.4f})")

    # ---- held-out per-event FOM (E6b protocol): the HONEST judge ------------
    geoms = {
        "uniform": uniform_profile(N, a0),
        "earlier_k16": build_geometry("earlier_k16", N, a0)[0],
        "E7_ADopt": two_region_profile_k(a_star, k, N, a0)[0],
    }
    print("\n# HELD-OUT per-event rel_RMSE (E6b protocol, frozen-from-uniform estimator)")
    print("geometry | held-out rel_RMSE | SE")
    ho = {}
    for name, prof in geoms.items():
        rr, se = heldout_fom(cfg, ctrl, prof, N, gap, args.n_events, args.test_seeds,
                             args.energies, w, b, args.workers)
        ho[name] = (rr, se)
        print(f"{name:12s} | {rr:.5f} | {se:.2e}")

    # refit-at-final diagnostic (fairer to E7): estimator refit on E7 geometry
    mE7 = fit_frozen_estimator_geom(cfg, ctrl, geoms["E7_ADopt"], N, gap,
                                    args.n_events, args.train_seeds, args.energies,
                                    args.ridge_lam, args.workers)
    wE7, bE7 = _wb(mE7)
    rr_refit, se_refit = heldout_fom(cfg, ctrl, geoms["E7_ADopt"], N, gap,
                                     args.n_events, args.test_seeds, args.energies,
                                     wE7, bE7, args.workers)
    print(f"# [diagnostic] E7_ADopt refit-at-final estimator: rel_RMSE={rr_refit:.5f} SE={se_refit:.2e}")

    # ---- verdict (SEPARATE from surrogate; per-event held-out is the judge) --
    u, e16, e7 = ho["uniform"], ho["earlier_k16"], ho["E7_ADopt"]
    imp_u = u[0] - e7[0]; se_u = np.sqrt(u[1] ** 2 + e7[1] ** 2)
    near16 = abs(e7[0] - e16[0]) <= np.sqrt(e16[1] ** 2 + e7[1] ** 2)
    print("\n# VERDICT (held-out per-event FOM — the honest judge, NOT the surrogate)")
    print(f"#   E7 vs uniform: improve={imp_u:+.2e} (3xSE={3*se_u:.2e}) "
          f"{'BEATS>3SE' if (np.isfinite(se_u) and se_u>0 and imp_u>3*se_u) else 'no'}")
    print(f"#   E7 vs earlier_k16: within 1xSE? {'YES' if near16 else 'no'} "
          f"(E7={e7[0]:.5f} vs earlier_k16={e16[0]:.5f})")


def fit_frozen_estimator_geom(cfg, ctrl, abs_profile, N, gap, n_events, train_seeds,
                              energies, ridge_lam, workers):
    """Ridge fit on an ARBITRARY geometry (for the refit-at-final diagnostic)."""
    X, y = [], []
    def _one(job):
        e, s = job
        return e, collect(cfg, ctrl, abs_profile, gap, N, n_events, s, e)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for e, P in ex.map(_one, [(e, s) for e in energies for s in train_seeds]):
            X.append(P); y.append(np.full(P.shape[0], e))
    return fit_ridge(np.vstack(X), np.concatenate(y), ridge_lam)


if __name__ == "__main__":
    main()

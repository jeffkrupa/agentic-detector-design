"""E6: energy-reconstruction figure of merit for structured absorber geometries.

Promotes from shape controllability (E3-E5) to a REAL detector FOM: how well can
we reconstruct the incident beam energy from per-event layer energies, and does a
structured (two-region) absorber profile beat a uniform one at FIXED total absorber
budget?

CRITICAL vs E3-E5: keep the ENERGY SCALE. Features/target use RAW per-event layer
energies E_l [MeV] (via depth_resolution._forward_per_event_profiles), NOT the
normalized p_l. Total visible energy ΣE_l is kept as a feature.

Discipline: train seeds fit the estimator (calibration/regression), disjoint
held-out seeds score the FOM. Geometries are FIXED IN ADVANCE (declared below,
from E4/E5) — never chosen on test data. No AD, no SLURM, no agentic loop.

Declared geometries (fixed budget A_tot=N*a0=115mm, uniform gap, k=16):
  uniform      : a=2.3
  earlier      : a_front=2.516 a_rear=2.198  (E5 earlier/k16 optimum)
  later        : a_front=2.042 a_rear=2.421  (E5 later/k16 optimum)
Energies: {3, 10, 30} GeV.

Estimators: (1) total-visible-E calibration Ê=c*ΣE; (2) ridge on raw layer E;
(3) optional total-E + normalized-profile features.
Primary FOM: held-out relative RMSE. Also: bias, resolution, worst-energy error.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.depth_resolution import _forward_per_event_profiles
from experiments.e3_region_scan import _dp_with_abs, uniform_profile
from experiments.e3b_two_region_splits import two_region_profile_k
from tools import sim as _sim
import tools.sim as _simmod

# N=1000 forward ~185s; raise cap above shared 300s under load (in-process only).
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_DEFAULT, A0_DEFAULT, GAP_DEFAULT = 50, 2.3, 5.7
ENERGIES_MEV = [3000.0, 10000.0, 30000.0]

# Declared-in-advance geometries (budget-conserving). Built at runtime so the
# exact profile arrays come from the same region maps E3/E5 used. a_front values
# are the E5 per-(target,k) optima. E6 used earlier/later (k=16 aliases); E6b uses
# the explicit *_k16/*_k25 names (declared in advance, frozen before eval).
GEOMETRIES = {
    "uniform": None,                            # filled with uniform_profile
    "earlier": ("two_region", 2.516, 16),       # E6 alias = earlier_k16
    "later":   ("two_region", 2.042, 16),       # E6 alias = later_k16
    "earlier_k16": ("two_region", 2.516, 16),
    "later_k16":   ("two_region", 2.042, 16),
    "earlier_k25": ("two_region", 2.480, 25),
    "later_k25":   ("two_region", 2.124, 25),
}


def build_geometry(name, N, a0):
    if name == "uniform":
        return uniform_profile(N, a0), {"a": a0}
    _, a_front, k = GEOMETRIES[name]
    prof = two_region_profile_k(a_front, k, N, a0)
    assert prof is not None, f"{name} infeasible"
    assert abs(sum(prof[0]) - N * a0) < 1e-6, "budget not conserved"
    return prof[0], prof[1]


def collect(cfg, ctrl, abs_profile, gap_mm, N, n_events, seed, energy):
    """Per-event (n_events, N) raw layer energies at one (geometry, seed, energy)."""
    dp = _dp_with_abs(cfg, abs_profile, gap_mm, energy, N)
    return _forward_per_event_profiles(cfg, ctrl, dp, n_events, seed, N)


# --------------------------------------------------------------------------- #
# Estimators. Each returns per-event predicted energy Ê given a feature matrix.
# Fit on TRAIN (X, y), predict on TEST X. y = true beam energy per event.
# --------------------------------------------------------------------------- #
def fit_calib(X_layers, y):
    """Ê = c * ΣE_l ; c fit by least squares on train (scalar gain)."""
    s = X_layers.sum(axis=1)
    c = float(np.dot(s, y) / np.dot(s, s))
    return {"c": c}


def pred_calib(model, X_layers):
    return model["c"] * X_layers.sum(axis=1)


def fit_ridge(X, y, lam):
    """Ê = X w + b ; ridge (bias unpenalized) fit on train."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    d = Xa.shape[1]
    R = lam * np.eye(d); R[-1, -1] = 0.0            # don't penalize bias
    w = np.linalg.solve(Xa.T @ Xa + R, Xa.T @ y)
    return {"w": w}


def pred_ridge(model, X):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xa @ model["w"]


def features_layers(P):
    """Raw per-event layer energies (n_events, N)."""
    return P


def features_total_plus_shape(P):
    """[ total visible E , normalized profile p_l ]  (keeps scale + shape)."""
    tot = P.sum(axis=1, keepdims=True)
    p = P / np.clip(tot, 1e-12, None)
    return np.hstack([tot, p])


# --------------------------------------------------------------------------- #
# Metrics on held-out data.
# --------------------------------------------------------------------------- #
def fom(y_true, y_pred):
    """Return dict of relative-RMSE, bias, resolution, per the pooled test set."""
    r = y_pred / y_true
    rel_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2 / y_true ** 2)))
    bias = float(np.mean(r) - 1.0)
    reso = float(np.std(r))
    return {"rel_rmse": rel_rmse, "bias": bias, "reso": reso}


def run_study(cfg, ctrl, N, a0, gap, n_events, train_seeds, test_seeds,
              energies, geom_names, ridge_lam, workers, estimator):
    # ---- collect per-event data for every (geometry, seed, energy) ----------
    geoms = {g: build_geometry(g, N, a0) for g in geom_names}
    jobs = []
    for g in geom_names:
        for e in energies:
            for s in list(train_seeds) + list(test_seeds):
                jobs.append((g, e, s))

    def _run(job):
        g, e, s = job
        P = collect(cfg, ctrl, geoms[g][0], gap, N, n_events, s, e)
        return (g, e, s), P
    data = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, P in ex.map(_run, jobs):
            data[key] = P

    feat_fn = {"layers": features_layers,
               "total_shape": features_total_plus_shape}[estimator["feat"]]

    # ---- per geometry: fit on train seeds, score on each test seed ----------
    results = {}
    for g in geom_names:
        # assemble train matrix (pool train seeds x energies)
        Xtr, ytr = [], []
        for e in energies:
            for s in train_seeds:
                P = data[(g, e, s)]
                Xtr.append(feat_fn(P)); ytr.append(np.full(P.shape[0], e))
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        if estimator["kind"] == "calib":
            model = fit_calib(Xtr if estimator["feat"] == "layers" else
                              np.vstack([data[(g, e, s)] for e in energies for s in train_seeds]),
                              ytr)
            predf = lambda X: pred_calib(model, X)
            # calib always uses raw layers for ΣE
            def score_feat(P): return P
        else:
            model = fit_ridge(Xtr, ytr, ridge_lam)
            predf = lambda X: pred_ridge(model, X)
            score_feat = feat_fn

        # per-test-seed FOM (so we get empirical across-seed SE), per energy too
        per_seed = []
        per_energy = {e: [] for e in energies}
        for s in test_seeds:
            yp_all, yt_all = [], []
            for e in energies:
                P = data[(g, e, s)]
                yp = predf(score_feat(P)); yt = np.full(P.shape[0], e)
                yp_all.append(yp); yt_all.append(yt)
                per_energy[e].append(fom(yt, yp))
            m = fom(np.concatenate(yt_all), np.concatenate(yp_all))
            per_seed.append(m)
        rr = np.array([m["rel_rmse"] for m in per_seed])
        results[g] = {
            "rel_rmse_mean": float(rr.mean()),
            "rel_rmse_se": float(rr.std(ddof=1) / np.sqrt(len(rr))) if len(rr) > 1 else float("nan"),
            "per_energy": {e: {"rel_rmse": float(np.mean([d["rel_rmse"] for d in per_energy[e]])),
                               "bias": float(np.mean([d["bias"] for d in per_energy[e]])),
                               "reso": float(np.mean([d["reso"] for d in per_energy[e]]))}
                           for e in energies},
            "worst_energy_rel_rmse": float(max(np.mean([d["rel_rmse"] for d in per_energy[e]])
                                               for e in energies)),
        }
    return results


def report(results, energies, geom_names, estimator_label):
    print(f"\n## Estimator: {estimator_label}")
    print("geom | held-out rel_RMSE | SE | worst-E rel_RMSE | per-E (rel_rmse,bias,reso)")
    for g in geom_names:
        r = results[g]
        pe = "  ".join(f"{int(e/1000)}G:({r['per_energy'][e]['rel_rmse']:.4f},"
                       f"{r['per_energy'][e]['bias']:+.4f},{r['per_energy'][e]['reso']:.4f})"
                       for e in energies)
        print(f"{g:8s} | {r['rel_rmse_mean']:.5f} | {r['rel_rmse_se']:.2e} | "
              f"{r['worst_energy_rel_rmse']:.5f} | {pe}")
    # verdict vs uniform
    u = results["uniform"]
    print("# verdict (structured vs uniform, held-out):")
    for g in geom_names:
        if g == "uniform":
            continue
        r = results[g]
        imp = u["rel_rmse_mean"] - r["rel_rmse_mean"]
        se = np.sqrt(u["rel_rmse_se"] ** 2 + r["rel_rmse_se"] ** 2)
        beats = np.isfinite(se) and se > 0 and imp > 3 * se
        impw = u["worst_energy_rel_rmse"] - r["worst_energy_rel_rmse"]
        print(f"#   {g}: rel_RMSE improve={imp:+.2e} (3xSE={3*se:.2e}) "
              f"{'BEATS>3SE' if beats else 'no'}; worst-E improve={impw:+.2e}")


def report_e6b(results, energies, geom_names):
    """E6b: ridge-only, per-energy sign-consistency check vs uniform."""
    u = results["uniform"]
    print("\n## E6b ridge-on-raw-layer-E — held-out")
    print("geometry | rel_RMSE | SE | worst-E | per-E rel_RMSE(bias,reso) | "
          "vs uniform | verdict")
    for g in geom_names:
        r = results[g]
        pe = " ".join(f"{int(e/1000)}G:{r['per_energy'][e]['rel_rmse']:.4f}"
                      f"({r['per_energy'][e]['bias']:+.3f},{r['per_energy'][e]['reso']:.4f})"
                      for e in energies)
        if g == "uniform":
            print(f"{g:12s} | {r['rel_rmse_mean']:.5f} | {r['rel_rmse_se']:.2e} | "
                  f"{r['worst_energy_rel_rmse']:.5f} | {pe} | — | reference")
            continue
        imp = u["rel_rmse_mean"] - r["rel_rmse_mean"]
        se = np.sqrt(u["rel_rmse_se"] ** 2 + r["rel_rmse_se"] ** 2)
        beats_rr = np.isfinite(se) and se > 0 and imp > 3 * se
        impw = u["worst_energy_rel_rmse"] - r["worst_energy_rel_rmse"]
        # per-energy: how many bins does structured beat uniform (rel_rmse lower)?
        wins = [e for e in energies
                if r["per_energy"][e]["rel_rmse"] < u["per_energy"][e]["rel_rmse"]]
        consistent = len(wins) >= 2
        # E6b PASS: >3xSE on rel_RMSE OR worst-E, AND win in >=2 of 4 bins
        beats_any = beats_rr or (impw > 0 and imp > 0)  # worst-E improved + overall improved
        verdict = ("PASS" if (beats_rr and consistent) else
                   "fragile" if (imp > 0 and consistent) else
                   "fragile(1-bin)" if (imp > 0 and not consistent) else
                   "no-gain")
        print(f"{g:12s} | {r['rel_rmse_mean']:.5f} | {r['rel_rmse_se']:.2e} | "
              f"{r['worst_energy_rel_rmse']:.5f} | {pe} | "
              f"rr{imp:+.2e}(3SE{3*se:.1e}){'>' if beats_rr else '≤'} "
              f"wE{impw:+.2e} bins{len(wins)}/4 | {verdict}")
    print("# PASS = >3xSE on rel_RMSE AND wins >=2/4 energy bins; "
          "fragile = improves but <3xSE or 1-bin; no-gain = uniform matches/beats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--energies", type=float, nargs="+", default=ENERGIES_MEV)
    ap.add_argument("--geoms", nargs="+", default=["uniform", "earlier", "later"])
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="1 geom, 1 energy, 1 train + 1 test seed; validate shapes")
    ap.add_argument("--e6b", action="store_true",
                    help="E6b robustness: ridge-on-layers ONLY + per-energy sign check")
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0 = args.n_layers, args.a0

    if args.smoke:
        print("# E6 SMOKE: uniform, 10 GeV, train=[1] test=[5]")
        res = run_study(cfg, ctrl, N, a0, args.gap, args.n_events, [1], [5],
                        [10000.0], ["uniform"], args.ridge_lam, args.workers,
                        {"kind": "ridge", "feat": "layers"})
        u = res["uniform"]
        print(f"# shapes OK. uniform held-out rel_RMSE={u['rel_rmse_mean']:.5f} "
              f"reso@10G={u['per_energy'][10000.0]['reso']:.4f} "
              f"bias@10G={u['per_energy'][10000.0]['bias']:+.4f}")
        return

    print(f"# E6 energy-reco FOM  N={N} a0={a0} A_tot={N*a0:.1f}mm gap={args.gap} "
          f"N_ev={args.n_events}")
    print(f"# train_seeds={args.train_seeds} test_seeds={args.test_seeds} "
          f"energies={[int(e/1000) for e in args.energies]}GeV geoms={args.geoms}")

    if args.e6b:
        # E6b: ridge-on-raw-layer-E ONLY, per-energy sign-consistency check.
        res = run_study(cfg, ctrl, N, a0, args.gap, args.n_events, args.train_seeds,
                        args.test_seeds, args.energies, args.geoms, args.ridge_lam,
                        args.workers, {"kind": "ridge", "feat": "layers"})
        report_e6b(res, args.energies, args.geoms)
        return

    for est in ({"kind": "calib", "feat": "layers", "label": "(1) total-E calibration Ê=c·ΣE"},
                {"kind": "ridge", "feat": "layers", "label": "(2) ridge on raw layer E"},
                {"kind": "ridge", "feat": "total_shape", "label": "(3) ridge on [totalE, norm-profile]"}):
        res = run_study(cfg, ctrl, N, a0, args.gap, args.n_events, args.train_seeds,
                        args.test_seeds, args.energies, args.geoms, args.ridge_lam,
                        args.workers, est)
        report(res, args.energies, args.geoms, est["label"])


if __name__ == "__main__":
    main()

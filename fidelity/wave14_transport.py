"""Wave-14: transportability of the canonical-severed absorber core undershoot.

Question: is the severed-absorber core-window (L5-18) AD/FD undershoot
(~0.77 on-design, a=2.3) a geometry-INDEPENDENT constant -- so a single blind
1/0.766 = 1.30x correction transports -- or does it drift with absorber
thickness (needs a scaling rule c(a))?

Design point is the uniform default (-l 50 -g 5.7 -e 10000 -t 400, e-),
canonical severed flags -x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000 (paper published
strategy). We vary only the absorber thickness a in {2.0, 2.3, 3.0} mm.

AD (forward, absorber seed -a a:1):
  a=2.3  : experiments/perlayer_adfd/absorber_ad_s*.jsonl  (46 seeds x 20k) REUSE
  a=2.0  : fidelity/wave14_runs/absorber_ad_a20_s*.jsonl   (this script --run)
  a=3.0  : fidelity/wave14_runs/absorber_ad_a30_s*.jsonl   (this script --run)

FD truth (central diff, DO NOT recompute the arms):
  a=2.3  : experiments/perlayer_adfd_1M_summary.json  params.absorber.per_layer.fd
  a=2.0  : fidelity/wave13_runs/abs_fd_a20_{plus,minus}_s*.jsonl  (a=2.1/1.9, h=0.1)
  a=3.0  : fidelity/wave11_runs/abs_fd_od_{plus,minus}_s*.jsonl   (a=3.1/2.9, h=0.1)
  FD arms are UNPAIRED (seeded 'energy', primal columns are severing-independent
  and byte-identical across builds); per-layer FD = (E+ - E-) / (2h), seed-scatter
  errors keep inter-layer covariance on the window sum.

All sim calls go through tools.sim.run_forward (temp-dir isolation + flag-hash
cache). The forward binary in config.yaml (shared build/HepEmShow) with canonical
flags is byte-identical to build_agent_fwd knobs-off (gated wave-14), so these AD
runs share provenance with the a=2.3 reference.

Usage:
    python -u -m fidelity.wave14_transport --run --a 2.0 --seeds 1-6 --n-events 2000
    python -u -m fidelity.wave14_transport --run --a 3.0 --seeds 1-6 --n-events 2000
    python -u -m fidelity.wave14_transport --analyze
        -> fidelity/wave14_transport_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       run_forward)

CORE = slice(5, 19)          # L5-18 inclusive (14 layers)
CORE_LAYERS = list(range(5, 19))
FD_H_ODD = 0.1               # off-design FD half-step (a=2.0 & a=3.0 arms)
FD_DEN_ODD = 2 * FD_H_ODD

_HERE = Path(__file__).resolve().parent
RUN_DIR = _HERE / "wave14_runs"
SUMMARY_PATH = _HERE / "wave14_transport_summary.json"

# a-tag -> (a_value, ad_glob_or_None_meaning_wave14_runs)
GEOM = {
    "2.0": {"a": 2.0, "ad_glob": str(RUN_DIR / "absorber_ad_a20_s*.jsonl"),
            "fd_plus": "fidelity/wave13_runs/abs_fd_a20_plus_s*.jsonl",
            "fd_minus": "fidelity/wave13_runs/abs_fd_a20_minus_s*.jsonl",
            "fd_h": FD_H_ODD},
    "2.3": {"a": 2.3, "ad_glob": "experiments/perlayer_adfd/absorber_ad_s*.jsonl",
            "fd_from_summary": "experiments/perlayer_adfd_1M_summary.json",
            "fd_h": 0.02},
    "3.0": {"a": 3.0, "ad_glob": str(RUN_DIR / "absorber_ad_a30_s*.jsonl"),
            "fd_plus": "fidelity/wave11_runs/abs_fd_od_plus_s*.jsonl",
            "fd_minus": "fidelity/wave11_runs/abs_fd_od_minus_s*.jsonl",
            "fd_h": FD_H_ODD},
}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _atag(a: float) -> str:
    return f"a{int(round(a*10)):02d}"


def unit_out_path(a: float) -> Path:
    return RUN_DIR / f"absorber_ad_{_atag(a)}_s{{seed}}.jsonl"


def _seed_path(a: float, seed: int) -> Path:
    return RUN_DIR / f"absorber_ad_{_atag(a)}_s{seed}.jsonl"


def _row_exists(path: Path, seed: int, n_events: int) -> bool:
    if not path.exists():
        return False
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("seed") == seed and r.get("n_events") == n_events and r.get("ok"):
            return True
    return False


def run(a: float, seeds: list, n_events: int) -> int:
    cfg = load_config()
    dp = default_design_point(cfg).with_param("a", a)
    ctrl = ctrl_flags_from_config(cfg)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tag = _atag(a)
    rc_all = 0
    for seed in seeds:
        out_path = _seed_path(a, seed)
        if _row_exists(out_path, seed, n_events):
            print(f"[run] row exists ({tag}, s{seed}, n={n_events}) -> skip", flush=True)
            continue
        print(f"[run] a={a} seed={seed} n={n_events} g={dp.g} seeded=a "
              f"flags={ctrl.to_cli_args()}", flush=True)
        rr = run_forward(dp, "a", n_events=n_events, seed=seed, ctrl=ctrl)
        ok = (rr.returncode == 0 and rr.edeps is not None and not rr.nan)
        row = {
            "config": f"absorber_ad_{tag}",
            "param": "a",
            "seed": seed,
            "n_events": n_events,
            "a": dp.a,
            "g": dp.g,
            "n_layers": dp.n_layers,
            "seeded_param": "a",
            "flags": ctrl.to_cli_args(),
            "ok": ok,
            "returncode": rr.returncode,
            "nan": bool(rr.nan),
        }
        if rr.edeps is not None:
            row["mean_E"] = rr.edeps[:, 0].tolist()
            row["var_E"] = rr.edeps[:, 1].tolist()
            row["mean_dE"] = rr.edeps[:, 2].tolist()
            row["var_dE"] = rr.edeps[:, 3].tolist()
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if not ok:
            print(f"[run] FAILED (rc={rr.returncode}, nan={rr.nan}) -> recorded",
                  file=sys.stderr, flush=True)
            rc_all = 1
        else:
            d = rr.edeps
            print(f"[run] OK core_dE(L5-18)={d[CORE, 2].sum():.2f}  "
                  f"total_dE={d[:, 2].sum():.2f} -> {out_path}", flush=True)
    return rc_all


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _load_ad_seeds(pattern: str):
    """seed -> row (first ok row per seed)."""
    by = {}
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and "mean_dE" in r:
                by.setdefault(r["seed"], r)
    return by


def _ad_core(pattern: str):
    """Return dict with per-layer AD (event-weighted mean, pooled SEM) and
    core-window sum (event-weighted mean, seed-scatter SE keeping covariance)."""
    by = _load_ad_seeds(pattern)
    if len(by) < 2:
        return None
    means, vars_, ns, core_sums = [], [], [], []
    for r in by.values():
        m = np.asarray(r["mean_dE"], float)
        v = np.asarray(r["var_dE"], float)
        n = int(r["n_events"])
        means.append(m); vars_.append(v); ns.append(n)
        core_sums.append(m[CORE].sum())
    means = np.stack(means); vars_ = np.stack(vars_)
    ns = np.asarray(ns, float)[:, None]
    ntot = float(ns.sum())
    mean_l = (means * ns).sum(0) / ntot
    var_pooled = (vars_ * ns).sum(0) / ntot
    se_l = np.sqrt(var_pooled / ntot)            # per-layer pooled SEM
    core_sums = np.asarray(core_sums)
    nseed = len(core_sums)
    core_mean = float(core_sums.mean())
    core_se_scatter = float(core_sums.std(ddof=1) / math.sqrt(nseed))
    # pooled SEM of the window sum (ignores inter-layer covariance) for reference
    core_se_pooled = float(np.sqrt(var_pooled[CORE].sum() / ntot))
    return {
        "per_layer_ad": mean_l.tolist(),
        "per_layer_ad_sem_pooled": se_l.tolist(),
        "core_mean": core_mean,
        "core_se_scatter": core_se_scatter,
        "core_se_pooled": core_se_pooled,
        "n_seeds": nseed,
        "n_events_total": int(ntot),
        "per_seed_core": core_sums.tolist(),
    }


def _fd_core_from_arms(plus_glob: str, minus_glob: str, h: float):
    """Unpaired per-layer FD = (mean(E+) - mean(E-)) / (2h). Per-seed layer
    values -> event-weighted per-layer FD; core-window sum with seed-scatter SE
    (unpaired: combine + and - scatter in quadrature on the window sums)."""
    def load(g):
        by = {}
        for p in sorted(glob.glob(g)):
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok") and "mean_E" in r:
                    by.setdefault(r["seed"], r)
        return by
    bp, bm = load(plus_glob), load(minus_glob)
    if len(bp) < 2 or len(bm) < 2:
        return None
    den = 2 * h

    def stack(by):
        ms, cs, ns = [], [], []
        for r in by.values():
            m = np.asarray(r["mean_E"], float)
            ms.append(m); cs.append(m[CORE].sum()); ns.append(int(r["n_events"]))
        return np.stack(ms), np.asarray(cs), np.asarray(ns, float)
    mp, cp, np_ = stack(bp)
    mm, cm, nm_ = stack(bm)
    # event-weighted per-layer means
    plus_l = (mp * np_[:, None]).sum(0) / np_.sum()
    minus_l = (mm * nm_[:, None]).sum(0) / nm_.sum()
    fd_l = (plus_l - minus_l) / den
    # core window sums
    cp_mean, cm_mean = cp.mean(), cm.mean()
    cp_se = cp.std(ddof=1) / math.sqrt(len(cp))
    cm_se = cm.std(ddof=1) / math.sqrt(len(cm))
    fd_core = (cp_mean - cm_mean) / den
    fd_core_se = math.sqrt(cp_se**2 + cm_se**2) / den
    return {
        "fd_core": float(fd_core),
        "fd_core_se": float(fd_core_se),
        "per_layer_fd": fd_l.tolist(),
        "plus_core": float(cp_mean), "minus_core": float(cm_mean),
        "plus_core_se": float(cp_se), "minus_core_se": float(cm_se),
        "n_plus_seeds": len(cp), "n_minus_seeds": len(cm),
        "h": h,
    }


def _fd_core_from_summary(summary_path: str):
    d = json.load(open(summary_path))
    per = d["params"]["absorber"]["per_layer"]
    fd = np.asarray(per["fd"], float)
    fd_se = np.asarray(per["fd_se"], float)
    fd_core = float(fd[CORE].sum())
    fd_core_se = float(np.sqrt((fd_se[CORE]**2).sum()))  # per-layer SEs in quad
    return {
        "fd_core": fd_core, "fd_core_se": fd_core_se,
        "per_layer_fd": fd[CORE].tolist(),
        "source": "perlayer_adfd_1M_summary (paired CRN, seed-scatter per-layer SE)",
    }


def _ratio(ad, ad_se, fd, fd_se):
    r = ad / fd
    rse = abs(r) * math.sqrt((ad_se / ad)**2 + (fd_se / fd)**2)
    return r, rse


def analyze() -> int:
    results = {}
    for tag, spec in GEOM.items():
        ad = _ad_core(spec["ad_glob"])
        if ad is None:
            print(f"[analyze] {tag}: AD missing ({spec['ad_glob']})", file=sys.stderr)
            results[tag] = {"a": spec["a"], "status": "AD_MISSING"}
            continue
        if "fd_from_summary" in spec:
            fd = _fd_core_from_summary(spec["fd_from_summary"])
        else:
            fd = _fd_core_from_arms(spec["fd_plus"], spec["fd_minus"], spec["fd_h"])
        if fd is None:
            print(f"[analyze] {tag}: FD missing", file=sys.stderr)
            results[tag] = {"a": spec["a"], "status": "FD_MISSING", "ad": ad}
            continue
        # window ratio (seed-scatter AD SE, keeps covariance)
        ratio, ratio_se = _ratio(ad["core_mean"], ad["core_se_scatter"],
                                  fd["fd_core"], fd["fd_core_se"])
        # per-layer core-ratio uniformity
        pl_ad = np.asarray(ad["per_layer_ad"], float)[CORE]
        pl_fd = np.asarray(fd["per_layer_fd"], float)
        if len(pl_fd) == 50:
            pl_fd = pl_fd[CORE]
        pl_ratio = pl_ad / pl_fd
        results[tag] = {
            "a": spec["a"],
            "core_layers": CORE_LAYERS,
            "ad_core": ad["core_mean"],
            "ad_core_se": ad["core_se_scatter"],
            "ad_core_se_pooled": ad["core_se_pooled"],
            "ad_n_seeds": ad["n_seeds"],
            "ad_n_events_total": ad["n_events_total"],
            "fd_core": fd["fd_core"],
            "fd_core_se": fd["fd_core_se"],
            "ratio": ratio,
            "ratio_se": ratio_se,
            "correction_factor": 1.0 / ratio,
            "correction_factor_se": ratio_se / ratio**2,
            "per_layer_core_ratio": pl_ratio.tolist(),
            "per_layer_core_ratio_mean": float(pl_ratio.mean()),
            "per_layer_core_ratio_std": float(pl_ratio.std(ddof=1)),
        }

    # constant vs drift across the three geometries
    fit = _fit_across(results)

    summary = {
        "meta": {
            "question": ("Is the severed-absorber core-window (L5-18) AD/FD "
                         "undershoot geometry-independent (single 1.30x blind "
                         "correction transports) or does it drift with absorber "
                         "thickness a (needs a scaling rule c(a))?"),
            "core_window_layers": CORE_LAYERS,
            "design": "-l 50 -g 5.7 -e 10000 -t 400 e-, canonical severed "
                      "-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000",
            "ad_binary_provenance": ("config forward_bin (shared build/HepEmShow) "
                                     "canonical flags == build_agent_fwd knobs-off "
                                     "(byte-identical, gated wave-14)"),
            "fd_provenance": ("a=2.3 from perlayer_adfd_1M_summary (paired CRN, "
                              "h=0.02); a=2.0 wave13 arms a=2.1/1.9 h=0.1; a=3.0 "
                              "wave11 arms a=3.1/2.9 h=0.1; off-design FD unpaired, "
                              "seed-scatter window SE"),
        },
        "geometries": results,
        "fit": fit,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"geometries": {k: {kk: v[kk] for kk in
            ("a", "ratio", "ratio_se", "correction_factor", "correction_factor_se",
             "per_layer_core_ratio_std") if kk in v} for k, v in results.items()},
            "fit": fit}, indent=2))
    print(f"[analyze] wrote {SUMMARY_PATH}")
    return 0


def _fit_across(results: dict):
    pts = [(v["a"], v["ratio"], v["ratio_se"]) for v in results.values()
           if "ratio" in v]
    if len(pts) < 2:
        return {"status": "insufficient", "n_points": len(pts)}
    a = np.asarray([p[0] for p in pts])
    r = np.asarray([p[1] for p in pts])
    se = np.asarray([p[2] for p in pts])
    w = 1.0 / se**2
    # weighted constant
    c0 = (w * r).sum() / w.sum()
    c0_se = math.sqrt(1.0 / w.sum())
    chi2_const = float((w * (r - c0)**2).sum())
    dof_const = len(pts) - 1
    # weighted linear r = m*a + b
    S = w.sum(); Sa = (w*a).sum(); Saa = (w*a*a).sum()
    Sr = (w*r).sum(); Sar = (w*a*r).sum()
    D = S*Saa - Sa*Sa
    m = (S*Sar - Sa*Sr) / D
    b = (Saa*Sr - Sa*Sar) / D
    m_se = math.sqrt(S / D)
    b_se = math.sqrt(Saa / D)
    resid = r - (m*a + b)
    chi2_lin = float((w * resid**2).sum())
    dof_lin = len(pts) - 2
    return {
        "n_points": len(pts),
        "a_values": a.tolist(),
        "ratios": r.tolist(),
        "ratio_ses": se.tolist(),
        "constant": {"c": float(c0), "c_se": float(c0_se),
                     "chi2": chi2_const, "dof": dof_const,
                     "chi2_per_dof": chi2_const / dof_const if dof_const else None},
        "linear": {"slope": float(m), "slope_se": float(m_se),
                   "intercept": float(b), "intercept_se": float(b_se),
                   "chi2": chi2_lin, "dof": dof_lin,
                   "slope_significance_sigma": float(abs(m) / m_se)},
        "implied_correction_const": float(1.0 / c0),
        "implied_correction_const_se": float(c0_se / c0**2),
    }


def _parse_seeds(spec: str) -> list:
    if "-" in spec and ".." not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--a", type=float)
    ap.add_argument("--seeds", type=str, default="1-6")
    ap.add_argument("--n-events", type=int, default=2000)
    args = ap.parse_args()
    if args.run:
        if args.a is None:
            ap.error("--run requires --a")
        return run(args.a, _parse_seeds(args.seeds), args.n_events)
    if args.analyze:
        return analyze()
    ap.error("choose --run or --analyze")


if __name__ == "__main__":
    raise SystemExit(main())

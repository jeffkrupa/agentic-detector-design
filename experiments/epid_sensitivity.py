"""FINITE-DIFFERENCE sensitivity of e/gamma separation to REGION thickness.

The question (Phase 2, A2 test)
-------------------------------
Starting from the UNIFORM N=40, 10 GeV design, where does e/gamma separability
respond most strongly to a granularity redistribution? We split the 40 layers
into three regions -- front [0:13], mid [13:27], rear [27:40] -- and ask, by
finite differences, how the separation metric changes when ONE region's layers
are made thinner/thicker. The scientific prediction (mirroring the
epid_separation hypothesis) is that the sensitivity POINTS TO THE FRONT: the
largest |dMetric/d(region-scale)| should be at the front region, because the
e/gamma onset contrast lives at the front.

NO reverse-AD: recon confirmed the separation objective is not differentiated by
the binary and wiring it would be non-trivial. This is a pure FINITE-DIFFERENCE
probe.

The FD perturbation (fixed-budget, pure granularity redistribution)
-------------------------------------------------------------------
For each region R and a small ``delta`` (default 5%):
  1. Start from the uniform per-layer scale w_l = 1.
  2. Multiply region R's scales by (1 +/- delta).
  3. RE-RENORMALIZE the WHOLE profile so Sigma w_l == N again
     (depth_resolution._renorm_scale), which restores total length / total gap /
     total absorber to their fixed-budget values. This makes the perturbation a
     pure granularity REDISTRIBUTION (thicken region R <=> thin everything else),
     exactly matching the fixed-budget premise of the separation study, rather
     than a "more material in region R" change.
  4. Re-run both particles, recompute the metric.
The reported sensitivity is the central difference
    fd = (metric(+delta) - metric(-delta)) / (2 * delta)
i.e. dMetric per unit fractional region-scale.

Objective metric: Fisher S of frac-energy-in-first-3-layers (documented choice)
-------------------------------------------------------------------------------
We use the FRONT-onset Fisher separation S (frac energy in the first 3 layers)
as the scalar objective, NOT the native-grid CV-AUC. Reasons, at the modest
event counts used for FD: (a) Fisher S is a smooth ratio of sample moments, so
its FD is stable; the k-fold CV-AUC is a bounded rank statistic whose fold-split
variance adds noise that swamps a 5% FD signal; (b) S is the cleaner readout of
the FRONT-onset effect this test is about. The metric name is recorded in the
output so the choice is explicit. (``--metric auc`` can override to the
native-grid CV-AUC for cross-checks.)

Output: one JSONL row per region:
    {region, metric_name, base_value, plus_value, minus_value, fd_sensitivity,
     n_layers, n_events, energy, delta, seed}
plus a printed summary line stating whether the sensitivity points to the front.

Usage
-----
    python -u -m experiments.epid_sensitivity \
        --n-layers 40 --n-events 5000 --seed 1 --energy 10000 --delta 0.05 \
        --out-jsonl experiments/epid_sensitivity.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.sim import load_config, ctrl_flags_from_config
from tools import separation
from experiments.depth_resolution import (
    ABSORBER_MM, GAP_PER_LAYER_MM, _renorm_scale,
    profiles_from_scale, append_row,
)
from experiments.epid_separation import (
    N_FRONT, _two_particle_profiles, compute_separation,
)


def region_bounds(n_layers: int):
    """Front/mid/rear half-open layer ranges (~ thirds) for N layers.

    For N=40 this is [0:13], [13:27], [27:40] as specified.
    """
    a = n_layers // 3
    b = (2 * n_layers) // 3
    return {"front": (0, a), "mid": (a, b), "rear": (b, n_layers)}


def region_scale(n_layers: int, region, factor: float) -> np.ndarray:
    """Uniform base scale with one region multiplied by ``factor``, renormalized.

    Renormalizing back to Sigma w_l == N makes this a pure granularity
    REDISTRIBUTION at fixed total length / gap / absorber.
    """
    w = np.ones(int(n_layers), dtype=float)
    s, e = region
    w[s:e] *= factor
    return _renorm_scale(w)


def _metric_value(cfg, ctrl, n_layers, w, energy, n_events, seed,
                  metric_name: str) -> float:
    """Run both particles for scale ``w`` and return the chosen scalar metric."""
    abs_p, gap_p = profiles_from_scale(w, ABSORBER_MM, GAP_PER_LAYER_MM)
    prof_e, prof_g = _two_particle_profiles(
        cfg, ctrl, n_layers, abs_p, gap_p, energy, n_events, seed)
    if metric_name == "fisher_front3":
        prof_e = np.nan_to_num(np.asarray(prof_e, float), nan=0.0)
        prof_g = np.nan_to_num(np.asarray(prof_g, float), nan=0.0)
        fe = separation.frac_in_first_layers(prof_e, N_FRONT)
        fg = separation.frac_in_first_layers(prof_g, N_FRONT)
        fe, fg = fe[np.isfinite(fe)], fg[np.isfinite(fg)]
        return separation.fisher_separation(fe, fg)
    if metric_name == "auc":
        return compute_separation(prof_e, prof_g, abs_p, gap_p)["auc_native_cv"]
    raise ValueError(f"unknown metric {metric_name!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--n-events", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--energy", type=float, default=10000.0)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="fractional region-thickness FD step (default 5%)")
    ap.add_argument("--metric", type=str, default="fisher_front3",
                    choices=("fisher_front3", "auc"),
                    help="FD objective (default fisher_front3: more stable)")
    ap.add_argument("--out-jsonl", type=str, required=True)
    args = ap.parse_args(argv)

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    n_layers = int(args.n_layers)
    n_events = int(args.n_events)
    seed = int(args.seed)
    energy = float(args.energy)
    delta = float(args.delta)
    metric_name = args.metric
    out_path = Path(args.out_jsonl)

    print(f"[config] {cfg['_source']}")
    print(f"[run] FD sensitivity @ uniform N={n_layers} E={energy} seed={seed} "
          f"delta={delta} metric={metric_name} n_events={n_events}")
    print(f"[out]  {out_path}")

    regions = region_bounds(n_layers)
    print(f"[regions] {regions}")

    # Baseline metric at the uniform design (renormalized identity).
    w_base = _renorm_scale(np.ones(n_layers))
    base_value = _metric_value(cfg, ctrl, n_layers, w_base, energy, n_events,
                               seed, metric_name)
    print(f"[base] uniform {metric_name}={base_value:.5f}")

    results = []
    for name, rng_ in regions.items():
        w_plus = region_scale(n_layers, rng_, 1.0 + delta)
        w_minus = region_scale(n_layers, rng_, 1.0 - delta)
        plus = _metric_value(cfg, ctrl, n_layers, w_plus, energy, n_events,
                             seed, metric_name)
        minus = _metric_value(cfg, ctrl, n_layers, w_minus, energy, n_events,
                              seed, metric_name)
        fd = (plus - minus) / (2.0 * delta)
        row = {
            "region": name,
            "metric_name": metric_name,
            "base_value": base_value,
            "plus_value": plus,
            "minus_value": minus,
            "fd_sensitivity": fd,
            "n_layers": n_layers,
            "n_events": n_events,
            "energy": energy,
            "delta": delta,
            "seed": seed,
        }
        append_row(out_path, row)
        results.append(row)
        print(f"[region={name}] +={plus:.5f} -={minus:.5f} "
              f"fd={fd:.5g} (d{metric_name}/d region-scale)")

    # Does the sensitivity point to the FRONT?
    mags = {r["region"]: abs(r["fd_sensitivity"]) for r in results}
    argmax_region = max(mags, key=mags.get)
    points_to_front = (argmax_region == "front")
    print(f"[summary] |fd| by region: " +
          ", ".join(f"{k}={v:.5g}" for k, v in mags.items()))
    print(f"[summary] largest |d{metric_name}/d(thickness)| at "
          f"'{argmax_region}' region -> POINTS TO FRONT? "
          f"{'YES' if points_to_front else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

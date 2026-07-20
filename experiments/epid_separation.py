"""e-/gamma PARTICLE-ID separation vs FRONT longitudinal granularity (no AD).

The question
------------
Electrons and photons make almost identical EM showers DOWNSTREAM, but differ at
the FRONT: an e- starts depositing immediately, while a gamma must first convert
(pair-produce) and so has a delayed, shallower onset. Recon confirmed the
e/gamma separation lives in this FRONT shape -- frac-energy-in-layer-0 reaches
AUC ~= 0.88, while total-edep is only ~0.65. The hypothesis here mirrors the
depth-resolution study but for a DIFFERENT per-event observable: FINER
longitudinal sampling AT THE FRONT should sharpen the onset contrast and improve
e/gamma separability, while coarse-at-front should degrade it.

This driver measures, for one (design, seed), the e/gamma separation of a
front-focused granularity design, holding total length / gap / absorber fixed
(pure granularity redistribution).

Designs (front-focused; built via depth_resolution.build_scale_profile with
shower_max_layer=0 so the Gaussian granularity contrast is centered on the FRONT)
-------------------------------------------------------------------------------
  * ``uniform``        : equal layer thicknesses (w_l = 1).
  * ``fine_at_front``  : thin (fine) layers at the front, thick far away
                         (== build_scale_profile 'fine_at_max' with sm=0).
  * ``coarse_at_front``: thick (coarse) layers at the front -- the control
                         (== build_scale_profile 'coarse_at_max' with sm=0).
The fixed-budget renormalization (Sigma w_l == N) holds total length, total gap,
total absorber fixed, so this is a pure GRANULARITY test, not "more material".

Metrics (all pure-numpy, see tools/separation.py)
-------------------------------------------------
  * ``auc_native_cv`` (+ std): K-fold cross-validated AUC of a ridge linear
    classifier on the per-layer profiles ON THE NATIVE GRID. CV + regularization
    stop a finer (higher-dimensional) design from winning by overfitting.
  * ``fisher_front3`` / ``fisher_layer0``: Fisher separation S of two single
    FRONT-onset features (frac energy in first 3 layers; frac in layer 0).
  * ``auc_gridfair``: the ARTIFACT CONTROL. Each design's per-event profile is
    rebinned onto ONE common physical-depth axis (1-mm bins), and we score the
    AUC of a single physical onset feature (energy-fraction in the first 30 mm).
    All designs share the SAME feature dimensionality, so a real onset effect
    survives but a pure dimensionality artifact vanishes.

Two INDEPENDENT event samples (e- and gamma) are simulated at matched
energy/seed; they consume the RNG differently so they are independent
realizations even at the same seed. We REUSE depth_resolution's forward driver
(_forward_per_event_profiles, which sets HEPEMSHOW_OUTPUT_ALL=1 and slices
boundary_stats.csv to (n_events, n_layers)) for both particles, constructing two
DesignPoints that differ only in ``particle``.

Output: one JSONL row per (design, seed), fsync'd, resumable:
    {design, seed, n_layers, n_events, energy, auc_native_cv, auc_native_cv_std,
     fisher_front3, fisher_layer0, auc_gridfair, mean_total_e, mean_total_g,
     total_length, total_gap, frac_gamma_zero_first3}

Usage
-----
    python -u -m experiments.epid_separation \
        --design fine_at_front --n-layers 40 --n-events 5000 --seed 1 \
        --energy 10000 --out-jsonl experiments/epid_fine_at_front_s1.jsonl

    # metric self-tests (no sim, no C++):
    python -u -m experiments.epid_separation --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.sim import load_config, ctrl_flags_from_config
from tools import separation
from experiments.depth_resolution import (
    ABSORBER_MM, GAP_PER_LAYER_MM,
    build_scale_profile, profiles_from_scale, layer_center_depths,
    _design_point, _forward_per_event_profiles,
    append_row, already_done,
)

# Front-focused design names mapped to the underlying build_scale_profile design
# (called with shower_max_layer=0 so the Gaussian is centered on the FRONT).
DESIGNS = ("uniform", "fine_at_front", "coarse_at_front")
_DESIGN_TO_SCALE = {
    "uniform": "uniform",
    "fine_at_front": "fine_at_max",
    "coarse_at_front": "coarse_at_max",
}

# Number of front layers used for the frac-in-first-N onset feature.
N_FRONT = 3
# Physical onset window (mm) for the grid-fair control feature.
ONSET_MM = 30.0


def _two_particle_profiles(cfg, ctrl, n_layers, abs_p, gap_p, energy,
                           n_events, seed):
    """Forward-sim e- and gamma independently; return (prof_e, prof_g) matrices.

    Both runs reuse depth_resolution._forward_per_event_profiles (which sets
    HEPEMSHOW_OUTPUT_ALL=1 and reads boundary_stats.csv -> (n_events, n_layers)).
    The two DesignPoints differ ONLY in ``particle``.
    """
    dp_e = _design_point(cfg, n_layers, abs_p, gap_p)
    dp_e = dp_e.__class__(a=ABSORBER_MM, g=GAP_PER_LAYER_MM, energy=float(energy),
                          n_layers=int(n_layers), transverse=400.0, particle="e-",
                          abs_profile=abs_p, gap_profile=gap_p)
    dp_g = dp_e.__class__(a=ABSORBER_MM, g=GAP_PER_LAYER_MM, energy=float(energy),
                          n_layers=int(n_layers), transverse=400.0, particle="gamma",
                          abs_profile=abs_p, gap_profile=gap_p)
    prof_e = _forward_per_event_profiles(cfg, ctrl, dp_e, n_events, seed, n_layers)
    prof_g = _forward_per_event_profiles(cfg, ctrl, dp_g, n_events, seed, n_layers)
    return prof_e, prof_g


def compute_separation(prof_e, prof_g, abs_p, gap_p):
    """All separation metrics for an (e-, gamma) pair of profile matrices.

    Returns a dict with the metric fields that go into the JSONL row (minus the
    run-identity fields the caller fills in).
    """
    prof_e = np.nan_to_num(np.asarray(prof_e, dtype=float), nan=0.0)
    prof_g = np.nan_to_num(np.asarray(prof_g, dtype=float), nan=0.0)
    n_layers = prof_e.shape[1]

    # Native-grid CV-AUC on the full per-layer profiles.
    auc_cv, auc_cv_std = separation.ridge_cv_auc(prof_e, prof_g, k=5, lam=1.0,
                                                 seed=0)

    # Single FRONT-onset features (drop zero-energy events).
    fe3 = separation.frac_in_first_layers(prof_e, N_FRONT)
    fg3 = separation.frac_in_first_layers(prof_g, N_FRONT)
    fe3, fg3 = fe3[np.isfinite(fe3)], fg3[np.isfinite(fg3)]
    fisher_front3 = separation.fisher_separation(fe3, fg3)

    fe0 = separation.frac_in_first_layers(prof_e, 1)
    fg0 = separation.frac_in_first_layers(prof_g, 1)
    fe0, fg0 = fe0[np.isfinite(fe0)], fg0[np.isfinite(fg0)]
    fisher_layer0 = separation.fisher_separation(fe0, fg0)

    # Grid-fair control on the common physical-depth axis.
    centers = layer_center_depths(abs_p, gap_p)  # noqa: F841 (kept for clarity)
    t = np.asarray(abs_p, dtype=float) + np.asarray(gap_p, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(t)])
    layer_lo, layer_hi = edges[:-1], edges[1:]
    total_length = float(edges[-1])
    auc_gf = separation.gridfair_auc(prof_e, prof_g, layer_lo, layer_hi,
                                     total_length, onset_mm=ONSET_MM, bin_mm=1.0)

    # Diagnostic: fraction of gamma events with ZERO energy in the first
    # N_FRONT layers (the late-onset signature that drives separation).
    g_front_sum = prof_g[:, :N_FRONT].sum(axis=1)
    g_has_energy = prof_g.sum(axis=1) > 0
    frac_gamma_zero_first3 = (
        float(np.mean((g_front_sum <= 0)[g_has_energy]))
        if np.any(g_has_energy) else float("nan"))

    return {
        "auc_native_cv": auc_cv,
        "auc_native_cv_std": auc_cv_std,
        "fisher_front3": fisher_front3,
        "fisher_layer0": fisher_layer0,
        "auc_gridfair": auc_gf,
        "mean_total_e": float(prof_e.sum(axis=1).mean()),
        "mean_total_g": float(prof_g.sum(axis=1).mean()),
        "frac_gamma_zero_first3": frac_gamma_zero_first3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=str, choices=DESIGNS)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--n-events", type=int, default=5000)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--energy", type=float, default=10000.0)
    ap.add_argument("--out-jsonl", type=str)
    ap.add_argument("--self-test", action="store_true",
                    help="run the metric self-tests (no sim) and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        print("[self-test] delegating to tools.separation metric self-test")
        return separation._self_test()

    if not (args.design and args.seed is not None and args.out_jsonl):
        ap.error("--design, --seed and --out-jsonl are required "
                 "(unless --self-test)")

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    n_layers = int(args.n_layers)
    n_events = int(args.n_events)
    seed = int(args.seed)
    design = args.design
    energy = float(args.energy)
    out_path = Path(args.out_jsonl)

    print(f"[config] {cfg['_source']}")
    print(f"[run] design={design} seed={seed} n_layers={n_layers} "
          f"n_events={n_events} energy={energy}")
    print(f"[out]  {out_path}")

    if already_done(out_path, design, seed):
        print(f"[resume] SKIP: (design={design}, seed={seed}) present in {out_path}")
        return 0

    # Build the FRONT-focused per-layer thickness profile (shower_max_layer=0).
    scale_design = _DESIGN_TO_SCALE[design]
    w = build_scale_profile(scale_design, n_layers, shower_max_layer=0)
    abs_p, gap_p = profiles_from_scale(w, ABSORBER_MM, GAP_PER_LAYER_MM)
    total_gap = float(np.sum(gap_p))
    total_length = float(np.sum(np.asarray(abs_p) + np.asarray(gap_p)))
    print(f"[profile] {design} (scale={scale_design}, sm=0): "
          f"w in [{w.min():.3f},{w.max():.3f}] sum_w={w.sum():.4f} "
          f"(==N={n_layers}); total_gap={total_gap:.4g}mm "
          f"total_length={total_length:.4g}mm")

    # Two independent forward samples: e- and gamma.
    print(f"[sim] forward e- and gamma, {n_events} events each, seed={seed} ...")
    prof_e, prof_g = _two_particle_profiles(
        cfg, ctrl, n_layers, abs_p, gap_p, energy, n_events, seed)
    print(f"[sim] profiles: e-={prof_e.shape} gamma={prof_g.shape}")

    metrics = compute_separation(prof_e, prof_g, abs_p, gap_p)

    row = {
        "design": design,
        "seed": seed,
        "n_layers": n_layers,
        "n_events": n_events,
        "energy": energy,
        **metrics,
        "total_length": total_length,
        "total_gap": total_gap,
    }
    append_row(out_path, row)
    print(f"[row] {json.dumps(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

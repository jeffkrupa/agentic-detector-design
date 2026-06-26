"""VALUE-ONLY shower-MAX-DEPTH resolution test (no AD, no optimizer).

The question (the "shape" objective)
------------------------------------
Every TOTAL/aggregate-energy objective we have tried (total_edep, visible
energy, energy resolution sigma_Evis/mu) optimizes to UNIFORM layer
thicknesses: a peaked per-layer gradient gets integrated away by a sum. The
hypothesis here is different in kind. The *shower-maximum DEPTH* -- the physical
depth x_max (mm) at which each event's longitudinal energy profile peaks -- is a
PER-EVENT SHAPE quantity, not a sum. Finer longitudinal sampling near shower max
should localize that peak more precisely, so the resolution

    depth_res = sigma(x_max)   (mm, over events; lower = better)

ought to genuinely reward non-uniform structure (thin layers packed near shower
max). This script measures depth_res for a design at fixed seed.

How x_max is measured (the crux -- see STEP-1 findings below)
-------------------------------------------------------------
x_max is PER EVENT. The standard sim outputs (``edeps_<seed>``,
``edeps_gap_<seed>``, ``vx_edep_hist.csv``) are all per-LAYER aggregates (mean +
meanSq over events) -- they cannot give a per-event peak. The ONLY per-event
per-layer dump in the sim is ``boundary_stats.csv``, written one row per event
with columns ``layer_edep_0 .. layer_edep_49`` = the COMBINED (abs+gap) energy
deposit of that event in each layer (``fEdepPerLayer_CurrentEvent`` flushed by
``EventLoop::ProcessEvents`` -> ``SteppingLoop::FlushBoundaryStatsForEvent``).

That dump is gated by a HARDCODED ``bool outputboundarylayers = false;``
(SteppingLoop.cc:46) with NO CLI/env toggle, so producing it requires a C++
recompile. This harness is therefore written to CONSUME ``boundary_stats.csv``
the moment that flag is enabled (ideally wired to an env var, mirroring
``outputall`` at SteppingLoop.cc:45). Until then it runs the forward binary,
detects the missing file, and exits with a precise, actionable error -- it does
NOT silently fabricate a value. The estimator itself is unit-tested on synthetic
profiles via ``--self-test`` (no sim, no C++ needed), which is what the
synchronous validation exercises.

Estimator: parabolic peak interpolation (physical mm)
-----------------------------------------------------
Per event, with combined edep e_0..e_{N-1} and layer-center depths c_0..c_{N-1}
(mm from calo start, computed from the abs+gap profiles this harness itself
passes in):
  1. k = argmax_i e_i.
  2. If 0 < k < N-1 and the 3 points (k-1,k,k+1) form a concave triple, fit a
     parabola through (c_{k-1},e_{k-1}),(c_k,e_k),(c_{k+1},e_{k+1}) and take its
     vertex depth as x_max -- SUB-layer resolution, the whole point of the test
     (raw argmax-depth would be discretized to layer centers and blind to
     granularity differences below one layer).
  3. Fallback to the layer-center depth c_k at the edge / non-concave cases.
Parabolic interpolation (not energy-weighted centroid) is chosen because the
longitudinal profile near max is locally parabolic in depth and the centroid is
biased by the long downstream tail; the vertex is the cleaner peak locator.

Metric: ``sigma_xmax_mm`` = std of x_max over events; also ``mean_xmax_mm`` and
``depth_res`` (== sigma_xmax_mm). Lower sigma = better depth resolution.

Fixed-budget designs (what is held fixed -- stated explicitly)
--------------------------------------------------------------
HELD FIXED across all designs: (a) total detector length L = Sigma(abs_l+gap_l),
and (b) total gap (active) budget Sigma(gap_l). Holding BOTH makes this a pure
GRANULARITY test (how the SAME material is sliced), not a "more material" test.
Per-layer absorber and gap are scaled together by a common per-layer factor
w_l so layer l's total thickness is w_l * (L/N), with Sigma w_l = N (=> total
length and the abs:gap ratio per layer, hence total gap and total absorber, are
all preserved exactly).
  * ``uniform``      : w_l = 1 (equal thicknesses).
  * ``fine_at_max``  : w_l < 1 (thin layers) near the shower-max layer m, w_l > 1
                       (thick) far away -> FINE granularity where the peak is.
  * ``coarse_at_max``: the reverse (thick at max, thin elsewhere) -- the control;
                       should be WORSE if the hypothesis holds.

CONFOUND (flagged honestly): changing per-layer thickness changes BOTH the
sampling granularity AND, slightly, the shower development (sampling fraction and
the depth at which energy lands), because absorber and gap move together. Holding
total length, total gap, and total absorber fixed removes the gross "more
material" confound, but a residual coupling between local thickness and local
shower physics remains and cannot be fully isolated in this sampling calorimeter.
Read a POSITIVE result (fine_at_max < uniform < coarse_at_max) as "granularity
near max helps, modulo this residual coupling," not as a clean causal isolation.

Output: one JSONL row per (design, seed), fsync'd, resumable:
    {design, seed, n_layers, n_events, sigma_xmax_mm, mean_xmax_mm, depth_res,
     shower_max_layer, total_gap, total_length, n_used_events}

Usage
-----
    python -u -m experiments.depth_resolution \
        --design fine_at_max --n-layers 40 --n-events 10000 --seed 1 \
        --out-jsonl experiments/depthres_fine_s1.jsonl

    # estimator validation (no sim, no C++):
    python -u -m experiments.depth_resolution --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       _common_args)

# Fixed geometry knobs / fairness constraints (mirror resolution_ceiling.py).
GAP_PER_LAYER_MM = 5.7   # default per-layer gap
ABSORBER_MM = 3.5        # default per-layer absorber
# Per-layer thickness-scale bounds (keeps geometry valid; granularity contrast).
W_MIN = 0.4              # thinnest layer = 0.4x nominal
W_MAX = 2.0              # thickest layer = 2.0x nominal

DESIGNS = ("uniform", "fine_at_max", "coarse_at_max")

# The per-event per-layer dump filename (see module docstring / STEP-1 findings).
BOUNDARY_CSV = "boundary_stats.csv"


# --------------------------------------------------------------------------- #
# Per-layer thickness-scale profiles (granularity at fixed total length + gap)
# --------------------------------------------------------------------------- #
def _renorm_scale(w: np.ndarray) -> np.ndarray:
    """Clip to [W_MIN, W_MAX] then renormalize so Sigma(w) == N exactly.

    Sigma(w) == N guarantees Sigma(abs_l+gap_l) == L (total length fixed) and,
    because abs and gap are scaled by the SAME w_l, Sigma(gap_l) and
    Sigma(abs_l) are each preserved too. One extra clip/renorm pass keeps the
    sum exact while staying close to the bounds for these smooth profiles.
    """
    w = np.asarray(w, dtype=float)
    n = w.size
    w = np.clip(w, W_MIN, W_MAX)
    w = w * (n / w.sum())
    w = np.clip(w, W_MIN, W_MAX)
    w = w * (n / w.sum())
    return w


def build_scale_profile(design: str, n_layers: int, shower_max_layer: int) -> np.ndarray:
    """Return length-N per-layer thickness scales w_l (Sigma w_l == N).

    Layer l's total thickness becomes w_l * (L/N); absorber and gap each scale
    by w_l, so the abs:gap ratio, total length, total gap, total absorber are
    all held fixed -- only the *distribution* of thickness (granularity) varies.
    """
    n = int(n_layers)
    x = np.arange(n, dtype=float)
    if design == "uniform":
        return _renorm_scale(np.ones(n))

    # Gaussian weight peaked at shower max, width ~ N/6.
    sigma = max(2.0, n / 6.0)
    peak = np.exp(-0.5 * ((x - shower_max_layer) / sigma) ** 2)  # in (0, 1]

    if design == "fine_at_max":
        # thin (small w) at max, thick (large w) far away: w = MAX - (MAX-MIN)*peak
        w = W_MAX - (W_MAX - W_MIN) * peak
        return _renorm_scale(w)
    if design == "coarse_at_max":
        # thick (large w) at max, thin (small w) far away: the control
        w = W_MIN + (W_MAX - W_MIN) * peak
        return _renorm_scale(w)
    raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")


def profiles_from_scale(w: np.ndarray, abs_mm: float, gap_mm: float):
    """Map per-layer scale w_l -> (abs_profile, gap_profile) in mm."""
    abs_profile = tuple(float(abs_mm * wi) for wi in w)
    gap_profile = tuple(float(gap_mm * wi) for wi in w)
    return abs_profile, gap_profile


def layer_center_depths(abs_profile, gap_profile) -> np.ndarray:
    """Physical depth (mm from calo start) at the CENTER of each layer.

    Layer l spans [Sigma_{j<l} t_j, Sigma_{j<=l} t_j) with t_j = abs_j + gap_j;
    its center is at the cumulative thickness up to its start plus t_l/2.
    """
    t = np.asarray(abs_profile, dtype=float) + np.asarray(gap_profile, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(t)])      # length N+1
    centers = 0.5 * (edges[:-1] + edges[1:])           # length N
    return centers


# --------------------------------------------------------------------------- #
# Per-event shower-max-depth estimator (parabolic peak interpolation, mm)
# --------------------------------------------------------------------------- #
def xmax_parabolic(edep: np.ndarray, centers: np.ndarray) -> Optional[float]:
    """Physical depth (mm) of the shower max for ONE event, or None if empty.

    edep, centers: length-N arrays (combined per-layer edep and layer-center
    depths). Parabolic vertex through the 3 points around argmax when concave
    and interior; else the argmax layer center. Returns None if the event has no
    energy (all edep <= 0), so the caller can drop it.
    """
    edep = np.asarray(edep, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if edep.size == 0 or not np.any(edep > 0) or not np.all(np.isfinite(edep)):
        return None
    k = int(np.argmax(edep))
    n = edep.size
    if k <= 0 or k >= n - 1:
        return float(centers[k])
    x0, x1, x2 = centers[k - 1], centers[k], centers[k + 1]
    y0, y1, y2 = edep[k - 1], edep[k], edep[k + 1]
    # Parabola y = a x^2 + b x + c through the 3 points; vertex at -b/2a.
    d0 = (x0 - x1) * (x0 - x2)
    d1 = (x1 - x0) * (x1 - x2)
    d2 = (x2 - x0) * (x2 - x1)
    if d0 == 0 or d1 == 0 or d2 == 0:
        return float(x1)
    a = y0 / d0 + y1 / d1 + y2 / d2
    b = (-y0 * (x1 + x2) / d0 - y1 * (x0 + x2) / d1 - y2 * (x0 + x1) / d2)
    if a >= 0:  # not concave (no interior max) -> fall back to the peak center
        return float(x1)
    vertex = -b / (2.0 * a)
    # Guard: the vertex must lie within the bracketing layer centers; otherwise
    # the local quadratic is a poor peak model -> fall back to the argmax center.
    if vertex < x0 or vertex > x2:
        return float(x1)
    return float(vertex)


def depth_resolution_from_profiles(profiles: np.ndarray, centers: np.ndarray):
    """sigma, mean, n_used of x_max over events.

    profiles: shape (n_events, N) per-event combined per-layer edep. NaN-padded
    trailing columns (the dump always writes 50 columns) must be sliced to N by
    the caller. Events with no energy are dropped.
    """
    xs = []
    for row in profiles:
        xm = xmax_parabolic(row, centers)
        if xm is not None:
            xs.append(xm)
    if len(xs) < 2:
        raise RuntimeError(f"only {len(xs)} usable events; cannot form sigma(x_max)")
    xs = np.asarray(xs, dtype=float)
    return float(xs.std(ddof=1)), float(xs.mean()), int(xs.size)


# --------------------------------------------------------------------------- #
# Forward binary execution
# --------------------------------------------------------------------------- #
def _design_point(cfg, n_layers: int, abs_profile, gap_profile):
    dp = default_design_point(cfg)
    return dp.__class__(
        a=ABSORBER_MM, g=GAP_PER_LAYER_MM, energy=10000.0,
        n_layers=int(n_layers), transverse=400.0, particle="e-",
        abs_profile=abs_profile, gap_profile=gap_profile,
    )


def _forward_edeps(cfg, ctrl, dp, n_events: int, seed: int) -> np.ndarray:
    """Run forward in a temp dir; return the aggregate edeps_<seed> array.

    Used to locate the shower-max layer (argmax of column-0 mean edep).
    """
    fwd_bin = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    with tempfile.TemporaryDirectory(prefix="depthres_sm_") as wd:
        proc = subprocess.run([fwd_bin, *args], cwd=wd,
                              env={**os.environ, "HEPEMSHOW_OUTPUT_ALL": "1"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"{fwd_bin} rc={proc.returncode}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
        out = Path(wd) / f"edeps_{seed}"
        if not out.exists():
            cand = list(Path(wd).glob("edeps_*"))
            if not cand:
                raise FileNotFoundError(f"{fwd_bin} produced no edeps_{seed}")
            out = cand[0]
        arr = np.loadtxt(out)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


def _forward_per_event_profiles(cfg, ctrl, dp, n_events: int, seed: int,
                                n_layers: int) -> np.ndarray:
    """Run forward, read boundary_stats.csv -> (n_events, n_layers) edep array.

    Raises a precise, actionable error if boundary_stats.csv is absent, which
    happens whenever the binary was NOT compiled with outputboundarylayers=true
    (see module docstring / STEP-1 findings). The columns of interest are
    ``layer_edep_0 .. layer_edep_{N-1}`` (the dump always has 50 such columns;
    we slice to N).
    """
    fwd_bin = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    with tempfile.TemporaryDirectory(prefix="depthres_") as wd:
        proc = subprocess.run([fwd_bin, *args], cwd=wd,
                              env={**os.environ, "HEPEMSHOW_OUTPUT_ALL": "1"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"{fwd_bin} rc={proc.returncode}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
        csv = Path(wd) / BOUNDARY_CSV
        if not csv.exists():
            raise FileNotFoundError(
                f"{BOUNDARY_CSV} not produced by {fwd_bin}. The per-event "
                f"per-layer dump is gated by the HARDCODED "
                f"`bool outputboundarylayers = false;` at "
                f"Simulation/src/SteppingLoop.cc:46 and needs a C++ recompile "
                f"(ideally wired to an env var like HEPEMSHOW_OUTPUT_ALL at "
                f"line 45). This harness is ready to consume the file the moment "
                f"that flag is on; until then the depth-resolution VALUE cannot "
                f"be measured. Run `--self-test` to validate the estimator "
                f"without the sim.")
        return _read_boundary_profiles(csv, n_layers)


def _read_boundary_profiles(csv: Path, n_layers: int) -> np.ndarray:
    """Parse boundary_stats.csv -> (n_events, n_layers) combined per-layer edep.

    Header columns: event,numTracks,...,edep,steplength_dot,layer_edep_0,...,
    layer_edep_49,maxConsecBoundary,... . We select the layer_edep_* columns by
    name and slice to the first n_layers (trailing ones are NaN for N<50).
    """
    import csv as _csvmod
    with open(csv) as fh:
        reader = _csvmod.reader(fh)
        header = next(reader)
        idx = [i for i, h in enumerate(header) if h.startswith("layer_edep_")]
        if not idx:
            raise RuntimeError(f"{csv}: no layer_edep_* columns in header {header}")
        # keep them in numeric layer order
        idx_sorted = sorted(idx, key=lambda i: int(header[i].split("_")[-1]))
        rows = []
        for rec in reader:
            if not rec:
                continue
            vals = [float(rec[i]) for i in idx_sorted[:n_layers]]
            rows.append(vals)
    if not rows:
        raise RuntimeError(f"{csv}: no event rows")
    return np.asarray(rows, dtype=float)


def find_shower_max_layer(cfg, ctrl, n_layers: int, seed: int,
                          n_events: int = 1000) -> int:
    """Argmax of per-layer mean COMBINED edep from a baseline UNIFORM forward run.

    Uses edeps_<seed> column 0 (per-layer mean combined edep) -- always present,
    no C++ change needed. Centers the granularity contrast on the true peak.
    """
    w = build_scale_profile("uniform", n_layers, shower_max_layer=0)
    abs_p, gap_p = profiles_from_scale(w, ABSORBER_MM, GAP_PER_LAYER_MM)
    dp = _design_point(cfg, n_layers, abs_p, gap_p)
    edeps = _forward_edeps(cfg, ctrl, dp, n_events, seed)
    return int(np.argmax(edeps[:, 0]))


# --------------------------------------------------------------------------- #
# JSONL append / resume
# --------------------------------------------------------------------------- #
def append_row(out_path: Path, row: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", buffering=1) as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(out_path: Path):
    rows = []
    if not out_path.exists():
        return rows
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def already_done(out_path: Path, design: str, seed: int) -> bool:
    for r in read_rows(out_path):
        if r.get("design") == design and int(r.get("seed", -10**9)) == int(seed):
            return True
    return False


# --------------------------------------------------------------------------- #
# Self-test: validate the estimator on synthetic profiles (no sim, no C++)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    """Synthetic check: a known parabolic peak is recovered to sub-layer mm, and
    a population of jittered peaks yields a finite, sensible sigma(x_max)."""
    print("[self-test] estimator validation on synthetic profiles (no sim)")
    n = 40
    abs_p = tuple(ABSORBER_MM for _ in range(n))
    gap_p = tuple(GAP_PER_LAYER_MM for _ in range(n))
    centers = layer_center_depths(abs_p, gap_p)
    layer_t = ABSORBER_MM + GAP_PER_LAYER_MM
    print(f"[self-test] N={n}, layer thickness={layer_t}mm, "
          f"total length={centers[-1] + layer_t/2:.1f}mm")

    # (1) exact-vertex recovery: a true peak placed BETWEEN two layer centers
    #     must be recovered to sub-layer precision (not snapped to a center).
    true_x = 0.5 * (centers[14] + centers[15])  # between layers 14 and 15
    prof = np.exp(-0.5 * ((centers - true_x) / (3 * layer_t)) ** 2)
    est = xmax_parabolic(prof, centers)
    err = abs(est - true_x)
    print(f"[self-test] single peak: true={true_x:.3f}mm est={est:.3f}mm "
          f"err={err:.3f}mm (layer={layer_t:.1f}mm)")
    assert err < 0.25 * layer_t, "parabolic estimate not sub-layer accurate"
    assert abs(est - centers[14]) > 1e-6 and abs(est - centers[15]) > 1e-6, \
        "estimate snapped to a layer center (no sub-layer resolution)"

    # (2a) clean population (no per-layer noise): the recovered sigma must
    #      track the injected peak-depth jitter closely.
    rng = np.random.default_rng(0)
    jit = 4.0  # mm of injected peak-depth jitter
    base = centers[18]
    xs_clean = []
    for _ in range(2000):
        xt = base + rng.normal(0, jit)
        p = np.exp(-0.5 * ((centers - xt) / (3 * layer_t)) ** 2)
        xm = xmax_parabolic(p, centers)
        if xm is not None:
            xs_clean.append(xm)
    xs_clean = np.asarray(xs_clean)
    sig_clean = xs_clean.std(ddof=1)
    print(f"[self-test] clean population: n_used={xs_clean.size} "
          f"sigma_xmax={sig_clean:.3f}mm vs injected jitter={jit}mm")
    assert np.isfinite(sig_clean) and sig_clean > 0
    assert 0.7 * jit < sig_clean < 1.5 * jit, \
        "sigma(x_max) does not track injected jitter on clean profiles"

    # (2b) noisy population: per-layer multiplicative noise adds its own peak-
    #      location variance, so sigma INCREASES -- check finite + ordering.
    xs_noisy = []
    for _ in range(2000):
        xt = base + rng.normal(0, jit)
        p = np.exp(-0.5 * ((centers - xt) / (3 * layer_t)) ** 2)
        p = p * rng.uniform(0.8, 1.2, size=p.shape)
        xm = xmax_parabolic(p, centers)
        if xm is not None:
            xs_noisy.append(xm)
    xs_noisy = np.asarray(xs_noisy)
    sig_noisy = xs_noisy.std(ddof=1)
    print(f"[self-test] noisy population: n_used={xs_noisy.size} "
          f"sigma_xmax={sig_noisy:.3f}mm (>= clean, as expected)")
    assert np.isfinite(sig_noisy) and sig_noisy >= sig_clean * 0.99

    # (3) empty / edge events handled (return None / edge center, no crash).
    assert xmax_parabolic(np.zeros(n), centers) is None
    assert xmax_parabolic(np.array([5.0] + [0.0] * (n - 1)), centers) == centers[0]
    print("[self-test] edge/empty events handled")

    # (4) the full pipeline on a synthetic (n_events, N) profile matrix.
    profiles = []
    for _ in range(200):
        xt = base + rng.normal(0, jit)
        p = np.exp(-0.5 * ((centers - xt) / (3 * layer_t)) ** 2)
        profiles.append(p)
    profiles = np.asarray(profiles)
    sig2, mean2, nused = depth_resolution_from_profiles(profiles, centers)
    print(f"[self-test] pipeline: sigma_xmax={sig2:.3f}mm mean_xmax={mean2:.3f}mm "
          f"n_used={nused}")
    assert np.isfinite(sig2) and sig2 > 0
    print("[self-test] PASS: finite sigma(x_max), sub-layer estimator validated")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=str, choices=DESIGNS)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--n-events", type=int, default=8000)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--shower-max", type=int, default=None,
                    help="shower-max layer index; skips the baseline probe run")
    ap.add_argument("--out-jsonl", type=str)
    ap.add_argument("--self-test", action="store_true",
                    help="validate the estimator on synthetic profiles and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not (args.design and args.seed is not None and args.out_jsonl):
        ap.error("--design, --seed and --out-jsonl are required "
                 "(unless --self-test)")

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    n_layers = int(args.n_layers)
    n_events = int(args.n_events)
    seed = int(args.seed)
    design = args.design
    out_path = Path(args.out_jsonl)

    print(f"[config] {cfg['_source']}")
    print(f"[run] design={design} seed={seed} n_layers={n_layers} "
          f"n_events={n_events}")
    print(f"[out]  {out_path}")

    if already_done(out_path, design, seed):
        print(f"[resume] SKIP: (design={design}, seed={seed}) present in {out_path}")
        return 0

    # --- shower-max layer -------------------------------------------------
    if args.shower_max is not None:
        sm_layer = int(args.shower_max)
        print(f"[showermax] using provided --shower-max={sm_layer}")
    else:
        sm_layer = find_shower_max_layer(cfg, ctrl, n_layers, seed, n_events=1000)
        print(f"[showermax] argmax per-layer mean edep -> layer {sm_layer} "
              f"(of {n_layers}) from baseline uniform probe")

    # --- build the per-layer thickness profiles for this design -----------
    w = build_scale_profile(design, n_layers, sm_layer)
    abs_p, gap_p = profiles_from_scale(w, ABSORBER_MM, GAP_PER_LAYER_MM)
    centers = layer_center_depths(abs_p, gap_p)
    total_gap = float(np.sum(gap_p))
    total_length = float(np.sum(np.asarray(abs_p) + np.asarray(gap_p)))
    print(f"[profile] w in [{w.min():.3f},{w.max():.3f}] sum_w={w.sum():.4f} "
          f"(==N={n_layers}); total_gap={total_gap:.4g}mm "
          f"total_length={total_length:.4g}mm")

    # --- forward run, read per-event profiles, compute sigma(x_max) -------
    dp = _design_point(cfg, n_layers, abs_p, gap_p)
    profiles = _forward_per_event_profiles(cfg, ctrl, dp, n_events, seed, n_layers)
    sigma_xmax, mean_xmax, n_used = depth_resolution_from_profiles(profiles, centers)

    row = {
        "design": design,
        "seed": seed,
        "n_layers": n_layers,
        "n_events": n_events,
        "sigma_xmax_mm": sigma_xmax,
        "mean_xmax_mm": mean_xmax,
        "depth_res": sigma_xmax,
        "shower_max_layer": sm_layer,
        "total_gap": total_gap,
        "total_length": total_length,
        "n_used_events": n_used,
    }
    append_row(out_path, row)
    print(f"[row] {json.dumps(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

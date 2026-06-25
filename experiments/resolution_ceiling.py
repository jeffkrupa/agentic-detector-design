"""VALUE-ONLY energy-resolution ceiling test (no AD, no optimizer, no C++ build).

The question
------------
Before any AD work: does a NON-UNIFORM longitudinal sampling layout give a
BETTER energy resolution ``sigma(E_vis)/<E_vis>`` than a UNIFORM one **at the
same total gap (active-material) budget**? ``E_vis`` is the total gap (active)
energy deposit per event. Lower sigma/mu = better. Resolution is the one
physically-motivated, structure-rewarding objective: if even the *value*
(gradient-free) shows no improvement, structure is dead here.

The forward sim already PRINTS this to stdout:

    Gap     : mean Edep = <mu> [MeV] and  Std-dev = <sigma> [MeV]

(see hepemshow ``Simulation/src/Results.cc``::WriteResults, the "Gap" line).
Resolution = sigma / mu. NB: ``sigma`` is the std of the per-event TOTAL gap
energy = E_vis, so it correctly includes cross-layer covariance.

Fairness constraint (critical)
------------------------------
The TOTAL gap budget Sigma(gap_l) is FIXED at ``--gap-budget`` (default
``n_layers * 5.7`` mm): every design samples the SAME amount of active material.
The absorber is fixed uniform at ``--abs`` mm (default 3.5). Each design is just
a different *distribution* of that fixed gap budget over the N layers, with each
per-layer gap clipped to ``[GAP_MIN, GAP_MAX]`` = [1.5, 12] mm and the vector
renormalized so the sum equals the budget to < 0.1 %.

Designs
-------
* ``uniform`` : gap_l = budget / N for all l (the reference).
* ``bump``    : Gaussian-ish bump (width ~ N/6) centered on the shower-max layer
                ``m``, with a thin floor elsewhere, renormalized to the budget
                (the hypothesis: thick sampling where the shower + its
                fluctuations peak).
* ``front``   : monotone front-thick -> tail-thin linear taper.
* ``tail``    : reverse (thin front -> thick tail) -- a control expected to be
                WORSE if the bump hypothesis holds.

The shower-max layer ``m`` is found once (deterministically) from a quick
baseline UNIFORM forward run at the design's budget: argmax of the per-layer
mean gap edep (``edeps_<seed>`` column 0). ``--shower-max`` skips this, but it is
always computed for the bump design unless given.

Output
------
One JSON line per (design, seed) appended (fsync'd) to ``--out-jsonl``:
    {"design","seed","n_layers","n_events","gap_budget","total_gap_check",
     "mu","sigma","resolution","shower_max"}
Resumable: a (design, seed) already present in the file is skipped.

Usage
-----
    python -u -m experiments.resolution_ceiling \
        --design bump --n-layers 40 --n-events 10000 --seed 1 \
        --out-jsonl experiments/resceil_bump_s1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       _common_args)

# Fixed geometry knobs / fairness constraints for this study.
GAP_PER_LAYER_MM = 5.7   # default per-layer gap -> default budget = N * 5.7 mm
ABSORBER_MM = 3.5        # default absorber (uniform) for all designs
GAP_MIN_MM = 1.5         # per-layer gap floor (keeps geometry valid)
GAP_MAX_MM = 12.0        # per-layer gap ceiling

DESIGNS = ("uniform", "bump", "front", "tail")

# Parses e.g. " Gap     : mean Edep = 1234.5 [MeV] and  Std-dev = 67.8 [MeV]"
_GAP_LINE_RE = re.compile(
    r"Gap\s*:\s*mean Edep\s*=\s*([-+0-9.eE]+)\s*\[MeV\]\s*and\s*Std-dev\s*=\s*([-+0-9.eE]+)\s*\[MeV\]"
)


# --------------------------------------------------------------------------- #
# Gap-profile construction
# --------------------------------------------------------------------------- #
def _renorm_to_budget(gap: np.ndarray, budget: float) -> np.ndarray:
    """Clip to [GAP_MIN, GAP_MAX], renormalize to ``budget``, re-clip + renorm.

    A single clip-then-scale can push values back outside [min, max]; one extra
    clip/renorm pass keeps the sum exact to floating point while staying very
    close to the bounds for these smooth profiles. If a profile genuinely cannot
    fit the budget within the bounds (sum of mins > budget or sum of maxes <
    budget), the final renorm still makes the SUM exact (the budget is sacred);
    individual entries may then sit slightly outside [min, max], which the
    caller surfaces via total_gap_check (always == budget) -- not an issue for
    the smooth profiles here, whose feasible budgets bracket N*5.7.
    """
    gap = np.asarray(gap, dtype=float)
    gap = np.clip(gap, GAP_MIN_MM, GAP_MAX_MM)
    gap = gap * (budget / gap.sum())
    gap = np.clip(gap, GAP_MIN_MM, GAP_MAX_MM)
    gap = gap * (budget / gap.sum())
    return gap


def build_gap_profile(design: str, n_layers: int, budget: float,
                      shower_max_layer: int) -> np.ndarray:
    """Return a length-``n_layers`` gap vector summing to ``budget``.

    Each entry is (essentially) in [GAP_MIN, GAP_MAX]; the SUM equals the fixed
    budget exactly (the fairness constraint). See _renorm_to_budget for the
    clip+renorm contract.
    """
    n = int(n_layers)
    x = np.arange(n, dtype=float)

    if design == "uniform":
        gap = np.full(n, budget / n, dtype=float)
        # already sums to budget; renorm is a no-op safeguard / bound check.
        return _renorm_to_budget(gap, budget)

    if design == "bump":
        # Gaussian bump centered on shower max (width ~ N/6), thin floor else.
        sigma = max(2.0, n / 6.0)
        bump = np.exp(-0.5 * ((x - shower_max_layer) / sigma) ** 2)
        gap = GAP_MIN_MM + (GAP_MAX_MM - GAP_MIN_MM) * bump
        return _renorm_to_budget(gap, budget)

    if design == "front":
        # Monotone front-thick -> tail-thin linear taper.
        gap = GAP_MAX_MM - (GAP_MAX_MM - GAP_MIN_MM) * (x / max(n - 1, 1))
        return _renorm_to_budget(gap, budget)

    if design == "tail":
        # Reverse of `front`: thin front -> thick tail (the control).
        gap = GAP_MIN_MM + (GAP_MAX_MM - GAP_MIN_MM) * (x / max(n - 1, 1))
        return _renorm_to_budget(gap, budget)

    raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")


# --------------------------------------------------------------------------- #
# Forward binary execution
# --------------------------------------------------------------------------- #
def _design_point(cfg, n_layers: int, abs_mm: float,
                  gap_profile: Optional[np.ndarray]):
    """DesignPoint: fixed uniform absorber, e- @ 10 GeV, given gap profile.

    When ``gap_profile`` is None the scalar uniform gap (budget/N via -g) is
    used -- that path is only for the baseline shower-max run.
    """
    dp = default_design_point(cfg)
    abs_profile = tuple(float(abs_mm) for _ in range(int(n_layers)))
    gp = tuple(float(v) for v in gap_profile) if gap_profile is not None else None
    return dp.__class__(
        a=float(abs_mm), g=float(GAP_PER_LAYER_MM), energy=10000.0,
        n_layers=int(n_layers), transverse=400.0, particle="e-",
        abs_profile=abs_profile, gap_profile=gp,
    )


def _forward_edeps(cfg, ctrl, dp, n_events: int, seed: int) -> np.ndarray:
    """Run the forward binary in an isolated temp dir; return edeps_<seed>.

    Used only to locate the shower-max layer; stdout is discarded here.
    """
    fwd_bin = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    with tempfile.TemporaryDirectory(prefix="resceil_sm_") as wd:
        proc = subprocess.run([fwd_bin, *args], cwd=wd,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"{fwd_bin} rc={proc.returncode}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
        out = Path(wd) / f"edeps_{seed}"
        if not out.exists():
            cand = list(Path(wd).glob("edeps_*"))
            if not cand:
                raise FileNotFoundError(f"{fwd_bin} did not produce edeps_{seed}")
            out = cand[0]
        arr = np.loadtxt(out)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


def _forward_gap_resolution(cfg, ctrl, dp, n_events: int, seed: int):
    """Run the forward binary CAPTURING stdout; parse the Gap mean/std line.

    Returns (mu, sigma). The tools/sim.py wrapper discards stdout (DEVNULL), so
    we call the binary directly here (like fd_check_perlayer._run) but with
    stdout=PIPE to recover the printed "Gap : mean Edep ... Std-dev ..." line.
    """
    fwd_bin = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    with tempfile.TemporaryDirectory(prefix="resceil_") as wd:
        proc = subprocess.run([fwd_bin, *args], cwd=wd,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"{fwd_bin} rc={proc.returncode}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
        text = proc.stdout.decode(errors="replace")
    m = _GAP_LINE_RE.search(text)
    if not m:
        raise RuntimeError("could not parse 'Gap : mean Edep ... Std-dev ...' "
                           f"line from forward stdout (last 400 chars):\n"
                           f"{text[-400:]}")
    mu = float(m.group(1))
    sigma = float(m.group(2))
    return mu, sigma


def find_shower_max_layer(cfg, ctrl, n_layers: int, abs_mm: float, budget: float,
                          seed: int, n_events: int = 1000) -> int:
    """Argmax of the per-layer mean GAP edep from a baseline UNIFORM forward run.

    The baseline uses the uniform gap profile at the design's budget (not the
    config default 5.7), so shower max is measured for the exact geometry under
    test.
    """
    uniform = build_gap_profile("uniform", n_layers, budget, shower_max_layer=0)
    dp = _design_point(cfg, n_layers, abs_mm, gap_profile=uniform)
    edeps = _forward_edeps(cfg, ctrl, dp, n_events, seed)
    mean_E = edeps[:, 0]  # column 0 = mean gap edep per layer
    return int(np.argmax(mean_E))


# --------------------------------------------------------------------------- #
# JSONL append / resume
# --------------------------------------------------------------------------- #
def append_row(out_path: Path, row: dict):
    """Append one JSON line, flush + fsync so a kill can't lose it."""
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
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=str, required=True, choices=DESIGNS)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--n-events", type=int, default=8000)
    ap.add_argument("--seed", type=int, required=True,
                    help="single integer RNG seed")
    ap.add_argument("--gap-budget", type=float, default=None,
                    help="total Sigma(gap) mm; default = n_layers * 5.7")
    ap.add_argument("--abs", dest="abs_mm", type=float, default=ABSORBER_MM,
                    help="uniform absorber thickness [mm]")
    ap.add_argument("--shower-max", type=int, default=None,
                    help="shower-max layer index; skips the baseline probe run")
    ap.add_argument("--out-jsonl", type=str, required=True)
    args = ap.parse_args(argv)

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    n_layers = int(args.n_layers)
    n_events = int(args.n_events)
    seed = int(args.seed)
    design = args.design
    abs_mm = float(args.abs_mm)
    out_path = Path(args.out_jsonl)
    budget = (float(args.gap_budget) if args.gap_budget is not None
              else n_layers * GAP_PER_LAYER_MM)

    print(f"[config] {cfg['_source']}")
    print(f"[run] design={design} seed={seed} n_layers={n_layers} "
          f"n_events={n_events} gap_budget={budget:.6g}mm abs={abs_mm}mm")
    print(f"[out]  {out_path}")

    # --- resume -----------------------------------------------------------
    if already_done(out_path, design, seed):
        print(f"[resume] SKIP: (design={design}, seed={seed}) already present "
              f"in {out_path}")
        return 0

    # --- shower-max layer -------------------------------------------------
    if args.shower_max is not None:
        sm_layer = int(args.shower_max)
        print(f"[showermax] using provided --shower-max={sm_layer}")
    else:
        sm_layer = find_shower_max_layer(cfg, ctrl, n_layers, abs_mm, budget,
                                         seed, n_events=1000)
        print(f"[showermax] argmax per-layer gap edep -> layer {sm_layer} "
              f"(of {n_layers}) from baseline uniform probe")

    # --- build the gap profile for this design ----------------------------
    gap_profile = build_gap_profile(design, n_layers, budget, sm_layer)
    total_gap_check = float(gap_profile.sum())
    budget_rel = abs(total_gap_check - budget) / max(abs(budget), 1e-12)
    print(f"[profile] sum={total_gap_check:.6g}mm (budget={budget:.6g}mm, "
          f"rel.diff={budget_rel:.2e}) min={gap_profile.min():.3g} "
          f"max={gap_profile.max():.3g} argmax_layer={int(np.argmax(gap_profile))}")
    if budget_rel >= 1e-3:
        raise RuntimeError(f"budget mismatch {budget_rel:.2e} >= 1e-3 "
                           f"(sum={total_gap_check} budget={budget})")

    # --- forward run, parse resolution from stdout ------------------------
    dp = _design_point(cfg, n_layers, abs_mm, gap_profile)
    mu, sigma = _forward_gap_resolution(cfg, ctrl, dp, n_events, seed)
    resolution = sigma / mu if mu != 0 else float("nan")

    row = {
        "design": design,
        "seed": seed,
        "n_layers": n_layers,
        "n_events": n_events,
        "gap_budget": budget,
        "total_gap_check": total_gap_check,
        "mu": mu,
        "sigma": sigma,
        "resolution": resolution,
        "shower_max": sm_layer,
    }
    append_row(out_path, row)

    print(f"[row] {json.dumps(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

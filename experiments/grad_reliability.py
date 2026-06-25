"""High-statistics gradient-reliability measurement for the AD gap gradient.

Question: is the AD gradient ``d(E_vis)/d(uniform gap thickness)`` reliable, or
just noisy/biased? A single 400-event reverse pass gave
``dE_vis/d(uniform gap) ~= 1836 MeV/mm`` while the realized E_vis only changes
~200 MeV/mm. 400 events is too few to trust. This script produces, FOR ONE
SEED, an AD estimate and a finite-difference (FD) estimate of the same scalar
gradient at a UNIFORM design (all gap=g0, all abs=--abs), at high statistics.

Aggregating many seeds (run as separate condor jobs writing distinct JSONLs)
to mean +/- stderr + SNR settles whether AD and FD agree.

Definitions (at the uniform design, N layers):
  * E_vis = total_gap_signal = sum_l E_gap,l  (energy read out of the gaps).
  * AD gradient: seed the per-layer GAP outputs with the NET-SIGNAL adjoints
    (-1 each), so the reverse pass returns d(-E_vis)/d(design). Rows N..2N-1
    col0 are d(-E_vis)/d(gap_thick[i]); negating and summing gives
    d(E_vis)/d(uniform gap) = -sum_i pl[N+i, 0].
  * FD gradient (same seed): central difference of E_vis in g0 with half-step h.

Each invocation does 1 reverse + 3 forward sims at n_events (heavy at 5000);
the config subprocess timeout covers it. INCREMENTAL + RESUMABLE: if the given
seed is already present in --out-jsonl, the run is skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

from tools import sim as _sim
from tools.gap_signal_target import total_gap_signal, net_signal_adjoints


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="High-statistics AD-vs-FD gap-gradient reliability probe "
                    "(one seed per invocation; resumable).")
    p.add_argument("--n-layers", type=int, default=20,
                   help="number of layers N (default 20)")
    p.add_argument("--n-events", type=int, default=5000,
                   help="events per simulation (default 5000)")
    p.add_argument("--seed", type=int, required=True,
                   help="single RNG seed for this measurement")
    p.add_argument("--gap", type=float, default=5.7,
                   help="uniform gap thickness g0 [mm] (default 5.7)")
    p.add_argument("--abs", dest="abs_", type=float, default=3.5,
                   help="uniform absorber thickness [mm] (default 3.5)")
    p.add_argument("--h", type=float, default=0.2,
                   help="FD half-step on gap thickness [mm] (default 0.2)")
    p.add_argument("--out-jsonl", type=str, required=True,
                   help="JSONL checkpoint file (one row appended per seed)")
    return p.parse_args(argv)


def _uniform_dp(n_layers, abs_mm, gap_mm):
    """A uniform design point: N layers, all abs=abs_mm, all gap=gap_mm."""
    dp = _sim.default_design_point()
    return dataclasses.replace(
        dp, n_layers=int(n_layers), a=float(abs_mm), g=float(gap_mm),
        abs_profile=None, gap_profile=None)


def seed_already_done(out_path: Path, seed: int) -> bool:
    if not out_path.exists():
        return False
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("seed", -10**9)) == int(seed):
                return True
    return False


def append_row(out_path: Path, row: dict):
    """Append one JSON line, flush + fsync so a kill can't lose it."""
    with open(out_path, "a", buffering=1) as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def measure(n_layers, n_events, seed, gap, abs_mm, h, ctrl):
    """Return the result-row dict for one seed (AD + FD gap gradient)."""
    N = int(n_layers)
    seed = int(seed)

    # --- AD: reverse pass seeding GAP outputs with net-signal adjoints (-1).
    dp0 = _uniform_dp(N, abs_mm, gap)
    pl = _sim.run_reverse_per_layer(
        dp0, adjoints=np.zeros(N, dtype=float), n_events=n_events, seed=seed,
        ctrl=ctrl, gap_adjoints=net_signal_adjoints(N))
    if pl is None:
        ad_grad = float("nan")
    else:
        # rows N..2N-1 col0 = d(-E_vis)/d(gap_thick[i]); negate+sum for
        # d(E_vis)/d(uniform gap).
        ad_grad = float(-np.sum(pl[N:2 * N, 0]))

    # --- FD: central difference of E_vis in g0 (same seed).
    evis0 = total_gap_signal(_uniform_dp(N, abs_mm, gap), n_events, [seed], ctrl)
    evisp = total_gap_signal(_uniform_dp(N, abs_mm, gap + h), n_events, [seed], ctrl)
    evism = total_gap_signal(_uniform_dp(N, abs_mm, gap - h), n_events, [seed], ctrl)
    fd_grad = float((evisp - evism) / (2.0 * float(h)))

    return {
        "seed": seed,
        "n_layers": N,
        "n_events": int(n_events),
        "gap": float(gap),
        "h": float(h),
        "ad_grad": ad_grad,
        "fd_grad": fd_grad,
        "evis0": float(evis0),
        "evisp": float(evisp),
        "evism": float(evism),
    }


def main(argv=None):
    args = parse_args(argv)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_already_done(out_path, args.seed):
        print(f"[resume] SKIP seed={args.seed} already in {out_path}")
        sys.stdout.flush()
        return

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)

    row = measure(args.n_layers, args.n_events, args.seed, args.gap,
                  args.abs_, args.h, ctrl)
    append_row(out_path, row)

    ratio = (row["ad_grad"] / row["fd_grad"]
             if row["fd_grad"] not in (0.0,) and np.isfinite(row["fd_grad"])
             else float("nan"))
    print(f"[done] seed={row['seed']} N={row['n_layers']} "
          f"n_events={row['n_events']} gap={row['gap']} h={row['h']} "
          f"ad_grad={row['ad_grad']:.4f} fd_grad={row['fd_grad']:.4f} "
          f"ratio(ad/fd)={ratio:.4f} "
          f"evis0={row['evis0']:.2f} evisp={row['evisp']:.2f} "
          f"evism={row['evism']:.2f} -> appended to {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

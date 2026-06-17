"""Phase-0 gate: per-layer gradient validation for the generalized geometry.

For a chosen set of layers and for each thickness parameter (absorber `a`, gap
`g`), this performs a **3-way cross-check** of d(total_edep)/d(param_of_layer_i):

  * reverse-AD : one reverse run with adjoints = ones -> the output bar equals
                 total_edep = sum_l edep_l; ``barInputsPerLayer`` then gives
                 d(total)/d a_i and d(total)/d g_i for every layer at once.
  * forward-AD : one forward run seeding only layer i's parameter dot=1 (via the
                 new ``--abs-layer i:val:1`` / ``--gap-layer i:val:1`` flags);
                 the summed output dot equals d(total)/d(param_i).
  * central FD : two value runs at param_i +/- h (same seed) ->
                 (O(+h) - O(-h)) / (2h).

All three runs share the SAME seed so they describe identical shower
realizations (mandatory: see the note in tools/sim.py::_selftest). With exact AD
the forward and reverse estimates must match to ~machine precision; the FD
estimate matches to O(h^2) + transport non-smoothness, accepted within
``reliability.fd_rel_tol`` from config.yaml.

Also checks the aggregate consistency: sum_i d(total)/d a_i (per-layer file) must
equal the legacy ``barInputs`` absorber row from the same run (and likewise gap).

Usage:
    python -m tools.fd_check_perlayer                 # defaults from config
    python -m tools.fd_check_perlayer --layers 3,8,15 --n-events 1000 --n-layers 50
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .sim import (load_config, default_design_point, ctrl_flags_from_config,
                  _common_args)


def _run(binary: str, args: list, seed: int, want: str):
    """Run ``binary`` in an isolated temp dir; return the requested output array.

    want = 'edeps'             -> (n_layers, 4) array
    want = 'barInputsPerLayer' -> (2N+1, 2) array (rows: abs[0..N-1], gap[0..N-1], energy)
    want = 'barInputs'         -> (3, 2) array (rows: abs, gap, energy)
    """
    with tempfile.TemporaryDirectory(prefix="hepemshow_fd_") as wd:
        proc = subprocess.run([binary, *args], cwd=wd,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"{binary} rc={proc.returncode}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
        name = f"edeps_{seed}" if want == "edeps" else want
        out = Path(wd) / name
        if not out.exists() and want == "edeps":
            cand = list(Path(wd).glob("edeps_*"))
            if cand:
                out = cand[0]
        if not out.exists():
            raise FileNotFoundError(f"{binary} did not produce {name}")
        arr = np.loadtxt(out)  # '#' header lines are skipped by loadtxt
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices (default: spread across depth)")
    ap.add_argument("--n-events", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    print(f"[config] {cfg['_source']}")
    fwd_bin = cfg["paths"]["forward_bin"]
    rev_bin = cfg["paths"]["reverse_bin"]
    ctrl = ctrl_flags_from_config(cfg)
    rel_tol = float(cfg["reliability"]["fd_rel_tol"])
    h_a = float(cfg["reliability"]["fd_step"]["a"])
    h_g = float(cfg["reliability"]["fd_step"]["g"])

    dp = default_design_point(cfg)
    if args.n_layers is not None:
        dp = dp.__class__(a=dp.a, g=dp.g, energy=dp.energy, n_layers=args.n_layers,
                          transverse=dp.transverse, particle=dp.particle)
    n = int(args.n_events if args.n_events is not None else cfg["stats"]["dev_events"])
    seed = int(args.seed if args.seed is not None else cfg["stats"]["dev_seeds"][0])
    N = dp.n_layers
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # default to the shower-maximum region, where the per-layer gradient is
        # strongly above MC noise so the FD comparison is meaningful (deep tail
        # layers carry near-zero gradient and give noisy finite differences).
        layers = sorted(set(int(round(f * (N - 1))) for f in (0.06, 0.12, 0.20)))
    print(f"[design] {dp}")
    print(f"[run]    n_events={n} seed={seed}  layers={layers}  fd_rel_tol={rel_tol}")

    base = _common_args(dp, ctrl, n, seed, cfg, seeded_param=None)

    # --- one reverse run: per-layer gradients of total_edep (adjoints = ones) ---
    adj = ":".join(repr(1.0) for _ in range(N))
    rev_args = base + ["-b", adj]
    bar_pl = _run(rev_bin, rev_args, seed, "barInputsPerLayer")
    bar_agg = _run(rev_bin, rev_args, seed, "barInputs")
    if bar_pl.shape[0] != 2 * N + 1:
        print(f"  FAIL: barInputsPerLayer has {bar_pl.shape[0]} rows, expected {2*N+1}")
        return 1
    rev_da = bar_pl[0:N, 0]          # d(total)/d a_i
    rev_dg = bar_pl[N:2*N, 0]        # d(total)/d g_i

    # --- aggregate consistency: sum_i per-layer == legacy barInputs row ---
    agg_ok = True
    for label, perlayer, aggrow in (("a", rev_da, bar_agg[0, 0]),
                                    ("g", rev_dg, bar_agg[1, 0])):
        s = float(perlayer.sum())
        denom = max(abs(aggrow), 1e-12)
        rel = abs(s - aggrow) / denom
        tag = "OK" if rel < 1e-6 else "FAIL"
        if rel >= 1e-6:
            agg_ok = False
        print(f"[aggregate {label}] sum(per-layer)={s:.6g}  legacy barInputs={aggrow:.6g}"
              f"  rel.diff={rel:.2e}  [{tag}]")

    # --- per-layer 3-way checks ---
    def three_way(i: int, which: str, val: float, h: float, rev_grad: float):
        flag = f"--{which}-layer"
        fwd = _run(fwd_bin, base + [flag, f"{i}:{val}:1"], seed, "edeps")
        fwd_grad = float(fwd[:, 2].sum())                      # summed output dot
        op = _run(fwd_bin, base + [flag, f"{i}:{val + h}"], seed, "edeps")
        om = _run(fwd_bin, base + [flag, f"{i}:{val - h}"], seed, "edeps")
        fd_grad = float((op[:, 0].sum() - om[:, 0].sum()) / (2 * h))
        scale = max(abs(rev_grad), abs(fwd_grad), abs(fd_grad), 1e-12)
        rel_fr = abs(fwd_grad - rev_grad) / scale
        rel_fd = abs(fd_grad - 0.5 * (fwd_grad + rev_grad)) / scale
        ad_ok = rel_fr < 1e-3
        fd_ok = rel_fd < rel_tol
        status = "OK" if (ad_ok and fd_ok) else "FAIL"
        print(f"  layer {i:>3} d(tot)/d{which}: rev={rev_grad: .6g}  fwd={fwd_grad: .6g}"
              f"  fd={fd_grad: .6g}  | AD rel={rel_fr:.2e}  FD rel={rel_fd:.2e}  [{status}]")
        return ad_ok and fd_ok

    print("[per-layer absorber]")
    ok = agg_ok
    for i in layers:
        ok &= three_way(i, "abs", float(dp.a), h_a, float(rev_da[i]))
    print("[per-layer gap]")
    for i in layers:
        ok &= three_way(i, "gap", float(dp.g), h_g, float(rev_dg[i]))

    print("[fd_check_perlayer] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

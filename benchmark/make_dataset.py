"""Build a sensitivity-reasoning ground-truth dataset (see docs/BENCHMARK.md).

For each (observable, parameter, design_point) triple we compute the exact AD
sensitivity with error bars and a reliability flag, and write one JSONL line.
The exact flags/seeds/event-counts are recorded for reproducibility.

    python benchmark/make_dataset.py --quick --out benchmark/data/quick.jsonl
    python benchmark/make_dataset.py --full  --out benchmark/data/full.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

# allow running as a script from the agentic/ root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import sim as _sim                       # noqa: E402
from tools import observables as _obs               # noqa: E402
from tools import reliability as _rel               # noqa: E402
from tools.schemas import DesignPoint, DIFFERENTIABLE_PARAMS  # noqa: E402


def grid(quick: bool):
    if quick:
        a_vals, g_vals, e_vals, parts = [2.3], [5.7], [10000.0, 25000.0], ["e-"]
    else:
        a_vals = [1.5, 2.3, 3.0]
        g_vals = [4.0, 5.7, 8.0]
        e_vals = [5000.0, 10000.0, 25000.0]
        parts = ["e-", "gamma"]
    obs = _obs.list_observables()
    for o, wrt, a, g, e, p in itertools.product(obs, DIFFERENTIABLE_PARAMS,
                                                a_vals, g_vals, e_vals, parts):
        yield o, wrt, DesignPoint(a=a, g=g, energy=e, particle=p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-events", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = _sim.load_config()
    n_events = args.n_events or int(cfg["stats"]["bench_events"])
    seeds = cfg["stats"]["bench_seeds"]
    ctrl = _sim.ctrl_flags_from_config(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quick = args.quick or not args.full

    n = 0
    with open(out, "w") as fh:
        for observable, wrt, dp in grid(quick):
            sens = _rel.annotated_sensitivity(observable, wrt, dp,
                                              n_events=n_events, seeds=seeds,
                                              method="forward-AD", cross_check=True)
            item = {
                "id": f"{observable}__{wrt}__a{dp.a}_g{dp.g}_E{int(dp.energy)}_{dp.particle}",
                "observable": observable,
                "wrt": wrt,
                "design_point": asdict(dp),
                "ctrl_flags": asdict(ctrl),
                "stats": {"n_events": n_events, "seeds": list(seeds)},
                "ground_truth": {
                    "value": sens.value, "stderr": sens.stderr,
                    "sign": sens.sign, "log10_abs": sens.log10_abs,
                    "reliability": sens.reliability, "method": sens.method,
                },
            }
            fh.write(json.dumps(item, default=str) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"  ... {n} items")
    print(f"[make_dataset] wrote {n} items -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

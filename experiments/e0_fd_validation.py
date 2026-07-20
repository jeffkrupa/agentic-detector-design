"""E0: validate finite-difference machinery against AD on a linear control.

Observable: total_edep (monotone in absorber thickness and beam energy).
For each parameter we compute one forward-AD sensitivity and an epsilon-scan of
central-difference FD using COMMON RANDOM NUMBERS (the same seed list is passed
to the +eps and -eps runs, so shared RNG state cancels part of the MC noise).

Decision rule (see docs/research/EXPERIMENTS.md E0):
  * AD/FD plateau near ~0.8 over a clean epsilon window  -> FD validated.
  * AD/FD stuck ~0.03-0.1                                 -> serious discrepancy.
  * no plateau / noise-dominated                          -> CRN/noise failure.

Small by design: forward-only, N<=10000, dev seeds. No SLURM, no reverse-AD.
The independent +/-eps sim invocations are dispatched through a thread pool
(each sim is a CPU-bound subprocess in its own temp dir, so this is safe).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tools import observables as _obs
from tools import sim as _sim


# epsilon grids per parameter (physical units): mm for 'a', MeV for 'energy'.
EPS_GRID = {
    "a": [0.005, 0.02, 0.05, 0.1, 0.25, 0.5],
    "energy": [1.0, 5.0, 20.0, 50.0, 100.0, 200.0],
}


def _observe_point(name, dp, n_events, seeds, ctrl):
    o = _obs.observe(name, dp, n_events=n_events, seeds=seeds, ctrl=ctrl)
    return float(o.value), float(o.stderr)


def run(wrt, n_events, seeds, workers, observable="total_edep", eps_grid=None):
    cfg = _sim.load_config()
    dp = _sim.default_design_point(cfg)
    ctrl = _sim.ctrl_flags_from_config(cfg)
    base = dp.get(wrt)
    eps_grid = eps_grid if eps_grid is not None else EPS_GRID[wrt]

    # Build the full job list: AD once, then (+eps, -eps) for each epsilon.
    # All sims are independent -> dispatch concurrently.
    def _ad():
        s = _obs.sensitivity(observable, wrt, dp, n_events=n_events,
                             seeds=seeds, ctrl=ctrl, method="forward-AD")
        return ("ad", float(s.value), float(s.stderr))

    def _pt(tag, delta):
        v, se = _observe_point(observable, dp.with_param(wrt, base + delta),
                               n_events, seeds, ctrl)
        return (tag, v, se)

    jobs = [_ad]
    for eps in eps_grid:
        jobs.append(lambda e=eps: _pt(f"+{e}", e))
        jobs.append(lambda e=eps: _pt(f"-{e}", -e))

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tag, v, se in ex.map(lambda f: f(), jobs):
            results[tag] = (v, se)

    ad_val, ad_se = results["ad"]
    print(f"\n# {observable}  wrt={wrt}  N={n_events}  seeds={seeds}")
    print(f"# AD = {ad_val:+.4g}  (SE {ad_se:.3g})")
    print("epsilon | AD | FD | AD/FD | FD_SE | notes")
    for eps in eps_grid:
        vp, sp = results[f"+{eps}"]
        vm, sm = results[f"-{eps}"]
        fd = (vp - vm) / (2.0 * eps)
        fd_se = np.sqrt(sp ** 2 + sm ** 2) / (2.0 * eps)
        ratio = ad_val / fd if fd != 0 else float("nan")
        note = "noise-dominated" if abs(fd) < 3 * fd_se else ""
        print(f"{eps:>7g} | {ad_val:+.4g} | {fd:+.4g} | {ratio:+.3f} | "
              f"{fd_se:.3g} | {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrt", nargs="+", default=["a", "energy"])
    ap.add_argument("--observable", default="total_edep")
    ap.add_argument("--eps", type=float, nargs="+", default=None)
    ap.add_argument("-n", "--n-events", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    for wrt in args.wrt:
        run(wrt, args.n_events, args.seeds, args.workers,
            observable=args.observable, eps_grid=args.eps)


if __name__ == "__main__":
    main()

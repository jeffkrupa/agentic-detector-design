"""E0b (SLURM): per-seed AD + FD for d(front_fraction)/da, common random numbers.

One job = one seed. For that seed we compute, on the default geometry (a=2.3mm):
  * forward-AD sensitivity of front_fraction w.r.t. absorber thickness a, and
  * observe(front_fraction) at a+eps and a-eps for each eps (CRN: the +/-eps
    runs share this seed, so shared RNG cancels part of the MC noise).
Writes ONE JSONL row (seed, AD, per-eps +/- values) plus an audit metadata block.
Resume-safe: if a row for this (seed, N, eps-list) already exists in the output
file, the job skips and exits 0.

Aggregation across seeds is done separately by experiments/e0b_aggregate.py.
Scope: front_fraction / wrt=a ONLY. No other observable, no scans.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import tools.sim as _simmod
from tools import observables as _obs
from tools import sim as _sim

# Forward N<=3000 (~570s uncontended); keep margin above the shared 300s cap
# (in-process only; does NOT edit config.yaml). Precedent: _proxy_*.py.
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

OBSERVABLE = "front_fraction"
WRT = "a"


def _audit_meta(n_events, seed, eps):
    """Auditable provenance: git state + exact config used for this row."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _git(args):
        try:
            return subprocess.check_output(["git", "-C", repo, *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = _git(["status", "--short"]) or ""
    driver = os.path.abspath(__file__)
    try:
        driver_size = os.path.getsize(driver)
    except OSError:
        driver_size = None
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_diff_stat": _git(["diff", "--stat"]),
        # driver script is untracked (git_diff_stat empty) -> record its size as a
        # cheap provenance signal, plus how many untracked paths exist.
        "git_untracked_count": sum(1 for ln in status.splitlines() if ln.startswith("??")),
        "driver_path": driver,
        "driver_size_bytes": driver_size,
        "command": f"e0b_ffront_slurm --seed {seed} --n-events {n_events} "
                   f"--eps {' '.join(str(e) for e in eps)}",
        "observable": OBSERVABLE,
        "parameter": WRT,
        "eps": list(eps),
        "n_events": n_events,
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("-n", "--n-events", type=int, default=3000)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    seed, n, eps = args.seed, args.n_events, args.eps

    # --- resume-skip: identical (seed, N, eps) row already present? ----------
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get("seed") == seed and row.get("n_events") == n
                        and row.get("eps") == list(eps)):
                    print(f"[e0b] seed={seed} N={n} eps={eps} already present -> skip")
                    return

    cfg = _sim.load_config()
    dp = _sim.default_design_point(cfg)
    ctrl = _sim.ctrl_flags_from_config(cfg)
    base = dp.get(WRT)
    seeds = [seed]

    ad = _obs.sensitivity(OBSERVABLE, WRT, dp, n_events=n, seeds=seeds,
                          ctrl=ctrl, method="forward-AD")

    per_eps = {}
    for e in eps:
        op = _obs.observe(OBSERVABLE, dp.with_param(WRT, base + e), n_events=n,
                          seeds=seeds, ctrl=ctrl)
        om = _obs.observe(OBSERVABLE, dp.with_param(WRT, base - e), n_events=n,
                          seeds=seeds, ctrl=ctrl)
        per_eps[str(e)] = {
            "plus_val": float(op.value), "plus_se": float(op.stderr),
            "minus_val": float(om.value), "minus_se": float(om.stderr),
        }

    row = {
        "seed": seed, "n_events": n, "eps": list(eps),
        "ad_val": float(ad.value), "ad_se": float(ad.stderr),
        "per_eps": per_eps,
        "meta": _audit_meta(n, seed, eps),
    }
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[e0b] wrote seed={seed} N={n} AD={ad.value:+.4g} SE={ad.stderr:.3g} "
          f"-> {args.out_jsonl}")


if __name__ == "__main__":
    main()

"""Aggregate E0b per-seed rows -> AD/FD table with the trust check.

Reads experiments/e0b_ff_slurm_s*.jsonl (or an explicit glob), combines across
seeds: AD = mean(ad_val); AD_SE = sqrt(sum(ad_se^2))/n (independent seeds).
FD(eps) per seed = (plus_val - minus_val)/(2 eps) with CRN; combined across seeds
by mean, FD_SE = std/sqrt(n) (empirical across-seed spread).

Trust criterion (see docs/research/EXPERIMENTS.md E0b): AD/FD in ~[0.8,1.2] for
>=2 adjacent eps with resolved AD (|AD|>3 AD_SE) and resolved FD (|FD|>3 FD_SE).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def load_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/e0b_ff_slurm_s*.jsonl")
    args = ap.parse_args()

    rows = load_rows(args.glob)
    if not rows:
        print(f"[e0b-agg] no rows matched {args.glob}")
        return

    n_seed = len({r["seed"] for r in rows})
    n_events = rows[0]["n_events"]
    eps_list = rows[0]["eps"]
    n_total = n_events * n_seed

    ad_vals = np.array([r["ad_val"] for r in rows])
    ad_ses = np.array([r["ad_se"] for r in rows])
    ad = float(np.mean(ad_vals))
    ad_se = float(np.sqrt(np.sum(ad_ses ** 2)) / len(ad_ses))

    print(f"# front_fraction  wrt=a  seeds={n_seed}  N/seed={n_events}  "
          f"N_total={n_total}")
    print(f"# AD = {ad:+.5g}  AD_SE = {ad_se:.4g}  "
          f"(resolved={abs(ad) > 3 * ad_se})")
    print("epsilon | AD | AD_SE | FD | FD_SE | AD/FD | notes")

    in_band = []
    for e in eps_list:
        key = str(e)
        fd_s = np.array([
            (r["per_eps"][key]["plus_val"] - r["per_eps"][key]["minus_val"]) / (2 * e)
            for r in rows if key in r.get("per_eps", {})
        ])
        fd = float(np.mean(fd_s))
        fd_se = float(np.std(fd_s) / np.sqrt(len(fd_s))) if len(fd_s) > 1 else float("nan")
        ratio = ad / fd if fd != 0 else float("nan")
        ad_ok = abs(ad) > 3 * ad_se
        fd_ok = np.isfinite(fd_se) and abs(fd) > 3 * fd_se
        band = 0.8 <= ratio <= 1.2
        in_band.append(band and ad_ok and fd_ok)
        note = []
        if not ad_ok:
            note.append("AD unresolved")
        if not fd_ok:
            note.append("FD noise-dominated")
        if band and ad_ok and fd_ok:
            note.append("IN BAND")
        print(f"{e:>7g} | {ad:+.5g} | {ad_se:.4g} | {fd:+.5g} | {fd_se:.4g} | "
              f"{ratio:+.3f} | {', '.join(note)}")

    # >=2 ADJACENT eps in band?
    adjacent = any(in_band[i] and in_band[i + 1] for i in range(len(in_band) - 1))
    print(f"\n# VERDICT: {'PASS (>=2 adjacent eps in [0.8,1.2] w/ resolved AD+FD)' if adjacent else 'NOT PASSED'}")


if __name__ == "__main__":
    main()

"""Wave-3 derivative microscopy: per-event, same-seed AD-vs-FD decomposition.

Consumes the env-gated per-event dumps (HEPEMSHOW_EVENT_DUMP) written by the
`knob/derivative-microscopy` hepemshow branch: one line per event with
    eventID  signature(hex)  trackCount  totalSteps  E_0..E_{L-1}  D_0..D_{L-1}
where E is the per-layer combined energy deposit and D its forward-AD dot.

For a same-seed triplet of runs at (g-h, g, g+h):
  - events whose decision-path signature is identical across the triplet
    ("same-path") must satisfy per-event AD == central FD up to O(h^2);
  - events whose signature diverges carry FD mass that pathwise AD cannot
    represent (missing score-function term).
The core-window deficit decomposes exactly:
    mean(FD) - mean(AD) = f_same*(FD-AD | same) + f_div*(FD-AD | diverged).

Usage:
  python fidelity/wave3_microscopy.py \
      --triplet SEED H CENTER_DUMP MINUS_DUMP PLUS_DUMP [--triplet ...] \
      [--window 5 18] [--out fidelity/wave3_microscopy_summary.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_dump(path: str):
    """Parse a per-event dump file -> (event_ids, sigs, ntrk, nsteps, E, D)."""
    ev, sigs, ntrk, nsteps, evals, dvals = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            n_layers = (len(parts) - 4) // 2
            ev.append(int(parts[0]))
            sigs.append(parts[1])
            ntrk.append(int(parts[2]))
            nsteps.append(int(parts[3]))
            evals.append([float(x) for x in parts[4:4 + n_layers]])
            dvals.append([float(x) for x in parts[4 + n_layers:4 + 2 * n_layers]])
    return (np.asarray(ev), np.asarray(sigs), np.asarray(ntrk),
            np.asarray(nsteps), np.asarray(evals), np.asarray(dvals))


def analyze_triplet(seed: int, h: float, center: str, minus: str, plus: str,
                    lo: int, hi: int) -> dict:
    ev0, sig0, ntrk0, nst0, e0, d0 = parse_dump(center)
    evm, sigm, ntrkm, nstm, em, _ = parse_dump(minus)
    evp, sigp, ntrkp, nstp, ep, _ = parse_dump(plus)

    n = min(len(ev0), len(evm), len(evp))
    if not (len(ev0) == len(evm) == len(evp)):
        print(f"[warn] seed {seed}: event-count mismatch "
              f"{len(ev0)}/{len(evm)}/{len(evp)}; truncating to {n}")
    assert (ev0[:n] == evm[:n]).all() and (ev0[:n] == evp[:n]).all(), \
        "event index mismatch across triplet"

    sig0, sigm, sigp = sig0[:n], sigm[:n], sigp[:n]
    e0, em, ep, d0 = e0[:n], em[:n], ep[:n], d0[:n]
    nst0, nstm, nstp = nst0[:n], nstm[:n], nstp[:n]

    same = (sigm == sig0) & (sigp == sig0)
    div = ~same
    n_same = int(same.sum())
    n_div = int(div.sum())

    fd = (ep - em) / (2.0 * h)              # (n, L) per-event central FD
    win = slice(lo, hi + 1)
    fd_w = fd[:, win].sum(axis=1)           # per-event core-window FD
    ad_w = d0[:, win].sum(axis=1)           # per-event core-window AD

    # --- exact decomposition of the core-window mean ---
    mean_fd = float(fd_w.mean())
    mean_ad = float(ad_w.mean())
    same_fd_contrib = float(fd_w[same].sum() / n)
    div_fd_contrib = float(fd_w[div].sum() / n)
    same_ad_contrib = float(ad_w[same].sum() / n)
    div_ad_contrib = float(ad_w[div].sum() / n)

    # --- same-path AD vs FD ---
    diff_w = ad_w[same] - fd_w[same]
    abs_diff = np.abs(diff_w)
    q = (lambda p: float(np.quantile(abs_diff, p))) if n_same else (lambda p: None)
    # outliers: largest same-path |AD-FD| events, localized to their worst layer
    outliers = []
    if n_same:
        idx_same = np.nonzero(same)[0]
        order = np.argsort(abs_diff)[::-1][:10]
        for k in order:
            i = int(idx_same[k])
            layer_diff = d0[i] - fd[i]
            worst_layer = int(np.abs(layer_diff).argmax())
            outliers.append({
                "event": int(ev0[i]),
                "ad_window": float(ad_w[i]),
                "fd_window": float(fd_w[i]),
                "abs_diff_window": float(abs_diff[k]),
                "worst_layer": worst_layer,
                "ad_worst_layer": float(d0[i, worst_layer]),
                "fd_worst_layer": float(fd[i, worst_layer]),
                "steps_center": int(nst0[i]),
            })

    per_layer = {
        "layers": list(range(e0.shape[1])),
        "ad_mean_all": d0.mean(axis=0).tolist(),
        "fd_mean_all": fd.mean(axis=0).tolist(),
        "ad_mean_same": d0[same].mean(axis=0).tolist() if n_same else None,
        "fd_mean_same": fd[same].mean(axis=0).tolist() if n_same else None,
    }

    steps_delta = np.abs(nstp - nstm) + np.abs(nst0 - nstm)
    res = {
        "seed": seed,
        "h": h,
        "n_events": n,
        "window": [lo, hi],
        "n_same_path": n_same,
        "n_diverged": n_div,
        "frac_diverged": n_div / n,
        "core_window": {
            "mean_ad_all": mean_ad,
            "mean_fd_all": mean_fd,
            "ratio_ad_fd_all": mean_ad / mean_fd if mean_fd else None,
            "same_path": {
                "fd_contrib_to_total_mean": same_fd_contrib,
                "ad_contrib_to_total_mean": same_ad_contrib,
                "mean_ad_minus_fd": float(diff_w.mean()) if n_same else None,
                "se_ad_minus_fd": (float(diff_w.std(ddof=1) / math.sqrt(n_same))
                                   if n_same > 1 else None),
                "abs_diff_quantiles": {"q50": q(0.5), "q90": q(0.9),
                                       "q99": q(0.99),
                                       "max": float(abs_diff.max()) if n_same else None},
            },
            "diverged": {
                "fd_contrib_to_total_mean": div_fd_contrib,
                "ad_contrib_to_total_mean": div_ad_contrib,
            },
            "decomposition_check": {
                "fd_same_plus_div": same_fd_contrib + div_fd_contrib,
                "ad_same_plus_div": same_ad_contrib + div_ad_contrib,
            },
        },
        "same_path_outliers": outliers,
        "steps": {
            "mean_steps_center": float(nst0.mean()),
            "mean_abs_step_delta_within_triplet": float(steps_delta.mean()),
            "same_path_all_steps_equal": bool(
                ((nst0 == nstm) & (nst0 == nstp))[same].all()) if n_same else None,
        },
        "per_layer": per_layer,
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", nargs=5, action="append", required=True,
                    metavar=("SEED", "H", "CENTER", "MINUS", "PLUS"),
                    help="seed h center_dump minus_dump plus_dump")
    ap.add_argument("--window", nargs=2, type=int, default=[5, 18])
    ap.add_argument("--out", default="fidelity/wave3_microscopy_summary.json")
    args = ap.parse_args()

    lo, hi = args.window
    results = []
    for seed_s, h_s, center, minus, plus in args.triplet:
        r = analyze_triplet(int(seed_s), float(h_s), center, minus, plus, lo, hi)
        results.append(r)

    print(f"\n=== wave-3 microscopy: core window L{lo}-L{hi} ===")
    hdr = (f"{'seed':>4} {'h':>5} {'N':>5} {'div%':>6} {'AD':>9} {'FD':>9} "
           f"{'AD/FD':>6} | {'FD_same':>9} {'FD_div':>9} {'AD_same':>9} "
           f"{'AD_div':>9} | {'same AD-FD':>11} {'q99|d|':>8}")
    print(hdr)
    for r in results:
        cw = r["core_window"]
        sp = cw["same_path"]
        dv = cw["diverged"]
        print(f"{r['seed']:>4} {r['h']:>5} {r['n_events']:>5} "
              f"{100*r['frac_diverged']:>5.1f}% "
              f"{cw['mean_ad_all']:>9.4f} {cw['mean_fd_all']:>9.4f} "
              f"{cw['ratio_ad_fd_all']:>6.3f} | "
              f"{sp['fd_contrib_to_total_mean']:>9.4f} "
              f"{dv['fd_contrib_to_total_mean']:>9.4f} "
              f"{sp['ad_contrib_to_total_mean']:>9.4f} "
              f"{dv['ad_contrib_to_total_mean']:>9.4f} | "
              f"{sp['mean_ad_minus_fd']:>11.6f} "
              f"{sp['abs_diff_quantiles']['q99']:>8.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"window": [lo, hi], "triplets": results}, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

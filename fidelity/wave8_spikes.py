"""Wave-8 spike mining: per-event derivative-spike population, unsevered regime.

Consumes the env-gated per-event dumps (HEPEMSHOW_EVENT_DUMP, wave-3 format):
    eventID  signature(hex)  trackCount  totalSteps  E_0..E_{L-1}  D_0..D_{L-1}
for three severing configs (all other knobs off):
    ref : -x 2 -y 1 -B 1   (canonical severing)
    uns : -x 0 -y 0 -B 0   (fully unsevered)
    int : -x 0 -y 1 -B 1   (descendant severing off, grazing/backward on)
plus one unsevered absorber-seed run.

Per run it characterizes the distribution of the per-event core-window
(L5-18) dot sum W_e = sum_{l in window} D_l: quantiles of |W_e|, variance,
tail concentration (share of the second central moment carried by the top
1% / 0.1% events), NaN/inf counts, raw/median/trimmed-mean locators, and
the anatomy of the largest spikes. Cross-config: whether unsevered spike
events are also spiky in the intermediate config (same seed => same primal
paths as long as severing is derivative-only).

Usage:
  python fidelity/wave8_spikes.py --base <run-base-dir> \
      [--window 5 18] [--out fidelity/wave8_spikes_summary.json] \
      [--spikes-out fidelity/wave8_spike_events.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FD_TRUTH = {"gap": (195.8, 9.4), "abs": (2233.7, 14.8)}  # 1M-event core FD


def parse_dump(path: Path):
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


def trimmed_mean(x: np.ndarray, frac: float) -> float:
    x = np.sort(x)
    k = int(math.floor(frac * len(x)))
    if k > 0:
        x = x[k:len(x) - k]
    return float(x.mean())


def tail_var_share(w: np.ndarray, frac: float) -> float:
    """Share of the second central moment carried by the top-frac events
    ranked by |W_e - mean|."""
    dev2 = (w - w.mean()) ** 2
    k = max(1, int(math.ceil(frac * len(w))))
    top = np.sort(dev2)[::-1][:k]
    tot = dev2.sum()
    return float(top.sum() / tot) if tot > 0 else float("nan")


def dipole_metrics(D: np.ndarray) -> dict:
    """Adjacent-bin opposite-sign dot pairs (relabeling dipoles): per event the
    largest min(|D_j|, |D_{j+1}|) over pairs with D_j*D_{j+1} < 0."""
    A, B = D[:, :-1], D[:, 1:]
    amp = np.where((A * B) < 0, np.minimum(np.abs(A), np.abs(B)), 0.0)
    amp = np.where(np.isfinite(amp), amp, 0.0)
    dip_amp = amp.max(axis=1)
    dip_layer = amp.argmax(axis=1)
    big = dip_amp > 1e4
    res = [float(abs(D[i, dip_layer[i]] + D[i, dip_layer[i] + 1]) / dip_amp[i])
           for i in np.where(big)[0]]
    hist = np.bincount(dip_layer[big], minlength=D.shape[1])
    return {
        "n_dipole_gt_1e4": int(big.sum()),
        "n_dipole_gt_1e5": int((dip_amp > 1e5).sum()),
        "n_dipole_gt_1e6": int((dip_amp > 1e6).sum()),
        "dipole_left_layer_hist_gt_1e4": hist.tolist(),
        "median_cancellation_residual": float(np.median(res)) if res else None,
        "max_dipole_amp": float(dip_amp.max()),
    }


def characterize(tag: str, dump: Path, lo: int, hi: int) -> dict:
    ev, sig, ntrk, nst, E, D = parse_dump(dump)
    win = slice(lo, hi + 1)
    W = D[:, win].sum(axis=1)
    S50 = D.sum(axis=1)
    finite = np.isfinite(W)
    nonfinite_any_layer = ~np.isfinite(D).all(axis=1)
    Wf = W[finite]
    S50f = S50[np.isfinite(S50)]
    absw = np.abs(Wf)
    n = len(W)
    out = {
        "tag": tag,
        "dump": str(dump),
        "n_events": n,
        "n_nonfinite_window": int((~finite).sum()),
        "n_nonfinite_any_layer": int(nonfinite_any_layer.sum()),
        "nonfinite_event_ids": ev[~finite].tolist(),
        "abs_W_quantiles": {q: float(np.quantile(absw, float(q) / 100.0))
                           for q in ("50", "90", "99", "99.9")},
        "abs_W_max": float(absw.max()),
        "var_W": float(Wf.var(ddof=1)),
        "mean_W": float(Wf.mean()),
        "se_mean_W": float(Wf.std(ddof=1) / math.sqrt(len(Wf))),
        "median_W": float(np.median(Wf)),
        "trimmed_mean_1pct": trimmed_mean(Wf, 0.01),
        "trimmed_mean_5pct": trimmed_mean(Wf, 0.05),
        "top1pct_var_share": tail_var_share(Wf, 0.01),
        "top0p1pct_var_share": tail_var_share(Wf, 0.001),
        "median_steps": float(np.median(nst)),
        "median_tracks": float(np.median(ntrk)),
        # total-detector dot sum (all 50 layers): relabeling dipoles cancel here
        "mean_sum50": float(S50f.mean()),
        "se_mean_sum50": float(S50f.std(ddof=1) / math.sqrt(len(S50f))),
        "var_sum50": float(S50f.var(ddof=1)),
        "window_over_total_var_ratio": float(Wf.var(ddof=1) / S50f.var(ddof=1)),
        "dipoles": dipole_metrics(D),
    }
    return out


def spike_anatomy(tag: str, dump: Path, lo: int, hi: int, topn: int = 10) -> list[dict]:
    ev, sig, ntrk, nst, E, D = parse_dump(dump)
    win = slice(lo, hi + 1)
    W = D[:, win].sum(axis=1)
    # rank finite events by |W|; nonfinite events reported separately
    finite = np.isfinite(W)
    order = np.argsort(np.where(finite, np.abs(W), -np.inf))[::-1][:topn]
    med_steps = float(np.median(nst))
    med_tracks = float(np.median(ntrk))
    rows = []
    for i in order:
        d = D[i]
        dw = d[win]
        adw = np.abs(dw)
        tot = adw.sum()
        top_layers = np.argsort(adw)[::-1][:3] + lo
        # full-detector view too: which layers (0..49) dominate |dot|
        ad_all = np.abs(np.where(np.isfinite(d), d, 0.0))
        top_layers_all = np.argsort(ad_all)[::-1][:3]
        rows.append({
            "event": int(ev[i]),
            "W": float(W[i]),
            "abs_W_over_median": float(abs(W[i]) / max(np.median(np.abs(W[finite])), 1e-300)),
            "steps": int(nst[i]),
            "steps_over_median": float(nst[i] / med_steps),
            "tracks": int(ntrk[i]),
            "tracks_over_median": float(ntrk[i] / med_tracks),
            "window_top3_layers": top_layers.tolist(),
            "window_top3_share": float(adw[top_layers - lo].sum() / tot) if tot > 0 else None,
            "alldet_top3_layers": top_layers_all.tolist(),
            "n_pos_window": int((dw > 0).sum()),
            "n_neg_window": int((dw < 0).sum()),
            "dot_hi_plus_1": float(d[hi + 1]) if hi + 1 < len(d) else None,
            "sum50": float(d.sum()),
            "window_dots": [float(x) for x in dw],
        })
    return rows


def top_dipoles(tag: str, dump: Path, topn: int = 10) -> list[dict]:
    ev, sig, ntrk, nst, E, D = parse_dump(dump)
    A, B = D[:, :-1], D[:, 1:]
    amp = np.where((A * B) < 0, np.minimum(np.abs(A), np.abs(B)), 0.0)
    amp = np.where(np.isfinite(amp), amp, 0.0)
    dip_amp = amp.max(axis=1)
    dip_layer = amp.argmax(axis=1)
    order = np.argsort(dip_amp)[::-1][:topn]
    return [{"event": int(ev[i]), "left_layer": int(dip_layer[i]),
             "amplitude": float(dip_amp[i]),
             "D_j": float(D[i, dip_layer[i]]),
             "D_j_plus_1": float(D[i, dip_layer[i] + 1]),
             "sum50": float(D[i].sum())} for i in order]


def cross_config(dump_a: Path, dump_b: Path, lo: int, hi: int, topn: int = 20) -> dict:
    """Are the top-|W| events of run A also spiky in run B (same seed)?"""
    eva, _, _, _, _, Da = parse_dump(dump_a)
    evb, _, _, _, _, Db = parse_dump(dump_b)
    win = slice(lo, hi + 1)
    Wa = Da[:, win].sum(axis=1)
    Wb = Db[:, win].sum(axis=1)
    fa = np.isfinite(Wa)
    fb = np.isfinite(Wb)
    order = np.argsort(np.where(fa, np.abs(Wa), -np.inf))[::-1][:topn]
    idx_b = {int(e): j for j, e in enumerate(evb)}
    # percentile ranks of |Wb| (finite only)
    absb = np.abs(Wb[fb])
    rows = []
    for i in order:
        e = int(eva[i])
        j = idx_b.get(e)
        if j is None:
            continue
        wb = Wb[j]
        pct = float((absb < abs(wb)).mean() * 100.0) if np.isfinite(wb) else None
        rows.append({"event": e, "W_a": float(Wa[i]), "W_b": float(wb),
                     "pct_rank_in_b": pct})
    pcts = [r["pct_rank_in_b"] for r in rows if r["pct_rank_in_b"] is not None]
    return {
        "top_n": topn,
        "events": rows,
        "median_pct_rank_in_b": float(np.median(pcts)) if pcts else None,
        "n_above_p99_in_b": sum(1 for p in pcts if p >= 99.0),
        "n_nonfinite_in_b": sum(1 for r in rows if r["pct_rank_in_b"] is None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--window", nargs=2, type=int, default=[5, 18])
    ap.add_argument("--out", default="fidelity/wave8_spikes_summary.json")
    ap.add_argument("--spikes-out", default="fidelity/wave8_spike_events.json")
    args = ap.parse_args()
    base = Path(args.base)
    lo, hi = args.window

    tags = {
        "ref_s1": ("gap", 1), "ref_s2": ("gap", 2),
        "uns_s1": ("gap", 1), "uns_s2": ("gap", 2),
        "int_s1": ("gap", 1), "int_s2": ("gap", 2),
        "uns_abs_s1": ("abs", 1),
    }
    runs = {}
    for tag in tags:
        dump = base / tag / "dump.txt"
        if dump.exists():
            runs[tag] = characterize(tag, dump, lo, hi)
        else:
            print(f"[warn] missing {dump}")

    summary: dict = {
        "window": [lo, hi],
        "fd_truth_core": {k: {"value": v[0], "err": v[1]} for k, v in FD_TRUTH.items()},
        "configs": {
            "ref": "-x 2 -y 1 -B 1 (canonical severing)",
            "uns": "-x 0 -y 0 -B 0 (unsevered)",
            "int": "-x 0 -y 1 -B 1 (descendant severing off)",
        },
        "runs": runs,
    }

    # variance ratios vs reference, per seed
    ratios = {}
    for s in (1, 2):
        ref = runs.get(f"ref_s{s}")
        for cfg in ("uns", "int"):
            r = runs.get(f"{cfg}_s{s}")
            if ref and r:
                ratios[f"{cfg}_over_ref_s{s}"] = r["var_W"] / ref["var_W"]
    summary["variance_ratios"] = ratios

    # spike anatomy: unsevered gap runs (spec), plus intermediate + absorber
    anatomy = {}
    spike_list = []
    dipole_list = []
    for tag in ("uns_s1", "uns_s2", "int_s1", "int_s2", "uns_abs_s1"):
        dump = base / tag / "dump.txt"
        if not dump.exists():
            continue
        anatomy[tag] = spike_anatomy(tag, dump, lo, hi)
        if tag.startswith("uns_s"):
            s = int(tag[-1])
            for r in anatomy[tag]:
                spike_list.append({"seed": s, "event": r["event"], "W": r["W"],
                                   "config": "-x 0 -y 0 -B 0", "seed_arg": "gap"})
            for r in top_dipoles(tag, dump):
                dipole_list.append({"seed": s, **r, "config": "-x 0 -y 0 -B 0",
                                    "seed_arg": "gap"})
    summary["spike_anatomy"] = anatomy

    # cross-config: unsevered spikes in the intermediate + reference runs
    xc = {}
    for s in (1, 2):
        a = base / f"uns_s{s}" / "dump.txt"
        for other in ("int", "ref"):
            b = base / f"{other}_s{s}" / "dump.txt"
            if a.exists() and b.exists():
                xc[f"uns_vs_{other}_s{s}"] = cross_config(a, b, lo, hi)
    summary["cross_config"] = xc

    Path(args.out).write_text(json.dumps(summary, indent=1))
    Path(args.spikes_out).write_text(json.dumps({
        "description": "Top spike events (largest |core-window dot sum|, L%d-%d) "
                       "in the unsevered (-x 0 -y 0 -B 0) gap-seed runs; "
                       "wave-8 mining, for step-level dissection." % (lo, hi),
        "base_args": "-n 2000 -p e- -e 10000 -l 50 -t 400 -a 2.3 -g 5.7:1 "
                     "-f 0.2 -N 1e-3 -C 1000 -x 0 -y 0 -B 0",
        "binary": "build_agent_fwd/HepEmShow @ knob/race-score 7ebab9e (all knobs off)",
        "events": spike_list,
        "top_dipoles_note": "Largest adjacent-bin opposite-sign dot pairs "
                            "(relabeling dipoles) — the actual spike mechanism; "
                            "the W-ranked list above only catches dipoles that "
                            "straddle the window edge (L18|L19).",
        "top_dipoles": dipole_list,
    }, indent=1))
    print(f"wrote {args.out} and {args.spikes_out}")

    # console digest
    for tag, r in runs.items():
        print(f"{tag:12s} mean={r['mean_W']:12.4g} +- {r['se_mean_W']:.4g} "
              f"med={r['median_W']:9.4g} tm1={r['trimmed_mean_1pct']:10.4g} "
              f"tm5={r['trimmed_mean_5pct']:10.4g} var={r['var_W']:.4g} "
              f"top1%%var={r['top1pct_var_share']:.3f} "
              f"nonfinite={r['n_nonfinite_any_layer']} "
              f"win/tot_var={r['window_over_total_var_ratio']:.3g} "
              f"dip>1e4={r['dipoles']['n_dipole_gt_1e4']}")
    for k, v in ratios.items():
        print(f"var ratio {k}: {v:.4g}")


if __name__ == "__main__":
    main()

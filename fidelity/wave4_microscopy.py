"""Wave-4 derivative microscopy: RNG-lineage-isolated AD-vs-FD decomposition.

Wave 3 found 100% of events diverge under any gap perturbation because one
branch flip decorrelates the entire downstream shower through the single
shared RNG stream. The wave-4 knob (`--rng-lineage 1`) gives every track its
own deterministic stream keyed by (base seed, track lineage), so divergence
localizes to subtrees where a discrete decision genuinely flips. This script
re-runs the wave-3 decomposition at both EVENT level (per-event dumps,
HEPEMSHOW_EVENT_DUMP, same format as wave 3) and TRACK level (per-track
binary dumps, HEPEMSHOW_TRACK_DUMP):

    record = (ev:int32, nsteps:uint32, lineage:uint64, subsig:uint64,
              ewin:f64, dwin:f64)          # 40 bytes, window = L5-18

For a same-seed triplet at (g-h, g, g+h), a track lineage is "matched" when
it exists in all three runs with an identical decision-path sub-signature;
matched tracks must satisfy per-track AD == central FD up to O(h^2). The
core-window FD mean decomposes exactly into matched + flip contributions.
Also reports the CRN pairing quality: variance of the per-event paired FD
versus the unpaired (independent-runs) variance.

Usage:
  python fidelity/wave4_microscopy.py \
      --triplet SEED H CENTER_DIR MINUS_DIR PLUS_DIR [--triplet ...] \
      [--window 5 18] [--out fidelity/wave4_microscopy_summary.json]

Each *_DIR must contain dump.txt (event dump) and tracks.bin (track dump).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

TRACK_DTYPE = np.dtype([("ev", "<i4"), ("nsteps", "<u4"), ("lin", "<u8"),
                        ("sig", "<u8"), ("ewin", "<f8"), ("dwin", "<f8")])


def parse_event_dump(path: str):
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


def load_tracks(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=TRACK_DTYPE)
    if arr.size == 0:
        raise RuntimeError(f"empty track dump: {path}")
    return arr


def match_tracks(t0: np.ndarray, tm: np.ndarray, tp: np.ndarray) -> dict:
    """Match tracks across the triplet by lineage hash (globally unique by
    construction: primary lineage already encodes the event index)."""
    out = {}
    # lineage-collision audit (drop colliding lineages from the matched set)
    dup_frac = {}
    keep = {}
    for name, t in (("c", t0), ("m", tm), ("p", tp)):
        lins, counts = np.unique(t["lin"], return_counts=True)
        dup = lins[counts > 1]
        dup_frac[name] = float(dup.size) / float(lins.size)
        if dup.size:
            keep[name] = ~np.isin(t["lin"], dup)
        else:
            keep[name] = np.ones(t.size, dtype=bool)
    t0u, tmu, tpu = t0[keep["c"]], tm[keep["m"]], tp[keep["p"]]

    s0 = np.argsort(t0u["lin"]); t0u = t0u[s0]
    sm = np.argsort(tmu["lin"]); tmu = tmu[sm]
    sp = np.argsort(tpu["lin"]); tpu = tpu[sp]

    common = np.intersect1d(t0u["lin"], tmu["lin"], assume_unique=True)
    common = np.intersect1d(common, tpu["lin"], assume_unique=True)

    i0 = np.searchsorted(t0u["lin"], common)
    im = np.searchsorted(tmu["lin"], common)
    ip = np.searchsorted(tpu["lin"], common)
    c0, cm, cp = t0u[i0], tmu[im], tpu[ip]
    same_sig = (c0["sig"] == cm["sig"]) & (c0["sig"] == cp["sig"])

    out["dup_lineage_frac"] = dup_frac
    out["n_tracks"] = {"c": int(t0.size), "m": int(tm.size), "p": int(tp.size)}
    out["n_common_lineage"] = int(common.size)
    out["n_matched_same_sig"] = int(same_sig.sum())
    out["_matched"] = (c0, cm, cp, same_sig)
    return out


def analyze_triplet(seed: int, h: float, center: str, minus: str, plus: str,
                    lo: int, hi: int) -> dict:
    # ---------- EVENT level (identical logic to wave 3) ----------
    ev0, sig0, ntrk0, nst0, e0, d0 = parse_event_dump(os.path.join(center, "dump.txt"))
    evm, sigm, _, nstm, em, _ = parse_event_dump(os.path.join(minus, "dump.txt"))
    evp, sigp, _, nstp, ep, _ = parse_event_dump(os.path.join(plus, "dump.txt"))

    n = min(len(ev0), len(evm), len(evp))
    assert (ev0[:n] == evm[:n]).all() and (ev0[:n] == evp[:n]).all()
    sig0, sigm, sigp = sig0[:n], sigm[:n], sigp[:n]
    e0, em, ep, d0 = e0[:n], em[:n], ep[:n], d0[:n]
    nst0, nstm, nstp = nst0[:n], nstm[:n], nstp[:n]

    same = (sigm == sig0) & (sigp == sig0)
    div = ~same
    n_same = int(same.sum())

    fd = (ep - em) / (2.0 * h)
    win = slice(lo, hi + 1)
    fd_w = fd[:, win].sum(axis=1)
    ad_w = d0[:, win].sum(axis=1)

    mean_fd = float(fd_w.mean())
    mean_ad = float(ad_w.mean())

    diff_w = ad_w[same] - fd_w[same]
    abs_diff = np.abs(diff_w)
    q = (lambda p: float(np.quantile(abs_diff, p))) if n_same else (lambda p: None)

    event_level = {
        "n_events": int(n),
        "n_same_path": n_same,
        "frac_same_path": n_same / n,
        "core_window": {
            "mean_ad_all": mean_ad,
            "mean_fd_all": mean_fd,
            "ratio_ad_fd_all": mean_ad / mean_fd if mean_fd else None,
            "same_path": {
                "fd_contrib_to_total_mean": float(fd_w[same].sum() / n),
                "ad_contrib_to_total_mean": float(ad_w[same].sum() / n),
                "mean_ad_minus_fd": float(diff_w.mean()) if n_same else None,
                "se_ad_minus_fd": (float(diff_w.std(ddof=1) / math.sqrt(n_same))
                                   if n_same > 1 else None),
                "mean_fd_same": float(fd_w[same].mean()) if n_same else None,
                "mean_ad_same": float(ad_w[same].mean()) if n_same else None,
                "abs_diff_quantiles": {"q50": q(0.5), "q90": q(0.9),
                                       "q99": q(0.99),
                                       "max": float(abs_diff.max()) if n_same else None},
            },
            "diverged": {
                "fd_contrib_to_total_mean": float(fd_w[div].sum() / n),
                "ad_contrib_to_total_mean": float(ad_w[div].sum() / n),
            },
        },
        "steps": {
            "mean_steps_center": float(nst0.mean()),
            "mean_abs_step_delta_within_triplet":
                float((np.abs(nstp - nstm) + np.abs(nst0 - nstm)).mean()),
        },
        "per_layer": {
            "layers": list(range(e0.shape[1])),
            "ad_mean_all": d0.mean(axis=0).tolist(),
            "fd_mean_all": fd.mean(axis=0).tolist(),
            "ad_mean_same": d0[same].mean(axis=0).tolist() if n_same else None,
            "fd_mean_same": fd[same].mean(axis=0).tolist() if n_same else None,
        },
    }

    # ---------- CRN check (event level) ----------
    ew0 = e0[:, win].sum(axis=1)
    ewm = em[:, win].sum(axis=1)
    ewp = ep[:, win].sum(axis=1)
    var_paired = float(fd_w.var(ddof=1))
    var_unpaired = float((ewp.var(ddof=1) + ewm.var(ddof=1)) / (2.0 * h) ** 2)
    crn = {
        "h": h,
        "per_event_fd_var_paired": var_paired,
        "per_event_fd_var_unpaired": var_unpaired,
        "variance_reduction_factor": var_unpaired / var_paired if var_paired else None,
        "se_mean_fd_paired": math.sqrt(var_paired / n),
        "se_mean_fd_unpaired": math.sqrt(var_unpaired / n),
        "corr_Ewin_plus_minus": float(np.corrcoef(ewp, ewm)[0, 1]),
        "note": "unpaired = var(E+)+var(E-) over events / (2h)^2, i.e. what "
                "independent-seed runs would give at the same n",
    }

    # ---------- TRACK level ----------
    t0 = load_tracks(os.path.join(center, "tracks.bin"))
    tm = load_tracks(os.path.join(minus, "tracks.bin"))
    tp = load_tracks(os.path.join(plus, "tracks.bin"))
    minfo = match_tracks(t0, tm, tp)
    c0, cm, cp, same_sig = minfo.pop("_matched")

    # exact decomposition of the total per-event-mean window FD into
    # matched (same-lineage same-sig) and flip (everything else) parts
    fd_total = float((tp["ewin"].sum() - tm["ewin"].sum()) / (2.0 * h) / n)
    ad_total = float(t0["dwin"].sum() / n)
    fd_matched = float((cp["ewin"][same_sig].sum() - cm["ewin"][same_sig].sum())
                       / (2.0 * h) / n)
    ad_matched = float(c0["dwin"][same_sig].sum() / n)

    # matched-population per-track AD vs FD
    fd_trk = (cp["ewin"][same_sig] - cm["ewin"][same_sig]) / (2.0 * h)
    ad_trk = c0["dwin"][same_sig].astype(float)
    dtrk = ad_trk - fd_trk
    nz = (np.abs(ad_trk) > 0) | (np.abs(fd_trk) > 0)
    n_matched = int(same_sig.sum())
    qq = (lambda a, p: float(np.quantile(a, p)) if a.size else None)

    steps_equal = None
    if n_matched:
        steps_equal = bool(((c0["nsteps"] == cm["nsteps"]) &
                            (c0["nsteps"] == cp["nsteps"]))[same_sig].all())

    track_level = {
        **minfo,
        "frac_matched_same_sig_of_center": minfo["n_matched_same_sig"] / t0.size,
        "matched_steps_all_equal": steps_equal,
        "core_window": {
            "fd_total_mean": fd_total,
            "ad_total_mean": ad_total,
            "fd_matched_contrib": fd_matched,
            "fd_flip_contrib": fd_total - fd_matched,
            "ad_matched_contrib": ad_matched,
            "ad_flip_contrib": ad_total - ad_matched,
            "matched_fd_mass_frac": fd_matched / fd_total if fd_total else None,
        },
        "matched_ad_vs_fd": {
            "n_matched": n_matched,
            "n_matched_with_window_signal": int(nz.sum()),
            "mean_ad_minus_fd": float(dtrk.mean()) if n_matched else None,
            "se_ad_minus_fd": (float(dtrk.std(ddof=1) / math.sqrt(n_matched))
                               if n_matched > 1 else None),
            "abs_diff_quantiles": {"q50": qq(np.abs(dtrk), 0.5),
                                   "q90": qq(np.abs(dtrk), 0.9),
                                   "q99": qq(np.abs(dtrk), 0.99),
                                   "max": float(np.abs(dtrk).max()) if n_matched else None},
            "signal_tracks_abs_diff_quantiles": {
                "q50": qq(np.abs(dtrk[nz]), 0.5),
                "q90": qq(np.abs(dtrk[nz]), 0.9),
                "q99": qq(np.abs(dtrk[nz]), 0.99)},
        },
    }

    return {"seed": seed, "h": h, "window": [lo, hi],
            "event_level": event_level, "track_level": track_level,
            "crn": crn}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", nargs=5, action="append", required=True,
                    metavar=("SEED", "H", "CENTER_DIR", "MINUS_DIR", "PLUS_DIR"))
    ap.add_argument("--window", nargs=2, type=int, default=[5, 18])
    ap.add_argument("--out", default="fidelity/wave4_microscopy_summary.json")
    args = ap.parse_args()

    lo, hi = args.window
    results = []
    for seed_s, h_s, center, minus, plus in args.triplet:
        r = analyze_triplet(int(seed_s), float(h_s), center, minus, plus, lo, hi)
        results.append(r)

    print(f"\n=== wave-4 microscopy (rng-lineage on): core window L{lo}-L{hi} ===")
    for r in results:
        el, tl, crn = r["event_level"], r["track_level"], r["crn"]
        cw, tcw = el["core_window"], tl["core_window"]
        print(f"\nseed {r['seed']}  h={r['h']}  n={el['n_events']}")
        print(f"  EVENT: same-path {el['n_same_path']}/{el['n_events']} "
              f"({100*el['frac_same_path']:.1f}%)  AD={cw['mean_ad_all']:.3f} "
              f"FD={cw['mean_fd_all']:.3f}")
        sp = cw["same_path"]
        if el["n_same_path"]:
            print(f"         same-path: FD={sp['mean_fd_same']:.4f} "
                  f"AD={sp['mean_ad_same']:.4f}  AD-FD={sp['mean_ad_minus_fd']:.4f}"
                  f"+-{sp['se_ad_minus_fd']:.4f}  q99|d|={sp['abs_diff_quantiles']['q99']:.4f}")
        print(f"  TRACK: matched {tl['n_matched_same_sig']}/{tl['n_tracks']['c']} "
              f"({100*tl['frac_matched_same_sig_of_center']:.1f}%)  "
              f"common-lineage {tl['n_common_lineage']}")
        print(f"         FD total={tcw['fd_total_mean']:.3f} = matched "
              f"{tcw['fd_matched_contrib']:.3f} + flip {tcw['fd_flip_contrib']:.3f}; "
              f"AD total={tcw['ad_total_mean']:.3f} (matched {tcw['ad_matched_contrib']:.3f})")
        mv = tl["matched_ad_vs_fd"]
        if mv["n_matched"]:
            print(f"         matched AD-FD={mv['mean_ad_minus_fd']:.5f}"
                  f"+-{mv['se_ad_minus_fd']:.5f}  q99|d|={mv['abs_diff_quantiles']['q99']:.4g}")
        print(f"  CRN:   var paired={crn['per_event_fd_var_paired']:.4g} "
              f"unpaired={crn['per_event_fd_var_unpaired']:.4g}  "
              f"reduction x{crn['variance_reduction_factor']:.2f}  "
              f"corr(E+,E-)={crn['corr_Ewin_plus_minus']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"window": [lo, hi], "triplets": results}, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

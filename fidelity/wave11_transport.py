#!/usr/bin/env python3
"""Wave 11-A gap transportability analysis (pre-registered items A and B).

A. Scale-aware boundary-dot-cap: cap = (80/5.7)*g => b=42.1 at g=3.0.
   Score the off-design b42 core L5-18 pooled mean vs the od FD truth
   188.05 +- 9.52 (wave-9 validation, g=3.1/2.9 severed arms), plus the
   b30/b60 bracket for the local cap slope.
B. Median transportability: pooled per-event median of the capped W_e
   distribution, off-design (b42 new runs; b80 from wave-9 validation
   sidecars) and on-design across the cap grid {b10,b40,b80,b160} plus
   the uncapped unsevered wave-8 dumps.  Median SE via nonparametric
   bootstrap over events (resampling within the pooled sample).

Inputs (all pre-existing except the wave-11 run dirs):
  fidelity/wave11_runs/{b42_s1..4,b30_s1..2,b60_s1..2}/dump.txt   (od, g=3.0)
  fidelity/wave9_runs/gap_b{10,40,80,160}_s{1,2}/dump.txt         (on, g=5.7)
  fidelity/wave8_runs/uns_s{1,2}/dump.txt                         (on, uncapped)
  fidelity/wave9val_runs/gap_b80_od_s*_events.npz                 (od, b80, 200k)

Writes fidelity/wave11_transport_summary.json.
"""
import glob
import json
import math
import os
import sys

import numpy as np

FID = os.path.dirname(os.path.abspath(__file__))
LO, HI = 5, 18  # core window, inclusive

FD_OD = (188.05392622948375, 9.515829015176568)   # g=3.0 own FD (wave-9 val)
FD_ON = (195.78335485036746, 9.356835570290496)   # g=5.7 1M FD truth
N_BOOT = 10000
BOOT_SEED = 20260725


def load_dump_W(path):
    """Per-event core-window derivative sum W_e from a 104-column dump."""
    dots = []
    with open(path) as f:
        for line in f:
            p = line.split()
            dots.append([float(x) for x in p[54:104]])
    dots = np.array(dots)
    return dots[:, LO:HI + 1].sum(axis=1), int((~np.isfinite(dots)).sum())


def pooled_stats(W, rng):
    n = len(W)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot[i] = np.median(W[rng.integers(0, n, n)])
    return {
        "n": n,
        "mean": float(W.mean()),
        "se_mean": float(W.std(ddof=1) / math.sqrt(n)),
        "median": float(np.median(W)),
        "se_median_bootstrap": float(boot.std(ddof=1)),
        "se_median_normal_approx": float(1.2533 * W.std(ddof=1) / math.sqrt(n)),
        "var": float(W.var(ddof=1)),
        "nonfinite_dump_values": 0,
    }


def z_vs(value, se, truth):
    return (value - truth[0]) / math.sqrt(truth[1] ** 2 + se ** 2)


def collect(tag_dirs):
    Ws, per_seed, nonfinite = [], {}, 0
    for tag, d in tag_dirs:
        p = os.path.join(d, "dump.txt")
        if not os.path.exists(p):
            print(f"MISSING {p}", file=sys.stderr)
            continue
        W, nf = load_dump_W(p)
        nonfinite += nf
        per_seed[tag] = {"n": len(W), "mean": float(W.mean()),
                         "median": float(np.median(W))}
        Ws.append(W)
    return (np.concatenate(Ws) if Ws else np.array([])), per_seed, nonfinite


def main():
    rng = np.random.default_rng(BOOT_SEED)
    w11 = os.path.join(FID, "wave11_runs")
    out = {
        "wave": 11,
        "items": "A,B (gap transportability, LOCAL)",
        "binary_mtime_ns": 1784902792072373144,
        "binary_size": 429472,
        "core_window_layers": [LO, HI],
        "fd_truth_od_g3.0": list(FD_OD),
        "fd_truth_on_g5.7": list(FD_ON),
        "median_se_method": (
            f"nonparametric bootstrap over pooled events, {N_BOOT} resamples, "
            f"rng seed {BOOT_SEED}; 1.2533*SE_mean also reported for reference "
            "(invalid for these heavy tails)"),
        "config": ("unsevered -x 0 -y 0 -B 0 -f 0.2 -N 1e-3 -C 1000, gap seed "
                   "-g <g>:1, n=2000/seed for dumps, 20000/seed for npz"),
    }

    # ---- off-design (a=2.30, g=3.00) ---------------------------------------
    od = {}
    for cap, tags in [("b42", [f"b42_s{s}" for s in (1, 2, 3, 4)]),
                      ("b30", [f"b30_s{s}" for s in (1, 2)]),
                      ("b60", [f"b60_s{s}" for s in (1, 2)])]:
        W, per_seed, nf = collect([(t, os.path.join(w11, t)) for t in tags])
        if not len(W):
            od[cap] = {"status": "no data"}
            continue
        st = pooled_stats(W, rng)
        st["nonfinite_dump_values"] = nf
        st["per_seed"] = per_seed
        means = [v["mean"] for v in per_seed.values()]
        if len(means) > 1:
            st["seed_scatter_se_mean"] = float(
                np.std(means, ddof=1) / math.sqrt(len(means)))
        st["z_mean_vs_od_fd"] = z_vs(st["mean"], st["se_mean"], FD_OD)
        st["z_median_vs_od_fd"] = z_vs(st["median"],
                                       st["se_median_bootstrap"], FD_OD)
        od[cap] = st

    # b80 off-design from wave-9 validation per-event sidecars
    npzs = sorted(glob.glob(os.path.join(FID, "wave9val_runs",
                                         "gap_b80_od_s*_events.npz")))
    Wb80 = np.concatenate([np.load(p)["W"] for p in npzs]) if npzs else np.array([])
    if len(Wb80):
        st = pooled_stats(Wb80, rng)
        st["n_seed_files"] = len(npzs)
        st["z_mean_vs_od_fd"] = z_vs(st["mean"], st["se_mean"], FD_OD)
        st["z_median_vs_od_fd"] = z_vs(st["median"],
                                       st["se_median_bootstrap"], FD_OD)
        od["b80"] = st
    out["off_design_g3.0"] = od

    # ---- on-design (g=5.70) cap grid + uncapped ----------------------------
    on = {}
    grid = [("b10", "wave9_runs/gap_b10_s%d"), ("b40", "wave9_runs/gap_b40_s%d"),
            ("b80", "wave9_runs/gap_b80_s%d"), ("b160", "wave9_runs/gap_b160_s%d"),
            ("uncapped", "wave8_runs/uns_s%d")]
    for cap, pat in grid:
        dirs = [(pat % s, os.path.join(FID, pat % s)) for s in (1, 2)]
        W, per_seed, nf = collect(dirs)
        if not len(W):
            on[cap] = {"status": "no data"}
            continue
        st = pooled_stats(W, rng)
        st["nonfinite_dump_values"] = nf
        st["per_seed"] = per_seed
        st["z_mean_vs_on_fd"] = z_vs(st["mean"], st["se_mean"], FD_ON)
        st["z_median_vs_on_fd"] = z_vs(st["median"],
                                       st["se_median_bootstrap"], FD_ON)
        on[cap] = st
    npzs = sorted(glob.glob(os.path.join(FID, "wave9val_runs",
                                         "gap_b80_val_s*_events.npz")))
    Wv = np.concatenate([np.load(p)["W"] for p in npzs]) if npzs else np.array([])
    if len(Wv):
        st = pooled_stats(Wv, rng)
        st["n_seed_files"] = len(npzs)
        st["z_mean_vs_on_fd"] = z_vs(st["mean"], st["se_mean"], FD_ON)
        st["z_median_vs_on_fd"] = z_vs(st["median"],
                                       st["se_median_bootstrap"], FD_ON)
        on["b80_200k"] = st
    out["on_design_g5.7"] = on

    # ---- pre-registered verdicts -------------------------------------------
    v = {}
    if "mean" in od.get("b42", {}):
        v["A_b42_mean_within_2sigma_of_od_fd"] = bool(
            abs(od["b42"]["z_mean_vs_od_fd"]) < 2)
    if "median" in od.get("b42", {}) and "median" in od.get("b80", {}):
        v["B_od_medians_within_2sigma"] = bool(
            abs(od["b42"]["z_median_vs_od_fd"]) < 2
            and abs(od["b80"]["z_median_vs_od_fd"]) < 2)
        m42, m80 = od["b42"]["median"], od["b80"]["median"]
        u42, u80 = od["b42"]["mean"], od["b80"]["mean"]
        v["B_median_rel_spread_caps_42_80"] = float(
            abs(m42 - m80) / (0.5 * (m42 + m80)))
        v["B_mean_rel_spread_caps_42_80"] = float(
            abs(u42 - u80) / (0.5 * (u42 + u80)))
        v["B_median_varies_lt_10pct_while_mean_16pct"] = bool(
            v["B_median_rel_spread_caps_42_80"] < 0.10)
    out["verdicts"] = v

    with open(os.path.join(FID, "wave11_transport_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["verdicts"], indent=1))
    for side in ("off_design_g3.0", "on_design_g5.7"):
        print(f"-- {side}")
        for cap, st in out[side].items():
            if "mean" not in st:
                print(f"  {cap}: {st}")
                continue
            print(f"  {cap}: n={st['n']} mean={st['mean']:.2f}+-"
                  f"{st['se_mean']:.2f} median={st['median']:.2f}+-"
                  f"{st['se_median_bootstrap']:.2f}")


if __name__ == "__main__":
    main()

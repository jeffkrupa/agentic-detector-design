#!/usr/bin/env python
"""Publication-quality per-layer AD/FD figures for the FINAL prescription.

Two figures (PNG + PDF, dpi=200) under fidelity/plots/:

  adfd_final_per_layer_ratio.png     : AD/FD ratio vs layer index, two panels
                                       (gap top, absorber bottom). Each panel
                                       overlays the "published flags (severed)"
                                       estimator (1M ev) against the "repaired"
                                       estimator (unsevered + cap/governor,
                                       200k ev), both divided by the common
                                       ~1M-event paired-FD truth.
  adfd_final_per_layer_absolute.png  : per-layer AD (repaired) and the FD truth
                                       in MeV/mm, linear y.

On-design point for both channels: a = 2.30 mm, g = 5.70 mm, 10 GeV e-, 50 layers.

SEM definition (repaired AD series): event-weighted mean of mean_dE across
seeds; pooled per-event variance, SEM = sqrt(sum_s(N_s var_dE_s)/N_tot / N_tot)
= sqrt( (sum_s N_s var_dE_s / N_tot) / N_tot ). Only rows with ok=True and
nan=False enter. The published-flags AD (ad, ad_se) and the FD truth (fd, fd_se)
come unchanged from experiments/perlayer_adfd_1M_summary.json. Ratio errors
propagate both contributions:
    sigma_r = |r| * sqrt((sem_AD/AD)^2 + (fd_se/FD)^2).

Layers where the FD truth is consistent with zero (|FD| < 2 fd_se) are drawn
with open (hollow) markers: the ratio is unreliable there.

Re-runnable: python fidelity/plots/make_adfd_final_plots.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
NLAYERS = 50

# Okabe-Ito, colorblind-safe.
COLORS = {
    "published": "#0072B2",   # blue  -- published flags (severed)
    "repaired": "#009E73",    # green -- unsevered + cap/governor
    "fd": "#000000",          # black -- FD truth
}
OFFSETS = {"published": -0.15, "repaired": 0.15}

# Config prefixes for the repaired (final-prescription) 200k-event runs.
REPAIRED = {
    "gap": (REPO / "fidelity" / "wave9val_runs", "gap_b80_val"),
    "abs": (REPO / "fidelity" / "wave11_runs", "abs_gov_val"),
}


def pooled_from_jsonl(run_dir: Path, config: str):
    """Event-weighted mean of mean_dE across seeds; pooled SEM.

    SEM = sqrt(var_pooled / N_tot) with var_pooled = sum_s(N_s var_dE_s)/N_tot.
    """
    n, m, v = [], [], []
    for p in sorted(glob.glob(str(run_dir / f"{config}_s*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("config") != config:
                    continue
                if not r.get("ok", False) or r.get("nan", False):
                    continue
                if "mean_dE" not in r or "var_dE" not in r:
                    continue
                n.append(r["n_events"])
                m.append(r["mean_dE"])
                v.append(r["var_dE"])
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)          # (nseed, 50)
    v = np.asarray(v, dtype=float)
    ntot = n.sum()
    mean = (m * n[:, None]).sum(axis=0) / ntot
    var_pooled = (v * n[:, None]).sum(axis=0) / ntot
    sem = np.sqrt(var_pooled / ntot)
    return mean, sem, int(ntot)


def ratio_with_err(ad, sem, fd, fd_se):
    r = ad / fd
    sig = np.abs(r) * np.sqrt((sem / ad) ** 2 + (fd_se / fd) ** 2)
    return r, sig


def fmt_ev(n):
    if n >= 1_000_000:
        return f"{n/1e6:g}M"
    return f"{n/1e3:g}k"


def draw_ratio_series(ax, x, r, sig, reliable, color, label, ylim, row):
    """One ratio series: filled (FD reliable) / hollow (|FD|<2sig) markers.

    Off-scale points get a small edge arrow; the value is printed only when
    the +-1 sigma interval lies entirely outside the y-range. In a contiguous
    run of off-scale points (e.g. the absorber deep tail, where the FD truth
    crosses zero and the ratio is ill-defined) the value labels are thinned to
    a minimum x-spacing so they stay readable -- the arrows still mark every
    off-scale point, and Figure 2 carries the quantitative tail story. `row`
    staggers the printed text so series labels don't collide vertically.
    """
    lo, hi = ylim
    inside = (r >= lo) & (r <= hi)
    for mask, filled in ((reliable & inside, True), (~reliable & inside, False)):
        if not mask.any():
            continue
        ax.errorbar(
            x[mask], r[mask], yerr=sig[mask], fmt="o", ms=4.5, capsize=2,
            color=color, markerfacecolor=color if filled else "none",
            markeredgecolor=color, markeredgewidth=1.0, elinewidth=1.0,
            linestyle="none", label=label if (filled and label) else None,
        )
    span = hi - lo
    signif = (r - sig > hi) | (r + sig < lo)
    min_label_gap = 3.0  # layers between printed value labels, per edge
    last_labeled = {True: -1e9, False: -1e9}  # keyed by top(bool)
    for xi, ri, is_sig in zip(x[~inside], r[~inside], signif[~inside]):
        top = ri > hi
        y_edge = hi if top else lo
        arrow_len = 0.055 * span
        y_tail = y_edge - arrow_len if top else y_edge + arrow_len
        ax.annotate(
            "", xy=(xi, y_edge), xytext=(xi, y_tail),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                            alpha=1.0 if is_sig else 0.35,
                            shrinkA=0, shrinkB=0),
            annotation_clip=False,
        )
        if is_sig and (xi - last_labeled[top]) >= min_label_gap:
            last_labeled[top] = xi
            y_txt = (y_edge - (0.075 + 0.05 * row) * span if top
                     else y_edge + (0.075 + 0.05 * row) * span)
            ax.text(xi, y_txt, f"{ri:.0f}" if abs(ri) >= 10 else f"{ri:.1f}",
                    ha="center", va="top" if top else "bottom",
                    fontsize=6.5, color=color, clip_on=False)


def style_axis(ax):
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "fidelity" / "plots"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    # ---------------- FD truth + published-flags AD (from 1M summary) --------
    with open(REPO / "experiments" / "perlayer_adfd_1M_summary.json") as fh:
        base = json.load(fh)
    fd, pub = {}, {}   # per channel: (val, se), n_events
    for ch, key in (("gap", "gap"), ("abs", "absorber")):
        pl = base["params"][key]["per_layer"]
        fd[ch] = (np.array(pl["fd"]), np.array(pl["fd_se"]),
                  int(base["params"][key]["n_events_fd_per_side"]))
        pub[ch] = (np.array(pl["ad"]), np.array(pl["ad_se"]),
                   int(base["params"][key]["n_events_ad"]))

    # ---------------- repaired (final prescription) AD, 200k ev -------------
    rep = {}
    for ch, (run_dir, cfg) in REPAIRED.items():
        rep[ch] = pooled_from_jsonl(run_dir, cfg)

    x = np.arange(NLAYERS, dtype=float)
    core = slice(5, 19)  # L5-18 inclusive

    # ---------------- core-window sanity print ------------------------------
    print("Core-window (L5-18) summed AD/FD:")
    for ch, name in (("gap", "gap d(E)/dg"), ("abs", "absorber d(E)/da")):
        fdv, fde, _ = fd[ch]
        fdw = fdv[core].sum()
        fdw_se = np.sqrt((fde[core] ** 2).sum())
        for lbl, (ad, ad_se, nev) in (
            ("published (severed)", pub[ch]),
            ("repaired            ", rep[ch]),
        ):
            adw = ad[core].sum()
            adw_se = np.sqrt((ad_se[core] ** 2).sum())
            rw = adw / fdw
            rw_se = abs(rw) * np.sqrt((adw_se / adw) ** 2 + (fdw_se / fdw) ** 2)
            print(f"  {name:18s} {lbl} ({fmt_ev(nev):>4s} ev): {rw:.3f} +- {rw_se:.3f}")

    panels = (
        ("gap", r"Gap-thickness derivative  $dE_{\mathrm{layer}}/dg$", (-1.0, 6.0)),
        ("abs", r"Absorber-thickness derivative  $dE_{\mathrm{layer}}/da$", (-0.5, 2.0)),
    )

    # ======================= Figure 1: ratio ===============================
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True)
    for ax, (ch, title, ylim) in zip(axes, panels):
        fdv, fde, _ = fd[ch]
        reliable = np.abs(fdv) >= 2.0 * fde
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        series = (
            ("published", pub[ch], "published flags (severed)"),
            ("repaired", rep[ch], "repaired (unsevered + cap/governor)"),
        )
        for row, (key, (ad, ad_se, nev), base_lbl) in enumerate(series):
            r, sig = ratio_with_err(ad, ad_se, fdv, fde)
            label = f"{base_lbl}, {fmt_ev(nev)} ev"
            draw_ratio_series(ax, x + OFFSETS[key], r, sig, reliable,
                              COLORS[key], label, ylim, row)
        ax.set_ylim(*ylim)
        ax.set_ylabel("AD / FD")
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_xlabel("Layer index")
    axes[1].set_xlim(-1.2, 50.2)
    fig.suptitle(
        "Per-layer derivative fidelity: AD / FD ratio (final prescription)\n"
        "on-design: $a$=2.30 mm, $g$=5.70 mm, 10 GeV $e^-$, 50 layers   "
        "open markers: |FD| < 2$\\sigma_{FD}$; arrows: off-scale points",
        fontsize=9, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"adfd_final_per_layer_ratio.{ext}", dpi=200)
    plt.close(fig)

    # ======================= Figure 2: absolute ============================
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True)
    abs_panels = (
        ("gap", r"Gap-thickness derivative", r"$dE_{\mathrm{layer}}/dg$  [MeV/mm]",
         (-25.0, 25.0)),
        ("abs", r"Absorber-thickness derivative", r"$dE_{\mathrm{layer}}/da$  [MeV/mm]",
         (-110.0, 260.0)),
    )
    for ax, (ch, title, ylab, ylim) in zip(axes, abs_panels):
        fdv, fde, fd_nev = fd[ch]
        ad, sem, nev = rep[ch]
        lo, hi = ylim
        span = hi - lo
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        # FD truth
        ax.errorbar(
            x - 0.15, fdv, yerr=fde, fmt="o", ms=4.5, capsize=2,
            color=COLORS["fd"], elinewidth=1.0, linestyle="none",
            label=f"FD truth ({fmt_ev(fd_nev)} ev/side)" if ch == "gap" else None,
        )
        # repaired AD, with off-scale arrows
        inside = (ad >= lo) & (ad <= hi)
        ax.errorbar(
            x[inside] + 0.15, ad[inside], yerr=sem[inside], fmt="o", ms=4.5,
            capsize=2, color=COLORS["repaired"], markerfacecolor=COLORS["repaired"],
            elinewidth=1.0, linestyle="none",
            label=f"AD repaired ({fmt_ev(nev)} ev)" if ch == "gap" else None,
        )
        for xi, ai in zip(x[~inside], ad[~inside]):
            top = ai > hi
            y_edge = hi if top else lo
            y_tail = y_edge - 0.055 * span if top else y_edge + 0.055 * span
            ax.annotate(
                "", xy=(xi, y_edge), xytext=(xi, y_tail),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["repaired"], lw=0.9,
                                shrinkA=0, shrinkB=0),
                annotation_clip=False,
            )
            ax.text(xi, y_edge - 0.075 * span if top else y_edge + 0.075 * span,
                    f"{ai:.0f}", ha="center", va="top" if top else "bottom",
                    fontsize=6.5, color=COLORS["repaired"], clip_on=False)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("Layer index")
    axes[1].set_xlim(-1.2, 50.2)
    fig.suptitle(
        "Per-layer derivative: repaired AD vs FD truth (final prescription)\n"
        "absorber FD truth turns negative past ~L26 (deep-tail sign structure); "
        "arrows: off-scale AD points",
        fontsize=9, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"adfd_final_per_layer_absolute.{ext}", dpi=200)
    plt.close(fig)

    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()

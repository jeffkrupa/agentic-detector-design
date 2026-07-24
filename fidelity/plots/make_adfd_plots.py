#!/usr/bin/env python
"""Publication-quality per-layer AD/FD fidelity plots.

Produces two figures (PNG + PDF, dpi=200) under fidelity/plots/:

  adfd_per_layer_ratio.png     : AD/FD ratio vs layer index, two panels
                                 (gap on top, absorber below), three AD
                                 variants against the common 1M-event FD truth.
  adfd_per_layer_absolute.png  : absolute per-layer derivative (MeV/mm) for
                                 the current best stack (prefix-anchor +
                                 surface mode 3) vs the FD truth.

All numbers are measured. SEM definition (uniform across all AD series):
pooled per-event variance, SEM = sqrt(mean(var_dE) / N_total) with an
event-weighted mean of mean_dE across seeds; only rows with ok=True and
nan=False enter. The FD truth and its SE come unchanged from
experiments/perlayer_adfd_1M_summary.json (per-seed central differences,
SE from seed-to-seed scatter). Ratio errors propagate both contributions:
sigma_r = |r| * sqrt((sem_AD/AD)^2 + (fd_se/FD)^2).

Layers where the FD truth is consistent with zero (|FD| < 2 fd_se) are drawn
with open (hollow) markers: the ratio is unreliable there.

Re-runnable: python fidelity/plots/make_adfd_plots.py
(wave-5 mode-3 inputs live in the session scratchpad; override with
--wave5-dir if that path has moved.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
NLAYERS = 50

DEFAULT_WAVE5 = (
    "/tmp/jekrupa/claude-114419/-eos-home-j-jeffkrup-agentic-agentic-detector-design/"
    "9b48278f-6e43-478e-b1fe-d360cf13ec20/scratchpad/wave5"
)

# Okabe-Ito, fixed assignment (same in both panels)
COLORS = {
    "baseline": "#0072B2",   # blue
    "prefix": "#D55E00",     # vermillion
    "mode3": "#009E73",      # green
}
OFFSETS = {"baseline": -0.15, "prefix": 0.0, "mode3": 0.15}


def pooled_from_rows(rows):
    """Event-weighted mean of mean_dE; pooled SEM = sqrt(mean(var_dE)/N_tot)."""
    n = np.array([r["n_events"] for r in rows], dtype=float)
    m = np.array([r["mean_dE"] for r in rows], dtype=float)   # (nseed, 50)
    v = np.array([r["var_dE"] for r in rows], dtype=float)
    ntot = n.sum()
    mean = (m * n[:, None]).sum(axis=0) / ntot
    var_pooled = (v * n[:, None]).sum(axis=0) / ntot
    sem = np.sqrt(var_pooled / ntot)
    return mean, sem, int(ntot)


def load_jsonl_rows(paths, want_config=None):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if want_config is not None and r.get("config") != want_config:
                    continue
                if not r.get("ok", False) or r.get("nan", False):
                    continue
                if "mean_dE" not in r or "var_dE" not in r:
                    continue
                rows.append(r)
    return rows


def load_wave5(wave5_dir: Path, dirs, n_events_per_run=2000):
    rows = []
    for d in dirs:
        run = wave5_dir / d
        cands = sorted(run.glob("edeps_[0-9]*"))
        cands = [c for c in cands if not c.name.startswith("edeps_gap")]
        if len(cands) != 1:
            raise FileNotFoundError(f"expected exactly one edeps_<seed> in {run}, got {cands}")
        arr = np.loadtxt(cands[0])
        if arr.shape != (NLAYERS, 4):
            raise ValueError(f"{cands[0]}: unexpected shape {arr.shape}")
        rows.append({
            "n_events": n_events_per_run,
            "mean_dE": arr[:, 2],
            "var_dE": arr[:, 3],
        })
    return pooled_from_rows(rows)


def ratio_with_err(ad, sem, fd, fd_se):
    r = ad / fd
    sig = np.abs(r) * np.sqrt((sem / ad) ** 2 + (fd_se / fd) ** 2)
    return r, sig


def fmt_ev(n):
    if n >= 1_000_000:
        s = n / 1e6
        return f"{s:g}M"
    return f"{n/1e3:g}k"


def draw_series(ax, x, r, sig, reliable, color, label, ylim, row):
    """Plot one ratio series.

    In-range points: filled markers (FD reliable) or hollow markers
    (|FD| < 2 fd_se). Off-scale points get a small arrow at the panel edge;
    the numerical value is printed only when the point is *significantly*
    off-scale (its +-1 sigma interval lies entirely outside the y-range) --
    otherwise the arrow is drawn faint with no text, since the value there
    is noise. `row` staggers the text height per series so labels don't
    collide.
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
        if is_sig:
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
    ap.add_argument("--wave5-dir", default=DEFAULT_WAVE5)
    ap.add_argument("--outdir", default=str(REPO / "fidelity" / "plots"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    wave5 = Path(args.wave5_dir)

    # ---------------- FD truth (denominator for everything) ----------------
    with open(REPO / "experiments" / "perlayer_adfd_1M_summary.json") as fh:
        base = json.load(fh)
    fd = {}
    for ch, key in (("gap", "gap"), ("abs", "absorber")):
        pl = base["params"][key]["per_layer"]
        fd[ch] = (np.array(pl["fd"]), np.array(pl["fd_se"]))

    # ---------------- AD variant 1: baseline (per-seed rows, pooled) -------
    data = {}  # data[ch][variant] = (ad, sem, n_events)
    for ch in ("gap", "abs"):
        data[ch] = {}
    for ch, cfg in (("gap", "gap_ad"), ("abs", "absorber_ad")):
        rows = load_jsonl_rows(sorted((REPO / "experiments" / "perlayer_adfd").glob(f"{cfg}_s*.jsonl")), want_config=cfg)
        data[ch]["baseline"] = pooled_from_rows(rows)
        # cross-check against the standing summary
        summ_ad = np.array(base["params"]["gap" if ch == "gap" else "absorber"]["per_layer"]["ad"])
        if not np.allclose(data[ch]["baseline"][0], summ_ad, rtol=1e-6, atol=1e-9):
            dmax = np.max(np.abs(data[ch]["baseline"][0] - summ_ad))
            print(f"[warn] baseline {ch}: recomputed mean differs from summary (max |d| = {dmax:.3g})")

    # ---------------- AD variant 2: prefix-anchor (wave 2) -----------------
    for ch, cfg in (("gap", "gap_ad_prefix"), ("abs", "abs_ad_prefix")):
        rows = load_jsonl_rows(sorted((REPO / "fidelity" / "wave2_runs").glob(f"{cfg}_s*.jsonl")), want_config=cfg)
        data[ch]["prefix"] = pooled_from_rows(rows)

    # ---------------- AD variant 3: prefix-anchor + surface mode 3 ---------
    data["gap"]["mode3"] = load_wave5(wave5, [f"t6v3_gap_s{i}_ctr_on" for i in (1, 2, 3, 4)])
    data["abs"]["mode3"] = load_wave5(wave5, [f"t6v3_abs_s{i}_ctr_on" for i in (1, 2)])

    x = np.arange(NLAYERS, dtype=float)
    panels = (
        ("gap", "Gap thickness derivative", (-1.0, 6.0)),
        ("abs", "Absorber thickness derivative", (-0.5, 1.5)),
    )
    variant_names = {
        "baseline": "baseline",
        "prefix": "prefix-anchor",
        "mode3": "+ surface m3",
    }

    # report core-window (L5-18) summed ratios for the caller
    core = slice(5, 19)
    print("Core-window (L5-18) summed AD/FD:")
    for ch in ("gap", "abs"):
        fdv, fde = fd[ch]
        fdw = fdv[core].sum()
        for v in ("baseline", "prefix", "mode3"):
            ad, sem, nev = data[ch][v]
            adw = ad[core].sum()
            adw_se = np.sqrt((sem[core] ** 2).sum())
            fdw_se = np.sqrt((fde[core] ** 2).sum())
            rw = adw / fdw
            rw_se = abs(rw) * np.sqrt((adw_se / adw) ** 2 + (fdw_se / fdw) ** 2)
            print(f"  {ch:4s} {variant_names[v]:14s} ({fmt_ev(nev):>5s} ev): {rw:.3f} +- {rw_se:.3f}")

    # ======================= Figure 1: ratio ===============================
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for ax, (ch, title, ylim) in zip(axes, panels):
        fdv, fde = fd[ch]
        reliable = np.abs(fdv) >= 2.0 * fde
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        for row, v in enumerate(("baseline", "prefix", "mode3")):
            ad, sem, nev = data[ch][v]
            r, sig = ratio_with_err(ad, sem, fdv, fde)
            label = None
            if ch == "gap":
                nev_other = data["abs"][v][2]
                label = f"{variant_names[v]} ({fmt_ev(nev)}/{fmt_ev(nev_other)} ev)"
            draw_series(ax, x + OFFSETS[v], r, sig, reliable, COLORS[v], label, ylim, row)
        ax.set_ylim(*ylim)
        ax.set_ylabel("AD / FD")
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, ncol=1)
    axes[1].set_xlabel("Layer index")
    axes[1].set_xlim(-1.2, 50.2)
    fig.suptitle(
        "Per-layer derivative fidelity: AD / FD ratio\n"
        "open markers: |FD| < 2$\\sigma_{FD}$ (ratio unreliable); arrows: off-scale points,\n"
        "labeled when $>1\\sigma$ outside the range",
        fontsize=9, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"adfd_per_layer_ratio.{ext}", dpi=200)
    plt.close(fig)

    # ======================= Figure 2: absolute ============================
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    abs_panels = (
        ("gap", "Gap thickness derivative", r"$dE_{layer}/dg$  [MeV/mm]", (-150.0, 160.0)),
        ("abs", "Absorber thickness derivative", r"$dE_{layer}/da$  [MeV/mm]", (-1100.0, 1100.0)),
    )
    for ax, (ch, title, ylab, ylim) in zip(axes, abs_panels):
        fdv, fde = fd[ch]
        ad, sem, nev = data[ch]["mode3"]
        lo, hi = ylim
        span = hi - lo
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        ax.errorbar(
            x - 0.15, fdv, yerr=fde, fmt="o", ms=4.5, capsize=2, color="black",
            elinewidth=1.0, linestyle="none",
            label=f"FD truth ({fmt_ev(base['params']['gap' if ch=='gap' else 'absorber']['n_events_fd_per_side'])} ev/side)"
            if ch == "gap" else None,
        )
        inside = (ad >= lo) & (ad <= hi)
        ax.errorbar(
            x[inside] + 0.15, ad[inside], yerr=sem[inside], fmt="o", ms=4.5,
            capsize=2, color=COLORS["mode3"], markerfacecolor=COLORS["mode3"],
            elinewidth=1.0, linestyle="none",
            label=f"AD, prefix-anchor + surface m3 ({fmt_ev(nev)}/{fmt_ev(data['abs']['mode3'][2])} ev)"
            if ch == "gap" else None,
        )
        for xi, ai in zip(x[~inside], ad[~inside]):
            top = ai > hi
            y_edge = hi if top else lo
            y_tail = y_edge - 0.055 * span if top else y_edge + 0.055 * span
            ax.annotate(
                "", xy=(xi, y_edge), xytext=(xi, y_tail),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["mode3"], lw=0.9,
                                shrinkA=0, shrinkB=0),
                annotation_clip=False,
            )
            ax.text(xi, y_edge - 0.075 * span if top else y_edge + 0.075 * span,
                    f"{ai:.0f}", ha="center", va="top" if top else "bottom",
                    fontsize=6.5, color=COLORS["mode3"], clip_on=False)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("Layer index")
    axes[1].set_xlim(-1.2, 50.2)
    fig.suptitle("Per-layer derivative: current best AD stack vs FD truth\n"
                 "arrows: off-scale AD points (value printed at the edge)",
                 fontsize=9, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"adfd_per_layer_absolute.{ext}", dpi=200)
    plt.close(fig)

    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()

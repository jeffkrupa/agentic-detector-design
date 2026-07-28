"""Read-only per-layer scrutiny of the GAP b80 estimator vs FD truth.
Mirrors fidelity/absorber_perlayer_diagnosis.json method exactly.
No sims. No C++.
"""
import json, glob, os
import numpy as np

ROOT = "/eos/home-j/jeffkrup/agentic/agentic-detector-design"
NL = 50

# ---- FD truth ----
fdj = json.load(open(f"{ROOT}/experiments/perlayer_adfd_1M_summary.json"))
pl = fdj["params"]["gap"]["per_layer"]
FD = np.array(pl["fd"]); FD_SE = np.array(pl["fd_se"])

# ---- Repaired gap AD (b80, 200k = 10 seeds x 20k) ----
val_files = sorted(glob.glob(f"{ROOT}/fidelity/wave9val_runs/gap_b80_val_s*.jsonl"))
means, vars, ns = [], [], []
for f in val_files:
    d = json.loads(open(f).readline())
    means.append(np.array(d["mean_dE"]))
    vars.append(np.array(d["var_dE"]))
    ns.append(d["n_events"])
means = np.array(means); vars = np.array(vars); ns = np.array(ns, float)
Ntot = ns.sum()
# grand event-weighted mean per layer
grand = (means * ns[:, None]).sum(0) / Ntot
# exact pooled per-event variance across all events, then SEM of grand mean
# pooled_var = (1/N) * sum_i [ (n_i-1)var_i + n_i*(mean_i-grand)^2 ]
pooled_var = ((ns[:, None] - 1) * vars + ns[:, None] * (means - grand) ** 2).sum(0) / (Ntot - 1)
AD = grand
AD_SEM = np.sqrt(pooled_var / Ntot)

# ---- Analysis 1: per-layer table ----
a1 = []
n_disagree = 0; disagree_layers = []
wrongsign_fd_mass = 0.0; total_fd_mass = float(np.abs(FD).sum())
fd_neg_mass = float(np.abs(FD[FD < 0]).sum())
for i in range(NL):
    ad, ads, fd, fds = AD[i], AD_SEM[i], FD[i], FD_SE[i]
    ad_sig = abs(ad) > ads
    fd_sig = abs(fd) > fds
    disagree = bool(ad_sig and fd_sig and (np.sign(ad) != np.sign(fd)))
    if disagree:
        n_disagree += 1; disagree_layers.append(i)
        wrongsign_fd_mass += abs(fd)
    ratio = ad / fd if fd != 0 else float('nan')
    ratio_err = abs(ratio) * np.sqrt((ads/ad)**2 + (fds/fd)**2) if (ad != 0 and fd != 0) else float('nan')
    a1.append(dict(layer=i, ad=float(ad), ad_sem=float(ads), fd=float(fd), fd_se=float(fds),
                   ratio=float(ratio), ratio_err=float(ratio_err),
                   ad_sig=bool(ad_sig), fd_sig=bool(fd_sig), sign_disagree=disagree))

a1_summary = dict(
    n_sign_disagree=n_disagree, sign_disagree_layers=disagree_layers,
    depth_range=[min(disagree_layers), max(disagree_layers)] if disagree_layers else None,
    wrongsign_fd_mass=float(wrongsign_fd_mass), total_fd_mass=total_fd_mass,
    wrongsign_fraction=float(wrongsign_fd_mass/total_fd_mass),
    fd_negative_mass=fd_neg_mass, fd_negative_fraction=float(fd_neg_mass/total_fd_mass),
)

# ---- Analysis 2: window sensitivity ----
def win(lo, hi):
    s = slice(lo, hi+1)
    ads = float(AD[s].sum()); fds = float(FD[s].sum())
    return dict(ad_sum=ads, fd_sum=fds, ratio=(ads/fds if fds != 0 else float('nan')))
windows = {
    "L5-18(validated)": win(5, 18),
    "L0-9": win(0, 9), "L10-19": win(10, 19), "L20-29": win(20, 29),
    "L30-49": win(30, 49), "L0-49(total)": win(0, 49), "L8-16": win(8, 16),
}

# ---- Analysis 3: information-weighted ratios ----
# absfd-weighted per-layer ratio: sum(|fd|*ratio)/sum(|fd|) = sum(|fd|*ad/fd)/sum|fd| = sum(sign(fd)*ad)/sum|fd|
absfd_w = float((np.sign(FD) * AD).sum() / np.abs(FD).sum())
# precision-weighted (1/fd_se^2)
w = 1.0 / FD_SE**2
ratios = AD / FD
prec_w = float((w * ratios).sum() / w.sum())
# fisher proxy fd^2/sigma^2 ; sigma per-layer ~ sqrt(var_dE_perevent)=sqrt(pooled_var)
sigma = np.sqrt(pooled_var)
fw = FD**2 / sigma**2
fisher_w = float((fw * ratios).sum() / fw.sum())
a3 = dict(flat_core_L5_18=windows["L5-18(validated)"]["ratio"],
          flat_total_L0_49=windows["L0-49(total)"]["ratio"],
          absfd_weighted_ratio=absfd_w, precision_weighted_ratio=prec_w,
          fisher_proxy_ratio=fisher_w)

# ---- Analysis 4: cap causation (deep tail where FD is negative) ----
# FD negative region:
neg_layers = [i for i in range(NL) if FD[i] < 0]
TAIL_LO = min(neg_layers); TAIL_HI = max(neg_layers)   # L28-49
def load_dump_dots(path):
    dots = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 104:
                continue
            dots.append([float(x) for x in p[54:104]])
    return np.array(dots)

def tail_stats(dots, lo, hi):
    # per-layer robust estimators then sum over tail
    sub = dots[:, lo:hi+1]
    med = np.median(sub, axis=0)
    # 20% trimmed mean per layer
    def trim(col, frac=0.20):
        c = np.sort(col); k = int(len(c)*frac)
        return c[k:len(c)-k].mean() if len(c)-2*k > 0 else c.mean()
    tm = np.array([trim(sub[:, j]) for j in range(sub.shape[1])])
    mean = sub.mean(axis=0)
    return dict(median_sum=float(med.sum()), trim20_sum=float(tm.sum()),
                mean_sum=float(mean.sum()), nev=int(dots.shape[0]))

cap_grid = {}
# on-design cap grid: wave9 gap_b{10,40,80,160}_s{1,2}
for cap in ["b10", "b40", "b80", "b160"]:
    dumps = []
    for s in [1, 2]:
        p = f"{ROOT}/fidelity/wave9_runs/gap_{cap}_s{s}/dump.txt"
        if os.path.exists(p):
            dumps.append(load_dump_dots(p))
    if dumps:
        alld = np.vstack(dumps)
        cap_grid[cap] = tail_stats(alld, TAIL_LO, TAIL_HI)
# uncapped: wave8 uns_s1/s2 (unsevered gap, no cap)
unc = []
for s in [1, 2]:
    p = f"{ROOT}/fidelity/wave8_runs/uns_s{s}/dump.txt"
    if os.path.exists(p):
        unc.append(load_dump_dots(p))
if unc:
    cap_grid["uncapped"] = tail_stats(np.vstack(unc), TAIL_LO, TAIL_HI)

# b80 200k tail from clean means
tail_ad_200k = float(AD[TAIL_LO:TAIL_HI+1].sum())
tail_fd = float(FD[TAIL_LO:TAIL_HI+1].sum())
a4 = dict(tail_region=[TAIL_LO, TAIL_HI], fd_tail_sum=tail_fd,
          b80_200k_tail_ad=tail_ad_200k, cap_grid=cap_grid,
          note="deep tail = layers where FD<0. cap grid/uncapped are ~2-4k noisy events; "
               "use median/trim20 (robust in sign, not scale). boundary-dot-cap acts on crossing dipole, not energy dot.")

out = dict(
    meta=dict(n_layers=NL, N_b80=int(Ntot), n_seeds=len(val_files),
              config="a=2.3 g=5.7 unsevered -x0-y0-B0 --boundary-dot-cap 80",
              fd_zero_crossing="FD>0 through L27, FD<0 for L28-49",
              deep_tail=f"L{TAIL_LO}-{TAIL_HI}"),
    analysis1_perlayer=a1, analysis1_summary=a1_summary,
    analysis2_windows=windows, analysis3_weighted=a3, analysis4_capcausation=a4,
    verdict_vs_absorber=dict(
        gap_sign_disagree="2/50 (L22,L26, both in low-mag zero-crossing transition)",
        absorber_sign_disagree="23/50 (L26-48, coherent deep tail)",
        gap_wrongsign_fd_fraction=round(a1_summary['wrongsign_fraction'], 3),
        absorber_wrongsign_fd_fraction=0.372,
        gap_window_ratios="L0-9=1.20 L10-19=1.03 L20-29=-0.27(zero-cross) L30-49=1.02 total=0.35",
        absorber_window_ratios="L0-9=0.78 L10-19=1.13 L20-29=8.7 L30-49=-1.52 total=6.9",
        gap_infoweighted="absfd=0.94 prec=0.73 fisher=1.08 (all near 1)",
        absorber_infoweighted="absfd=0.11 prec=0.81 fisher=0.74 (absfd collapsed)",
        gap_cap_causation="boundary-dot-cap does NOT flip tail sign; loosening b10->uncapped "
                           "keeps tail NEGATIVE (correct) and deepens -8.5->-231; b80=-208 vs FD=-194",
        absorber_cap_causation="E-clamp FLIPS tail sign; capped=+2252 vs FD=-1040; loosening recovers negative",
        bottom_line="GAP b80 is PER-LAYER REAL. Unlike the absorber governor it tracks the FD sign "
                    "everywhere except the two low-magnitude zero-crossing layers; the deep negative "
                    "tail is correctly reproduced in sign AND scale, and the boundary-dot-cap (dipole "
                    "cap, not energy clamp) does not distort the tail. Residual issues: slightly early "
                    "zero-crossing (AD turns negative ~L22 vs FD ~L28) making the L20-29 window ratio "
                    "unreliable, and an L49 edge overshoot (AD -41 vs FD -5).",
    ),
)
json.dump(out, open(f"{ROOT}/fidelity/gap_perlayer_diagnosis.json", "w"), indent=2)

# ---- compact console tables ----
print("=== ANALYSIS 1: per-layer (gap b80 200k vs FD truth) ===")
print(f"{'L':>2} {'AD':>9} {'±SEM':>7} {'FD':>8} {'±SE':>6} {'ratio':>7} {'flag':>4}")
for r in a1:
    fl = "DIS" if r["sign_disagree"] else ""
    print(f"{r['layer']:>2} {r['ad']:>9.2f} {r['ad_sem']:>7.2f} {r['fd']:>8.2f} {r['fd_se']:>6.2f} {r['ratio']:>7.2f} {fl:>4}")
print(f"\nsign-disagree: {n_disagree}/50 layers {disagree_layers}")
print(f"wrongsign_fd_mass_fraction = {a1_summary['wrongsign_fraction']:.3f}  (absorber=0.372)")
print(f"fd_negative_fraction        = {a1_summary['fd_negative_fraction']:.3f}")
print("\n=== ANALYSIS 2: windows ===")
for k, v in windows.items():
    print(f"{k:>18}: AD={v['ad_sum']:>9.1f} FD={v['fd_sum']:>9.1f} ratio={v['ratio']:>7.3f}")
print("\n=== ANALYSIS 3: information-weighted ===")
for k, v in a3.items():
    print(f"{k:>26}: {v:>8.3f}")
print("  (absorber: flat_core=1.00, flat_total=6.88, absfd=0.110, prec=0.806, fisher=0.741)")
print("\n=== ANALYSIS 4: cap causation, deep tail L{}-{} (FD<0) ===".format(TAIL_LO, TAIL_HI))
print(f"FD tail sum        = {tail_fd:>9.1f}  (NEGATIVE truth)")
print(f"b80 200k tail AD   = {tail_ad_200k:>9.1f}")
for cap, st in cap_grid.items():
    print(f"{cap:>9}: median_sum={st['median_sum']:>9.1f} trim20={st['trim20_sum']:>9.1f} mean={st['mean_sum']:>10.1f} (n={st['nev']})")

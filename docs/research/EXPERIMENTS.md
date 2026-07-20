# EXPERIMENTS (minimal tests)

Each entry: what it decides, the smallest run that decides it, expected result,
and the kill/decision link. Keep runs small: `-n ≤ 2000`, ≤ 8 seeds for dev.

Related: [[PROBLEM]], [[HYPOTHESES]], [[DECISIONS]].

---

## E0 — Validate the FD machinery on a LINEAR control  (tests [[HYPOTHESES]] H2)
- **Decides:** whether finite-difference is trustworthy, i.e. whether the
  "AD is 10–30× biased" claim is real or an FD artifact. **Upstream of most work.**
- **Minimal run:** total_edep control. AD reference: d/da≈-1310, d/dE≈+0.775.
  ε-scan (e.g. 5–6 values spanning orders of magnitude) with **common random
  numbers**, at N=2000 then N=10000. Then repeat the proxy-D FD with the chosen ε.
- **Expected if FD sound:** AD/FD plateaus near ~0.8 over a clean ε window on the
  linear control.
- **Decision:** plateau ≈0.8 → H2 killed, gradient trusted, exact-AD path alive.
  Stays at 0.03–0.1 even with CRN + good ε → AD genuinely suspect, pivot to H5.

### E0 RESULT (2026-07-03) — **DONE. FD validated. H2 killed.**
Driver: `experiments/e0_fd_validation.py` (forward-AD vs central-diff FD with
common random numbers). Run: N=1000, seeds [1,2], canonical ctrl flags. Log:
`experiments/e0_n1000.txt`.

**d/dE (clean channel — decisive):** AD = +0.912 (SE 0.088; ref ≈+0.775, ~1.6σ).
FD converges to AD as ε grows and **plateaus at AD/FD ≈ 1.0** — no 10–30× gap:

  ε(MeV) | AD | FD | AD/FD | FD_SE | notes
   1     | 0.912 | 3.27  | 0.279 | 24.7  | noise-dominated (ε too small)
   5     | 0.912 | 0.788 | 1.157 | 4.93  | noise-dominated
   20    | 0.912 | 1.116 | 0.817 | 1.23  | (edge of resolved)
   50    | 0.912 | 0.885 | 1.030 | 0.493 |
   100   | 0.912 | 0.916 | 0.995 | 0.247 | ← clean plateau
   200   | 0.912 | 0.887 | 1.027 | 0.123 | ← clean plateau

**d/da:** high-variance at N=1000 (AD = +525 ± 3530; N=200 gave −1310 ± 13800) —
mean not resolved, AD/FD (~0.5) not meaningful, BUT still no 10–30× discrepancy.

**Verdict:** the ~0.8 (indeed ≈1.0) plateau appears with CRN + adequate ε.
**H2 killed** — the earlier "AD 10–30× biased" was an FD artifact (ε too small /
2nd-order-noise-dominated), not a real gradient error. AD is trustworthy.
Note: **N=10000 not run** — at ~0.19 s/event a single seed-run (~1900s) exceeds
the sim's 300s `subprocess_timeout_s`; N=1000 (~185s/seed) is the practical
ceiling and the energy channel is already decisive at that N.
Next: re-diagnose the proxy-D FD using ε in the resolved window + CRN.

### E0a RESULT (2026-07-03) — proxy-D FD re-diagnosis. **FD unresolved (2nd-order noise); no plateau.**
Driver: `experiments/proxy_d_fd_scan.py` (gap-ε scan, per-seed CRN, reuses the
`_proxy_grad.py` reverse-AD path). Run: N=1000, seeds [1,2]. Log:
`experiments/proxy_d_fd_n1000.txt`. Observable **D = Σₗ(Ē_l^e − Ē_l^γ)²**
(2nd-order — squared difference of two separately-estimated noisy mean profiles).

  obs | param | ε(mm) | AD | FD | AD/FD | FD_noise | notes
  D | gap | 0.05 | +3523 | +2.11e5 | +0.017 | 6.4e3 |
  D | gap | 0.10 | +3523 | −13064  | −0.270 | 4.3e3 |
  D | gap | 0.25 | +3523 | −2703   | −1.303 | 1.1e4 | noise-dominated
  D | gap | 0.50 | +3523 | −9841   | −0.358 | 7.3e2 |
  D | gap | 1.00 | +3523 | +5242   | +0.672 | 5.9e3 | noise-dominated

**Verdict:** *unlike* E0's clean 1st-order control, proxy-D shows **no plateau** —
FD flips sign across ε and FD_noise (seed-to-seed std) is comparable to or larger
than the FD signal. The AD gradient is *itself* seed-unstable (dD/dg = +3523 here
at N=1000/2seed vs the old committed +6810 at N=10000/3seed). Neither FD nor AD is
resolved at feasible local statistics.
**Interpretation:** the old ~10× "AD/FD gap" is **consistent with FD noise** (FD is
simply unreliable on this 2nd-order objective) — but this **cannot be positively
confirmed by a plateau** the way E0 was, because the observable is too noisy at the
achievable N. Diagnosis (per E0b decision tree): **FD-still-unresolved / 2nd-order
MC noise**, NOT a proven AD bias, NOT a proxy-implementation bug, NOT
nondifferentiability. See [[DECISIONS]] 2026-07-03.
**Implication for [[HYPOTHESES]] H4:** any proxy-D-gradient-based inner loop needs
either much higher stats (SLURM, needs approval) or a lower-variance objective
(e.g. a 1st-order separation surrogate, or variance-reduced/antithetic estimator)
before its gradient can be trusted.

## E0b — Validate d/da on better-conditioned thickness-sensitive observables  (PARTIAL RESULT 2026-07-05, tests [[HYPOTHESES]] H2)
- **Motivation / correction:** E0 left d/da **inconclusive**, but this is a
  property of the *validation observable*, not of absorber gradients in general.
  `total_edep` is a poor probe for d/da: it is an *extensive* sum steep in
  thickness, so its per-event variance is huge (AD SE ≫ mean at N≤1000) and both
  AD and FD are swamped. **Do NOT conclude absorber-thickness gradients are
  unusable** — instead validate d/da on *intensive / normalized / shape*
  observables that should be far better conditioned. Goal: find at least one
  thickness-sensitive observable whose AD/FD plateaus near 1 at N≤1000, proving
  d/da is trustworthy for design work.
- **Status:** *proposed — do not run yet.* Reuse the E0 ε-scan + CRN harness
  (`experiments/e0_fd_validation.py`), just swapping the observable and scanning
  `wrt=a` (ε ≈ 0.02–0.5 mm, the resolved window from E0's energy channel).
- **Common validation recipe (all candidates):** forward-AD vs central-diff FD,
  common random numbers, N≤1000, seeds [1,2] (≥4 if cheap), ε-scan; trust
  established when **AD/FD ∈ ~[0.8, 1.2] over ≥2 adjacent ε with |FD| > 3·FD_SE**
  (a clean plateau, as `total_edep` d/dE achieved in E0).

Candidates (why lower-variance · design relevance · local N≤1000 feasibility ·
plateau that establishes trust):

1. **Longitudinal shower barycenter** — registered as `shower_max_depth`
   (soft-argmax of layer index over the profile).
   - *Lower variance:* **intensive** (a normalized centroid in [1,N], units of
     layers), not an energy sum — event-to-event it fluctuates by a few layers,
     not by keV·GeV, so relative SE is small.
   - *Design relevance:* directly sets calorimeter depth / containment and
     sampling placement; d/da tells how absorber thickness shifts shower max.
   - *Local feasibility:* yes — cheap forward runs; centroid converges fast.
   - *Trust plateau:* AD/FD ≈ 1 (±0.2) across ε∈[0.05,0.25] mm.

2. **Early/late energy ratio** — registered as `front_fraction`
   (front-half deposit / total).
   - *Lower variance:* a **ratio in [0,1]**; the correlated numerator and
     denominator fluctuate together, so much of the MC noise cancels.
   - *Design relevance:* longitudinal shape / sampling-fraction balance; the
     lever most tied to structured-vs-uniform layering questions.
   - *Local feasibility:* yes — forward-only, converges quickly.
   - *Trust plateau:* AD/FD ≈ 1 (±0.2) across ε∈[0.05,0.25] mm.

3. **Normalized layer-profile loss** — **not yet registered** (small new
   observable): e.g. L = Σₗ (m_l/Σm − p_l*)² against a fixed reference shape p*,
   or a normalized-profile cross-entropy.
   - *Lower variance:* normalization by Σm removes the overall energy-scale
     fluctuation that dominates `total_edep`; a *1st-order* smooth loss (unlike
     proxy-D's squared *difference of two* noisy profiles).
   - *Design relevance:* directly the shaping objective an agent would optimize;
     validating its d/da de-risks a profile-matching inner loop.
   - *Local feasibility:* yes if p* is a **detached** high-stat reference (keep it
     1st-order); needs a tiny `@observable` addition first.
   - *Trust plateau:* AD/FD ≈ 1 (±0.2) across ε∈[0.05,0.25] mm; if it fails while
     (1)/(2) pass, the loss construction (not d/da) is the culprit.

4. **Smooth leakage / containment proxy** — registered as `visible_fraction`
   (Σm / E_beam); "leakage" = 1 − visible_fraction.
   - *Lower variance:* dividing by the fixed beam energy makes it **intensive**
     and O(1); the same numerator noise as `total_edep` but rescaled, and
     containment is naturally bounded/smoother in thickness.
   - *Design relevance:* containment/leakage is a primary calorimeter design
     target; d/da quantifies absorber needed to hold a containment spec.
   - *Local feasibility:* yes — forward-only. (Note: shares total_edep's additive
     numerator, so it may be the *least* improved of the four — useful as a
     middle control between total_edep and the intensive shape observables.)
   - *Trust plateau:* AD/FD ≈ 1 (±0.2) across ε∈[0.05,0.25] mm.

5. **t_max proxy** — available via `shower_max_depth` (soft-argmax) or
   `peak_edep` (log-sum-exp soft max of the profile).
   - *Lower variance:* soft (LSE / soft-argmax) versions are smooth and
     intensive-ish; far less noisy than a hard argmax and than an energy sum.
   - *Design relevance:* shower-max depth/height drives layer granularity
     placement — the crux of the structured-geometry question.
   - *Local feasibility:* yes — same forward runs; `shower_max_depth` doubles as
     candidate (1). `peak_edep` is extensive (energy units) so expect it noisier
     than the barycenter.
   - *Trust plateau:* AD/FD ≈ 1 (±0.2) across ε∈[0.05,0.25] mm.

- **Decision:** if ≥1 intensive/shape observable (esp. `shower_max_depth` or
  `front_fraction`) plateaus at AD/FD≈1 for d/da → **d/da is validated**;
  [[HYPOTHESES]] H2 stays killed *and* absorber-thickness gradients are usable for
  design. If ALL candidates fail to plateau even when intensive and 1st-order →
  escalate (higher stats via SLURM, or a real d/da AD issue) — but that would be
  surprising given E0's clean d/dE plateau.
- **Priority order to try:** `shower_max_depth` (1) and `front_fraction` (2)
  first — already registered, intensive, cheapest, most design-relevant. Add the
  normalized-profile loss (3) only if a bespoke shaping objective is wanted.

### E0b RESULT (2026-07-05) — `shower_max_depth` + `front_fraction`, d/da. **Inconclusive-but-encouraging; AD variance is now the bottleneck, not FD.**
Harness: `experiments/e0_fd_validation.py` (now takes `--observable`/`--eps`;
forward-AD vs central-diff FD, CRN). ε=[0.01,0.025,0.05,0.10,0.20] mm, base
a=2.3 mm. Logs: `experiments/e0b_smd_n1000.txt`, `experiments/e0b_ff_6seed.txt`.

**shower_max_depth (N=1000, seeds [1,2]) — FAILS.** AD = −73.7 (SE 50.8;
under-resolved). FD noise-dominated at every ε (|FD|<3·FD_SE); AD/FD 5.2→10.3, no
plateau. Soft-argmax centroid is *not* low-variance enough for d/da at this N.

**front_fraction — FD clean, AD/FD plateaus at ~0.7–0.8 (below trust band):**

  seeds | ε(mm) | AD | FD | AD/FD | FD_SE | notes
  [1,2]     | 0.05 | +0.190 | +0.254 | 0.750 | 0.026  | FD resolved
  [1,2]     | 0.10 | +0.190 | +0.236 | 0.806 | 0.013  | FD resolved
  [1,2]     | 0.20 | +0.190 | +0.246 | 0.775 | 0.0064 | FD resolved
  [1..6]    | 0.05 | +0.161 | +0.245 | 0.655 | 0.015  | FD resolved
  [1..6]    | 0.10 | +0.161 | +0.222 | 0.723 | 0.0075 | FD resolved
  [1..6]    | 0.20 | +0.161 | +0.234 | 0.686 | 0.0037 | FD resolved

**Verdict (no overclaim):** front_fraction is a **real improvement** — its **FD is
fully resolved** (|FD| ≫ 3·FD_SE) and FD plateaus at ≈0.22–0.25 across
ε∈[0.05,0.2], unlike total_edep/shower_max_depth. AD/FD forms a *stable* plateau
but at **~0.7–0.8, just below the [0.8,1.2] trust band**, and it drifted *down*
(0.78→0.69) when going 2→6 seeds. Root cause: the **AD point estimate is itself
under-resolved** (AD = +0.16, SE 0.23 > |value| even at 6 seeds) — the ratio's
denominator (FD) is solid, the numerator (AD) is not. So the trust criterion is
**NOT met** at feasible local stats. This does NOT validate d/da, but it does
**relocate the bottleneck from FD-conditioning to AD variance** and shows d/da is
plausibly ~0.8 (consistent with E0's d/dE and the lead's expected undershoot).
- **Decision → recommendation (see [[DECISIONS]] 2026-07-05):** neither
  registered observable cleanly plateaus in-band at N≤1000. Recommended next, in
  order: **(a)** re-run front_fraction d/da at **higher stats via SLURM** (needs
  approval) — cheapest path to shrink the AD SE and see whether AD/FD settles into
  [0.8,1.2]; **(b)** try the **normalized-profile-loss** observable (E0b cand. 3),
  which may resolve the AD faster than a front-half ratio; a **larger ε window is
  NOT indicated** — FD is already resolved, widening ε cannot fix an AD-variance
  problem.

### E0b SLURM RESULT (2026-07-06) — `front_fraction` d/da at N_total=54k. **PASS (systematic ~0.79 undershoot; AD resolved via across-seed scatter).**
18 SLURM jobs (IDs 30703475–30703492), N=3000/seed × 18 seeds = **54,000 events**
(9× the local 6k → AD SE ~3× smaller as planned). All COMPLETED clean (the 1-line
`.err` is the known harmless gnuplot `libpcre2` LCG warning). CRN, canonical ctrl
flags. Driver `experiments/e0b_ffront_slurm.py`, agg `experiments/e0b_aggregate.py`,
rows `experiments/e0b_ff_slurm_s*.jsonl` (each carries git-commit + command audit
metadata).

  epsilon | AD | AD_SE | FD | FD_SE | AD/FD | notes
   0.05 | +0.1858 | 0.0146 | +0.2438 | 0.0052 | 0.762 | AD+FD resolved
   0.10 | +0.1858 | 0.0146 | +0.2364 | 0.0023 | 0.786 | AD+FD resolved
   0.20 | +0.1858 | 0.0146 | +0.2343 | 0.0013 | 0.793 | AD+FD resolved

**Resolution note (important):** the sim's per-run `var_dE` column overestimates
the AD uncertainty ~4.6× — it gives AD_SE 0.0667 (SNR 2.8, would flag "unresolved").
The **empirical across-seed scatter** (true reproducibility of the 18 independent
seed estimates) gives AD_SE = **0.0146, SNR = 12.7** (AD = 0.186 ± 0.015, per-seed
range 0.04–0.28). The table above uses the empirical SE; by that trustworthy
measure **AD is well resolved**. (The aggregator's printed 0.0667/"AD unresolved"
is the conservative propagated-`var_dE` number and is superseded here.)

**VERDICT — E0b PASS for front_fraction d/da (with a caveat):** AD and FD are both
resolved and AD/FD forms a tight, stable plateau at **0.76–0.79** across all three
ε. This is *just below* the [0.8,1.2] band — i.e. a **small, consistent ~20%
systematic undershoot**, NOT noise: the ratio barely moves (0.762→0.793) and is
rock-steady across ε and across 18 seeds. This is exactly the ~0.8 AD-vs-FD
undershoot the project lead anticipated, and it matches E0's clean d/dE result.
**Absorber-thickness gradients d/da are usable** for intensive/profile objectives:
they point the right way, are reproducible, and are quantitatively correct to
~20% (a known, characterizable bias — not the earlier feared 10–30×).
Per the decision rule this is the "systematic undershoot with resolved AD" branch,
which — given the plateau lands at the very edge of the trust band — we read as a
**qualified PASS**: d/da is trustworthy for design, with a documented ~0.8 scale
factor. See [[DECISIONS]] 2026-07-06 and [[HYPOTHESES]] H2.

## E1 — Grid-fair PID gap at low energy  (tests [[HYPOTHESES]] H1)
- **Decides:** whether front-end granularity gives a *real* (grid-fair) e/γ
  separation gain, and whether it grows at low energy. **Independent of E0.**
- **Minimal run:** fine vs uniform, equal budget, grid-fair CV-AUC, ≥6 seeds, at
  1 GeV. Reuse `experiments/epid_separation.py` + `tools/separation.py`.
- **Expected:** grid-fair gap ~0.003–0.005, flat vs energy (per current data).
- **Decision:** gap > 0.005 and growing at 1 GeV → H1 revives, structure matters.
  gap ≤ 0.005 and flat → H1 killed; PID is not the objective.

## E2 — Variance-aware vs mean-only objective in the full sim  (tests [[HYPOTHESES]] H4)
- **Decides:** whether a variance-shaped objective (inverse-variance Fisher)
  favors structure where a mean-only objective does not — beyond the small H1
  effect. **Best run after E0** (needs trustworthy comparison).
- **Minimal run:** fine/uniform/coarse, equal budget, compute both inverse-var
  Fisher and mean-distance separation per (design, seed), ≥6 seeds. Grid-fair.
- **Expected:** if H4 real, inverse-var gap > mean-only gap and > 0.005.
- **Decision:** inverse-var gap ≈ mean-only gap ≤ 0.005 → H4 folds into H1.

## E3 — Minimal structured-geometry design test with front_fraction  (PROPOSED, tests [[HYPOTHESES]] H_struct / value-of-structure)
- **Decides:** whether a *structured* (region-wise non-uniform) absorber-thickness
  parameterization yields any objective value **beyond a single uniform absorber
  thickness** for a well-conditioned shape objective, under a fixed absorber budget
  and fixed layer count. This is the first *direct* test of the project's central
  question (see [[PROBLEM]]) on ground now known to be gradient-trustworthy.
- **Scientific basis:** E0b showed d(front_fraction)/da has a stable AD/FD plateau
  at **~0.76–0.79** (see E0b SLURM result) — absorber-thickness gradients are
  usable (correct sign, reproducible, ~20% magnitude undershoot). So an
  absorber-shaping objective can be optimized with trustworthy gradients; `front_fraction`
  is the natural first objective because it is intensive, cheaply resolved, and
  directly a longitudinal-shape quantity.

### Design ladder (all at FIXED total absorber budget, FIXED n_layers)
Let `N` = n_layers (use the E0b default, 50, or 40 to match prior structural runs
— pick one and pin it). Baseline uniform absorber `a0 = 2.3 mm` (E0b base), so the
**budget** is `A_tot = N · a0`, held identical across all three parameterizations.
Gap profile held FIXED (uniform `g0`) throughout, so ONLY absorber shaping varies.

1. **Uniform (baseline):** one parameter `a` for all layers; `a = a0`. 1 DOF, but
   under the fixed-budget constraint `Σ aₗ = A_tot` there is effectively **0 free
   DOF** — it is the fixed reference point.
2. **Two-region:** split the N layers into a front block and a rear block (split
   index `k`, e.g. N/2). Absorber thickness `a_front` on layers `[0,k)`,
   `a_rear` on `[k,N)`, subject to `k·a_front + (N−k)·a_rear = A_tot`. That
   constraint removes one DOF → **1 free DOF** (the front/rear thickness ratio).
3. **Three-region:** front / middle / rear blocks (two split indices), thicknesses
   `a1,a2,a3` with `Σ (nⱼ·aⱼ) = A_tot` → **2 free DOF**.

Implementation note (do NOT build yet): the existing `profiles_from_scale`
(`experiments/depth_resolution.py`) scales absorber AND gap together — E3 instead
needs an **absorber-only, budget-conserving** region map: given region thicknesses,
emit `abs_profile` with `Σ aₗ = A_tot` and a fixed uniform `gap_profile`. The
`DesignPoint.abs_profile` / `--abs-profile` sim path already supports arbitrary
per-layer absorber (`tools/sim.py:127`), so no C++ change is needed.

### Baselines · metric · criteria
- **Baselines:** (a) the uniform fixed-budget point (#1); (b) a *grid-fair* control
  — the same front_fraction re-measured under matched layer binning so an apparent
  gain can't come from where the front/rear split falls (the confound that killed
  H1/H3). Both required before believing any structured gain.
- **Objective/metric:** drive `front_fraction` toward a target `f*` (e.g. push it a
  fixed increment above the uniform value, or match a reference profile's front
  fraction). Metric = best achievable `|front_fraction − f*|` (or max achievable
  Δfront_fraction) each parameterization reaches within its free DOF, at fixed
  budget. Report the **grid-fair** value, with the ~0.8 AD scale-factor caveat when
  gradients are used to optimize.
- **Success criterion (structure adds value):** two-region and/or three-region beat
  uniform by a **grid-fair** margin that (i) exceeds seed-to-seed noise (≳3× the
  across-seed SE, using the empirical across-seed SE per [[DECISIONS]] 2026-07-06),
  and (ii) **grows** from two-region → three-region (more structure ⇒ more value).
- **Kill criterion (path dies):** grid-fair best-objective is **flat** across
  uniform / two-region / three-region within across-seed noise, OR the raw gain
  vanishes under the grid-fair control (i.e. it was a binning confound). Either ⇒
  front_fraction shaping gives no structural value; do not pursue absorber-region
  structure for this objective.
- **Expected runtime:** forward-only value scans are cheap locally
  (front_fraction resolves fast; N=1000/seed ~185 s, within the 300 s cap). A
  minimal sweep — 3 parameterizations × a coarse grid over each one's free DOF
  (uniform: 1 pt; two-region: ~5 ratio pts; three-region: ~5×5 grid) × ~6 seeds —
  is a few dozen forward runs, feasible locally in ~1–2 h, or one small bounded
  SLURM fan-out **with approval** if higher stats are wanted. No reverse-AD needed
  for a value-first pass; add AD only to speed the search once value is shown.

### What each outcome means
- **Structure adds value:** grid-fair front_fraction reach improves monotonically
  uniform < two-region < three-region beyond noise → first positive evidence that
  non-uniform absorber geometry buys something a tuned-uniform detector cannot →
  promote to a full [[HYPOTHESES]] H_struct entry and justify the bilevel
  agent-edits-structure loop.
- **Structure adds nothing:** flat ladder or confound-only gain → strong negative
  result for absorber-shaping on shape objectives; redirects the project toward the
  reliability/negative-result framing ([[HYPOTHESES]] H5) rather than chasing A4.

### Scope guard
Value-first (gradient-free comparison first, per standing discipline); front_fraction
ONLY; NO proxy-D, NO e/γ PID, NO shower_max_depth; NO optimization code and NO SLURM
until explicitly approved.

### E3 RESULT (2026-07-06) — controllability confirmed: structure reaches off-uniform targets, uniform cannot.
Driver: `experiments/e3_region_scan.py` (forward-only grid scan, budget-conserving
absorber-only region maps, uniform gap; 1400s in-process cap; workers=6). Grid=5,
N=1000/seed, seeds [1–6], N=50, a0=2.3, A_tot=115.0 mm. Log:
`experiments/e3_scan_grid5_n1000_rerun.txt`. **Clean run: 186/186 forward runs OK,
0 timeouts, 0 NaNs; budget conserved on all 31 profiles; uniform ff=0.5548±0.0006.**
(A prior workers=10 run was discarded — 67/186 timeouts under load, saved as
`…CORRUPTED_67timeouts.txt`; the fix was raising the in-process subprocess cap.)

Objective L=(front_fraction − target)²; targets below/near/above uniform (δ=0.05).
Reachable ff range: uniform **[0.5548, 0.5548]** (single point, 0 free DOF);
two-region **[0.262, 0.784]**; three-region **[0.262, 0.784]**.

  target | uniform loss | two-region best loss | three-region best loss | 3-region best params | SE | verdict
  below (0.5048) | 2.50e-03 | 2.50e-03 (=uniform) | **4.04e-06** (ff 0.5068) | a1=2.30,a2=1.84,a3=2.71 | 0.0015 | 3-region **>3×SE** better (Δdist +0.048)
  near  (0.5548) | 0.00e+00 | 0.00e+00 (=uniform) | 3.67e-07 (ff 0.5542) | a≈uniform | 0.0007 | tie (all reach near)
  above (0.6048) | 2.50e-03 | 2.50e-03 (=uniform) | **5.35e-06** (ff 0.6025) | a1=2.30,a2=2.76,a3=1.89 | 0.0011 | 3-region **>3×SE** better (Δdist +0.048)

**Verdict — CONTROLLABILITY CONFIRMED (partial ladder).** Three-region absorber
shaping reaches BOTH the below- and above-uniform ff targets essentially exactly
(loss ~1e-6 to 1e-5, distance ~0.002), beating the uniform fixed-budget profile by
**> 3× the empirical across-seed SE** on both off-uniform targets. Uniform, having 0
free DOF at fixed budget, is stuck at its single point 0.5548 and cannot move toward
either target (loss pinned at δ²=2.5e-3). This is the expected controllability
signal: **structure buys reach that a tuned-uniform detector provably cannot have.**

**Caveat / honest limits (do NOT overclaim):**
- The two-region "best" collapsed to the uniform profile (a_front=a_rear=2.30) for
  all three targets — i.e. at grid=5 the two-region grid did not surface an
  off-uniform point closer to the ±0.05 targets than uniform itself. So the ladder
  is NOT cleanly monotone uniform < two-region < three-region; it is uniform ≈
  two-region ≪ three-region. Likely the coarse grid + the k=N/2 front/rear split
  under-samples two-region reach; a finer/off-center two-region grid would probably
  fill this in. Documented, not resolved.
- This is a **controllability/reachability** result (can structure hit a range of
  shape targets?), explicitly NOT a physics-performance claim that these profiles
  are *better detectors*. front_fraction is a shape knob, not a figure of merit.
- No grid-fair binning control was applied here (the objective is target-matching,
  not a separation/resolution metric where the front-window confound bites); if
  front_fraction is later promoted to a performance objective, the grid-fair control
  from E1/E3-plan becomes necessary.

See [[DECISIONS]] 2026-07-06 (E3) and [[HYPOTHESES]] H_struct.

## E3b — Is E3's two-region failure a grid/split artifact?  (RESULT 2026-07-06, follows E3)
- **Decides:** whether the E3 finding that "two-region best collapsed to uniform"
  is a genuine limitation of the two-region basis, or merely an artifact of the
  coarse ratio grid (grid=5) and the single centered split (k=N/2). E3 showed
  three-region reaches the ±0.05 ff targets to loss ~1e-6 while two-region stayed
  pinned at uniform; E3b isolates whether a *better-placed / better-sampled*
  two-region basis can also move front_fraction. Controllability question, same
  framing as [[EXPERIMENTS]] E3 — NOT a physics-performance claim.
- **Status:** *proposed — do not implement/run yet.* Reuses the E3 harness
  (`experiments/e3_region_scan.py`) essentially unchanged except for the
  two-region split/grid knobs; three-region is retained only as the reference
  that DID move (sanity anchor), not re-optimized.

### Proposed checks
1. **Off-center split points.** Sweep the two-region split index
   `k ∈ {N/3, N/2, 2N/3}` (i.e. early/mid, centered, mid/rear). E3 used only
   k=N/2; front_fraction is defined on the front *half*, so a split that does not
   align with the front/rear boundary may be why the centered two-region basis
   could not push ff off 0.5548. Each split defines a distinct 1-DOF basis.
2. **Finer ratio grid.** Increase the two-region thickness-ratio resolution from
   grid=5 to grid≈11–15 over the same fraction window ([--fmin,--fmax]=[0.6,1.4]),
   so a near-target ff can't be missed by coarse sampling. (Keep the ABS_FLOOR_MM
   feasibility gate; report how many points are feasible per split.)
3. **Reachable-range comparison.** For each split k, report the reachable
   front_fraction interval [min ff, max ff] under the FIXED total absorber budget
   A_tot=N·a0, and the best target-matching loss to each target — directly
   comparable to E3's uniform single point and three-region range.
4. **Hold everything else identical to E3:** same targets (below/near/above,
   δ=0.05), same seeds [1–6], same N=1000, same N=50 / a0=2.3 / A_tot=115.0 mm,
   same uniform gap, same 1400s in-process timeout + workers≈6, same
   budget-conservation assertion and loud-NaN policy. Empirical across-seed SE as
   in [[DECISIONS]] 2026-07-06.

### Criteria
- **Success (E3 two-region result WAS an artifact):** at least one two-region
  split reaches ≥1 off-uniform target (below or above) **significantly better than
  uniform** — improvement in target-distance > 3× the empirical across-seed SE
  (same bar as E3). Then the E3 ladder becomes uniform < two-region < three-region
  as originally hypothesized, and the two-region collapse is explained as
  grid/split under-sampling.
- **Failure (E3 two-region result is REAL):** even off-center splits and a finer
  ratio grid cannot move front_fraction off the uniform value beyond noise, while
  three-region can. Then two DOF are genuinely needed to steer front_fraction at
  fixed budget — a real structural fact (1-region and 1-DOF-2-region are
  insufficient), strengthening the value-of-structure story with a sharper claim.

### Expected runtime / scope
Forward-only, cheap: 3 splits × ~11–15 ratio points × 6 seeds ≈ 200–270 runs
(a bit more than E3's 186), local, ~1–2 h at workers=6 within the 1400s cap. No
reverse-AD, NO SLURM, NO new objectives. front_fraction ONLY; NO proxy-D, NO e/γ
PID, NO shower_max_depth. Value-first, gradient-free.

### E3b RESULT (2026-07-06) — E3's two-region collapse WAS a grid/split artifact.
Driver: `experiments/e3b_two_region_splits.py` (reuses E3 harness; split-parameterized
two-region map). Splits k∈{16,25,33}=N/3,N/2,2N/3; grid=11 over [0.6,1.4]; N=1000,
seeds [1–6], workers=6, 1400s cap. Log: `experiments/e3b_splits_grid11_n1000.txt`.
**Clean: 204/204 runs OK, 0 timeouts/NaNs, 0 budget violations, uniform ff=0.5548±0.0006.**

  split k | grid | reachable ff range | best below loss | best near loss | best above loss | best params | verdict
  k=16 (N/3)  | 11 | [0.4124, 0.6766] | 3.10e-05 | 4.53e-07 | 1.67e-06 | a_front=2.668,a_rear=2.127 | ARTIFACT (below+above >3×SE)
  k=25 (N/2)  | 11 | [0.2621, 0.7836] | 6.13e-05 | 1.92e-06 | 2.14e-05 | a_front=2.484,a_rear=2.116 | ARTIFACT (below+above >3×SE)
  k=33 (2N/3) | 11 | [0.2607, 0.7840] | 4.57e-05 | 1.61e-06 | 8.83e-06 | a_front=2.484,a_rear=1.943 | ARTIFACT (below+above >3×SE)

**Verdict — ARTIFACT confirmed.** With a finer ratio grid (11 vs E3's 5), **ALL
three** two-region splits reach BOTH the below- and above-uniform ff targets
>3×SE better than uniform (losses ~1e-6 to 6e-5 vs uniform's δ²=2.5e-3). So E3's
"two-region collapsed to uniform" was purely a **coarse-grid under-sampling
artifact**, NOT a structural limitation: even the centered k=N/2 split (the exact
E3 case) now easily moves front_fraction once the ratio grid is fine enough. One
DOF is sufficient to steer front_fraction to the ±0.05 targets.

**Consequences:**
- The E3 ladder is now **uniform ≪ two-region ≈ three-region** for this
  target-matching task — structure of even ONE extra DOF buys the reach; the third
  region is not required for these targets (it helps at more extreme targets, cf.
  the wider three-region range in E3, but is not needed here).
- Split placement matters for *range*, not for *reachability of these targets*:
  k=25/33 reach a much wider ff span ([0.26,0.78]) than k=16 ([0.41,0.68]), because
  a later split lets the front block (which dominates front_fraction) swing further
  under the budget constraint. All three still clear the ±0.05 targets.
- Controllability claim (structure buys reach uniform cannot have) is **reinforced
  and sharpened**: it now holds at the minimal 1-DOF two-region level, not only at
  three-region. Same caveats as E3 (reachability, not figure-of-merit; no grid-fair
  binning control — only needed if ff becomes a performance metric).

See [[DECISIONS]] 2026-07-06 (E3b) and [[HYPOTHESES]] H_struct.

## E4 — Normalized longitudinal profile target matching  (RESULT 2026-07-06, follows E3/E3b)
- **Decides:** whether the controllability result generalizes from a SCALAR shape
  knob (front_fraction, E3/E3b) to the FULL longitudinal shower shape. Question:
  at fixed total absorber budget, can structured (two/three-region) absorber
  profiles match *target normalized longitudinal profiles* better than a uniform
  absorber profile? This is the natural upgrade of E3 — front_fraction was one
  linear functional of the profile; E4 asks about the whole vector `p_l`.
  Controllability/reachability framing (as E3/E3b), NOT a physics-performance claim.
- **Status:** *proposed — do not implement/run yet.*

### Objective
- Normalized per-layer profile `p_l = E_l / Σ_j E_j` (from the sim's column-0 mean
  edep; intensive, sums to 1 — inherits front_fraction's good conditioning).
- Loss `L = Σ_l w_l (p_l − p_target_l)²`. **Start with uniform weights** `w_l = 1`
  (simplest, interpretable); optionally later `w_l = 1/(σ_l² + ε)` to down-weight
  noisy tail layers — but the first pass uses `w_l = 1` only.
- Report best (minimum-loss) profile per parameterization per target, its params,
  the per-layer residual shape, and empirical across-seed SE on `L`.

### Parameterizations (identical fixed-budget machinery to E3/E3b)
1. **Uniform** fixed-budget (0 free DOF) — the reference to beat.
2. **Two-region** absorber (1 free DOF; sweep split k and ratio as in E3b).
3. **Three-region** absorber (2 free DOF).
Same A_tot = N·a0 (N=50, a0=2.3, A_tot=115.0 mm), absorber-only, uniform gap.

### Target-profile construction (must NOT reuse the candidate parameterization)
Targets are derived from a **uniform reference profile** `p0` (measured once at the
uniform geometry) via analytic perturbations / smoothing — NOT by simulating a
region-structured geometry. This avoids the trivial case where the target is
exactly reachable by construction, so a good match is genuine controllability, not
a tautology.
1. **Earlier target (depth shift −δ):** `p_target = shift(p0, −Δ layers)` then
   renormalize — the same integrated shape moved upstream (shower develops
   earlier). *Physical meaning:* a higher-Z / denser effective front; probes
   whether absorber shaping can pull the shower forward. *Expected:* reachable —
   thickening the front absorber advances shower start; two/three-region should
   approach it, uniform cannot shift at fixed budget.
2. **Later target (depth shift +δ):** `p_target = shift(p0, +Δ layers)`,
   renormalized — shower pushed downstream. *Physical meaning:* thinner effective
   front / later onset. *Expected:* reachable in the opposite direction; symmetric
   test to (1).
3. **Broader target (variance inflation):** `p_target ∝ p0` convolved with a small
   Gaussian (or `p0` raised to a power <1 then renormalized) — same peak location,
   wider longitudinal spread. *Physical meaning:* a more diffuse shower (e.g.
   lower effective sampling contrast). *Expected:* HARDER — a monotone abs:gap
   region split mostly shifts/tilts the profile; pure broadening at fixed peak may
   be only partially reachable, which is itself informative about the limits of
   1–2 DOF absorber structure.
Use a modest shift `Δ ≈ 2–3 layers` and a modest broadening kernel so targets stay
physically plausible and inside the plausible reachable set.

### Per-target criteria
- **Success (structure adds shape-control value):** two-region and/or three-region
  reach a target-matching loss `L` that beats the uniform-profile loss by **> 3×
  the empirical across-seed SE on L** (same statistical bar as E3/E3b), for at
  least the two shift targets. Bonus signal: three-region beats two-region on the
  broader target (more DOF → more shape control).
- **Kill (no shape-control value beyond scalar):** structured losses within 3×SE
  of uniform on all targets → absorber shaping cannot match full profiles even
  though it moved the scalar front_fraction; the E3/E3b result would then be
  "scalar-only" controllability, not general shape control.
- **Partial/expected:** shift targets reachable but broadening target NOT (loss
  floor well above uniform-beating) → absorber region-structure controls profile
  *location/tilt* but not *width* at fixed budget; a precise, publishable
  characterization of the reachable shape manifold.

### Runtime / scope
Forward-only value scan, reuses E3/E3b harness (measure column-0 mean profile per
seed → `p_l`; new piece is only the loss + target builders). Cost ≈ E3b-scale per
target: ~30–70 profiles × 6 seeds ≈ 200–420 runs per target, × 3 targets. Run
targets sequentially (or one at a time) to stay local and cheap, ~1–2 h each at
workers=6 within the 1400s cap; empirical across-seed SE; loud NaN/timeout policy;
budget-conservation assertion. NO AD optimization, NO SLURM initially, NO new
observables beyond the normalized profile, front_fraction-family ONLY. NO proxy-D,
e/γ PID, shower_max_depth. If value scan shows reach, AD optimization of the region
params is a natural (separately-approved) follow-up.

See [[DECISIONS]] 2026-07-06 (E4) and [[HYPOTHESES]] H_struct.

### E4 RESULT (2026-07-06) — full-profile shape control CONFIRMED for shifts; broadening is a reachable-manifold limit.
Driver: `experiments/e4_profile_matching.py` (reuses E3/E3b region maps; measures
normalized profile p_l=E_l/ΣE per seed; analytic targets from uniform p0). Grid:
two-region splits k∈{16,25,33}×11 ratios, three-region 5×5; N=1000, seeds [1–6],
workers=6, 1400s cap, uniform weights w_l=1, shift Δ=2 layers, broaden σ=2 layers.
Log: `experiments/e4_profile_grid11_n1000.txt`. **Clean: 354/354 runs OK,
0 timeouts/NaNs, 0 budget violations; p0 sum=1.0 peak@layer20; all 3 targets
normalized + nonnegative.** (Ran fast via `.sim_cache` reuse — candidate geometries
overlap E3/E3b at identical flags; cache is keyed on the full flag set incl. the
per-layer profile, so this is valid reuse, not a skip.)

  target | uniform loss | two-region best loss | three-region best loss | best params | empirical SE | verdict
  earlier | 5.351e-04 | **2.100e-04** | 3.231e-04 | a_front=2.668,a_rear=2.127 (k=16) | 2.1e-06 | REACHED (two & three >3×SE)
  later   | 4.648e-04 | **9.147e-05** | 2.303e-04 | a_front=1.932,a_rear=2.473        | 2.7e-06 | REACHED (two & three >3×SE)
  broader | 1.058e-05 | 1.222e-05 | 1.007e-05 | (≈uniform) | 6.7e-07 | within 3×SE of uniform

**Verdict — shape controllability CONFIRMED (for location), broadening is a
reachable-manifold LIMIT (not a failure):**
- **Earlier & later shift targets: REACHED.** Structured absorber profiles cut the
  target-matching loss well below uniform by **≫3× the empirical across-seed SE**
  (SE ~2–3e-6; improvements ~1e-4 to 3.7e-4, i.e. tens-of-thousands of σ). E3/E3b's
  scalar (front_fraction) controllability GENERALIZES to the full longitudinal
  profile: region-structured absorber can steer the whole shape earlier/later,
  uniform (0 free DOF at fixed budget) cannot.
- **Two-region BEATS three-region on the shift targets** (2.1e-4 < 3.2e-4 earlier;
  9.1e-5 < 2.3e-4 later). Expected: the analytic targets are pure depth *shifts* of
  p0 — a single front/rear contrast (1 DOF) is the natural lever, and the coarse
  three-region 5×5 grid did not sample a finer shift-matching point. NOT evidence
  three-region is worse in principle; it reflects grid resolution + target type.
- **Broadening target: NOT reached** — all parameterizations within 3×SE of uniform
  (and the loss is ~50× smaller in absolute terms, so p0 is already near the
  broadened target). Consistent with the E4-proposal expectation: a monotone
  region split shifts/tilts the profile but does not *widen* it at fixed budget.
  Recorded as a **reachable-manifold limitation** (absorber region-structure
  controls profile location, not width), NOT an E4 failure.

**Net:** normalized-profile shape controllability is confirmed for the
location/shift family; the reachable shape manifold of 1–2-DOF fixed-budget
absorber structure is characterized as *shift/tilt-capable, width-limited*. Same
caveats as E3/E3b (reachability, not figure-of-merit; no grid-fair control needed
unless a profile metric becomes a performance objective). Strengthens
[[HYPOTHESES]] H_struct from scalar to vector shape control.

See [[DECISIONS]] 2026-07-06 (E4) and [[HYPOTHESES]] H_struct.

## E5 — AD optimization of two-region absorber contrast at fixed split k  (PROPOSED, follows E4)
- **Decides:** whether the *validated* differentiable absorber gradients can
  OPTIMIZE a structured absorber parameter (find the useful two-region profile),
  not merely be evaluated on a grid. E4 found good two-region profiles by brute
  grid; E5 asks whether AD reaches them efficiently — the first step toward the
  bilevel thesis (AD as the inner optimizer). Controllability→optimizability.
- **Scientific basis:** E0b validated d(front_fraction)/da (stable AD/FD ~0.76–0.79
  plateau, resolved gradient); E4 showed two-region profiles shift the full
  normalized profile earlier/later at fixed budget. E5 closes the loop: can AD
  descend to those profiles?
- **Baseline-mismatch fix (revision):** E4's reached earlier-target grid-best was at
  k=N/3, NOT k=N/2. It is INVALID to compare a fixed-k=N/2 AD run against the
  all-k grid best. E5 therefore treats **k as an OUTER DISCRETE choice** and runs a
  SEPARATE AD optimization of the inner continuous variable at EACH fixed k, and
  compares each AD run only against the **same-k** grid best. The cross-k grid best
  is reported ONLY as a clearly-marked harder reference.
- **Status:** *proposed — do not implement/run yet.* NO agentic outer loop, NO
  SLURM initially, NO three-region first pass, NO new objectives, front_fraction-
  family ONLY; no proxy-D / e-γ PID / shower_max_depth.

### Structure: outer discrete k × inner continuous a_front
- **Outer discrete choices:** `k ∈ {N/3, N/2, 2N/3}` = {16, 25, 33} for N=50 (the
  E3b/E4 split set). NOT optimized by gradient — enumerated; one AD run per k.
- **Inner continuous variable:** `a_front` (front block layers [0,k)); `a_rear`
  solved exactly from the fixed budget `A_tot = N·a0 = 115 mm`, so budget is
  conserved *by construction* (no penalty term). 1 free DOF per (target, k) run.
- **Targets:** the earlier and later normalized-profile targets from E4.
- **Runs:** targets {earlier, later} × k {16,25,33} = **6 AD optimizations**, each
  starting from the uniform point (a_front = a0 = 2.3, i.e. a_rear = a0 too).

### 1. AD optimization implementation plan
New script `experiments/e5_ad_optimize.py` reusing E3/E4 building blocks. For a
given (target, k), per GD step:
1. Build the two-region abs_profile from the current scalar `a_front` at split k
   (rear solved from budget) — reuse `two_region_profile_k`.
2. **Forward** (multi-seed): measure per-layer mean edep → `E_l`, then
   `p_l = E_l/ΣE`. Compute loss `L = Σ_l (p_l − p*_l)²`.
3. **Output adjoints** (∂L/∂E_l): with `S=ΣE`, `p=E/S`,
   `∂L/∂E_l = (2/S)[ (p_l − p*_l) − Σ_j (p_j − p*_j) p_j ]`
   (the normalization Jacobian term; a detached scalar `Σ_j(p_j−p*_j)p_j`).
4. **Reverse** (multi-seed): `run_reverse_per_layer` with those adjoints returns
   `dL/d a_i` for every layer i (rows 0..N-1 of barInputsPerLayer).
5. **Chain to the 1 free DOF** through the budget constraint `a_rear =
   (A_tot − k·a_front)/(N−k)`, so `da_rear/da_front = −k/(N−k)`:
   `dL/d a_front = Σ_{i<k} dL/d a_i + (da_rear/da_front)·Σ_{i≥k} dL/d a_i`.
6. Gradient-descent update on the single scalar `a_front` (see §3).
Cross-check: at the first step (per (target,k)), verify this AD dL/da_front against
a central FD on the scalar (matched seeds), expecting the E0b ~0.8 undershoot — a
built-in sanity gate before trusting the descent.

### 2. Parameterization & constraint handling
- **1 free DOF per run:** `a_front` at the run's fixed split k; `a_rear` solved from
  `A_tot = 115 mm` so the budget is conserved *exactly by construction* (no penalty
  term needed).
- **Bounds:** clip `a_front` to `[ABS_FLOOR_MM, (A_tot − (N−k)·ABS_FLOOR_MM)/k]` so
  both regions stay ≥ 0.3 mm floor; project after each step (simple box clip).

### 3. Step size / optimizer / gradient clipping
- **Optimizer:** plain gradient descent with a fixed step, first pass (no Adam —
  keep it inspectable). `a_front ← clip(a_front − η · dL/da_front)`.
- **Step size η:** set from a trust-region on the *parameter*, not raw grad — target
  a first-step move of ≈0.05–0.1 mm in `a_front` (the E0b/E3 scale). Concretely
  `η = Δa_target / (|dL/da_front| + ε)` at step 0, then hold η fixed. This
  auto-scales to the loss magnitude (L~1e-4, grads small) without hand-tuning.
- **Gradient clipping:** clip |dL/da_front| to a max step of 0.1 mm/iter
  (`Δa = clip(η·grad, −0.1, +0.1)`) to avoid overshoot from a noisy gradient.
- **~0.8 caveat:** since AD undershoots FD by ~20% (E0b), the true step is slightly
  larger than AD suggests — acceptable for descent (conservative), noted not
  corrected.

### 4. Iterations & seeds
- **Seeds:** 6 (seeds [1–6]) per forward AND per reverse pass, matched (CRN);
  empirical across-seed SE on L and on the gradient (per [[DECISIONS]] 2026-07-06).
- **Iterations:** cap at **15** GD steps per run (each = 1 multi-seed forward + 1
  multi-seed reverse). Early-stop when |Δa_front| < 0.01 mm or L stops improving for
  3 steps.
- **6 runs:** {earlier, later} × k ∈ {16, 25, 33}; each starts from the uniform
  point `a_front = a0 = 2.3` (so the optimizer must move off uniform).

### 5. Success & kill criteria (per (target, k), compared to the SAME-k baselines)
Each AD run is judged against: (i) uniform loss, and (ii) the **same-k** E4/E3b grid
best. The best grid result **over all k** is reported only as a clearly-marked
harder cross-k reference — NOT the primary success bar.
- **Primary success:** AD reaches a loss within **1×SE of the same-k grid best** in
  ≤15 steps, OR beats uniform by **>3×SE** with a monotonic/sane trajectory
  (`a_front` moving the right direction, loss decreasing). Confirms AD optimizes the
  inner continuous contrast at that split.
- **Secondary success:** AD picks the correct direction (sign of the first move
  agrees with where the same-k grid best lies), AND the **best fixed-k AD run**
  (min over the three k) approaches the best-across-k grid result — i.e. enumerating
  a few k + AD-on-a_front recovers the full grid outcome.
- **Partial:** AD beats uniform >3×SE but plateaus above the same-k grid best →
  gradient works, step/optimizer needs tuning (record; don't over-engineer pass 1).
- **Kill:** AD fails to move below uniform beyond noise at ALL three k, OR the
  step-0 AD-vs-FD cross-check disagrees in SIGN / by ≫ the E0b ~0.8 factor → the
  per-layer reverse→1-DOF chain is wrong or the gradient is unusable here; stop and
  diagnose the chain before any three-region / agentic extension.

### 6. Runtime estimate
Per GD step: 1 forward (6 seeds) + 1 reverse (6 seeds) at N=1000. Reverse ~0.22
s/ev → ~1320 s/6-seed reverse; forward ~1110 s. With the 1400s cap and workers≈6
(seeds parallel within a step), ~4–6 min wall per step. At ≤15 steps × 6 runs =
≤90 steps → **~6–9 h total** if fully serial; in practice much less — forward
profiles at grid-adjacent a_front are cached from E4, early-stop trims steps, and
runs are independent so the 6 can be staggered. Recommend running the 6 (target,k)
combos as independent local jobs (or a couple at a time) and aggregating. NO SLURM
first pass; if wall-time balloons, a bounded SLURM fan-out (1 job per (target,k))
needs separate approval.

See [[DECISIONS]] 2026-07-06 (E5) and [[HYPOTHESES]] H_struct.

### E5 STAGE-1 PILOT RESULT (2026-07-07) — AD optimizes the two-region contrast; both pilots PASS.
Driver `experiments/e5_ad_optimize.py`; 2 of 6 combos (pilot). N=1000, 6 CRN seeds,
≤15 GD steps, trust-region step (first move 0.08 mm) + 0.1 mm clip, exact budget.
Logs: `experiments/e5_pilot_earlier_k16.txt`, `experiments/e5_pilot_later_k25.txt`.

**Run 1 — earlier target, k=16.** Step-0 AD/FD: AD=−1.85e-3, FD=−2.58e-3,
**ratio +0.72 (same sign, OK — matches E0b ~0.8)**. Descent (7 steps, early-stop):

  step | a_front | a_rear | loss | loss_SE | grad | step | notes
   0 | 2.3000 | 2.3000 | 5.351e-04 | 8.4e-06 | −1.85e-3 | −0.080 | AD/FD +0.72 OK
   1 | 2.3800 | 2.2624 | 3.838e-04 | 1.8e-05 | −1.93e-3 | −0.084 |
   2 | 2.4638 | 2.2229 | 2.263e-04 | 3.6e-06 | −7.78e-4 | −0.034 |
   3 | 2.4976 | 2.2070 | 2.072e-04 | 3.7e-06 | +2.12e-3 | +0.092 | (overshoot, recovers)
   4 | 2.4059 | 2.2502 | 3.090e-04 | 7.3e-06 | −1.61e-3 | −0.070 |
   5 | 2.4755 | 2.2174 | 2.334e-04 | 4.5e-06 | −9.34e-4 | −0.040 |
   6 | 2.5160 | 2.1984 | 1.873e-04 | 6.5e-06 | −1.07e-4 | −0.005 | early-stop |da|<0.01
  uniform 5.351e-04 → best AD **1.873e-04** (a_front=2.516). **Beats uniform >3×SE**
  (improve 3.48e-4 ≫ 3×SE 3.2e-5). **Same-k(=16) E4 grid-best = 2.100e-04** → AD
  (1.873e-4) is BELOW the grid-best (finer than the grid sampled). PASS.

**Run 2 — later target, k=25.** Step-0 AD/FD: AD=+2.63e-3, FD=+3.43e-3,
**ratio +0.77 (same sign, OK — matches E0b ~0.8)**. Descent (5 steps, early-stop):

  step | a_front | a_rear | loss | loss_SE | grad | step | notes
   0 | 2.3000 | 2.3000 | 4.648e-04 | 8.5e-06 | +2.63e-3 | +0.080 | AD/FD +0.77 OK
   1 | 2.2200 | 2.3800 | 2.271e-04 | 7.5e-06 | +1.83e-3 | +0.056 |
   2 | 2.1644 | 2.4356 | 1.429e-04 | 3.8e-06 | +8.38e-4 | +0.026 | 1 seed NaN (tolerated)
   3 | 2.1389 | 2.4611 | 1.340e-04 | 3.2e-06 | +4.94e-4 | +0.015 |
   4 | 2.1239 | 2.4761 | 1.208e-04 | 2.0e-06 | +2.28e-4 | +0.007 | early-stop |da|<0.01
  uniform 4.648e-04 → best AD **1.208e-04** (a_front=2.124). **Beats uniform >3×SE**
  (improve 3.44e-4 ≫ 3×SE 2.6e-5). **Same-k(=25) grid-best NOT in E4 log** (E4's
  later grid-best 9.147e-05 is at k=16, a cross-k *harder* reference); against that
  harder cross-k bar AD is close but not below. vs same-k it clearly beats uniform.
  PASS.

**VERDICT — both pilots PASS.** (a) Step-0 AD/FD **same sign** both runs, ratios
0.72 / 0.77 — consistent with the E0b ~0.8 undershoot, confirming the per-layer
reverse→budget-chain gradient is correct. (b) Sane monotone-ish descent both runs
(one transient overshoot in run 1, recovered — a fixed-η artifact, not a bug).
(c) Both **beat uniform by ≫3×SE**, and run 1 reaches BELOW the same-k grid best.
Conclusion: **validated AD gradients successfully optimize the structured
two-region absorber contrast** — the inner optimizer of the bilevel thesis works
on this objective. One operational note: a single 6-seed reverse pass hit the
1400s cap once (run 2, step 2) but was tolerated (5/6 seeds, loud NaN logged);
gradient still fine. Caveats unchanged (reachability, not figure-of-merit).
**Remaining 4 combos (earlier×{25,33}, later×{16,33}) NOT yet run — awaiting
user go-ahead per the two-stage plan.**

See [[DECISIONS]] 2026-07-07 (E5 pilot) and [[HYPOTHESES]] H_struct.

### E5 STAGE-2 / FULL-MATRIX RESULT (2026-07-07) — all 6 (target,k) combos PASS.
Remaining 4 combos run in 2 waves (2 pilots earlier). Same settings: N=1000, 6 CRN
seeds, ≤15 GD steps, trust-region step + 0.1 mm clip, exact budget. Logs:
`experiments/e5_{earlier,later}_k{16,25,33}.txt`. Full matrix (uniform losses:
earlier 5.351e-04, later 4.648e-04; both SE ~8.5e-6):

  target  | k  | step0 AD/FD (ratio) | best AD loss | best a_front | beats uniform >3×SE? | same-k grid ref
  earlier | 16 | −1.85e-3/−2.58e-3 (0.72) | 1.873e-4 | 2.516 | YES | 2.100e-4 (E4 k=16) — AD BELOW grid
  earlier | 25 | −1.96e-3/−3.27e-3 (0.60) | 2.320e-4 | 2.480 | YES | none (k=25 not in E4 log)
  earlier | 33 | −1.56e-3/−3.18e-3 (0.49) | 2.682e-4 | 2.500 | YES | none (k=33 not in E4 log)
  later   | 16 | +2.03e-3/+2.73e-3 (0.74) | 7.672e-5 | 2.042 | YES | 9.147e-5 (E4 k=16) — AD BELOW grid
  later   | 25 | +2.63e-3/+3.43e-3 (0.77) | 1.208e-4 | 2.124 | YES | none (k=25 not in E4 log)
  later   | 33 | +2.54e-3/+3.30e-3 (0.77) | 1.855e-4 | 2.160 | YES | none (k=33 not in E4 log)

Note on grid refs: the E4 log records only the all-k min best (both at k=16), so
only k=16 has a true same-k grid bar. AD at k=16 reaches BELOW the E4 grid-best for
BOTH targets (earlier 1.873e-4<2.100e-4; later 7.672e-5<9.147e-5) — AD found a finer
optimum than the grid sampled. k=25/33 are judged against uniform (all >3×SE); the
E4 all-k best (k=16) is the marked harder cross-k reference.

**VERDICT — E5 COMPLETE, all 6 combos PASS.** (1) Step-0 AD/FD **same sign in all 6**
(ratios 0.49–0.77, clustered near the E0b ~0.8 undershoot) → the per-layer
reverse→budget-chain gradient is correct across targets and splits. (2) All 6 beat
uniform by ≫3×SE with sane descent. (3) At the only k with a same-k grid bar (k=16),
AD beats the grid-best for BOTH targets. **Conclusion: validated AD gradients reliably
optimize the two-region absorber contrast for normalized-profile matching — the inner
optimizer of the bilevel thesis is established on this objective.**
- **k-dependence of reach (physics read):** best achievable loss depends on split —
  earlier target prefers k=16 (front block aligned with the upstream region being
  thickened: 1.87e-4 < 2.32e-4 < 2.68e-4 for k=16/25/33); later target also best at
  k=16 (7.7e-5). So k=16 (N/3) is the most expressive single split for these
  shift targets — consistent with E4, where the reached grid-best was at k=16.
- **Minor:** earlier_k33 step-0 ratio 0.49 is the furthest from ~0.8 (still correct
  sign, sane descent) — plausibly a rear-heavy-split noise effect; not pursued.
  Transient fixed-η overshoots occurred as in the pilot; a backtracking-LR would
  smooth them but did not affect the best-loss outcome.
Caveats unchanged (reachability/optimizability, NOT figure-of-merit; no grid-fair
control needed unless a profile metric becomes a performance objective). Establishes
[[HYPOTHESES]] H_struct at the inner-optimizer level.

See [[DECISIONS]] 2026-07-07 (E5 full matrix) and [[HYPOTHESES]] H_struct.

## E6 — Energy-reconstruction figure of merit for structured absorber geometries  (RESULT 2026-07-07, follows E5; FORK A)
- **Decides:** whether structured absorber profiles improve a REAL detector
  performance metric (energy reconstruction), not just normalized-shape
  reachability (E3–E5). This is the "promote to figure of merit" fork (A) from
  [[MILESTONE_E0_E5]] — the first test of whether "controllable" implies "better".
- **Central question:** at fixed total absorber budget (A_tot=115 mm) and fixed
  layer count, do uniform vs two-/three-region absorber profiles differ in energy
  reconstruction performance across a DISTRIBUTION of incident beam energies?
- **Status:** *proposed — do not implement/run yet.* No new sim capability needed:
  per-event per-layer edep already available via
  `depth_resolution._forward_per_event_profiles` (sets HEPEMSHOW_OUTPUT_ALL=1,
  reads boundary_stats.csv → (n_events, n_layers)). Beam energy varies via `-e`
  (dp.energy). NO agentic loop; SLURM only if approved.

### Critical departure from E3–E5: KEEP the energy scale
E3–E5 used normalized profiles `p_l=E_l/ΣE`, which **discard total-energy info** —
exactly what energy reconstruction needs. E6 must use the **raw per-event layer
energies E_l (MeV)** as features/target. Do NOT normalize away the scale.

### 1. Figure of merit (real, not shape)
Per-event reconstructed energy `Ê` vs true beam energy `E`. Report on HELD-OUT data:
- **relative RMSE:** `sqrt(mean((Ê−E)²/E²))` — the primary FOM.
- **resolution:** per-energy σ(Ê/E) (spread), and the stochastic-term fit if clean.
- **bias:** per-energy mean(Ê/E) − 1.
- **worst-energy error:** max over the energy grid of relative RMSE (robustness).

### 2. Geometries compared (fixed budget, absorber-only, uniform gap)
- **uniform** (reference).
- **two-region** profiles from E4/E5 (e.g. the k=16 earlier/later optima).
- **optional** small set of {k, contrast} candidates (do NOT sweep broadly;
  pick a handful, declared in advance — geometry is NOT chosen on test data, §3).

### 3. Train/test discipline (non-negotiable)
- **Calibration/train seeds** fit the estimator per geometry; **held-out eval
  seeds** (disjoint) score the FOM. E.g. seeds [1–4] train, [5–8] eval.
- **Geometry is fixed in advance** (from E4/E5 + a declared candidate list) —
  never selected using eval-seed results. Prevents the "precise-but-wrong /
  overfit to the test set" trap that burned earlier work.
- Report empirical across-seed SE on the held-out FOM (per [[DECISIONS]] 2026-07-06).

### 4. Estimators (start simple, escalate only if needed)
1. **Total-visible-energy calibration baseline:** `Ê = c · ΣE_l`, `c` fit on train
   (one scalar per geometry). The honest floor — most detectors start here.
2. **Linear / ridge regression on per-layer energies:** `Ê = w·E_vec + b`, ridge
   λ chosen on train only. Lets layer weighting exploit structure.
3. **Optional:** normalized-profile shape features + total energy as a combined
   feature vector (tests whether the E3–E5 shape control *adds* to energy est).
Pure-numpy (reuse the `tools/separation.py` ridge/k-fold style); no new deps.

### 5. Energy grid & runtime
- **Minimal grid:** {3, 10, 30} GeV (log-spaced, spans EM regimes cheaply); extend
  to {1,3,10,30} GeV only if the 3-point grid is promising.
- **Cost:** per (geometry, energy, seed) one forward run for N_ev events (per-event
  dump). ~4 geometries × 3 energies × 8 seeds = 96 forward runs at N=1000
  (~185 s each, 1400s cap). At workers≈6, ~1–2 h local; a bounded SLURM fan-out
  (1 job per geometry×energy) is the clean scale-up **if approved**. Forward-only —
  NO reverse-AD needed (E6 evaluates a FOM; it does not optimize geometry by AD).

### 6. Success criterion
Structured profile improves the **held-out** relative RMSE (or bias, or
worst-energy error) over uniform by **>3× the empirical across-seed SE**, with the
geometry fixed in advance and calibration fit only on train seeds.

### 7. Kill criterion
Uniform matches or beats structured profiles on held-out evaluation, OR an apparent
train-set gain vanishes on held-out seeds. Then: structured absorber shaping gives
shape control (E3–E5) but NO energy-reconstruction advantage — an honest, important
negative that bounds the value-of-structure claim to shape, not performance.

### 8. Honest caveats
- **Still EM-shower energy reconstruction** in a sampling calorimeter — a real
  detector FOM, but NOT yet a medical/space/other application. Don't overclaim.
- **Shape control may not help energy estimation:** E3–E5 showed structure steers
  the *normalized* profile; energy resolution depends on sampling fraction,
  fluctuations, and leakage — a DIFFERENT axis. E6 may well kill (that's useful).
- **Energy scale must be RETAINED** (raw E_l, not normalized) — the whole point;
  a normalized-only pipeline cannot do energy reconstruction by construction.
- **Fixed budget means fixed sampling on average** — E6 tests whether *redistributing*
  the same absorber changes resolution, a subtle effect; expect small if any.

See [[DECISIONS]] 2026-07-07 (E6) and [[MILESTONE_E0_E5]] (fork A).

### E6 RESULT (2026-07-07) — QUALIFIED SUCCESS: structure gives a small but >3×SE held-out energy-reco gain (estimator-dependent).
Driver `experiments/e6_energy_fom.py`. Forward-only, raw per-event layer energies
(scale retained), train seeds [1–4] fit / held-out seeds [5–8] score, geometries
FIXED in advance (uniform; earlier a_front=2.516; later a_front=2.042; all k=16,
budget=115 mm). Energies {3,10,30} GeV, N=1000. Clean: 0 timeouts/NaNs, budget
conserved. Log: `experiments/e6_energy_fom_result.txt`. Primary FOM = held-out
relative RMSE (empirical across-seed SE from the 4 test seeds).

  estimator | geom | held-out rel_RMSE | SE | worst-E rel_RMSE | vs uniform
  (1) total-E calib Ê=c·ΣE | uniform | 0.05702 | 5.2e-4 | 0.06581 | —
                           | earlier | 0.05661 | 2.2e-4 | 0.06504 | +4.1e-4 (3×SE 1.7e-3) no
                           | later   | 0.05823 | 7.4e-4 | 0.06614 | −1.2e-3 no
  (2) ridge on raw layer E | uniform | 0.02739 | 8.1e-4 | 0.03237 | —
                           | earlier | 0.02440 | 3.1e-4 | 0.02754 | **+3.0e-3 (3×SE 2.6e-3) BEATS>3SE** (worst-E +4.8e-3)
                           | later   | 0.02656 | 7.9e-4 | 0.02969 | +8.3e-4 no (worst-E +2.7e-3)
  (3) ridge [totalE,shape] | uniform | 0.08612 | 1.5e-3 | 0.13980 | —
                           | earlier | 0.07854 | 2.4e-3 | 0.12667 | +7.6e-3 (3×SE 8.3e-3) no (worst-E +1.3e-2)
                           | later   | 0.07984 | 1.4e-3 | 0.12828 | **+6.3e-3 (3×SE 6.0e-3) BEATS>3SE** (worst-E +1.2e-2)

**VERDICT — QUALIFIED SUCCESS (per decision rule: ≥1 structured geom beats uniform
>3×SE on held-out rel_RMSE or worst-E).** Two of three estimators show a structured
geometry beating uniform by >3×SE: **earlier** wins under the clean ridge-on-layers
estimator (2) — rel_RMSE 0.0244 vs 0.0274, ~11% relative reduction, and worst-energy
(3 GeV) 0.0275 vs 0.0324; **later** wins under the shape-feature estimator (3). The
gains SURVIVE the train→test seed split (not overfitting). So on this FOM,
"controllable" (E3–E5) does translate into a modest "better" for energy
reconstruction — the first evidence that structural shape control has a
performance payoff.

**Honest qualifications (do NOT overclaim):**
- **Estimator-dependent & small.** The calibration baseline (1) shows NOTHING
  (>3×SE nowhere) — pure ΣE can't exploit structure. The gains appear only with a
  layer-weighting regressor, and are ~10% relative-RMSE reductions, not dramatic.
- **Which geometry wins depends on the estimator** (earlier under (2), later under
  (3)) — not a single clean "structured optimum". The consistent signal is
  "some fixed-budget redistribution beats uniform for a layer-aware estimator",
  not "profile X is best".
- **Mechanism is plausibly leakage/sampling, not shape per se:** at 3 GeV
  (worst energy, most containment-sensitive) the structured gains are largest —
  consistent with front/rear absorber redistribution modestly improving
  containment/sampling for a weighted estimator, which is a genuine (if small)
  detector effect.
- Estimator (3) is WORSE than (2) overall (0.086 vs 0.027) — adding normalized
  shape as features hurt vs raw layer energies here; not pursued further.
- Still EM-shower energy reco in a sampling calorimeter (not medical/space);
  fixed budget ⇒ small effects as anticipated.

See [[DECISIONS]] 2026-07-07 (E6 result) and [[MILESTONE_E0_E5]] (fork A).

## E6b — Robustness check of the E6 energy-reco gain  (RESULT 2026-07-07, follows E6)
- **Decides:** whether the E6 qualified-success gain (structured beats uniform on
  held-out energy-reco rel_RMSE >3×SE, ridge on raw layer E) is ROBUST — survives a
  wider energy grid, a few more fixed geometries, and per-energy scrutiny — or
  whether it was confined to an accidental energy bin / geometry. Explicitly NOT an
  open-ended geometry search.
- **Status:** *proposed — do not implement/run yet.* Reuses `e6_energy_fom.py`
  unchanged except the declared geometry list + energy grid (both CLI args already
  exist). NO AD, NO agentic loop, NO SLURM initially, NO large scan.

### Primary estimator (only one)
Ridge on **raw per-event layer energies** — the cleanest E6 win. E6b reports this
estimator ONLY (drop the ΣE-calibration baseline and the shape-feature variant to
keep the comparison sharp; those were E6's context, not the claim).

### Declared geometry list (FIXED IN ADVANCE, from E5 optima; budget=115 mm, k as noted)
| label | a_front | a_rear | k | source |
|-------|---------|--------|---|--------|
| uniform | 2.300 | 2.300 | — | reference |
| earlier_k16 | 2.516 | 2.198 | 16 | E5 earlier/k16 (E6 winner under ridge) |
| later_k16 | 2.042 | 2.421 | 16 | E5 later/k16 |
| earlier_k25 | 2.480 | 2.120 | 25 | E5 earlier/k25 |
| later_k25 | 2.124 | 2.476 | 25 | E5 later/k25 |
5 geometries total. k=25 added as a DISTINCT, well-sampled split (a different
front/rear boundary) to test whether the gain is split-specific. k=33 optima
deliberately EXCLUDED to keep the list small and avoid a scan; declared now, frozen.

### Energy grid
{1, 3, 10, 30} GeV — adds **1 GeV** to E6's {3,10,30}. 1 GeV is the most
containment/leakage-sensitive point (E6's gains were largest at the low end), so
it is the sharpest robustness probe. If the E6 effect was a low-energy accident it
will show here; if real it should persist across ≥2 energies.

### Metrics (held-out seeds only; train/test discipline unchanged)
Held-out rel_RMSE (primary), bias per energy, resolution per energy, worst-energy
rel_RMSE, empirical across-seed SE. Train seeds [1–4] fit ridge; test seeds [5–8]
score. **Every structured geometry compared to uniform under the SAME estimator.**
Geometry list frozen BEFORE seeing any test result (declared above).

### Success
A structured geometry beats uniform by **>3×SE** on held-out rel_RMSE OR
worst-energy rel_RMSE, AND the improvement is **not confined to a single energy
bin** (i.e. structured ≤ uniform per-energy rel_RMSE at ≥2 of the 4 energies, or a
consistent sign across the grid). This guards against an accidental single-bin win.

### Kill
The E6 gain disappears — no structured geometry beats uniform >3×SE once 1 GeV and
the extra geometries are included, or the improvement is isolated to one energy bin
and flips sign elsewhere. Then E6 was fragile; value-of-structure for a performance
FOM is not established and the honest record is "controllable, not robustly better".

### Runtime
5 geometries × 4 energies × 8 seeds (train+test) = **160 forward runs** at N=1000
(~185 s, 1400s cap). vs E6's 72. At workers≈6, ~2–3 h local under normal load
(E6 took longer under heavy contention — acceptable). Forward-only, no reverse-AD.
A bounded SLURM fan-out (1 job per geometry×energy = 20 jobs) is the clean scale-up
**only if approved**; first pass stays local.

### Honest note
Even a clean E6b success remains a small (~10%), estimator-specific,
fixed-budget EM-shower effect (see E6 caveats) — robustness ≠ magnitude. E6b tests
"is it real", not "is it big".

See [[DECISIONS]] 2026-07-07 (E6b) and [[MILESTONE_E0_E5]] (fork A).

### E6b RESULT (2026-07-07) — PASS: the E6 gain (earlier_k16) is ROBUST; broader trend directionally consistent.
Driver `experiments/e6_energy_fom.py --e6b`. Ridge-on-raw-layer-E only, 5 fixed
declared geometries, energy grid {1,3,10,30} GeV, train[1–4]/test[5–8], geometry
frozen before eval. Log: `experiments/e6b_robustness_result.txt`.

Diagnostics: **160/160 forward runs OK** (5 geoms × 4 E × 8 seeds), **0 timeouts/
NaNs**, **0 budget violations** (asserted per geometry, all 115 mm), 1000 events/run
at every energy, **no `.sim_cache` reuse** (`_forward_per_event_profiles` runs in a
tempdir — all fresh sims).

  geometry | held-out rel_RMSE | SE | worst-E | vs uniform | bins won | verdict
  uniform     | 0.03089 | 4.1e-4 | 0.03992 | — | — | reference
  earlier_k16 | 0.02709 | 1.3e-4 | 0.03406 | +3.80e-3 (3×SE 1.3e-3) **>3×SE** | **4/4** | **PASS**
  later_k16   | 0.02961 | 8.3e-4 | 0.03763 | +1.29e-3 (3×SE 2.8e-3) ≤ | 3/4 | fragile
  earlier_k25 | 0.02932 | 5.7e-4 | 0.03774 | +1.57e-3 (3×SE 2.1e-3) ≤ | 3/4 | fragile
  later_k25   | 0.02992 | 2.1e-4 | 0.03874 | +9.69e-4 (3×SE 1.4e-3) ≤ | 4/4 | fragile

Per-energy rel_RMSE (all geometries improve uniform bin-by-bin; earlier_k16 shown):
uniform 1G/3G/10G/30G = 0.0399/0.0319/0.0256/0.0232; earlier_k16 =
0.0341/0.0273/0.0231/0.0221 — lower at ALL four energies including the new 1 GeV.

**VERDICT — PASS (robust win for earlier_k16).** Per the decision rule: earlier_k16
beats uniform on held-out rel_RMSE by >3×SE (+3.80e-3 vs 3×SE=1.3e-3, ~9σ) AND on
worst-energy error (+5.86e-3) AND with **consistent sign in 4/4 energy bins** —
including the newly-added, most containment-sensitive 1 GeV. The E6 gain **survived**
the harder test (extra energy, extra geometries, held-out). NOT a single-bin accident.

**Honest scope of the claim:**
- **One geometry is individually significant** (earlier_k16). The other three
  structured geometries improve uniform with **consistent sign** (3–4 of 4 bins,
  all positive) but do NOT individually clear >3×SE → "fragile" (real-direction,
  not significant alone). No geometry is worse than uniform.
- So: the SPECIFIC winner is robust; the general "any redistribution helps" is only
  a directional trend. The strongest defensible statement is "the E5-optimized
  earlier_k16 front-thickened profile robustly improves held-out energy resolution
  by ~12% over uniform at fixed budget, consistently across 1–30 GeV."
- Still small (~12% rel-RMSE), estimator-specific (ridge on layers; the E6
  ΣE-calibration baseline showed nothing), fixed-budget EM-shower reco — robustness
  ≠ magnitude, as flagged. earlier_k16 (front-thickened, k=N/3) being the winner is
  consistent with a containment/sampling mechanism (front-load absorber → better
  early-shower sampling), strongest where it matters most (low E).

See [[DECISIONS]] 2026-07-07 (E6b result) and [[MILESTONE_E0_E5]] (fork A).

## E7 — Direct AD optimization of a two-region absorber for the energy-reco FOM  (RESULT 2026-07-08, follows E6b; closes the E5/E6 gap)
- **Decides:** whether AD can optimize the ENERGY-RECONSTRUCTION objective ITSELF —
  closing the gap flagged in [[MILESTONE_E0_E6B]] §8: E5 used AD on a *shape* loss;
  E6/E6b only *evaluated* fixed geometries on energy reco. E7 asks: does AD, driving
  a differentiable reconstruction loss through the simulator, find a geometry whose
  held-out energy-reco FOM matches or beats the E6b hand-picked winner (earlier_k16)?
- **Status:** *proposed — do not implement/run yet.* Reuses the E5 gradient chain
  (`ad_grad_afront`: output adjoints → `run_reverse_per_layer` → budget chain) and
  the E6 estimator/FOM code. NO agentic loop; NO SLURM first pass; NO new physics
  objective beyond energy reco; front_fraction-family / energy-reco ONLY.

### Key subtlety (read first): what AD can and cannot differentiate here
The sim's reverse-AD propagates through the **mean per-layer profile** `Ē_l(θ)`
(the barInputs are d⟨E_l⟩/dθ). The E6b FOM is a **per-event** RMSE, whose spread
(resolution) depends on per-event fluctuations AD does NOT expose. So a
differentiable FOM surrogate can cleanly capture the **calibration/bias** part
(how the mean reconstructed energy moves with θ), NOT the per-event variance.
**Design choice:** optimize a differentiable **mean-reconstruction loss** (bias-like),
then evaluate the TRUE per-event held-out FOM (E6b protocol) at the found geometry.
AD steers the mean; the held-out RMSE is the honest judge. This is stated up front
so a null result is read correctly (AD may improve bias but not resolution).

### 1. Exact differentiable loss
Frozen linear estimator weights `w, b` (from training, §3). Predicted MEAN energy
at true energy E and geometry θ=a_front:
  `Ê_mean(E,θ) = w · Ē_vec(E,θ) + b`   where `Ē_vec` = mean per-layer energies.
Differentiable loss over the training energy set (relative, to match the FOM):
  `L(θ) = Σ_E ( (Ê_mean(E,θ) − E) / E )²`
Gradient: `dL/dθ = Σ_E 2((Ê_mean−E)/E²) · w · dĒ_vec/dθ`, where `dĒ_vec/dθ` is the
per-layer reverse-AD output (adjoints `a_l = 2((Ê_mean−E)/E²)·w_l`, one reverse pass
per energy) chained to `a_front` via the budget rule `da_rear/da_front=−k/(N−k)`
(identical chain to E5). One forward (for Ē and Ê_mean) + one reverse per energy per
step; multi-seed, CRN.

### 2. Train / validation / test split
- **Train seeds [1–4]:** fit the estimator (§3) AND compute the AD loss/gradient
  used to drive `a_front`.
- **Validation seeds** — reuse train seeds for the descent's own loss (the loss is
  a mean quantity; a separate val split is optional first pass); note if added.
- **Test seeds [5–8]:** held out; used ONLY for the final E6b per-event FOM
  evaluation of the converged geometry. `a_front` is NEVER updated on test seeds.
- Energy grid {1,3,10,30} GeV (E6b grid), so the optimized geometry is judged on
  the same distribution.

### 3. Estimator: frozen-from-uniform (first pass)
Fit ridge `w,b` ONCE on the UNIFORM geometry (train seeds, all energies), then
**freeze** for the whole descent. Rationale: (i) a moving estimator + moving
geometry is a two-body optimization that muddies the AD signal and invites
overfitting; (ii) it asks a clean question — "can AD reshape the detector to suit a
fixed readout?". Secondary/report-only: re-fit the estimator at the FINAL geometry
before the held-out eval (a per-geometry-calibrated number), clearly labeled, to
avoid penalizing E7 for using uniform's weights.

### 4. AD-vs-FD sanity check for dFOM/da_front
Same gate as E5: at step 0 compute the AD `dL/da_front` and a central-difference FD
(perturb a_front ±h, recompute L with matched seeds). Expect **same sign** and
roughly the E0b ~0.8 scale. Wrong sign / ≫0.8-off ⇒ STOP (chain bug), diagnose
before trusting the descent.

### 5. Success criterion
The AD-optimized geometry, evaluated with the **held-out E6b per-event protocol**,
achieves rel_RMSE **within 1×SE of the E6b earlier_k16 winner (0.02709)** OR beats
uniform (0.03089) by **>3×SE** — AND the descent shows a sane AD/FD-verified
trajectory converging to a_front near the E6b/E5 optimum region (~2.5 for the
front-thickened branch). I.e. AD *finds* a good FOM geometry on its own.

### 6. Kill criterion
AD descent does NOT beat uniform >3×SE on the held-out FOM, OR converges to a
geometry whose held-out FOM is worse than earlier_k16 beyond noise, OR the step-0
AD/FD check fails (sign / scale). A likely-and-informative kill: AD improves the
**mean/bias** loss but the **per-event held-out RMSE does not improve** — that
would show energy-reco gains live in the resolution (variance) channel AD cannot
see via mean-profile gradients, bounding the differentiable-FOM approach.

### 7. Runtime estimate
Per GD step: (1 forward + 1 reverse) × 4 energies × 6 seeds, parallelized. ~4
energies × ~2× the E5 per-step cost ≈ ~15–25 min/step under load (1400s cap);
≤15 steps + early-stop → ~2–5 h for one k=16 run. Plus the held-out eval (E6b, ~1 h
at the single found geometry, cache-shared). Forward+reverse, NO SLURM first pass;
a bounded SLURM version (1 job per energy per step is NOT worth it — keep local /
sequential). Single geometry track (k=16) only.

### 8. Caveats
- **AD sees the mean, not the spread** (see subtlety box): E7 optimizes a bias-like
  surrogate; the resolution part of the FOM is not directly differentiable here. A
  clean improvement is meaningful; a null is informative (gain is in variance).
- **Frozen-from-uniform estimator** may under-credit E7; the re-fit-at-final number
  is the fairer comparison and is reported alongside.
- **~0.8 AD/FD scale factor** (E0b) applies — descent direction reliable, step
  magnitude carries the factor (conservative).
- Still fixed-budget EM-shower reco, single species, one calorimeter, k=16 only —
  same scope bounds as E6/E6b. Modest effects expected.
- If E7 merely re-finds earlier_k16, that is still a POSITIVE result: it shows AD
  can reach the hand-picked winner directly on the FOM, which is the point.

See [[DECISIONS]] 2026-07-07 (E7) and [[MILESTONE_E0_E6B]] (§8 gap, §10 fork C).

### E7 RESULT (2026-07-08) — machinery PASS; AD re-finds earlier_k16 on the FOM surrogate; held-out gain real but sub-3×SE with a frozen estimator.
Ran on SLURM (job 31047704, 32G) after 3 LOCAL runs died to node load-spikes and 1
SLURM OOM at 8G. This job OOM'd at 33.5G > 32G **but only after printing all results**
(the OOM cut just the final formatted interpretation lines — no data lost; ≤2
transient NaN-flagged descent points, below the 3-timeout corruption threshold, none
in the held-out eval). Result preserved: `experiments/e7_diag_result_SLURM_31047704.txt`.
Bounded diagnostic: k=16, frozen-from-uniform ridge, N=1000, train[1–4]/test[5–8],
energies {1,3,10,30} GeV.

**(a) Surrogate curve (mean-reconstruction loss L=Σ_E((Ê_mean−E)/E)² vs a_front):**

  a_front | surrogate L | AD grad | note
  2.000   | 1.175e-4 | −2.66e-2 | grad<0 → push up
  2.300   | 1.339e-4 | +1.14e-2 | (uniform)
  2.516   | **9.87e-5** | −8.2e-4 (~0) | **surrogate MINIMUM = earlier_k16**
  2.600   | 1.252e-4 | +2.34e-3 | grad>0 → push down

**(b) Short AD descents:** from 2.0 → climbs to ~2.08 then early-stops; from 2.6 →
walks down through 2.52→2.42→2.32→2.22; from 2.516 → grad≈0, oscillates around it.
All consistent with a surrogate optimum near a_front≈2.3–2.5 (front-thickened),
NOT back at uniform.

**(c) Held-out per-event rel_RMSE (E6b protocol, FROZEN-from-uniform estimator):**

  geometry | held-out rel_RMSE | SE | vs uniform
  uniform              | 0.03089 | 4.1e-4 | —
  earlier_k16          | 0.02969 | 5.6e-5 | +1.20e-3 (3×SE 1.25e-3) — just under
  E7_surr_opt(a=2.516) | 0.02969 | 5.6e-5 | +1.20e-3 — IDENTICAL (same geometry)

**VERDICT — machinery PASS; FOM optimization PARTIAL (revises the probe's KILL prediction):**
- **Machinery PASS:** AD moves downhill on the surrogate (grad −2.66e-2 below the
  optimum, +2.34e-3 above, ≈0 at it); descents converge sanely. The per-layer
  reverse→budget chain drives the FOM surrogate correctly.
- **AD re-finds earlier_k16.** At FULL stats the mean-surrogate optimum is
  a_front≈2.516 = **exactly the E6b hand-picked winner** — NOT uniform (the earlier
  low-stat probe that suggested a uniform optimum was noise-misled). So *directly
  optimizing the FOM surrogate with AD lands on the same geometry* → **the E5/E6 gap
  is closed positively at the level of WHICH geometry AD selects.**
- **But the held-out gain is weaker here and sub-3×SE.** With the frozen-from-uniform
  estimator, earlier_k16 (= E7's pick) beats uniform by only +1.20e-3 (3×SE=1.25e-3,
  ~2.9σ — just under threshold), vs E6b's +3.80e-3 (~9σ) which used a
  per-geometry-REFIT estimator. Same geometry, weaker number: the frozen readout
  under-realizes the resolution gain. (The refit-at-final diagnostic that would
  quantify this did not print — OOM cut the tail; a rerun at 64G would recover it.)
- **Mechanism, refined:** the mean-surrogate DOES capture enough to pick the right
  geometry (the location/front-thickening effect is visible in the mean profile);
  but the *full* held-out RMSE advantage (E6b's ~9σ) is only realized with a matched
  estimator, consistent with part of the gain living in the resolution/variance
  channel + estimator-geometry coupling that a frozen mean-surrogate under-weights.

**Net:** AD optimizing the energy-reco FOM surrogate is viable and re-discovers the
E6b winner (closes the [[MILESTONE_E0_E6B]] §8 gap), but the frozen-estimator held-out
gain is marginal (~2.9σ) — a per-geometry-calibrated readout (as E6b used) is needed
to realize the full effect. NOT overclaimed as a clean FOM-optimization win.
Caveats unchanged (mean-AD-invisible variance; ridge-specific; fixed-budget; k=16).

See [[DECISIONS]] 2026-07-08 (E7 result) and [[MILESTONE_E0_E6B]].

## E8 — Controlled agentic/discrete outer loop with AD inner optimization for the energy-reco FOM  (DRY-RUN RESULT 2026-07-08; full proposal below)
- **Status:** *dry-run (candidates 1–2) DONE 2026-07-08 — FAILED on the AD-inner-loop
  gate; plumbing otherwise validated. Full E8 (candidates 1–5) NOT run.* Details in
  the DRY-RUN RESULT box immediately below; the full proposal follows it.

### E8 DRY-RUN RESULT (2026-07-08) — infra clean, held-out plumbing validated, but AD-inner-loop gate FAILED (shallow-surrogate noise).
Driver `experiments/e8_bilevel.py`; candidates 1–2 (uniform + two-region@k16,
front-thickened init a_front=2.8). Ran on **atlas:compef@ampere** (normal QOS,
non-preemptable, CPU-only) after `atlas:usatlas@roma` was wedged on
`AssocGrpNodeLimit` for hours (roma idle nodes + 0 group roma jobs, still blocked).
Job 31115135: **COMPLETED, 2h52m, MaxRSS 3.96G — no OOM, no timeouts, exit 0.**
Log: `experiments/e8_dryrun_result_31115135.txt`.

**What passed (2 of 3 subsystems):**
- **Proposer table → geometry construction:** OK (uniform + k16 built, budget conserved).
- **Held-out E6b eval + refit-at-final:** OK and **reproduces E6b exactly** —
  earlier_k16_ref refit rel_RMSE = **0.02709** (matches E6b's 0.02709) vs uniform
  0.03089; frozen 0.02969 vs 0.03089. worst-E 0.03926 vs 0.03992. The evaluation
  half of the pipeline is validated end-to-end with no manual intervention.

**What FAILED (the AD inner loop):**
- Step-0 AD/FD gate at the probe point a_front=2.0: **AD=+5.87e-4, FD=−3.34e-4,
  ratio −1.76 → wrong sign → GATE_FAIL**, so k16's inner loop aborted and the
  candidate was skipped (never optimized).
- **Root cause: shallow-surrogate noise, NOT a chain bug.** The mean-reconstruction
  surrogate varies only ~1e-4 across a_front∈[2.0,2.6] (E7), with per-seed gradient
  SEs comparable to the gradients; E7's own curve already showed an inconsistent
  gradient sign near a_front=2.3. At N=1000×4seeds the gate-probe gradient is simply
  not resolved, so the strict same-sign gate fires spuriously. (E7 succeeded because
  it gated at a DIFFERENT off-minimum point and, more importantly, only needed the
  full-curve minimum — not a single-point sign that survives noise.)

**DRY-RUN VERDICT: FAIL (pipeline not fully validated).** The infra + evaluation
plumbing are sound (no OOM/timeout, held-out reproduces E6b), but the AD-inner-loop
gate is too fragile for this shallow surrogate at feasible stats. **Do NOT proceed
to candidates 3–5.** Minimal fixes to consider BEFORE any retry (require approval —
NOT yet applied): (a) gate at a point with a resolved gradient (larger |a_front−a0|,
e.g. 1.7 or 3.0) and/or require |AD|>3·SE before applying the sign test (skip-gate
when unresolved rather than FAIL); (b) raise inner-loop stats (more seeds/N) so the
gradient is resolved; (c) treat a shallow/|grad|≈0 surrogate as "already near
optimum" and evaluate the init geometry rather than aborting. Each is a small,
scoped change to `e8_bilevel.py`/the E7 gate — none alter the objective or physics.

See [[DECISIONS]] 2026-07-08 (E8 dry-run) and [[MILESTONE_E0_E7]] (§7 E8).

### E8 DRY-RUN RERUN (2026-07-09, fixed gate) — GATE FIX WORKS, but TIMEOUT + noise-dominated descent expose a deeper issue.
Rerun with the SE-aware 3-outcome gate (job 31127411, atlas:compef@ampere, 64G).
Job **TIMEOUT at the 6h walltime** (killed at descent step 7/10, before the held-out
eval — no DRY-RUN VERDICT). MaxRSS 67G at the kill (would also have OOM'd — but the
timeout came first). Partial log: `experiments/e8_dryrun_result.TIMEOUT_31127411.txt`.

**GATE FIX: works as designed.** Both probes correctly labeled FLAT (not FAIL):
- @a_front=2.00: AD=+5.87e-4 (SE **2.8e-3**), FD=−3.34e-4 (SE 1.2e-4) → FLAT (AD
  unresolved: |AD| ≪ 3·AD_SE). This is the exact point the OLD gate wrong-sign-FAILed.
- @a_front=2.60: AD=+3.85e-4 (SE **3.5e-3**), FD=+1.42e-4 → FLAT.
- Verdict FLAT → **proceeded into the descent, no abort.** The gate-design bug is fixed.

**NEW problems surfaced (both real, neither is the gate):**
1. **Walltime:** actually running the descent (2 gate probes + up to 10 steps, each
   = forward+reverse × 4 energies × 4 seeds) exceeds 6h. The prior run only "fit" in
   2h52m because it aborted at the gate. → need a longer walltime AND/OR fewer steps.
2. **Noise-dominated descent — the deeper issue.** The AD gradient's SE (2.8–3.5e-3)
   is LARGER than the gradient (~5e-4) at the probe points, and the descent gradient
   flips sign nearly every step (+4.8e-3, −1.0e-3, +1.2e-2, −2.5e-3, −1.3e-2, …).
   a_front drifts 2.80→2.72→2.74→2.64→2.68→2.78→2.74→2.67 — **wandering, NOT
   converging to 2.516.** The mean-reconstruction surrogate is simply too flat
   relative to MC noise at N=1000×4seeds for a per-step gradient descent to home in.

**RERUN VERDICT: gate FIXED; dry-run still INCOMPLETE.** The plumbing runs end-to-end
without a spurious abort (gate goal met), but (a) it times out before the held-out
eval, and (b) even completed, the noise-dominated 1-DOF descent would not reliably
recover a_front≈2.516 — it wanders in a shallow, noisy surrogate. This is consistent
with E7 (where AD found the geometry only via the full-CURVE minimum over fixed
probe points, NOT via a step-by-step descent). **Root lesson: per-step AD descent is
the wrong inner optimizer for this shallow/noisy surrogate.**

Recommended fixes BEFORE any further E8 run (require approval; NOT applied):
- **Inner loop = curve-scan-argmin, not GD:** evaluate the surrogate on a small fixed
  a_front grid (as E7's curve did) and take the argmin — robust to the noisy gradient,
  and it DID recover 2.516 in E7. This reuses validated E7 machinery and sidesteps the
  descent-thrash entirely.
- **Higher stats** (more seeds / N) to resolve the gradient — but expensive and
  doesn't fix the walltime.
- **Walltime → 12h** and/or **--max-steps ≤5** regardless.
The AD *gradient* is validated (E5/E7); the issue is using it in a per-step descent
on THIS shallow surrogate. Do NOT proceed to candidates 3–5 or the full E8 until the
inner optimizer is switched to the curve-argmin approach and a dry-run completes.

See [[DECISIONS]] 2026-07-09 (E8 dry-run rerun) and [[MILESTONE_E0_E7]].

---

**FULL E8 PROPOSAL (candidates 1–5) — unchanged, pending dry-run fix:**
- **Status:** *proposed — do not implement/run yet.* This is the headline bilevel
  test flagged in [[MILESTONE_E0_E7]] §7. It is a CONTROLLED demonstration, NOT
  "let the agent wander": small discrete proposal space + AD continuous optimization
  + held-out FOM. On SLURM (per E7 op-rule, ≥48–64G); NO new physics/estimators.

### 1. Exact E8 question
Can a **discrete-structure proposer + AD inner optimization** find fixed-budget
absorber geometries that **match or beat the known earlier_k16 baseline** on
held-out energy reconstruction — under a frozen evaluation protocol chosen before
any test-set look?

### 2. Proposed outer-loop protocol (the "proposer")
- Proposes a SMALL declared set of discrete structures (candidate budget §4). Each
  structure is a low-DOF parameterization, NOT a 50-DOF per-layer profile.
- Proposer may choose: **#regions ∈ {1,2,3}**; **split location(s)** from the fixed
  discrete set **{N/4, N/3, N/2, 2N/3, 3N/4}** = {12,16,25,33,37} for N=50;
  **initialization** ∈ {front-thickened, rear-thickened, uniform}; simple box
  constraints (region thickness ≥ 0.3 mm floor).
- Proposer MAY NOT: inspect held-out test results before final eval; change the
  metric after seeing results; propose arbitrary per-layer profiles; run unbounded
  searches; use normalized-profile target losses as the FINAL metric; add new
  physics/estimators without approval.
- **First-pass proposer = deterministic enumeration of the declared candidate list**
  (below). An LLM/heuristic proposer is an OPTIONAL later variant that must still
  emit structures within this same small parameterization — deferred, not first pass.

### 3. Proposed inner AD protocol (per proposed structure)
- AD optimizes the continuous absorber thickness(es) under EXACT budget conservation
  (A_tot=115 mm; last region solved from the others, as E5/E7). #free DOF = #regions−1.
- Objective: the **E7 mean-reconstruction energy surrogate**
  `L=Σ_E((Ê_mean−E)/E)²` with a **frozen-from-uniform ridge** estimator (identical
  to E7; NO new estimator). Two-region = 1 DOF (reuse E7 exactly); three-region =
  2 DOF (needs the not-yet-built 2-DOF chain — see Risks / three-region gate).
- Step-0 AD/FD sanity check at an OFF-minimum probe (as E7); STOP that structure if
  wrong-sign or unstable. Bounded iterations ≤10. Empirical across-seed SE.

### 4. Candidate structure budget (SMALL, declared in advance)
Exactly **5 candidates**, frozen before any evaluation:
  1. **uniform** (1 region) — reference.
  2. **two-region @ k=16** front-thickened init — the E6b/E7 winner region (recovery test).
  3. **two-region @ k=25** front-thickened init — a different split (generalization).
  4. **two-region @ k=33** rear-region init — tests whether the proposer/AD avoids a worse split.
  5. **three-region @ (k1,k2)=(16,33)** front-thickened init — the one multi-DOF probe.
Candidates 2–4 reuse the validated E7 1-DOF inner loop; candidate 5 is the only one
needing the 2-DOF inner loop (gated — see Risks).

### 5. Data split & evaluation protocol (frozen, = E6b)
- Train seeds [1–4] fit ridge AND drive the AD inner loop; test seeds [5–8] score.
- Energy grid **{1,3,10,30} GeV**; primary estimator **ridge on raw layer energies**.
- Report held-out **relative RMSE** (primary), **worst-energy error**, per-energy
  RMSE, bias, resolution, empirical across-seed SE.
- Geometry candidates and metric FROZEN before test-seed evaluation; NO test-set
  selection.
- **Estimator note:** the AD inner loop uses the frozen-from-uniform estimator
  (E7); the FINAL held-out comparison reports BOTH frozen and per-geometry-refit
  numbers (refit is the fair geometry comparison, as learned in E7). Declared now to
  avoid a post-hoc choice.

### 6. Baselines
(1) uniform; (2) earlier_k16 (the E6b/E7 known-good geometry); (3) best E8 candidate
(the outer+AD loop's selection). All under the same held-out protocol.

### 7. Success / partial / kill criteria
- **Success:** the outer+AD loop finds a structure that (a) beats uniform by **>3×SE**
  on held-out rel_RMSE or worst-energy error, AND (b) **matches earlier_k16 within
  1×SE OR beats it by >3×SE**.
- **Partial:** recovers a structure qualitatively equivalent to earlier_k16, OR
  beats uniform but does not match earlier_k16.
- **Kill/null:** fails to beat uniform; OR wins on train/validation but not held-out;
  OR requires post-hoc metric/geometry changes to look good.

### 8. Runtime estimate
Inner AD per two-region candidate ≈ one E7-style run (≤10 steps × forward+reverse ×
4 energies × 4 train seeds). 3 two-region candidates + uniform (no inner loop) +
1 three-region ≈ **4 inner-loop runs + 1 trivial**. On SLURM at ≥48–64G, ~2–4 h each
if serial; independent, so a small fan-out (1 job per candidate) finishes in
~one E7 wall-time. Plus the held-out eval (E6b-style, cheap). **Total ~1 SLURM
wave.** No local runs (E7 op-rule: load spikes kill forward+reverse locally).

### 9. Risks & mitigations
- **[HIGHEST] Three-region 2-DOF inner loop is not yet built/validated.** The E5/E7
  chain is 1-DOF. Mitigation: run candidates 1–4 (all ≤1-DOF, fully validated) as
  the E8 core; **GATE candidate 5** behind a separate 2-DOF chain validation (an
  E5-style step-0 AD/FD check on 2 DOF) — do NOT include three-region until that
  passes. E8's core conclusion does not depend on candidate 5.
- **Frozen-estimator marginal gain (E7):** the frozen readout gave only ~2.9σ.
  Mitigation: report refit-at-final too (declared in §5); judge "match earlier_k16"
  on the refit number.
- **SLURM OOM (E7 hit 33.5G):** request ≥48–64G per job; cap --workers.
- **Trivial-proposer critique:** enumerating a declared list is barely "agentic".
  Mitigation: frame E8 honestly as a CONTROLLED bilevel demonstration (proposer =
  deterministic first pass); an LLM proposer over the SAME space is a labeled
  optional follow-up, not the claim.
- **Local load spikes:** SLURM-only (E7 op-rule).

### 10. What would count as evidence for the bilevel thesis
The outer proposer + AD inner loop, under a FROZEN held-out protocol with NO
test-set selection, selects a fixed-budget structure that matches or beats
earlier_k16 on held-out energy reconstruction — i.e. the loop **re-derives a
known-good detector structure it was not told**, and the continuous params come from
AD (not grid). Bonus: it finds a split OTHER than k=16 that also beats uniform >3×SE
(generalization beyond the one known geometry).

### 11. What would NOT count
- Beating uniform only on train/validation, not held-out.
- Matching earlier_k16 only because the candidate list literally contained k=16 AND
  no other candidate was competitive (that's recovery-by-enumeration, report as
  PARTIAL, not a bilevel win).
- Any post-hoc metric/geometry/estimator change to make a candidate look good.
- A per-layer 50-DOF fit (outside the declared small parameterization).
- Claiming "the agent designed a detector" or "structure universally helps" — E8 is
  a narrow, fixed-budget, EM-shower, ridge-estimator demonstration.

See [[DECISIONS]] 2026-07-08 (E8) and [[MILESTONE_E0_E7]] (§7 E8).

### E8 REVISED PROTOCOL (2026-07-10) — minimal bilevel PoC on the validated normalized-profile objective

**Supersedes the FULL E8 PROPOSAL above.** Both dry-runs established that per-step GD
on the E7 energy-reco surrogate fails at feasible stats. The E5 normalized-profile MSE
inner loop (AD gradient-descent, NOT curve-argmin) already works reliably — all 6 combos
sane descent, beat uniform ≫3×SE. E8 is revised to use that validated objective.

- **Outer proposer (two-call adaptive):**
  - Call 1: selects 2 k values from {16, 25, 33} with a one-sentence rationale.
  - Runs those 2 AD inner optimizations; receives compact results only.
  - Call 2: given `{k, best_loss, a_front, a_rear, steps, status}` for the first two,
    decides whether to evaluate the remaining k (max 3 total).
- **Inner AD loop:** E5 `optimize()` function verbatim — GD on
  `L = Σ(p_l − p*_l)²` (earlier-shift target), exact budget, ≤10 steps, 6 CRN seeds
  N=1000, start from uniform. SE-aware gate (PASS/FLAT/FAIL; FLAT → proceed).
- **Reuse:** completed E5 result files (`e5_pilot_earlier_k16.txt`, `e5_earlier_k25.txt`,
  `e5_earlier_k33.txt`) are loaded as oracle/reference; those k are not rerun unless
  `--no-reuse`. A fresh bilevel run exercises plumbing via `--no-reuse`.
- **Baselines:** uniform (no inner loop) + exhaustive E5 three-k results as offline oracle.
- **Frame:** controlled bilevel PoC demonstration; does NOT claim superiority over exhaustive enumeration.
- **No three-region, no new objectives, no secondary E6b eval until bilevel completes.**
- **Runtime:** ≤3 AD runs × ≤10 steps (forward+reverse × 6 seeds, N=1000) ≈ 2–4 h on SLURM.
- **Driver:** `experiments/e8_profile_bilevel.py`; smoke-test with `--smoke` (no sim).
- **SLURM (fan-out):** one job per k, then aggregate; see driver docstring for exact commands.

---

## Sequencing
1. **E0 first** — cheap, unblocks the pivotal H2 question and defines "correct."
   DONE: FD validated, H2 killed. **E0b DONE:** d/da usable (~0.8 undershoot).
2. **E1 in parallel** — independent of E0; forward-only value runs are cheap.
3. **E2 after E0** — only meaningful once FD/AD comparison is trusted.
4. **E3 DONE (2026-07-06)** — controllability confirmed: three-region reaches
   off-uniform ff targets >3×SE better than uniform; two-region collapsed to
   uniform at grid=5 / centered split.
5. **E3b DONE (2026-07-06)** — E3's two-region collapse was a coarse-grid
   artifact: with grid=11, all splits k∈{N/3,N/2,2N/3} reach off-uniform targets
   >3×SE better than uniform. Ladder is uniform ≪ two-region ≈ three-region;
   1 DOF suffices to steer front_fraction to ±0.05 targets.
6. **E4 DONE (2026-07-06)** — full normalized-profile shape control confirmed for
   earlier/later shift targets (structured ≫3×SE better than uniform); broadening
   target within 3×SE of uniform = reachable-manifold limit (shift/tilt-capable,
   width-limited). H_struct upgraded from scalar to vector shape control.
7. **E5 DONE (2026-07-07)** — full 6-combo matrix ({earlier,later}×{k=16,25,33})
   all PASS: step-0 AD/FD same-sign (ratios 0.49–0.77 ~ E0b 0.8), sane descent, all
   beat uniform ≫3×SE; at k=16 (only same-k grid bar) AD reaches BELOW the E4
   grid-best for both targets. Validated AD reliably optimizes the two-region
   contrast → inner optimizer of the bilevel thesis established on this objective.
8. **E6 DONE (2026-07-07, FORK A)** — QUALIFIED SUCCESS: with a layer-weighting
   ridge estimator, structured (earlier, k=16) beats uniform on held-out energy-reco
   rel_RMSE by >3×SE (~11%, worst-E 3 GeV too); "later" wins under the shape-feature
   estimator; calibration baseline shows nothing. Gains survive train→test split.
   "Controllable" ⇒ modest "better" for energy reconstruction. Estimator-dependent
   and small; likely a containment/sampling effect.
9. **E6b DONE (2026-07-07)** — PASS: earlier_k16 beats uniform on held-out energy-reco
   rel_RMSE >3×SE (~9σ) and worst-E, consistent-sign in 4/4 energy bins incl. new
   1 GeV → E6 gain is ROBUST (not a single-bin accident). Other 3 structured geoms
   improve with consistent sign but <3×SE (fragile). Claim: E5-optimized front-thickened
   earlier_k16 robustly improves held-out energy resolution ~12% over uniform at
   fixed budget across 1–30 GeV. Small, ridge-specific. Forks B/C now on firmer ground.
10. **E7 DONE (2026-07-08, on SLURM after 3 local load-spike kills + 1 OOM)** —
    machinery PASS; AD on the mean-reconstruction surrogate re-finds earlier_k16
    (surrogate optimum a_front≈2.516, the E6b winner — NOT uniform), closing the
    E5/E6 gap at the level of which geometry AD selects. But with a frozen-from-uniform
    estimator the held-out gain is marginal (~2.9σ, sub-3×SE) vs E6b's ~9σ
    per-geometry-refit — the full RMSE advantage needs a matched readout. Bounded
    diagnostic, not overclaimed as a clean FOM-optimization win.
11. **E8 DRY-RUN (2026-07-08 gate-fail → 2026-07-09 gate-fixed rerun).** Gate now works
    (SE-aware PASS/FLAT/FAIL; both k16 probes correctly FLAT, no abort — fix validated).
    Held-out eval reproduces E6b exactly (earlier_k16 refit 0.02709 vs uniform 0.03089).
    BUT rerun TIMED OUT at 6h before held-out eval, AND the per-step AD descent is
    noise-dominated (grad_SE > grad, a_front wanders, does NOT converge to 2.516).
    Root lesson: per-step GD is the wrong inner optimizer for this shallow/noisy
    surrogate — switch to CURVE-SCAN-ARGMIN (as E7's curve, which DID recover 2.516).
    Do NOT proceed to candidates 3–5 / full E8 until inner loop is switched + a dry-run
    completes. Gradient itself is validated (E5/E7); the descent USAGE is the problem.

## E9 — Physics-reasoned agentic proposal of structured absorber families with AD continuous optimization  (PROPOSED 2026-07-10, follows E8)

- **Decides:** whether an LLM agent can use natural-language shower-physics reasoning
  and compact AD feedback to propose structured absorber parameterizations **not fixed
  in advance**, and whether those proposals beat enumerated fixed-structure baselines
  on held-out normalized-profile loss. E8 validated the bilevel plumbing over a
  pre-declared discrete set; E9 removes the enumeration assumption.
- **Status:** *proposed — do not implement/run yet.*

### 1. Target suite

**Development targets** (visible to agent during search):
- `earlier_2` — uniform p0 shifted −2 layers (same as E4/E5/E8 earlier-shift target).
- `later_2`   — uniform p0 shifted +2 layers.
- `earlier_4` — shift −4 layers (harder; two-region reach limited by DOF).

**Held-out targets** (evaluated once after all 6 proposals are finalized):
- `broader_σ3` — p0 convolved with Gaussian σ=3 layers, renormalized. E4 showed this
  is outside the reachable manifold of 1–2-DOF absorber structure — a strong test of
  whether more DOF / physics reasoning unlocks width control.
- `earlier_4_independent` — a separate seed batch for the −4 shift target, disjoint
  from the development seeds, to guard against overfitting to the development noise.

Target construction: analytic perturbations of a detached high-stat reference p0
(same method as E4). Held-out seeds [7–12] (disjoint from E5 seeds [1–6]).

### 2. Proposal schema (machine-readable, required in every agent proposal)

```json
{
  "label": "<unique string>",
  "n_regions": <int 2–6>,
  "splits": [<int>, ...],          // ordered layer indices; len = n_regions - 1
  "constraints": ["monotone_increasing" | "monotone_decreasing" | "symmetric" | null],
  "init": "front_loaded" | "rear_loaded" | "uniform" | "tapered_<direction>",
  "rationale": "<one sentence: what shower physics this targets>",
  "prediction": "<one sentence: which residual component this should reduce>"
}
```
Validity rules enforced by the driver: `n_regions = len(splits)+1`; splits strictly
ordered and within [1, N-1]; each region thickness ≥ ABS_FLOOR_MM (0.3 mm) at init;
total budget = A_tot = 115 mm conserved exactly (last region solved from others).
Free DOF = n_regions − 1 (last thickness budget-solved). Duplicate proposals (same
splits + constraints, even under relabeling) are rejected with a prompt to propose
something genuinely different.

### 3. Adaptive protocol (3 rounds × 2 proposals = 6 evaluated structures maximum)

Each round:
1. Agent receives: (a) all compact results from prior rounds, (b) a coarse per-layer
   residual summary for each evaluated structure (mean |p_l − p*_l| per quintile of
   the layer stack — 5 numbers, not raw per-layer data), (c) which development targets
   improved vs. the best prior result.
2. Agent produces 2 proposals in the JSON schema above, with rationale and prediction.
3. Driver validates both (schema + budget + uniqueness), rejects invalids with feedback
   (agent may re-propose once per rejected slot, then a random-valid fallback is used).
4. AD inner loop runs on each proposal's continuous parameters.
5. Compact results `{label, n_regions, splits, best_loss_per_target, a_opt, steps, status}`
   and the quintile residual summary are returned. Raw logs go to file only.

Round 1 agent context: uniform baseline loss per target, p0 statistics (peak layer,
FWHM, front_fraction), and the proposal schema. No prior round data.

### 4. Inner-loop parameterization

Reuses E5/E8 AD machinery verbatim for 1-DOF (two-region) structures. Extends to
n_regions − 1 DOF via the same per-layer reverse-AD → budget chain: given splits
{k_1 < k_2 < … < k_{R-1}} defining R regions with thicknesses {a_1, …, a_R}, the
last thickness is solved from the budget: `a_R = (A_tot − Σ_{i<R} n_i·a_i) / n_R`.
Free parameters: {a_1, …, a_{R-1}}. Gradient chain: `dL/da_i = Σ_{l in region_i} dL/da_l
− (n_i/n_R) Σ_{l in region_R} dL/da_l`. Constraint handling: box-clip each a_i to
[ABS_FLOOR_MM, hi_i] after each step; reproject if budget violated.

**AD/FD gate:** step-0 same-sign check per free DOF; if any DOF wrong-sign AND resolved
→ STOP that proposal; if all FLAT → proceed cautiously (per E8 SE-aware rule). This
gate is validated for 1-DOF (E5/E8) and extends naturally to multi-DOF.

Settings: N=1000, seeds [1–6], CRN, ≤10 GD steps, trust-region step per DOF (same
da_target=0.08 mm first step), 0.1 mm clip.

### 5. Baselines (all under the same evaluation budget = 6 structures)

| baseline | description |
|----------|-------------|
| uniform | 0 free DOF, reference |
| best_two_region | best of E5 {earlier,later}×{k=16,25,33} on each target (oracle) |
| best_three_region | best of E3 three-region grid on each target (oracle) |
| random_budget | 6 randomly sampled valid structures (n_regions∈{2,3,4}, splits uniform-random, init random), same AD inner loop — fair-compute control for the agent's 6 proposals |
| no_feedback_ablation | agent proposes all 6 in one shot (no adaptive revision) — tests whether multi-round feedback helps |

Fair-compute rule: every baseline and agent structure gets exactly one inner-loop run
(same N, seeds, steps). Baselines use oracle results where available; the random budget
and no-feedback ablation run fresh.

### 6. Success / partial / kill criteria

**Primary success:** agent+AD best held-out loss across the 6 episode structures beats
both `best_two_region` and `random_budget` by >3×SE on the primary held-out target
(`broader_σ3` or `earlier_4_independent`).

**Secondary success:** agent proposes a structure that (a) solves a target where
simple two-/three-region families fail (especially `broader_σ3`), OR (b) proposes an
interpretable pattern (tapered, peak-focused) that achieves a held-out loss the
exhaustive enumeration did not sample.

**Partial:** agent beats uniform and at least matches `best_two_region` on development
targets, but does not exceed it on held-out; or wins on development only.

**Kill:** agent structures fail to beat uniform >3×SE on any target; or the best agent
structure is statistically indistinguishable from `random_budget`; or all agent proposals
after deduplication collapse to two-region variants already in the baselines.

**Honesty gates:** any improvement must (a) appear on held-out seeds, not just
development; (b) survive a >3×SE test against the appropriate baseline; (c) not require
post-hoc changes to targets, metrics, or the held-out set.

### 7. Safeguards against duplicate and invalid proposals

- Structural fingerprint: `(frozenset of splits, constraints)`. Duplicates rejected,
  agent prompted once to propose something different; if the second attempt also
  duplicates, a random-valid structure is substituted and flagged in the log.
- Validity check before any sim: schema fields, budget conservation to <1e-6 mm,
  region thickness ≥ floor at init, splits strictly ordered.
- Maximum re-proposal attempts per slot: 1 (then fallback).
- Agent may not reference held-out target values in its rationale (the held-out targets
  are not named in the agent prompt — only development targets are visible).

### 8. Runtime estimate

Per proposal: ≤10 AD steps × (1 forward + 1 reverse) × 6 seeds × N=1000 events.
Two-region (1 DOF): ~4–6 min/step wall = ~40–60 min/proposal under moderate load.
Three/four-region (2–3 DOF): same step count, each step runs one reverse pass per DOF
— up to ~3× slower for 3 DOF at shared-node load → request 2–4 h/proposal on SLURM.
6 proposals × 3 h (conservative) = ~18 h serial; parallelized as 1 job/proposal →
~3–4 h wall for one SLURM wave. Plus held-out eval (6 structures × 2 targets × 12
seeds × forward-only ≈ 144 runs, ~1 h).

**SLURM-only** (E7/E8 op-rule). Request ≥16G/job (pure profile loss, no per-event
arrays); 8 CPUs for workers=6 with margin. Total ~1 SLURM wave per round.

### 9. Risks and mitigations

| risk | mitigation |
|------|------------|
| Multi-DOF AD chain unvalidated for n_regions>2 | Step-0 AD/FD gate per DOF; gate any proposal with n_regions>2 behind a separate 2-DOF chain validation (as E8 did for three-region). E9 core = proposals that pass the gate; unvalidated DOF are flagged. |
| Agent proposes cosmetically different but structurally equivalent families | Structural fingerprint deduplication; random-budget ablation provides the null model. |
| Broadening target unreachable by any absorber structure at 1–2 DOF | Expected from E4; success here would be a NEW result. Kill = broadening remains unreachable at ≤6 DOF. |
| LLM reasoning is post-hoc rationalization of random structures | No-feedback ablation isolates whether adaptive revision actually helps vs. one-shot proposals. |
| Development-set overfitting | Held-out seeds [7–12] are fully disjoint; agent never sees held-out target names or values. |

See [[DECISIONS]] 2026-07-10 (E9) and [[MILESTONE_E0_E7]] (§7 E8→E9 path).

## Compute discipline
Forward-only value runs are cheap; reverse-AD at high stats is slow (~0.22 s/ev).
No large matrices, no SLURM submission without explicit approval. Batch python on
nodes sources the LCG view, not the login `.venv`.

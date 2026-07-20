# Milestone: E0 → E6b (gradient validation → controllability → AD optimization → robust performance gain)

Complete evidence chain from untrusted gradients to a robust, held-out energy-
reconstruction improvement from structured absorber geometry. Companion to
[[PROBLEM]], [[HYPOTHESES]], [[EXPERIMENTS]], [[DECISIONS]], and the earlier
[[MILESTONE_E0_E5]]. Dates 2026-07-01 … 2026-07-07.
Sim build: `/sdf/data/atlas/u/jkrupa/agentic/diffcalo`.

---

## 1. Original motivation

**Bilevel detector optimization.** An agent proposes discrete STRUCTURE (region
layout, split points) in an outer loop; a **differentiable Geant-like EM
calorimeter** (hepemshow + g4hepem + CoDiPack) provides EXACT AD gradients so an
inner loop optimizes continuous parameters. The thesis only holds if (a) those
gradients are trustworthy and (b) structure buys something a tuned uniform
detector cannot — and ideally (c) that "something" is a real performance metric.

**Why we were stuck (start of E-series).** No trusted gradient (a proxy reported
AD/FD ≈ 0.03–0.1, a claimed 10–30× bias) and no confirmed value of structure
(every prior "win" — depth resolution, e/γ PID — collapsed under a fair control).

---

## 2. Gradient validation — E0 / E0b

| exp | test | result |
|-----|------|--------|
| E0 | FD machinery on `total_edep` (linear control), CRN, ε-scan | d/dE plateaus AD/FD ≈ 1.0 → **FD validated, "10–30× bias" was an FD artifact (H2 killed)** |
| E0b | d/da on well-conditioned `front_fraction`, SLURM N=54k | AD/FD plateau **0.76–0.79** → **absorber gradients usable**, ~0.8 scale factor |

Also established: judge AD resolution by **empirical across-seed scatter**, not the
sim's `var_dE` column (which overstated SE ~4.6×). **AD gradients are trustworthy.**

---

## 3. Shape controllability — E3 / E3b / E4

At FIXED total absorber budget (A_tot = 115 mm), fixed layer count, absorber-only
shaping, uniform gap, budget conserved by construction.

| exp | question | result |
|-----|----------|--------|
| E3 | can structure hit `front_fraction` targets uniform can't? | three-region reaches ±targets >3×SE better than uniform; two-region "collapsed" (coarse grid) |
| E3b | was the two-region collapse a grid artifact? | YES — with grid=11, all splits k∈{N/3,N/2,2N/3} reach targets >3×SE; **1 DOF suffices** |
| E4 | full normalized profile `p_l=E_l/ΣE`, shift/broaden targets | earlier/later shifts REACHED >3×SE; broadening NOT (reachable manifold is shift/tilt-capable, width-limited) |

**Controllability confirmed:** structure controls longitudinal shape (scalar and
full-profile location) that uniform-at-fixed-budget provably cannot.

---

## 4. AD inner-loop optimization — E5

Can validated gradients OPTIMIZE the structured parameter (not just be gridded)?
Two-region, 1 free DOF (`a_front`, `a_rear` budget-solved), k as outer discrete
choice; gradient via output adjoints → per-layer reverse-AD → budget chain-rule.
Plain GD, ≤15 steps, 6 CRN seeds. Full matrix: 2 targets × k∈{16,25,33}.

| check | result |
|-------|--------|
| step-0 AD/FD sign (all 6) | same sign; ratios 0.49–0.77 (near E0b ~0.8) |
| descent | all 6 beat uniform ≫3×SE, sane trajectories |
| vs grid | at k=16 (only same-k grid bar) AD reaches BELOW the E4 grid-best, both targets |

**AD reliably optimizes the two-region contrast** — the inner optimizer of the
bilevel thesis is established on this objective.

---

## 5. FOM promotion — E6 / E6b

Move from shape reachability to a REAL detector metric: **energy reconstruction**.
Raw per-event layer energies (scale RETAINED, not normalized); train seeds fit the
estimator, disjoint held-out seeds score; geometries fixed in advance.

| exp | setup | result |
|-----|-------|--------|
| E6 | 3 geoms, {3,10,30} GeV, 3 estimators | QUALIFIED SUCCESS: earlier_k16 beats uniform >3×SE under ridge-on-layers; ΣE-baseline shows nothing; estimator-dependent |
| E6b | 5 fixed geoms, {1,3,10,30} GeV, ridge-on-layers only | **PASS: robust** (see §6) |

---

## 6. Final robust E6b result

Held-out relative RMSE, ridge on raw layer energies, train [1–4] / test [5–8],
geometries frozen before evaluation. 160/160 forward runs, 0 timeouts/NaNs, budget
conserved, no cache reuse.

| geometry | held-out rel_RMSE | SE | worst-E | vs uniform | bins improved | verdict |
|----------|-------------------|-----|---------|-----------|---------------|---------|
| uniform | 0.03089 | 4.1e-4 | 0.03992 | — | — | reference |
| **earlier_k16** | **0.02709** | 1.3e-4 | 0.03406 | **+3.80e-3 (>3×SE, ~9σ)** | **4/4** | **PASS** |
| later_k16 | 0.02961 | 8.3e-4 | 0.03763 | +1.29e-3 (≤3×SE) | 3/4 | fragile |
| earlier_k25 | 0.02932 | 5.7e-4 | 0.03774 | +1.57e-3 (≤3×SE) | 3/4 | fragile |
| later_k25 | 0.02992 | 2.1e-4 | 0.03874 | +9.69e-4 (≤3×SE) | 4/4 | fragile |

**earlier_k16 (front-thickened, k=N/3): rel_RMSE 0.02709 vs uniform 0.03089 —
~12% relative improvement, all 4/4 energy bins improved (incl. new 1 GeV),
held-out.** Per-energy earlier_k16 vs uniform: 1G 0.0341/0.0399, 3G 0.0273/0.0319,
10G 0.0231/0.0256, 30G 0.0221/0.0232. The win survived a harder test than E6 and
is not a single-bin accident.

---

## 7. What is established

1. **Trustworthy AD gradients** on this differentiable calorimeter (E0/E0b),
   ~0.8 absorber-thickness scale factor.
2. **Structural shape controllability** at fixed budget — scalar and full-profile
   location/shift (E3/E3b/E4).
3. **AD optimizability** — validated gradients descend to good structured
   profiles, matching/beating a grid (E5).
4. **A robust, held-out PERFORMANCE gain** — one E5-optimized structured geometry
   (earlier_k16) improves energy-reco resolution ~12% over uniform at fixed budget,
   consistently across 1–30 GeV (E6b). "Controllable" → (modestly) "better".
5. Robust methodology throughout: value-first, grid-fair / train-test discipline,
   empirical across-seed SE, budget conservation, loud NaN/timeout handling.

---

## 8. What is NOT established

1. **No agentic outer loop yet** — the discrete structure (k, region count) was
   enumerated by hand, not proposed/searched by an agent. The full bilevel loop is
   unbuilt.
2. **AD not used to optimize the FOM** — E5 optimized a shape-matching loss; E6/E6b
   only EVALUATED energy-reco on fixed geometries. AD has not descended the energy
   FOM directly.
3. **Only 1-DOF two-region AD** — three-region / 2-DOF AD optimization not run.
4. **Effect is modest and narrow** — ~12%, one significant geometry, one estimator
   family (see §9).
5. **Width control absent** — E4 broadening target unreached; structure steers
   location/tilt, not spread.
6. **No absolute-scale physics** — sampling fraction, calibration transfer, or a
   fitted stochastic/constant resolution term not characterized.

---

## 9. Caveats

- **Ridge-on-layer-energies-specific:** the gain requires a layer-weighting
  estimator. The pure ΣE-calibration baseline showed **no** significant gain — a
  detector reading out only total energy would not see this.
- **ΣE-only baseline null:** reinforces that the benefit is in longitudinal
  weighting, not raw collected energy.
- **Fixed-budget EM shower only:** single incident species, one calorimeter, fixed
  total absorber. Not yet a cross-detector or non-EM claim.
- **Modest effect size:** ~12% relative RMSE. Robustness ≠ magnitude — E6b showed
  it is real, not that it is large.
- **One geometry individually significant:** earlier_k16 clears >3×SE; the other
  three structured geometries improve uniform with consistent sign but individually
  <3×SE ("fragile"). The robust claim is specific to the front-thickened profile,
  not "any redistribution helps".
- Mechanism plausibly containment/sampling (front-thickened, gains largest at low
  E) — physically sensible, not shape magic.

---

## 10. Recommended next experiments

Three coherent directions (each needs separate approval; nothing running):

- **A. Agentic outer loop over discrete structure + AD inner optimization.** The
  headline bilevel demo: an agent proposes region layout / split points (discrete);
  the validated AD inner loop (E5) solves the continuous contrast; score on the
  held-out energy FOM (E6b). First end-to-end test of the original thesis. Highest
  narrative value; build on E5 (inner) + E6b (FOM) which are now both established.
- **B. Three-region / 2-DOF AD optimization.** Incremental, cheap, de-risks the AD
  gradient chain past 1 DOF before wiring the agent; also the route to testing
  whether more DOF unlock the width control E4 missed. Natural precursor to A.
- **C. Stochastic-term / energy-resolution characterization.** Fit σ(E)/E =
  a/√E ⊕ b for uniform vs earlier_k16 across a denser energy grid; determine
  WHICH resolution term the structure improves (sampling `a` vs constant `b`),
  turning the ~12% number into a physics statement. Also directly optimize the
  energy FOM with AD (not just evaluate it).

**Suggested order:** B (cheap, de-risks multi-DOF gradients) → A (the bilevel
demonstration, the project's core claim) → C (physics depth / resolution decomposition,
parallelizable and largely independent machinery).

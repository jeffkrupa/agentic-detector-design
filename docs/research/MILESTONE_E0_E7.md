# Milestone: E0 → E7 (gradient validation → controllability → AD optimization → held-out performance gain → AD geometry selection)

Extends [[MILESTONE_E0_E6B]] (kept intact for audit) with the E7 result. Complete
evidence chain from untrusted gradients to: AD selecting the same front-thickened
absorber geometry that improves held-out energy reconstruction. Companion to
[[PROBLEM]], [[HYPOTHESES]], [[EXPERIMENTS]], [[DECISIONS]]. Dates 2026-07-01 … 2026-07-08.
Sim build: `/sdf/data/atlas/u/jkrupa/agentic/diffcalo`.

**Headline (specific, not hyped):** *AD selects the same front-thickened geometry
that improves held-out energy reconstruction.* This is NOT "the agent designed a
detector" and NOT "structure universally improves calorimetry".

---

## 1. Original motivation

**Bilevel detector optimization.** An agent proposes discrete STRUCTURE (region
layout, split points); a **differentiable Geant-like EM calorimeter** (hepemshow +
g4hepem + CoDiPack) provides EXACT AD gradients for an inner loop over continuous
parameters. The thesis needs: (a) trustworthy gradients, (b) structure buying
something a tuned uniform detector cannot, (c) ideally on a real performance metric.

**Why we were stuck (start).** No trusted gradient (a proxy reported AD/FD ≈
0.03–0.1, a claimed 10–30× bias) and no confirmed value of structure (prior "wins"
collapsed under fair controls).

---

## 2. Evidence chain (E0 → E7)

| stage | question | outcome |
|-------|----------|---------|
| E0/E0b | are the gradients trustworthy? | FD validated; "10–30×" was an FD artifact; d/da usable, ~0.8 scale factor |
| E3/E3b | can structure control a scalar shape (front_fraction)? | yes, 1 DOF suffices, >3×SE vs uniform |
| E4 | can it control the full normalized profile? | shifts reachable; broadening not (shift/tilt-capable, width-limited) |
| E5 | can AD optimize the structured param (shape loss)? | yes, all 6 (target,k) combos; AD ≤ grid-best |
| E6/E6b | does structure improve a real FOM (energy reco)? | yes, held-out: earlier_k16 ~12% better than uniform, robust across 1–30 GeV (ridge estimator) |
| E7 | can AD SELECT the energy-FOM geometry itself? | yes — AD surrogate optimum = earlier_k16; held-out gain real but marginal with a frozen estimator |

Details for E0–E6b are in [[MILESTONE_E0_E6B]]; E7 is new below.

---

## 3. E7 — direct AD optimization of the energy-reco FOM

### 3.1 Motivation
The gap E7 closes: **E5** used AD to optimize a *normalized-profile shape* loss;
**E6/E6b** only *evaluated* fixed geometries on energy reconstruction. Neither
showed AD **selecting** the energy-FOM geometry itself. E7 tests exactly that.

### 3.2 Setup
- Fixed split **k=16**; one free absorber parameter **a_front**; **a_rear solved by
  the fixed total absorber budget** (A_tot = 115 mm, conserved by construction).
- **Frozen-from-uniform** ridge estimator: fit once on the uniform geometry, then
  held fixed through the descent.
- **Differentiable mean-reconstruction surrogate**
  `L(θ) = Σ_E ((Ê_mean(E,θ) − E)/E)²`, `Ê_mean = w·Ē_vec + b`, gradient via output
  adjoints → per-layer reverse-AD → budget chain (the validated E5 chain).
- **Held-out E6b-style per-event RMSE** (train seeds [1–4] fit, test seeds [5–8]
  score) is the **honest final judge** — surrogate improvement alone is not success.

### 3.3 Core result
- **AD surrogate optimum at a_front ≈ 2.516**, which **matches the E6b earlier_k16
  geometry**. (An earlier low-stat probe had suggested a uniform optimum; that was
  noise-misled — the full-stats curve is minimized at 2.516, not 2.3.)
- **Machinery PASS:** AD gradients drove the surrogate downhill correctly (grad < 0
  below the optimum, > 0 above, ≈ 0 at it); short descents converged sanely.
- **Held-out per-event rel_RMSE (frozen-from-uniform estimator):**

  | geometry | held-out rel_RMSE | SE |
  |----------|-------------------|-----|
  | uniform | 0.03089 | 4.1e-4 |
  | earlier_k16 | 0.02969 | 5.6e-5 |
  | E7_surr_opt (a≈2.516) | 0.02969 | 5.6e-5 (identical — same geometry) |

  Improvement over uniform ≈ **1.20e-3, about 2.9σ** (just under the 3×SE bar).
- This is **weaker than E6b** (where earlier_k16 beat uniform by ~3.80e-3, ~9σ)
  because **E6b refit the ridge estimator per geometry**, whereas E7 froze it from
  uniform. Same geometry, weaker number.

### 3.4 Revised interpretation
- E7 **closes the gap** that AD had not previously *selected* the energy-FOM
  geometry itself: AD FOM-surrogate optimization **recovers the same front-thickened
  geometry** (earlier_k16) that E6b found by hand.
- The **full** performance gain **depends partly on geometry–estimator
  co-adaptation**: the frozen-from-uniform readout under-realizes the E6b gain
  (2.9σ vs 9σ). The mean-surrogate captures enough to pick the right geometry (the
  location/front-thickening effect is visible in the mean profile), but the full
  held-out RMSE advantage needs a matched (refit) estimator — part of the gain lives
  in the resolution/variance channel and estimator coupling a frozen mean-surrogate
  under-weights.

---

## 4. What is established

1. **AD gradients are trustworthy enough for design** (E0/E0b), ~0.8 absorber scale
   factor.
2. **Structured absorbers control longitudinal shower shape** at fixed budget —
   scalar and full-profile location (E3/E3b/E4).
3. **AD optimizes structured absorber parameters** (E5, shape loss; all 6 combos).
4. **Fixed-budget structure improves held-out energy reconstruction** — earlier_k16
   ~12% better rel_RMSE than uniform, robust across 1–30 GeV (E6b, ridge estimator).
5. **AD can SELECT the same FOM-useful geometry through a differentiable surrogate**
   (E7): the AD surrogate optimum is earlier_k16, machinery verified.

---

## 5. What is NOT established

1. **No agentic outer loop yet** — the discrete structure (k, region count) was
   enumerated / hand-set, not proposed by an agent. The full bilevel loop is unbuilt.
2. **No proof of a global optimum** — E5/E7 are local gradient descents on 1 DOF
   at a fixed split; the reachable set was probed, not proven optimal.
3. **Only simple two-region / one-DOF structures tested for the FOM** — no
   three-region / multi-DOF FOM optimization.
4. **The strongest energy-reco gain still requires a layer-aware estimator and
   appears estimator-dependent** — the ΣE-calibration baseline showed no gain; the
   frozen-estimator E7 gain is marginal (2.9σ); the full ~9σ needs a refit readout.
5. **No medicine/space (or any cross-domain) transfer claim** — this is fixed-budget
   EM-shower energy reconstruction in one simulated sampling calorimeter.

---

## 6. E7 caveats

- **SLURM memory / OOM:** E7 ran on SLURM after 3 local runs died to shared-node
  load spikes. The successful job OOM'd (MaxRSS ~33.5G > 32G) **after the core data
  printed** — results are usable (only trailing format lines lost), but E7-class
  forward+reverse multi-energy jobs need **≥48–64G** (or fewer concurrent workers).
- **Frozen vs refit estimator:** the frozen-from-uniform estimator gives a weaker
  held-out gain (2.9σ) than E6b's per-geometry refit (9σ). The frozen number is the
  conservative one; the refit number is the fairer geometry comparison.
- **Surrogate, not full RMSE:** E7 optimizes a MEAN-reconstruction surrogate (the
  sim's AD exposes mean-profile gradients, not per-event spread), so it targets a
  bias/calibration-like quantity. The per-event held-out RMSE remains the judge.
- **Optional refit-at-final diagnostic:** would quantify E7's geometry under a
  matched estimator (likely recovering the ~9σ gain); it did not print (OOM tail).
  A single 64G rerun recovers it — **not needed for the main conclusion**.

---

## 7. Recommended next steps

- **E8 (headline): controlled agentic / discrete outer loop over structure, with AD
  inner optimization.** An agent proposes the discrete structure (split count /
  positions); the validated AD inner loop (E5/E7) solves the continuous contrast;
  score on the held-out energy FOM (E6b). First end-to-end test of the bilevel
  thesis. Builds directly on now-established E5 (inner) + E6b (FOM) + E7 (AD selects
  geometry). *Not started; needs separate approval.*
- **Optional: three-region / 2-DOF AD** — de-risks the gradient chain past 1 DOF;
  also probes whether more DOF unlock the width control E4 missed.
- **Optional: stochastic-term / energy-resolution characterization** — fit
  σ(E)/E = a/√E ⊕ b for uniform vs earlier_k16; localize which term structure
  improves, turning the ~12% into a physics statement.
- **Optional: 64G E7 rerun** for the refit-at-final diagnostic table.

**Suggested order:** three-region/2-DOF AD (cheap, de-risks multi-DOF) → E8 (the
bilevel demonstration) → resolution characterization (physics depth, parallelizable).

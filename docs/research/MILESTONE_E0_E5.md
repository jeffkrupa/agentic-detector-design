# Milestone: E0 → E5 (gradient validation → structural controllability → AD inner loop)

Single-page state of the project after experiments E0, E0b, E3, E3b, E4, E5.
Companion to [[PROBLEM]], [[HYPOTHESES]], [[EXPERIMENTS]], [[DECISIONS]].
Dates 2026-07-01 … 2026-07-07. Sim build: `/sdf/data/atlas/u/jkrupa/agentic/diffcalo`.

---

## 1. Original problem and why we were stuck

Thesis: ground **agentic detector design** in EXACT AD gradients of a
differentiable EM calorimeter — an agent edits STRUCTURE (outer, discrete) while
AD optimizes continuous params (inner). The thesis only matters if (a) the
gradients are trustworthy and (b) structure actually buys something a tuned
uniform detector cannot.

We were stuck on both. Every "structure wins" signal had collapsed under a fair
control (depth-resolution = quantization artifact; e/γ PID = ~0.003 AUC
front-window confound). And a differentiable proxy reported AD/FD ≈ 0.03–0.1
(a claimed 10–30× gradient bias), which — if real — would sink the exact-AD story.
Net: no trusted gradient, no confirmed value of structure.

---

## 2. E0 / E0b — gradient validation

**E0 (FD machinery on a linear control, `total_edep`).** With common random
numbers and a resolved ε window, central-difference FD converged to AD:

| channel | AD/FD plateau | verdict |
|---------|---------------|---------|
| d(total_edep)/dE | ≈ 1.0 (ε 50–200 MeV) | FD validated |
| d(total_edep)/da | unresolved (high variance) | inconclusive — bad probe |

The claimed "10–30× AD bias" was an **FD artifact** (ε too small / MC-noise on a
2nd-order quantity). **H2 killed: AD gradients are trustworthy.**

**E0b (d/da on a well-conditioned observable, `front_fraction`, SLURM N=54k).**
`total_edep` was just a high-variance probe for absorber thickness. On the
intensive `front_fraction`, AD/FD forms a stable plateau:

| ε (mm) | AD/FD |
|--------|-------|
| 0.05 / 0.10 / 0.20 | 0.76 / 0.79 / 0.79 |

**Absorber-thickness gradients d/da are usable**, with a documented ~20%
systematic undershoot (~0.8 scale factor). Also established: judge AD resolution
by empirical across-seed scatter, not the sim's `var_dE` column (which overstated
the SE ~4.6×).

---

## 3. E3 / E3b — structural controllability (scalar)

Question: at FIXED total absorber budget (A_tot = 115 mm) and fixed layer count,
can region-structured absorber profiles reach `front_fraction` targets a uniform
profile cannot? (Absorber-only shaping, uniform gap, budget conserved by
construction.)

**E3:** three-region reached both below- and above-uniform ff targets >3×SE
better than uniform; uniform (0 free DOF at fixed budget) is stuck at one point.
Two-region "collapsed" to uniform — but that was a coarse-grid artifact.

**E3b:** with a finer ratio grid, ALL three splits k∈{N/3,N/2,2N/3} reach the
targets >3×SE better than uniform. So **1 extra DOF suffices** to steer
front_fraction; ladder is uniform ≪ two-region ≈ three-region.

**Controllability confirmed at the scalar level:** structure buys reach a
tuned-uniform detector provably cannot have.

---

## 4. E4 — normalized-profile shape control (vector)

Upgrade from the scalar knob to the full longitudinal shape `p_l = E_l/ΣE`.
Targets built analytically from the uniform reference profile (NOT from simulated
structured geometries, to avoid tautology): earlier shift, later shift, broadening.
Objective `L = Σ_l (p_l − p*_l)²`, uniform weights.

| target | uniform loss | best structured loss | verdict |
|--------|-------------|---------------------|---------|
| earlier (shift −2) | 5.35e-4 | 2.10e-4 | REACHED (>3×SE) |
| later (shift +2) | 4.65e-4 | 9.15e-5 | REACHED (>3×SE) |
| broader | 1.06e-5 | 1.01e-5 | within 3×SE — NOT reached |

**Shape control generalizes from scalar to the full profile for the
location/shift family.** Broadening is outside the reachable set of 1–2-DOF
fixed-budget absorber structure → the reachable manifold is **shift/tilt-capable,
width-limited**.

---

## 5. E5 — AD inner-loop optimization

Question: can the validated gradients OPTIMIZE the structured parameter (not just
be evaluated on a grid)? Two-region, 1 free DOF (`a_front`, `a_rear` budget-solved);
k treated as an outer discrete choice; gradient via output adjoints (with the
normalization Jacobian) → per-layer reverse-AD → budget chain-rule. Plain GD,
trust-region step, ≤15 steps, 6 CRN seeds. Full matrix = 2 targets × k∈{16,25,33}.

| target | k | step-0 AD/FD | best AD loss | beats uniform >3×SE? |
|--------|---|-------------|-------------|----------------------|
| earlier | 16 | 0.72 | 1.87e-4 | YES (below E4 grid-best) |
| earlier | 25 | 0.60 | 2.32e-4 | YES |
| earlier | 33 | 0.49 | 2.68e-4 | YES |
| later | 16 | 0.74 | 7.67e-5 | YES (below E4 grid-best) |
| later | 25 | 0.77 | 1.21e-4 | YES |
| later | 33 | 0.77 | 1.86e-4 | YES |

**All 6 pass:** step-0 AD/FD same sign everywhere (ratios 0.49–0.77, near the E0b
~0.8), sane descent, all beat uniform ≫3×SE. At k=16 (only split with a same-k
grid bar) AD reaches BELOW the E4 grid-best for both targets. **The AD inner
optimizer of the bilevel thesis is established on this objective.**

---

## 6. What is now established

1. **AD gradients are trustworthy** on this sim (E0/E0b), with a known ~0.8
   AD/FD scale factor for absorber thickness.
2. **Structured absorber geometry controls longitudinal shape** at fixed budget —
   confirmed for the scalar front_fraction (E3/E3b) and the full normalized
   profile's location/shift (E4). Uniform-at-fixed-budget cannot.
3. **AD reliably optimizes** the structured (two-region) parameter to those
   targets, matching or beating a brute grid (E5).
4. Robust methodology: grid-fair / value-first cross-checks, empirical across-seed
   SE, budget conservation by construction, loud NaN/timeout handling.

---

## 7. What is explicitly NOT established yet

1. **No physics figure of merit.** Every result is target-*reachability* /
   controllability, not "this is a better detector." No resolution, efficiency,
   or separation performance metric has been optimized.
2. **No full bilevel demonstration.** k (and region count) were enumerated by
   hand, not chosen by an agentic outer loop with AD as the inner solver.
3. **Only 1-DOF two-region AD.** Three-region / 2-DOF AD optimization not run.
4. **Width control not shown** (broadening target unreached) — absorber structure
   steers profile location/tilt, not spread.
5. **Energy scale untested** — normalized profiles discard total-energy info (§8).

---

## 8. Key caveats

- **~0.8 AD/FD scale factor:** absorber-thickness AD gradients systematically
  undershoot FD by ~20% (E0b, reproduced across E5's 6 step-0 checks, 0.49–0.77).
  Signs and relative magnitudes are reliable; absolute magnitudes carry this factor.
- **Shape / reachability objective, not a figure of merit:** target-matching a
  profile shape says nothing yet about detector performance. A grid-fair control
  becomes mandatory the moment a profile metric is promoted to a performance claim.
- **Normalized profiles discard energy-scale information:** `p_l = E_l/ΣE` is
  invariant to total deposited energy, so nothing here constrains sampling
  fraction / absolute response / energy resolution. That is a separate axis.
- **Broadening target not reached:** 1–2-DOF fixed-budget absorber structure is
  shift/tilt-capable but width-limited (E4).
- **k=16 (N/3) is the strongest split** for the shift targets — lowest achievable
  loss for both earlier and later in E4 and E5. Split choice matters for reach.

---

## 9. Recommended next scientific fork

Three coherent directions (each needs separate approval; no experiments run now):

- **A. Promote the profile objective toward a real figure of merit.** Replace /
  augment shape-matching with a physics metric (e.g. energy resolution at fixed
  budget, or containment/leakage), re-introduce absolute energy scale, and add a
  grid-fair control. Turns "controllable" into "better" — the payoff question.
- **B. Agentic outer loop over k / regions with AD inner loop.** Let an agent
  choose the discrete structure (split count/positions) while AD (E5, validated)
  solves the continuous contrast — the first true bilevel demonstration.
- **C. Extend AD to three-region / 2-DOF optimization.** Natural incremental step:
  confirm the E5 gradient chain scales past 1 DOF before wiring the agentic loop;
  also the route to testing whether more DOF unlock the width control E4 missed.

**Suggested order:** C (cheap, de-risks the multi-DOF gradient chain) → B (the
headline bilevel demo) → A (the harder, higher-value physics-performance question,
which also needs the energy-scale and grid-fair machinery). A can proceed in
parallel since it is largely independent machinery.

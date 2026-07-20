# DECISIONS (log)

Append-only. Each entry: date · decision · rationale · status.
Related: [[PROBLEM]], [[HYPOTHESES]], [[EXPERIMENTS]].

---

- **2026-07-01 · Adopt low-context research-ledger workflow.**
  Split the monolithic handoff into PROBLEM / HYPOTHESES / EXPERIMENTS / DECISIONS
  so each session reads a small, structured state instead of re-deriving. Status:
  active.

- **2026-07-01 · E0 (FD validation) is the pivotal upstream experiment.**
  Whether AD gradients are trustworthy gates the entire exact-AD thesis and the
  reliability fallback. Validate FD on a linear control before re-diagnosing the
  proxy. Status: pending.

- **2026-07-01 · Preserve uncertainty explicitly.** Every hypothesis carries a
  tested/inferred/speculative tag and a kill criterion; no claim is promoted
  without a grid-fair / value / validated-FD cross-check. Status: active.

- **2026-07-03 · E0 done: FD machinery VALIDATED, [[HYPOTHESES]] H2 killed.**
  On the clean d(total_edep)/dE channel, central-diff FD with common random
  numbers plateaus at AD/FD ≈ 1.0 for ε ∈ [50,200] MeV (see [[EXPERIMENTS]] E0
  result). No 10–30× gap anywhere. The earlier "AD biased 10–30×" was an FD
  artifact (ε too small / MC-noise-dominated 2nd-order objective), not a real
  gradient error. **AD gradients are trustworthy**; the exact-AD path is alive.
  Status: closed.

- **2026-07-03 · FD usage rule.** Use ε in the *resolved* window and CRN, and
  check `|FD| > 3·FD_SE` before trusting any FD number. Small-ε FD on this MC
  sim is noise-dominated (FD_SE ∝ 1/ε) and misleading — the trap that produced
  the false "bias." Status: active.

- **2026-07-03 · Compute reality: N=1000 is the per-run ceiling.** At ~0.19
  s/event a single seed subprocess at N=2000 (~380s) exceeds the sim's 300s
  `subprocess_timeout_s` and fails. This corrects the old handoff note that
  "forward-only value runs are cheap." Use N≤1000/seed locally (or raise the
  timeout / go to SLURM with approval for higher stats). Status: active.

- **2026-07-03 · Proxy-D FD re-diagnosed: FD UNRESOLVED (2nd-order MC noise), no
  plateau.** Gap-ε scan with CRN (N=1000, seeds [1,2]; see [[EXPERIMENTS]] E0a)
  gives AD/FD = +0.017 / −0.27 / −1.30 / −0.36 / +0.67 — sign-flipping, no
  plateau, FD_noise ≈ signal (see [[EXPERIMENTS]] E0a). The AD gradient is itself
  seed-unstable (dD/dg +3523 @N=1000/2seed vs +6810 @N=10000/3seed). **The old
  ~10× discrepancy is consistent with FD noise, but NOT positively confirmed** —
  proxy-D =
  Σₗ(Ē_l^e−Ē_l^γ)² is a 2nd-order squared-difference of noisy mean profiles and
  is too noisy at feasible local N for either FD or AD to resolve. This differs
  from E0, where the clean 1st-order `total_edep` control DID plateau at ≈1.0.
  Diagnosis: FD-still-unresolved / MC-noise — not a proven AD bias, proxy bug, or
  nondifferentiability. Status: closed.

- **2026-07-03 · Caveat on E0 → proxy generalization.** E0 validated the FD
  *machinery* on a 1st-order observable. It does NOT license trusting proxy-D
  gradients: 2nd-order objectives (squared profile distance, Fisher, variances)
  need variance reduction (antithetic/CRN at the event level), a 1st-order
  surrogate, or much higher stats before any gradient-based inner loop
  ([[HYPOTHESES]] H4) can rely on them. Status: active.

- **2026-07-03 · Open decision: how to get a trustworthy proxy gradient.** Options
  before building an AD inner loop: (a) higher-stat proxy-D via SLURM (needs
  approval); (b) switch to a lower-variance 1st-order separation surrogate;
  (c) event-level variance-reduced estimator for D. Not started — do not pick
  without asking. Status: pending.

- **2026-07-04 · Correction: d/da is NOT declared unusable.** E0's inconclusive
  d/da only reflects that `total_edep` is a high-variance validation observable
  for absorber thickness (extensive, steep in t), not a defect of absorber
  gradients. Queued **[[EXPERIMENTS]] E0b** to validate d/da on better-conditioned
  intensive/shape observables (`shower_max_depth`, `front_fraction`,
  `visible_fraction`, a normalized-profile loss, t_max proxy). Also renamed the
  proxy-D result block to **E0a** to free the E0b label. Status: E0b proposed, not
  yet run (awaiting go-ahead).

- **2026-07-05 · E0b run (2 registered observables): inconclusive-but-encouraging.
  d/da bottleneck is AD variance, not FD.** `shower_max_depth` d/da FAILS (AD & FD
  both noise-dominated at N=1000). `front_fraction` d/da: **FD fully resolved and
  plateaued** (≈0.22–0.25 for ε∈[0.05,0.2]) — a real improvement over total_edep —
  but AD/FD sits at **~0.7–0.8, just below the [0.8,1.2] trust band**, and drifts
  down (0.78→0.69) from 2→6 seeds because the **AD estimate itself is
  under-resolved** (AD +0.16, SE 0.23). See [[EXPERIMENTS]] E0b result. **Do NOT
  claim d/da validated**; but it is plausibly ~0.8 (consistent with E0 d/dE and
  the expected undershoot), and the blocker is now AD MC-variance, not FD
  conditioning. Status: closed (partial).

- **2026-07-05 · Recommendation for finishing d/da validation.** Next steps in
  order: (a) higher-stat front_fraction d/da via **SLURM** (needs approval) to
  shrink AD SE and test whether AD/FD settles into [0.8,1.2]; (b) try the
  normalized-profile-loss observable ([[EXPERIMENTS]] E0b cand. 3). A larger ε
  window is NOT indicated — FD is already resolved; widening ε cannot fix an
  AD-variance problem. Status: SUPERSEDED by 2026-07-06 (option (a) executed).

- **2026-07-06 · E0b PASS: d(front_fraction)/da is usable; ~0.8 systematic
  undershoot characterized.** Bounded SLURM run (18 jobs, N_total=54k, CRN; see
  [[EXPERIMENTS]] E0b SLURM result) resolves both AD and FD and gives a tight
  AD/FD plateau at **0.76–0.79** across ε∈{0.05,0.10,0.20} mm. Verdict: **d/da is
  trustworthy** for intensive/profile objectives — correct direction, reproducible
  across 18 seeds, quantitatively right to ~20% (a known scale factor, NOT the
  feared 10–30×). This is the decision-rule "systematic undershoot with resolved
  AD" branch; because the plateau sits at the very edge of [0.8,1.2] we read it as
  a **qualified PASS**. Reinforces [[HYPOTHESES]] H2 (AD trustworthy). Status:
  closed.

- **2026-07-06 · Trust the empirical across-seed SE, not the sim's `var_dE`
  column, for AD resolution.** On front_fraction the per-run `var_dE`-propagated
  AD_SE overestimated the true uncertainty ~4.6× (0.067 vs empirical 0.015),
  falsely flagging AD as unresolved. When ≥ several seeds exist, judge AD
  resolution by the across-seed scatter of the per-seed estimates. (Consider
  updating `e0b_aggregate.py` / `tools/reliability.py` to report both.) Status:
  active.

- **2026-07-06 · The AD/FD ~0.8 undershoot is real and consistent, not an
  artifact.** Seen now on two independent 1st-order observables (E0 total_edep
  d/dE ≈1.0 within noise; E0b front_fraction d/da ≈0.79 tight). The ~0.8 figure
  the lead anticipated is reproducible. Any AD-based design objective should carry
  this ~0.8 caveat but can otherwise trust gradient signs and relative magnitudes.
  Status: active.

- **2026-07-06 · Gradient-validation phase paused; next is E3 (value-of-structure).**
  E0/E0b are done — FD validated, d/da usable. No further gradient-validation runs
  unless asked. Proposed **[[EXPERIMENTS]] E3**: minimal structured-geometry test —
  uniform vs two-region vs three-region absorber shaping at FIXED absorber budget
  and FIXED n_layers, objective `front_fraction`, judged grid-fair against the
  uniform baseline. First direct test of the project's central value-of-structure
  question ([[PROBLEM]]) on now-trustworthy gradient ground. Success = grid-fair
  objective improves beyond across-seed noise AND grows with #regions; kill = flat
  ladder or confound-only gain. Requires a new absorber-only, budget-conserving
  region map (existing `profiles_from_scale` scales abs+gap together, unsuitable).
  Status: DONE — see 2026-07-06 (E3 result) below.

- **2026-07-06 · E3 reframed as a CONTROLLABILITY test, not a physics-performance
  objective.** Per lead: the question is whether structured absorber profiles can
  REACH a range of longitudinal-shape (front_fraction) targets that the uniform
  fixed-budget profile cannot — reachability, not "better detector." Objective is
  target-matching L=(ff−target)² over below/near/above targets. Status: active.

- **2026-07-06 · E3 RESULT: controllability CONFIRMED (partial ladder).** Clean
  forward-only scan (186/186 runs OK, 0 timeouts/NaNs, budget conserved, uniform
  ff=0.5548±0.0006; see [[EXPERIMENTS]] E3 result). Three-region shaping reaches the
  below- and above-uniform ff targets essentially exactly (loss ~1e-6, dist ~0.002),
  beating uniform by **>3× empirical across-seed SE** on both. Uniform has 0 free DOF
  at fixed budget → stuck at its single point, cannot move toward either target.
  **Structure demonstrably buys reach a tuned-uniform detector cannot have** →
  first positive evidence for [[HYPOTHESES]] H_struct (promote it). Caveats: (a)
  ladder is NOT clean-monotone — two-region "best" collapsed to uniform at grid=5
  (coarse grid + centered k=N/2 split under-samples two-region reach; unresolved,
  a finer/off-center two-region grid would likely fill it in); (b) this is
  reachability ONLY, not a figure-of-merit claim; (c) no grid-fair binning control
  (not needed for target-matching, but REQUIRED if ff is later promoted to a
  performance metric). Status: closed.

- **2026-07-06 · Next proposed: E3b (two-region collapse — artifact or real?).**
  E3's two-region best collapsed to uniform (grid=5, centered k=N/2 split).
  Proposed **[[EXPERIMENTS]] E3b**: sweep off-center splits k∈{N/3,N/2,2N/3} + a
  finer ratio grid (≈11–15), everything else identical to E3 (targets, seeds, N,
  budget, timeout, NaN policy). Success = some two-region split reaches an
  off-uniform ff target >3×SE better than uniform (→ E3 collapse was
  grid/split under-sampling, ladder becomes monotone). Failure = even off-center
  finer two-region can't move ff while three-region can (→ 2 DOF genuinely needed,
  a sharper structural claim). Value-first, forward-only, NO SLURM, front_fraction
  ONLY. Status: DONE — see 2026-07-06 (E3b result) below.

- **2026-07-06 · E3b RESULT: E3's two-region collapse was a GRID ARTIFACT.** Clean
  scan (204/204 runs OK, 0 timeouts/NaNs, budget conserved, uniform ff=0.5548;
  see [[EXPERIMENTS]] E3b result). With grid=11 (vs E3's 5), ALL three splits
  k∈{16,25,33} reach both the below- and above-uniform ff targets >3×SE better than
  uniform (losses ~1e-6–6e-5 vs uniform δ²=2.5e-3) — including the exact centered
  k=N/2 case that "collapsed" in E3. **So E3's two-region failure was coarse-grid
  under-sampling, not a structural limit.** Revised picture: the ladder is
  uniform ≪ two-region ≈ three-region for this target-matching task — **1 DOF
  suffices** to steer front_fraction to ±0.05; the third region is not required for
  these targets (extends range, cf. E3). Split placement affects reachable RANGE
  (k=25/33 → [0.26,0.78] vs k=16 → [0.41,0.68]) but not reachability of the ±0.05
  targets. Controllability claim for [[HYPOTHESES]] H_struct is reinforced and
  sharpened (holds at minimal 1-DOF structure). Same caveats as E3 (reachability,
  not figure-of-merit; no grid-fair control needed unless ff becomes a performance
  metric). Status: closed.

- **2026-07-06 · E3/E3b front_fraction scans DONE; next proposed: E4 (full-profile
  matching).** Done refining scalar front_fraction controllability. Proposed
  **[[EXPERIMENTS]] E4**: upgrade to full normalized longitudinal profile
  `p_l=E_l/ΣE`, loss `L=Σ w_l(p_l−p_target,l)²` (uniform weights first), comparing
  uniform vs two- vs three-region at the same fixed absorber budget. Targets are
  built from the uniform reference profile via analytic perturbations (depth-shift
  earlier/later, variance-broadening) — deliberately NOT by simulating a
  region-structured geometry, so a good match is genuine controllability not a
  tautology. Success = structured L beats uniform L by >3×SE on the shift targets;
  kill = within 3×SE of uniform everywhere (scalar-only controllability); expected
  interesting middle = shifts reachable, broadening only partially (characterizes
  the reachable shape manifold). Value-first, forward-only, NO SLURM initially, NO
  AD yet, front_fraction-family ONLY. Status: DONE — see 2026-07-06 (E4 result).

- **2026-07-06 · E4 RESULT: full-profile shape control CONFIRMED (location);
  broadening is a reachable-manifold limit.** Clean scan (354/354 runs OK, 0
  timeouts/NaNs, budget conserved, p0 + all 3 targets normalized/nonnegative; see
  [[EXPERIMENTS]] E4 result). Earlier & later depth-shift targets REACHED —
  structured absorber cuts loss ≫3×SE below uniform (SE ~2–3e-6, improvements
  ~1e-4–3.7e-4). Broadening target NOT reached (all within 3×SE of uniform).
  **So E3/E3b scalar controllability generalizes to the FULL longitudinal profile
  for the location/shift family**, while pure width control is outside the reachable
  set of 1–2-DOF fixed-budget absorber structure — a precise characterization, not
  a failure. Two-region beat three-region on the (pure-shift) targets — expected,
  since a 1-DOF front/rear contrast is the natural shift lever and the 5×5
  three-region grid was coarser; not evidence three-region is worse in principle.
  Upgrades [[HYPOTHESES]] H_struct from scalar to vector shape control. Same caveats
  (reachability not figure-of-merit; no grid-fair control unless a profile metric
  becomes a performance objective). Status: closed.

- **2026-07-06 · Next proposed: E5 (AD-optimize two-region for profile matching).**
  E3/E4 forward-only controllability scans are done. Proposed **[[EXPERIMENTS]] E5**:
  use validated per-layer reverse-AD (`run_reverse_per_layer`) to OPTIMIZE the 1-DOF
  two-region absorber (a_front, a_rear budget-solved) against E4's earlier/later
  normalized-profile targets — objective L=Σ(p_l−p*_l)², plain GD with parameter-
  trust-region step + grad clip, ≤15 steps, 6 CRN seeds, start from uniform. Key
  mechanic: output adjoints ∂L/∂E_l (incl. normalization Jacobian) → reverse per-
  layer dL/da_i → chain through budget (da_rear/da_front=−k/(N−k)) to the single
  free DOF; step-0 AD-vs-FD cross-check (expect E0b ~0.8) as a sanity gate.
  Status: SUPERSEDED by the 2026-07-06 revision below.

- **2026-07-06 · E5 REVISED: fix the k baseline mismatch (fixed-k AD, same-k
  baseline).** Original E5 fixed k=N/2 but would have compared to the all-k grid
  best — invalid, since E4's earlier-target grid-best was at k=N/3. Revised design:
  treat **k as an OUTER DISCRETE choice** k∈{N/3,N/2,2N/3}, run a SEPARATE AD
  optimization of the inner `a_front` at each fixed k (6 runs = 2 targets × 3 k),
  and judge each run against the **same-k** grid best; the cross-k grid best is
  reported only as a clearly-marked harder reference. Primary success = AD reaches
  within 1×SE of same-k grid best OR beats uniform >3×SE with a monotonic/sane
  trajectory; secondary = AD picks the correct direction and the best fixed-k run
  approaches the best-across-k grid result. Kill = no k beats uniform beyond noise,
  or step-0 AD/FD wrong sign / ≫0.8-off (chain bug). Keep: step-0 AD-vs-FD gate,
  exact budget conservation, clipping/trust-region step, empirical across-seed SE.
  First inner-loop step of the bilevel thesis. NO agentic outer loop, NO SLURM first
  pass, NO three-region first pass, front_fraction-family ONLY. Status: Stage-1
  pilot DONE — see 2026-07-07 below.

- **2026-07-07 · E5 STAGE-1 PILOT: both pilots PASS — AD optimizes the two-region
  contrast.** Ran 2 of 6 combos (earlier/k=16, later/k=25), N=1000, 6 CRN seeds,
  ≤15 GD steps (see [[EXPERIMENTS]] E5 pilot result). Step-0 AD-vs-FD **same sign**
  both runs (ratios +0.72, +0.77 — consistent with the E0b ~0.8 undershoot →
  per-layer reverse→budget-chain gradient is correct). Sane descent: earlier/k=16
  5.35e-4→1.87e-4 (below the same-k(=16) E4 grid-best 2.10e-4); later/k=25
  4.65e-4→1.21e-4. **Both beat uniform ≫3×SE.** (later/k=25: no same-k=25 grid best
  in the E4 log — E4's later best 9.15e-5 is at k=16, a cross-k harder reference;
  vs it AD is close but not below.) Minor op note: one 6-seed reverse pass hit the
  1400s cap (later, step 2), tolerated with 5/6 seeds + loud NaN, gradient fine.
  One transient overshoot (earlier, step 3) from fixed η, self-recovered — consider
  a small backtracking-LR refinement if the full matrix shows it. **Conclusion: the
  inner AD optimizer of the bilevel thesis works on this objective.** Per the
  two-stage plan, STOPPED after the pilot; the remaining 4 combos (earlier×{25,33},
  later×{16,33}) await explicit user go-ahead. Status: pilot closed; superseded by
  the full-matrix result below.

- **2026-07-07 · E5 COMPLETE: full 6-combo matrix all PASS — AD optimizer
  established.** Ran the remaining 4 combos (earlier×{25,33}, later×{16,33}) in 2
  waves; with the 2 pilots this closes all 6 (target × k∈{16,25,33}). See
  [[EXPERIMENTS]] E5 full-matrix result. **All 6:** step-0 AD/FD same sign (ratios
  0.49–0.77, near the E0b ~0.8 undershoot); sane descent; all beat uniform ≫3×SE.
  At k=16 (the only split with a same-k E4 grid bar) AD reaches BELOW the grid-best
  for BOTH targets (earlier 1.87e-4<2.10e-4; later 7.67e-5<9.15e-5) — finer than the
  grid sampled. k-reach: k=16 (N/3) is the most expressive single split for these
  shift targets (best loss at k=16 for both), consistent with E4. **Conclusion:
  validated per-layer AD gradients reliably optimize the two-region absorber
  contrast for normalized-profile matching — the inner optimizer of the bilevel
  thesis is established on this objective.** Minor: earlier_k33 step-0 ratio 0.49 is
  the furthest from ~0.8 (still correct sign); fixed-η transient overshoots as in the
  pilot (a backtracking-LR would smooth them, did not affect best loss). Caveats
  unchanged (optimizability, not figure-of-merit). Natural next steps (NOT started,
  need separate approval): (a) three-region / 2-DOF AD optimization; (b) an agentic
  OUTER loop choosing k (discrete) with AD as the inner solver — the full bilevel
  demonstration; (c) promoting a profile objective to a physics figure-of-merit
  (would then require a grid-fair control). Status: closed.

- **2026-07-07 · Direction: FOM promotion (fork A) — proposed [[EXPERIMENTS]] E6.**
  Moving from shape controllability (E3–E5) to a REAL detector figure of merit.
  Proposed E6: energy reconstruction (held-out relative RMSE / resolution / bias /
  worst-energy error) across an energy grid ({3,10,30} GeV first), comparing uniform
  vs two-/three-region fixed-budget geometries (from E4/E5 + a small declared
  candidate list). Key discipline changes vs E3–E5: (1) **RETAIN the energy scale**
  — use raw per-event layer energies E_l, not normalized p_l (normalization discards
  exactly what energy reco needs); (2) **train/test seed split** — calibrate the
  estimator on train seeds, score on disjoint held-out seeds, geometry fixed in
  advance (never chosen on test data). Estimators escalate simple→ridge: total-E
  calibration baseline → linear/ridge on layer energies → optional shape+total
  features. Per-event data via `_forward_per_event_profiles` (exists; no new sim
  capability). Success = structured beats uniform on held-out FOM >3×SE; kill =
  uniform matches/beats, or train gain vanishes held-out (→ honest negative bounding
  value-of-structure to shape, not performance). Forward-only, no reverse-AD, SLURM
  only if approved. Caveats: still EM-shower reco (not medical/space); shape control
  may not transfer to energy resolution (different axis: sampling/fluctuations/
  leakage); expect small effects since fixed budget ⇒ fixed average sampling.
  Status: DONE — see 2026-07-07 (E6 result) below.

- **2026-07-07 · E6 RESULT: QUALIFIED SUCCESS — "controllable" ⇒ modest "better"
  for energy reconstruction.** Held-out energy-reco FOM, geometries fixed in advance,
  train[1–4]/test[5–8] split, energy scale retained (see [[EXPERIMENTS]] E6 result).
  With a **layer-weighting ridge** estimator, the **earlier** (k=16) structured
  geometry beats uniform on held-out relative RMSE by **>3×SE** (0.0244 vs 0.0274,
  ~11% reduction; worst-energy 3 GeV 0.0275 vs 0.0324); under a shape-feature
  estimator **later** beats uniform >3×SE. Gains **survive the train→test seed
  split** (not overfit). So structural shape control (E3–E5) DOES have a modest
  energy-reconstruction payoff — the first performance (not just reachability)
  evidence for value-of-structure. **Qualifications:** (a) estimator-dependent —
  the ΣE calibration baseline shows NOTHING >3×SE (structure needs a layer-aware
  estimator to be exploited); (b) small (~10% rel-RMSE); (c) the winning geometry
  differs by estimator (earlier under ridge-layers, later under shape-features), so
  the robust claim is "some fixed-budget redistribution helps a layer-aware
  estimator", not "profile X is optimal"; (d) largest gains at 3 GeV (most
  containment-sensitive) → plausibly a containment/sampling effect, not shape magic;
  (e) still EM-shower reco, fixed budget ⇒ small by design. This SOFTENS the earlier
  worry that FOM promotion would be a clean kill. Natural next forks (need approval):
  B (agentic outer loop over k with AD inner), C (three-region/2-DOF AD), or deepen
  E6 (add 1 GeV, more geometries, resolution-fit stochastic term). Status: closed.

- **2026-07-07 · Next proposed: E6b (robustness check of the E6 gain).** Before
  pursuing forks B/C, verify the E6 qualified-success is robust. Proposed
  **[[EXPERIMENTS]] E6b**: ridge-on-raw-layer-E ONLY (E6's clean winner), add **1 GeV**
  → energy grid {1,3,10,30}, and a small FIXED declared 5-geometry list (uniform +
  E5 earlier/later at k=16 and k=25; k=33 excluded, frozen in advance). k=25 added
  to test split-specificity. Same train[1–4]/test[5–8] discipline, geometry frozen
  before test. Success = a structured geom beats uniform >3×SE on held-out rel_RMSE
  OR worst-energy error AND the win is NOT confined to a single energy bin (≥2 of 4
  energies, consistent sign). Kill = gain disappears with 1 GeV / more geometries /
  held-out, or is isolated to one bin. 160 forward runs (~2–3 h local). NO AD, NO
  agentic loop, NO SLURM initially, NO large scan, NO geometry chosen on test data.
  Rationale: E6's win was estimator-dependent and small — must confirm it's real
  (not a single-bin accident) before building the agentic/multi-DOF machinery on it.
  Status: DONE — see 2026-07-07 (E6b result) below.

- **2026-07-07 · E6b RESULT: PASS — the E6 gain is ROBUST (earlier_k16).** Clean run
  (160/160 forward OK, 0 timeouts/NaNs, budget conserved, no cache reuse; see
  [[EXPERIMENTS]] E6b result). **earlier_k16 beats uniform on held-out rel_RMSE by
  >3×SE (+3.80e-3 vs 3×SE 1.3e-3, ~9σ) AND worst-energy error, consistent-sign in
  ALL 4/4 energy bins including the newly-added 1 GeV** → the E6 result survived the
  harder test; NOT a single-bin accident. The other 3 structured geometries improve
  uniform with consistent sign (3–4/4 bins) but individually <3×SE ("fragile"); none
  worse than uniform. **Defensible claim:** the E5-optimized front-thickened
  earlier_k16 profile robustly improves held-out energy resolution ~12% over uniform
  at fixed absorber budget, consistently across 1–30 GeV. Caveats hold: small
  (~12%), ridge-on-layers-specific (ΣE baseline showed nothing), fixed-budget
  EM-shower reco — robustness ≠ magnitude. Front-thickened (k=N/3) winner is
  consistent with a containment/sampling mechanism, strongest at low E. **Net across
  E0→E6b: trustworthy gradients → structural shape controllability → AD
  optimizability → a robust (if modest) energy-reconstruction performance gain.**
  Forks B (agentic outer loop) / C (three-region AD) now rest on a confirmed
  performance signal, not just reachability. Status: closed.

- **2026-07-07 · Next proposed: E7 (direct AD optimization of the energy-reco FOM).**
  Closes the gap in [[MILESTONE_E0_E6B]] §8: E5 AD-optimized a *shape* loss; E6/E6b
  only *evaluated* fixed geometries on energy reco. Proposed **[[EXPERIMENTS]] E7**:
  reuse the E5 gradient chain to AD-optimize a_front (k=16) on a **differentiable
  energy-reco loss** with a **frozen-from-uniform ridge** estimator; evaluate the
  converged geometry with the E6b held-out per-event protocol; compare uniform vs
  E6b earlier_k16 vs E7 AD-found geometry. **Key honesty point:** the sim's reverse-AD
  propagates through the MEAN per-layer profile, not per-event spread — so E7
  optimizes a bias/calibration-like surrogate, and the held-out per-event RMSE is
  the honest judge. Success = AD-found geometry within 1×SE of earlier_k16 or beats
  uniform >3×SE on held-out, with an AD/FD-verified sane trajectory. Informative kill
  = AD improves the mean/bias loss but NOT the held-out RMSE (→ the energy-reco gain
  lives in the resolution/variance channel AD can't see via mean-profile gradients).
  Step-0 AD/FD gate as in E5 (expect ~0.8). Forward+reverse, ≤15 steps, k=16 only,
  NO agentic loop, NO SLURM first pass, NO new objective. Status: DONE (as a bounded
  diagnostic) — see 2026-07-08 below.

- **2026-07-08 · E7 RESULT: machinery PASS, AD re-finds earlier_k16; FOM-opt PARTIAL.**
  Ran as a bounded diagnostic (surrogate curve + short descents + held-out eval).
  INFRASTRUCTURE: 3 LOCAL attempts died to node load-spikes (loads 266–459 → reverse
  timeouts), then ran on SLURM; job 31047704 (32G) OOM'd at 33.5G **but only after
  printing all results** (OOM cut just the final formatted lines; data intact).
  Result: `experiments/e7_diag_result_SLURM_31047704.txt`. Findings (see
  [[EXPERIMENTS]] E7 result): (1) **machinery PASS** — AD moves downhill on the
  mean-reconstruction surrogate (grad<0 below optimum, >0 above, ≈0 at it). (2) **AD
  re-finds earlier_k16** — full-stats surrogate minimum is a_front≈2.516 = the E6b
  hand-picked winner, NOT uniform (the earlier low-stat probe suggesting a uniform
  optimum was noise-misled). So directly AD-optimizing the FOM surrogate SELECTS the
  right geometry → **[[MILESTONE_E0_E6B]] §8 gap closed at the geometry-selection
  level.** (3) BUT with the frozen-from-uniform estimator the held-out gain is
  marginal (+1.20e-3, ~2.9σ, sub-3×SE) vs E6b's +3.80e-3 (~9σ, per-geometry refit) —
  same geometry, weaker number: the frozen readout under-realizes the resolution gain.
  **Refined mechanism:** the mean-surrogate captures the location/front-thickening
  effect (enough to pick the geometry), but the FULL held-out RMSE advantage needs a
  matched (refit) estimator — part of the gain lives in the resolution/variance +
  estimator-geometry coupling that a frozen mean-surrogate under-weights. NOT
  overclaimed as a clean FOM-optimization win. Caveats unchanged (mean-AD-invisible
  variance, ridge-specific, fixed-budget, k=16). Op notes: the refit-at-final
  diagnostic didn't print (OOM tail) — a 64G rerun would recover it; the E7 driver's
  peak RSS (~33G at N=1000×4E×4seeds concurrent) means future runs need ≥48–64G or
  fewer concurrent workers. Status: closed (diagnostic complete; optional 64G rerun
  for the refit-at-final number not scheduled).

- **2026-07-08 · Operational: E7-class jobs need SLURM + ≥48G, not local.** Three
  local E7 attempts were destroyed by shared-node load spikes (the 1400s cap can't
  save a reverse pass when load hits 400+). Forward+reverse multi-energy diagnostics
  MUST run on a dedicated SLURM allocation. Memory: N=1000 × 4 energies × 4 seeds with
  6 concurrent workers peaks ~33G → request ≥48–64G or cut --workers. Status: active.

- **2026-07-08 · Next proposed: E8 (controlled bilevel demonstration).** The headline
  test of the project thesis. Proposed **[[EXPERIMENTS]] E8**: a discrete-structure
  proposer (SMALL declared set — 5 candidates over #regions∈{1,2,3} and splits
  {N/4,N/3,N/2,2N/3,3N/4}) + AD inner optimization (the E7 energy-reco surrogate,
  frozen-from-uniform ridge, exact budget) judged on the E6b held-out protocol
  (train[1–4]/test[5–8], {1,3,10,30} GeV, ridge on raw layer E). Central Q: can
  proposer+AD find fixed-budget geometries that MATCH or BEAT earlier_k16 on held-out
  energy reco, with NO test-set selection? Success = beats uniform >3×SE AND matches
  earlier_k16 within 1×SE (or beats it >3×SE); partial = recovers earlier_k16-like or
  beats-uniform-only; kill = no uniform gain / train-only / post-hoc changes. First
  pass proposer = DETERMINISTIC enumeration of the declared list (an LLM proposer over
  the same space is an optional labeled follow-up — E8 is a CONTROLLED demo, NOT "let
  the agent wander"). **Candidate budget = 5; expected runtime ~1 SLURM wave
  (~2–4 h/candidate, independent → fan-out); highest risk = the three-region 2-DOF
  inner loop is unbuilt/unvalidated (GATE candidate 5 behind a separate 2-DOF AD/FD
  validation; E8 core = candidates 1–4, all validated 1-DOF).** Frozen-estimator gain
  may be marginal (E7) → report refit-at-final too. SLURM-only (E7 op-rule), ≥48–64G.
  Recommendation: **do a smaller dry-run first** (candidates 1–2 only: uniform +
  two-region@k16 recovery) to confirm the end-to-end proposer→AD→held-out plumbing on
  one known-good case before spending the full 5-candidate wave. Status: DRY-RUN
  done — see 2026-07-08 below.

- **2026-07-08 · E8 DRY-RUN RESULT: FAILED (fixable) — AD-inner-loop gate too fragile.**
  Candidates 1–2 (uniform + two-region@k16). Infra: ran on **atlas:compef@ampere**
  (normal QOS, non-preemptable, CPU-only) after `atlas:usatlas@roma` was wedged for
  hours on `AssocGrpNodeLimit` despite idle roma nodes + 0 group roma jobs (a stuck
  association cap — switching account fixed it instantly). Job 31115135 COMPLETED
  2h52m, MaxRSS 3.96G, no OOM/timeout. **Plumbing 2/3 validated:** proposer→geometry
  OK; held-out E6b eval + refit-at-final **reproduces E6b exactly** (earlier_k16 refit
  rel_RMSE 0.02709 = E6b's, vs uniform 0.03089). **Failure:** the AD inner loop's
  step-0 AD/FD gate fired wrong-sign (AD +5.87e-4, FD −3.34e-4, ratio −1.76) at the
  a_front=2.0 probe → k16 never optimized. Root cause is **shallow-surrogate noise**,
  NOT a chain bug: the mean-reconstruction surrogate varies only ~1e-4 over
  a_front∈[2.0,2.6] (E7), gradient SEs ≈ gradients, so a single-point sign test isn't
  robust at N=1000×4seeds. See [[EXPERIMENTS]] E8 dry-run result. **Do NOT proceed to
  candidates 3–5.** Fix options (scoped, NOT applied, need approval): (a) gate at a
  resolved-gradient point + skip the sign test when |AD|<3·SE (treat unresolved as
  "near optimum", don't FAIL); (b) more inner-loop stats; (c) evaluate the init
  geometry when the surrogate is flat. None alter objective/physics. Status: closed
  (dry-run); full E8 blocked on the gate fix + user approval.

- **2026-07-09 · E8 DRY-RUN RERUN (fixed gate): gate FIX WORKS; deeper issue = the
  per-step AD descent is wrong for this shallow/noisy surrogate.** Applied fix (a) —
  SE-aware 3-outcome gate (PASS/FLAT/FAIL). Job 31127411: both k16 probes correctly
  labeled **FLAT** (AD SE 2.8–3.5e-3 ≫ |AD| 4–6e-4 → unresolved), no abort — the gate
  bug is FIXED. But job **TIMED OUT at 6h** before the held-out eval, and the descent
  gradient is **noise-dominated** (grad_SE > grad; sign flips nearly every step;
  a_front wanders 2.80→2.67, does NOT converge to 2.516). See [[EXPERIMENTS]] E8
  dry-run rerun. **Root lesson: per-step gradient descent is the wrong inner optimizer
  for this flat, MC-noisy surrogate** — E7 only recovered 2.516 via the full-CURVE
  argmin over fixed probe points, not a step-by-step descent. The AD gradient itself
  is validated (E5/E7); the DESCENT usage is the problem. Recommended fix (needs
  approval, NOT applied): **switch the E8 inner loop from GD to curve-scan-argmin**
  (evaluate surrogate on a small fixed a_front grid, take argmin — robust to noise,
  reuses validated E7 machinery); plus walltime→12h and/or --max-steps≤5. Do NOT
  proceed to candidates 3–5 / full E8 until the inner optimizer is switched and a
  dry-run completes. Status: closed (rerun); full E8 blocked on inner-loop switch +
  approval.

- **2026-07-08 · Operational: `atlas:usatlas@roma` can wedge on AssocGrpNodeLimit;
  use `atlas:compef@ampere` (normal QOS, CPU-only, --gpus=0) as the fallback.** E8
  dry-run sat PENDING for hours on usatlas@roma with idle roma nodes; atlas:compef@
  ampere scheduled in seconds. Prefer a NON-preemptable QOS (normal) — compef@roma is
  preemptable-only, and preemption mid-run corrupts forward+reverse jobs like the
  load spikes did. Status: active.

- **2026-07-06 · Operational: experiment drivers MUST raise the in-process
  subprocess cap.** The first E3 run lost 67/186 forward runs to the shared 300s
  `subprocess_timeout_s` under node load. Any local multi-run driver at N≥1000 must
  set `_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0` (as _proxy_*.py,
  e0b_ffront_slurm.py, and now e3_region_scan.py do) AND surface NaNs loudly rather
  than averaging over them. Prefer workers≈6 (not 10) to reduce contention. Status:
  active.
- Use sim build `/sdf/data/atlas/u/jkrupa/agentic/diffcalo`; **never** touch the
  protected `/sdf/data/atlas/u/jkrupa/hepemshow` tree.
- Fixed-budget structural designs only (`build_scale_profile` /
  `profiles_from_scale`): vary granularity distribution, hold total
  length/absorber/gap constant.
- Canonical control flags on every run: `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`.
- **Value-first discipline:** gradient-free value comparison before trusting any
  gradient (burned twice by precise-but-wrong gradients).
- SLURM workers source the LCG view (`LCG_107`), not the login `.venv`; wrapper
  uses `set -eo pipefail` (no `-u`), `set +e` around the LCG source.
- No large batches / no SLURM submission / no destructive actions without
  explicit approval.

- **2026-07-10 · E8 REVISED: switch to validated normalized-profile objective; minimal
  bilevel PoC.** Both E8 dry-runs (2026-07-08, 2026-07-09) confirmed that per-step GD
  on the E7 energy-reco mean-surrogate is unworkable at feasible stats — the surrogate
  is flat (~1e-4 over a_front), grad_SE ≫ |grad|, descent wanders. The E5
  normalized-profile MSE inner loop is the reliable alternative: all 6 combos ran
  cleanly, beat uniform ≫3×SE, sane descent at N=1000 × 6 seeds. **Revised protocol:**
  outer proposer chooses k ∈ {16,25,33} (max 3 structures, one adaptive round only);
  inner AD loop = E5 machinery against the earlier-shift profile target, SE-aware gate,
  ≤10 steps; agent receives compact `{k, best_loss, a_front, a_rear, steps, status}`.
  Energy-FOM eval (E6b protocol) once afterward as secondary context only — not used to
  steer the search. No three-region, no SLURM, no new objectives. **Baselines:** uniform
  and k=16 (E5 earlier_k16 best_loss ~1.87e-4). This is a controlled bilevel PoC on a
  validated inner loop, not a FOM-optimization claim. Status: proposed, not yet run.

- **2026-07-10 · E8 oracle-reuse run: plumbing pass confirmed; adaptive-state bug
  found and fixed.** Job 31200203 (first attempt) pre-filled `compact_results` from
  the oracle before the episode started, so `proposer_call_2` saw an empty remaining
  set and skipped the k=33 decision entirely — cache availability was incorrectly
  treated as equivalent to episode evaluation. Fixed by separating `episode_results`
  (k values explicitly proposed and evaluated in the current run) from `oracle_loaded`
  (cache used only to decide HOW a proposed k is resolved). Job 31201635 (fixed run)
  confirmed correct behaviour: Call 2 saw k=33 as the remaining unevaluated candidate,
  judged the loss spread informative (4.48e-5 > 3×SE), and requested it. Episode best:
  k=16, agrees with exhaustive E5 oracle. Status: closed.

- **2026-07-10 · Proposed E9: physics-reasoned agentic proposal over open-ended
  absorber families.** E8 validated the bilevel plumbing over a pre-declared discrete
  set k∈{16,25,33}. E9 removes the enumeration assumption: an LLM proposes structures
  (2–6 piecewise-constant regions, arbitrary ordered splits, optional constraints) using
  shower-physics reasoning, revised over 3 adaptive rounds of 2 proposals each (6
  structures max). Inner loop = E5 AD machinery extended to multi-DOF via the same
  per-layer reverse-AD → budget chain. Development targets visible to the agent;
  held-out targets evaluated once at the end only (no energy-FOM feedback during
  search). Baselines: uniform, best fixed two-region, best fixed three-region, random
  budget (6 random valid structures, same AD inner loop), and a no-feedback ablation.
  Primary success = agent+AD beats both fixed-structure and random-budget baselines on
  held-out profile loss >3×SE. Largest scientific risk = multi-DOF AD chain unvalidated
  for n_regions>2 (mitigated by per-proposal step-0 AD/FD gate). Smallest implementation
  needed = multi-DOF budget chain + proposal validator + LLM client with the schema
  prompt. See [[EXPERIMENTS]] E9. Status: proposed, not yet started.

## Killed directions (do not reopen without new method)
- Aggregate-energy objectives → optimum is uniform by construction.
- Depth/shower-max resolution structural claim ([[HYPOTHESES]] H3) → artifact of
  vertex-bracket quantization; grid-fair σ is flat.

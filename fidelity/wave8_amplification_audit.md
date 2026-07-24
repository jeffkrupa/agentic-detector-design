# Wave 8, Agent B — dot-amplification audit of the differentiable shower sim

Read-only line-by-line audit, 2026-07-24. No code changes, no builds, no runs.
Serves the wave-8 hypothesis (ledger.jsonl wave 8): severing (`-x/-y/-B`) exists
only to kill spike variance; if the ~1e6 unsevered variance explosion is driven
by a handful of identifiable dot-amplification sites, surgical KeepPrimal caps
at those lines can replace track-killing stop-grads.

Sources audited (working trees, branch `phaseA-perlayer-gap-energy` /
g4hepem `91cbee3`, byte-identical install copies — the compiled physics):

- `/eos/user/j/jeffkrup/agentic/hepemshow/Simulation/src/{Box,Geometry,SteppingLoop}.cc`
- `/eos/user/j/jeffkrup/agentic/g4hepem/install/include/G4HepEm/{G4HepEmElectronManager,G4HepEmGammaManager,G4HepEmElectronInteractionUMSC,G4HepEmElectronEnergyLossFluctuation,G4HepEmRunUtils,G4HepEmElectronInteractionBrem,G4HepEmElectronInteractionIoni,G4HepEmGammaInteractionCompton,G4HepEmGammaInteractionConversion,G4HepEmGammaInteractionPhotoelectric,G4HepEmPositronInteractionAnnihilation,G4HepEmInteractionUtils}.icc`

Physical scales used for worst-case estimates (defaults: 50 layers, 2.3 mm Pb
absorber + 5.7 mm lAr gap, e- @ 10 GeV; physical per-layer derivative scale
~10–200 MeV/mm/seed):

- lAr (gap, the sensed medium): X0 ≈ 140 mm. Restricted brems mfp @ 0.1–1 GeV
  ≈ 1e2–1e3 mm; e+ annihilation mfp ≈ 4e3 mm @ 100 MeV, ≈ 3e4 mm @ 1 GeV;
  restricted ioni/brems mfp → ∞ as Ekin → production cut (mxsec → 0).
- Pb (absorber): X0 ≈ 5.6 mm; all mfps 10–100× shorter; λ_tr1 at ~MeV is
  µm–mm scale (relevant for the MSC z↔t Jacobians).
- Accumulated `numIALeft` dot over a 100-step track: |Σ ṗ/m + p·ṁ/m²| =
  O(1–100) per unit seed (each step's step-length dot is O(0.1–10) and the
  denominators are the small in-material mfps where the path was traveled).

Notation: "dot" = pathwise AD derivative (forward tangent / reverse adjoint —
identical pathwise, per prior waves). "KeepPrimal" = the established pattern
`primal + (x − stop_grad(x))·dfdx_reg` (primal untouched, derivative
coefficient clamped) — see RECON.md §C.

---

## Ranked catalog (rank = expected contribution to unsevered variance × gap relevance)

| # | Site (file:line) | Expression | Dot multiplier, worst case | Fires when | Existing knob | Single-event 1e3–1e6× capable? |
|---|---|---|---|---|---|---|
| S1 | ElectronManager.icc:447 (in loop :439-452; mfp defined :441) | `dStepLimit = mfp*numIALeft[ip]` (ioni/brems/annih), winner assigned to pStepLength :448-456 | `mfp` × accumulated numIA dot: mfp up to 1e3–3e4 mm (lAr brems/annih), **unbounded** near the production-cut threshold (mfp=1/mxsec, :441); plus `numIA·ṁfp` with ṁfp = −ṁxsec/mxsec² unbounded near threshold. Winner dot up to 1e5–1e7 mm/seed | e-/e+ any step where a discrete channel wins the min with small numIALeft; preferentially **in the gap** (lAr mfps ≫ Pb) and at high E (shower front/max) | **NONE** (RAW). Gamma analogue capped by `-C` (GammaManager.icc:75-95,:139); `-A` floors only the numIA *decrement* (icc:162-178,:575-577), not this product | **YES — prime suspect** |
| S2 | Box.cc:126-137 (helper :33-45; called from Geometry.cc:273,316,335,347 every step, SteppingLoop.cc:645/1011) | `t = (copysign(hD,v)-p)/v` per axis, min over axes | `1/vx` per crossing: unbounded raw (vx down to 1e-6+ ⇒ 1e6×); with floor f=0.2 ≤ 5× per crossing but **compounds**: dot(t) → position dot (SteppingLoop.cc:799/1213) → next crossing's numerator ṗ ⇒ ~ (1/vx)^k over k ping-pong crossings, 5^k even when floored | grazing/backward tracks at any layer/gap/abs boundary; ping-pong sustained by the same-boundary re-hits + 1e-6 zero-step pushes (SteppingLoop.cc:793-797/1206-1211). Gap-relevant: *both* faces of the gap scoring bin carry the seed dot (hD = 0.5·fGapThick[i], Geometry.cc:310) | `-V` box-dir-den-floor (default **0=off**; canonical runs *sever* via `-y/-B/-f/-q…` instead) | **YES** (this is what `-y/-B` were built to kill) |
| S3 | ElectronManager.icc:1083-1084 (helper :118-131), :1047-1051, :1089 (helper :133-160); called from UpdatePStepLength :550-557 | g→t: `−λ·log(1−z/λ)` ⇒ dfdz = −1/(λ−z); t→g full model: `par1=(λ−λ1)/(λ·t)`, ratios 1/(par1·λ), 1/(par1·par3); g→t par-pos: `(1−(1−dum)^{1/par3})/par1` | `1/(λ−z)` with λ_tr1(Pb, MeV) ~ µm–mm and z → λ on boundary-limited steps: raw up to ~1e9; with `-N`=1e-3 capped at 1e3. Ratio dfdd = −num/den² with den = λ·t, t → 0: 1/t² | every boundary-limited e± step (gStepLength < fZPathLength), i.e. exactly the steps whose dots carry the geometry seed; mostly low-E electrons in Pb but adjacent to the gap faces | `-N` conversion-reg-eps (derivative-only; canonical `-N 1e-3`). **Unbounded when off** — unsevered forensics must keep `-N` on | **YES when unregularized** (up to 1e9); ≤1e3 with `-N 1e-3` |
| S4 | Systemic feedback loop: SteppingLoop.cc:799/1213 (`AddTo3Vect(globalPosition, curDirection, stepLength)`) + ElectronManager.icc:575-577 / GammaManager.icc:194-196 (numIA integration) | position dot += dot(stepLength); numIA dot −= Σ dot(p/m) | not a single line but the **closed loop** that turns any one-step amplification (S1–S3) into track-lifetime growth: amplified step dot → position dot → boundary numerator (S2) and → numIA dot → re-multiplied by mfp (S1) next steps. Geometric growth per loop iteration | every live track, every step; the reason hard severing "works" — it opens the loop | none (severing `-x/-y/-B` = opening the loop by killing the state dots; SanitizeTrackState :362-373 zeroes the position dot but re-injects prefix dots — waves 1-2) | YES (the compounding mechanism behind S1/S2 spikes) |
| S5 | UMSC.icc:607-654 SampleCosineTheta tail algebra | `1/(parC−2.)` in xmean2Num :647; `1/(1−d)` :648,:651; `1/(1−dumEa)` :619; pow/log sample branches :666,:676,:685-687 | value-guard `parC=2.001` (:614-616) bounds primal but derivative coefficient `1/(parC−2)` up to ~1e3 raw just outside the guard (parC = 2.0011); `1/(1−d)` similar | e± MSC scattering sampling, every non-tiny step in both materials | `-P` umsc-cos-den-floor, `-R` (SimpleScattering :711-719) — default 0=off | marginal (≤1e3, but multiplies direction dots that then feed S2) |
| S6 | UMSC.icc:776-777 SampleDisplacement + SteppingLoop.cc:1318 | `r = 0.73·sqrt((t−z)(t+z))` ⇒ dfdx = 0.5/sqrt(radicand); displacement clip `scale = clipSafety/dispR` | 1/sqrt(t−z) unbounded as t→z (near-straight steps); the clip ratio is bounded (dispR ≥ clipSafety ⇒ net ≤ dot(dispR)) | e± non-boundary steps with displacement on (`-m 1`) | `-S` umsc-disp-rad-floor (default 0=off); clip ratio itself RAW but bounded | marginal alone; feeds position dots |
| S7 | Cancellation-fragile pair sites: UMSC.icc:574-578 (theta0 ∝ sqrt(p/X0), tsmall branch sqrt(p/tsmall)), :607-610 (`log(pStep/(tau·radLength))`), Brem.icc:174-179 (e+ 1/sqrt(e2) → exp(−∞) product), Fluct.icc:84/:90 (`w3/(1−w·u)`, per-sample dot ≤ ṫcut by exact cancellation) | individually singular factors whose *products* cancel exactly in real arithmetic | the singular factor is 1/sqrt(p), 1/(1−wu)² etc., but the analytically-cancelled net dot is O(relative). **Danger: a floor applied to one member of the pair breaks the cancellation and *creates* net dots** | small steps / hard-tail samples | partially covered by `-P/-N` — which is exactly how the cancellation gets broken | no (unless floors are applied inconsistently) |
| S8 | GammaManager.icc:139 gamma step-limit product | `GammaRegularizedStepLimit(mfp, numIA)` | same mechanism as S1; gamma mfp in lAr @ 1–100 MeV is 1e2–1e4 mm | every gamma step | **`-C`** gamma-mfp-cap (canonical 1000) — capped, dfdMfp zeroed when capped (:87) | residual ≤ cap: 1000 mm × numIA dot still allows 1e3–1e4 spikes with `-C 1000` |
| S9 | Compton.icc:42-49 | secondary e- dir = (E_g·d − E'·d')·(1/norm), norm = p_e | 1/p_e up to ~1e2 (threshold 100 eV ⇒ p_e ≥ 0.01 MeV) applied to O(Ė) numerator dots | forward Compton with slow secondary; gamma-rich tail of the shower (deep layers, gap-relevant) | none | no (≤1e2) but feeds descendant position dots |
| S10 | Long tail: PE Sandia `inv=1/ekin` GammaManager.icc:245-246 + Photoelectric.icc:51,:92 (`ac=(1−β)/β`); annihilation in-flight 1/(eps·sqrt(τ(τ+2))) PositronAnnihilation.icc:75-77; conversion `epsp=0.5−0.5·sqrt(1−δmin/δmax)` Conversion.icc:89 and `delta=deltaFactor/(eps(1−eps))` :190/:194; LPM `sqrt(1/(1−redegamma))` InteractionUtils.icc:34; rotate `1/up` RunUtils.icc:54-86; low-E spline edges | assorted 1/x, 1/sqrt | each ≤ 1e1–1e3 and rare; rotate pole is the z-axis (⊥ beam) so `up≈|vx|≈1` for shower tracks | `-U` (PE), `-F` (rotate), none for the rest | no |

Sub-notes:

- **S1 detail — why the raw product spikes in single events.** The winner
  channel satisfies `mfp·numIA < min(other limits)` so the *value* is small,
  but the *dot* is `mfp·dot(numIA) + numIA·dot(mfp)`. A long-mfp channel
  (lAr brems ~1e3 mm; annih ~1e4 mm; near-cut ioni → ∞) wins whenever its
  numIALeft has been decremented near zero — guaranteed to happen at some
  rate over ~1e6 track-steps per 2k-event run. At that moment the accumulated
  numIA dot (O(1–100), integrated with the *short* mfps of the material where
  the path was traveled — the amplification is precisely the ratio
  mfp_now/mfp_then) is multiplied by 1e3–1e4+ mm ⇒ step-length dots of
  1e4–1e6 mm/seed, converted to MeV by dEdx (lAr 0.21, Pb ≥1.3 MeV/mm) and
  by the range-out full-deposit branch (icc:598-603) ⇒ **single-event edep
  dots of 1e4–1e6+ MeV/mm vs the ~1e1–1e2 physical scale**. This matches the
  observed spike-event structure and the 1e6 variance scale.
- **S1 near-threshold divergence** (RECON missed): both dot terms blow up as
  mxsec → 0 at the production-cut threshold (mfp = 1/mxsec :441 raw). The
  gamma cap pattern handles this automatically — when mfp > cap the
  d/d(mfp) coefficient is zeroed (GammaManager.icc:87), killing the
  `numIA·ṁfp` term as well.
- **S2 detail — what is true physics vs artifact.** The *single-crossing*
  1/vx is legitimate (path through a slab of thickness t is t/vx; its
  derivative truly is 1/vx — heavy-tailed but true derivative mass; capping
  it is a bias-variance dial). The *compounded* part — same-boundary
  re-hits, vx sign flips, 1e-6 pushes, positions dots re-entering the
  numerator — is an artifact of the simplified navigation and can be capped
  with far less bias concern. The audit therefore recommends capping the
  *accumulated track-state dot* (proposal 2) rather than only flooring the
  per-crossing denominator.
- **S3 status**: with the canonical `-N 1e-3` every Jacobian coefficient in
  the conversion family is already bounded at ~1e3. For the wave-8 unsevered
  forensics runs `-N` (and `-C`) must stay ON; the unsevered variance
  attributable to S3 beyond that is bounded per step but still feeds S4.
- **Severed-scoring interaction** (context, not amplification): edep is
  scored per-step into discrete bins (SteppingLoop.cc:1442-1479) with
  `edep.setGradient(0.)` only under `-c` (:1449-1453) — in unsevered runs a
  single S1/S2 spike propagates into all subsequent deposits of the track
  and its descendants (`-x 2` semantics off), which is why one bad step
  contaminates a whole event.

## What RECON missed (new findings of this audit)

1. **The `numIA·ṁfp` near-threshold divergence** at ElectronManager.icc:441/:447
   (and the reason the gamma cap's dfdMfp-zeroing (:87) matters): RECON D2
   described only the `mfp × accumulated-dot` term.
2. **Raw pow inside an otherwise-regularized branch**: ElectronManager.icc:1038
   `G4HepEmPow(1.−t/range, fPar3)` is outside any KeepPrimal wrapper (mild:
   par3 > 1 keeps dfdBase finite; dfdPar3 has only a log divergence).
3. **Raw clip ratio** `scale = clipSafety/dispR` SteppingLoop.cc:1318
   (bounded net effect because dispR ≥ clipSafety on that branch, but
   unflagged AD-active division on the transport path).
4. **The value-guard/derivative-gap in UMSC parC** (:614-616): guards bound
   the primal at 2.001/3.001 but leave `1/(parC−2)` derivative coefficients
   up to ~1e3 just outside the guard window.
5. **The cancellation-fragility class (S7)**: several singular factors cancel
   exactly (theta0-sqrt/thex chain; `log(p/(tau·X0)) ≡ log(λ/X0)`;
   fluct-loop per-sample dot ≤ ṫcut; brems e+ correction) — flooring *one*
   member of such a pair (e.g. `-P` on dumEaa but not thex) breaks the
   cancellation and manufactures spurious dots. Any new cap must be applied
   to the *net* quantity, not a factor.
6. **The compounding loop S4 as the object to cap**: a per-crossing floor
   (`-V`) cannot bound 5^k ping-pong growth; only a cap on the accumulated
   track-state dots (position, numIALeft) closes the loop without killing
   the track.

## Bias assessment per site

| Site | Nature of the large dots | Cap consequence |
|---|---|---|
| S1 | **Legitimate heavy tail**: interaction-position sensitivity of a long-mfp channel truly scales with mfp (δ(optical depth) → mfp·δτ position shift). Capping biases the estimator by the mass above the cap — a *bias-variance dial*, exactly like `-C` | bias grows smoothly with (mfp−cap)+; scan the cap |
| S1-threshold | numerics-dominated (1/mxsec² against a vanishing xsec) — near-zero true information content | cap is essentially free |
| S2 single-crossing | legitimate heavy tail (1/vx is the true Jacobian) | floor = bias dial |
| S2 compounding | navigation artifact (same-boundary re-hits, pushes) | cap ≈ bias-free |
| S3 | coordinate-map singularity of the mean-value z↔t transform (the *mean* map's Jacobian at a measure-zero point) — mostly artifact | eps floor low-bias (already the `-N` design) |
| S4 governor | mixes both: caps whatever the upstream site produced | bias only on events whose true derivative exceeds the cap; choose cap ~10–100× physical scale |
| S5/S9/S10 | bounded; mixed | low stakes |
| S7 | net dots are small and true; the *danger* is cap-induced bias | do NOT floor pair members independently |

## Top-3 cap proposals (ready for implementation, all default-off, derivative-only, KeepPrimal)

1. **`--el-mfp-cap <mm>` — electron mirror of `-C`** (site S1; = RECON D2).
   Copy `GammaRegularizedMfpProductKeepPrimal` (GammaManager.icc:75-91)
   into the ElectronManager anonymous namespace, use it at
   ElectronManager.icc:447 (`dStepLimit = ElRegularizedStepLimit(mfp,
   theTrack->GetNumIALeft(ip))`), add
   `G4HepEmElectronManager::ConfigureMfpCapRegularization` + one CLI flag in
   InputParameters.hh + wiring in HepEmShow.cc:204-209. Semantics identical
   to `-C`: when mfp > cap, d/d(mfp) zeroed (also kills the near-threshold
   ṁfp divergence) and d/d(numIA) clamped to cap. **Suggested scale:
   100 mm** (≈12 layer pitches ≈ 0.7 X0_lAr; the value-scale where a step
   limit stops being locally meaningful in an 8 mm-pitch stack); scan
   {10, 100, 1000} mm against the 1M FD truth. ~40 lines, mechanical,
   primal bit-identical when off. Decisive wave-8 test: unsevered
   (`-x0 -y0 -B0`) + this cap + `-C 1000 -N 1e-3` should collapse the
   spike tail if S1 dominates.
2. **`--track-dot-cap <val>` — track-state dot governor** (sites S2+S4).
   Bounded-cap replacement for the `-y/-B` kills: after the position update
   (SteppingLoop.cc:799/:1213) clamp the derivative part of each position
   component to ±cap_pos, and after UpdateNumIALeft
   (ElectronManager.icc:575-577, GammaManager.icc:194-196) clamp the
   derivative part of each numIALeft to ±cap_numia. KeepPrimal trivially
   (values untouched; forward mode: clamp GET_DOTVALUE; reverse mode:
   implement as `x = stop_grad(x) + clamp_dot(x − stop_grad(x))` via a
   custom clamp on the tangent/adjoint — for reverse this needs the small
   CoDiPack external-function helper, or first validate forward-only).
   **Suggested scales**: cap_pos = N_layers × max|seed| ≈ 50 (the true
   |∂x/∂θ| of a transported point under a uniform gap seed is ≤ 49) × a
   grazing allowance 1/f = 5 ⇒ **250 mm/seed**; cap_numia = **100**
   (≈ calo thickness / shortest gap mfp). This is the only proposal that
   provably bounds the S4 compounding loop (5^k → ≤ cap) while keeping
   every track alive and scoring. ~30 lines in SteppingLoop.cc + 6 in the
   two managers.
3. **`--boundary-dot-cap <mm/seed>` — capped Box distance** (site S2,
   per-crossing; complement to 2, or fallback if the reverse-mode governor
   is deferred). In Box::DistanceToOut wrap the returned `tmax`
   (Box.cc:137-139) as `primal + clamp(corr, ±cap)` where
   `corr = tmax − stop_grad(tmax)` (KeepPrimal on the *net* result — one
   cap after the min, so no cancellation-pair breakage, unlike flooring
   vx/vy/vz separately). **Suggested cap: 8 mm (layer pitch) / f(=0.2)
   = 40 mm/seed** per crossing. Together with `-V` left OFF this preserves
   the exact 1/vx Jacobian for all normal crossings and only truncates the
   grazing tail. ~15 lines in Box.cc + flag.

Recommended forensics configuration for the wave-8 unsevered runs:
`-x 0 -y 0 -B 0` + `-N 1e-3 -C 1000` (keep the already-bounded S3/S8
regularizers) + proposals 1(+3) — then measure the per-event dot
distribution; if the spike census attributes the residual tail to S4
compounding, add proposal 2. Predicted signature if the wave-8 reframe is
right: variance within 1e1–1e2 of canonical (not 1e6) and absorber core
AD/FD moving 0.78 → ~1.

Minor watch-list for the spike miner (cheap to instrument, not worth caps
yet): S5 parC-window events, S6 near-straight displacement steps, S9 slow
Compton secondaries, Fluct.icc:64 `alfa·log(alfa)/(alfa−1)` 0/0 numerics
at alfa → 1.

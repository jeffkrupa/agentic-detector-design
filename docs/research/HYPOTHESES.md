# HYPOTHESES (ledger)

Status tags: **tested** (evidence exists) · **inferred** (indirect support) ·
**speculative** (no direct evidence). Each hypothesis has a minimal experiment
and a **kill criterion** — the observation that would falsify it.

Related: [[PROBLEM]], [[EXPERIMENTS]], [[DECISIONS]].

---

## H1 — e/γ PID rewards front-end granularity (structure helps classification)
- **Claim:** A design that concentrates granularity at the shower front
  separates e⁻ from γ better than an equal-budget uniform design.
- **Status:** *tested — mostly refuted as a structural effect.* Native CV-AUC
  ordered fine>uniform>coarse (0.88/0.84/0.79 @10 GeV; 0.92/0.85/0.77 @1 GeV),
  but the grid-fair control shrinks the true effect to **~+0.003 AUC** and it did
  **not** grow at 1 GeV. Most of the native gap is a front-window-depth confound.
- **Minimal experiment:** [[EXPERIMENTS]] E1 — grid-fair CV-AUC, fine vs uniform,
  ≥6 seeds, at 1 GeV (where any front-granularity effect should be largest).
- **Kill criterion:** grid-fair AUC gap ≤ 0.005 and not increasing at low energy
  → structure does not meaningfully help PID. (Currently ~at kill threshold.)

## H2 — The AD gradient is genuinely biased (~10–30×) vs finite difference
- **Claim:** Exact AD gradients on this sim are wrong by 10–30×, so the exact-AD
  bilevel story is unsound.
- **Status:** *tested — KILLED (2026-07-03).* E0 validated FD on the clean
  d(total_edep)/dE channel: with CRN and resolved ε (50–200 MeV) FD plateaus at
  AD/FD ≈ 1.0 — no 10–30× gap. The earlier proxy AD/FD ~0.03–0.1 was an FD
  artifact (ε too small / 2nd-order MC noise), not a real gradient error. AD is
  trustworthy. See [[EXPERIMENTS]] E0 result and [[DECISIONS]] 2026-07-03.
- **Minimal experiment:** [[EXPERIMENTS]] E0 — validate FD machinery on a LINEAR
  control (total_edep: AD d/da≈-1310, d/dE≈+0.775) via ε-scan with common random
  numbers at N=2000 and 10000; expect an AD/FD≈0.8 plateau. Then re-run proxy D
  FD with that ε + CRN + high N.
- **Kill criterion:** on the linear control, AD/FD plateaus near 0.8 with a clean
  ε window → FD machinery is sound and the "10–30× bias" claim is killed; gradient
  is trustworthy.

## H3 — Depth/shower-max resolution rewards non-uniform layering
- **Claim:** Non-uniform layer spacing improves longitudinal (shower-max / depth)
  resolution vs uniform.
- **Status:** *tested — refuted (artifact).* Apparent ordering came from parabolic
  vertex-bracket quantization + extensive-edep argmax bias. Grid-fair σ is
  identical (5.30/5.33/5.32 mm). Depth resolution is flat across designs.
- **Minimal experiment:** none — already killed. Reopen only with a
  quantization-free estimator.
- **Kill criterion:** (already met) grid-fair σ differences < measurement noise.

## H4 — A resolution/variance-shaped objective (not aggregate energy) favors structure
- **Claim:** Objectives sensitive to the *shape/variance* of the longitudinal
  profile (e.g. inverse-variance Fisher separation, energy-resolution at fixed
  budget) can favor structured granularity where aggregate-energy objectives
  cannot.
- **Status:** *inferred.* The inverse-variance Fisher **proxy** reproduced the
  value ordering fine>uniform>coarse, while plain mean-distance did not (ranked
  coarse first). Suggests the *variance-weighting* is what carries any structural
  signal — but this is a proxy, not the full sim, and the effect may be the same
  small PID effect in disguise.
- **Minimal experiment:** [[EXPERIMENTS]] E2 — in the full sim, compare a
  variance-aware objective (inverse-variance Fisher) vs a mean-only objective
  across fine/uniform/coarse, grid-fair, multi-seed.
- **Kill criterion:** variance-aware objective shows the same ≤0.005 grid-fair
  gap as the mean-only one → variance-shaping adds nothing; H4 folds into H1.

## H5 — Gradient reliability itself is the publishable contribution (fallback)
- **Claim:** Even if structure barely helps, the recurring "precise-but-wrong
  gradient" pattern (high SNR ≠ correct) plus a reliability layer
  (gradient SNR + validated AD-vs-FD gate) is a real, demonstrable result.
- **Status:** *inferred.* Precise-but-wrong has appeared 3× (net-signal GD,
  depth-res quantization, proxy 10–30×). Needs E0 to fix what "correct" means
  before it can be framed as reliability rather than noise.
- **Minimal experiment:** depends on E0; then quantify how often the reliability
  flag would have caught each of the 3 past artifacts.
- **Kill criterion:** after E0, all three artifacts turn out to be trivially
  avoidable (e.g. just an ε typo) → no general reliability story, only bugs.

---

## Dependency note
E0 (H2) is upstream of almost everything: until FD is validated we cannot tell
whether AD gradients, the proxy, or the reliability story are real. E1 (H1) is
independent of E0 and directly tests the strongest current structural signal.

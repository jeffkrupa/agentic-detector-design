# Optimization campaign plan — "wrong gradients build a wrong detector"

Draft v1, 2026-07-28. The payoff experiment for the gradient-repair program:
run the *same* gradient-based detector optimization twice — once with the
published-flag gradients (gap 3× inflated), once with the repaired gradients
— and let a gradient-free primal ground-truth scan adjudicate which design is
real. The thesis the whole program earns: **gradient fidelity is not a
numerical nicety; a 3× error in one gradient component steers an optimizer to
a measurably different, worse detector.**

Foil / context: arXiv:2604.21804 puts an LLM agent in a differentiable
simulator and never uses the derivatives. We use them — and show that using
the *wrong* ones (the simulator's out-of-the-box gradients under the paper's
own flags) produces a wrong design, which is the concrete argument for why the
repair program mattered.

---

## 0. What the trust map lets us do (and not do)

Established, validated (see fidelity/LEDGER.md, waves 1–14):

| gradient | status | mode | usable for optimization? |
|---|---|---|---|
| d(μ_i)/dE (energy) | exact (≈1.00) | fwd + rev | yes, per-layer |
| d(μ_i)/dg (gap thickness) | **repaired, per-layer real** (3.02→1.08), unsevered + boundary-dot-cap ∝ g | **forward only** | yes, per-layer, Fisher-safe |
| d(μ_i)/da (absorber thickness) | severed × 1.30 (0.77 uniform undershoot, per-layer honest, mild drift not excluded) | forward | yes for shape; scale good to ~5–7% |

Two hard constraints this imposes on the campaign:

1. **Forward mode only.** The repaired gap cap is a tangent clamp with no
   linear transpose (wave 9) — it does not exist in reverse mode. So the
   "all-gradients-in-one-reverse-pass" payoff is NOT available for the repaired
   estimator. Each design-parameter gradient is a separate forward run (seed
   that one input). This bounds the design dimensionality we can afford.
2. **No dσ/dθ.** The adjoint interface gives var_dE (variance *of* the
   derivative), not d(variance)/d(design). So objectives that need the
   derivative of the per-layer *noise* w.r.t. design (most energy-resolution
   objectives) are out of reach at first order. Objectives must be functionals
   of the per-layer *mean* response μ_i(θ), whose θ-gradient is exactly what we
   validated.

---

## 1. Primary experiment — gap-profile response shaping

The cleanest, most affordable, most dramatic showcase: optimize the **gap
thickness profile only** (absorber fixed uniform), because gap is the channel
that is simultaneously (a) 3× wrong under published flags and (b) genuinely
repaired per-layer — so broken-vs-repaired is maximally separated, and no
absorber correction factor enters.

**Design space.** Group the 50 layers into K = 5 longitudinal zones; each zone
z has one gap thickness g_z (10 layers/zone). Constraint: fixed total gap
Σ_i g_i = 50·5.70 mm (pure redistribution of sensitive material — no free
lunch from adding material). Absorber held uniform at 2.30 mm. That is a
4-DOF constrained space (5 zones − 1 constraint), each gradient = 5 forward
runs (one per zone's gap seed, via the per-layer `--gap-layer` seeding).

**Objective (first-order, mean-response only).** Response-profile shaping:
maximize J(θ) = −Σ_i w_i (μ_i(θ) − t_i)² toward a physically-motivated target
profile t_i. Candidate targets (pre-register one; the choice must guarantee an
interior optimum, not a boundary):
  - **equalized sampling** t_i = const (drive energy-per-gap-layer flat — a
    calibration/compensation goal; nontrivial because showers peak, so the
    best-match redistribution puts more sampling in the tail);
  - or a **shower-max-centering** target that rewards a specified longitudinal
    centroid of the sampled energy.
dJ/dθ = 2 Σ_i w_i (t_i − μ_i) ∂μ_i/∂g_z — a weighted sum of exactly the
per-layer gap derivatives we repaired. Under published flags each ∂μ_i/∂g_z is
~3× too large → the optimizer believes gap moves are 3× more effective than
they are → it over-steps and mis-weights the zones → converges to a different
profile. Under the repaired gradient → correct.

**Why this survives the "structure helps nothing" prior.** That negative
result (memory: project-state-2026-07) was about *aggregate* observables being
pinned by fixed totals and swamped by intrinsic fluctuations. Response-profile
*shaping* is explicitly a per-layer functional; redistributing gap demonstrably
reshapes μ_i, so the objective has a real, primal-findable interior optimum.
The pre-check below confirms this before any compute is committed.

---

## 2. Mandatory pre-check (cheap, before the full campaign)

The single biggest risk is a vacuous demo: if the 3× gap inflation does NOT
displace the optimum far enough to matter, the broken and repaired optimizers
land in the same place and there is no result (this is the trap that stalled
the old E3 plan). Kill or confirm it for ~15 CPU-h before spending the campaign:

- Evaluate the true objective gradient (repaired) and the published-flag
  gradient at ~5 points spanning the design space (including the uniform
  starting profile and 2–3 displaced profiles).
- Compute, at each point, the angle between the repaired and published-flag
  gradient vectors in the 4-DOF constrained space, and the implied
  displacement of the stationary point (Newton step under a cheap curvature
  estimate).
- **Acceptance to proceed:** the two gradients point in materially different
  constrained directions (angle ≳ 15°) AND the predicted optimum displacement
  exceeds the primal-scan resolution. If not, escalate the zone count or switch
  the target profile to one where the gap direction is more load-bearing —
  before the full run.

---

## 3. Ground truth (gradient-free, the adjudicator)

The whole demonstration rests on an independent truth. Primal-only, no AD:

- **Map J over the 4-DOF space** by Latin-hypercube sampling (~200–400 designs)
  at n = 20k events each, then refine the best region with a gradient-free
  optimizer (Nelder–Mead or CMA-ES) on primal J evaluations to locate the true
  optimum θ*_true and its uncertainty.
- Re-evaluate θ*_true, θ*_repaired, θ*_published at high statistics (200k
  events) so the three designs' *true* J values carry real error bars.
- **The headline result:** J(θ*_repaired) consistent with J(θ*_true), and
  J(θ*_published) significantly worse — with the *design vectors* θ*_published
  vs θ*_true visibly different (a plot of the two gap profiles). That is
  "wrong gradients → wrong detector," quantified.

---

## 4. The two gradient-optimizer runs

Identical projected-gradient-ascent optimizers on the constraint surface,
differing ONLY in the gradient source:

- **Published-flag optimizer:** gradients from canonical severed flags
  (-x 2 -y 1 -B 1 …), the simulator's out-of-the-box AD — gap ~3× inflated.
- **Repaired optimizer:** gradients from unsevered + boundary-dot-cap ∝ g
  (forward, per-layer validated).

Both: ≤ 30 iterations, 3 restarts (uniform + 2 displaced starts), gradient =
5 forward runs × 20k events per step, line search on primal J. Log full
trajectories. Same step-size schedule, same convergence criterion, same seeds
— the ONLY difference is the gradient, so any divergence in the endpoint is
attributable to gradient fidelity alone.

---

## 5. Compute budget & condor layout

Per gradient evaluation = 5 forward runs (5 gap-zone seeds) × 20k events.
Per optimizer ≈ 30 steps × 3 restarts × 5 runs = 450 sims ≈ 9M events.

| phase | content | ~events | ~CPU-h |
|---|---|---|---|
| Pre-check | 5 points × (repaired+published grad) | 1M | 15 |
| Ground-truth map | 300 LHC designs × 20k + CMA refine (~150) | 9M | 130 |
| Published optimizer | 450 sims × 20k | 9M | 130 |
| Repaired optimizer | 450 sims × 20k | 9M | 130 |
| High-stat re-eval | 3 designs × 200k (+ per-layer) | 0.6M | 8 |
| **Total** | | **~29M** | **~415** |

All condor, forward mode, unsevered+cap binary (knob/dot-caps fc388aa) for the
repaired arm and the shared baseline binary for the published arm, per-(design)
jsonl outputs, 10800 s timeout, AFS submit. Phased; each phase gated on the
previous (pre-check must pass before ground truth; ground truth before the
optimizer runs). **Every phase needs explicit approval before submission.**

---

## 6. Extension experiments (after the primary lands)

1. **Joint (a, g) profile.** Add per-zone absorber, using the severed × 1.30
   absorber gradient. 8–10 DOF, ~2× the forward runs per step. Tests whether
   the absorber's ~5% scale uncertainty and possible mild drift matter to the
   design; also a richer broken-vs-repaired contrast (both channels wrong under
   published flags, one 3× and one 0.77×).
2. **Energy-resolution / Fisher objective** — flagged as future work: needs
   d²μ/dEdθ (mixed second derivative) and/or dσ²/dθ, neither available at first
   order through the current adjoint interface. Requires forward-over-reverse or
   a differentiable variance estimator — a genuine extension of the simulator,
   out of scope for v1. Do NOT promise a resolution result the interface can't
   support.
3. **Agent-in-the-loop.** Wrap the repaired optimizer in the self-auditing
   agent: it runs the gradient SNR + AD-vs-FD spot check (tools/reliability.py)
   *before* trusting each gradient, and would itself flag the published-flag
   gradient as untrustworthy — closing the loop with arXiv:2604.21804 (agent
   that ignores gradients) vs ours (agent that audits, then uses them).

---

## 7. Risks & honesty ledger

- **Vacuous demo** (optimum not displaced): mitigated by the §2 pre-check;
  do not run the campaign until it passes.
- **Forward-only cost**: dimensionality is bounded by "one forward run per
  DOF"; K = 5 gap zones is the affordable sweet spot. State plainly that the
  reverse-mode one-pass payoff is future work.
- **Absorber scale/drift**: kept OUT of the primary (gap-only) experiment;
  enters only the extension, with the ×1.30 and its ~5% band quoted honestly.
- **dσ/dθ unavailable**: forecloses resolution objectives at first order —
  scope the primary to mean-response shaping and say why.
- **"Structure helps nothing" prior**: the objective is a per-layer shape
  functional with a pre-verified interior optimum, not an aggregate — the
  pre-check is the guard.
- **Provenance**: every design's gradient records the binary git-rev + cap/flag
  set (tools/sim.py provenance keying), so the published vs repaired arms can
  never be confused in the results.

## 8. What "done" looks like

One figure: the true objective landscape (primal), with the published-flag
optimizer's trajectory ending at a wrong optimum and the repaired optimizer's
ending on the true one; a companion panel of the two final gap profiles side by
side; and a table of J(θ*) for published / repaired / true with error bars. One
sentence of result: *the simulator's out-of-the-box gradients, 3× wrong in the
gap direction, drive the optimizer to a design whose true objective is worse by
[X]σ; the repaired gradients recover the true optimum.*

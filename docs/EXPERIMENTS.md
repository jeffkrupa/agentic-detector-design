# Experiments

The experiment matrix that turns the scaffold into a paper. Start every arm at
**small scale** (`-n 1000..2000`, ≤8 seeds) to validate plumbing, then scale.

## E0 — Tool & ground-truth validation (prerequisite)
- `tools.sim --selftest` passes against forward + reverse binaries.
- AD vs finite-difference agree (within error) for `total_edep` w.r.t. `a` and
  `E` at a benign design point → confirms ground truth and the `cross_check`.
- Reproduce a known physics sanity check: shower-max depth increases ~logarithmically
  with `E`; `∂(total_edep)/∂E > 0`; thicker absorber → earlier shower max.

**Deliverable:** a one-page validation note + the FD/AD agreement plot.

## E1 — Sensitivity-reasoning benchmark (headline)
See `BENCHMARK.md`. Run closed-book vs grounded for ≥2 frontier models.
- **Metrics:** sign accuracy, magnitude MAE (dex), within-1-dex, ECE, all split by
  reliability flag.
- **Figures:** per-observable sign-accuracy bars; calibration curves; the
  `untrusted`-slice breakout.

**Claim:** first measurement of LLM physical-sensitivity reasoning; models are
miscalibrated in the low-reliability regime; tool grounding helps.

## E2 — Constrained design loop (agent vs baselines)
Task: reach a target longitudinal profile / resolution under NL constraints
(e.g. "minimize channels while keeping containment > 90% and sampling fraction
> 3%").

| Arm | Optimizer | Tests |
|-----|-----------|-------|
| A | Agent + **exact reverse-AD** gradient tool | the proposed method |
| B | Agent + **finite-difference** tool | value of *exact* vs noisy/expensive grads |
| C | **Plain projected gradient descent** (no agent) | value of the agent (NL constraints, trap escape) |
| D | Agent **without** gradient tool (physics priors only) | grounding effect |

- **Metrics:** loss/objective vs compute budget (events used); #constraint
  violations; #steps to target; final design quality; **predicted-vs-realized ΔO
  honesty** (does the agent's gradient-based prediction match the measured
  change?).
- **Figures:** objective vs budget for A–D; honesty scatter (predicted vs realized).

**Claim:** A converges with less compute than B/C and, unlike D, makes
physically-correct moves; A can satisfy NL constraints that C cannot express.

## E3 — Reliability-aware judgment (the "physics judgment" result)
Construct/identify design regions where raw AD is `untrusted` (grazing/boundary
discontinuities, near-singular MSC conversion — the unregularized CRE regime).
- **Arm A+** (reliability policy on) vs **A−** (policy off, trusts AD blindly).
- **Metrics:** rate of trusting bad gradients; resulting bad steps; recovery via
  FD fallback; final design quality in these regions.

**Claim:** an agent that consults the reliability flag avoids the catastrophic
steps that blind-AD agents take — operationalizing "physics-motivated judgment."

## E4 — Discovery / explanation (stretch, optional)
Let the Sensitivity-Analyst form and test hypotheses about scaling laws
(shower-max ∝ ln E; sampling-fraction vs absorber/gap ratio) using gradient
queries, and report which it confirms. "AI-scientist" framing.

## Ablations
- Statistics: gradient SNR and benchmark metrics vs `-n` (events).
- Reliability threshold sweep (what SNR / FD-disagreement defines `untrusted`).
- Model size / family on E1.
- Query budget in the grounded benchmark condition.

## Compute discipline
- Cache everything (`tools/sim.py` flag-hash cache).
- Dev at `-n<=2000`; only scale the *final* numbers.
- No SLURM from agent code without explicit approval; the existing
  `../jobs/submit_many.sh` is the human's tool for big grids.

---

## Log
_Append dated notes here as experiments run (keep it terse)._

- (template) `YYYY-MM-DD` E0 selftest: PASS/FAIL; AD-vs-FD rel.err = …; notes …

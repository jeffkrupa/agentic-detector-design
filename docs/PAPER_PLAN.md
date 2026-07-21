# Experiment plan — "An agent that audits the gradients before using them"

Draft v1, 2026-07-21. Merged paper: self-auditing gradient agent (spine) +
tolerance/uniformity budget (payoff section) + energy-calibration positive
control + head-to-head vs the gradient-free agentic co-design baseline.

Foil / nearest prior work: arXiv:2604.21804 (agentic co-design in a
differentiable framework, **no gradients ever used** — verified against full
text). Our contribution: an agent that actually consumes the exact physical
derivatives, *after* measuring where they can be trusted.

---

## 0. Claims the paper makes (each mapped to experiments)

| # | Claim | Experiments |
|---|-------|-------------|
| C1 | The simulator's AD gradients have a measurable, structured reliability landscape ("trust map"): energy ≈ exact, absorber ≈ 0.8× (uniform through shower core), gap ≈ 2–3× inflated (depth-dependent, sign-correct) | E1 (partly **done**) |
| C2 | The biases are mechanistically attributable: boundary-moving parameters (absorber, gap) are biased pathwise; non-boundary (energy) is exact; stop-grad/regularization flags control the bias level | E2 |
| C3 | Blind trust in raw gradients drives an optimizer to a wrong design; the true landscape (primal-only ground truth) proves it | E3 |
| C4 | A reliability-gated agent gets the payoff safely: one reverse pass → per-layer tolerance budget, validated by FD spot checks it chose itself; ~100× fewer sim calls than an FD table | E4 |
| C5 | Positive control: the energy gradient supports quantitative extrapolation (calibration transfer) with no correction | E5 |
| C6 | The audited-gradient agent beats a gradient-free agentic scan (2604.21804-style) in simulator calls on the same design task | E6 |
| — | Sidebar: agents must audit estimators too — depth-resolution "win" was estimator variance-shrinkage bias | **done** (depthres autopsy) |

Baseline design point everywhere unless stated: uniform 50 layers,
a = 2.30 mm, g = 5.70 mm, 10 GeV e-, t = 400 mm, canonical flags
`-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`. Runtime ≈ 0.187 s/event nominal
(thicker absorber is slower — the a=2.32 FD+ config hit 2 h at 20k events on
slow nodes).

Condor conventions (proven by cluster 3911355): submit from AFS cwd,
`workday` flavour, wrapper in `/afs/cern.ch/user/j/jekrupa/condor_bin/`,
one output jsonl per (config, seed) unit under `experiments/<study>/`,
idempotent runner with `--analyze` aggregation mode (pattern:
`experiments/perlayer_adfd_1M.py`). Job sizing rule from the timeout
post-mortem: **≤ 20k events/job at a ≤ 2.5 mm, ≤ 10k events/job at
a > 2.5 mm or E > 10 GeV**; keep the 7200 s in-process sim timeout.

---

## E1 — Trust map (C1). Status: core result DONE, three gaps to close

**Done (cluster 3911355, `experiments/perlayer_adfd_1M_summary.json`):**
per-layer d(E_layer)/da and /dg at the baseline point, 1M events/config,
CRN-paired central FD h = ±0.02 mm. Absorber plateau AD/FD = 0.77–0.80
(k = 0.754 ± 0.007), gap ~2–3× depth-dependent (k = 2.67 ± 0.10,
χ²/ndof = 385/42), L49 gap outlier parked per user direction.

**E1a — repair absorber pairing.** Rerun the 32 timed-out units
(20 absorber_fd_plus, 7 fd_minus, 4 absorber_ad, 1 gap_fd_plus) as 2 × 10k-event
sub-jobs each, restoring 50 → ~46+ paired absorber FD seeds.
*Cost: 32 × 20k = 0.64M ev ≈ 35 CPU-h.*

**E1b — energy row of the trust map.** Same per-layer treatment for
d(E_layer)/dE: forward `-e 10000:1`, FD at E = 10000 ± 20 MeV, CRN,
seeds 1–10 × 20k = 200k ev per config (the gradient is clean; e0 showed
AD/FD = 0.995–1.03 at n = 1000, so 200k is overkill-safe, not exploratory).
*Cost: 3 × 200k = 0.6M ev ≈ 31 CPU-h.*

**E1c — FD step-size scan (kills the "FD is wrong, not AD" rebuttal).**
Gap only (the contested direction): h ∈ {0.005, 0.01, 0.05, 0.1} mm
(h = 0.02 exists), ± sides, 10 seeds × 20k = 200k ev per side.
Acceptance: layer-summed FD stable to < 10% across h; then the 2–3× gap is
attributable to AD.
*Cost: 4 h-values × 2 sides × 200k = 1.6M ev ≈ 83 CPU-h.*

**E1d — generality beyond one design point.** Repeat the 6-config AD/FD
comparison at 200k ev/config (totals + coarse layer bins, not full per-layer
precision) at TWO more points: (a = 1.5, g = 3.0) and (a = 3.5, g = 8.0),
10 GeV. Acceptance: absorber ratio in 0.7–0.9 and gap ratio in 1.5–4 at both
points → trust map is a property of the parameter, not the point.
*Cost: 2 points × 6 × 200k = 2.4M ev ≈ 125 CPU-h.*

Figure F1: heatmap, parameter × layer, AD/FD with significance masking —
the paper's centerpiece.

---

## E2 — Mechanism axis (C2)

**E2a — stop-grad / regularization flag scan.** The FD truth from E1 is
flag-independent (flags only touch the derivative path), so scan **AD only**
against the existing E1 FD baseline. One-factor-at-a-time from canonical:

| Scan | Values (canonical bold) |
|------|------------------------|
| `-x` stop-grad-mode | 0, 1, **2** |
| `-y` grazing-stop | 0, **1** |
| `-B` backward-boundary-stop | 0, **1** |
| `-N` CRE | 0, 1e-4, **1e-3**, 1e-2 |
| `-C` gmc | 100, **1000**, 10000 |

= 10 non-canonical configs × (absorber-AD + gap-AD) × 200k ev
(10 seeds × 20k). Output per config: layer-summed and shower-core AD/FD
ratio ± SE, NaN/outlier event counts (NaN rate is itself a result — the CRE
study's tradeoff: bias vs variance/NaNs).
Deliverable: "bias attribution" table/figure (F2) — which severing mechanism
buys which part of the 0.8× absorber and 2–3× gap factors, and at what NaN
cost. This is the paper's *why*, and no prior work has it for gap gradients.
*Cost: 20 configs × 200k = 4M ev ≈ 208 CPU-h.* (Trim option: drop `-C`/`-y`
rows → 12 configs ≈ 125 CPU-h.)

**E2b — boundary vs non-boundary thesis.** No new runs: assembled from
E1b (energy exact, moves no boundary) vs E1a/E1c-d (absorber & gap biased,
both move material boundaries) + the depth profile (bias grows where
boundary-crossing track density peaks). Forward=reverse equality (already
established to all printed digits) closes the "reverse-path bug" alternative.

---

## E3 — Negative demo: blind trust corrupts a design (C3)

Task: maximize visible energy fraction f_vis = E_gap,tot / E_beam over
(a, g) with the material/total-length constraint 50·(a + g) = 400 mm
(1-D feasible line, parameterized by a). Objectives stay **linear in mean
layer energies** — that's what the adjoint interface differentiates.

1. **Ground truth landscape (primal only, no gradients):** scan a from 0.5 to
   4.5 mm in 0.25 mm steps (17 points), g = 8 − a; 5 seeds × 20k = 100k
   ev/point. Locate the true optimum ± uncertainty.
   *Cost: 1.7M ev ≈ 88 CPU-h.*
2. **Blind-gradient optimizer:** projected gradient ascent on the constraint
   line using raw reverse-mode df_vis/da − df_vis/dg, 20k ev/iteration,
   ≤ 30 iterations, 3 restarts (a₀ = 1.0, 2.3, 4.0). Prediction: the 2–3×
   inflated gap direction drags the iterate off the true optimum / pins it to
   a bound.
   *Cost: ≤ 90 × 20k = 1.8M ev ≈ 94 CPU-h.*
3. **Audited optimizer (teaser for E4/E6):** same loop but gradients pass
   through `tools/reliability.py` gating — gap component deflated by the
   trust-map factor (or FD-refreshed every 5 steps where flagged
   `untrusted`). Should land on the true optimum in ≤ half the primal cost
   of the scan.
   *Cost: ≈ 60 × 20k + 10 FD refreshes × 40k = 1.6M ev ≈ 83 CPU-h.*

Figure F3: trajectories of (2) and (3) overlaid on the true landscape (1).

---

## E4 — Positive demo: one-pass tolerance budget (C4)

Physics task: per-layer mechanical tolerance allocation. Given a total
response-shift budget |ΔE_vis|/E_vis ≤ 0.5%, allocate per-layer absorber
thickness tolerances δaᵢ (uniformity/constant-term budget framing — d(σ/μ)
is *not* available from the adjoint interface; do not promise it).

1. **One reverse pass:** per-layer adjoints on E_vis → d(E_vis)/d(aᵢ) and
   d(E_vis)/d(gᵢ) for all 50 layers at once. 50 seeds × 20k = 1M ev (need
   per-layer SNR; the gap-locality probe showed the profile is sign-changing:
   −28 / −2.1 / +33 MeV/mm at layers 10/20/35).
   *Cost: 1M ev ≈ 52 CPU-h.*
2. **Agent audit step:** reliability layer flags each of the 100 components
   (SNR + trust-map class). Absorber components → usable with the 0.80
   plateau correction; gap components → sign-only (rank ordering), FD spot
   checks where the budget decision is sensitive.
3. **FD validation table (chosen BY the agent, not exhaustively):** ~8 layers
   (front / max / tail / sign-change region) × absorber ± 0.02 mm, CRN,
   25 paired seeds × 20k = 500k ev/side.
   *Cost: 8 × 2 × 500k = 8M ev — too much; use 10 seeds × 20k = 200k/side
   → 3.2M ev ≈ 167 CPU-h. Trim option: 5 layers ≈ 104 CPU-h.*
4. **Cost comparison (headline number):** exhaustive FD table = 100 params
   × 2 sides × 200k ev = 40M ev (~2100 CPU-h) vs delivered: 1M (reverse)
   + 3.2M (spot checks) ≈ **10× fewer events, one pass for all 100
   sensitivities**; scales with #params, FD doesn't.

Figure F4: per-layer sensitivity profile with error bars + allocated
tolerances + spot-check agreement markers.

---

## E5 — Positive control: calibration transfer (C5)

Use d(E_layer)/dE (exact per E1b) for first-order response extrapolation:
predict per-layer response at E = 8, 9, 11, 12 GeV from the 10 GeV state +
gradient; validate against direct sims at those energies (5 seeds × 20k
each). Acceptance: prediction within stat errors at ±10%, controlled
deviation at ±20% (honest nonlinearity). Message: where the trust map says
"exact," the agent may extrapolate without simulating.
*Cost: 4 × 100k = 0.4M ev ≈ 21 CPU-h.*

---

## E6 — Head-to-head vs gradient-free agentic scan (C6)

Same task as E3 (constrained f_vis optimum), three agents, identical tool
access except gradients; primary metric = **simulated events to locate the
optimum within the ground-truth error band**; secondary = wall-clock and
final-design error.

| Agent | Gradient access | Expected result |
|-------|-----------------|-----------------|
| A: scan agent (2604.21804-style) | none — proposes scan ranges/steps, reasons over primal results | finds optimum, most events |
| B: blind-gradient agent | raw AD | fewer events, **wrong or unstable** answer (E3.2) |
| C: audited agent | AD through reliability gate + trust map | fewest events to the *correct* answer |

Implementation: `agent/orchestrator.py` loop + `agent/llm.py`; dry-run
heuristic policies for all three (deterministic, CI-safe) + one real-LLM run
per agent for the paper's trajectory logs. Sim budget ≤ 40 calls × 20k each
per agent.
*Cost: ≤ 3 × 0.8M = 2.4M ev ≈ 125 CPU-h (partially shared with E3).*

Table F5: events-to-solution / calls / correctness, the direct quantitative
comparison against the foil's methodology.

---

## Sidebar (done): estimator auditing

Depth-resolution autopsy (`experiments/depthres_autopsy/`): published 7–10 mm
σ(x_max) spread across layer-structure designs collapses to ≤ 1.6 mm under
sound estimators — variance-shrinkage bias of the extensive-energy argmax.
Framing: the audit habit applies to estimators, not only gradients. No new
compute.

---

## Compute budget & phasing

| Phase | Experiments | New events | ≈ CPU-h | Blocking? |
|-------|-------------|-----------|---------|-----------|
| P1 (map repair + controls) | E1a, E1b, E1c, E5 | 3.2M | 170 | yes — C1 airtight first |
| P2 (mechanism) | E2a (trimmed 12-config option) | 2.4M | 125 | independent of P3 |
| P3 (demos) | E3, E4 (5-layer option), E6 | 6.5M | 340 | needs P1 trust map numbers |
| P4 (generality) | E1d | 2.4M | 125 | polish; can run last |
| **Total** | | **~14.5M** | **~760** | |

All condor (300-job cluster 3911355 = 6M ev completed fine in < 1 day
including fair-share queuing). Nothing here runs locally beyond ≤ 2000-event
smoke tests. **Every phase needs explicit approval before submission** per
CLAUDE.md guardrails.

Trim-to-minimum option (~400 CPU-h): drop E1d, E2a → 12 configs, E4 spot
checks → 5 layers, share E3/E6 runs fully.

## Code work implied (all through existing layers, no C++)

1. Generalize `experiments/perlayer_adfd_1M.py` → parameterized study runner
   (design point, parameter, h, flags) reused by E1b–E2a.
2. `tools/reliability.py`: wire the trust-map class (parameter → prior
   ratio/policy) into `recommend_policy()`.
3. E3/E6: constraint-line optimizer + agent policies (dry-run heuristics
   first) in `agent/`; trajectory logs to `outputs/`.
4. Analysis/figure scripts per experiment, one `--analyze` entry point each.

## Seed & determinism policy

Seeds s = 1..K per unit, disjoint studies may reuse seeds (CRN pairing is
*within* a study's ± configs). Record (seeds, n_events, flags, binary git
rev) in each jsonl row, as `perlayer_adfd_1M.py` already does. Note in the
paper: the sim is not bit-reproducible at fixed seed (~0.4 mm σ jitter
observed) — quote seed-scatter errors, never single-seed numbers.

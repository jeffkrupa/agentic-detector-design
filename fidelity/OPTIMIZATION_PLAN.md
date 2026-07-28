# Optimization campaign plan — information-optimal sampling profile

Draft v2, 2026-07-28 (supersedes the v1 broken-vs-repaired framing, dropped per
direction). The payoff experiment: use the **exact per-layer energy derivative**
that differentiable simulation provides to compute the Fisher information of any
longitudinal sampling profile *directly*, optimize it, and demonstrate (by direct
measurement) that the information-optimal profile resolves beam energy better than
uniform sampling — or cleanly prove uniform is already optimal.

Through-line to the repair program (part 1 of the paper): the objective is only
meaningful because the energy gradient is *exact* — a fact we validated (trust
map: energy ≈ 1.00, the one channel that never needed repair). A biased energy
gradient would give a wrong I(θ) and a wrong design. So the method — validate the
gradient, then use it — is the spine; this experiment is what the validated
gradient buys.

---

## 1. The objective (why it's cheap and clean)

Fisher information about beam energy E from the per-layer deposits {E_i}, Gaussian
(location-term) form:

    I(θ) = gᵀ Σ⁻¹ g ,   g_i = ∂μ_i/∂E ,   Σ = Cov(E_i, E_j)

The Cramér–Rao bound on any unbiased energy estimator is σ_E/E ≥ 1/(E·√I). A
single **energy-seeded forward run** at design θ yields every ingredient:

- `edeps` column `mean_dE`  = g_i = ∂μ_i/∂E  — **exact AD** (validated ≈1.00);
- `edeps` column `var_E`    = Σ_ii (per-layer variance);
- the off-diagonal Σ_ij from the per-event dump (`HEPEMSHOW_EVENT_DUMP`), so the
  **full covariance** version I = gᵀΣ⁻¹g is available, not just the diagonal
  approximation I_diag = Σ_i g_i²/σ_i² (which ignores the strong layer-to-layer
  shower correlations and would overstate the information).

So **one sim per candidate design** gives I(θ). No design-gradient, no dσ/dθ, no
finite differences in geometry. This is what makes a gradient-free search over
designs affordable.

Refinement (secondary): the full single-measurement Fisher info adds a
noise-energy-slope term ½(∂σ_i²/∂E)²/σ_i⁴; ∂σ_i²/∂E is obtainable by
finite-difference in E (two energy points, CRN-paired via `--rng-lineage`, which
we built in wave 4). Report the location-term result as primary; add the
noise-slope term as a cross-check that it doesn't change the optimal profile.

---

## 2. Design space

Optimize the **longitudinal sampling profile**: K = 6 zones (≈8 layers/zone),
per-zone gap thickness g_z, at **fixed total gap** Σ_i g_i = 50·5.70 mm (fixed
sensitive volume / cost) and **uniform absorber** a_i = 2.30 mm (fixed radiation-
length profile). This asks the classic calorimetry question directly — *where
longitudinally should sampling be concentrated to best measure energy?* — with a
genuine interior optimum (fine sampling near shower max captures more info where
∂μ_i/∂E is largest, but also more noise; the balance is nontrivial). DOF = K−1 = 5
after the fixed-total constraint.

Bounds: g_z ∈ [1.0, 12.0] mm (physical, keeps every zone sampling). The constraint
is enforced by projecting each CMA-ES proposal onto Σg = const.

Extension (after primary): joint (a_z, g_z) at fixed total absorber AND gap
(sampling granularity *and* fraction) — uses the absorber design implicitly but
the objective still only needs the energy gradient, so still one run per design.

---

## 3. Pre-flight checks (cheap, before the search)

Two load-bearing assumptions, verified for ~20 CPU-h before committing the campaign:

1. **Energy-gradient exactness OFF the baseline design.** The energy gradient was
   validated exact at (a=2.3, g=5.7). Confirm it stays exact at 2–3 displaced gap
   profiles (front-loaded, back-loaded, extreme) via AD vs CRN-paired FD-in-E.
   Acceptance: AD/FD_E within ~2% at every design. If it drifts, the objective's
   ingredient is compromised and the plan needs revisiting first. (Expected to
   pass — energy is a non-boundary parameter in any geometry — but it's the one
   assumption the whole result rests on, so it gets checked.)
2. **Covariance stability at n=20k.** Confirm the 50×50 Σ (and I=gᵀΣ⁻¹g) is stable
   between independent 20k-event runs — i.e. that Σ⁻¹ isn't ill-conditioned at
   that statistics. If it is, either raise n per design or regularize/reduce Σ to
   the zone level (6×6, far better conditioned) and optimize the zone-Fisher info.

---

## 4. The optimization (gradient-free)

CMA-ES over the 5-DOF constrained gap profile. Each objective evaluation = one
energy-seeded forward run at n = 20k events (unsevered or canonical — the primal
μ_i/σ_i and the exact energy derivative are severing-independent for the energy
channel; use the shared baseline binary for simplicity). Population ~12, ~30
generations ≈ 360 evaluations. Log the full I(θ) history and the evolving profile.
Restarts from uniform and 2 random feasible starts to check the optimum is global.

Determinism/provenance: per-design jsonl records the profile, seeds, n, binary
git-rev; each design's I computed by a committed analysis module so the run is
reproducible.

---

## 5. Physics validation (the actual result)

The optimization runs on predicted information; the *result* is a measured
resolution improvement. Take the info-optimal profile θ* and the uniform profile
θ_0 and, for each:

- **Energy scan**: run at E ∈ {5, 7, 10, 14, 20} GeV, high statistics (200k
  events/point), full per-event dumps.
- **Build an actual energy estimator** (optimal linear / likelihood from the
  measured Σ and response) and measure the achieved σ_E/E vs E at each design.
- **Compare** achieved σ_E/E to the Cramér–Rao prediction 1/(E√I) — confirming
  (a) the Fisher prediction is faithful, and (b) θ* genuinely resolves energy
  better than θ_0, by the predicted margin.

Headline: *the information-optimal sampling profile achieves [X]% better energy
resolution than uniform, predicted from exact differentiable-simulation gradients
and confirmed by direct measurement* — or, if uniform wins, *uniform sampling is
information-optimal for this calorimeter, established with exact per-layer
gradients rather than folklore.* Both are clean results.

---

## 6. Compute budget & condor layout

| phase | content | ~events | ~CPU-h |
|---|---|---|---|
| Pre-flight | energy-grad exactness (3 designs × AD+FD_E) + Σ stability | 1.5M | 20 |
| CMA-ES search | ~360 designs × 20k | 7.2M | 100 |
| Validation | 2 designs × 5 energies × 200k | 2.0M | 28 |
| Refinement (noise-slope Fisher) | recompute on ~top-5 designs, 2 E-points | 0.4M | 6 |
| **Total** | | **~11M** | **~155** |

Much cheaper than the discarded broken-vs-repaired campaign (~415 CPU-h) because
it is one sim per design, not five. All forward mode, condor, AFS submit, 10800 s
timeout, per-design jsonl + event-dump sidecars. Phased and individually gated:
pre-flight must pass before the search; the search's winner before the validation
scan. **Each phase needs explicit approval before submission.**

---

## 7. Risks & honesty ledger

- **Energy gradient is the single load-bearing input** — pre-flight §3.1 guards it;
  if it drifts off-baseline the objective is unreliable and we stop.
- **Covariance conditioning** at 20k — §3.2 guard; fall back to zone-level 6×6
  Fisher if the 50×50 inverse is unstable.
- **"Uniform wins" (structure-helps-nothing again)** — a real possibility, but here
  it is a *publishable* outcome (exact-gradient proof, not folklore), unlike the
  earlier aggregate-observable nulls. The objective is the information content
  itself, computed exactly, not an aggregate swamped by fluctuations.
- **Diagonal vs full covariance** — report both; the full-Σ version is the honest
  one (correlations reduce information), the diagonal is the naive comparison.
- **Location-term vs full Fisher** — primary result is the location term; the
  noise-slope refinement (§1) confirms it doesn't move the optimum.
- **Gap/absorber repair not directly used here** — stated plainly: this experiment
  uses the energy channel (exact). The gap/absorber repair is part 1's result (the
  trust map + the repair method); its role in part 2 is that the *method* —
  validate before you trust — is what licenses using the energy gradient as ground
  truth. The joint-(a,g) extension (§2) is where the repaired design gradients
  could enter a follow-up.

## 8. What "done" looks like

One figure: I(θ) rising over CMA-ES generations to the optimal profile, with that
profile drawn against uniform; one companion: measured σ_E/E vs E for optimal vs
uniform, with the Cramér–Rao bands overlaid. One table: I and σ_E/E(10 GeV) for
uniform / optimal / (extension) joint-optimal. One sentence of result as in §5.

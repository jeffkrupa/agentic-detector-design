# The Idea: Grounding Agentic Detector Design in Exact Physical Derivatives

## 1. Motivation

Differentiable simulation makes the gradient of a physics observable with
respect to a design parameter directly available. In `g4hepem`/`hepemshow` these
gradients are **numerically exact** (machine precision via CoDiPack), not
finite-difference approximations, and they flow through *real* EM physics:
ionization, bremsstrahlung, multiple Coulomb scattering, pair conversion,
annihilation, and the geometry stepping.

Separately, LLM agents are now competent at orchestrating multi-step scientific
workflows. The recent work of Chung et al.
([arXiv:2604.21804](https://arxiv.org/abs/2604.21804)) demonstrates AI agents
driving a bilevel optimization over a vertically-integrated differentiable
detector simulation (geometry → digitization → reconstruction). It is an
important proof of concept, but the authors are explicit about the limitation:

> *"While the ability to make autonomous leaps of physics-motivated judgment or
> insight is not demonstrated in this work, this study defines the current
> frontier of experimental design methods in high-energy physics."*

In their framework the agent **orchestrates**; the gradients live inside the
optimizer as an implementation detail. The agent never *reads*, *reasons about*,
or *is evaluated on* physical sensitivities.

## 2. Thesis

> Exact physical derivatives should be a **first-class input to the agent's
> reasoning**, not just a signal inside an optimizer. An agent that can query,
> interpret, and stress-test physical sensitivities — and that knows when a
> gradient is trustworthy — can make and defend physics-motivated design
> decisions.

This reframes the gradient from "optimizer fuel" to "a sense the agent uses to
perceive the physics."

## 3. Three concrete contributions

### C1 — Gradients as a tool the agent calls
We expose `sensitivity(observable, parameter, design_point)` returning the exact
`∂O/∂θ` (with statistical error bars) as a function-call tool. The agent uses it
to:
- rank which design parameters control a target observable and explain why;
- optimize under **natural-language constraints** ("depth < 30 X₀", "minimize
  channel count", "keep sampling fraction > 3%") that plain gradient descent
  cannot express;
- produce explanations that are *checkable* against the true gradient.

### C2 — A sensitivity-reasoning benchmark
Before revealing a gradient, ask the model to predict its **sign** and
**order of magnitude**. Score predictions against AD ground truth across a grid
of (observable, parameter, configuration) triples. This is, to our knowledge, the
first benchmark of whether LLMs can reason about *physical* sensitivities. It is
cheap (reuses simulation outputs + API calls) and the result is quotable on its
own. See `BENCHMARK.md`.

### C3 — Gradient-reliability awareness
Our stop-gradient and conversion-regularization (CRE) studies characterize
*where* gradients are trustworthy: near geometric grazing/boundary
discontinuities and near-singular MSC conversion denominators, raw AD gradients
can spike (the unregularized "no CRE" outliers). We attach a **reliability flag**
to every sensitivity (from gradient SNR and an AD-vs-FD cross-check) and give the
agent a policy: when a gradient is `untrusted`, fall back to finite differences,
widen statistics, or reason qualitatively. An agent that *declines to trust a bad
gradient* is exactly the physics-motivated judgment prior work lacked.

## 4. The differentiable substrate (what makes this possible)

- **Forward mode** propagates one seeded input perturbation to all per-layer
  observables in one run — ideal for sensitivity *profiles* across depth.
- **Reverse mode** backpropagates per-layer output adjoints to gradients w.r.t.
  the full continuous design vector `(absorber thickness, gap thickness, beam
  energy)` in a single pass — ideal for loss-gradient queries.
- Exactness lets us treat the gradient as a *physical claim* with a definite
  sign/magnitude that the model can be graded against — impossible with noisy FD.

## 5. Design space and observables (small but real)

**Continuous design parameters (differentiable):** absorber thickness `a` [mm],
gap thickness `g` [mm], beam energy `E` [MeV].
**Discrete / non-differentiable:** number of layers `L`, transverse size `t`.

**Observables:** per-layer mean energy deposit; total deposited energy;
shower-maximum depth; longitudinal containment fraction; sampling fraction;
energy-resolution proxy (from event-to-event variance). Definitions and formulas
are in `SIMULATION_INTERFACE.md`.

These are textbook calorimeter quantities with well-known qualitative
sensitivities (e.g. shower max ∝ ln E; thicker absorber → earlier shower, worse
sampling fraction). That known physics is what makes the benchmark meaningful: we
can sanity-check both the AD ground truth and the model's reasoning.

## 6. Positioning vs. Chung et al.

We are complementary, not competing. Chung et al. show agents can *run* a
differentiable design loop. We show agents can *reason with the physics* the loop
exposes, and we provide the first quantitative measure (the benchmark) of how good
that reasoning is — plus a reliability mechanism that begins to address their
stated open problem.

## 7. Suggested narrative for the paper

1. Differentiable EM simulation gives exact, physically-meaningful gradients.
2. Expose them to an LLM agent as a first-class tool.
3. Benchmark: can the model predict sensitivities? (headline number)
4. Design loop: agent optimizes under NL constraints, beats no-gradient and
   matches/į beats plain GD, and *explains* itself consistently with the gradient.
5. Reliability: agent detects and routes around untrustworthy gradients
   (the stop-grad/CRE regime), demonstrating physics-motivated judgment.

Working title: *"Grounding Agentic Detector Design in Exact Physical Derivatives
from Differentiable Simulation."*

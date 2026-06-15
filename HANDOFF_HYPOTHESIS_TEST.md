# Bilevel Detector Design: Differentiable Inner Loop + Agentic Representation Editing

A self-contained scientific brief for an agent or collaborator. It states the
underlying capability, the central hypothesis, the method, the assumptions that
must be tested, the decisive experiment, and the claim discipline. Scaffolding /
implementation details are intentionally omitted; they are not the point.

---

## 1. What the code does (the capability)

We have a **differentiable simulation of electromagnetic showers** in a layered
sampling calorimeter:

- **`hepemshow`** — a simplified particle-transport stepping loop that develops EM
  showers through alternating absorber and active/gap layers.
- **`g4hepem`** — the physics engine it calls each step: a compact implementation
  of Geant4 EM physics for e⁻/e⁺/γ (ionization, bremsstrahlung, multiple Coulomb
  scattering, pair conversion, annihilation, photon interactions).
- The full transport+physics program is **differentiable end-to-end** (CoDiPack)
  in forward and reverse mode.

This yields **exact pathwise derivatives of the differentiable transport program**
— per-layer energy-deposition observables w.r.t. continuous parameters (absorber
thickness, gap thickness, beam energy, etc.). Non-smoothness intrinsic to MC
transport (boundary crossings, discrete interaction sampling) is handled with
stop-gradient policies and targeted regularization of near-singular terms; that
methodology is the existing physics paper and is assumed working.

> Claim discipline: these are exact derivatives **of the differentiable
> physics-transport program**, validated against finite-difference checks where
> applicable — *not* "exact gradients of the real physics" (stop-gradient
> policies make that overclaim attackable).

**Capability statement:** *given a calorimeter configuration and beam, we can
obtain exact gradients of detector observables w.r.t. continuous design
parameters, through realistic EM physics, cheaply enough to optimize them.*

---

## 2. Central hypothesis

Detector design has two qualitatively different parts:

1. **Continuous variables** (absorber/gap/sensor thickness, spacing, thresholds,
   material-budget fractions) — smooth enough for AD gradients to optimize.
2. **Discrete / structural variables** (number of regions, layer grouping,
   material ordering, segmentation pattern, channel allocation, split/merge,
   graded profiles) — not naturally differentiable. A gradient cannot say
   "invent a three-region calorimeter."

> **Hypothesis:** a physics-grounded agent can improve detector design by editing
> the *structural representation*, while exact AD optimizes the continuous degrees
> of freedom within each proposed representation.

The agent does **not** replace gradient descent. The agent **decides what space
gradient descent operates in.**

---

## 3. Method: a bilevel loop

```
Agent proposes a structural representation
        ↓
AD optimizer tunes continuous parameters within it   (inner loop)
        ↓
Full simulation evaluates the optimized design
        ↓
"Optimization autopsy" summarizes gradients, constraints, failures
        ↓
Agent proposes the next structural edit               (outer loop)
```

- **Outer loop (agent, structural search):** split into front/middle/rear
  regions; merge low-value rear layers; increase granularity near shower maximum;
  reduce readout where gradients are small; introduce a graded absorber; change a
  section's material; reallocate depth or channel budget across regions.
- **Inner loop (AD, continuous optimization):** for each proposed structure,
  optimize region thicknesses, gaps, spacings, continuous depth/material
  allocations, thresholds; then run full simulation to evaluate.

---

## 4. The key mechanism: the optimization autopsy

The agent must receive **structured derivative evidence from the inner loop**, not
just the final score. The autopsy is the real bridge between AD and agency:

- final objective value; active/binding constraints; saturated parameter bounds;
- residual gradients after optimization; layer-wise sensitivity maps;
- regions of high vs. low sensitivity; where derivative estimates are unstable;
- failure modes: leakage, poor containment, poor resolution, wasted readout.

**Worked example.** Autopsy reports: middle-region gap pushed to its upper bound;
large residual gradient near layers 14–20; small rear-layer sensitivity; leakage
up only for high-energy events. Grounded agent response: *split the middle region
around shower maximum, allocate finer sampling there, coarsen rear readout to stay
within channel budget.* The structural edit is justified by derivative evidence —
that is the scientific content.

---

## 5. Assumptions that must be tested (not asserted)

- **A1 — Design is representation-limited.** A fixed parametrization can be too
  restrictive; revising the representation itself has value. *Test fairly:* the
  baseline must also have access to structural moves, or the agent "wins" only by
  being handed a richer design language.
- **A2 — AD gradients carry useful local design information.** Sensitivity maps
  should reveal which layers matter, which parameters are bottlenecks, where more
  material/finer sampling helps, where readout can be cut, which constraints bind.
  *Early test:* do gradient-derived sensitivity maps **predict** which structural
  moves help after full simulation? (e.g. large `dJ/dgap` near shower max →
  does finer sampling there actually improve the objective?)
- **A3 — Discrete moves have useful continuous shadows.** Each structural edit
  maps to a continuous proxy (add absorber→regional depth; finer sampling→
  gap/absorber ratio; coarsen readout→channel density; split region→independent
  regional params; change material→radiation-length/density proxy). The proxy
  need not be accurate — only good enough to **price/prioritize** candidate moves
  before expensive simulation. *Test directly.*
- **A4 — The agent beats scripted structural search (most fragile).** The correct
  skeptical objection is: *why not a hand-coded split/merge search with the same
  gradients?* So the real claim is: **given the same structural grammar, same
  simulation budget, and same AD autopsy, the agent proposes better/more efficient
  edits than non-agentic structural search.** The strong baseline is *scripted
  structural search + gradient pricing + AD inner optimization*, not random search.
- **A5 — Exact gradients matter.** Ablate: exact AD vs. finite-difference vs.
  noised gradients vs. objective-only. Outcome A (exact AD helps) → the
  differentiable simulator is central. Outcome B (noisy works as well) → the claim
  weakens to "local sensitivity, even approximate, guides structural search."
- **A6 — The task is hard but not artificial.** Design space must be too large to
  enumerate, physically meaningful, structurally rich, and not engineered to
  flatter the agent. Good: *design a compact layered calorimeter / radiation
  monitor under fixed depth, mass, and channel budget* with nonuniform
  longitudinal structure. Weak: *optimize number of layers* (enumerable).

---

## 6. The minimal decisive experiment

One physically meaningful task — **optimize a layered calorimeter under fixed
total depth and channel budget** — with these arms:

1. uniform design + AD continuous optimization;
2. fixed two/three-region parametrization + AD;
3. Bayesian / evolutionary search over a predefined structural space;
4. scripted split/merge structural search using gradient autopsies;
5. agent using only final objective scores;
6. agent using full AD autopsies.

- **Primary question (A4):** does arm 6 discover better structural
  representations faster (per simulation budget) than arms 3–4?
- **Secondary question (A5):** do exact gradients beat noisy/finite-difference
  sensitivities for guiding structural edits (arm 6 vs. its ablations)?

Run the cheap, agent-free pieces first: A2 (do sensitivity maps predict helpful
moves?) and A3 (do continuous proxies price discrete moves?). These yield a solid
result regardless of the agentic outcome.

---

## 7. Claim discipline

**Strongest defensible claim:**

> We introduce a bilevel detector-design framework in which an exact
> differentiable simulation engine performs continuous optimization within a
> detector representation, while an agent proposes structural edits to that
> representation using gradient-based optimization autopsies. We test whether this
> representation-editing loop improves design quality per simulation budget
> compared with fixed-parametrization and non-agentic structural-search baselines.

**Do not claim:** "LLMs optimize detector design" (vague); "agents beat standard
optimizers" (too broad); "continuous gradients solve discrete detector design"
(false); "exact gradients of the real physics" (attackable — say "exact AD
derivatives of the differentiable transport program, FD-validated").

**Decision rule.** If arm 6 beats arms 3–4 on a non-enumerable space, the agentic
claim is earned. If not, lead with the A2/A3 result (sensitivity-guided structural
pricing — agent-free, rigorous) and demote the agent to a demonstrator. Both
outcomes are honest and worth reporting.

---

## 8. One-sentence summary

The strongest idea is not "use agents for detector design"; it is **use an agent
to edit the detector-design representation, guided by exact differentiable-
simulation autopsies, while AD optimizes the continuous parameters inside each
representation** — and to test that against scripted structural search with the
same gradients, so the agent has to earn its place.

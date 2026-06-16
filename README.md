# Grounding Agentic Detector Design in Exact Physical Derivatives

> An LLM-agent framework that treats the **numerically exact gradients** of a
> differentiable EM calorimeter simulation (`hepemshow` + `g4hepem` + CoDiPack)
> as a first-class *reasoning substrate* — not just an optimizer's black box.

This project plants a flag at the intersection of **agentic AI** and
**differentiable simulation** for high-energy physics. Where existing work
(e.g. Chung et al., [arXiv:2604.21804](https://arxiv.org/abs/2604.21804)) uses
agents to *orchestrate* a bilevel optimization over a differentiable detector
simulation, here the agent **senses, reasons about, and explains physical
sensitivities** using exact derivatives extracted from the simulation via
automatic differentiation (AD).

The headline claims this repo is built to support:

1. **Gradients as a tool.** Exact `∂(observable)/∂(design parameter)` is exposed
   as a callable tool. The agent queries physical sensitivities and uses them to
   plan, optimize under natural-language constraints, and justify its decisions.
2. **A sensitivity-reasoning benchmark.** Can frontier LLMs predict the *sign and
   magnitude* of physical derivatives before seeing them? We score model
   predictions against AD ground truth — a first-of-its-kind eval.
3. **Gradient-reliability awareness.** Building on our stop-gradient / conversion
   regularization studies, each sensitivity carries a *reliability flag*. An agent
   that knows **when not to trust a gradient** — and switches strategy — is the
   "physics-motivated judgment" prior work identified as missing.

---

## Why this is novel (the gap we exploit)

Chung et al. concede: *"the ability to make autonomous leaps of physics-motivated
judgment or insight is not demonstrated in this work."* In their setup the agent
traverses parameter space and calls the optimizer; the gradients stay buried
inside the optimizer.

Our differentiator is the **truly differentiable physics**: `g4hepem` is
differentiated through real EM processes (ionization, bremsstrahlung,
multiple scattering, conversion, annihilation, ...) with CoDiPack, yielding
derivatives that are *numerically exact* (to machine precision) rather than
finite-difference estimates. That makes the gradient a meaningful *physical*
quantity the agent can reason with and be evaluated against.

| | Chung et al. 2026 | This work |
|---|---|---|
| Agent role | Orchestrate bilevel optimization | **Sense + reason + explain** with gradients |
| Gradient use | Inside optimizer (black box) | **First-class tool** the agent calls |
| Gradient source | Differentiable full sim | Differentiable full sim (exact AD) |
| Reliability | Not modeled | **Per-query reliability flag** |
| Eval of physics insight | Not demonstrated | **Sensitivity-prediction benchmark** |

---

## The simulation in one paragraph

`hepemshow` is a simplified stepping loop wrapped around `g4hepem`'s EM physics.
The default geometry is a 50-layer Pb(WO4)/lAr sampling calorimeter. The
executable is compiled in two AD flavors: a **forward-mode** build
(`build/HepEmShow`) that propagates one input perturbation to all outputs, and a
**reverse-mode** build (`build_reverse/HepEmShow`) that backpropagates output
adjoints to gradients w.r.t. all design inputs in a single pass. Outputs are
per-layer energy deposits and their derivatives. See
[`docs/SIMULATION_INTERFACE.md`](docs/SIMULATION_INTERFACE.md) for the exact CLI,
file formats, and the parameter/observable catalog.

---

## Repository map

```
agentic-detector-design/
├── README.md                  # this file
├── CLAUDE.md                  # operating manual for Claude Code (read this first)
├── requirements.txt
├── config.example.yaml        # copy to config.yaml and edit paths/model/budgets
├── docs/
│   ├── IDEA.md                # scientific framing, related work, contributions
│   ├── SIMULATION_INTERFACE.md# exact binary CLI, AD modes, file formats, catalog
│   ├── AGENTS.md              # orchestrator + subagent architecture and message flow
│   ├── BENCHMARK.md           # sensitivity-prediction benchmark protocol + scoring
│   └── EXPERIMENTS.md         # experiment arms (A/B/C/D), ablations, metrics, figures
├── tools/                     # FUNCTIONAL python layer that drives the real binaries
│   ├── schemas.py             # dataclasses: DesignPoint, Sensitivity, Observable, ...
│   ├── sim.py                 # forward/reverse HepEmShow wrappers + caching
│   ├── observables.py         # derived observables (total, peak, depth, fractions)
│   └── reliability.py         # gradient reliability flag heuristics
├── agent/                     # model-agnostic agent scaffolding
│   ├── tool_registry.py       # JSON tool schemas wrapping tools/*.py
│   ├── llm.py                 # pluggable LLM client (Anthropic/OpenAI/dry-run)
│   ├── subagents.py           # subagent definitions + routing
│   ├── orchestrator.py        # main agent loop
│   └── prompts/               # system prompts (orchestrator + each subagent)
├── benchmark/
│   ├── make_dataset.py        # generate (observable, param, config) ground-truth grid
│   └── run_benchmark.py       # score LLM sign/magnitude predictions vs AD truth
└── tasks/                     # example natural-language design goals
```

---

## Quickstart

```bash
git clone git@github.com:jeffkrupa/agentic-detector-design.git
cd agentic-detector-design
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml      # edit paths if needed

# 1) Smoke-test the differentiable simulation tool layer (no LLM needed):
python -m tools.sim --selftest

# 2) Build a small sensitivity ground-truth grid:
python benchmark/make_dataset.py --quick --out benchmark/data/quick.jsonl

# 3) Run the agent in dry-run mode (heuristic planner, no API key required):
python -m agent.orchestrator --task tasks/example_resolution.md --dry-run

# 4) Run the grounding benchmark against a model:
python benchmark/run_benchmark.py --dataset benchmark/data/quick.jsonl --model dry-run
```

The tool layer is fully functional against the compiled binaries. The LLM layer
ships with a `dry-run` client so everything runs end-to-end without an API key;
plug in a real client in `agent/llm.py` (clearly marked `TODO`).

---

## Status / handoff

This is a **scaffold designed to be handed to Claude Code** to flesh out. Start
with [`CLAUDE.md`](CLAUDE.md), which contains build/run/test commands, the exact
simulation interface, coding conventions, guardrails, and a prioritized task
list. The tool layer is real; the agent layer is a working skeleton with marked
extension points.

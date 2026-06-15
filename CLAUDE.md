# CLAUDE.md — Operating Manual for Claude Code

This file orients an AI coding agent (Claude Code) working in this directory.
Read it fully before editing. It encodes the **ground-truth interfaces**,
**commands**, **conventions**, and **guardrails** for this project.

Project goal: *Grounding agentic detector design in exact physical derivatives.*
See `README.md` for the pitch and `docs/IDEA.md` for the scientific framing.

---

## 0. TL;DR for the agent

- The differentiable simulation is **already built and working**. Your job is to
  build the **agent + benchmark layers** on top of the functional `tools/` layer.
- Never launch large batches of jobs or long sweeps without being asked. Default
  to **small event counts** (`-n 1000`..`2000`) and a handful of seeds for dev.
- Prefer extending `tools/observables.py` and `agent/*` over touching the C++.
- The C++ simulation lives outside this folder and is **out of scope** unless
  explicitly requested. Do not rebuild it casually (builds are slow).
- Keep the `dry-run` path working at all times so the repo runs without API keys.

---

## 1. Paths (ground truth)

```
REPO_ROOT      = /sdf/data/atlas/u/jkrupa/hepemshow/hepemshow
FORWARD_BIN    = $REPO_ROOT/build/HepEmShow            # forward-mode AD (CoDiPack)
REVERSE_BIN    = $REPO_ROOT/build_reverse/HepEmShow    # reverse-mode AD (CoDiPack)
HEPEM_DATA     = $REPO_ROOT/data/hepem_data.json
AGENTIC_DIR    = $REPO_ROOT/build/hepemshow_utils/agentic   # you are here
```

All four of the first paths are confirmed to exist and are used by the existing
optimizer at `../optimize/optimize.py`. If a path is wrong, fix it in
`config.yaml` (not hard-coded in source).

---

## 2. The simulation interface (exact)

The binary is configured entirely by CLI flags (parsed in
`$REPO_ROOT/Simulation/include/InputParameters.hh::GetOpt`).

### Core flags

| Flag | Meaning | Units / form |
|------|---------|--------------|
| `-d <file>` | G4HepEm data file | path (use `$HEPEM_DATA`) |
| `-n <int>` | number of events | e.g. `2000` |
| `-s <num>` | random seed | integer-valued |
| `-p <name>` | primary particle | `e-`, `e+`, `gamma` |
| `-e <E[:dot]>` | beam energy [MeV] | `10000` or `10000:1` (forward-seed) |
| `-a <t[:dot]>` | absorber thickness [mm] | `2.3` or `2.3:1` (forward-seed) |
| `-g <t[:dot]>` | gap thickness [mm] | `5.7` or `5.7:1` (forward-seed) |
| `-l <int>` | number of layers | discrete, **non-differentiable** |
| `-t <size>` | transverse size [mm] | continuous |
| `-b <a0:a1:...>` | **reverse only**: per-layer output adjoints | colon-separated, len = nlayers |

### Differentiation-control flags (from the physics paper)

These select stop-gradient policy and the regularizers studied in the paper. For
agent work, use the **paper defaults** unless an experiment varies them.
**Verified CLI mapping** (from `InputParameters.hh`): `-N` is `conversion-reg-eps`
(CRE), `-A` is `numia-mfp-floor`, `-C` is `gamma-mfp-cap` (gmc), `-x` is
`--stop-grad-mode`.

```
-x 2     # --stop-grad-mode : 0=off, 1=track-only, 2=track+descendants
-y 1     # grazing-stop-track
-B 1     # backward-boundary-stop
-f 0.2   # grazing |vx| threshold
-N 1e-3  # conversion-reg-eps (CRE)
-C 1000  # gamma mfp cap (gmc)
-A <v>   # numia-mfp-floor (off by default)
```

> The full list of regularization flags is in `InputParameters.hh`. The canonical
> "good" configuration matches the paper's main result:
> `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000` (i.e. CRE = 1e-3, gmc = 1000).

### AD modes

- **Forward mode** (`FORWARD_BIN`): seed exactly one input with `:1`
  (`-a 2.3:1` *or* `-e 10000:1` *or* `-g 5.7:1`). The output file then contains
  the derivative of **every** layer observable w.r.t. that one input. Best for
  "how does the whole shower respond to parameter X?".
- **Reverse mode** (`REVERSE_BIN`): pass per-layer output adjoints with `-b`.
  One backward pass yields gradients w.r.t. **all three** continuous design
  params at once. Best for "gradient of my scalar loss w.r.t. the design vector".

### Output files (written to the process CWD)

- `edeps_<seed>`: shape `(nlayers, 4)`, columns:
  `[ mean_E, var_E, mean_dE, var_dE ]`
  - `mean_E`  : mean energy deposit in the layer [MeV]
  - `var_E`   : variance of the per-event energy deposit
  - `mean_dE` : mean of the seeded derivative (forward mode) [MeV / unit-input]
  - `var_dE`  : variance of the derivative
- `barInputs` (reverse mode): 3 rows of `mean var`, in this order:
  1. `barThicknessAbsorber` → ∂L/∂(absorber thickness)
  2. `barThicknessGap`      → ∂L/∂(gap thickness)
  3. `barParticleEnergy`    → ∂L/∂(beam energy)

Because each process writes to its CWD, the `tools/sim.py` wrapper runs every
invocation in a unique temp dir and parses the result. **Do not** run two sims in
the same CWD.

### Statistics / error bars

The mean derivative's standard error over `N` events is
`SE = sqrt(var_dE / N)`. The reliability layer uses the signal-to-noise ratio
`|mean_dE| / SE` and (optionally) an AD-vs-finite-difference cross-check.

---

## 3. Commands

```bash
# Install python deps
python -m pip install -r requirements.txt
```

> **Python version (cluster gotcha).** The system `python3` is **3.6**, which is
> too old (the code uses `from __future__ import annotations` and 3.10+ idioms).
> `/usr/bin/python3.9` exists but has no numpy/yaml. Use the local venv:
> ```bash
> python3.9 -m venv .venv
> source .venv/bin/activate
> python -m pip install -r requirements.txt
> ```
> Activate `.venv` before running anything below. The venv is git-ignored.

```bash
# Tool-layer smoke test (calls the real forward & reverse binaries on a tiny job)
python -m tools.sim --selftest

# One forward sensitivity query (CLI helper)
python -m tools.sim --observable total_edep --wrt a --a 2.3 --energy 10000 -n 1000

# Build a quick benchmark dataset (small grid)
python benchmark/make_dataset.py --quick --out benchmark/data/quick.jsonl

# Score a model on the grounding benchmark (dry-run needs no API key)
python benchmark/run_benchmark.py --dataset benchmark/data/quick.jsonl --model dry-run

# Run the design agent on a task (dry-run = heuristic, no API key)
python -m agent.orchestrator --task tasks/example_resolution.md --dry-run
```

There is no compiled component here. If you must (re)build the simulation, ask
first — the CMake builds under `$REPO_ROOT/build*` are slow and shared.

---

## 4. Conventions

- **Python 3.10+**, standard library + `numpy`, `pyyaml`, `pydantic` optional.
  Keep hard dependencies minimal; the tool layer must run on a bare cluster node.
- All external paths come from `config.yaml` (loaded by `tools/sim.py`), never
  hard-coded in new modules.
- Tool functions return **dataclasses from `tools/schemas.py`**, not bare dicts,
  so the agent layer and the benchmark share one contract.
- Every simulation call goes through `tools/sim.py` (temp-dir isolation + caching
  keyed on the full flag set). Do not shell out to the binary elsewhere.
- Keep the `dry-run` LLM client functional. New tools must have a deterministic
  heuristic fallback so CI and demos work offline.
- Do not add comments/docstrings to code you didn't change. Match existing style.

---

## 5. Guardrails (important)

- **No large compute without explicit approval.** Dev/test uses `-n <= 2000` and
  ≤ 8 seeds. Anything bigger (or any SLURM submission) must be requested.
- **No SLURM submission** from agent code by default. The existing
  `../jobs/submit_many.sh` is for the human; the agent's `sim` tool runs **local
  subprocesses** only.
- **No destructive actions.** Do not delete `outputs/`, build dirs, or job logs.
- **Determinism for the benchmark.** Ground-truth datasets pin seeds and event
  counts; record them in the dataset file so results are reproducible.
- **Cache, don't spam.** `tools/sim.py` caches by flag-hash under
  `.sim_cache/`; reuse it. Clear it only on request.
- If a sim call returns NaNs or non-zero exit, surface it as a *low-reliability*
  result rather than silently dropping it (the NaN/outlier behavior is physics-
  relevant — see the CRE study).

---

## 6. Prioritized task list (suggested order)

1. **Verify the tool layer.** Run `python -m tools.sim --selftest`; fix any path
   or parsing mismatch against the real binaries. (Highest priority — everything
   builds on this.)
2. **Flesh out `tools/observables.py`.** Implement the derived observables in
   `docs/SIMULATION_INTERFACE.md` §Observables (total edep, shower max,
   longitudinal containment, sampling fraction, resolution proxy) with unit tests
   on a cached tiny run.
3. **Implement `tools/reliability.py`.** Gradient SNR + optional AD-vs-FD
   cross-check → a `ReliabilityFlag` (`ok` / `marginal` / `untrusted`).
4. **Wire the tool registry** (`agent/tool_registry.py`) so each tool has a JSON
   schema and a Python dispatcher.
5. **Implement the orchestrator loop** (`agent/orchestrator.py`): plan → call
   tools → reflect → act, with a step/compute budget from `config.yaml`.
6. **Implement subagents** (`agent/subagents.py`) per `docs/AGENTS.md`:
   Sensitivity-Analyst, Optimizer, Physics-Critic, Reporter.
7. **Build the benchmark** (`benchmark/*`): dataset generator + scorer +
   sign/magnitude/calibration metrics per `docs/BENCHMARK.md`.
8. **Add a real LLM client** in `agent/llm.py` (Anthropic + OpenAI), keeping
   `dry-run` as default.
9. **Reproduce the experiment matrix** in `docs/EXPERIMENTS.md` at small scale,
   then document how to scale up.

Mark progress by leaving short notes in `docs/EXPERIMENTS.md` under "Log".

---

## 7. What "done" looks like for v0.1

- `tools.sim --selftest` passes against both binaries.
- A 50–100 row benchmark dataset exists and a model (even dry-run) is scored with
  sign-accuracy, magnitude-MAE-in-dex, and a reliability-stratified breakdown.
- The agent completes `tasks/example_resolution.md` end-to-end in dry-run, emits a
  trajectory log, and produces a short report whose claimed sensitivities match
  the tool's gradients (consistency check passes).

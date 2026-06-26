# Experiment results & reproduction (companion to `HANDOFF.md`)

All runs: PbWO4/lAr sampling calorimeter, e⁻ at 10 GeV, paper ctrl flags
`-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`. Binaries:
`/eos/user/j/jeffkrup/agentic/hepemshow/build{,_reverse}/HepEmShow`. Python repo branch
`phase1-e2-agentic-loop`; activate `.venv`.

---

## A. Headline numbers

### A.1 Energy-resolution ceiling (gradient-free) — 40 layers, 10000 events, 6 seeds, fixed gap budget
`σ(E_vis)/⟨E_vis⟩`, **lower = better**. Designs redistribute the *same* total gap.
```
  uniform : 0.0319 ± 0.0002   ← BEST
  tail    : 0.0551
  front   : 0.0697
  bump    : 0.0769 ± 0.0013   ← WORST (gap concentrated at shower max)
```
**Conclusion: uniform sampling gives the best total-energy resolution; concentrating
structure makes it 1.7–2.4× worse.** Aggregate objectives don't reward longitudinal structure.
(Script: `experiments/resolution_ceiling.py`; raw rows `experiments/resceil_*_s*.jsonl`.)

### A.2 Gradient reliability — `d(E_vis)/d(uniform gap)` at g=5.7, 20 layers, 5000 events, 8 seeds
```
  AD (reverse, --bar-gap)      = 1951 ± 24   (SNR 82)   ← precise
  FD (central, same seed)      =  198 ± 3
  ratio AD/FD                  = 9.86
```
**Conclusion: the AD gap-thickness gradient is ~10× biased (high SNR ⇒ bias, not noise).**
It silently misled the optimizer. (Script: `experiments/grad_reliability.py`; raw
`experiments/gradrel_s*.jsonl`.)

### A.3 The retracted "net-signal structure win" (artifact)
Net-signal `L = −E_vis + μ·gap_material`. Hand-rolled GD reported K=40/K=1 ≈ 0.67–0.85 with
bang-bang front-thick/tail-thin profiles. **Artifact** of under-converged GD + the biased
gradient (A.2). Direct gradient-free loss check at μ=70, 20 layers:
```
  uniform g=5.7 : E_vis 1273, gap material 114, LOSS 6707
  non-unif 12/4 : E_vis 1521, gap material 160, LOSS 9679   ← WORSE
  uniform g=3.0 : E_vis  729, gap material  60, LOSS 3471   ← BEST (uniform-thin)
```
L-BFGS-B independently converges to ~uniform (ratio ≈ 0.998). **Structure does not help
net-signal.** Do not cite the GD ratios.

### A.4 Earlier ceiling tests (all NO)
- total_edep / containment, K=1 vs K=n (full per-layer freedom), 20 & 40 layers: every-layer-
  free collapses to **uniform** (e.g. 40-layer K=40 → 2.24 mm at all 40 layers). Structure = no gain.
- flat-profile MSE: target unreachable; no discrimination.

---

## B. Code map (branch `phase1-e2-agentic-loop`)
- `tools/sim.py` — binary wrapper; `run_forward`, `run_reverse`, `run_forward_gap`,
  `run_reverse_per_layer(..., gap_adjoints=)`, `reverse_*_multiseed`, `region_gradients`,
  `_common_args`, flag-hash cache, `sim.subprocess_timeout_s` (config; currently 1800).
- `tools/observables.py` — `observe`, `sensitivity`, profiles; obs: total_edep, peak_edep,
  shower_max_depth, visible_fraction, front_fraction. (NOTE: `observe` is profile-safe — it
  seeds `energy` not `a` when a per-layer profile is set, because the binary forbids seeding
  `-a`/`-g` under a profile.)
- `tools/schemas.py` — frozen `DesignPoint` (+ `abs_profile`/`gap_profile` tuples), `Region`,
  `regions_to_profiles`, `profiles_from_design`.
- `tools/optimizer.py` — inner loops; **use `optimize_inner_netsignal_lbfgs`** (scipy L-BFGS-B).
  Others are fragile GD (reference only). `InnerResult` fields: `final_rep`, `history`
  (per-iter dicts: iter/objective/loss/evis/region_grads/step), `final_objective`,
  `final_residual`, `converged`, `predicted_vs_realized`.
- `tools/gap_signal_target.py` — `total_gap_signal` (=E_vis), `net_signal_loss/adjoints`,
  `gap_length_region_grad`; also the older shortfall `gap_signal_*`.
- `tools/constraints.py` — `parse_constraints`, `check_constraints`, `clip_to_bounds`.
- `tools/reliability.py` — SNR + AD-vs-FD cross-check → ok/marginal/untrusted. **Use this to
  gate gradients.** `tools/fd_check_perlayer.py` — per-layer AD-vs-AD/FD checker.
- `tools/autopsy.py` — per-iteration optimization autopsy (per-region grads+SNR, heatmap).
- `agent/structural_moves.py` — `DesignRepresentation` + `Split/Merge/SetRegionThickness/
  SplitEqual` + `apply_move`/`legal_moves`.
- `agent/e2_loop.py` — bilevel Arm A (agent: inner-opt→autopsy→structural move) vs Arm C
  (fixed-structure baseline); dry-run heuristic structural-analyst (LLM clients are stubs).
- `agent/orchestrator.py`, `agent/llm.py` (Anthropic/OpenAI = NotImplementedError stubs),
  `agent/subagents.py`, `agent/tool_registry.py`, `agent/prompts/`.
- `experiments/` — `e2_discriminate*.py` (containment/gapsignal ceilings, `--objective
  {shortfall,netsignal}`, `--optimizer {gd,lbfgs}`), `grad_reliability.py`,
  `resolution_ceiling.py`, `depth_resolution.py`.
- `benchmark/` — E1 sensitivity-benchmark scaffolding (dataset + scorer; not the focus).
- `condor/` — `*.sub` + `run_*.sh` (AFS-deployed). See `HANDOFF.md` §5.
- `patches/` — portable diffs of the hepemshow C++ changes (apply on tag `v1.0-paper`).

## C. Git / branches
- `hepemshow`: `phase0-perlayer-geometry` (per-layer geometry), `phaseA-perlayer-gap-energy`
  (per-layer gap-energy AD output; current). Base tag `v1.0-paper`.
- `agentic-detector-design`: `phase1-e2-agentic-loop`. Notable commits (see `git log` for all):
  per-layer plumbing, containment objective+gate, Phase-A python, net-signal objective,
  subprocess timeout (`f34cd91`), L-BFGS-B (`ddc9892`), `--gap-max` (`aba2c57`),
  `grad_reliability` (`d0165c5`), per-coordinate trust-region fix (`3b88dbf`),
  `resolution_ceiling` (`dc745d7`). depth_resolution + condor committed by the depth subagent.
- **Uncommitted/in-flight to verify:** the one-line `SteppingLoop.cc:46` env-gate (HANDOFF §6)
  — confirm whether it was applied/rebuilt before redoing it.

## D. Reproduce the headline results
```bash
cd /eos/user/j/jeffkrup/agentic/agentic-detector-design && source .venv/bin/activate
python -m tools.sim --selftest                 # fwd/rev AD cross-check PASS (~1e-13)
# resolution ceiling (gradient-free; needs the existing forward binary):
#   experiments/resolution_ceiling.py  (per design: parses the "Gap ... Std-dev" stdout line)
# gradient reliability (AD vs FD):    experiments/grad_reliability.py
# (both launched at scale via condor/ — see HANDOFF §5; raw rows in experiments/*.jsonl)
```
Aggregation snippets used in this work are simple (mean ± stderr over seed JSONLs); see the
tables in §A for the format.

## E. Known reliability caveats (must respect)
- Gradients w.r.t. **gap thickness** are ~10× biased (A.2) — gate with AD-vs-FD before use.
- Same-seed finite differences over **geometry changes** desync the RNG stream → noisy/biased
  FD; the project's stop-gradient machinery is about exactly these discontinuities. Prefer
  gradient-free *value* comparisons (e.g. §A.1) where possible.
- HTCondor `RemoteUserCpu` lags; read result rows, not the CPU counter, to judge progress.

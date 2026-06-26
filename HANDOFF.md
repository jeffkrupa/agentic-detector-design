# HANDOFF — Agentic Detector Design: state, findings, and what to do next

> Audience: a new agent/collaborator picking up this work cold. Read this top to
> bottom before touching anything. Companion file: `docs/EXPERIMENT_RESULTS.md`
> (concrete numbers + how to reproduce).

---

## 0. TL;DR (the honest bottom line)

- **The original headline thesis (A4: "an agent that edits detector *structure* beats
  non-agentic baselines") is NOT supported in this testbed.** Every *aggregate-energy*
  objective we tried optimizes to a **uniform** layer geometry — structure buys nothing.
- **Two real, defensible findings came out of it instead** (the A2/A3/A5 "fallback"
  framing the original handoff named):
  1. **A clean negative-structure result**: in a simple sampling calorimeter, single
     aggregate-energy objectives (total deposit, visible/sampled energy, *energy
     resolution*) are optimized by **uniform** sampling. Concentrating structure hurts.
  2. **A concrete AD-reliability failure**: the exact reverse-AD gradient
     `d(E_vis)/d(gap-thickness)` is **high-confidence but ~10× biased** vs finite
     difference (AD 1951 ± 24, SNR 82; FD 198 ± 3; ratio ≈ 9.9). It *silently misled* a
     gradient-based optimizer into a false "structure helps" result that evaporated under
     a gradient-free check.
- **One physically-motivated avenue remains genuinely open and is set up but not yet run:**
  *shape/discrimination* objectives (shower-max **depth** resolution; particle ID). A
  peaked shower's peaked sensitivity gets **integrated away** by *aggregate* objectives,
  but it should matter for objectives that resolve the *shape*. The shower-max-depth
  resolution test is built (`experiments/depth_resolution.py`) and **gated on one
  remaining one-line C++ change** (see §6).

---

## 1. The project & the scientific thesis

Goal: *grounding agentic detector design in exact physical derivatives.* A differentiable
particle-transport calorimeter simulator (`hepemshow`, a Geant4-EM-style stepping loop on
G4HepEm, made differentiable with CoDiPack — forward and reverse AD binaries) lets you take
exact gradients of detector observables w.r.t. design parameters. The **bilevel hypothesis**
(`HANDOFF_HYPOTHESIS_TEST.md`):

- **Outer loop (agent, discrete):** propose *structural* edits — split/merge longitudinal
  regions, change granularity, etc.
- **Inner loop (AD, continuous):** for each fixed structure, optimize the continuous
  per-region thicknesses with exact gradients.
- **Claim A4 (strong, fragile):** the agent finds better structures, per compute budget,
  than non-agentic structural search.
- **Fallback claims (A2/A3/A5):** gradient *sensitivities predict where structure helps*;
  exact gradients vs finite-difference reliability matters. The original handoff explicitly
  says: *if A4 isn't earned, lead with A2/A3/A5.* **We are now in that situation.**

The simulator is a sampling calorimeter: `N` layers, each a PbWO4 **absorber** + liquid-Ar
**gap** (the active/readout medium). Design knobs: per-layer absorber & gap **thickness**.

---

## 2. Infrastructure that was built (this works and is committed)

### C++ (`/eos/user/j/jeffkrup/agentic/hepemshow`, cloned at tag `v1.0-paper`)
- **Phase 0 — per-layer geometry** (branch `phase0-perlayer-geometry`): generalized the
  uniform `(absorber,gap)×N` geometry to **per-layer** `std::vector<G4double>` thicknesses,
  each a registered AD input. New CLI: `--abs-profile`, `--gap-profile`,
  `--abs-layer i:v[:dot]`, `--gap-layer i:v[:dot]`. Reverse writes `barInputsPerLayer`
  (2N+1 rows). Validated: fwd/rev AD agree ~1e-14; per-layer sums == legacy aggregate.
- **Phase A — per-layer GAP energy as an AD *output*** (branch `phaseA-perlayer-gap-energy`):
  per-layer gap (active) energy is now differentiable. Forward writes `edeps_gap_<seed>`;
  reverse accepts `--bar-gap` adjoints. Validated fwd/rev to 8e-14 **w.r.t. absorber**
  (NOTE: never validated w.r.t. *gap thickness* — see §3 finding 2).
- Both binaries built: `hepemshow/build/HepEmShow` (forward), `build_reverse/HepEmShow`
  (reverse). Portable patches saved in `patches/` in the python repo.

### Python (`/eos/user/j/jeffkrup/agentic/agentic-detector-design`, branch `phase1-e2-agentic-loop`)
See `docs/EXPERIMENT_RESULTS.md` §"Code map" for the full file list. Key pieces:
- `tools/sim.py` — the binary wrapper (temp-dir isolation, flag-hash cache, forward/reverse,
  per-layer + gap adjoints, **subprocess timeout** `sim.subprocess_timeout_s` in config).
- `tools/observables.py` — total_edep, peak_edep, shower_max_depth, visible/front_fraction.
- `tools/schemas.py` — `DesignPoint` (+ per-layer `abs_profile`/`gap_profile`), `Region`,
  `regions_to_profiles`.
- `tools/optimizer.py` — inner optimizers (see §4: several variants; **use the L-BFGS-B one**).
- `tools/gap_signal_target.py`, `tools/constraints.py`, `tools/autopsy.py`,
  `tools/reliability.py`, `tools/fd_check_perlayer.py`.
- `agent/structural_moves.py` (DesignRepresentation + Split/Merge/…), `agent/e2_loop.py`
  (bilevel Arm A vs baseline Arm C), `agent/orchestrator.py` (older scalar loop).
- `experiments/` — the ceiling/sweep scripts (containment, gapsignal, netsignal,
  grad_reliability, resolution_ceiling, depth_resolution).
- `condor/` — **HTCondor** durable-compute wiring (see §5).

---

## 3. The journey & findings — READ THIS so you don't repeat dead ends

### What we tested and what happened
| Objective | Result | Why |
|---|---|---|
| `total_edep` (total deposited energy) | **uniform optimum** | sum of a peaked gradient is ~flat in layer index |
| containment (= total_edep) | **uniform optimum** | global effect; position-insensitive when under-contained |
| flat-profile MSE (match a flat longitudinal profile) | NO (unreachable target) | thickness can't reshape the shower that much |
| **net-signal** `L = −E_vis + μ·gap_material` | **FALSE "structure wins" → artifact** | see below |
| **energy resolution** `σ(E_vis)/⟨E_vis⟩` | **uniform is BEST; structure HURTS** | total-energy resolution needs the *whole* profile sampled |

### Finding 1 — aggregate objectives don't reward structure (physical, robust)
The per-layer **sensitivity** `∂E_l/∂θ_r` *is* peaked (shower is peaked). But every objective
above is an **aggregate** (a sum or ratio of sums). The gradient of a sum is the sum of the
peaked per-layer gradients — which is ~flat. **A peaked integrand does not make the integral's
optimum peaked.** At fixed budget, an aggregate is best served by *balanced* (uniform)
sampling. This is real physics, not a bug. The strongest evidence is the **resolution ceiling
test** (gradient-free): uniform σ/μ = 0.032 vs bump 0.077 / front 0.070 / tail 0.055 — uniform
wins decisively (see `docs/EXPERIMENT_RESULTS.md`).

### Finding 2 — the AD gap-thickness gradient is ~10× biased (the big one)
The "net-signal structure win" (ratios 0.67–0.85, bang-bang front-thick/tail-thin profiles)
was an **artifact**. Two independent gradient-free checks killed it:
- **Direct loss comparison:** a non-uniform config has *higher* net-signal loss than uniform;
  the true optimum is uniform-thin (material cost dominates).
- **AD-vs-FD (8 seeds, 5000 events):** `d(E_vis)/d(gap-thickness)` AD = **1951 ± 24 (SNR 82)**
  while realized FD = **198 ± 3** → **ratio 9.86**. High SNR ⇒ this is **bias, not noise**.
The pathwise AD gradient w.r.t. *gap thickness* is systematically ~10× too large. Likely
causes: it misses the **score-function/discrete-fluctuation** term (track-counting across
gaps), and/or the project's `stop-gradient` regularization severs exactly the
boundary-crossing channel. We had validated gap-energy AD only w.r.t. *absorber*, never *gap*.
**Lesson: gate every gradient with the AD-vs-FD reliability check (`tools/reliability.py`)
before trusting it for optimization.**

### Lessons that cost us a lot of time (don't re-pay them)
- **K=n ⊇ K=1**: a converged every-layer-free optimum can NEVER be worse than uniform. If a
  "ceiling" run reports ratio > 1, the optimizer is **under-converged** — not a real result.
- **Hand-rolled trust-region GD is fragile.** We hit three separate convergence bugs
  (lr-limited steps; a *global* inf-norm trust-region where one absorber coordinate's big step
  crushed all gap steps; early-stop when a sim hit the subprocess timeout). **Use
  `optimize_inner_netsignal_lbfgs` (scipy L-BFGS-B) — it dropped the hand-tuned `lr`/
  `trust_region` and the per-coordinate clip pathologies.** (The GD variants are still in
  `tools/optimizer.py` for reference but are not trustworthy at scale.)
- **Always check the VALUE before the gradient.** The gradient-free loss/resolution
  comparisons are what caught the artifact. Do value-only ceiling tests first.
- **Compute/infra:**
  - HTCondor `RemoteUserCpu` **lags** — do NOT read a "frozen" counter as a hung job (we
    killed jobs prematurely doing this). Read the actual result rows.
  - Detached local jobs (`setsid`/`nohup`) **die when the session hops lxplus nodes**. Use
    **HTCondor** for anything that must outlive a node tenure.
  - Reverse-AD at 40 layers is ~0.22 s/event; the gap-output tape made it heavier. 5000-event
    reverse sims can blow a 300 s timeout under load → set `sim.subprocess_timeout_s` high
    (currently 1800) and/or use fewer events.

---

## 4. Inner optimizers (which to use)
`tools/optimizer.py` contains, in order of trustworthiness:
- **`optimize_inner_netsignal_lbfgs(rep, mu, constraints, max_iters, n_events, seed, ctrl)`**
  — scipy **L-BFGS-B**, analytic AD gradient (`jac=True`), box bounds from constraints, fixed
  seed (deterministic objective). **This is the one to use.** (Caveat: it relies on the AD
  gradient, which for gap thickness is the unreliable one from Finding 2 — gate with FD.)
- `optimize_inner_netsignal`, `optimize_inner_gapsignal`, `optimize_inner_containment`,
  `optimize_inner_profile`, `optimize_inner` — hand-rolled trust-region GD. **Fragile; for
  reference only.** The `--optimizer {gd,lbfgs}` flag in `experiments/e2_discriminate_gapsignal.py`
  selects between them.

---

## 5. HTCondor (how to run durable compute) — IMPORTANT lxplus gotchas
The reliable pattern (the *only* one that survives session/node hops):
- Wrappers live on **AFS**: `/afs/cern.ch/user/j/jekrupa/condor_bin/run_*.sh`. Condor logs
  go to AFS (`/afs/cern.ch/user/j/jekrupa/condor_logs/`). Submit files in repo `condor/*.sub`.
- **Submit from an AFS cwd** (`cd /afs/cern.ch/user/j/jekrupa/condor_bin`). The schedd
  **rejects any `/eos` path in the submit context** (including an EOS `initialdir`). The
  wrapper `cd`s into the EOS repo at *runtime* on the worker and writes results to EOS.
- After editing a `run_*.sh`, re-copy it to the AFS path the `.sub`'s `executable` points at.
- Existing submit setups: `condor/{gapsignal,gradrel,resceil,depthres}.sub`. Example:
  ```bash
  cd /afs/cern.ch/user/j/jekrupa/condor_bin
  condor_submit /eos/user/j/jeffkrup/agentic/agentic-detector-design/condor/<X>.sub \
    -append "TAG = <tag>" \
    -append "arguments = <python module args>"
  ```
- Results land in `experiments/*.jsonl` on EOS; poll with `condor_q <cluster>` /
  `condor_history`. Jobs are durable — independent of this session/credits.

---

## 6. IN-FLIGHT: shower-max-depth resolution test (the open opportunity)

This is the *shape* objective — the remaining physically-motivated place structure could win
(finer sampling near shower max → better localization of the peak depth). **It is the direct
test of the "peaked shower ⇒ structure should help *something*" intuition.**

- **Built & validated:** `experiments/depth_resolution.py`. Per-event shower-max depth via a
  **parabolic sub-layer interpolation in physical mm**; metric `σ(x_max)` (lower = better).
  Designs `uniform` / `fine_at_max` / `coarse_at_max` at **fixed total length AND total gap**
  (so only *granularity* varies). Condor-wired (`condor/depthres.sub`, `run_depthres.sh`).
  Self-test PASSES.
- **GATE IS DONE — the test is LAUNCH-READY (verified end-to-end).** The per-event dump is
  now env-gated: `hepemshow/Simulation/src/SteppingLoop.cc:46` reads `HEPEMSHOW_OUTPUT_ALL`
  (default-off; committed on `phaseA-perlayer-gap-energy` as `9ceda1e`), the **forward binary
  is rebuilt** with it, `experiments/depth_resolution.py` sets that env var in the sim
  subprocess (committed `47bd6b4`), and `run_depthres.sh` is deployed to AFS. A local smoke
  produces real numbers (uniform σ(x_max) ≈ 24.8 mm vs fine_at_max ≈ 29.0 mm at 20 layers /
  500 ev — **NOT decisive**, just proof it runs; the 40-layer/10000-event matrix is the test).
  Nothing is blocked — just launch.
- **Run it** (value-only, gradient-free — robust to the AD bias):
  ```bash
  cd /afs/cern.ch/user/j/jekrupa/condor_bin
  for d in uniform fine_at_max coarse_at_max; do for s in 1 2 3 4 5 6; do
    condor_submit /eos/user/j/jeffkrup/agentic/agentic-detector-design/condor/depthres.sub \
      -append "TAG = depthres_${d}_s${s}" \
      -append "RUN_ARGS = --design ${d} --n-layers 40 --n-events 10000 --seed ${s} --out-jsonl experiments/depthres_${d}_s${s}.jsonl"
  done; done
  ```
  Aggregate `σ(x_max)` per design. **If `fine_at_max` < `uniform` < `coarse_at_max`**, that is
  the first genuine "structure helps" result → then (and only then) it's worth making depth
  resolution *differentiable* and running the agent loop on it. **If not**, structure is dead
  even for shape objectives in this testbed.
- **Honest confound:** changing per-layer thickness changes both *granularity* and (slightly)
  *local shower development*. We hold total length/gap/absorber fixed to kill the gross "more
  material" confound, but a residual coupling remains; read a positive result accordingly.

---

## 7. Recommended next steps (in priority order)
1. **Finish the depth-resolution test** (§6): the one-line C++ flag + forward rebuild, then the
   value-only matrix. This is cheap and decisive for whether *any* objective rewards structure.
2. **If depth resolution shows structure:** make it differentiable (scope in the resolution
   investigation — it's a Phase-A-sized change: register per-event `E_vis`, `E_vis²` /
   shower-max depth as AD outputs), **gate the gradient with AD-vs-FD**, then run the agent
   bilevel loop (`agent/e2_loop.py`) on it as the real A4 test.
3. **If it doesn't:** pivot the writeup to **A2/A3/A5** — the negative-structure result + the
   AD-reliability finding (Finding 2). Both are honest, real, and publishable. Characterize
   *why* the gap-thickness gradient is 10× off (score-function term vs stop-gradient
   regularization) — that's a clean differentiable-transport-reliability contribution.
4. **Don't** keep hunting aggregate-energy objectives — they're exhausted (uniform every time).
5. **Don't** trust any gradient-based "structure win" without a gradient-free value check first.

---

## 8. Claim discipline (from the original handoff, still binding)
State only what's earned. As of now: **A4 is unsupported in this testbed; A5 (gradient
reliability) is demonstrated; the structure question for *shape* objectives is open and
testable.** Lead with what's true.

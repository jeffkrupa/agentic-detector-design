# Claude Session Handoff

Branch: `phase1-e2-agentic-loop`. Repo: `/sdf/data/atlas/u/jkrupa/agentic/agentic-detector-design`.
Role: orchestrator; delegate code/sim work to subagents. See memory files for full detail.

## 1. Goal
- Ground agentic detector design in EXACT AD gradients of a differentiable EM calorimeter sim (hepemshow+g4hepem+CoDiPack).
- Bilevel thesis: agent edits STRUCTURE (outer), AD optimizes continuous params (inner). Test whether structure helps (A4 strong; A2/A3/A5 fallback = sensitivity-pricing + gradient-reliability).

## 2. Task state
- DONE: SDF toolchain rebuilt + verified; Slurm pipeline; depth-res experiment; e/gamma PID experiment (10 GeV + 1 GeV + sensitivity); differentiable-proxy test.
- IN PROGRESS / BLOCKED: **FD-reference validation is UNRESOLVED** — the audit subagent died on a 429 budget-exceeded error before returning. This is the single open pivotal question (see §5).

## 3. Files changed/created this session
- `slurm/depthres.sbatch`, `slurm/submit_depthres.sh` — Slurm worker + fan-out for depth-res.
- `slurm/epid.sbatch`, `slurm/submit_epid.sh` — Slurm worker (separation|sensitivity modes) + fan-out for PID.
- `tools/separation.py` — pure-numpy Fisher, ROC-AUC, ridge k-fold CV classifier, grid-fair control metric.
- `experiments/epid_separation.py` — 2-particle (e-/gamma) driver; native CV-AUC + grid-fair + Fisher per (design,seed).
- `experiments/epid_sensitivity.py` — FD region-thickness sensitivity of separation (front/mid/rear).
- `experiments/_proxy_test.py` / `_proxy_grad.py` / `_proxy_fd.py` / `_proxy_uniform.npz` — SCRATCH: differentiable mean-profile proxy value+gradient+FD. (dev artifacts, not committed)
- `docs/CLAUDE_SESSION_HANDOFF.md` — this file.
- `config.yaml`, `.venv/`, `.sim_cache/` — local only, gitignored. config.yaml points at diffcalo build.

## 4. Key design decisions
- Rebuilt sim at `/sdf/data/atlas/u/jkrupa/agentic/diffcalo` (CoDiPack v3.1.0, g4hepem v1.0-paper, hepemshow branch phaseA-perlayer-gap-energy). NEVER touch `/sdf/data/atlas/u/jkrupa/hepemshow` (protected).
- Slurm: partition `roma`, account `atlas:usatlas`. Workers source LCG view `/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt/setup.sh` (python3.11+numpy+pyyaml) — NOT the login .venv (dangles on el8.6 nodes w/ only py3.6). Wrapper needs `set -eo pipefail` (NO -u) + `set +e` around the LCG source (unbound $COMPILER; gnuplot nonzero probe).
- Fixed-budget structural designs via `build_scale_profile(design,N,shower_max_layer)` + `profiles_from_scale` (holds total length/gap/absorber; only granularity distribution varies).
- Always use canonical ctrl flags `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`, identical across all runs.
- Value-first discipline: gradient-free value comparison before trusting any gradient (project burned twice by precise-but-wrong gradient results).

## 5. Findings (not all in docs)
- **Depth-resolution result was an ARTIFACT** (confirmed): backwards ordering came from parabolic vertex-bracket quantization + extensive-edep argmax bias. Grid-fair metric → sigma identical (5.30/5.33/5.32mm). Depth resolution is FLAT across designs.
- **PID e/gamma: first POSITIVE structure signal but MARGINAL.** Native CV-AUC fine>uniform>coarse (0.88/0.84/0.79 @10GeV; 0.92/0.85/0.77 @1GeV) but grid-fair control shows the true effect is only ~+0.003 AUC and did NOT grow at 1 GeV. Larger native gap is mostly front-window-depth confound.
- **Multi-seed FD sensitivity (fisher_front3): pure noise** — all regions consistent with zero; A2 "gradient points to front" NOT established at this stat level.
- **Differentiable proxy:** inverse-variance Fisher (denominator DETACHED; sim has var_E but no d(var)/d(input)) REPRODUCES the value ordering fine>uniform>coarse. Plain mean-distance FAILS (ranks coarse first).
- **DISPUTED / LIKELY WRONG:** proxy AD gradient reported AD/FD ~0.03-0.1 (10-30x "bias"). Project lead: real AD-vs-FD undershoot on this sim is only ~20% (AD/FD~0.8), and AD fwd-vs-reverse selftest = 1e-13. So the 10-30x is almost certainly a BROKEN FD (wrong epsilon and/or MC-noise-dominated; D=Σ(diff_l)² is 2nd-order so its ΔD is noise-prone; possible common-random-number/seed desync). DO NOT trust the "AD gradient is biased" conclusion until FD is validated.

## 6. Known failures/errors
- FD-validation subagent: killed by 429 budget-exceeded (incomplete, no result).
- Sensitivity FD: noise-dominated at 5000ev×6seed.
- Prior known: d/d(gap) AD flagged ~10x vs FD (A.2 in docs/EXPERIMENT_RESULTS.md) — may ALSO be an FD-epsilon artifact; re-examine with validated FD.

## 7. Commands that matter
- Selftest: `source .venv/bin/activate; python -m tools.sim --selftest` (PASS = rel.diff ~1e-13).
- Submit PID: `bash slurm/submit_epid.sh [N_LAYERS] [N_EVENTS] [ENERGY] "SEEDS" "DESIGNS"`.
- Aggregate results: read `experiments/epid_*_s*.jsonl` (10GeV), `experiments/epid_*_e1000.jsonl` (1GeV), `experiments/epid_sensitivity_s*.jsonl`.
- Batch python on nodes: source LCG view (NOT venv). Monitor: `squeue -u jkrupa`.

## 8. Next steps (fresh session)
1. **RE-RUN THE FD VALIDATION** (the blocked task). First validate FD machinery on a LINEAR control (total_edep, AD d/da≈-1310, d/dE≈+0.775) via an epsilon scan with COMMON RANDOM NUMBERS at N=2000 & 10000 — expect AD/FD~0.8 plateau. Then re-diagnose proxy D FD with the stable epsilon + CRN + high N. Decisive question: does AD/FD move to ~0.8, or stay ~0.03-0.1?
2. If FD was broken → proxy AD gradient likely FINE → exact-AD bilevel story back on table → build agent inner-loop on proxy B (inverse-var Fisher).
3. If AD genuinely biased → pivot to A2/A3/A5 reliability/negative-result paper (the "precise-but-wrong" theme, demonstrated 3x).
4. Decide strategic framing: chase exact-AD/A4 vs write reliability+negative-structure paper. (Even if gradient is clean, grid-fair PID effect is only ~0.003 AUC — small hill.)
5. Clean up scratch `experiments/_proxy_*` files; gitignore `slurm/logs/`.

## 9. Avoid repeating
- Don't trust a tight/monotone/impressive number without a grid-fair / value / validated-FD cross-check (burned 3x: net-signal GD, depth-res quantization, proxy 10-30x).
- Don't conclude "AD biased" from an unvalidated FD. High SNR ≠ correct, but also FD ≠ ground truth until its epsilon is validated.
- Don't use login .venv in Slurm jobs (dangles on el8.6). Don't run `set -u` around the LCG source.
- Don't chase aggregate-energy objectives (all → uniform by construction).
- Don't launch huge matrices; forward-only value runs are cheap, reverse-AD at high stats is slow (~0.22s/ev).

## 10. Open questions / assumptions
- Is the 10-30x AD/FD purely an FD-epsilon/noise artifact? (blocking Q)
- Is the ~0.003 grid-fair PID effect large enough to be worth an agentic demonstration, even if the gradient is trustworthy?
- Does the prior A.2 "gap gradient 10x biased" survive a validated FD, or was it the same artifact?
- Assumed: subagents may read/run/write without asking permission; confirm before slow rebuilds / large batches / destructive-outward actions.

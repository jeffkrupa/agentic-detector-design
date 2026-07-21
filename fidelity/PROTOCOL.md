# Wave protocol — agentic gradient-fidelity repair loop

The project: an agent iteratively improves the derivative fidelity of the
differentiable simulator (HepEmShow + G4HepEm/CoDiPack) against empirical
finite-difference truth. Unit of work = **wave**:
hypothesis → C++ edit → gate → validation → verdict → ledger entry.
Strictly serial; one wave in flight at a time; human verdict checkpoint at
the end of each wave.

## Git layout

- **hepemshow** (`/eos/user/j/jeffkrup/agentic/hepemshow`) and **g4hepem**
  (`/eos/user/j/jeffkrup/agentic/g4hepem`): long-lived integration branch
  `agent-knobs` cut from `phaseA-perlayer-gap-energy` (hepemshow @ 9ceda1e)
  / master (g4hepem @ 91cbee3). One short branch per knob
  (`knob/local-frame-anchor`, `knob/el-mfp-cap`, …). Accepted knobs merge to
  `agent-knobs`, tagged `wave-NN-accepted`. Rejected knobs stay as branches
  with the verdict in the ledger (a measured null is paper content).
- **Shared binaries `build/` and `build_reverse/` never move off baseline.**
  Agent builds live in `build_agent_fwd/` and `build_agent_rev/`.
  Caveat from RECON §E: physics `.icc` edits under
  `g4hepem/install*/include/G4HepEm/` are seen by all build dirs of that
  mode → every change MUST be behind a default-off knob (KeepPrimal
  pattern) so baseline behavior is bit-identical with knobs off.
- **agentic-detector-design** (this repo): the loop lives in `fidelity/`.
  Ledger = `fidelity/LEDGER.md` (prose) + `fidelity/ledger.jsonl` (one JSON
  object per wave).

## Provenance rules

- Every validation row records: hepemshow SHA, g4hepem SHA, knob flags,
  seeds, n_events, binary path.
- `tools/sim.py` cache keys by flag-hash only — **must be extended to
  include the binary git revs / build dir before any wave-1 validation runs**
  (else knob-branch binaries collide with baseline cache entries).
- The sim is not bit-reproducible at fixed seed → always quote seed-scatter
  errors, never single-seed numbers.

## Wave steps

1. **Diagnose.** Hypothesis + *pre-registered predicted signature* written to
   the ledger BEFORE any edit. (No post-hoc storytelling; the prediction is
   what makes acceptance a test.)
2. **Implement.** Knob on its own branch, default-off, derivative-only,
   KeepPrimal pattern (`primal + (x−stop_grad(x))·reg`). CLI flag +
   config plumbing per RECON §C conventions.
3. **Gate** (local, seconds-to-minutes, before any condor):
   a. Both modes build in `build_agent_fwd/rev`.
   b. **Primal identity**: `edeps_<seed>` mean_E/var_E columns byte-identical
      to baseline with knob OFF *and* with knob ON (derivative-only edits),
      2 seeds × 2k events.
   c. Forward = reverse derivative spot check (same seed, printed digits).
   d. NaN scan on the derivative columns.
4. **Validate** (one small condor batch, ~10–30 CPU-h): AD (knob on vs off)
   against the standing FD truth. Fitness = gradient error at fixed compute
   (bias AND variance AND NaN rate), per-layer. See methodology facts below.
5. **Verdict + ledger.** Accept/reject against the pre-registered prediction.
   Accept ⇒ merge to `agent-knobs`, tag. Either way: append full ledger
   entry, update LEDGER.md, next wave takes the top remaining candidate from
   RECON §D.

## Validation methodology facts (measured; do not rediscover)

- **CRN seed-pairing is worthless here**: sim non-reproducibility decorrelates
  ± runs (measured variance reduction 1.16× gap / 0.87× absorber). Use the
  unpaired FD estimator over all surviving seeds; SE from seed scatter.
- **FD noise ∝ 1/h** (per-side σ(E_tot) ≈ 3 MeV per 20k-event unit,
  h-independent). Small-h FD is unwinnable; use large h (0.05–0.2 mm) and a
  Richardson/h² extrapolation with the existing h=0.02 1M-event point as
  low-h anchor if truncation matters.
- **Estimator**: core-plateau ratio over a fixed layer window (e.g. L5–18),
  pre-registered. Never headline layer-summed totals (the gap total is a
  cancellation of opposite-sign contributions; absorber total is
  tail-noise-dominated).
- **Standing FD truth** (baseline design: uniform 50 layers, a=2.30 mm,
  g=5.70 mm, 10 GeV e-, t=400, flags `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`):
  `experiments/perlayer_adfd_1M_summary.json` — absorber AD/FD plateau
  0.77–0.80 (k=0.754±0.007); gap ~2–3× depth-dependent (k=2.67, χ²/ndof=385/42
  ⇒ quote ±0.31, not ±0.10); gap L49 AD=−265 vs FD=−5.4 (ratio 49±11);
  energy exact (n=1000-level so far). FD truth is knob-independent
  (derivative-only edits) ⇒ reusable across all waves.
- Job sizing: ≤20k events/job at a ≤ 2.5 mm, ≤10k at a > 2.5 mm; 7200 s
  in-process sim timeout; condor from AFS cwd, `workday` flavour, wrapper
  dir `/afs/cern.ch/user/j/jekrupa/condor_bin/`, per-(config,seed) jsonl
  outputs; runner pattern = `experiments/perlayer_adfd_1M.py`.

## Ledger entry schema (ledger.jsonl)

```json
{
  "wave": 1,
  "knob": "--stopgrad-local-frame",
  "branch": "knob/local-frame-anchor",
  "hypothesis": "…",
  "predicted_signature": "…",
  "status": "planned | implemented | gated | validated | accepted | rejected",
  "shas": {"hepemshow": "…", "g4hepem": "…", "agentic": "…"},
  "gate": {"primal_identical": null, "fwd_eq_rev": null, "nan_scan": null},
  "validation": {"batch": null, "before": {}, "after": {}, "events": null},
  "verdict": null,
  "date": "2026-07-21"
}
```

## Session bootstrap

Fresh sessions start from `fidelity/BOOTSTRAP.md`. This session-independence
is deliberate: no conversation is load-bearing.

# Session bootstrap — gradient-fidelity repair project

Read these, in order, to be fully operational (≈15 min of reading replaces
the entire founding conversation):

1. `fidelity/PROTOCOL.md` — what a wave is, git layout, gate steps,
   validation methodology facts (CRN pairing worthless, FD noise ∝ 1/h,
   core-plateau estimator), provenance rules, condor conventions.
2. `fidelity/LEDGER.md` + `fidelity/ledger.jsonl` — where the loop stands:
   which wave is in flight, its pre-registered prediction, past verdicts.
3. `fidelity/RECON.md` — the C++ derivative-path map (file:line), the
   non-differentiable construct inventory, the ranked intervention list
   (§D — the wave queue), build facts (~30 s rebuilds, build_agent_* dirs).
4. `experiments/perlayer_adfd_1M_summary.json` — the standing 1M-event
   AD-vs-FD trust map (the "before" measurement and FD truth reference).
5. `docs/PAPER_PLAN.md` — the earlier audit-paper plan; its trust-map and
   methodology sections feed this project; its demo sections are superseded
   by the repair loop.

Standing working agreements (from the founding session):
- Operator: Jeffrey Krupa. Act as orchestrator; code tasks go to subagents
  that summarize concisely. One wave in flight at a time; human verdict
  checkpoint ends each wave.
- Platform is HTCondor (lxplus), NOT SLURM. Submit from AFS cwd only.
- No condor submission or >2k-event local runs without explicit user
  approval per batch.
- Shared baseline binaries (hepemshow build/, build_reverse/) are never
  modified; all C++ changes behind default-off knobs (KeepPrimal pattern).
- Parked by user direction: the L49 gap anomaly as a *research topic* (it
  IS wave 1's diagnostic target, but don't spiral into it beyond that);
  paper items E6/E2a/E1d (deferred, see PAPER_PLAN + critics' verdicts in
  git history of docs/).
- Established context: structure/redistribution helps no objective
  (memory: project-state-2026-07); depth-resolution win was an estimator
  artifact; absorber gradient quantitatively correct (~0.8×), gap
  directionally correct (~2-3× inflated) — user's words.

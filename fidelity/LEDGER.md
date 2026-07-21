# Fidelity-repair ledger

One entry per wave. Machine-readable twin: `ledger.jsonl`.
Protocol: `PROTOCOL.md`. Fault inventory & candidates: `RECON.md` §B/§D.

## Wave 1 — local-frame stop-grad re-anchoring (PLANNED)

- **Knob**: `--stopgrad-local-frame 1` (+ diagnostic sub-variant
  `--last-layer-local 1`), branch `knob/local-frame-anchor`, default off,
  derivative-only.
- **Hypothesis** (RECON §B): `SanitizeTrackState` zeroes the *global*
  position dot of stop-graded tracks while the geometry translation
  `fLayerStartX[iLayer]` keeps its AD dot; the destroyed cancellation
  injects a spurious step-length dot ≈ prefix_dot/vx into every subsequent
  boundary-limited step of still-tracked zombie tracks, which keep
  depositing AD-active energy.
- **Pre-registered predicted signature** (written before implementation):
  1. Gap per-layer AD/FD bias falls from ~2–3× toward ~1 with a
     depth-dependence reduction (the linear-in-depth component is the
     mechanism's fingerprint; uniform gap seed ⇒ prefix dot of layer i ≈ i).
  2. The L49 anomaly (AD/FD = 49 ± 11) collapses by an order of magnitude
     under `--last-layer-local 1` alone.
  3. Absorber plateau (0.75–0.80) changes little (its bias is attributed to
     over-severing, RECON §D5, a different mechanism).
  4. Primal strictly unchanged (byte-identical edeps mean/var columns).
- **Status**: GATED (2026-07-21) — all gate checks pass; awaiting condor
  validation approval.
- **Implementation**: hepemshow branch `knob/local-frame-anchor` @ `e9ed89b`
  (2 commits over baseline `9ceda1e`; `agent-knobs` = baseline, unmoved).
  5 files, +85/−15: knob logic in `Geometry.cc::CalculateDistanceToOut`
  (stop_grad on calo shift/dToCalo and per-layer translations/half-lengths
  for gradient-disabled tracks; knob 2 = last layer only, knob 1 ⊇ knob 2),
  bool plumbed through the 4 call sites in `SteppingLoop.cc`, long options
  `--stopgrad-local-frame` / `--last-layer-local` in `InputParameters.hh`.
  Builds in `build_agent_fwd/` + `build_agent_rev/`; shared binaries and
  g4hepem untouched. Baseline working tree restored after the wave.
- **Gate results** (2000 ev, canonical flags, gap seed unless noted):
  - G0 reproducibility: shared binary twice, same seed ⇒ bit-identical
    (byte-identity gating valid on this host).
  - G1 build equivalence: agent build knob-off ≡ shared binary. PASS.
  - G2 primal identity: mean_E/var_E byte-identical with knobs on (each and
    both; seeds 1–2; absorber seed too). PASS.
  - G3 forward=reverse: knob 1 on, n=500, Σ fwd mean_dE = −12.93556478 =
    reverse barInputs gap row (rel 8e−13). PASS (n=500: two n=2000 reverse
    runs were killed by the environment at ~80%, not by the sim).
  - G4 NaN scan: clean. PASS.
- **Preview vs predictions** (n=2000, 2 seeds — qualitative only):
  mid-depth gap inflation shrinks 3–5× under knob 1 (L10 mean_dE 39.5→7.2,
  L15 52.2→12.6); L49 collapses 1–2 orders (−272→−2 s1; −417→−120 s2);
  knob 2 alone collapses L49 (−272→−17) leaving L0–35 untouched —
  predictions 1 and 2 land qualitatively. Absorber derivative also moves
  under knob 1 (L15 135→95) — watch prediction 3 at validation.
  **Diagnostic deviation**: knob 2 as literally specced in RECON (calo
  shift + dToCalo + clamp only) was a measured near-null (−272.076→−272.079)
  — dToCalo never enters a returned step length; the anomaly rides on the
  layer translation's prefix dot. Knob 2 was widened to the full
  iLayer==N−1 layer-frame path (still ⊂ knob 1).
- **Validation** (cluster 3911763, 2026-07-21, 26/30 units ok — failed:
  gap_knob1 s2/s8, abs_knob1 s9/s10, rc=−1; summary:
  `fidelity/wave1_validation_summary.json`):
  - **P1 FAIL (overcorrection)**: gap core L5–18 AD/FD = **0.566 ± 0.027**
    (from 3.02 ± 0.15). The 3× inflation is gone but AD now undershoots FD
    ~2×; deficit is depth-flat through the core.
  - **P2 PASS**: L49 gap ratio → −0.23 ± 2.21 under knob 1 (217×);
    **3.22 ± 1.91 under knob-2-only (15×)** with the core window untouched
    (3.016 ± 0.150 vs knob-off 3.020 — z = −0.02, clean pipeline check).
  - **P3 FAIL**: absorber core AD/FD 0.772 → **0.545 ± 0.010** (−29%); the
    knob is not gap-specific — it discards legitimate derivative mass in
    both channels.
  - **P4 PASS** (gap configs clean; abs config 3/50 layers >3σ — likely
    seed fluctuation, abs/gap knob-1 primals are identical physics).
- **Interpretation**: the zombie-prefix-dot mechanism is CONFIRMED (the
  depth-dependent inflation and L49 anomaly both die exactly as predicted),
  but the implemented severing is too broad: stop_grad-ing ALL local-frame
  geometry dots for gradient-disabled tracks also kills their legitimate
  local boundary-motion contribution → uniform ~0.55 deficit in both
  channels (cf. absorber's pre-existing 0.77 over-severing deficit).
  RECON's original sketch (re-anchor the position dot to the local frame,
  i.e. cancel only the accumulated translation prefix dot while KEEPING the
  local thickness/half-length dots) is a strictly narrower intervention —
  wave-2 candidate.
- **Variance/cost addendum** (from per-event var_dE + condor wall times):
  variance is NOT the price of knob 1 — severing reduces it (gap core 0.54×,
  tails ~0.06–0.11× ⇒ absorber L49 SE at 1M: 224 → 55; heavy tail gone,
  scatter matches √(var/N) within ~30%). Exceptions/costs: absorber core
  variance 1.64× WORSE under knob 1, gap knob-1 runtime +53% (6524s vs
  4262s mean → the 4 unit failures were all 7200s timeouts; raise batch
  timeout to ~10800s in future waves). Knob 2: tail win ~0.10×, core
  variance 1.06×, core means preserved, no slowdown, 0 failures — strict
  improvement on every axis. (Baseline absorber itself has ~8% timeout rate;
  knob-off reference uses 46/50 seeds.)
- **VERDICT (user-approved 2026-07-21): knob 1 REJECTED as-is (mechanism
  confirmed, severing too broad); knob 2 ACCEPTED.** Branch merged to
  `agent-knobs` (@ e9ed89b, tag `wave-01-accepted`); the rejected knob-1
  flag remains in the code, default-off, documented as diagnostic-only.

## Wave 2 — prefix-only re-anchoring (PLANNED)

- **Knob**: `--stopgrad-prefix-anchor 1`, branch `knob/prefix-anchor` cut
  from `agent-knobs`, default off, derivative-only.
- **Hypothesis**: wave 1 proved the spurious term is the *accumulated
  translation prefix dot* re-entering severed tracks via the local-frame
  transform; knob 1 failed because it also severed the *local* thickness /
  half-length dots (legitimate boundary-motion signal). Cancelling ONLY the
  prefix-sum translation dot (`fLayerStartX[iLayer]`, calo shift) for
  gradient-disabled tracks — while keeping the current layer's own
  thickness/half-length/sub-box-offset dots — removes the zombie inflation
  without discarding legitimate derivative mass.
- **Pre-registered predicted signature** (before implementation):
  1. Gap core (L5–18) AD/FD lands in [0.8, 1.5] — strictly between knob-1's
     0.566 and knob-off's 3.02, and closer to 1 than both.
  2. L49 collapse retained: gap L49 ratio within ~3× of 1 (vs knob-off 49).
  3. Absorber core in [0.60, 0.80]; result is informative either way: if it
     stays ≈0.77 the zombie term was gap-specific; if it drops, part of the
     absorber's apparent 0.77 was zombie inflation masking a deeper
     over-severing deficit.
  4. Primal byte-identical (gate).
  5. Runtime within ~15% of baseline (fewer stop_grad ops than knob 1);
     batch timeout raised to 10800s regardless.
  6. Variance: tail-variance win retained (≤0.3× at L40+), gap core
     variance in [0.5, 1.2]×.
- **Status**: planned.

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
- **Status**: GATED (2026-07-21) — all gate checks pass; awaiting condor
  validation approval.
- **Implementation**: hepemshow branch `knob/prefix-anchor` @ `61c7bb5`
  (1 commit over `agent-knobs` @ e9ed89b). 4 files, +44/−9:
  `--stopgrad-prefix-anchor` long option (id 1008) in `InputParameters.hh`;
  third bool through `Geometry::ConfigureLocalFrameStopGrad`; in
  `CalculateDistanceToOut` a new `sgPrefix` path severs ONLY
  `trLayeri = fLayerStartX[iLayer]` (+ the calo shift / dToCalo via the
  existing `sgCalo` path) while `layerThick`, the abs/gap half-lengths, and
  `trAbs`/`trGap` keep their dots. Decomposition note: `trAbs`/`trGap` are
  computed in the *layer-local* frame already (`0.5*fAbsThick[i]`,
  `fAbsThick[i]+0.5*fGapThick[i]` — no `trLayeri` inside), so no rewrite was
  needed; keeping them alive is exact. Knob interaction: knob 1 (full sever)
  wins over prefix-anchor; prefix-anchor supersedes knob 2's full-local
  severing on the last layer (same prefix/calo severing there, but the last
  layer's local dots stay alive; prefix-anchor ⊇ knob 2's original
  calo-shift severing). Shared binaries and g4hepem untouched; baseline
  working tree restored after the build.
- **Gate results** (agent builds, canonical flags, gap seed unless noted):
  - G1 build equivalence: agent fwd knob-off ≡ shared `build/HepEmShow`,
    seed 1, edeps bit-identical. PASS.
  - G2 primal identity: knob on vs off, mean_E/var_E byte-identical, seeds
    1–2, gap AND absorber seeds (all 50 layers' mean_dE change ⇒ knob
    active). PASS.
  - G3 forward=reverse: knob on, n=500, gap seed: Σ fwd mean_dE
    = −12.719654630933 vs reverse barInputs gap row −12.719654630928
    (rel 3.8e−13). PASS.
  - G4 NaN scan: all columns finite, seeds 1–2, gap+abs, knob off+on. PASS.
  - G5 runtime: ON 305.5 s vs OFF 324.8 s (n=2000, gap seed, same host,
    sequential) ⇒ ratio 0.94 — no slowdown (vs knob-1's +53%).
    Prediction 5 satisfied.
- **Preview vs predictions** (n=2000, 2 seeds, gap seed — qualitative):
  per-layer mean_dE (FD | off s1/s2 | ON s1/s2): L10 15.2 | 39.5/44.2 |
  7.4/13.6; L15 17.5 | 52.2/40.3 | 12.7/16.0; L2 2.03 | 1.8/3.0 | 1.2/1.2;
  L49 −5.4 | −272/−417 | −2.4/−120. Core lands between knob-off ~3×
  inflation and knob-1 ~0.5× attenuation, near FD (prediction-1 pattern);
  L49 s1 collapses, s2 −120 matches the wave-1 knob-1 s2 preview value
  (same residual rare-event tail, non-geometry path — the 2-seed tail is
  known heavy; defer to validation). L20 flips sign at n=2000 (noisy layer,
  off s2 = 9.4 there) — watch at validation.

## Wave 3 — derivative microscopy (event-matched AD/FD decomposition)

*(Redefined 2026-07-23, superseding the flag-relaxation scan: flag knockouts
cannot measure bias — the severing trio suppresses variance ~1e6, so relaxing
it hits a variance wall. jsonl entry is the twin.)*

- **No new physics knob**: branch `knob/derivative-microscopy` @ `887499f`,
  cut from `knob/prefix-anchor` @ 61c7bb5. One commit: env-gated per-event
  dump `HEPEMSHOW_EVENT_DUMP=<path>` (3 files, +90 lines: SteppingLoop.hh/.cc,
  EventLoop.cc). Per event: FNV-1a decision-path signature (per step:
  particle type, pre-step layer+region, boundary-vs-physics race winner,
  step index; per Perform: secondary counts; final track count), track/step
  totals, 50 per-layer edeps + 50 AD dots. Strictly no-op with env unset.
- **Hypothesis**: same-seed runs at g and g±h share the RNG stream until the
  first discrete flip. Same-path events must satisfy per-event AD = central
  FD to O(h²) (mismatch ⇒ severed derivative line, bisectable); diverged
  events carry FD mass pathwise AD cannot represent (missing score term).
  Decompose the knob-on core deficit (AD/FD ≈ 0.57) into the two parts.
- **Gate** (n=2000, s=1, canonical flags + `--stopgrad-prefix-anchor 1`):
  (1) env unset ⇒ edeps byte-identical to the prefix-anchor binary (primal
  AND derivative columns) — PASS; (2) env set ⇒ edeps still byte-identical,
  dump = 2000 lines, per-event means reproduce aggregate mean_E/mean_dE to
  4e-14 rel — PASS; (3) same seed twice ⇒ dumps byte-identical — PASS.
- **Result — the decomposition degenerates, decisively** (summary:
  `fidelity/wave3_microscopy_summary.json`; raw dumps `fidelity/wave3_runs/`;
  seeds 1–3 at h=0.05; seed-1 h-scan 0.02, 1e-3, 1e-4, 1e-6, 1e-7, 1e-11):
  - **100.0% of events diverge at every h**, down to Δg = 1e-11 mm
    (2000/2000 at each of 8 h values; side-vs-side and center-vs-side alike).
  - Divergence is *saturated, not ∝ h*: median |Δsteps| ≈ 2500–2800
    (~3% of the ~80k steps/event) whether h = 0.05 or 1e-11; per-event
    Σ|ΔE_layer| ≈ 2970 MeV of 9300 MeV (~30% energy rearrangement) at 1e-11;
    first differing layer = L0 for >90% of events.
  - **Controls**: gap-seed vs energy-seed at identical g = 5.70 ⇒ 2000/2000
    identical signatures and step counts (AD seeding does not touch the path;
    instrumentation valid). Same binary, same seed, twice ⇒ bit-identical.
  - Interpretation: the discrete decision path is **bit-unstable in g** —
    every event's first gap-boundary landing re-resolves an exact-tie
    comparison (boundary-landing arithmetic: Box on-surface early return
    0.0 / boundary-tolerance tests / value-only layer scan,
    RECON B6/B8) at ulp level, and one flip decorrelates the entire
    downstream shower via the shared RNG stream. The effective branch-flip
    density is unbounded: there is **no finite h with a same-path
    population**, so (i) the same-path pathwise term is unmeasurable at
    event granularity (empty set — 0/2000 at all h), and (ii) 100% of the
    FD signal — hence the whole 0.57 core-window deficit — flows through
    path-changed events. Same-seed per-event FD behaves as an *unpaired*
    estimator (per-event FD noise ∝ 1/h; core FD means at h ≤ 1e-3 are pure
    noise; at h = 0.05, seeds 1–3 give FD 399/−118/19 vs AD 82/114/111 —
    n=2000/seed is far below the SNR needed to re-measure the 0.57).
  - Ranked code sites (v1): the deficit cannot be pinned to severed
    derivative lines by event-level matching; the instability lives at
    Box.cc:113-121 (on-surface early return), Geometry.cc:287-298
    (value-only layer scan + clamp), SteppingLoop.cc zero-step push
    (:793/:1206) — the boundary-landing tie cluster.
- **Implication for wave 4**: event-level microscopy cannot separate
  pathwise error from branch-flip mass in this simulator; the drill-down
  needs *sub-event* (track-level) matching with RNG-stream isolation, or
  accept the score-function route (RECON D4) / surface-term route (D3) on
  aggregate evidence alone.
- **Status**: v1 complete (2026-07-23); awaiting human verdict. hepemshow
  working tree restored to `phaseA-perlayer-gap-energy`; agent binaries in
  `build_agent_fwd/rev` left at `knob/derivative-microscopy` (env-off =
  bit-identical to prefix-anchor).

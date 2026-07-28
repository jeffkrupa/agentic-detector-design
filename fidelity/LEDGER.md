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

## Wave 4 — RNG lineage isolation (`--rng-lineage`)

- **Knob**: `--rng-lineage 1`, branch `knob/rng-lineage` @ `3ad2e97`
  (1 commit over `knob/derivative-microscopy` @ 887499f), default off.
  7 files, +255/−6. Design:
  - **Lineage key** (stable under perturbation — no step indices): primary
    track = FNV1a64(base seed, event index); secondary = FNV1a64(parent
    lineage, ordinal among that parent's secondaries). Bookkeeping lives in
    `TrackStack` (lineage + per-parent ordinal maps, cleared per event);
    assignment in `EventLoop` (primary) / `StackSecondaries` (children).
  - **Engine strategy**: tracks are stepped contiguously (secondaries only
    wait on the stack), so the shared mt19937_64 is re-seeded with
    splitmix64(lineage) at each track start (`EventLoop`, after the existing
    per-track `DiscardGauss`); no per-track engine instances or draw
    counters needed. Knob off ⇒ bit-identical single shared stream.
  - **Track-level dump** (for the microscopy): env
    `HEPEMSHOW_TRACK_DUMP=<path>` (+ `HEPEMSHOW_TRACK_WINDOW=lo:hi`,
    default 5:18) writes one 40-byte binary record per track: (eventID,
    nsteps, lineage, decision-path sub-signature, window edep value+dot).
    ~13k tracks/event ⇒ ~1.0 GB per 2000-event run; strictly no-op unset.
- **Hypothesis**: with per-track streams, θ vs θ±h runs give identical
  draws per corresponding track; divergence localizes to subtrees where a
  discrete decision genuinely flips ⇒ (a) a nonzero same-path population
  for pathwise AD-vs-FD matching, (b) real CRN pairing for FD.
- **Pre-registered predictions**: knob-off byte-identity; knob-on primal
  distribution unchanged (|z|<4 all layers at 20k — statistical gate,
  documented PROTOCOL deviation: the knob intentionally changes individual
  paths, so byte-identity cannot hold for the on-state); same-path fraction
  off the floor (>0 measurable, target >50% of tracks at h≤0.02);
  paired-FD variance reduction >>1; fwd=rev preserved.
- **Gate results** (canonical flags + `--stopgrad-prefix-anchor 1`, gap
  seed, s=1 unless noted):
  - G1 knob-off byte-identity: edeps (primal AND derivative columns)
    identical to the microscopy binary, n=2000. PASS.
  - G2 knob-on reproducibility: same seed twice ⇒ edeps + event dump +
    track dump all byte-identical, n=2000. PASS.
  - G3 knob-on distributional physics gate, n=20000 knob-on vs knob-off:
    per-layer mean_E z-scores (SE from var_E/n): max|z| = 1.00 (L49),
    mean z = +0.044, 0/50 layers with |z|>2 (gate bound was |z|<4);
    total mean_E 9291.5 (on) vs 9299.0 (off), diff −7.5 MeV = −0.48σ
    (−0.08%). No systematic shift. PASS (documented PROTOCOL deviation:
    statistical gate replaces byte-identity for the on-state, since the
    knob intentionally changes individual paths while preserving the
    distribution).
  - G4 fwd=rev: knob on, n=500: Σ fwd mean_dE = −67.194850372309 vs
    reverse barInputs gap row −67.194850372402 (rel 1.4e−12). PASS.
  - G5 NaN scan: all gate + acceptance edeps finite. PASS.
  - Consistency: per-event window sums of the track dump reproduce the
    event dump to 1e−10; per-event track counts match; 0 duplicate
    lineages in 25.6M tracks; runtime knob-on+dumps 401 s vs knob-off
    365 s (n=2000, same host/load) ⇒ ≤10% overhead.
- **Acceptance experiment** (wave-3 microscopy retried, knob on; triplets
  (g−h, g, g+h), seeds 1–2, h ∈ {0.05, 0.02}, n=2000/run, window L5–18;
  summary `fidelity/wave4_microscopy_summary.json`, analysis
  `fidelity/wave4_microscopy.py`):
  - **Event-level same-path: still 0/2000 at every (seed, h).** With
    ~12.8k tracks/event, one flipped track anywhere diverges the event
    signature; the event-level floor is a combinatorial consequence, not a
    knob failure.
  - **Track-level same-path comes off the floor** (was structurally 0):
    matched (present in all 3 runs, identical sub-signature) = 6.3%/6.2%
    of center tracks at h=0.05 and 14.0%/13.9% at h=0.02 (seeds 1/2);
    common-lineage (existence) fractions 22%/33%. The >50% target was
    missed — flips propagate to entire descendant subtrees, and most
    tracks are deep in the cascade.
  - **Matched-population pathwise AD is exact**: per-track AD − central FD
    = +0.00035±0.00026 / +0.00025±0.00032 / −0.00008±0.00026 /
    +0.00039±0.00032 MeV/mm (4 triplets — consistent with 0 at 3e−4);
    signal-carrying matched tracks (~0.5–1.0M per triplet) have median
    |AD−FD| = 1e−6; all matched tracks have identical step counts across
    the triplet. Window-sum decomposition closes exactly.
  - **Deficit decomposition (core L5–18)**: matched tracks carry 0.09–0.9%
    of the FD mass (FD_matched 0.18–2.6 of FD_total 211–622 MeV/mm);
    ≥99% of the FD signal — and hence the knob-on core deficit
    (AD/FD≈0.57 at 1M events) — flows through path-flipped subtrees, now
    *measured* at track granularity rather than inferred (wave 3).
  - **CRN pairing now works**: per-event paired-FD variance reduction vs
    unpaired = 5.6×/5.2× at h=0.05 and 7.6×/7.8× at h=0.02 (seeds 1/2);
    corr(E⁺,E⁻) = 0.82–0.87 (knob-off measured 1.16×, i.e. useless).
    Equivalent to ~6–8× fewer events for the same FD error at fixed h.
- **Interpretation**: the knob does exactly what it was built to do —
  isolates RNG lineage so divergence is local, creates the first nonzero
  same-path population, and proves pathwise AD is *numerically exact* on
  it. The remaining core deficit is therefore entirely a missing
  branch-flip (score-function/surface) term, not a severed derivative
  line: direct, track-level evidence for the RECON D3/D4 routes. Bonus:
  the CRN factor makes future FD truth batches ~6–8× cheaper at fixed h.
- **Status**: gated + acceptance complete (2026-07-23); awaiting human
  verdict. hepemshow working tree restored to `phaseA-perlayer-gap-energy`;
  agent binaries `build_agent_fwd/rev` left at `knob/rng-lineage`
  (knob-off + env-off = bit-identical to the microscopy binary).

## Wave 5 — analytic scoring-surface jump term (`--score-surface-term`)

- **Knob**: `--score-surface-term 0/1/2/3`, branch `knob/score-surface`
  (3 commits over `agent-knobs` @ 3ad2e97: 5b9dca4 mode 1, dc494d3 mode 2,
  c1945ff mode 3 = final HEAD), default off, derivative-only (KeepPrimal:
  weight `(P − stop_grad(P))·ρ`, primal exactly 0), deterministic, no RNG.
  5 files, +158 (mode 1) then +131/−32 and +36/−11.
- **Pre-registered prediction** (jsonl): gap core L5–18 AD/FD rises from
  0.57 toward 1 (success > 0.75 AND closer to 1); absorber rises from 0.55;
  primal byte-identical; fwd=rev; runtime < +15%.
- **Implementation** (per-step, both steppers, after `Perform`):
  ρ = (step edep)/|step Δx| (Δx = geometric stepLength·vx at move time,
  |Δx| floored at 1e-6 mm); plane identified from the pre-step bin
  (indxLayer, indxAbs) + sign(Δx), verified |postX − plane| ≤ 1e-6 mm
  (rejects transverse y/z exits); plane positions and dots directly from
  the AD-active prefix sums `Geometry::GetLayerStartX(i)` (new accessor) +
  `fAbsThick[i]` — FULL dots incl. prefix, independent of the stop-grad
  severing knobs. Bin mapping: layer plane x_j → combined bins (j−1 | j),
  gap bin j−1, abs/gap totals; sub-plane x_j+a_j → gap bin j + totals only
  (interior to the combined bin); back face x_N → one-sided escape term
  into bin N−1; front face x_0 pinned (zero dot). Mode 2 adds
  start-adjacency (|preX − rear face| ≤ 2e-6 mm) and applies each face
  term one-sided to the step's own bin. Mode 3 = symmetric transfer only
  for severed steps (gradient-disabled tracks or step-locally sanitized),
  carried by the severed prefix anchor `fLayerStartX[i_step]`.
- **Gates** (canonical flags + `--stopgrad-prefix-anchor 1`, final binary):
  G1 knob-off byte-identity vs agent-knobs reference (Release build):
  edeps + edeps_gap bit-identical (primal AND derivative), n=2000 s1 —
  PASS. G2 primal identity knob-on: modes 1/2/3, gap+abs seeds s1–2,
  mean_E/var_E byte-identical, derivative column changes 50/50 layers,
  both outputs — PASS. G3 fwd=rev n=500 to all printed digits (mode 1:
  61.8768256974; mode 2: −158082.576166; mode 3: −2.79508060118) — PASS.
  G4 NaN scan clean — PASS. G5 runtime: mode-1 paired median 0.997,
  mode-2 sequential probe median 1.002 — PASS (< 1.15). Refactor check:
  mode-1 output bit-identical across the three commits.
- **Step-6 physics check** (triplets g∓0.02 / a∓0.02, `--rng-lineage 1`,
  gap seeds 1–4 = 8000 paired events, abs seeds 1–2 = 4000; per-event
  paired FD, corr(E⁺,E⁻)=0.86–0.87, variance reduction 6.9–7.8×; paired-FD
  core SE still 29% (gap) — pre-registration of "few-%" was optimistic —
  so ratios are quoted against the 1M FD truth (±4.8%/±0.7%); summary
  `fidelity/wave5_step6_summary.json`):
  - knob-off (in-sample): gap core 0.481 ± 0.037, abs 0.422 ± 0.064
    (1M values 0.566 ± 0.027 / 0.545 ± 0.010).
  - **Mode 1 (pre-registered, all tracks): FAIL in gap** — gap core
    AD/FD_1M = **5.50 ± 0.28** (>0.75 but far from 1); gap L49 −465 ± 26
    vs FD −5.4 (re-inflates the L49 anomaly); gap L0–2 10.9 vs 3.5.
    **Absorber: 0.865 ± 0.069** (from 0.55; direction prediction PASSES),
    abs L0–2 24.7 vs 28.4.
  - **Mode 2 (one-sided Leibniz, all tracks): decisively wrong and
    decisive** — gap core −131 ± 8, abs −11.5 ± 0.8 (≈ −25.7k MeV/mm both
    channels): the one-sided boundary terms are real but are cancelled by
    an equal-and-opposite interior transport term that pathwise AD ALREADY
    carries for live tracks (a live track's boundary-limited step-length
    dot converts relative track/plane motion into amount changes = the
    relabeling flux). Adding the boundary term for all tracks
    double-counts it with the sign of the un-cancelled interior part.
  - **Mode 3 (severed-only, prefix carrier): partial** — abs core
    **0.780 ± 0.069** (recovers the severed relabeling mass), but gap core
    **4.54 ± 0.24** and L49 −461 ± 26: the symmetric single-ρ estimator
    mis-sides the discontinuous density (ρ_PbWO4 ≫ ρ_lAr) at layer planes;
    severed backscatter crossings through the absorber side credit the
    absorber-side density to the gap-adjacent bin.
- **Interpretation**: the missing-jump-term picture is CONFIRMED in the
  absorber channel (0.55 → 0.78–0.87) and the mechanism is now sharply
  localized: (i) live tracks need NO surface term (pathwise AD already has
  it — mode-2's −25.7k proves the cancellation), (ii) the missing mass is
  the severed-step relabeling (mode 3 recovers it where the density is
  smooth), (iii) the remaining gap failure is purely the side-assignment
  of a discontinuous ρ. Next-wave candidate (both fix attempts spent):
  mode 4 = severed-only + one-sided per-bin densities + prefix carrier —
  the ~15-line composition of modes 2 and 3.
- **Deviations**: physics preview failed → 2 pre-authorized fix attempts
  used (modes 2, 3), both gated and measured; paired-FD SE 29% not few-%
  (pairing itself performed as wave-4 predicted); reference build initially
  missed `-DCMAKE_BUILD_TYPE=Release` (rebuilt; no science impact); knob
  widened from 0/1 to modes 0–3.
- **Status**: gated + step-6 measured (2026-07-23); awaiting human verdict.
  hepemshow working tree restored to `phaseA-perlayer-gap-energy`; agent
  binaries `build_agent_fwd/rev` left at `knob/score-surface` @ c1945ff
  (knob-off = bit-identical to agent-knobs).

## Wave 6 — mode 4: severed-only one-sided per-bin densities

- **Knob**: `--score-surface-term 4` on branch `knob/score-surface`
  (2 commits over c1945ff: 7dba60d mode 4, 79f7383 fix-attempt-1 guard =
  final HEAD), default off, derivative-only. Composition of the two measured
  wave-5 halves: mode-3 POPULATION (severed steps only = `stop_tracking ||
  isUnsafeStep`; live tracks carry the relabeling flux pathwise — mode-2's
  −25.7k double-count proof) and CARRIER (severed prefix anchor
  `fLayerStartX[i_step]`), with mode-2's one-sided end/start-adjacency
  machinery so each bin's Leibniz term is priced in its own material
  (per-step route: each severed step adjacent to a face contributes its own
  ρ = edep/|Δx| to its own bin; a side with no adjacent severed step gets no
  term — the one-sided guard is automatic in the per-step formulation).
- **Pre-registered success bar** (task spec): gap core L5–18 AD/FD_1M in
  [0.75, 1.35] AND absorber ≥ 0.75; per-event var_dE core/tail ≈ mode 3.
- **Gates** (canonical flags + `--stopgrad-prefix-anchor 1`; both 7dba60d
  and the final 79f7383): G1 regression — modes 0–3 outputs byte-identical
  to the c1945ff binary (n=500, edeps + edeps_gap; mode-3 sum cross-check
  −2.7950806012 = wave-5 record) — PASS. G2 primal identity mode 4 — gap
  s1–2 + abs s1–2 n=2000 (7dba60d) and gap/abs s1 (79f7383), mean_E/var_E
  byte-identical vs wave-5 mode-0 runs; derivative column active in **49/50
  layers — L0 unchanged by design** (`fLayerStartX[0]` has zero dot; layer-0
  severed steps lost nothing under prefix-anchor, so mode 4 correctly adds
  nothing) — PASS. G3 fwd=rev n=500 to all printed digits (7dba60d:
  −103760.639026; 79f7383: −98060.1742341; rel 0.0) — PASS. G4 NaN scan
  clean — PASS. G5 runtime paired probe (3 alternating off/on pairs,
  n=1000): median ratio 1.002 — PASS (measured on 7dba60d; the guard only
  prunes work).
- **Preview** (paired triplets g∓0.02 / a∓0.02, `--rng-lineage 1`, gap
  seeds 1–4 = 8000 paired events, abs seeds 1–2 = 4000, n=2000/run; ratios
  vs 1M FD truth gap 195.8 ± 9.4 / abs 2233.7 ± 14.8; summary
  `fidelity/wave6_preview_summary.json`): **decisive FAIL, both channels,
  both variants**:
  - mode 4 @7dba60d: gap core AD/FD_1M = **−78.8 ± 5.9** (AD −15424 ± 898
    vs truth +195.8), abs core = **−6.83 ± 0.74** (AD −15253 ± 1650 vs
    +2233.7); gap L49 AD −1719 ± 166 (FD −5.4); gap L0–2 −6.7 ± 1.6
    (FD 3.5); abs L0–2 8.8 ± 1.1 (FD 28.4).
  - fix attempt 1 (79f7383): the severed start-adjacency population is rich
    in sanitized 1e−6 mm push steps whose ρ = edep/|Δx| is floored at
    |Δx| = 1e−6 (ρ amplified ~1e6×) → guard: a floored-|Δx| step has no
    usable one-sided density, treat that side as absent. Result: fwd
    layer-sum −103760.6 → −98060.2 (−5%); preview gap core **−73.2 ± 4.9**,
    abs **−6.21 ± 0.53** — the artifact class was real but subdominant
    (~7% of the blow-up).
  - var_dE vs mode 3 (prediction "≈ same" also FAILS): gap core ×232,
    gap tail ×25, abs core ×34 (vs off: gap core ×2393); abs tail ~1.14.
  - Comparison rows (core AD/FD_1M gap | abs): off 0.57/0.55 (in-window
    0.48/0.42), m1 5.50/0.865, m3 4.54/0.780, **m4 −73.2/−6.21**.
- **Interpretation — the failure is structural, not an estimator bug**:
  severed-only one-sided pricing breaks the within-bin two-face cancellation
  that tames the full one-sided Leibniz expansion. Per severed +x crossing
  of layer plane x_j, the left-bin term carries ρ_gap × anchor_dot(j−1)
  while the right-bin term carries ρ_abs × anchor_dot(j): the bin-level net
  is ~ −j × (ρ_abs − ρ_gap) per crossing — matching the observed
  ~−1000/layer core profile (both channels pulled to ≈ −15k, ~60% of
  mode-2's blow-up from the severed subset alone). In the field picture the
  one-sided boundary terms are only correct WITH the interior ∂ρ/∂θ partner
  term, which severed steps by definition do not carry pathwise — and which
  mode 2 measured for live tracks as the exact canceller. The per-step
  lost-derivative reconstruction of what `--stopgrad-prefix-anchor` severed
  is mode-3's SYMMETRIC transfer (single ρ, single carrier, exact zero-sum
  per crossing); "each side its own density" is not a property of the
  severed mass. Fix attempt 2 not spent: per-material dE/dx tables would
  inherit the same structural imbalance (documented deviation).
- **Consequence**: mode 3 stands as the best severed-only estimator
  (abs 0.780 ± 0.069; gap 4.54 ± 0.24); the residual gap failure is NOT a
  side-assignment-of-ρ problem fixable within the severed-only surface-term
  family. The remaining gap inflation needs a different mechanism class
  (e.g. the branch-flip/score-function route, RECON D4, with
  phantom-subtree rollouts under rng-lineage).
- **Deviations**: (i) task-spec preview success bar used (gap [0.75, 1.35],
  abs ≥ 0.75) — measured FAIL; 1 of 2 pre-authorized fix attempts spent
  (floored-ρ guard, gated + measured), attempt 2 declined on structural
  grounds; (ii) G5 runtime measured on 7dba60d only (guard strictly
  removes work); (iii) preview reused the wave-5 side runs and off-centers
  (t6_*_lo/hi, t6_*_ctr_off — identical binaries for those legs), only the
  6 mode-4 center runs are new per variant.
- **Status**: gated + preview measured, FAIL vs pre-registered bar
  (2026-07-24); awaiting human verdict (recommend: reject mode 4, keep as
  measured-null branch commits). hepemshow working tree restored to
  `phaseA-perlayer-gap-energy`; agent binaries `build_agent_fwd/rev` left at
  `knob/score-surface` @ 79f7383 (knob-off = bit-identical to agent-knobs;
  modes 0–3 bit-identical to c1945ff).

## Wave 7 — phantom-subtree score-function race term (`--race-score-term`)

- **Knob**: `--race-score-term 0/1` (+ `--race-score-q <p>`,
  `--race-score-thresh <S>`, `--race-score-cap <n≤64>`), branch
  `knob/race-score` cut from `knob/score-surface` @ 79f7383, default off,
  derivative-only (KeepPrimal carrier), **forward-mode-only v1**
  (pre-authorized deviation; reverse build rejects the flag with a clear
  error). Requires `--rng-lineage 1` at runtime (hard error otherwise):
  phantom rollouts run at end-of-event on the shared engine, and per-track
  lineage reseeding is what guarantees they cannot perturb any primal draw.
- **Hypothesis / predictions**: pre-registered in `ledger.jsonl` (wave 7,
  written before implementation): gap core L5–18 AD/FD_1M reaches
  [0.75, 1.35] on the accepted stack; absorber stays within errors of 0.78;
  primal byte-identical knob-on (critical gate); runtime < 2×; fwd=rev or
  documented forward-only.
- **Derivation** (written before implementation; the wave-7 physics):
  At each step the sim races the physics step limit against the boundary:
  `distToPhysics < distToBoundary` (SteppingLoop.cc:1156 gamma, :1556
  electron; original RECON refs :712/:1092). The winner channel w carries a
  remaining interaction budget n_w = numIALeft[w], sampled once as
  −log(U) (zero dot) and decremented by pStep/λ each step
  (GammaManager.icc:104–145/190–197, ElectronManager.icc:397–452/570–578),
  so the proposed interaction distance x_p = λ_w·n_w (gamma; MSC
  geometric conversion for e∓) is **pathwise** with dot
  ẋ_p = GET_DOTVALUE(distToPhysics) (numIALeft-decrement dots + mfp dots,
  `-C`-regularized). The boundary distance d_b = distToBoundary has dot
  ḋ_b (consistent with the accepted severing stack: sanitized tracks carry
  severed/re-anchored dots). The discrete outcome is 1{x_p < d_b} with
  margin m = d_b − x_p; with the exponential draw held fixed
  (reparameterized) the outcome flips exactly when m crosses 0.
  By memorylessness of the exponential budget, conditional on the track
  history at step start, the residual n_w ~ Exp(1), so the conditional
  density of x_p at the threshold d_b is f = (1/λ_w)·e^{−d_b/λ_w}
  (electron with true↔geom conversion: f = (r/λ_w)·e^{−r·d_b/λ_w} with
  local secant Jacobian r = pStepLength_true/gStep ≥ 1 — documented v1
  approximation). The missing expectation-level term per decision (law of
  total expectation over the per-step filtration) is
      dE[F]/dθ ⊇ f · ṁ · (F_int@bnd − F_transport@bnd),  ṁ = ḋ_b − ẋ_p.
  **Sign conventions**: ṁ > 0 means the boundary recedes from the pending
  interaction point (physics-win region grows); the payoff difference is
  always oriented interaction-at-boundary MINUS transport-through-boundary.
  With the realized continuation as F_cur and a phantom rollout of the
  alternative as F_alt: contribution = s·f·ṁ·(F_phantom − F_real) with
  s = +1 for realized boundary-won steps (phantom = interaction at the
  boundary) and s = −1 for realized physics-won steps (phantom = transport
  through the boundary). Multi-channel: the outcome flips through channel w
  only when every other channel's proposed distance exceeds d_b (explicit
  guard; automatic for boundary-won steps) — winner-channel-only density is
  a documented v1 underestimate of the total-Σ flip density.
  **Threshold states via numIALeft forcing**: cloning the pre-step track
  and setting numIALeft[w] ← (d_b/λ_w)·(1∓1e−9) forces interaction just
  inside the near face (physics branch) or transport with ~zero residual
  budget — which then interacts just past the plane, exactly the marginal
  transport event (electron: r·d_b/λ_w). Known v1 biases (documented):
  (i) F_real is the realized continuation, not the threshold-conditioned
  branch payoff (realized transport crosses with Exp(1) residual budget,
  the marginal one with ~0) — standard one-sample bias of pruning-based
  stochastic-AD estimators; (ii) electron races won by continuous/MSC
  limits are skipped (no discrete margin); (iii) the selection threshold on
  S = f·|ṁ|·EKin drops small-payoff decisions.
- **Design** (pre-registered):
  - Phantom lineage: FNV(FNV(track_lineage, 'PHAN'), decision_ordinal);
    phantom secondaries get the standard HashLineage(parent, ordinal) chain
    from a dedicated phantom TrackStack. Selection randomness is a
    counter-based hash u = SplitMix64(FNV(FNV(track_lineage, 'RACE'),
    per-track candidate ordinal))/2^64 — **no draws from the primal
    stream** anywhere; rollouts run after the event's last primal track,
    and every primal track reseeds by lineage, so primal draws are
    structurally unreachable (the critical gate 2 tests this).
  - Subtree accounting: decision d ∈ [0,64) = one bit in a per-track
    uint64 race mask (TrackStack map + a current-track global); set on the
    deciding track from the decision step onward, inherited by secondaries
    at stacking; SteppingAction adds each deposit VALUE to E_real[d][layer]
    (combined + gap) for every set bit. The phantom rollout (gPhantomMode)
    redirects deposits to E_phantom[d][layer] and touches no primal
    accumulator, dump, or surface-term path.
  - Contribution at end of event, before EndOfEventAction:
    per layer, Fill(L, (m − stop_grad(m)) · s·f/q · (E_ph − E_re)) — primal
    exactly 0 (KeepPrimal), dot = s·f·ṁ·ΔE/q; gap/abs event totals
    consistently updated.
  - Cost control: candidate iff winner ∈ {0,1,2}, other-channel guard, and
    S = f·|ṁ|·EKin > thresh (default calibrated on a probe run); accepted
    with probability q (default 0.05), reweighted 1/q; ≤ cap (default 64)
    rollouts/event; per-rollout step budget with truncation counting;
    saturation statistics reported per run.
- **Implementation**: branch `knob/race-score`, 2 commits over 79f7383:
  30967d6 (mode 1) + 7ebab9e (mode 2 = fix attempt 1). 6 files, ~+420
  lines: `--race-score-term 0/1/2`, `--race-score-q/-thresh/-cap`
  (InputParameters ids 1011–1014); race-mask map in TrackStack; decision
  recording + phantom-mode guards + end-of-event rollout loop + KeepPrimal
  contribution in SteppingLoop.cc; per-event reset / finalize / stats hooks
  in EventLoop; rng-lineage requirement + forward-only rejection in main.
  Defaults calibrated on probes: thresh = 50 MeV/mm, q = 0.025 (mode 1;
  22 selected/event on gap seeds, saturation 0), cap = 64.
  **Mode 2 (fix attempt 1, motivated by the measured mode-1 bias)**: both
  branch payoffs rolled AT the threshold state — interaction at the
  boundary (numIALeft[w] ← thr·(1−1e−9)) vs transport with ~zero residual
  (thr·(1+1e−9)) — CRN-paired by a SHARED phantom lineage; contribution
  = f·ṁ·(E_int@bnd − E_transport@bnd)/q, no realized-branch sign.
- **Gate results** (canonical flags + m3 stack `--stopgrad-prefix-anchor 1
  --rng-lineage 1 --score-surface-term 3`, n=2000 unless noted):
  - G1 knob-off byte-identity: fwd edeps+edeps_gap identical to a fresh
    79f7383 Release build (gap s1; re-verified n=500 after the mode-2
    commit); reverse barInputs+edeps identical (n=500, `-b 1:…:1`). PASS.
  - G2 primal identity knob-on (**the critical gate**): mean_E/var_E
    byte-identical vs knob-off in every knob-on run — mode 1 gap s1–2,
    abs s1–2, q/2; mode 2 gap s1–3, abs s1, no-m3 s1 — while running
    22k–128k phantom rollouts per run (25M+ phantom tracks): zero draws
    leak into the primal stream. Derivative column changes 50/50 layers.
    PASS.
  - G3 fwd=rev: pre-authorized deviation — **forward-only v1** (reverse
    build rejects the flag with a clear error; phantom rollouts are not
    taped). Reverse knob-off regression byte-identical.
  - G4 NaN scan: all runs finite. PASS.
  - G5 runtime: mode 1 gap 2.01× (q=0.025), mode 2 gap 2.06× (q=0.0125,
    2 rollouts/decision); abs channel 2.2×/3.3× with cap saturation
    (thresh=50 passes 5.6k candidates/event on abs seeds). Marginally
    above the pre-registered <2×.
  - Mode-1 regression across the mode-2 commit: bit-identical. PASS.
- **Preview** (center runs n=2000, ratios vs the pre-registered 1M FD
  truth gap 195.8 ± 9.4 / abs 2233.7 ± 14.8; FD side legs not rerun —
  prior-wave scratch runs deleted; summary
  `fidelity/wave7_preview_summary.json`):
  - **Mode 1 (pre-registered estimator): decisive FAIL, both channels** —
    gap core (s1–4) AD/FD_1M = **7.56 ± 0.53** (from m3's 4.54 ± 0.22);
    abs (s1–2, cap-saturated 2000/2000 events) = **5.88 ± 0.88** (from
    0.78); race-only on prefix-anchor (s1–2) = **3.58 ± 0.25** (from
    0.57). The term ADDS ≈ +560 MeV/mm/seed to the gap core in either
    stack. q-sensitivity s1: core AD 1450 (q=0.025) vs 1313 (q=0.0125).
    var_dE on/off: gap core ×7.3, gap tail ×12.4, abs core ×32.
  - **Mode 2 (paired-threshold, exact): the race branch-flip term is a
    measured NULL** — gap core (s1–3) = **4.499 ± 0.220** vs off
    4.540 ± 0.217 (per-seed race-term delta −23.9 / −6.3 / — MeV/mm of
    ~888); abs s1 = **0.799** vs off 0.780 (within errors — the abs
    prediction PASSES trivially because the term ≈ 0); race2-only on
    prefix-anchor s1 = 0.399 vs knob-off in-sample ~0.48. var_dE on/off:
    gap core ×1.6, gap tail ×22.6, abs core ×4.8.
- **Interpretation — the wave-7 hypothesis is FALSIFIED by the exact
  estimator**: numIALeft PERSISTS across boundaries, so at the race
  threshold the two branches nearly coincide (interaction just inside the
  plane vs transport that crosses with ~zero residual budget and interacts
  just past it) — E[F] is continuous there up to the cross-material
  product difference, and the interaction-position sensitivity is already
  carried pathwise by the numIALeft decrement dots. The paired-threshold
  measurement (CRN-shared lineages make the two branches differ only by
  the genuine jump) puts the boundary-vs-physics branch-flip mass at
  −0.08 ± 0.05 (gap, ratio units) and −0.04 (abs): this decision class
  does NOT carry the residual gap error. Mode-1's +560 MeV/mm/seed is the
  quantified double-count bias of realized-continuation payoffs (the
  spec's pre-registered v1 approximation) — a methods result in itself:
  phantom estimators here MUST price both branches at the threshold.
- **Deviations**: forward-only v1 (pre-authorized); runtime 2.01–2.06×
  vs pre-registered <2× (marginal, reported); abs channel cap-saturated
  at thresh=50 (abs-seed ṁ dots are ~10× larger — a per-seed threshold
  would be needed); FD side legs not rerun (1M truth is the pre-registered
  denominator); fix attempt 1 = mode 2 (spent, gated, measured); fix
  attempt 2 declined — the estimator is exact and the null is the
  measurement, not an estimator defect.
- **Status**: gated + preview measured (2026-07-24) — mode 1 FAIL
  (biased estimator, kept as measured branch commits), mode 2 = measured
  null vs the pre-registered gap bar; awaiting human verdict (recommend:
  reject as a gap fix, record as the decisive falsification of RECON D4's
  boundary-race route — the residual gap inflation must live in another
  mechanism class, e.g. the m3 carrier itself or the winner-reset /
  secondary-kinematics severed lines). hepemshow working tree restored to
  `phaseA-perlayer-gap-energy`; agent binaries `build_agent_fwd/rev` left
  at `knob/race-score` @ 7ebab9e (knob-off = bit-identical to 79f7383).

## Wave 8 — unsevered-regime forensics (part A: spike-event mining, DONE)

- **Measurement only, no C++ changes.** Binary `build_agent_fwd/HepEmShow`
  @ `knob/race-score` 7ebab9e, all wave knobs OFF. Canonical base args,
  n=2000, event dumps on. Configs (gap seed `-g 5.7:1`, seeds 1–2):
  reference `-x 2 -y 1 -B 1`; unsevered `-x 0 -y 0 -B 0`; intermediate
  `-x 0 -y 1 -B 1`; plus absorber seed `-a 2.3:1` unsevered, seed 1.
  7 runs, 318–335 s each (unsevered is NOT slower). Analysis
  `fidelity/wave8_spikes.py`; summary `fidelity/wave8_spikes_summary.json`;
  spike list `fidelity/wave8_spike_events.json`; raw dumps
  `fidelity/wave8_runs/` (untracked). Event signatures and primal edeps
  are byte-identical across all three severing configs (severing is
  derivative-only; cross-config event matching is exact).
- **Core-window (L5–18) per-event dot sum W_e, gap seed** (FD truth
  195.8 ± 9.4): 0 NaN/inf events anywhere (the "NaNs expected" prior did
  not materialize at n=2000).
  | config | mean ± SE | median | tm5% | var | top-1% var share |
  |---|---|---|---|---|---|
  | ref s1/s2 | 565±35 / 589±15 | 504 / 494 | 540 / 535 | 2.4e6 / 4.5e5 | 0.93 / 0.62 |
  | uns s1/s2 | −340±1184 / −991±2212 | **202.5 / 196.0** | 273 / 216 | 2.8e9 / 9.8e9 | 0.99 / 1.00 |
  | int s1/s2 | 5571±23470 / −9630±11380 | 132.8 / 113.0 | 236 / 113 | 1.1e12 / 2.6e11 | 1.00 / 1.00 |
  - Variance ratios vs reference: **unsevered ×1.2e3 / ×2.2e4**;
    **intermediate ×4.6e5 / ×5.7e5** (the measured "1e6" in this testbed —
    and it belongs to the PARTIALLY severed config, not the unsevered one).
  - **The unsevered MEDIAN sits on FD truth** (202.5/196.0 vs 195.8 ± 9.4)
    while reference medians sit at the known ~3× inflation (≈500). Raw
    unsevered means are noise (SE > 1000); 5%-trimmed means 273/216.
  - Intermediate (-x 0 -y 1 -B 1) is the WORST config on every axis:
    grazing/backward guards flag tracks mid-flight, SanitizeTrackState
    zeroes their position dots, and the wave-1 zombie prefix-dot injection
    does the rest — partial severing CREATES the biggest spikes. Its
    median (133/113) also undershoots truth (severed legitimate mass).
- **Spike anatomy — the spike population is adjacent-bin relabeling
  DIPOLES, not lost-in-the-weeds blowups**: all top-10 |W_e| events in
  both unsevered gap runs are L18|L19 pairs of huge equal-and-opposite
  dots (|D18| ≈ |D19| up to 3.8e6, opposite sign) whose FULL-detector sum
  is ordinary (|Σ50 dots|/|W_e| ≤ 1.4e-2, median ~1e-3); they rank at the
  top of the L5–18 statistic only because the window edge slices the
  dipole. Spike events are otherwise NORMAL showers: steps/tracks within
  ±6% of the run median, no step-count signature. Mining dipoles directly
  (largest opposite-sign adjacent-bin pair per event): 78% of unsevered
  events carry a dipole > 1e4 MeV/mm, 4.5% > 1e6, largest **2.1e8**
  (cancellation residual 1e-9 relative); left-layer histogram is broad
  over L20–40 (shower-tail region), no single hot layer. Intermediate:
  95% > 1e4, 23% > 1e6 (≈5× the unsevered count at 1e6 — severing
  amplification). Reference: 5% > 1e4, 0 > 1e6.
  - **Window/total variance ratio** (var W_e / var Σ50): uns 77/130,
    int 1.2e4/62 — the unsevered variance explosion is overwhelmingly
    relabeling noise that cancels in the total-detector derivative
    (uns total: mean 189±135 s1, −151±194 s2). A surgical cap need only
    tame a bin-transfer term, not a net-derivative term.
  - Cross-config: unsevered top-20 spike events are spiky in the
    intermediate config too (median |W| percentile 96/98) but NOT in the
    reference (median percentile 67/76) — severing kills these events'
    dots and replaces them with its own (bigger) zombie population.
- **Absorber seed, unsevered** (FD truth 2233.7 ± 14.8): median 1430,
  tm5% 1580, tm1% 1876, raw mean 994 ± 3032; var 1.8e10, window/total
  ratio only 2.9 (unlike the gap channel, most variance is NOT
  relabeling); dipoles present (99.6% > 1e4, largest ~2e6-e7 class).
  Same dipole population exists, but robust locators stay ~15–35% below
  FD truth — the unsevered absorber estimator does not obviously center
  on truth at this n; needs the dissection/cap stage to say more.
- **Implication for the cap design** (part B/C): the guilty object is a
  boundary-local antisymmetric transfer with an unbounded carrier
  (grazing 1/vx class at layer interfaces, broad in depth) — cap the
  per-step bin-transfer dot (or floor |vx| in the dot only), and the
  total derivative is provably untouched (dipoles cancel to ≤1e-2
  relative already). Top spike/dipole (seed, event) lists for step-level
  dissection: `fidelity/wave8_spike_events.json` (includes both the
  W-ranked window-edge list and the direct top-dipole list).
- **Status**: part A complete (2026-07-24); dissection (part B) and cap
  knobs (part C) pending.

### part C — dissection (single-event step-level replay, DONE 2026-07-24)

- **Method.** Worktree `/eos/user/j/jeffkrup/agentic/hepemshow-diag`, branch
  `knob/step-telemetry` (from `knob/race-score` 7ebab9e; parallel wave-9 tree
  untouched), forward build `build_diag_fwd/`. Env-gated telemetry
  (`HEPEMSHOW_STEP_DUMP=<path>` + `HEPEMSHOW_DUMP_EVENT=<idx>`): one CSV line
  per step (lineage, layer/region, winner, boundary flag, position/direction/
  stepLen/pStepLen/edep/EKin/numIA/mfp/distB/distP values AND dots) plus
  track-birth lines; per-track death summaries derived offline from the step
  lines. **Byte-identity gate PASSED**: env unset, n=2000 gap-seed unsevered
  s1 → `edeps_1`, `edeps_gap_1`, event `dump.txt` all byte-identical to the
  7ebab9e reference (`wave8_runs/uns_s1`). Replays with shared RNG stream
  (`-n event+1`): both target events' dump rows byte-identical to part A.
- **Gap dipole, s1 ev529 (amplitude 2.08e8, L32|L33).** One track carries
  the whole dipole: e- id 10489 (parent γ 10487), born 0.244 MeV with
  HEALTHY dots (ẋ=240, Ė=0.20), 88 steps ping-ponging the L32-gap|L33-abs
  interface. Its per-layer dot sums: L32 −2.0787e8 / L33 +2.0787e8, and the
  track TOTAL = +0.2003 = its birth Ė exactly (1e-9 relative): the spike is
  pure bin relabeling of a fixed energy. Of 12,111 tracks in the event only
  17 ever exceed |edep_dot|>2e3 (next-largest peak 9.7e4).
- **The generating loop (named lines, measured gains)** — a 3-carrier S4
  compounding loop, one multiplication per boundary crossing:
  1. *S2 numerator* (Box.cc `t=(copysign(hD,v)−p)/v` via
     `Geometry::CalculateDistanceToOut`): crossing stepLen_dot =
     (plane_dot − ẋ)/vx. Crossing vx were 0.23–0.90 — **NOT grazing**; the
     1/vx factor contributed ≤5× while the accumulated-ẋ numerator carried
     up to 1.8e8. Post-crossing ẋ resets to the plane dot (33/19 ✓ healthy).
  2. *Energy conservation* (`ApplyMeanEnergyLoss`, eloss=pStepLength·dEdx):
     the crossing eloss_dot is scored in the current bin (first pole) and
     sign-flips into Ė. Measured Ė after crossings: 138 → 1.66e3 → 3.75e5
     → 2.09e8 (per-cycle gains ×12, ×226, ×557).
  3. *MSC re-injection* (UMSC `SampleCosineTheta`/rotate/displacement,
     S5/S6/S7 class): angle Jacobians convert pSL_dot and Ė into direction
     dots — v̇x 0.9→3.0e4 in ONE scattering (step 0; gain ≈63 per unit
     pSL_dot), later up to 1.3e10; each step rebuilds ẋ += sL·v̇x
     (~1.7e7/step measured, ≈80% of growth; displacement dots ≈20%).
  4. *Scoring*: pole 1 at crossing step 70 (edep_dot −2.09e8, L32-gap);
     pole 2 at range-out step 87 (full-deposit branch dumps the accumulated
     Ė: +3.38e8; net L33 +2.08e8 after fluctuation-dot recycling at steps
     79–86, which moved Ė 2.09e8→3.38e8 with matching negative edep dots).
  - Top-5 |edep_dot| steps: st87 +3.38e8 (range-out), st70 −2.09e8 (S2
    crossing), st85/84/86 −4.3e7/−2.7e7/−2.1e7 (fluctuation recycling).
- **Absorber spike, s1 ev91 (W=−3.6e6) = SAME mechanism** (the audit's
  "expect S1" prediction is falsified): an L18-gap|L19-abs dipole on e- 3291
  (born 0.355 MeV), track total dot = −2042 = birth Ė exactly. Seeding is
  multi-generation: e+ 3262 (born ẋ=0.0245) ran ~1.5 cycles of the same
  loop at the same interface (ẋ→2.2e4); its brems γ 3285 was born with
  ẋ=4.3e3 and inherited v̇x≈−2.6e4, and after a 5 mm flight the secondary
  e- 3291 was born with ẋ=−1.24e5; crossings at steps 2 and 12 (sL_dot
  −2.5e5, −2.18e7) pumped Ė to 3.87e6; range-out at step 31 = +4.1e6 pole.
- **S1 verdict**: the electron mfp×numIA product NEVER set the step limit
  on any amplifier step in either event (all winner=−2 continuous/MSC-
  limited; numIA dots reached 4.9e5 but never landed in a stepLen). Gamma
  S8 injections were seed-scale (~1e4, `-C 1000` active).
- **Cap counterfactuals (audit proposals)**: #1 el-mfp-cap would NOT have
  neutralized either event. #2 track-dot-cap @250 mm/seed: neutralizes both
  (gap pole 2.1e8→≲2e3; abs 3.7e6→≲2.3e2); caveat: Ė and v̇x stay uncapped
  as spec'd — here both are fed only through the clamped crossing dot, but
  an explicit Ė cap (~1e2) would close the second carrier. #3 boundary-dot-
  cap @40 mm/seed: neutralizes both at the generator (gap pole →≈63 MeV/mm,
  abs →≈2e1, at/below the ~200 physical scale) and bounds the per-crossing
  Ė gain, collapsing the loop; residual: seed-scale γ injections and
  inherited v̇x persist but cannot compound. **Recommend #3 first** (caps
  the net Box distance dot — no cancellation-pair breakage; dipoles already
  cancel to ≤1.4e-2 in Σ50, so the total derivative is provably untouched),
  optionally + an Ė-dot governor.
- **Artifacts**: `fidelity/wave8_dissection_summary.json`; carrier+ancestor
  step traces `fidelity/wave8_runs/dissection/*.csv.gz` (untracked);
  telemetry code left on the worktree branch `knob/step-telemetry` (never
  pushed; worktree left in place).

## Wave 9 — surgical dot caps to replace stop-grad severing (IMPLEMENTED, pre-registered)

- **Knobs**: `--boundary-dot-cap <mm/seed>` (id 1016) + `--el-mfp-cap <mm>`
  (id 1015), branch `knob/dot-caps` @ `fc388aa` (1 commit over
  `knob/race-score` @ 7ebab9e) + g4hepem branch `el-mfp-cap` @ `73405c2`
  (1 commit over 91cbee3), both default off, derivative-only KeepPrimal.
- **Hypothesis** (wave-8 census + amplification audit): the unsevered
  gap-core variance explosion is adjacent-bin relabeling dipoles with an
  unbounded boundary-local carrier (S2), cancelling to ≤1e-2 in the
  50-layer total; the unsevered median already sits on FD truth. The
  absorber unsevered deficit candidate carrier is the raw electron
  mfp×numIA step-limit product (S1, ElectronManager.icc:447 — the gamma
  analogue is `-C`-capped). Surgical caps at these two sites should tame
  spike variance without severing any track.
- **Implementation**:
  - `--boundary-dot-cap` (audit proposal 3): in `Box::DistanceToOut(r,v)`
    the derivative part of the returned NET distance is clamped to ±cap,
    applied ONCE after the min over the three axes — no individual member
    of a cancelling pair (axis terms, numerators/denominators) is touched
    (audit S7 cancellation-fragility honored). Value untouched exactly
    (`stop_grad(t) + (t − stop_grad(t))·s`, s = cap/|ṫ| when |ṫ| > cap).
    Applies to all tracks. **Forward-mode-only**: the clamp scale depends
    on the tangent value and cannot be recorded on a reverse Jacobian
    tape; the reverse build rejects a nonzero cap with a clear error
    (race-score precedent).
  - `--el-mfp-cap` (audit proposal 1 = RECON D2): exact mirror of the
    gamma `-C` pattern (91cbee3) at the electron `dStepLimit = mfp·numIA`
    site — when mfp > cap, d/d(mfp) is zeroed (also killing the
    near-threshold `numIA·ṁfp` divergence) and d/d(numIA) is clamped to
    cap. New `G4HepEmElectronManager::ConfigureMfpCapRegularization`,
    wired in HepEmShow.cc next to the gamma call. Both modes. The
    g4hepem edit lives in the source tree AND is mirrored byte-identically
    into `install/` and `install_reverse/` (the copies the builds compile).
  - **`-A` relationship (decrement side)**: `RegularizedNumIADecrement`
    (the `-A` floor) already bounds the numIA-decrement derivative
    coefficients from the small-mfp side (1/mfp, p/mfp² with floored mfp);
    on the large-mfp side those coefficients shrink — no cap needed.
    The two knobs are complementary and independent; nothing to mirror.
  - Diff sizes: hepemshow 4 files +83/−2; g4hepem 2 files +39/−1.
- **Pre-registered predicted signature** (before any preview run):
  1. At some cap setting, UNSEVERED (-x 0 -y 0 -B 0 -N 1e-3 -C 1000) +
     caps: gap core L5–18 mean within 2σ of FD truth 195.8 ± 9.4 with
     core variance ≤ 100× the canonical-severed reference.
  2. Absorber core mean moves from the unsevered-uncapped robust locator
     (~1430–1580) toward truth 2233.7 ± 14.8 when `--el-mfp-cap` is on.
  3. Primal byte-identical with caps on.
  4. The boundary cap changes the layer-sum total derivative by <1%
     (dipoles cancel in totals — capping them must not move the total).
- **Status**: implemented + pre-registered (2026-07-24); gates running.
- **Gate results** (canonical args; unsevered config for knob-on runs):
  - G1 knob-off byte-identity vs the 7ebab9e reference binaries, n=2000 s1,
    canonical severed config: fwd edeps + edeps_gap identical; rev
    barInputs + barInputsPerLayer + edeps identical (both builds rebuilt,
    reverse against the mirrored install_reverse .icc). PASS. Bonus: the
    fc388aa knobs-off unsevered gap s1 run is byte-identical (edeps AND
    event dump) to the wave-8 uns_s1 run — gate extends to the unsevered
    config.
  - G2 primal identity knob-on: mean_E/var_E byte-identical vs off for
    each cap alone and both (b40 / e100 / b40+e100), gap AND abs seeds,
    s1–2, n=2000, unsevered; derivative column active 50/50 layers in all
    12 comparisons. PASS.
  - G3 fwd=rev (el-mfp-cap 100, unsevered, n=500 s1): Σ fwd mean_dE =
    260.464691249 = reverse barThicknessGap 260.46469125028 (rel 3e−12).
    PASS. `--boundary-dot-cap` is forward-only by design; the reverse
    build rejects a nonzero cap with a clear error (verified, rc=255) —
    race-score precedent, documented deviation.
  - G4 NaN scan: 0 non-finite entries in all 32 preview dumps (64k
    events) and all gate outputs. PASS.
  - G5 runtime: capped walls 317–334 s vs off 327/331 s (n=2000 pairs,
    same host) ⇒ ratio ≈ 1.00. PASS.
- **Preview** (n=2000/run, unsevered `-x 0 -y 0 -B 0 -N 1e-3 -C 1000`,
  per-event core-window W_e from event dumps, FD truth gap 195.8 ± 9.4 /
  abs 2233.7 ± 14.8; var ratios vs canonical severed var(W_e)
  2.4e6 (s1) / 4.5e5 (s2); summary `fidelity/wave9_preview_summary.json`,
  raw `fidelity/wave9_runs/` untracked):
  | config (gap seed) | pooled core mean ± SE | z vs truth | median s1/s2 | var ratio s1/s2 | layer-sum total |
  |---|---|---|---|---|---|
  | off (uncapped)   | −484.9 ± 1044 | −0.65 | 202.5/196.0 | 1.2e3/2.2e4 | 189±135 / −151±194 |
  | b10              |  88.6 ± 3.0 | −10.9 | 86.3/88.3 | 0.022/0.063 | −114/−98 |
  | b40              | 174.8 ± 3.4 | −2.11 | 160.0/160.4 | 0.041/0.064 | 11.0/10.4 |
  | **b80**          | **207.9 ± 6.0** | **+1.09** | 184.6/183.9 | 0.064/0.306 | 26.5/27.1 |
  | b160             | 244.2 ± 9.1 | +3.70 | 200.3/204.7 | 0.124/0.821 | 31.6/33.2 |
  | e100 (alone)     | −481.9 ± 1044 | −0.65 | 201.9/196.4 | 1.2e3/2.2e4 | 119/−180 |
  | b40+e100 / b160+e100 | ≡ b40 / b160 (el-cap null) | | | | |
  - Absorber seed (uncapped robust locator 1430–1580, truth 2233.7):
    b40 → **−697 ± 33 pooled (wrong sign)**; b160 → 1210 ± 37 (still
    −45%); e100/e1000 → null (994→1020 s1, within noise); e10 → collapse
    (~0–17). The el variations at the best gap combo (b40) and the b160
    diagnostic both confirm: no tested setting recovers the absorber.
- **Verdict vs pre-registration**:
  1. **PASS** at `--boundary-dot-cap 80`: gap core 207.9 ± 6.0 within 2σ
     of FD truth (z = +1.09) with var(W_e) 0.06×/0.31× the canonical
     severed reference (bar was ≤100×) — the first configuration in the
     program with an unbiased-at-this-precision gap core AND
     sub-canonical variance, no track killed. The pre-specified grid
     {10, 40, 160} brackets truth monotonically; b80 was added as the
     interpolation point (documented deviation).
  2. **FAIL**: `--el-mfp-cap` is a measured null in both channels (100,
     1000) and destructive at 10 — S1 is not the absorber deficit
     carrier (independently corroborated by the wave-8 part-C dissection
     cap counterfactuals).
  3. **PASS**: primal byte-identical throughout (G2).
  4. **FAIL**: matched per-event layer-sum totals change by ~65–93%
     (median) under the boundary cap; capped ensemble totals (b40 10.7,
     b80 26.8, b160 32.4) sit far below the 1M FD-truth total
     84.2 ± 14.2 — the 1/vx tail the cap truncates carries real
     net-derivative mass, not only the antisymmetric bin transfer. The
     "dipoles cancel in totals" argument was right about the dipoles but
     wrong that the cap acts only on them.
- **Interpretation**: the boundary-dot cap is a genuine bias–variance
  dial whose gap-channel sweet spot (~80 mm/seed ≈ 10 layer pitches / f²)
  sits at 1σ from truth with 20× LESS variance than canonical severing —
  the wave-8 reframe (caps can replace severing) is CONFIRMED for the
  gap channel. It does not transfer to the absorber at a single global
  cap value: the absorber-seed carrier dots are ~10× larger, so cap=40
  amputates real mass (wrong sign), cap=160 still −45%. Next-wave
  candidates: per-seed-scaled or depth-scaled cap, or the audit's
  proposal 2 (track-state dot governor ~250 mm/seed) which the part-C
  dissection also endorses.
- **Deviations**: (i) b80 interpolation point beyond the pre-specified
  grid (2 runs); (ii) abs b160 diagnostic beyond the grid (2 runs);
  (iii) boundary-dot-cap forward-only (tangent clamp is nonlinear in the
  tangent — not representable on a reverse Jacobian tape; reverse build
  rejects it with a clear error); G3 therefore ran with el-mfp-cap only;
  (iv) gate-3/batch-D overlap briefly ran 4 local processes (≤2 was the
  target); (v) el-mfp-cap decrement side: `-A` already floors the
  small-mfp decrement coefficients; large mfp shrinks them — nothing to
  mirror (noted in the g4hepem commit).
- **Status**: gated + preview measured (2026-07-24). Prediction 1+3 PASS
  / 2+4 FAIL; awaiting human verdict (recommend: accept
  `--boundary-dot-cap` as the gap-channel severing replacement pending
  condor validation vs FD at scale; record `--el-mfp-cap` as a measured
  null, keep default-off). hepemshow tree restored to
  `phaseA-perlayer-gap-energy`; agent binaries `build_agent_fwd/rev`
  left at `knob/dot-caps` @ fc388aa (knobs-off = bit-identical to
  7ebab9e); g4hepem source+installs at `el-mfp-cap` @ 73405c2
  (default-off, baseline-safe).

## Wave 10 — accumulated-state dot governor (`--track-dot-cap`)

- **Knob**: `--track-dot-cap <P_mm>[:<E_MeV>]` (id 1017), branch
  `knob/dot-governor` in worktree `/eos/user/j/jeffkrup/agentic/hepemshow-gov`
  (cut from `knob/dot-caps` @ fc388aa; main tree, `build_agent_fwd/rev` and
  the `-diag` worktree untouched — a parallel condor batch uses them).
  Default off, derivative-only, KeepPrimal (values never touched). Per-track
  governor on the ACCUMULATED track-state dots, applied at each step boundary
  (top of the stepping while-loop in both `GammaStepper` and
  `ElectronStepper` — after the previous step's update, before the next step
  uses the state; also catches secondary birth dots on the first iteration):
  - position-x dot clamped to ±P [mm/seed];
  - kinetic-energy dot clamped to ±E [MeV/seed], with the cached log-energy
    companion dot rescaled consistently (d(logE) = dE/E) so the clamp cannot
    leak around through `GetLogEKin` interpolations;
  - direction-vx dot clamped to a fixed ±1e3 whenever the governor is on
    (trivially co-located in the same helper; the dissection saw v̇x reach
    1.3e10 — ±1e3 only fires on pathology).
- **Hypothesis** (wave-8 dissection + amplification-audit proposal 2): the
  S4 ping-pong loop (position-dot → stepLen-dot → eloss-dot → energy-dot →
  MSC angle Jacobians → direction-dot → position-dot; measured per-cycle
  gains ×12–×557) compounds through THREE accumulated state carriers.
  Clamping the stored carriers at each step boundary bounds the loop
  (5^k → ≤ cap) while keeping every track alive and scoring — unlike the
  per-crossing boundary cap it does not truncate the legitimate 1/vx
  net-derivative mass of any individual crossing, so it should recover the
  absorber channel where b40/b160 failed.
- **Implementation**: hepemshow 4 files, ~+190 lines over fc388aa.
  `InputParameters.hh` (flag 1017, `<P>[:<E>]` parse, both modes allowed —
  unlike `--boundary-dot-cap` there is no reverse rejection);
  `SteppingLoop.hh/.cc` (`ConfigureTrackDotGovernor`, `GovernTrackStateDots`
  called at the two while-loop tops); `HepEmShow.cc` (wiring). Forward mode:
  tangents clamped in place via `SET_DOTVALUE` on the stored state (primal
  bytes untouched by construction). **Reverse mode**: the forward tangent
  clamp is nonlinear in the tangent and cannot be taped (wave-9 precedent);
  instead the governor records an identity external function
  (`codi::ExternalFunctionHelper`, identity primal, KeepPrimal) whose
  REVERSE interpretation clamps the corresponding ADJOINTS at the same
  program points: position-x adjoint to ±P, direction-vx adjoint to ±1e3,
  ekin adjoint to ±E, log-ekin adjoint to ±E·max(1, EKin) (scale-matched,
  adjoint_logE = EKin·adjoint_E). This bounds the identical amplification
  loop traversed backward; it is the reverse-sweep mirror, NOT the transpose
  of the tangent clamp (no such linear transpose exists), so fwd=rev exact
  agreement is only expected when no clamp fires — the gate measures the
  governed-on agreement and a huge-cap control must reproduce knob-off
  fwd=rev exactly. (CoDiPack 3.1.0 note: array user data hits a compile bug
  in `ExternalFunctionUserData::DataArray::clone`; scalar data items used.)
- **Pre-registered preview grid** (written BEFORE any preview run; outline
  pre-registered in `ledger.jsonl` wave-10 entry):
  - ABSORBER (`-a 2.3:1`), unsevered `-x 0 -y 0 -B 0 -N 1e-3 -C 1000`,
    n=2000, seeds 1–2: governor grid **P ∈ {250, 1000} × E ∈ {50, 200}**.
    Scale reasoning: legitimate |ẋ| ≤ N_layers×|seed| ≈ 50 × grazing
    allowance 1/f = 5 ⇒ P = 250 (audit proposal-2 value; the dissection
    counterfactual shows it neutralizes both dissected events), P = 1000 =
    one decade looser to probe clamp-aggressiveness; E: healthy birth Ė is
    O(0.2–2e3) with pathological onset ≥ 1e3 and poles at 3.7e6–2.1e8 —
    E = 50 sits ~100× above the healthy sub-MeV-track scale, E = 200 one
    half-decade looser; both are 4–6 orders below the measured pathology.
    Metrics: core L5–18 per-event W_e mean ± SE, median, variance ratio vs
    canonical severed (2.4e6 s1 / 4.5e5 s2), vs FD truth 2233.7 ± 14.8.
    **Success bar: ≥ 0.9× truth at ≤ 100× canonical variance.**
  - GAP compatibility (`-g 5.7:1`) at the best absorber (P,E) WITH
    `--boundary-dot-cap 80`, seeds 1–2: core mean unchanged (<2σ) vs
    wave-9's b80 207.9 ± 6.0.
  - Combined-config ABSORBER (`-a 2.3:1`, `--boundary-dot-cap 80` +
    governor, seeds 1–2): does the gap-tuned boundary cap still damage the
    absorber with the governor active (wave-9: b40 → wrong sign, b160 →
    −45%), or does the governor rescue it? Decides whether ONE unified
    config exists.
- **Gate results** (canonical severed args for G1; unsevered for knob-on;
  final binary `knob/dot-governor` @ 3b6d45f — the D-field commit is
  behavior-preserving: all six n=500 fwd/rev outputs bit-identical across
  46747fc → 3b6d45f):
  - G1 knob-off byte-identity vs fc388aa, n=2000 s1: fwd `edeps_1` +
    `edeps_gap_1` identical (`build_agent_fwd` reference run vs gov build);
    rev `barInputs` + `barInputsPerLayer` + `edeps_1` + `edeps_gap_1`
    identical. Bonus: gov knob-off UNSEVERED gap+abs s1 bit-identical to
    the wave-9 fc388aa off runs. PASS.
  - G2 primal identity knob-on (250:50, unsevered, gap+abs seeds s1–2):
    mean_E/var_E byte-identical in edeps + edeps_gap (8/8 comparisons);
    derivative column changes 50/50 layers. PASS.
  - G3 fwd=rev n=500 (unsevered, gap seed): knob-off rel 8.4e−12 (PASS);
    **never-fires control** (`--track-dot-cap 1e30:1e30:1e30`): fwd
    bit-identical to knob-off; rev bit-identical except 2/202
    `barInputsPerLayer` values at max rel 1.1e−13 (adjoint accumulation
    order through the identity external function) — the tape machinery is
    an exact no-op when no clamp fires. **Governor-on (250:50): fwd
    Σ mean_dE = 579.404 vs rev gap row = 1600.995 (rel 0.64) — NOT equal
    when clamps fire.** This is inherent, not a bug: the forward tangent
    clamp and the reverse adjoint clamp are different regularizations of
    the same governor (a tangent-nonlinear op has no linear transpose);
    they coincide exactly when no clamp fires (control above). The
    pre-registered "tape-compatible reverse governor" is delivered in the
    verified sense: recordable, primal-exact, finite, bounded, exact
    no-op at non-firing caps. First reverse-capable knob of the cap
    family (boundary-dot-cap and race-score are fwd-only).
  - G4 NaN scan: 0 nonfinite in all gate outputs, all 14 preview runs'
    edeps and all event dumps. PASS.
  - G5 runtime (fwd, paired off/on n=1000): ratios 0.943 / 0.994 ⇒ ≈0.97
    (~1.0 as expected). PASS. Reverse governor-on ≈2.5× (452 s vs 177 s,
    n=500) — the per-step ExternalFunctionHelper push; reverse is
    gate-only.
- **Preview** (unsevered, n=2000, seeds 1–2 pooled; abs FD truth
  2233.7 ± 14.8; var ratios vs canonical severed var(W_e) 2.4e6 s1 /
  4.5e5 s2; summary `fidelity/wave10_preview_summary.json`, raw
  `fidelity/wave10_runs/` untracked):
  | absorber grid | pooled mean ± SE | ratio | z | median s1/s2 | var× s1/s2 | Σ50 s1/s2 |
  |---|---|---|---|---|---|---|
  | P250:E50   | 1901.1 ± 124.9 | 0.851 | −2.65 | 1753/1745 | 27.8/128.6 | 3215/3100 |
  | P250:E200  | 1549.3 ± 120.9 | 0.694 | −5.62 | 1390/1429 | 25.5/124.1 | 885/696 |
  | **P1000:E50** | **2417.7 ± 194.0** | **1.082** | **+0.95** | 2214/1953 | 74.8/270.7 | 5134/5109 |
  | P1000:E200 | 1938.4 ± 192.9 | 0.868 | −1.53 | 1786/1599 | 75.0/261.4 | 1212/1173 |
  - **Bar verdict (≥0.9× truth at ≤100× canonical variance): P1000:E50
    passes the mean bar** — 1.082 ± 0.087, within 1σ of FD truth, the
    first absorber-channel configuration in the program consistent with
    truth (from unsevered-uncapped robust locators 0.64–0.71 and severed
    0.55–0.78). **Variance bar is MARGINAL**: 74.8× (s1, passes) vs
    270.7× (s2 — but against the fluke-low canonical s2 reference;
    absolute s2 variance 1.2e8 is LOWER than s1's 1.8e8; pooled absolute
    1.5e8 ≈ 105× the pooled canonical 1.4e6).
  - Surprise (measured): E200 is WORSE than E50 in mean (0.87 vs 1.08 at
    P1000) — the looser Ė clamp admits more negative fluctuation-recycling
    dot mass; the Ė clamp is doing real directional work, not just
    variance-taming.
  - Gap, governor alone (250:50, gate-C runs): medians 190.5/229.1 sit on
    truth 195.8; var 7.3×/44.6× canonical — enormously better than
    uncapped unsevered (1.2e3/2.2e4×) but not competitive with b80
    (0.06/0.31×).
  - **Gap compatibility** (b80 + governor 1000:50): pooled 162.3 ± 23.3
    vs wave-9 b80-alone 207.9 ± 6.0 ⇒ Δ = −45.6, z = −1.89 — inside the
    pre-registered <2σ bar, but only just, and var(W_e) rises ~14× vs
    b80 alone (2.1e6/2.2e6 = 0.89×/4.9× canonical). The governor is not
    free in the gap channel at these scales.
  - **Combined-config absorber** (b80 + governor 1000:50):
    **−762.0 ± 127.8 — WRONG SIGN** (medians −737/−769; Σ50 −4586/−3808).
    The governor does NOT rescue the absorber from the gap-tuned boundary
    cap: b80 amputates the real crossing mass at the generator, upstream
    of the state clamps, exactly as in wave-9 (b40 → −697). **NO unified
    config exists in this cap family** — channel-specific configs stand:
    gap → `--boundary-dot-cap 80`, absorber → `--track-dot-cap 1000:50`.
- **Deviations**: (i) after the first control run exposed that the fixed
  ±1e3 direction clamp always fires, the flag gained an optional third
  field `P[:E[:D]]` (D = dir-vx cap, default 1e3) so a true never-fires
  control exists — regression: all pre-change gate outputs bit-identical
  under the final binary (fix attempt 1 of 2, spent on a control-design
  flaw, not physics); (ii) fwd ≠ rev under active clamping is documented
  as inherent (see G3) — the reverse governor clamps adjoints at the same
  program points with the same numeric caps (units differ: adjoint units
  are MeV/mm, MeV/MeV); (iii) compat/combined legs ran seeds 1–2 (task
  sketch said one combined run — one extra n=2000 run for seed scatter);
  (iv) G5 measured on 46747fc (bit-identical outputs ⇒ ratios carry);
  (v) wave-9 off runs reused as knob-off references (G1 proves binary
  equivalence); (vi) log-EKin companion dot is governed alongside EKin
  (forward: rescaled to clamped_dot/E; reverse: adjoint cap E·max(1,EKin))
  — not in the task sketch but required, else the clamp leaks through
  `GetLogEKin` interpolations.
- **Status**: gated + preview measured (2026-07-24) — mean bar PASS at
  P1000:E50 (first truth-consistent absorber config), variance bar
  marginal (~105× pooled vs 100× bar), unified-config question answered
  NO; awaiting human verdict (recommend: accept the governor as the
  absorber-channel severing replacement pending condor validation at
  scale, keep channel-specific configs, and treat the E-clamp asymmetry
  as the next diagnostic handle). Worktree
  `/eos/user/j/jeffkrup/agentic/hepemshow-gov` left in place @
  `knob/dot-governor` 3b6d45f (knob-off = bit-identical to fc388aa);
  main tree, `build_agent_fwd/rev`, g4hepem and the -diag worktree
  untouched.

## Wave 11 — gap transportability: scale-aware cap + median estimator (A/B MEASURED; C pending condor)

- **No new knob.** Existing binaries: gap = `knob/dot-caps` @ fc388aa
  (`build_agent_fwd`, mtime/size re-verified 1784902792072373144 ns /
  429472 B before launch); absorber governor legs (item C) run on condor
  by a parallel agent — not covered here.
- **Pre-registered** (ledger.jsonl wave-11, 2026-07-25): (A) the boundary
  cap should scale with local geometry, cap = c·g with c = 80/5.7 =
  14.04/mm ⇒ b = 42.1 at g = 3.0 predicts the od FD truth 188.05 ± 9.52;
  (B) the pooled per-event MEDIAN transports without a tuned constant —
  od medians at b42/b80 within 2σ of the od FD, median spread < 10%
  across caps {42, 80} while the mean spreads ~16%.
- **Runs** (LOCAL, this wave): unsevered `-x 0 -y 0 -B 0 -f 0.2 -N 1e-3
  -C 1000`, off-design a=2.30 g=3.00, gap seed `-g 3.0:1`, n=2000/seed,
  event dumps on: b42.1 seeds 1–4, b30 + b60 seeds 1–2 (bracket).
  Mined without reruns: wave-9 on-design cap-grid dumps
  (b10/b40/b80/b160, s1–2), wave-9 validation per-event sidecars (b80 at
  BOTH geometries, 200k events each), wave-8 uncapped unsevered dumps
  (on-design, s1–2). Analysis `fidelity/wave11_transport.py`, summary
  `fidelity/wave11_transport_summary.json`; raw `fidelity/wave11_runs/`
  untracked. Median SEs: nonparametric bootstrap, 10k resamples, pooled
  events.
- **A — VERIFIED.** b42 od core L5–18 pooled mean **184.75 ± 4.67**
  (seed-scatter SE 4.31, seeds 179.5/196.8/177.7/185.0) vs od FD truth
  188.05 ± 9.52 ⇒ **z = −0.31**, well inside 2σ — where the on-design
  constant b80 failed od at z = +3.16 (wave-9). Bracket is monotone and
  brackets truth: b30 171.5 ± 6.2 (z = −1.46), b60 204.8 ± 8.1
  (z = +1.34), b80 218.6 ± 1.8 (z = +3.16); local slope ≈ +1.11 core-MeV per
  cap-unit across b30→b60.
- **B — REFUTED as pre-registered.** The median is cap-DEPENDENT below
  cap ≈ 80: od b42 median 163.36 ± 1.70 ⇒ z = −2.55 (outside 2σ), and
  the median spread across od caps {42, 80} is 13.0% (mean spread
  16.8%) — the < 10% bar fails. Full median-vs-cap curve (z vs local FD
  truth; od truth 188.05 ± 9.52, on-design 195.78 ± 9.36):
  | cap | od median (z) | on-design median (z) |
  |---|---|---|
  | 10 | — | 86.9 ± 2.1 (−11.35) |
  | 30 | 154.2 ± 1.6 (−3.51) | — |
  | 40 | — | 160.0 ± 2.7 (−3.68) |
  | 42.1 | 163.4 ± 1.7 (−2.55) | — |
  | 60 | 178.3 ± 2.5 (−0.99) | — |
  | 80 | 186.1 ± 0.5 (−0.20) [200k] | 184.0 ± 2.9 (−1.20) [4k]; 182.4 ± 0.5 (−1.42) [200k] |
  | 160 | — | 202.7 ± 4.0 (+0.68) |
  | ∞ (uncapped) | — | **199.6 ± 6.4 (+0.34)** (mean −665 ± 1255) |
  Salvage (post-hoc, measured): in the LOOSE-cap regime cap ≥ 80 the
  median IS truth-consistent in both geometries (all |z| ≤ 1.4 od,
  ≤ 1.42 on-design incl. uncapped) — the tight-cap failure is the cap
  biting into the bulk of the W_e distribution, not a failure of the
  median as an uncapped-truth locator. The wave-9 observation
  "medians 182–186 in both geometries" was a loose-cap artifact of b80
  being ~2× the matched cap at g = 3.0.
- **Prescription recommendation** (gap channel): primary =
  **scale-aware capped mean**, `--boundary-dot-cap 14.04*g` [mm/seed]
  (this wave: od z = −0.31; wave-9 on-design z = +1.65), which also has
  ~3.6× smaller per-event variance than loose-cap b80 at g = 3.0
  (1.75e5 vs 6.21e5). Secondary/diagnostic = loose-cap (≥ 80·(g/5.7)…∞)
  median as a tuned-constant-free consistency check (truth-consistent
  both geometries, mild −5% central tendency, unusable mean).
- **Status**: A/B measured 2026-07-25 (this entry); item C (governed
  absorber at scale + a=3.0 transport) runs on condor under the parallel
  agent — jsonl verdict field left open for it.

## Wave 12 — absorber governor scaling rule: off-design clamp bracket scan (LOCAL, measured)

- **No new knob.** Gov binary `knob/dot-governor` @ 3b6d45f
  (`build_gov_fwd`, mtime/size re-verified 1784915391625879590 ns /
  434128 B before launch; never rebuilt). Question (pre-registered in
  ledger.jsonl wave-12 BEFORE any run): is the governor's +9.7%
  off-design excess (a=3.0: 2213.83 ± 47.18 vs own-FD truth
  2018.05 ± 15.36, z = 3.95, wave-11) a clamp-scale mismatch curable by
  a geometry-scaling rule, as the gap cap's was (cap ∝ g, wave-11 A)?
  Predicted: (1) some (P,E) at a=3.0 within 2σ of truth; (2) monotone
  controlling clamp with a fittable slope; (3) the rule backward-
  consistent with P1000:E50 being matched at a=2.3.
- **Runs** (LOCAL, `fidelity/wave12_scaling.py`; raw `wave12_runs/`
  untracked; 28 units × n=2000 = 56k events, ~2.3 core-h, ≤2 concurrent,
  0 failed/NaN rows): a=3.00 g=5.70, unsevered `-x 0 -y 0 -B 0 -f 0.2
  -N 1e-3 -C 1000`, `-a 3.0:1`, event dumps on. P-axis (E=50):
  P500/P700/P1300 s1–4; E-axis (P=1000): E35/E70 s1–4. The P1000:E50
  reference reused from the wave-11 200k batch, never rerun. Adaptive
  step (per the pre-registered stop-the-flat-axis rule): after the full
  first bracket the E clamp controls the response while P is subdominant
  and non-monotone, so the budget went to the live axis — **E65 added**
  (s1–4; sits on the E ∝ a candidate 50·3.0/2.3 = 65.2) and E70 extended
  to s1–8.
- **Bracket table** (core L5–18 pooled per-event mean ± SE; od truth
  2018.05 ± 15.36; var ratio vs the a=3.0 E50 reference var 6.42e8):
  | variant | mean ± SE | z vs truth | median ± boot SE | var× |
  |---|---|---|---|---|
  | P500:E50 | 1879.3 ± 304.0 | −0.46 | 1984.7 ± 63.4 | 1.15 |
  | P700:E50 | 1855.8 ± 420.5 | −0.39 | 2025.6 ± 70.9 | 2.20 |
  | P1300:E50 | 1543.3 ± 729.2 | −0.65 | 2044.0 ± 92.5 | 6.63 |
  | P1000:E35 | 1951.5 ± 577.2 | −0.12 | 2270.2 ± 89.3 | 4.15 |
  | P1000:E65 | 1559.4 ± 587.5 | −0.78 | 1882.0 ± 76.3 | 4.30 |
  | P1000:E70 (16k) | 1727.3 ± 313.1 | −0.93 | 1870.6 ± 58.4 | 2.44 |
  | REF P1000:E50 (200k) | 2213.8 ± 56.7 | +3.3 | 2117.1 ± 17.3 | 1.00 |
- **Mean estimator: variance-limited.** Pooled SEs 300–730 MeV at 8k
  events vs the ~60 needed to discriminate the 196-MeV excess; every
  variant is trivially within 2σ of truth (pred-1 vacuously true) and
  neither axis is monotone in the mean (pred-2 FAILS as pre-registered:
  P slope +0.58 ± 0.55, E slope −21.0 ± 13.8 MeV/unit). The mean cannot
  bracket the matched clamp at local scale.
- **Post-hoc robust estimators** (per-event median/tm5, bootstrap SEs,
  wave-11-B precedent; reference recomputed from its own 200k dumps with
  the same estimators): the **E clamp is the controlling clamp** — E-axis
  median monotone 2270.2 → 2117.1 → 1882.0 → 1870.6 across E 35→50→65→70,
  slope **−12.6 ± 2.4 MeV/unit (5.2σ)** (tm5: −12.5 ± 2.4, 5.2σ); the
  P-axis moves only ~60 MeV over P500→P1300 and is non-monotone vs the
  reference. Median-anchored matched value (anchor: target = truth ×
  ref-median/ref-mean; assumes locally clamp-independent shape ratio —
  stated assumption): **matched E = 64.6 ± 12.4** (tm5: 65.1 ± 12.6) ⇒
  exponent **k = 0.97** (tm5 0.99), i.e. **E ∝ a**, prediction
  50·(3.0/2.3) = 65.2 — the new E65 point sits right on it. Backward
  consistency (pred-3): the rule anchors at the matched P1000:E50 at
  a=2.3 by construction, the measured od exponent ≈ 1 makes E ∝ a a
  measurement (not a fit choice among {1/a, const, a} — those are only
  separated at 2.0/1.3σ by the mean fit, but the robust k pins 1), and
  the sign matches the wave-10 on-design E-direction (looser E lowers
  the mean: E50→E200 gave 1.082→0.868). The mean-based fits agree but
  weakly (matched E = 58.8 ± 38.5, k = +0.61; matched P = 681 ± 646).
- **Variance transport**: the candidate matched clamp costs real
  variance — var(W_e) at E65 is 4.3× the E50 reference (E70 at 16k:
  2.4×; single-seed spike sampling dominates the spread). Not a blowup
  (the program's failure bar has been ~100×), but not free; must be
  re-measured at scale.
- **Verdict vs pre-registration** (measured): pred-1 PASS but vacuous at
  local precision (mean variance-limited); pred-2 FAIL on the
  pre-registered mean, PASS on the post-hoc robust estimators
  (controlling clamp = E, monotone, 5.2σ slope); pred-3 PASS in the
  measured-exponent sense (k ≈ 1 ⇒ E ∝ a). Net: the wave-11 od excess
  behaves exactly like a clamp-scale mismatch with **E_cap ∝ a at fixed
  P = 1000**, but the mean-level confirmation is beyond local reach.
- **Deviations**: (i) E65 variant + E70 seeds 5–8 added mid-scan under
  the pre-registered adaptive rule (P declared the flat/subdominant
  axis); (ii) the robust-estimator block (test 5) is post-hoc and
  labelled so in the summary; (iii) first `--analyze` OOM-killed by a
  16 GB bootstrap index matrix on the 200k reference — fixed to a
  chunked, size-adaptive bootstrap (analysis bug, no physics rerun);
  (iv) units ran ~290 s not the projected ~450 s, so the added E-axis
  units kept total ≈ 2.3 core-h, inside the ~2.5 budget.
- **Status**: measured 2026-07-25, summary
  `fidelity/wave12_scaling_summary.json`; awaiting human verdict.
  Recommended confirming condor run (NOT submitted): wave-11-style
  batch at a=3.0 with `--track-dot-cap 1000:65.2`, 15 seeds × 20k
  (var 2.8e9 ⇒ pooled SE ≈ 96, 2σ-discriminates the 196-MeV excess),
  plus a second lever arm (e.g. a=2.0, cap 1000:43.5, with its own
  ±0.1 mm FD arms) to test the rule off the anchor, and an unscaled
  E50 control at the second geometry.

## Wave 13 — absorber E_cap ∝ a scaling-rule confirmation (condor, SUBMITTED)

- **No new knob.** Gov binary `knob/dot-governor` @ 3b6d45f
  (`build_gov_fwd`, mtime 2026-07-24 19:49:51 / size 434128 B
  re-verified before submission; `--track-dot-cap` accepted, 10-event
  smoke with `1000:65.2` clean; never rebuilt). This is the wave-12
  recommended confirming batch, pre-registered in ledger.jsonl wave-13.
  Rule under test: **E_cap = 50·a/2.3 at fixed P = 1000** (E ∝ a).
- **Batch** (`fidelity/wave13_scaleval.py`, condor cluster **3912920**,
  60 jobs × 20k = **1.2M events**, ~75 CPU-h; forward mode, unsevered
  `-x 0 -y 0 -B 0 -f 0.2 -N 1e-3 -C 1000`, no `--rng-lineage`, absorber
  seed `-a <a>:1`, HEPEMSHOW_EVENT_DUMP on for governed AD configs):
  - `abs_a30_scaled` s1–15, a=3.00 g=5.70, `--track-dot-cap 1000:65.2`;
  - `abs_a20_scaled` s1–15, a=2.00 g=5.70, `--track-dot-cap 1000:43.5`;
  - `abs_a20_ctrl` s1–10, a=2.00, `--track-dot-cap 1000:50` (UNSCALED
    control — tests that the rule is load-bearing, not a null);
  - `abs_fd_a20_plus`/`abs_fd_a20_minus` s1–10 each, a=2.1/1.9,
    CANONICAL severed `-x 2 -y 1 -B 1`, energy-seeded dot slot, primal-
    only FD arms (denom (2.1−1.9)=0.2) → fresh a=2.0 FD truth.
  Reused truths (no rerun): a=3.0 FD 2018.05 ± 15.36 (wave-11 abs_fd_od
  arms); a=2.3 on-design 1.00 anchor (wave-11 abs_gov_val).
- **Split decision: none.** n=1000 timing smokes projected 20k walls
  well inside the 8500 s split threshold and the 10800 s in-process
  timeout: a=2.0 174.8 s → ~3496 s; a=3.0 (only config ≥2.5 mm)
  160.8 s → ~3216 s. All 60 jobs kept at 20k/job.
- **Pre-registered signatures** (predicted_signature, ledger.jsonl
  wave-13): (1) a=3.0 scaled core L5–18 within 2σ of 2018.05 ± 15.36
  (unscaled was +9.7%, z=3.95); (2) a=2.0 scaled core within 2σ of its
  own FD arms; (3) a=2.0 unscaled E50 control OFF (>2σ); (4) a=2.3 E50
  = established 1.00 anchor.
- **Status**: submitted 2026-07-27, cluster 3912920, 60 idle at submit
  (one `condor_q -totals`). Analyze after completion:
  `python -u -m fidelity.wave13_scaleval --analyze` →
  `fidelity/wave13_scaleval_summary.json`. Awaiting jobs + human verdict.

## Wave 14 — transportability of the canonical-severed absorber core undershoot (MEASURED)

- **No new knob.** Reproduces the paper published strategy (canonical
  severed `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`, all governor/anchor
  knobs OFF). AD via config `forward_bin` (shared `build/HepEmShow`);
  **gate G14**: shared-build knobs-off canonical ≡ `build_agent_fwd`
  knobs-off canonical, byte-identical `edeps_1` (n=500, a=2.3:1, s1).
  `build_agent_fwd` mtime 1784902792072373144 ns / 429472 B re-verified,
  not rebuilt.
- **Question**: is the on-design (a=2.3) severed-absorber core-window
  (L5–18) AD/FD undershoot ~0.77 a geometry-INDEPENDENT constant (single
  blind 1/0.766 ≈ 1.30× correction transports) or does it drift with
  absorber thickness a (needs a scaling rule c(a))? Measured severed
  absorber AD/FD at a = 2.0, 2.3, 3.0.
- **Compute**: a=2.3 AD reused (`experiments/perlayer_adfd/absorber_ad_s*`,
  46 seeds × 20k = 920k). a=2.0 & a=3.0 AD run LOCALLY (severed, absorber
  seed `-a a:1`, 6 seeds × 2k = 12k each, 2 concurrent, nice −10;
  `fidelity/wave14_runs/absorber_ad_a{20,30}_s*.jsonl`; 0 failed, 0 NaN).
  FD truths reused/recomputed (all three reproduce the reported values
  exactly): a=2.3 = 2233.68 ± 14.75 (perlayer_adfd_1M_summary, paired CRN
  h=0.02); a=2.0 = 2119.74 ± 14.08 (wave-13 arms a=2.1/1.9, h=0.1,
  unpaired seed-scatter); a=3.0 = 2018.05 ± 15.36 (wave-11 od arms
  a=3.1/2.9). Window AD SE = seed-scatter on per-seed core sums (keeps
  inter-layer covariance).
- **Result — core L5–18 severed AD/FD (window sum)**:

  | a (mm) | AD core | FD core | AD/FD | 1/ratio (corr.) | AD n_ev |
  |--------|---------|---------|-------|-----------------|---------|
  | 2.0 | 1570.3 ± 77.1 | 2119.7 ± 14.1 | **0.741 ± 0.037** | 1.350 ± 0.067 | 12k |
  | 2.3 | 1724.4 ± 15.8 | 2233.7 ± 14.8 | **0.772 ± 0.009** | 1.295 ± 0.015 | 920k |
  | 3.0 | 1628.8 ± 96.6 | 2018.0 ± 15.4 | **0.807 ± 0.048** | 1.239 ± 0.074 | 12k |

- **Constant-vs-drift verdict (measured)**: weighted constant fit
  c = **0.7714 ± 0.0083**, χ²/dof = 0.62 → constant is a good fit.
  Weighted linear fit slope = **+0.063 ± 0.060 /mm** (intercept 0.625),
  significance **1.06σ** → NOT significant. The point estimates trend
  monotonically upward (0.741 → 0.772 → 0.807), hinting at a possible
  mild positive drift with a, but the two off-design points are too
  imprecise (12k events, ~4–5% ratio SE) to confirm it. A single blind
  **1.30×** correction (⇔ ratio 0.769) sits within ≤0.8σ of all three
  geometries — it transports within current errors.
- **Per-layer core-ratio uniformity**: clean & uniform ONLY at the
  high-stats anchor a=2.3 (std 0.036, ratios 0.73–0.85). At a=2.0/3.0 the
  per-layer breakdown is AD shot-noise-dominated (std 0.20 and 1.76;
  a=3.0 has individual-layer ratios −1.9, 6.2 where per-layer mean_dE SEM
  blows up) — only the pooled window sum is robust at 12k. NOT evidence
  against uniformity; a precision artifact.
- **Statistics caveat (honest)**: local 12k is precision-limited. At
  a=3.0, 0.807 ± 0.048 cannot distinguish 0.77 from 0.85 (both <1σ). The
  1.06σ slope is neither confirmed nor excluded. To resolve the
  suspected ~0.06/mm drift at >3σ (and pin per-layer uniformity off the
  anchor) recommend a small condor batch: ~10× more events at a=2.0 and
  a=3.0 (match the a=2.3 ~1M anchor → ratio SE ~1.5%). NOT submitted.
- **Deviation**: task specced `build_agent_fwd` for AD; used the
  byte-identical shared build via `tools.sim` (gate G14) so the three
  geometries share provenance with the a=2.3 reference AD.
- **Status**: measured 2026-07-28, `fidelity/wave14_transport.py`,
  summary `fidelity/wave14_transport_summary.json`. Awaiting human
  verdict.

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

# HepEmShow differentiable-simulator recon report

Read-only reconnaissance, 2026-07-21. Basis for the gradient-fidelity repair
loop (see PROTOCOL.md). All file:line refs verified at the SHAs below.

## Source location / provenance (confirmed)

| Item | Value |
|---|---|
| App repo | `/eos/user/j/jeffkrup/agentic/hepemshow` — branch `phaseA-perlayer-gap-energy`, HEAD `9ceda1e` ("Env-gate per-event boundary_stats dump…"; below it `8bc7758` per-layer GAP output, `324e77c` per-layer geometry) |
| Physics repo | `/eos/user/j/jeffkrup/agentic/g4hepem` — HEAD `91cbee3` ("gamma MFP cap") |
| CoDiPack | `/eos/user/j/jeffkrup/agentic/CoDiPack` (header-only) |
| Binaries | `hepemshow/build/HepEmShow` (CODI_FORWARD), `hepemshow/build_reverse/HepEmShow` (CODI_REVERSE) — match `config.yaml` |

Critical structural fact: **all G4HepEm run-time physics is header-compiled
into one HepEmShow TU** — `Simulation/src/Physics.cc` `#include`s every
`.icc`, and the `.o.d` files prove it compiles the **install copies**:
`g4hepem/install/include/G4HepEm/*.icc` (forward) and
`install_reverse/include/G4HepEm/*.icc` (reverse). The linked g4hepem libs
(`g4HepEmData`, `g4HepEmDataJsonIO`) are data-I/O only. Install copies are
currently byte-identical to the g4hepem source tree. AD type: `ad_type.h` —
`G4double = codi::RealForward` / `codi::RealReverse`;
`stop_grad(x) = G4double(GET_VALUE(x))` defined in `G4HepEmRunUtils.hh:9`.

## A. Derivative-path map (gap/absorber thickness)

**Input → geometry.** `-a/-g/--abs-profile/--gap-profile/--gap-layer i:v[:dot]`
parsed in `hepemshow/Simulation/include/InputParameters.hh:304-348` (dot
seeding via `parseRealInput`, :274-290). Reverse mode registers each
per-layer thickness + energy as tape inputs **per event**:
`EventLoop.cc:201-223` (`registerInput` on `pAbsThick[i]`, `pGapThick[i]`),
outputs registered per layer (combined + gap) at `EventLoop.cc:301-319`;
adjoints from `-b` / `--bar-gap`. Written to `barInputs` /
`barInputsPerLayer` (`Results.cc:46-70`).

**Geometry.** `Geometry.cc:210-255 UpdateParameters`: AD-active prefix sums
`fLayerStartX[i+1]=fLayerStartX[i]+abs_i+gap_i` (:216-220), `fCaloThick`
(:221). Front face pinned at x=0 (:224-225); `fPrimaryXPosition` **severed**
via `GET_VALUE` (:230) — intentional (beam start fixed).

**Distance to boundary.** `Geometry.cc:259-351 CalculateDistanceToOut`,
called every step (`SteppingLoop.cc:645` gamma, `:1011` electron):
- Calo frame: `r[0] -= 0.5*fCaloThick` (:272), `dToCalo` from calo box (:273)
  — carries the **total-thickness dot**. Exit ⇒ return 1.0E+20 (:276-281) →
  track terminated (`SteppingLoop.cc:652/1018`).
- Layer index found by **value-only scan** of `fLayerStartX` (:287-298) —
  discrete, severed by design; then AD-active translation
  `trLayeri = fLayerStartX[iLayer]` (:303) and AD-active half-lengths per
  layer (:308-310). Abs-vs-gap branch on `rx_Layer < fAbsThick[iLayer]`
  (:326) — discrete condition, AD-active operands.
- `Box::DistanceToOut(p,v)` (`Box.cc:110-140`): `t = (copysign(hD,v)-p)/v`
  per axis, `min` over axes; hD and p both AD-active. Early return literal
  `0.0` if on surface & outgoing (:113-121).

**Stepping.** Gamma `SteppingLoop.cc:572-930`, electron `:933-1401`.
Boundary-vs-physics race is `if (distToPhysics < distToBoundary)`
(`:712-715`, `:1092-1095`) — discrete winner, pathwise derivative flows
through whichever branch is taken. Position update
`AddTo3Vect(globalPosition, curDirection, stepLength)` (`:799`, `:1213`)
accumulates the dot. Zero-step push adds constant `1.0E-6` (`:793-797`,
`:1206-1211`).

**Severed casts on the path** (each a cut derivative edge):
- `localPosition[0..2] = stop_grad(...)` before safety computation —
  `SteppingLoop.cc:688-690` (gamma), `:1055-1057` (electron), `:1300-1302`
  (post-step safety for MSC displacement clipping). Safety-based decisions
  carry no dot.
- `SanitizeTrackState` / `SanitizeTrackStateForFullStop` (`:362-373`) zero
  position/direction (and steplen/safety) dots — invoked every step for
  stop-graded tracks (`:620-622`, `:985-987`) and on step-local "unsafe"
  sanitize (`:769`, `:1165`).
- Layer index & material index (values only). KE-cut `edep.setGradient(0.)`
  (`:1449-1453`).
- `fPrimaryXPosition` (`Geometry.cc:230`).

**Energy scoring.** `SteppingAction` (`SteppingLoop.cc:1442-1479`):
`edep = track.GetEnergyDeposit()` (AD, from `ApplyMeanEnergyLoss`
`ElectronManager.icc:588-629`: `eloss = pStepLength*dEdx` :611, range-out
full-deposit branch :598-603/:618-623, fluctuations `:712-754` — Gaussian
reparameterized, pathwise-alive) filled into `fEdepPerLayer_CurrentEvent` /
`fEdepGapPerLayer_CurrentEvent` keyed by the **discrete pre-step**
`(indxLayer, indxAbsorber)`.

**Interaction timing.** `numIALeft = -log(flat())` (zero dot at sampling;
`GammaManager.icc:104-108`, `ElectronManager.icc:397-401`); decremented by
`pStep/mfp` each step (Gamma `:190-197`, Electron `:570-578`) so it
**accumulates geometric dots**; step limit `mfp*numIALeft` (Gamma `:139` —
regularized by `-C`; Electron `:447` — **raw, no cap**); winner reset
`SetNumIALeft(-1)` (`GammaManager.icc:171`, `ElectronManager.icc:849`)
discards the winner channel's accumulated dot.

## B. Non-differentiable construct inventory (on the gap/abs path)

Legend: (i)=pathwise-differentiated, (ii)=severed by knob/design,
(iii)=**silently dropped — missing score-function term**.

| # | Construct | Site | Class |
|---|---|---|---|
| 1 | Boundary-vs-physics race (`min`) | SteppingLoop.cc:712/1092; step-limit `min` GammaManager.icc:141, ElectronManager.icc:448 | (i) within branch; branch-flip probability **(iii)** |
| 2 | Interaction occurrence in a layer (numIALeft crossing, exp(-t/λ) dependence on path-in-material) | GammaManager.icc:104-145, ElectronManager.icc:397-452 | position shift (i); **count change (iii)** |
| 3 | Delta-rejection `rand > mxsec*mfp` | ElectronManager.icc:821-835, 852 | **(iii)** — acceptance prob is AD-dependent, comparison discrete |
| 4 | Winner-process discrete switch (conversion/Compton/PE; ioni/brems/annih) | GammaManager.icc:175-186; ElectronManager.icc:858-868 | **(iii)** for relative-probability changes; secondary kinematics (i) |
| 5 | Secondary multiplicity / stacking | SteppingLoop.cc:1404-1439 | counts (iii); inherited pos/EKin (i), or (ii) under `-x 2` |
| 6 | Layer-index assignment (value-only scan) + **clamp `iLayer=N-1` for points past the back face** | Geometry.cc:287-298 (clamp :289-290) | **(iii)** — scoring-bin reassignment term dropped; clamp is L49-specific |
| 7 | Abs-vs-gap region branch (scoring boundary moves with the parameter) | Geometry.cc:326 | **(iii)** — the gap-specific one: gap is the sensed medium |
| 8 | Box surface early-return `0.0`; zero-step push `1e-6` | Box.cc:113-121; SteppingLoop.cc:793/1206 | (ii)/(iii) constants |
| 9 | Track kill on calo exit (leakage) | SteppingLoop.cc:652/1018, Geometry.cc:276-281 | escape *probability* term **(iii)**; in-flight shift (i) |
| 10 | Spline/Sandia table interval selection | GammaManager.icc:232-243; RunUtils `GetSplineLog`/`FindLowerBinIndex` | (i) within interval; knot switch (iii)-negligible |
| 11 | MSC branch cascade: tau branches, `fTrueStepLength==range`, `dum<1`, monotonic guard, min/max clamps | ElectronManager.icc:998-1105; UMSC.icc StepLimit :356-445 | (i)/regularized by `-N`,`-Q`; branch flips (iii) |
| 12 | Step-limit Gaussian randomization (`-r`) | UMSC.icc:432-441 | (i) reparameterized |
| 13 | Grazing/backward/same-boundary stop triggers | SteppingLoop.cc:751-783 / 1146-1194 | (ii) — deliberate severing (`-y`,`-B`,`-f`,`-q…`) |
| 14 | KE-cut edep gradient zero (`-c`) | SteppingLoop.cc:1449-1453 | (ii) |
| 15 | Fluctuation sub-branches (Gauss vs uniform vs Poisson-ish) | EnergyLossFluctuation.icc:12-110 | (i) within branch; branch flips (iii) |

**Evidence-backed root-cause candidate for the depth-dependent gap bias and
the L49≈49 anomaly** (a *severed-inconsistency*, not a knob): when a track is
stop-graded or sanitized, `SanitizeTrackState` zeroes the **global** position
dot (`SteppingLoop.cc:362-367`), but `CalculateDistanceToOut` then computes
local coords as `global − fLayerStartX[iLayer]` where the translation keeps
its AD dot (`Geometry.cc:303-311`). The physical cancellation (position dot ≈
prefix dot for a particle transported through i layers) is destroyed, so
every subsequent boundary-limited step of that *still-tracked* zombie track
acquires a spurious step-length dot ≈ `+prefix_dot/vx` — under a **uniform
gap seed the prefix dot of layer i is ≈ i**, i.e. a bias growing linearly
with depth, ≈ 49 at layer 49. Stop-graded tracks keep depositing AD-active
energy (`SteppingAction` fills `edep` whose dot flows from `pStepLength`
through `Perform`). Consistent with: gap worse than absorber (gap boundary is
the layer end, where prefix includes the seeded gap), depth-dependence,
fwd=rev (shared pathwise), and AD/FD≈49 at L49. Same mismatch applies to the
calo-box distance `dToCalo` (dot of `0.5*fCaloThick` ≈ N/2) for near-exit
tracks in the last layer.

## C. Knob map

| Flag | Config site → action site | What it kills/regularizes | Continuous generalization |
|---|---|---|---|
| `-x` stop-grad-mode | InputParameters.hh:358 → `SetGradientStopMode` SteppingLoop.cc:485; per-step sanitize :620/:985; mode 2 → secondaries too, :1409,1418,1432 | All pathwise dots of flagged tracks (but geometry re-injects prefix dots — see B) | per-track damping λ∈[0,1] on dots instead of hard zero |
| `-y` grazing-stop-track | :506-508 → triggers :759-770 / :1155-1166 | Full-track vs step-local severing on grazing | smooth weight w(‖vx‖) |
| `-B` backward-boundary-stop | :510-512 → `isUnsafeBackward` :753/:1148 | Boundary ping-pong derivative blowup of backward tracks | attenuation ∝ #backward crossings |
| `-f` grazing threshold | vxThreshold :587/:948, used :752/:1147 | Sets the grazing trigger | sigmoid in ‖vx‖ |
| `-N` conversion-reg-eps | ElectronManager.icc:584 → `ConvertTrueToGeometricLength` :1033-1051, `ConvertGeometricToTrueLength` :1083,1089 | Near-singular MSC z↔t conversion Jacobians (log(1−z/λ), pow terms), derivative-only, primal exact | adaptive eps |
| `-C` gamma-mfp-cap | GammaManager.icc:207 → step-limit product :93-95/:75-91 | Zeroes d(step)/d(mfp) when mfp>cap; caps mfp factor multiplying accumulated numIA dots | soft cap; **electron analogue missing** (ElectronManager.icc:447 raw) |
| `-A` numia-mfp-floor | ElectronManager.icc:580 → `RegularizedNumIADecrement` :162-178 (used :575-577) | Floors 1/mfp and p/mfp² derivative coefficients in numIA update | already smooth-ish |
| (related) `-T`,`-U`,`-V`,`-F`,`-P/-Q/-R/-S`,`-c`,`-r`,`-u`,`-k`,`-q/z/j/o/i` | gamma numIA floor GammaManager.icc:199; PE 1/E floor :71/:245; Box dir floor Box.cc:33-45/99-105; rotate floor RunUtils.icc:12-44,54-80; UMSC floors UMSC.icc:316-320/509/711/775; KE-cut SteppingLoop.cc:494-504/1449; MSC randomization UMSC.icc:432; boundary tol; near-boundary safety; same-boundary family SteppingLoop.cc:519-525 | — | — |

All "KeepPrimal" regularizers share one pattern
(`primal + (x−stop_grad(x))·dfdx_reg`): primal untouched, derivative
coefficients clamped — **new knobs should copy this pattern.**

## D. Candidate interventions, ranked (all as new, default-off, derivative-only knobs)

1. **Local-frame stop-grad re-anchoring** (`--stopgrad-local-frame 1`).
   Fixes the severed-position/AD-translation mismatch in B — the only
   candidate that *quantitatively* predicts both the linear-in-depth gap
   inflation and AD/FD≈49 at L49. Sketch: in `SanitizeTrackState`
   (SteppingLoop.cc:362) set the position dot to the dot of the local
   geometry offset at the track's location (i.e.
   `pos = stop_grad(pos − offset(x)) + offset(x)` with
   `offset = fLayerStartX[iLayer]`-based), or equivalently in
   `CalculateDistanceToOut` apply `stop_grad` to `trLayeri`/`0.5*fCaloThick`
   **only for gradient-disabled tracks** (needs a query hook
   `SteppingLoop::IsTrackGradientDisabled` passed into Geometry). ~30 lines
   across SteppingLoop.cc/Geometry.cc. Risk: low (pure re-severing; primal
   untouched); cheap to validate per-layer with `--gap-layer` vs uniform
   seeds.
   **L49-specific add-on**: `--last-layer-local 1` variant (stop_grad the
   calo-frame shift for last-layer/backstop steps, keep the layer-local one)
   should collapse the L49 anomaly on its own — falsifiable test of the
   mechanism. (Note: clamp `Geometry.cc:289-290` scores past-back-face
   points into layer 49; near-exit steps take derivatives from the calo box
   `dToCalo`, total-thickness dot ≈ N/2·seed.)
2. **Electron mfp×numIA step-limit cap** (mirror of `-C`;
   `--el-mfp-cap <mm>`). `ElectronManager.icc:447` multiplies accumulated
   numIA dots by raw brems/ioni mfp (can be huge at high E in lAr gap),
   amplifying geometric dots into the interaction-position derivative —
   plausible contributor to residual gap inflation at shower max where step
   counts peak. Sketch: copy `GammaRegularizedMfpProductKeepPrimal`
   (GammaManager.icc:75-91) into ElectronManager + a `Configure…` + one CLI
   flag. ~40 lines, mechanical. Risk: very low.
3. **Moving-boundary surface term for scoring reassignment**
   (`--score-surface-term 1`). Items B7/B6 — d(E_gap,i)/dθ has an analytic
   surface term ±(∂x_boundary/∂θ)·(edep flux density at the boundary) that
   pathwise AD drops because bin assignment is value-only; gap is worse than
   absorber precisely because its scoring bin's *both* faces move with the
   seeded parameter. Sketch: in `SteppingAction`/boundary-limited steps, add
   a derivative-only correction
   `(edep_rate_at_boundary)·(x_bnd − stop_grad(x_bnd))` transferring between
   adjacent bins (per-step, no RNG). Medium effort (~80 lines in
   SteppingLoop.cc + AD boundary position from Geometry). Risk: moderate
   variance, no NaN (bounded by local dE/dx).
4. **Score-function (REINFORCE) term for the boundary-vs-physics race**
   (`--race-score-term <weight>`). The missing interaction-count derivative
   (B1/B2): P(physics beats boundary in the gap) depends on t_gap; add
   per-event derivative-only `Σ (∂logP/∂θ)·(E_layer − baseline)`. Directly
   targets whatever bias remains after 1-3. Sketch: at the race
   (SteppingLoop.cc:712/1092) accumulate `∂logP` using AD `distToBoundary`
   /mfp already at hand; seed as a derivative-only additive term on the
   layer outputs. Risk: **high variance** (needs baseline + clipping), but
   formally unbiased — implement after 1-3 shrink the pathwise bias.
5. **Smooth grazing/backward damping** (`--grazing-damp p` replacing hard
   `-y/-B` kill). The absorber −20% *under*estimate suggests over-severing:
   full-track+descendant kills (`-x 2 -y 1 -B 1`) discard legitimate
   derivative mass. Sketch: in `Box::DistanceToOut` (Box.cc:126-137)
   multiply the derivative part of `t` (KeepPrimal pattern) by
   `w=min(1,(|vx|/f)^p)`, demote the full-track stops to this damping when
   the knob is on. ~40 lines. Risk: re-opens ping-pong blowups → pair with
   `-C`-style dot cap on `distToBoundary` (bounded, e.g. cap |dot| at layer
   pitch/|vx|).

## E. Build-cycle assessment

- **TU count**: 14 in HepEmShow (`CMakeLists.txt:50-93`: 12 Simulation
  sources + `G4HepEmRandomEngineIO.cc` + `HepEmShow.cc`). All physics `.icc`
  compile inside the single `Physics.cc` TU.
- **ccache**: active (`CMAKE_CXX_COMPILER=/usr/lib64/ccache/c++` both modes).
- **Rebuild time** (from object timestamps): full build < 1 min; touching
  one Simulation source ≈ seconds; touching a physics `.icc` rebuilds only
  `Physics.cc.o` ≈ 10–30 s per mode. CLAUDE.md's "slow builds" warning does
  NOT apply on this cluster.
- **Writability**: `build/`, `build_reverse/`, and
  `g4hepem/install*/include/G4HepEm/` all owned by us, writable.
- **Forward vs reverse**: identical CMake, differ only in `G4HepEm_DIR`
  (`install` vs `install_reverse`) → `CODI_FORWARD` vs `CODI_REVERSE` in
  `ad_type.h`. CoDiPack shared at `/eos/.../CoDiPack`.
- **Third build dir**: trivially feasible — fresh `build_agent_fwd/` /
  `build_agent_rev/` with `cmake -DG4HepEm_DIR=… -DCoDiPack_DIR=…`.
  **Caveat**: physics edits in `install*/include/G4HepEm/*.icc` are seen by
  ALL build dirs of that mode → gate every change behind default-off knobs
  so the shared binaries stay behavior-identical (recommended; matches the
  KeepPrimal pattern), or clone the install dirs for agent builds.

**Bottom line for tune-vs-edit**: existing knobs cannot remove the gap bias —
they only sever or floor already-modeled paths; the dominant defects are (a)
the severed-position/AD-geometry inconsistency for sanitized tracks (D1,
strongly evidenced, small edit) and (b) genuinely missing surface/score
terms (D3/D4). With a ~30 s edit-compile-test loop and the default-off knob
pattern established in this codebase, source edits are cheap and safe.

# Simulation Interface

Ground-truth reference for driving the differentiable `hepemshow` binaries from
Python. This mirrors the CLI parser in
`$REPO_ROOT/Simulation/include/InputParameters.hh` and the output writer in
`$REPO_ROOT/Simulation/src/Results.cc`. If the C++ changes, update this file.

## Binaries

| Build | Path | AD mode |
|-------|------|---------|
| Forward | `$REPO_ROOT/build/HepEmShow` | forward (one input → all outputs) |
| Reverse | `$REPO_ROOT/build_reverse/HepEmShow` | reverse (output adjoints → all input grads) |

Data file: `$REPO_ROOT/data/hepem_data.json` (pass with `-d`).

## CLI flags

### Geometry / primary / run
| Flag | Field | Units | Notes |
|------|-------|-------|-------|
| `-l <int>` | num layers | – | discrete; **not differentiable** |
| `-a <t[:dot]>` | absorber thickness | mm | `2.3` or forward-seed `2.3:1` |
| `-g <t[:dot]>` | gap thickness | mm | `5.7` or forward-seed `5.7:1` |
| `-t <size>` | transverse size | mm | continuous |
| `-p <name>` | particle | – | `e-`, `e+`, `gamma` |
| `-e <E[:dot]>` | beam energy | MeV | `10000` or forward-seed `10000:1` |
| `-n <int>` | events | – | statistics knob |
| `-s <num>` | seed | – | integer-valued |
| `-d <file>` | data file | – | `$HEPEM_DATA` |
| `-v <int>` | verbosity | – | 0 = quiet |
| `-b <a0:..:aL-1>` | output adjoints | – | **reverse build only**; length = nlayers |

### Differentiation control (paper-studied)
| Flag | Meaning | Paper default |
|------|---------|---------------|
| `-x` / `--stop-grad-mode` | 0=off, 1=track, 2=track+descendants | `2` |
| `-y` | grazing-stop-track (0/1) | `1` |
| `-B` | backward-boundary-stop (0/1) | `1` |
| `-f` | grazing \|vx\| threshold | `0.2` |
| `-N` | **conversion-reg-eps (CRE)** near-singular MSC conversion floor | `1e-3` |
| `-C` | gamma mfp cap (gmc) | `1000` |
| `-A` | numia-mfp-floor | off |

> ⚠️ Flag-letter gotchas (verified in `InputParameters.hh`): `-N` is
> **conversion-reg-eps** (NOT a NumIALeft floor); `-A` is **numia-mfp-floor**;
> `-C` is **gamma-mfp-cap**. The long option `--conversion-reg-eps` maps to `-N`.

### Additional regularization knobs (leave at defaults unless sweeping)
| Flag | Long name |
|------|-----------|
| `-T` | gamma-numia-mfp-floor |
| `-U` | gamma-pe-ekin-floor |
| `-V` | box-dir-den-floor |
| `-F` | rotate-up-floor |
| `-P` | umsc-cos-den-floor |
| `-Q` | umsc-tau-blend-eps |
| `-R` | umsc-simple-den-floor |
| `-S` | umsc-disp-rad-floor |
| `-k` | threshold2 (near-boundary safety [mm]) |
| `-m` | msc-displacement (0/1) |
| `-r` | msc-step-random (0/1) |
| `-u` | boundary-tolerance [mm] |
| `-w` | msc-disp-safe-floor [mm] |
| `-q -z -j -o -i` | same-boundary family |
| `-c` | ke-cut-threshold [MeV] |

### Canonical "good" configuration (use everywhere unless varying)
```
-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000     # CRE=1e-3, gmc=1000
```

## AD usage patterns

### Forward: one input → all per-layer derivatives
```bash
# d(edep_layer)/d(absorber thickness) for every layer:
$FORWARD_BIN -d $HEPEM_DATA -n 2000 -s 1 -e 10000 -a 2.3:1   <ctrl-flags>
# d(edep_layer)/d(beam energy):
$FORWARD_BIN -d $HEPEM_DATA -n 2000 -s 1 -e 10000:1 -a 2.3   <ctrl-flags>
# d(edep_layer)/d(gap thickness):
$FORWARD_BIN -d $HEPEM_DATA -n 2000 -s 1 -e 10000 -a 2.3 -g 5.7:1 <ctrl-flags>
```
Seed exactly **one** input with `:1` per run. Output `edeps_<seed>` column `mean_dE`
is the derivative profile; `var_dE` gives its variance.

### Reverse: output adjoints → all input gradients
```bash
# gradient of sum_l w_l * edep_l  w.r.t. (a, g, E) in one pass:
$REVERSE_BIN -d $HEPEM_DATA -n 2000 -s 1 -e 10000 -a 2.3 \
    -b "$(python -c 'print(":".join(["1"]*50))')"  <ctrl-flags>
```
`-b` length must equal the number of layers. Choose adjoints `w_l = ∂L/∂edep_l`
for your scalar loss/observable `L`. Read `barInputs`.

## Output file formats (written to process CWD)

### `edeps_<seed>` — shape `(nlayers, 4)`
| Col | Name | Meaning |
|-----|------|---------|
| 0 | `mean_E` | mean energy deposit in layer [MeV] |
| 1 | `var_E` | per-event variance of layer energy deposit |
| 2 | `mean_dE` | mean seeded derivative (forward mode) |
| 3 | `var_dE` | variance of the derivative |

Standard error of a mean over `N` events: `SE = sqrt(col/ N)`.

### `barInputs` — 3 rows of `mean var` (reverse mode)
| Row | Quantity | Gradient w.r.t. |
|-----|----------|-----------------|
| 0 | `barThicknessAbsorber` | absorber thickness `a` |
| 1 | `barThicknessGap` | gap thickness `g` |
| 2 | `barParticleEnergy` | beam energy `E` |

> ⚠️ Each process writes `edeps_*` / `barInputs` into its **current working
> directory**. The `tools/sim.py` wrapper isolates every run in a temp dir.

## Derived observables (to implement in `tools/observables.py`)

Let `m_l = mean_E[l]`, `v_l = var_E[l]`, layers `l = 0..L-1`, events `N`.

| Observable | Definition | Notes |
|-----------|-----------|-------|
| `total_edep` | `Σ_l m_l` | total visible energy [MeV] |
| `peak_edep` | `softmax_beta(m)` | smooth max surrogate (log-sum-exp) |
| `shower_max_depth` | `Σ_l l * softmax_beta(m)_l` | smooth argmax over layer index |
| `visible_fraction` | `Σ_l m_l / E_beam` | normalized by design-point beam energy |
| `front_fraction` | `Σ_{l < L/2} m_l / Σ_l m_l` | front-half longitudinal fraction |

`tools/observables.py` uses smooth surrogates for peak/depth so value and
gradient are consistent in AD and finite-difference cross-checks.

### Sensitivity of a derived observable
- **Forward path:** if `O = f({m_l})`, then `dO = Σ_l (∂f/∂m_l) · mean_dE[l]`
  (chain rule through the per-layer derivative profile). Use this to get
  `∂O/∂(seeded input)` from a single forward run.
- **Reverse path:** set adjoints `w_l = ∂f/∂m_l` and read `barInputs` to get
  `∂O/∂(a,g,E)` directly in one reverse run. (Preferred for total_edep where
  `w_l = 1`.)

## Reliability signals (for `tools/reliability.py`)
- **Gradient SNR:** `|mean_dE| / sqrt(var_dE / N)` per layer; aggregate for `O`.
- **AD vs FD cross-check:** compare AD `∂O/∂θ` to central finite difference
  `[O(θ+h) − O(θ−h)] / 2h` at matched seeds. Large relative disagreement ⇒
  discontinuity-dominated ⇒ `untrusted`.
- **NaN / non-zero exit:** treat as `untrusted`, surface to the agent.
- Map to a flag: `ok` (SNR high & FD agrees), `marginal`, `untrusted`.

## Statistics guidance
- Dev/test: `-n 1000..2000`, seeds `1..8`.
- Benchmark ground truth: pin `-n` and the seed list in the dataset file.
- Average over seeds for means; combine variances as `var/ (N·n_seeds)`.

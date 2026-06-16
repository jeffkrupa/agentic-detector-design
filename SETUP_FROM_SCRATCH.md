# Setup From Scratch — Differentiable EM Calorimeter (paper tag `v1.0-paper`)

Reproducible build + run instructions starting from a clean `git clone`, written
for an agent or collaborator who will then carry out the bilevel design study
(see `HANDOFF_HYPOTHESIS_TEST.md`, "Phase 0 onward"). No prior checkout is
assumed.

The end state is two working executables — a **forward-mode** and a
**reverse-mode** AD build of `HepEmShow` — plus a Python environment for the
analysis/agent layer.

---

## 0. What you are building (and the dependency chain)

```
CoDiPack (header-only AD library)
   └── G4HepEm   built TWICE:  forward-mode AD   and   reverse-mode AD   (Geant4 OFF)
           └── HepEmShow  built TWICE:  forward (build/)  and  reverse (build_reverse/)
```

- AD mode is selected when building **G4HepEm** (`-DCODI_FORWARD=ON` or
  `-DCODI_REVERSE=ON`). HepEmShow then **auto-inherits** the mode from the
  G4HepEm package it links against — you do not pass AD flags to HepEmShow.
- **Geant4 is NOT required.** It is only used to *generate* the physics data
  file, and that file (`hepemshow/data/hepem_data.json`, ~747 KB) is already
  committed in the repo at the paper tag. Build G4HepEm with
  `-DG4HepEm_GEANT4_BUILD=OFF`.

---

## 1. Prerequisites

- C++17 compiler (GCC ≥ 9 or Clang ≥ 10; GCC 11+ recommended)
- CMake ≥ 3.16
- `git`, `make`
- Python ≥ 3.9 with `venv` (for the analysis/agent layer in step 6)

> Cluster note (verified on this system): the default `python3` may be 3.6, which
> is too old for the Python layer. Use `python3.9` (or newer) explicitly when
> creating the venv.

---

## 2. Clone all three repositories at the pinned refs

```bash
# Choose a workspace root:
export WORK=$HOME/diffcalo            # <-- edit to taste
mkdir -p "$WORK" && cd "$WORK"

# 1) CoDiPack (AD library; header-only). Paper used ~v3.1.0.
git clone https://github.com/SciCompKL/CoDiPack.git
git -C CoDiPack checkout v3.1.0       # pin; omit to take latest 3.x

# 2) G4HepEm — differentiated fork, paper tag
git clone git@github.com:jeffkrupa/g4hepem.git
git -C g4hepem checkout v1.0-paper

# 3) HepEmShow — differentiated fork, paper tag
git clone git@github.com:jeffkrupa/hepemshow.git
git -C hepemshow checkout v1.0-paper
```

> HTTPS alternatives if you lack SSH keys:
> `https://github.com/jeffkrupa/g4hepem.git`,
> `https://github.com/jeffkrupa/hepemshow.git`.

After this you have `$WORK/{CoDiPack,g4hepem,hepemshow}`.

---

## 3. Build G4HepEm twice (forward + reverse), Geant4 OFF

The AD mode and install prefix differ between the two builds; everything else is
identical.

```bash
export CODIPACK_CMAKE="$WORK/CoDiPack/cmake"

# ---- forward-mode AD -> installs to g4hepem/install ----
cd "$WORK/g4hepem"
cmake -S . -B build \
  -DCODI_FORWARD=ON \
  -DCoDiPack_DIR="$CODIPACK_CMAKE" \
  -DG4HepEm_GEANT4_BUILD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$WORK/g4hepem/install"
cmake --build build -j --target install

# ---- reverse-mode AD -> installs to g4hepem/install_reverse ----
cmake -S . -B build_reverse \
  -DCODI_REVERSE=ON \
  -DCoDiPack_DIR="$CODIPACK_CMAKE" \
  -DG4HepEm_GEANT4_BUILD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$WORK/g4hepem/install_reverse"
cmake --build build_reverse -j --target install
```

> `CODI_FORWARD` and `CODI_REVERSE` are mutually exclusive (CMake will error if
> both are ON). That is why G4HepEm is configured/installed twice into separate
> prefixes.

Note the install library dir: CMake may use `install/lib` **or** `install/lib64`.
Check which exists — you need it for the next step:

```bash
ls -d "$WORK"/g4hepem/install*/lib*/cmake/G4HepEm
```

---

## 4. Build HepEmShow twice (forward + reverse)

Point each HepEmShow build at the matching G4HepEm install prefix. HepEmShow
auto-detects the AD mode from that package. **Use the build directory names
`build` (forward) and `build_reverse` (reverse)** — the Python layer's default
config expects exactly those paths.

```bash
# adjust lib vs lib64 to match step 3's check
export G4HEPEM_FWD="$WORK/g4hepem/install/lib64/cmake/G4HepEm"
export G4HEPEM_REV="$WORK/g4hepem/install_reverse/lib64/cmake/G4HepEm"

cd "$WORK/hepemshow"

# ---- forward-mode HepEmShow -> build/HepEmShow ----
cmake -S . -B build \
  -DG4HepEm_DIR="$G4HEPEM_FWD" \
  -DCoDiPack_DIR="$CODIPACK_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# ---- reverse-mode HepEmShow -> build_reverse/HepEmShow ----
cmake -S . -B build_reverse \
  -DG4HepEm_DIR="$G4HEPEM_REV" \
  -DCoDiPack_DIR="$CODIPACK_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_reverse -j
```

During configuration you should see a message like *"G4HepEm has been configured
with forward-mode AD … so we will build a forward-mode AD version of HepEmShow"*
(and the reverse analogue). That confirms the mode was inherited correctly.

You now have:
- `$WORK/hepemshow/build/HepEmShow`          (forward)
- `$WORK/hepemshow/build_reverse/HepEmShow`  (reverse)

---

## 5. Smoke-test the binaries directly

The physics data file ships in the repo: `$WORK/hepemshow/data/hepem_data.json`.
Run each build in a scratch dir (each process writes outputs to its CWD).

```bash
cd "$WORK/hepemshow"
mkdir -p /tmp/diffcalo_check && cd /tmp/diffcalo_check

# Forward mode: seed absorber thickness (a) with dot-value 1 -> d(edep)/da per layer
"$WORK/hepemshow/build/HepEmShow" \
  -d "$WORK/hepemshow/data/hepem_data.json" \
  -n 1000 -s 1 -e 10000 -a 2.3:1
head -3 edeps_1     # 4 cols: mean_E, meanSq_E, mean_dE/da, meanSq_dE/da

# Reverse mode: adjoint=1 on layer 1 -> gradients wrt (a, g, E) in one pass
"$WORK/hepemshow/build_reverse/HepEmShow" \
  -d "$WORK/hepemshow/data/hepem_data.json" \
  -n 1000 -s 1 -e 10000 -a 2.3 -b 1
cat barInputs       # 3 rows (mean meanSq): [d/da, d/dgap, d/dE]
```

Sanity expectations: `edeps_1` has 50 rows × 4 columns; `barInputs` has 3 rows.
A stronger correctness check (forward vs reverse agree on the same seed) is built
into the Python self-test below.

---

## 6. Python analysis/agent layer

Clone this standalone repository for the Python layer (tool wrappers, schemas,
reliability, and design docs), then set it up in a venv:

```bash
cd "$WORK"
git clone git@github.com:jeffkrupa/agentic-detector-design.git
cd "$WORK/agentic-detector-design"
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt    # numpy, pyyaml

# Point the tool layer at THIS checkout's paths:
cp config.example.yaml config.yaml
#   then edit config.yaml paths.* to:
#     repo_root:   $WORK/hepemshow
#     forward_bin: $WORK/hepemshow/build/HepEmShow
#     reverse_bin: $WORK/hepemshow/build_reverse/HepEmShow
#     hepem_data:  $WORK/hepemshow/data/hepem_data.json

# End-to-end validation: forward and reverse AD must agree to ~1e-13 on a shared seed
python -m tools.sim --selftest
```

A passing self-test prints `rel.diff` ≈ 1e-13 and `[selftest] PASS`, confirming
the whole differentiable pipeline is wired correctly.

> If you cloned to a different layout, the only things that must be correct are
> the four `paths.*` entries in `config.yaml`. Everything else is derived.

---

## 7. CLI reference (what the agent will drive)

- `-d <file>` data file; `-n <events>`; `-s <seed>`; `-p <e-|e+|gamma>`
- `-e <E[:dot]>` beam energy [MeV]; `-a <t[:dot]>` absorber [mm];
  `-g <t[:dot]>` gap [mm]; `-l <int>` #layers; `-t <size>` transverse [mm]
- Forward mode: append `:1` to exactly one of `-e/-a/-g` to seed that derivative.
- Reverse mode: `-b w0:w1:...` per-layer output adjoints (len = #layers) →
  `barInputs` rows `[d/da, d/dgap, d/dE]`.
- Differentiation-control flags (verified mapping; paper defaults):
  `-x 2` (stop-grad mode), `-y 1`, `-B 1`, `-f 0.2`, `-N 1e-3` (conversion-reg-eps),
  `-C 1000` (gamma mfp cap). Full table: `docs/SIMULATION_INTERFACE.md`.

---

## 8. Known constraint that gates the study (read before Phase 0)

The shipped geometry is **uniform**: it repeats a single
`(absorber, gap)` thickness `N` times, and only three continuous inputs
`(absorber, gap, energy)` are registered as AD inputs (see
`hepemshow/Simulation/src/Geometry.cc`, `UpdateParameters`/`CalculateLocation`).
There is **no per-region or per-layer parametrization**, so the structural design
space the bilevel study needs does not yet exist in the code.

**Therefore Phase 0 of the study is a C++ task**, not an agent task: generalize
the geometry to per-region thicknesses/materials and register those as AD inputs,
with finite-difference validation at each step. The plan, hypotheses, baselines,
and decision rules are in `HANDOFF_HYPOTHESIS_TEST.md`. Do not start the agent /
baseline work until the per-region geometry passes its FD checks.

---

## 9. Quick rebuild / clean reference

```bash
# rebuild after editing HepEmShow C++:
cmake --build "$WORK/hepemshow/build" -j
cmake --build "$WORK/hepemshow/build_reverse" -j

# rebuild after editing G4HepEm C++ (must reinstall, then rebuild HepEmShow):
cmake --build "$WORK/g4hepem/build" -j --target install
cmake --build "$WORK/g4hepem/build_reverse" -j --target install
cmake --build "$WORK/hepemshow/build" -j
cmake --build "$WORK/hepemshow/build_reverse" -j

# from-clean: delete the build*/ and install*/ dirs and re-run steps 3–4.
```

---

## 10. Troubleshooting

- **`Could NOT find CoDiPack`** → fix `-DCoDiPack_DIR=$WORK/CoDiPack/cmake`.
- **`Could NOT find G4HepEm`** → wrong `-DG4HepEm_DIR`; check `lib` vs `lib64`
  (step 3's `ls`).
- **HepEmShow built non-AD by accident** → you pointed `-DG4HepEm_DIR` at a
  non-AD G4HepEm install, or at the wrong prefix. Each HepEmShow build must point
  at the matching forward/reverse G4HepEm install prefix.
- **`Ignoring -b argument …` / `Ignoring dot value …`** → you used a reverse flag
  on the forward binary or vice-versa. `-b` is reverse-only; `:dot` seeding is
  forward-only.
- **Python `SyntaxError: future feature annotations`** → you ran with Python 3.6;
  use the `python3.9` venv from step 6.
- **CMake too old** → load a newer CMake module or `pip install cmake`.

# Sensitivity-Reasoning Benchmark

> **Question:** Can frontier LLMs predict the *sign* and *order of magnitude* of a
> physical derivative `∂O/∂θ` before seeing it — and do they know when they're
> uncertain?

This is the cheapest high-impact deliverable in the project: it reuses simulation
outputs you already generate, needs only API calls on top, and yields a single
quotable headline number. To our knowledge it is the first benchmark of LLM
reasoning about *physical sensitivities* from a differentiable simulator.

## 1. Ground truth

For each item we compute the exact AD derivative `∂O/∂θ` at a design point, with
statistical error bars and a reliability flag, using `tools/sim.py` +
`tools/observables.py` + `tools/reliability.py`.

A benchmark **item** is a triple:
```
(observable O, parameter θ, design_point P)  ->  ground_truth {sign, log10|value|, value, stderr, reliability}
```

### Grid (quick vs full)
- **Observables:** `total_edep`, `shower_max_depth`, `peak_edep`,
  `visible_fraction`, `front_fraction`.
- **Parameters:** `a` (absorber), `g` (gap), `E` (beam energy).
- **Design points:** Cartesian grid, e.g.
  `a ∈ {1.5, 2.3, 3.0}` mm, `g ∈ {4, 5.7, 8}` mm, `E ∈ {5, 10, 25}` GeV,
  `particle ∈ {e-, gamma}`. (Quick mode: 1 value each + 2 energies.)
- **Statistics:** pin `-n` (e.g. 5000) and a fixed seed list; record in the file.

Full grid ≈ `5 obs × 3 par × (3·3·3·2) points ≈ 800` items — still cheap because
each forward run yields all observables' profiles for one seeded parameter, and
the `sim` cache deduplicates.

## 2. Model task (prompt protocol)

The model is given **only** the physical setup in words — geometry description,
material stack (Pb/lAr sampling calorimeter), particle, energy, and the design
point — and is asked, **without tools**, to predict for each `(O, θ)`:
- `sign ∈ {+, −, ~0}`
- `magnitude` as `log10|∂O/∂θ|` in the stated units (a point estimate)
- `confidence ∈ [0,1]`
- one-sentence physical justification

Two conditions:
- **closed-book** (no tools): tests parametric physics knowledge / reasoning.
- **grounded** (gradient tool enabled): the model may call `sensitivity` for a
  *subset* of items (a query budget) and must generalize to held-out items —
  tests whether tool access improves reasoning and calibration.

## 3. Metrics

| Metric | Definition |
|--------|-----------|
| **Sign accuracy** | fraction of items with correct sign (with a `~0` band from stderr) |
| **Magnitude MAE (dex)** | mean abs error of `log10|value|` predictions |
| **Within-1-dex rate** | fraction predicted within ±1 order of magnitude |
| **Calibration (ECE)** | expected calibration error of `confidence` vs correctness |
| **Reliability stratification** | all metrics split by ground-truth flag (`ok` / `marginal` / `untrusted`) |
| **Closed vs grounded Δ** | improvement from enabling the gradient tool |

Headline figure: sign-accuracy and within-1-dex bar charts per observable, with
the `untrusted` slice broken out (we expect models — and naive AD — to do worst
exactly where reliability is low; that's the story).

## 4. Why the reliability split matters

The `untrusted` items are the discontinuity-dominated regimes from the
stop-grad/CRE study. Three findings would each be paper-worthy:
- Models are **miscalibrated** precisely in the `untrusted` regime (overconfident
  where physics is subtle).
- The **grounded** condition improves sign/magnitude but the agent must still
  apply the reliability policy to avoid trusting bad gradients.
- Exact AD ground truth is itself only trustworthy where the flag says so — a
  methodological point that strengthens the physics paper too.

## 5. Files

- `benchmark/make_dataset.py` — builds the grid, runs the sims via `tools/`,
  writes `benchmark/data/*.jsonl` (one ground-truth item per line, with the exact
  flags/seeds used).
- `benchmark/run_benchmark.py` — loads a dataset, queries a model
  (`--model dry-run|anthropic:...|openai:...`), scores all metrics, writes
  `benchmark/results/*.json` + plots.

### Dataset line schema (JSONL)
```json
{
  "id": "total_edep__a__a2.3_g5.7_E10000_e-",
  "observable": "total_edep",
  "wrt": "a",
  "design_point": {"a":2.3,"g":5.7,"energy":10000,"n_layers":50,"particle":"e-"},
  "ctrl_flags": {"stop_grad_mode":2,"grazing_stop_track":1,"backward_boundary_stop":1,"grazing_threshold":0.2,"conversion_reg_eps":"1e-3","gamma_mfp_cap":1000},
  "stats": {"n_events":5000, "seeds":[1,2,3,4]},
  "ground_truth": {"value": -1234.5, "stderr": 42.0, "sign":"-",
                    "log10_abs": 3.09, "reliability":"ok", "method":"reverse-AD"}
}
```

### Result schema
```json
{
  "model":"anthropic:claude-...","condition":"closed_book",
  "n_items": 1000,
  "sign_accuracy": 0.78, "mag_mae_dex": 0.62, "within_1_dex": 0.71,
  "ece": 0.13,
  "by_reliability": {"ok":{...}, "marginal":{...}, "untrusted":{...}},
  "by_observable": {"total_edep":{...}, ...}
}
```

## 6. Reproducibility
Pin model name+version, temperature (0 for scoring), seeds, event counts, and the
exact ctrl-flags in every dataset/result file. Commit the dataset; it is the
artifact reviewers will want.

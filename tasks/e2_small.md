# Task E2 (small): bilevel agentic design loop

This is the small-scale Experiment-E2 capstone task. The agent runs a *bilevel*
loop: an OUTER structural loop proposes split/merge edits to the region tiling,
and an INNER AD loop tunes the per-region absorber/gap thicknesses to drive a
derived observable toward a target.

The driver reads the YAML block below (see `agent/e2_loop.py`,
`load_e2_task`). Comments after `#` on a value line are tolerated. The
`target_value` may also be recomputed at runtime as ~0.9x the measured initial
`total_edep` (see `--retarget`), which keeps a real, non-trivial residual.

```yaml
description: Reach a target total_edep by tuning per-region absorber/gap, escalating structure as needed.
target_observable: total_edep
target_value: 8400.0          # ~90% of nominal 50-layer total_edep, gives a real residual
constraints:
  - "1.0 <= a <= 3.5"
  - "3.0 <= g <= 9.0"
n_layers: 20
init_absorber_mm: 2.3
init_gap_mm: 5.7
energy: 10000.0
particle: "e-"
max_outer: 3                  # outer structural iterations
max_inner: 6                  # inner AD iters per structure
n_events: 500
seeds: [1]
```

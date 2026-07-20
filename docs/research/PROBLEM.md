# PROBLEM

## Central question
Are there detector-optimization objectives where a **structured / non-uniform
geometry** plausibly outperforms an **optimized uniform (or smooth) baseline**
under **fixed constraints** (total length, total absorber, total gap, event
budget)?

## Why it matters
The project's thesis is a bilevel loop: an agent edits *structure* (outer,
discrete) while exact AD optimizes *continuous* params (inner). That thesis only
has value if structure actually buys something a well-tuned uniform detector
cannot. So far every structured win has collapsed under a fair control. We need
to find (or rule out) an objective with a real, robust structural advantage.

## Fixed constraints (the "fair fight" rules)
- Hold **total length, total absorber, total gap** constant across designs
  (`build_scale_profile` / `profiles_from_scale`): only the *distribution* of
  granularity varies, not the budget.
- Identical control flags on every run: `-x 2 -y 1 -B 1 -f 0.2 -N 1e-3 -C 1000`.
- Any comparison must pass a **grid-fair control** — otherwise apparent gains
  are usually front-window-depth or binning confounds.

## What counts as success
A structured design beats the best uniform/smooth baseline on a chosen objective
by a margin that (a) survives the grid-fair control and (b) is stable across
seeds / energies. Absent that, the honest result is a *negative* one worth
documenting.

## Non-goals / dead ends (do not revisit)
- Aggregate-energy objectives (total edep etc.) — optimum is uniform by
  construction.
- Trusting a tight/monotone metric without a grid-fair, value, or validated-FD
  cross-check (burned 3x: net-signal GD, depth-res quantization, proxy 10–30x).
- Rebuilding the C++ sim casually; touching the protected
  `/sdf/data/atlas/u/jkrupa/hepemshow` tree.

## Ground-truth artifacts
- Sim build: `/sdf/data/atlas/u/jkrupa/agentic/diffcalo` (use this, not the
  protected tree).
- Related ledgers: [[HYPOTHESES]], [[EXPERIMENTS]], [[DECISIONS]].

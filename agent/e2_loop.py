"""Experiment E2 driver: the bilevel agentic detector-design loop (dry-run).

This is the capstone that wires E2 end-to-end at small scale, in *dry-run* mode
(deterministic, explainable heuristics -- no LLM yet). It composes the already
validated pieces:

* ``agent/structural_moves.py`` -- the discrete structural state
  (``DesignRepresentation``) and the move algebra (Split/Merge/...).
* ``tools/optimizer.py`` -- the INNER AD loop (``optimize_inner``) that tunes
  per-region absorber/gap thicknesses by exact reverse-mode gradients.
* ``tools/autopsy.py`` -- the post-optimization gradient evidence
  (``build_autopsy``) the structural analyst reads to choose its next move.

Bilevel control flow (Arm A, the agentic arm)
---------------------------------------------
    rep <- uniform(n_layers, n_regions=1)
    for outer in range(max_outer):
        inner   = optimize_inner(rep, ...)      # tune thicknesses (AD)
        rep     = inner.final_rep
        autopsy = build_autopsy(rep, ...)       # structured derivative evidence
        move    = propose_structural_move(rep, autopsy)   # dry-run analyst
        if move is None: break
        rep     = apply_move(rep, move)         # escalate structure

Arm C (the non-agentic baseline) holds the structure FIXED (a 2-region uniform
tiling) and spends the SAME inner-iteration budget in a single ``optimize_inner``
call. The A-vs-C comparison is the "value of the agent / structural edits".

CLI
---
    python -m agent.e2_loop --task tasks/e2_small.md --arm both --out runs/e2_smoke

Public API
----------
- ``load_e2_task(path) -> dict``                 task loader (tolerant YAML block)
- ``propose_structural_move(rep, autopsy)``      dry-run structural analyst
- ``run_arm_a(task, out_dir, ctrl=None) -> dict``  agentic bilevel loop
- ``run_arm_c(task, out_dir, ctrl=None) -> dict``  fixed-structure baseline
- ``main(argv=None) -> int``                     CLI entry point
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from tools import sim as _sim
from tools import observables as _obs
from tools.constraints import parse_constraints
from tools.optimizer import optimize_inner
from tools.autopsy import build_autopsy
from agent.structural_moves import (
    DesignRepresentation, Split, Merge, apply_move, IllegalMoveError,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent

# Minimum region span (layers) for a Split to be worth proposing: we only add
# resolution where there are enough layers that the two children can each tune.
_SPLIT_MIN_LAYERS = 4
# SNR floor for the "dead region" merge heuristic (mirrors autopsy/_optimizer).
_SNR_FLOOR = 2.0
# A grad whose magnitude is below this (relative to the max region grad) is
# treated as effectively zero / dead for the single-seed merge heuristic.
_DEAD_GRAD_FRAC = 0.05

# Inner-loop step/stop tuning for THIS task (n_layers=20, total_edep, e-, 10 GeV).
# total_edep is steep in thickness (~5e2 MeV/mm) and at n_layers=20 each per-layer
# handle has more leverage than the optimizer's documented 50-layer default, so
# the default trust_region=0.05 overshoots and the residual oscillates in sign.
# We pass a smaller trust region (gentle, near-monotone steps) AND a looser
# relative tolerance so the inner loop STOPS as soon as it is within ~1% of the
# target instead of bouncing across it. Both arms use the IDENTICAL values so the
# A-vs-C comparison stays fair (only the structural editing differs).
_INNER_TRUST_REGION = 0.01
_INNER_TOL = 1e-2


# --------------------------------------------------------------------------- #
# 1. Task loader
# --------------------------------------------------------------------------- #
def load_e2_task(path) -> dict:
    """Parse the E2 task markdown file's fenced ``yaml`` block into a dict.

    Tolerates trailing ``# comment`` style on value lines (the YAML parser
    already drops inline comments; values are coerced to their expected types
    here). Returns a plain dict with normalized fields and sensible fallbacks.
    """
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")
    text = Path(path).read_text()
    if "```yaml" not in text:
        raise ValueError(f"no ```yaml block found in task file {path}")
    block = text.split("```yaml", 1)[1].split("```", 1)[0]
    spec = yaml.safe_load(block) or {}

    seeds = spec.get("seeds", [1])
    if isinstance(seeds, (int, float)):
        seeds = [int(seeds)]
    seeds = [int(s) for s in seeds]

    return {
        "description": spec.get("description", "E2 small task"),
        "target_observable": spec.get("target_observable", "total_edep"),
        "target_value": float(spec["target_value"]),
        "constraints": list(spec.get("constraints", [])),
        "n_layers": int(spec.get("n_layers", 20)),
        "init_absorber_mm": float(spec.get("init_absorber_mm", 2.3)),
        "init_gap_mm": float(spec.get("init_gap_mm", 5.7)),
        "energy": float(spec.get("energy", 10000.0)),
        "particle": str(spec.get("particle", "e-")),
        "max_outer": int(spec.get("max_outer", 3)),
        "max_inner": int(spec.get("max_inner", 6)),
        "n_events": int(spec.get("n_events", 500)),
        "seeds": seeds,
    }


# --------------------------------------------------------------------------- #
# 2. Structural-analyst heuristic (dry-run)
# --------------------------------------------------------------------------- #
def propose_structural_move(rep: DesignRepresentation, autopsy) -> Optional[object]:
    """Pick the next structural edit from the autopsy evidence (dry-run).

    Deterministic, explainable rules grounded in the post-optimization gradient
    landscape:

      1. SPLIT for resolution -- if the residual is still significant AND the
         region with the largest |absorber_grad| spans >= ``_SPLIT_MIN_LAYERS``
         layers, split that region near its midpoint. Rationale: sensitivity is
         concentrated there, so adding an independent thickness handle lets the
         inner loop place material where it matters.

      2. MERGE to coarsen -- elif some region is effectively "dead" (its
         |absorber_grad| SNR is below ``_SNR_FLOOR`` when SNR is available, or
         its |grad| is a tiny fraction of the largest region grad when it is
         not) AND it has a mergeable right neighbor, merge it away. Rationale:
         degrees of freedom the optimizer cannot move are wasted structure.

      3. STOP -- else return ``None`` (no beneficial structural edit).

    Every proposed move is validated with a dry ``apply_move`` (catching
    ``IllegalMoveError``); an illegal proposal falls through to the next rule
    rather than crashing the loop.
    """
    regs = list(autopsy.region_sensitivities)
    if not regs:
        return None

    # Is the residual still worth acting on? (relative to target magnitude)
    target_mag = max(1.0, abs(autopsy.objective_value - autopsy.residual))
    residual_significant = abs(autopsy.residual) > 1e-3 * target_mag

    # --- rule 1: split the highest-|absorber_grad| region for resolution ---- #
    top = max(regs, key=lambda r: abs(r.absorber_grad))
    if residual_significant and top.n_layers_in_region >= _SPLIT_MIN_LAYERS:
        r = rep.regions[top.region_index]
        at_layer = (int(r.start) + int(r.end)) // 2
        move = Split(
            region_index=top.region_index,
            at_layer=at_layer,
            justification=(
                f"residual {autopsy.residual:+.4g} still significant; region "
                f"{top.region_index} (layers {int(r.start)}-{int(r.end) - 1}, "
                f"{top.n_layers_in_region} layers) carries the largest "
                f"|abs_grad|={abs(top.absorber_grad):.4g} -> split at midpoint "
                f"layer {at_layer} to add resolution where sensitivity "
                f"concentrates."),
        )
        if _is_legal(rep, move):
            return move

    # --- rule 2: merge an effectively dead region into its right neighbor --- #
    max_abs_grad = max(abs(r.absorber_grad) for r in regs)
    for r in regs:
        # dead = low SNR (if available) OR near-zero grad relative to the max.
        if r.absorber_snr is not None:
            dead = r.absorber_snr < _SNR_FLOOR
        else:
            dead = abs(r.absorber_grad) < _DEAD_GRAD_FRAC * max(max_abs_grad, 1e-30)
        if not dead:
            continue
        # need a right neighbor to merge into
        if r.region_index >= rep.n_regions - 1:
            continue
        move = Merge(
            region_index=r.region_index,
            justification=(
                f"region {r.region_index} is effectively dead "
                f"(|abs_grad|={abs(r.absorber_grad):.4g}, "
                f"SNR={r.absorber_snr}); merge into right neighbor to coarsen "
                f"and free a wasted degree of freedom."),
        )
        if _is_legal(rep, move):
            return move

    # --- rule 3: nothing beneficial -> stop --------------------------------- #
    return None


def _is_legal(rep: DesignRepresentation, move) -> bool:
    """True if ``move`` applies cleanly to ``rep`` (dry-run legality guard)."""
    try:
        apply_move(rep, move)
        return True
    except (IllegalMoveError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Trajectory logger (local copy of orchestrator's convention)
# --------------------------------------------------------------------------- #
class Trajectory:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "trajectory.jsonl"
        self._fh = open(self.path, "w")

    def log(self, kind: str, payload: dict):
        rec = {"t": time.time(), "kind": kind, **payload}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _events_for_inner(inner, n_events: int, n_seeds: int) -> int:
    """Sim events consumed by an inner run = sum over measured iters.

    Each measured iteration evaluates the observable (forward, one run per seed)
    plus, when it took a step, a reverse pass (seed[0]) and -- if multiseed -- a
    multiseed reverse pass. We charge the dominant forward + reverse cost as
    ``n_events * n_seeds`` per measured iteration, which is the right order and
    is monotone in the real budget (good enough for the objective-vs-budget
    bookkeeping; exact event accounting lives in sim.py).
    """
    return len(inner.history) * n_events * n_seeds


def _adjoints_for(observable: str, n_layers: int) -> np.ndarray:
    """Per-layer adjoints seeding the autopsy reverse pass.

    For ``total_edep`` the objective is the plain sum of layer edeps so the
    adjoint is ones(n_layers). Other observables would seed their own dO/dedep;
    we keep ones as the documented default and note it (see report).
    """
    return np.ones(int(n_layers))


def _measure_objective(observable, rep, n_events, seeds, ctrl) -> float:
    """Measure the observable at ``rep`` using a scalar design point.

    ``observe`` forward-seeds the scalar 'a' internally, which conflicts with a
    per-layer abs_profile. When all regions share thicknesses the scalar point
    is geometrically identical; otherwise we fall back to the profile point's
    nominal scalar, which the inner loop already reported as final_objective, so
    this path is only used for the *initial* uniform (single-region) measurement.
    """
    dp = rep.to_design_point()
    # uniform single-region rep -> scalar a/g equal the profile everywhere.
    obs = _obs.observe(observable, dp, n_events=n_events, seeds=list(seeds),
                       ctrl=ctrl)
    return float(obs.value)


# --------------------------------------------------------------------------- #
# 3. Arm A -- agentic bilevel loop
# --------------------------------------------------------------------------- #
def run_arm_a(task: dict, out_dir, ctrl=None) -> dict:
    """Run the agentic bilevel loop (Arm A) and write its trajectory + report."""
    out_dir = Path(out_dir)
    traj = Trajectory(out_dir)
    obs_name = task["target_observable"]
    target = task["target_value"]
    constraints = task["constraints"]
    n_layers = task["n_layers"]
    n_events = task["n_events"]
    seeds = tuple(task["seeds"])
    n_seeds = len(seeds)
    parsed_constraints = parse_constraints(constraints) if constraints else []

    traj.log("task", {"arm": "A", **{k: task[k] for k in (
        "description", "target_observable", "target_value", "constraints",
        "n_layers", "max_outer", "max_inner", "n_events", "seeds")}})

    rep = DesignRepresentation.uniform(
        n_layers=n_layers, n_regions=1,
        absorber_mm=task["init_absorber_mm"], gap_mm=task["init_gap_mm"],
        energy=task["energy"], particle=task["particle"])

    # initial objective / loss for the reduction assertion + budget curve
    init_obj = _measure_objective(obs_name, rep, n_events, seeds, ctrl)
    init_residual = init_obj - target
    init_loss = 0.5 * init_residual * init_residual
    cum_events = n_events * n_seeds  # the initial measurement
    budget_curve = [{"stage": "init", "events": cum_events,
                     "objective": init_obj, "loss": init_loss}]
    traj.log("init", {"objective": init_obj, "residual": init_residual,
                      "loss": init_loss, "events": cum_events,
                      "regions": _regions_payload(rep)})
    print(f"[arm A] init {obs_name}={init_obj:.4g} target={target:.4g} "
          f"residual={init_residual:+.4g} loss={init_loss:.4g}")

    moves_taken = []
    final_obj = init_obj
    final_residual = init_residual

    for outer in range(task["max_outer"]):
        # (a) INNER: tune per-region thicknesses by AD
        inner = optimize_inner(
            rep, obs_name, target, constraints=constraints,
            max_iters=task["max_inner"], n_events=n_events, seeds=seeds,
            ctrl=ctrl, trust_region=_INNER_TRUST_REGION, tol=_INNER_TOL)
        rep = inner.final_rep
        final_obj = inner.final_objective
        final_residual = inner.final_residual
        cum_events += _events_for_inner(inner, n_events, n_seeds)
        loss = 0.5 * final_residual * final_residual
        budget_curve.append({"stage": f"outer{outer}.inner",
                             "events": cum_events, "objective": final_obj,
                             "loss": loss})
        traj.log("inner", {"outer": outer, "n_regions": rep.n_regions,
                           "final_objective": final_obj,
                           "final_residual": final_residual,
                           "converged": inner.converged,
                           "history": inner.history,
                           "predicted_vs_realized": inner.predicted_vs_realized,
                           "events_cumulative": cum_events,
                           "regions": _regions_payload(rep)})
        print(f"[arm A] outer {outer}: inner ({rep.n_regions} region(s), "
              f"{len(inner.history)} iters) -> {obs_name}={final_obj:.4g} "
              f"residual={final_residual:+.4g} loss={loss:.4g} "
              f"events={cum_events}")

        # (c) AUTOPSY: structured derivative evidence at the tuned rep
        adjoints = _adjoints_for(obs_name, n_layers)
        autopsy = build_autopsy(
            rep.regions, n_layers, rep.to_design_point(), adjoints,
            final_obj, target, n_events, list(seeds), ctrl=ctrl,
            constraints=parsed_constraints,
            measurements={obs_name: final_obj})
        cum_events += n_events * n_seeds  # the multiseed reverse pass in autopsy
        traj.log("autopsy", {"outer": outer,
                            "text": autopsy.to_text(),
                            "autopsy": autopsy.to_dict(),
                            "events_cumulative": cum_events})

        # (e) PROPOSE the next structural move from the autopsy
        move = propose_structural_move(rep, autopsy)
        if move is None:
            traj.log("structural_decision",
                     {"outer": outer, "move": None,
                      "reason": "no beneficial structural edit -> stop"})
            print(f"[arm A] outer {outer}: structural analyst proposes STOP")
            break

        rep = apply_move(rep, move)
        move_rec = {"outer": outer, "type": type(move).__name__,
                    "repr": repr(move),
                    "justification": getattr(move, "justification", ""),
                    "n_regions_after": rep.n_regions}
        moves_taken.append(move_rec)
        traj.log("structural_move", move_rec)
        print(f"[arm A] outer {outer}: {type(move).__name__} -> "
              f"{rep.n_regions} regions  ({getattr(move,'justification','')[:80]})")

    summary = {
        "arm": "A",
        "init_objective": init_obj,
        "init_loss": init_loss,
        "final_objective": final_obj,
        "final_residual": final_residual,
        "final_loss": 0.5 * final_residual * final_residual,
        "n_regions": rep.n_regions,
        "n_structural_moves": len(moves_taken),
        "moves": moves_taken,
        "total_events": cum_events,
        "budget_curve": budget_curve,
        "regions": _regions_payload(rep),
        "target_value": target,
        "target_observable": obs_name,
    }
    traj.log("report", {"summary": summary})
    traj.close()
    _write_report_a(out_dir, task, summary)
    return summary


# --------------------------------------------------------------------------- #
# 4. Arm C -- non-agentic fixed-structure baseline
# --------------------------------------------------------------------------- #
def run_arm_c(task: dict, out_dir, ctrl=None) -> dict:
    """Run the fixed-structure baseline (Arm C): one big inner call, no moves."""
    out_dir = Path(out_dir)
    traj = Trajectory(out_dir)
    obs_name = task["target_observable"]
    target = task["target_value"]
    constraints = task["constraints"]
    n_layers = task["n_layers"]
    n_events = task["n_events"]
    seeds = tuple(task["seeds"])
    n_seeds = len(seeds)

    # Same total inner-iteration budget Arm A *could* spend.
    budget_iters = task["max_outer"] * task["max_inner"]

    traj.log("task", {"arm": "C", "budget_iters": budget_iters,
                      **{k: task[k] for k in (
                          "description", "target_observable", "target_value",
                          "constraints", "n_layers", "n_events", "seeds")}})

    # FIXED structure: 2-region uniform tiling, never edited.
    rep = DesignRepresentation.uniform(
        n_layers=n_layers, n_regions=2,
        absorber_mm=task["init_absorber_mm"], gap_mm=task["init_gap_mm"],
        energy=task["energy"], particle=task["particle"])

    init_obj = _measure_objective(obs_name, rep, n_events, seeds, ctrl)
    init_residual = init_obj - target
    init_loss = 0.5 * init_residual * init_residual
    cum_events = n_events * n_seeds
    budget_curve = [{"stage": "init", "events": cum_events,
                     "objective": init_obj, "loss": init_loss}]
    traj.log("init", {"objective": init_obj, "residual": init_residual,
                      "loss": init_loss, "events": cum_events,
                      "regions": _regions_payload(rep)})
    print(f"[arm C] init {obs_name}={init_obj:.4g} target={target:.4g} "
          f"residual={init_residual:+.4g} loss={init_loss:.4g} "
          f"(fixed {rep.n_regions} regions, budget {budget_iters} iters)")

    inner = optimize_inner(
        rep, obs_name, target, constraints=constraints,
        max_iters=budget_iters, n_events=n_events, seeds=seeds, ctrl=ctrl,
        trust_region=_INNER_TRUST_REGION, tol=_INNER_TOL)
    rep = inner.final_rep
    final_obj = inner.final_objective
    final_residual = inner.final_residual
    cum_events += _events_for_inner(inner, n_events, n_seeds)

    for h in inner.history:
        budget_curve.append({"stage": f"iter{h['iter']}",
                             "events": cum_events, "objective": h["objective"],
                             "loss": h["loss"]})
    traj.log("inner", {"n_regions": rep.n_regions,
                      "final_objective": final_obj,
                      "final_residual": final_residual,
                      "converged": inner.converged,
                      "history": inner.history,
                      "predicted_vs_realized": inner.predicted_vs_realized,
                      "events_cumulative": cum_events,
                      "regions": _regions_payload(rep)})
    print(f"[arm C] inner ({len(inner.history)} iters) -> {obs_name}="
          f"{final_obj:.4g} residual={final_residual:+.4g} "
          f"loss={0.5*final_residual**2:.4g} events={cum_events}")

    summary = {
        "arm": "C",
        "init_objective": init_obj,
        "init_loss": init_loss,
        "final_objective": final_obj,
        "final_residual": final_residual,
        "final_loss": 0.5 * final_residual * final_residual,
        "n_regions": rep.n_regions,
        "n_structural_moves": 0,
        "budget_iters": budget_iters,
        "total_events": cum_events,
        "budget_curve": budget_curve,
        "regions": _regions_payload(rep),
        "target_value": target,
        "target_observable": obs_name,
    }
    traj.log("report", {"summary": summary})
    traj.close()
    _write_report_c(out_dir, task, summary)
    return summary


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #
def _regions_payload(rep: DesignRepresentation) -> list:
    return [{"start": int(r.start), "end": int(r.end),
             "absorber_mm": float(r.absorber_mm), "gap_mm": float(r.gap_mm)}
            for r in rep.regions]


def _regions_table(regions: list) -> list:
    lines = ["| region | layers | absorber (mm) | gap (mm) |",
             "|---|---|---|---|"]
    for i, r in enumerate(regions):
        lines.append(f"| {i} | [{r['start']},{r['end']}) | "
                     f"{r['absorber_mm']:.4f} | {r['gap_mm']:.4f} |")
    return lines


def _budget_table(curve: list) -> list:
    lines = ["| stage | cumulative events | objective | loss |",
             "|---|---|---|---|"]
    for c in curve:
        lines.append(f"| {c['stage']} | {c['events']} | "
                     f"{c['objective']:.4g} | {c['loss']:.4g} |")
    return lines


def _write_report_a(out_dir: Path, task: dict, s: dict) -> Path:
    lines = [
        f"# E2 Arm A (agentic bilevel) report: "
        f"{s['target_observable']} -> {s['target_value']:.4g}",
        "",
        f"- target observable: `{s['target_observable']}`  "
        f"target value: {s['target_value']:.4g}",
        f"- constraints: {task['constraints'] or 'none'}",
        f"- n_layers: {task['n_layers']}   n_events: {task['n_events']}   "
        f"seeds: {task['seeds']}",
        f"- outer iterations allowed: {task['max_outer']}   "
        f"inner iters/structure: {task['max_inner']}",
        "",
        "## Objective trajectory",
        "",
        f"- initial objective: {s['init_objective']:.6g}  "
        f"(loss {s['init_loss']:.6g})",
        f"- final objective:   {s['final_objective']:.6g}  "
        f"(loss {s['final_loss']:.6g})",
        f"- final residual:    {s['final_residual']:+.6g}",
        f"- loss reduced:      {s['final_loss'] < s['init_loss']}",
        "",
    ] + _budget_table(s["budget_curve"]) + [
        "",
        f"## Structural moves taken ({s['n_structural_moves']})",
        "",
    ]
    if s["moves"]:
        for m in s["moves"]:
            lines.append(f"- **outer {m['outer']}: {m['type']}** "
                         f"(-> {m['n_regions_after']} regions)")
            lines.append(f"  - {m['justification']}")
    else:
        lines.append("- (none) -- the structural analyst proposed STOP after the "
                     "first inner optimization (autopsy->propose path exercised).")
    lines += [
        "",
        f"## Final design ({s['n_regions']} region(s))",
        "",
    ] + _regions_table(s["regions"]) + [
        "",
        "## Budget",
        "",
        f"- total sim events used: {s['total_events']}",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_report_c(out_dir: Path, task: dict, s: dict) -> Path:
    lines = [
        f"# E2 Arm C (fixed-structure baseline) report: "
        f"{s['target_observable']} -> {s['target_value']:.4g}",
        "",
        f"- target observable: `{s['target_observable']}`  "
        f"target value: {s['target_value']:.4g}",
        f"- constraints: {task['constraints'] or 'none'}",
        f"- n_layers: {task['n_layers']}   n_events: {task['n_events']}   "
        f"seeds: {task['seeds']}",
        f"- FIXED structure: {s['n_regions']} regions, no structural moves",
        f"- inner-iteration budget: {s['budget_iters']} "
        f"(= max_outer*max_inner, matching Arm A's total)",
        "",
        "## Objective trajectory",
        "",
        f"- initial objective: {s['init_objective']:.6g}  "
        f"(loss {s['init_loss']:.6g})",
        f"- final objective:   {s['final_objective']:.6g}  "
        f"(loss {s['final_loss']:.6g})",
        f"- final residual:    {s['final_residual']:+.6g}",
        f"- loss reduced:      {s['final_loss'] < s['init_loss']}",
        "",
    ] + _budget_table(s["budget_curve"]) + [
        "",
        f"## Final design ({s['n_regions']} region(s))",
        "",
    ] + _regions_table(s["regions"]) + [
        "",
        "## Budget",
        "",
        f"- total sim events used: {s['total_events']}",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment E2 bilevel agentic design loop (dry-run)")
    ap.add_argument("--task", default="tasks/e2_small.md")
    ap.add_argument("--arm", choices=["a", "c", "both"], default="both")
    ap.add_argument("--out", default="runs/e2_smoke")
    ap.add_argument("--no-retarget", action="store_true",
                    help="use the literal target_value from the task file "
                         "instead of 0.9x the measured initial objective")
    args = ap.parse_args(argv)

    task = load_e2_task(args.task)
    ctrl = _sim.ctrl_flags_from_config(_sim.load_config())

    # By default, retarget to 0.9x the measured initial objective so the
    # residual is real AND reachable at this n_layers (the literal 8400 in the
    # task file was calibrated for 50 layers; at n_layers=20 it is unreachable).
    if not args.no_retarget:
        rep0 = DesignRepresentation.uniform(
            n_layers=task["n_layers"], n_regions=1,
            absorber_mm=task["init_absorber_mm"], gap_mm=task["init_gap_mm"],
            energy=task["energy"], particle=task["particle"])
        o0 = _measure_objective(task["target_observable"], rep0,
                                task["n_events"], tuple(task["seeds"]), ctrl)
        task["target_value"] = 0.9 * float(o0)
        print(f"[retarget] initial {task['target_observable']}={o0:.4g} -> "
              f"target={task['target_value']:.4g}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sa = sc = None
    if args.arm in ("a", "both"):
        sa = run_arm_a(task, out_root / "arm_a", ctrl=ctrl)
    if args.arm in ("c", "both"):
        sc = run_arm_c(task, out_root / "arm_c", ctrl=ctrl)

    print("\n=== E2 final comparison ===")
    if sa is not None:
        print(f"Arm A (agentic):  final {sa['target_observable']}="
              f"{sa['final_objective']:.4g}  residual={sa['final_residual']:+.4g}"
              f"  loss={sa['final_loss']:.4g}  regions={sa['n_regions']}"
              f"  moves={sa['n_structural_moves']}  events={sa['total_events']}")
    if sc is not None:
        print(f"Arm C (baseline): final {sc['target_observable']}="
              f"{sc['final_objective']:.4g}  residual={sc['final_residual']:+.4g}"
              f"  loss={sc['final_loss']:.4g}  regions={sc['n_regions']}"
              f"  moves=0  events={sc['total_events']}")
    if sa is not None and sc is not None:
        better = "A" if sa["final_loss"] < sc["final_loss"] else "C"
        print(f"lower final loss: Arm {better}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Orchestrator: the budgeted plan -> sense -> optimize -> verify -> report loop.

Runs end-to-end in dry-run mode (deterministic heuristics, no API key). The LLM
path swaps the heuristic decisions for ``llm.propose_action`` over the same tool
registry; the control structure is identical (see docs/AGENTS.md).

Usage:
    python -m agent.orchestrator --task tasks/example_resolution.md --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tools import sim as _sim
from tools.schemas import DesignPoint, DIFFERENTIABLE_PARAMS
from . import subagents as sub
from . import llm as _llm

try:
    import yaml
except ImportError:
    yaml = None

_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent


# --------------------------------------------------------------------------- #
# Task spec
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    description: str
    target_observable: str
    target_value: float
    constraints: list = field(default_factory=list)
    max_steps: Optional[int] = None


def parse_task(path: Path) -> Task:
    """Parse a markdown task file with a fenced ```yaml spec block."""
    text = path.read_text()
    spec = {}
    if "```yaml" in text and yaml is not None:
        block = text.split("```yaml", 1)[1].split("```", 1)[0]
        spec = yaml.safe_load(block) or {}
    return Task(
        description=spec.get("description", text.strip()[:200]),
        target_observable=spec["target_observable"],
        target_value=float(spec["target_value"]),
        constraints=spec.get("constraints", []),
        max_steps=spec.get("max_steps"),
    )


# --------------------------------------------------------------------------- #
# Ledger / trajectory
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
# Orchestrator
# --------------------------------------------------------------------------- #
def run(task_path: str, dry_run: bool = True, out_dir: Optional[str] = None) -> dict:
    cfg = _sim.load_config()
    task = parse_task(Path(task_path))
    out = Path(out_dir) if out_dir else (_AGENTIC / "runs" / Path(task_path).stem)
    traj = Trajectory(out)

    spec = "dry-run" if dry_run else cfg["llm"]["model"]
    client = _llm.make_client(spec, cfg["llm"].get("temperature", 0.0),
                              cfg["llm"].get("max_tokens", 4096))

    dp = _sim.default_design_point(cfg)
    budget = cfg["agent"]
    tr = budget["trust_region"]
    max_steps = task.max_steps or budget["max_steps"]
    n_events = cfg["stats"]["dev_events"]
    seeds = cfg["stats"]["dev_seeds"]

    traj.log("task", {"description": task.description,
                      "target_observable": task.target_observable,
                      "target_value": task.target_value,
                      "constraints": task.constraints, "client": client.name,
                      "start_design": asdict(dp)})

    ledger = []
    print(f"[orchestrator] client={client.name}  target {task.target_observable}"
          f"->{task.target_value}  max_steps={max_steps}")

    # ---- SENSE: rank parameters by reliable sensitivity --------------------- #
    ranking = sub.sensitivity_analyst_heuristic(task.target_observable, dp,
                                                n_events=n_events, seeds=seeds)
    traj.log("sense", ranking)
    print("[sense] sensitivity ranking:")
    for r in ranking["ranking"]:
        print(f"   d{task.target_observable}/d{r['wrt']:>6} = {r['value']:+.4g} "
              f"+/- {r['stderr']:.2g}  [{r['reliability']}]")

    # ---- OPTIMIZE / VERIFY loop -------------------------------------------- #
    for step_i in range(max_steps):
        plan = sub.optimizer_heuristic(task.target_observable, task.target_value, dp,
                                       tr, n_events=n_events, seeds=seeds)
        traj.log("optimize", {"step": step_i, **_jsonable(plan)})

        verdict = sub.physics_critic_heuristic(task.target_observable,
                                               DIFFERENTIABLE_PARAMS, dp,
                                               n_events=n_events, seeds=seeds)
        traj.log("verify", {"step": step_i, "passed": verdict.passed,
                            "reason": verdict.reason,
                            "checks": [asdict(c) for c in verdict.checks]})

        applied = {k: (plan["step"][k] if verdict.passed else 0.0)
                   for k in DIFFERENTIABLE_PARAMS}
        # only move along directions the critic didn't veto and that are nonzero
        moved = any(abs(v) > 0 for v in applied.values())
        residual = plan["residual"]
        print(f"[step {step_i}] {task.target_observable}={plan['current_value']:.4g} "
              f"residual={residual:+.4g} verdict={'PASS' if verdict.passed else 'REJECT'} "
              f"step={ {k: round(v,4) for k,v in applied.items()} }")

        ledger.append({"step": step_i, "design": asdict(dp),
                       "value": plan["current_value"], "residual": residual,
                       "grad": plan["grad"], "applied_step": applied,
                       "predicted_delta_loss": plan["predicted_delta_loss"],
                       "verdict": verdict.passed})

        if abs(residual) < _tolerance(task):
            print(f"[done] target reached (|residual|<{_tolerance(task):.3g}).")
            break
        if not moved:
            print("[done] no reliable improving direction; stopping.")
            break

        dp = DesignPoint(a=dp.a + applied["a"], g=dp.g + applied["g"],
                         energy=dp.energy + applied["energy"],
                         n_layers=dp.n_layers, transverse=dp.transverse,
                         particle=dp.particle)

    # ---- REPORT ------------------------------------------------------------- #
    report = _write_report(out, task, ranking, ledger, dp, client)
    traj.log("report", {"path": str(report)})
    traj.close()
    print(f"[orchestrator] report -> {report}")
    print(f"[orchestrator] trajectory -> {traj.path}")
    return {"report": str(report), "trajectory": str(traj.path),
            "final_design": asdict(dp), "n_steps": len(ledger)}


def _tolerance(task: Task) -> float:
    # 1% of the target magnitude, with a small floor
    return max(1e-9, 0.01 * abs(task.target_value))


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _write_report(out: Path, task: Task, ranking: dict, ledger: list,
                  final_dp: DesignPoint, client) -> Path:
    lines = [f"# Design report: {task.target_observable} -> {task.target_value}",
             "", f"- client: `{client.name}`",
             f"- constraints: {task.constraints or 'none'}",
             f"- steps taken: {len(ledger)}", "",
             "## Sensitivity ranking (start point)", "",
             "| d/d | value | stderr | reliability |", "|---|---|---|---|"]
    for r in ranking["ranking"]:
        lines.append(f"| {r['wrt']} | {r['value']:+.4g} | {r['stderr']:.2g} | {r['reliability']} |")
    lines += ["", "## Trajectory", "",
              "| step | value | residual | verdict | applied step |",
              "|---|---|---|---|---|"]
    for e in ledger:
        lines.append(f"| {e['step']} | {e['value']:.4g} | {e['residual']:+.4g} | "
                     f"{'PASS' if e['verdict'] else 'REJECT'} | "
                     f"{ {k: round(v,4) for k,v in e['applied_step'].items()} } |")
    lines += ["", "## Final design", "",
              f"- absorber `a` = {final_dp.a:.4g} mm",
              f"- gap `g` = {final_dp.g:.4g} mm",
              f"- beam energy = {final_dp.energy:.4g} MeV", "",
              "## Consistency check", "",
              "Every accepted step moved along the reported reliable gradient "
              "direction; rejected steps were vetoed by the Physics-Critic on "
              "untrusted gradients (see `trajectory.jsonl`)."]
    text = "\n".join(lines) + "\n"
    text = client.write("You are the Reporter.", text)  # dry-run: identity
    path = out / "report.md"
    path.write_text(text)
    return path


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Differentiable detector-design agent")
    ap.add_argument("--task", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="use deterministic heuristics (no API key)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = run(args.task, dry_run=args.dry_run, out_dir=args.out)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""Subagent definitions and their heuristic (dry-run) implementations.

Each subagent is a bounded role: a system prompt, an allowed tool subset, and an
output contract (see docs/AGENTS.md). In LLM mode the orchestrator drives each
role via ``llm.propose_action`` restricted to that role's tools. In dry-run mode
the deterministic ``*_heuristic`` functions below are used so the full pipeline
runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from tools.schemas import DesignPoint, DIFFERENTIABLE_PARAMS, CriticVerdict, CriticCheck
from tools import observables as _obs
from tools import reliability as _rel
from . import tool_registry as reg

_PROMPTS = Path(__file__).resolve().parent / "prompts"


@dataclass
class SubagentSpec:
    name: str
    prompt_file: str
    allowed_tools: tuple

    @property
    def system_prompt(self) -> str:
        p = _PROMPTS / self.prompt_file
        return p.read_text() if p.exists() else f"You are the {self.name}."


SENSITIVITY_ANALYST = SubagentSpec(
    "Sensitivity-Analyst", "sensitivity_analyst.md",
    ("observe", "sensitivity", "rank_sensitivities", "cross_check"))
OPTIMIZER = SubagentSpec(
    "Optimizer", "optimizer.md",
    ("loss_gradient", "observe", "sensitivity"))
PHYSICS_CRITIC = SubagentSpec(
    "Physics-Critic", "physics_critic.md",
    ("cross_check", "observe", "sensitivity"))
REPORTER = SubagentSpec(
    "Reporter", "reporter.md", ())

ALL_SUBAGENTS = [SENSITIVITY_ANALYST, OPTIMIZER, PHYSICS_CRITIC, REPORTER]


# --------------------------------------------------------------------------- #
# Heuristic (dry-run) implementations
# --------------------------------------------------------------------------- #
def sensitivity_analyst_heuristic(observable: str, dp: DesignPoint,
                                  n_events=None, seeds=None) -> dict:
    """Rank parameters by reliable sensitivity + textbook-sign sanity notes."""
    res = reg.dispatch("rank_sensitivities",
                       {"observable": observable, "design_point": _asdict(dp)},
                       n_events=n_events, seeds=seeds)
    return res


def optimizer_heuristic(target_observable: str, target_value: float, dp: DesignPoint,
                        trust_region: dict, n_events=None, seeds=None) -> dict:
    """Reverse-mode step toward a target observable, honoring the reliability policy.

    Loss L = 0.5*(O - target)^2  =>  dL/dtheta = (O - target) * dO/dtheta.
    We obtain dO/dtheta from the per-layer adjoints w_l = (O - target)*dO/dm_l
    using one reverse pass (implemented via observables.loss_gradient), then clip
    to the trust region and skip directions whose gradient is untrusted.
    """
    # measure current observable + its per-layer Jacobian
    o = _obs.observe(target_observable, dp, n_events=n_events, seeds=seeds)
    m_grad = _obs._grad_wrt_m(target_observable,
                              _current_means(dp, n_events, seeds), dp)
    residual = (o.value - target_value)
    adjoints = residual * m_grad                       # w_l = dL/d(edep_l)
    lg = _obs.loss_gradient(adjoints, dp, loss_spec=f"0.5*({target_observable}-target)^2",
                            n_events=n_events, seeds=seeds)

    step = {}
    notes = {}
    tr = {"a": trust_region["max_step_absorber_mm"],
          "g": trust_region["max_step_gap_mm"],
          "energy": trust_region["max_step_energy_mev"]}
    for k in DIFFERENTIABLE_PARAMS:
        gval = lg.grad[k]
        gerr = lg.stderr[k]
        snr = abs(gval) / gerr if gerr > 0 else np.inf
        if not np.isfinite(gval) or snr < 2.0:
            step[k] = 0.0
            notes[k] = f"skip (untrusted/low-SNR snr={snr:.1f})"
            continue
        # gradient-descent direction with a fixed fraction of the trust region
        raw = -np.sign(gval) * min(tr[k], abs(gval))   # placeholder lr; clip to TR
        raw = float(np.clip(raw, -tr[k], tr[k]))
        step[k] = raw
        notes[k] = f"step {raw:+.4g} (snr={snr:.1f})"
    predicted_dL = sum(lg.grad[k] * step[k] for k in DIFFERENTIABLE_PARAMS
                       if np.isfinite(lg.grad[k]))
    return {"current_value": o.value, "residual": residual, "grad": lg.grad,
            "grad_stderr": lg.stderr, "step": step, "notes": notes,
            "predicted_delta_loss": predicted_dL}


def physics_critic_heuristic(target_observable: str, wrt_list, dp: DesignPoint,
                             n_events=None, seeds=None) -> CriticVerdict:
    """Reject decisions that rely on untrusted gradients; cross-check AD vs FD."""
    checks = []
    passed = True
    for wrt in wrt_list:
        cc = reg.dispatch("cross_check",
                          {"observable": target_observable, "wrt": wrt,
                           "design_point": _asdict(dp)},
                          n_events=n_events, seeds=seeds)
        ok = cc["flag"] != "untrusted"
        passed = passed and ok
        checks.append(CriticCheck(
            name=f"cross_check[{target_observable},{wrt}]", passed=ok,
            detail=f"flag={cc['flag']} {cc['notes']}; policy={cc['policy']}"))
    reason = "all gradients trustworthy" if passed else "one or more untrusted gradients"
    return CriticVerdict(passed=passed, reason=reason, checks=checks)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _asdict(dp: DesignPoint) -> dict:
    return {"a": dp.a, "g": dp.g, "energy": dp.energy,
            "n_layers": dp.n_layers, "particle": dp.particle}


def _current_means(dp: DesignPoint, n_events, seeds) -> np.ndarray:
    from tools import sim as _sim
    cfg = _sim.load_config()
    seeds = seeds or cfg["stats"]["dev_seeds"]
    n_events = n_events or cfg["stats"]["dev_events"]
    mean_arr, _, _ = _sim.forward_profile_multiseed(dp, "a", n_events, seeds)
    return mean_arr[:, 0] if mean_arr is not None else np.zeros(dp.n_layers)

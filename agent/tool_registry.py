"""Tool registry: JSON schemas (for LLM function-calling) + Python dispatch.

Each tool wraps the deterministic ``tools/`` layer and returns JSON-serializable
results. The same registry serves the real LLM (which sees ``schema``) and the
dry-run path (which calls ``dispatch`` directly).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import numpy as np

from tools.schemas import DesignPoint, DIFFERENTIABLE_PARAMS
from tools import sim as _sim
from tools import observables as _obs
from tools import reliability as _rel


# --------------------------------------------------------------------------- #
# JSON schemas (OpenAI/Anthropic function-calling compatible)
# --------------------------------------------------------------------------- #
_DESIGN_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "absorber thickness [mm]"},
        "g": {"type": "number", "description": "gap thickness [mm]"},
        "energy": {"type": "number", "description": "beam energy [MeV]"},
        "n_layers": {"type": "integer"},
        "particle": {"type": "string", "enum": ["e-", "e+", "gamma"]},
    },
    "required": ["a", "g", "energy"],
}

TOOL_SCHEMAS = [
    {
        "name": "observe",
        "description": "Measure a scalar observable at a design point (no derivative).",
        "parameters": {
            "type": "object",
            "properties": {
                "observable": {"type": "string", "enum": _obs.list_observables()},
                "design_point": _DESIGN_POINT_SCHEMA,
            },
            "required": ["observable", "design_point"],
        },
    },
    {
        "name": "sensitivity",
        "description": ("Exact dO/dtheta for an observable w.r.t. one design "
                        "parameter, stamped with a reliability flag (ok/marginal/"
                        "untrusted)."),
        "parameters": {
            "type": "object",
            "properties": {
                "observable": {"type": "string", "enum": _obs.list_observables()},
                "wrt": {"type": "string", "enum": list(DIFFERENTIABLE_PARAMS)},
                "design_point": _DESIGN_POINT_SCHEMA,
                "method": {"type": "string",
                           "enum": ["forward-AD", "reverse-AD", "finite-diff"],
                           "default": "forward-AD"},
            },
            "required": ["observable", "wrt", "design_point"],
        },
    },
    {
        "name": "rank_sensitivities",
        "description": ("Rank all differentiable parameters by reliable |dO/dtheta| "
                        "for one observable. Returns a sorted table with flags."),
        "parameters": {
            "type": "object",
            "properties": {
                "observable": {"type": "string", "enum": _obs.list_observables()},
                "design_point": _DESIGN_POINT_SCHEMA,
            },
            "required": ["observable"],
        },
    },
    {
        "name": "loss_gradient",
        "description": ("Reverse-mode gradient of a per-layer-weighted loss "
                        "Sum_l w_l*edep_l w.r.t. (a,g,energy) in ONE pass."),
        "parameters": {
            "type": "object",
            "properties": {
                "adjoints": {"type": "array", "items": {"type": "number"},
                             "description": "per-layer weights w_l (len=n_layers)"},
                "design_point": _DESIGN_POINT_SCHEMA,
                "loss_spec": {"type": "string"},
            },
            "required": ["adjoints", "design_point"],
        },
    },
    {
        "name": "cross_check",
        "description": "Compare AD vs finite-difference for a sensitivity; returns reliability report.",
        "parameters": {
            "type": "object",
            "properties": {
                "observable": {"type": "string", "enum": _obs.list_observables()},
                "wrt": {"type": "string", "enum": list(DIFFERENTIABLE_PARAMS)},
                "design_point": _DESIGN_POINT_SCHEMA,
            },
            "required": ["observable", "wrt", "design_point"],
        },
    },
]


def schemas_for_llm() -> list:
    return [{"schema": s} for s in TOOL_SCHEMAS]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def _dp(d: dict) -> DesignPoint:
    base = _sim.default_design_point()
    return DesignPoint(
        a=float(d.get("a", base.a)),
        g=float(d.get("g", base.g)),
        energy=float(d.get("energy", base.energy)),
        n_layers=int(d.get("n_layers", base.n_layers)),
        transverse=float(d.get("transverse", base.transverse)),
        particle=str(d.get("particle", base.particle)),
    )


def dispatch(tool: str, args: dict, *, n_events: Optional[int] = None, seeds=None) -> dict:
    """Execute a tool by name; returns a JSON-serializable result dict."""
    if tool == "observe":
        o = _obs.observe(args["observable"], _dp(args["design_point"]),
                         n_events=n_events, seeds=seeds)
        return _ser(o)

    if tool == "sensitivity":
        s = _rel.annotated_sensitivity(args["observable"], args["wrt"],
                                       _dp(args["design_point"]),
                                       n_events=n_events, seeds=seeds,
                                       method=args.get("method", "forward-AD"))
        return _ser_sens(s)

    if tool == "rank_sensitivities":
        dp = _dp(args["design_point"]) if "design_point" in args else _sim.default_design_point()
        rows = []
        for wrt in DIFFERENTIABLE_PARAMS:
            s = _rel.annotated_sensitivity(args["observable"], wrt, dp,
                                           n_events=n_events, seeds=seeds)
            rows.append(_ser_sens(s))
        # sort by reliable magnitude: untrusted entries sink to the bottom
        order = {"ok": 2, "marginal": 1, "untrusted": 0}
        rows.sort(key=lambda r: (order[r["reliability"]], abs(r["value"])), reverse=True)
        return {"observable": args["observable"], "ranking": rows}

    if tool == "loss_gradient":
        lg = _obs.loss_gradient(np.asarray(args["adjoints"], dtype=float),
                                _dp(args["design_point"]),
                                loss_spec=args.get("loss_spec", "custom"),
                                n_events=n_events, seeds=seeds)
        return {"loss_spec": lg.loss_spec, "grad": lg.grad, "stderr": lg.stderr,
                "reliability": lg.reliability, "method": lg.method}

    if tool == "cross_check":
        dp = _dp(args["design_point"])
        s = _obs.sensitivity(args["observable"], args["wrt"], dp,
                             n_events=n_events, seeds=seeds, method="forward-AD")
        rep = _rel.assess(s, dp, cross_check=True, n_events=n_events, seeds=seeds)
        return {"observable": args["observable"], "wrt": args["wrt"],
                "flag": rep.flag, "snr": rep.snr, "ad_value": rep.ad_value,
                "fd_value": rep.fd_value, "fd_rel_disagreement": rep.fd_rel_disagreement,
                "policy": _rel.recommend_policy(rep.flag), "notes": rep.notes}

    raise KeyError(f"unknown tool {tool!r}")


def _ser(o) -> dict:
    d = asdict(o)
    d["design_point"] = asdict(o.design_point) if o.design_point else None
    return d


def _ser_sens(s) -> dict:
    return {"observable": s.observable, "wrt": s.wrt, "value": s.value,
            "stderr": s.stderr, "sign": s.sign, "log10_abs": s.log10_abs,
            "reliability": s.reliability, "method": s.method}

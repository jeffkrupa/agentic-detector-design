"""Gradient-reliability assessment -- the project's differentiating mechanism.

Building on the stop-gradient / conversion-regularization (CRE) studies, we flag
*where* an exact AD gradient can be trusted. Two signals (see
docs/SIMULATION_INTERFACE.md and docs/AGENTS.md):

  1. Signal-to-noise ratio (SNR) of the sensitivity:  |value| / stderr.
  2. AD-vs-finite-difference agreement: large relative disagreement implies the
     observable is discontinuity-dominated at this design point (the regime that
     produced unregularized gradient spikes in the CRE study).

The result is a flag in {"ok", "marginal", "untrusted"} that the agent uses to
decide whether to trust AD, gather more statistics, or fall back to FD / physics
priors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .schemas import DesignPoint, CtrlFlags, Sensitivity, worst_reliability
from . import sim as _sim
from . import observables as _obs


@dataclass
class ReliabilityReport:
    flag: str                      # ok | marginal | untrusted
    snr: float
    ad_value: float
    fd_value: Optional[float]
    fd_rel_disagreement: Optional[float]
    notes: str = ""


def _snr_flag(snr: float, cfg: dict) -> str:
    r = cfg["reliability"]
    if snr >= r["snr_ok"]:
        return "ok"
    if snr >= r["snr_marginal"]:
        return "marginal"
    return "untrusted"


def assess(sens: Sensitivity, dp: DesignPoint, cross_check: bool = True,
           n_events: Optional[int] = None, seeds=None,
           ctrl: Optional[CtrlFlags] = None) -> ReliabilityReport:
    """Assess a previously-computed AD sensitivity and return a reliability flag."""
    cfg = _sim.load_config()
    r = cfg["reliability"]

    # NaN / failed run -> untrusted immediately
    if not np.isfinite(sens.value) or not np.isfinite(sens.stderr):
        return ReliabilityReport("untrusted", 0.0, sens.value, None, None,
                                 notes="non-finite AD value/stderr")

    snr = abs(sens.value) / sens.stderr if sens.stderr > 0 else np.inf
    flag = _snr_flag(snr, cfg)

    fd_val = None
    fd_rel = None
    if cross_check:
        fd = _obs.sensitivity(sens.observable, sens.wrt, dp, n_events=n_events,
                              seeds=seeds, ctrl=ctrl, method="finite-diff")
        fd_val = fd.value
        denom = max(abs(sens.value), abs(fd_val), 1e-12)
        fd_rel = abs(sens.value - fd_val) / denom
        if np.isfinite(fd_rel) and fd_rel > r["fd_rel_tol"]:
            flag = worst_reliability(flag, "untrusted")
        elif np.isfinite(fd_rel) and fd_rel > 0.5 * r["fd_rel_tol"]:
            flag = worst_reliability(flag, "marginal")

    notes = f"snr={snr:.2f}"
    if fd_rel is not None:
        notes += f", fd_rel={fd_rel:.2f}"
    return ReliabilityReport(flag, float(snr), float(sens.value),
                             None if fd_val is None else float(fd_val),
                             None if fd_rel is None else float(fd_rel), notes)


def annotated_sensitivity(observable: str, wrt: str, dp: DesignPoint,
                          n_events: Optional[int] = None, seeds=None,
                          ctrl: Optional[CtrlFlags] = None,
                          method: str = "forward-AD",
                          cross_check: bool = True) -> Sensitivity:
    """Compute a sensitivity and stamp it with a reliability flag in one call.

    This is the function the agent's Sensitivity-Analyst should call: it returns a
    ``Sensitivity`` whose ``reliability`` field reflects SNR + AD/FD agreement.
    """
    sens = _obs.sensitivity(observable, wrt, dp, n_events=n_events, seeds=seeds,
                            ctrl=ctrl, method=method)
    rep = assess(sens, dp, cross_check=cross_check, n_events=n_events, seeds=seeds, ctrl=ctrl)
    sens.reliability = rep.flag
    return sens


def recommend_policy(flag: str) -> str:
    """Map a reliability flag to the agent's recommended action (docs/AGENTS.md)."""
    return {
        "ok": "trust AD; take the gradient step",
        "marginal": "increase statistics (more events/seeds) and re-query; take a small step",
        "untrusted": "do NOT trust AD; use finite-difference or reason qualitatively, and flag the region",
    }[flag]

"""Per-layer longitudinal-profile target + loss for E2 profile shaping.

The fundamental observable here is the per-layer mean energy-deposit profile
``E = (E_0, ..., E_{N-1})`` of a detector design. Given a target profile ``T``
(length N) we minimize the mean-squared profile mismatch::

    L_profile(E) = (1/N) * sum_l (E_l - T_l)^2

Because the loss is *directly* on the per-layer profile (no scalar-observable
Jacobian in the chain), the reverse-AD adjoints are simply::

    w_l = dL/dE_l = (2/N) * (E_l - T_l)

Those adjoints are fed straight to ``run_reverse_per_layer`` to obtain
``dL/d(per-layer thickness)`` and then aggregated by ``region_gradients``.

This module provides:
  * ``natural_profile``      -- measure the per-layer mean edep of a design,
  * ``make_plateau_target``  -- build a FLAT (constant) target profile,
  * ``profile_loss``         -- the scalar MSE loss,
  * ``profile_adjoints``     -- the per-layer adjoints (2/N)(E - T).
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from . import sim as _sim
from .schemas import DesignPoint, CtrlFlags
from agent.structural_moves import DesignRepresentation


def _to_design_point(dp_or_rep) -> DesignPoint:
    """Accept either a DesignPoint or a DesignRepresentation."""
    if isinstance(dp_or_rep, DesignRepresentation):
        return dp_or_rep.to_design_point()
    return dp_or_rep


def natural_profile(
    dp_or_rep,
    n_events: int,
    seeds: Sequence[int],
    ctrl: Optional[CtrlFlags] = None,
) -> np.ndarray:
    """Per-layer mean edep profile (length n_layers) of a design.

    Uses ``forward_profile_multiseed``. Because ``observe``/sim forbid seeding
    the scalar 'a'/'g' when a per-layer profile is set, we seed 'energy' for
    profile designs (column 0 -- the value -- is identical regardless of which
    differentiable input carries the forward dot).
    """
    dp = _to_design_point(dp_or_rep)
    seed_param = "a"
    if dp.abs_profile is not None or dp.gap_profile is not None:
        seed_param = "energy"
    mean_arr, _n, _runs = _sim.forward_profile_multiseed(
        dp, seed_param, n_events, list(seeds), ctrl=ctrl)
    if mean_arr is None:
        raise RuntimeError("natural_profile: all forward seeds failed")
    return np.asarray(mean_arr[:, 0], dtype=float).copy()


def make_plateau_target(natural: np.ndarray, total_match: bool = True) -> np.ndarray:
    """Build a FLAT target profile -- constant value across all layers.

    If ``total_match`` (default True), the constant equals ``mean(natural)`` so
    the target's *total* equals the natural total. This forces REDISTRIBUTION of
    deposit (flattening the shower) rather than a trivial overall rescale.
    Otherwise the constant is the natural peak (a deeper/uniform-at-peak plateau).
    """
    nat = np.asarray(natural, dtype=float)
    N = nat.size
    if total_match:
        const = float(nat.mean())
    else:
        const = float(nat.max())
    return np.full(N, const, dtype=float)


def profile_loss(profile: np.ndarray, target: np.ndarray) -> float:
    """Mean-squared profile mismatch ``(1/N) sum (E - T)^2``."""
    E = np.asarray(profile, dtype=float)
    T = np.asarray(target, dtype=float)
    diff = E - T
    return float(np.mean(diff * diff))


def profile_adjoints(profile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-layer adjoints ``w_l = dL/dE_l = (2/N)(E_l - T_l)``."""
    E = np.asarray(profile, dtype=float)
    T = np.asarray(target, dtype=float)
    N = E.size
    return (2.0 / N) * (E - T)

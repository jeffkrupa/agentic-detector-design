"""Gap-signal (sampled-energy) objective for the structure-sensitivity E2 test.

This MIRRORS ``tools.containment_target`` but the differentiable handle is the
TOTAL SAMPLED energy ``Evis = sum_l E_gap,l`` (sum over per-layer mean GAP
energy) rather than the total combined edep. The question is whether an
objective on the SAMPLED (gap) signal -- the energy actually read out -- finally
rewards longitudinal structure, where the previous combined-energy objectives
did not.

We want the sampled energy at least ``evis_target`` while paying a per-layer
LENGTH cost on GAP thickness only (the absorber is passive material; only the
gap is the active/sampling medium being grown), so the design balances:

    L = (lam/2) * relu(evis_target - Evis)^2
        + mu * sum_r n_r * g_r

The first (gap-signal) term is differentiable via the reverse-AD ``--bar-gap``
adjoints; its per-layer adjoint is UNIFORM across layers::

    w_l = dL/d(E_gap,l) = -lam * relu(evis_target - Evis)

because ``d(Evis)/d(E_gap,l) = 1`` and ``d/d(Evis)`` of the relu^2 term is
``-lam * relu(...)``. Note ``w_l <= 0``; it seeds the GAP outputs (``--bar-gap``)
so the returned thickness gradient is NEGATIVE -> a GD step (``-lr * grad``)
pushes thickness UP, INCREASING the sampled energy toward the target.

The second (length) term is EXPLICIT in the GAP thicknesses (no simulation): for
a region with ``n_r`` layers, ``dL/d g_r = mu*n_r`` (positive -> a GD step pushes
gap thickness DOWN). The absorber carries no length cost here. Tail regions
(little sampled energy) have a small gap-signal gradient, so the length cost wins
there and their gaps go thin; front regions go thick -> a non-uniform optimum.

This module provides:
  * ``total_gap_signal``      -- measure Evis (sum of per-layer gap energy),
  * ``gap_signal_loss``       -- the scalar loss L above,
  * ``gap_signal_adjoints``   -- the uniform per-layer gap adjoint vector ``w``,
  * ``gap_length_region_grad`` -- the explicit per-region length-term gradient
                                  (on GAP thickness only).
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .schemas import DesignPoint, CtrlFlags
from . import sim as _sim


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def total_gap_signal(
    dp: DesignPoint,
    n_events: int,
    seeds: Sequence[int],
    ctrl: Optional[CtrlFlags] = None,
) -> float:
    """Return ``Evis = sum_l E_gap,l`` averaged over seeds.

    ``E_gap,l`` is the per-layer mean GAP energy (column 0 of the
    ``(n_layers, 4)`` array from :func:`tools.sim.run_forward_gap`). The forward
    seed param is irrelevant to column 0 (the value), so we seed ``a`` (or
    ``energy`` under a profile). Returns ``nan`` if every seed failed.
    """
    seed_param = "a"
    if dp.abs_profile is not None or dp.gap_profile is not None:
        seed_param = "energy"
    vals = []
    for s in seeds:
        arr = _sim.run_forward_gap(dp, seed_param, n_events, int(s), ctrl=ctrl)
        if arr is None or bool(np.isnan(arr).any()):
            continue
        vals.append(float(arr[:, 0].sum()))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def gap_signal_loss(
    evis: float,
    evis_target: float,
    lam: float,
    mu: float,
    regions,
) -> float:
    """Scalar minimum-length gap-signal loss ``L``.

    ``L = (lam/2) * relu(evis_target - evis)^2 + mu * sum_r n_r * g_r``
    (length cost on GAP thickness only).
    """
    short = _relu(float(evis_target) - float(evis))
    signal_term = 0.5 * float(lam) * short * short
    length_term = 0.0
    for r in regions:
        n_r = int(r.end) - int(r.start)
        length_term += n_r * float(r.gap_mm)
    length_term *= float(mu)
    return float(signal_term + length_term)


def gap_signal_adjoints(
    evis: float,
    evis_target: float,
    lam: float,
    n_layers: int,
) -> np.ndarray:
    """Uniform per-layer GAP adjoint vector ``w`` (length ``n_layers``).

    ``w_l = -lam * relu(evis_target - evis)`` for every layer (since
    ``d(Evis)/d(E_gap,l) = 1``). ``w_l <= 0`` -> pushes to INCREASE Evis. These
    are the ``gap_adjoints`` fed to the reverse pass (``--bar-gap``).
    """
    short = _relu(float(evis_target) - float(evis))
    w_val = -float(lam) * short
    return np.full(int(n_layers), w_val, dtype=float)


def net_signal_adjoints(n_layers: int) -> np.ndarray:
    """Uniform per-layer GAP adjoint vector for the NET-SIGNAL objective.

    The net-signal objective is ``L = -Evis + mu * sum_r n_r * g_r``; since
    ``d(-Evis)/d(E_gap,l) = -1`` for every layer, the per-layer GAP adjoints are
    ``-1`` uniformly. These seed the GAP outputs (``--bar-gap``) so the returned
    thickness gradient pushes gap thickness UP where sampling is efficient. The
    explicit length term is handled separately by ``gap_length_region_grad``.
    """
    return -1.0 * np.ones(int(n_layers), dtype=float)


def net_signal_loss(evis: float, mu: float, regions) -> float:
    """Scalar NET-SIGNAL loss ``L = -evis + mu * sum_r n_r * g_r``.

    A clean linear trade: each unit of sampled energy is worth 1, each unit of
    GAP length (summed over layers) costs ``mu``. Layers want max gap where the
    marginal sampling ``d(Evis)/d(gap_l)`` exceeds ``mu`` and min gap otherwise.
    """
    length_term = 0.0
    for r in regions:
        n_r = int(r.end) - int(r.start)
        length_term += n_r * float(r.gap_mm)
    return float(-float(evis) + float(mu) * length_term)


def gap_length_region_grad(regions, mu: float) -> dict:
    """Explicit per-region length-term gradient (GAP thickness only).

    Maps region index -> ``{"gap": mu*n_r, "absorber": 0.0}``. The gap entry is
    > 0 (pushing gap thickness DOWN under gradient descent); the absorber carries
    no length cost in this objective.
    """
    out = {}
    for ridx, r in enumerate(regions):
        n_r = int(r.end) - int(r.start)
        out[ridx] = {"gap": float(mu) * n_r, "absorber": 0.0}
    return out

"""Minimum-length containment target + loss for the structure-sensitive E2 test.

The differentiable handle here is the TOTAL energy deposit ``total_edep =
sum_l m_l`` (sum over per-layer mean combined edep). Containment is the fraction
``total_edep / E_beam``. We want containment at least ``contain_target`` while
paying a per-layer LENGTH cost for material, so the design balances:

    L = (lam/2) * relu(contain_target - total_edep/E_beam)^2
        + mu * sum_r n_r * (a_r + g_r)

The first (containment) term is differentiable via the reverse-AD ``-b``
adjoints; its per-layer adjoint is UNIFORM across layers::

    w_l = dL/d(edep_l) = -lam * relu(contain_target - total_edep/E_beam) / E_beam

because ``d(total_edep)/d(edep_l) = 1`` and ``d/d(total_edep)`` of the relu^2
term is ``-lam * relu(...) / E_beam``. Note ``w_l <= 0`` and
``d(edep_l)/d(thickness) > 0``, so the containment gradient is NEGATIVE -> a GD
step (``-lr * grad``) pushes thickness UP, reducing the containment shortfall.

The second (length) term is EXPLICIT in the thicknesses (no simulation): for a
region with ``n_r`` layers, ``dL/d a_r = mu*n_r`` and ``dL/d g_r = mu*n_r``,
both POSITIVE -> a GD step pushes thickness DOWN. Tail regions (little shower
energy) have a small containment gradient, so the length cost wins there and
they go thin; front regions go thick -> a non-uniform optimum.

This module provides:
  * ``containment``        -- measure (total_edep, frac) of a design,
  * ``containment_loss``   -- the scalar loss L above,
  * ``containment_adjoints`` -- the uniform per-layer adjoint vector ``w``,
  * ``length_region_grad`` -- the explicit per-region length-term gradient.
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .schemas import DesignPoint, CtrlFlags
from . import observables as _obs


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def containment(
    dp: DesignPoint,
    n_events: int,
    seeds: Sequence[int],
    ctrl: Optional[CtrlFlags] = None,
):
    """Return ``(total_edep, frac)`` for a design point.

    ``total_edep`` is the scalar ``observe('total_edep', ...)`` value (sum of
    per-layer mean combined edep, MeV); ``frac = total_edep / dp.energy`` is the
    containment fraction. ``observe`` is profile-safe.
    """
    obs = _obs.observe("total_edep", dp, n_events=n_events,
                       seeds=list(seeds), ctrl=ctrl)
    total_edep = float(obs.value)
    e_beam = float(dp.energy)
    frac = total_edep / e_beam if e_beam != 0.0 else float("nan")
    return total_edep, frac


def containment_loss(
    total_edep: float,
    contain_target: float,
    e_beam: float,
    lam: float,
    mu: float,
    regions,
) -> float:
    """Scalar minimum-length containment loss ``L``.

    ``L = (lam/2) * relu(contain_target - total_edep/e_beam)^2
          + mu * sum_r n_r * (a_r + g_r)``.
    """
    frac = float(total_edep) / float(e_beam)
    short = _relu(float(contain_target) - frac)
    contain_term = 0.5 * float(lam) * short * short
    length_term = 0.0
    for r in regions:
        n_r = int(r.end) - int(r.start)
        length_term += n_r * (float(r.absorber_mm) + float(r.gap_mm))
    length_term *= float(mu)
    return float(contain_term + length_term)


def containment_adjoints(
    total_edep: float,
    contain_target: float,
    e_beam: float,
    lam: float,
    n_layers: int,
) -> np.ndarray:
    """Uniform per-layer adjoint vector ``w`` (length ``n_layers``).

    ``w_l = -lam * relu(contain_target - total_edep/e_beam) / e_beam`` for every
    layer (since ``d(total_edep)/d(edep_l) = 1``). ``w_l <= 0``.
    """
    frac = float(total_edep) / float(e_beam)
    short = _relu(float(contain_target) - frac)
    w_val = -float(lam) * short / float(e_beam)
    return np.full(int(n_layers), w_val, dtype=float)


def length_region_grad(regions, mu: float) -> dict:
    """Explicit per-region length-term gradient.

    Maps region index -> ``{"absorber": mu*n_r, "gap": mu*n_r}`` (both > 0,
    pushing thickness DOWN under gradient descent).
    """
    out = {}
    for ridx, r in enumerate(regions):
        n_r = int(r.end) - int(r.start)
        g = float(mu) * n_r
        out[ridx] = {"absorber": g, "gap": g}
    return out

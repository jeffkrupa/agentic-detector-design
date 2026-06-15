"""Derived observables and their exact sensitivities.

Design (see docs/SIMULATION_INTERFACE.md, "Derived observables"):

Each observable is a pure function ``f(m, dp)`` of the per-layer mean energy
vector ``m`` (shape n_layers) and the design point ``dp``. We obtain exact
sensitivities by combining:

  * dm_l/dθ           -- exact AD, from the forward run's ``mean_dE`` column
                         (forward mode) or from a reverse pass with adjoints
                         w_l = ∂f/∂m_l (reverse mode);
  * ∂f/∂m_l           -- numerical Jacobian of f over the 50-dim vector m
                         (cheap, central differences);
  * (∂f/∂θ)_explicit  -- numerical partial of f w.r.t. θ holding m fixed
                         (nonzero only when f references dp directly, e.g. the
                         beam-energy normalization of ``visible_fraction``).

Total derivative:  dO/dθ = Σ_l (∂f/∂m_l)·(dm_l/dθ) + (∂f/∂θ)_explicit.

Smooth surrogates (``peak_edep``, ``shower_max_depth``) are used so that value
and gradient are mutually consistent; temperatures are module constants.
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .schemas import (DesignPoint, CtrlFlags, Observation, Sensitivity, Profile,
                      LossGradient, DIFFERENTIABLE_PARAMS)
from . import sim as _sim

# ----- smooth-surrogate temperatures -------------------------------------- #
_SOFTMAX_BETA = 0.5      # 1/layer; higher -> closer to true argmax/max


# ----- observable registry ------------------------------------------------ #
# Each entry: name -> callable f(m: np.ndarray, dp: DesignPoint) -> float
_OBSERVABLES: dict[str, Callable[[np.ndarray, DesignPoint], float]] = {}


def observable(name: str):
    def _reg(fn):
        _OBSERVABLES[name] = fn
        return fn
    return _reg


@observable("total_edep")
def _total_edep(m, dp):
    return float(np.sum(m))


@observable("peak_edep")
def _peak_edep(m, dp):
    # smooth max via log-sum-exp (consistent value & gradient)
    b = _SOFTMAX_BETA
    return float(np.log(np.sum(np.exp(b * m))) / b)


@observable("shower_max_depth")
def _shower_max_depth(m, dp):
    # soft-argmax over layer index (1-based), differentiable in m
    b = _SOFTMAX_BETA
    layers = np.arange(1, m.size + 1)
    w = np.exp(b * (m - m.max()))
    return float(np.sum(layers * w) / np.sum(w))


@observable("visible_fraction")
def _visible_fraction(m, dp):
    # total visible (deposited) energy divided by beam energy
    return float(np.sum(m) / dp.energy)


@observable("front_fraction")
def _front_fraction(m, dp):
    # fraction of deposit in the front half (longitudinal shape; differentiable)
    half = m.size // 2
    tot = np.sum(m)
    return float(np.sum(m[:half]) / tot) if tot > 0 else 0.0


def list_observables() -> list:
    return sorted(_OBSERVABLES)


def _fn(name: str) -> Callable[[np.ndarray, DesignPoint], float]:
    if name not in _OBSERVABLES:
        raise KeyError(f"unknown observable {name!r}; have {list_observables()}")
    return _OBSERVABLES[name]


# ----- numerical helpers --------------------------------------------------- #
def _grad_wrt_m(name: str, m: np.ndarray, dp: DesignPoint) -> np.ndarray:
    """∂f/∂m_l via central differences (relative step, abs floor)."""
    f = _fn(name)
    g = np.zeros_like(m, dtype=float)
    scale = max(1.0, float(np.max(np.abs(m))))
    h = 1e-4 * scale
    for l in range(m.size):
        mp = m.copy(); mp[l] += h
        mm = m.copy(); mm[l] -= h
        g[l] = (f(mp, dp) - f(mm, dp)) / (2 * h)
    return g


def _explicit_partial(name: str, m: np.ndarray, dp: DesignPoint, wrt: str) -> float:
    """(∂f/∂θ) holding m fixed -- captures direct dp references inside f."""
    f = _fn(name)
    base = dp.get(wrt)
    h = 1e-4 * max(1.0, abs(base))
    dp_p = dp.with_param(wrt, base + h)
    dp_m = dp.with_param(wrt, base - h)
    return (f(m, dp_p) - f(m, dp_m)) / (2 * h)


def _seeds(seeds, cfg) -> list:
    if seeds is not None:
        return list(seeds)
    return list(cfg["stats"]["dev_seeds"])


# ----- public API ---------------------------------------------------------- #
def observe(name: str, dp: DesignPoint, n_events: Optional[int] = None,
            seeds=None, ctrl: Optional[CtrlFlags] = None) -> Observation:
    """Measure a scalar observable at ``dp`` (averaged over seeds)."""
    cfg = _sim.load_config()
    n_events = n_events or int(cfg["stats"]["dev_events"])
    seeds = _seeds(seeds, cfg)
    # any forward run yields the per-layer means; seed 'a' arbitrarily.
    mean_arr, n_total, _ = _sim.forward_profile_multiseed(dp, "a", n_events, seeds, ctrl=ctrl)
    if mean_arr is None:
        return Observation(name, float("nan"), float("inf"), dp, 0)
    m = mean_arr[:, 0]
    v = mean_arr[:, 1]
    val = _fn(name)(m, dp)
    # crude value stderr via linear propagation of per-layer SE through ∂f/∂m
    gm = _grad_wrt_m(name, m, dp)
    se = float(np.sqrt(np.sum((gm ** 2) * (v / n_total))))
    return Observation(name, float(val), se, dp, n_total)


def sensitivity(name: str, wrt: str, dp: DesignPoint, n_events: Optional[int] = None,
                seeds=None, ctrl: Optional[CtrlFlags] = None,
                method: str = "forward-AD") -> Sensitivity:
    """Exact dO/dθ for a derived observable.

    method="forward-AD"  : one forward run seeding ``wrt`` (chain rule through m).
    method="reverse-AD"  : one reverse run with adjoints ∂f/∂m (implicit part)
                           plus the explicit partial.
    method="finite-diff" : central difference of ``observe`` in θ (for cross-checks).
    """
    if wrt not in DIFFERENTIABLE_PARAMS:
        raise ValueError(f"wrt must be one of {DIFFERENTIABLE_PARAMS}")
    cfg = _sim.load_config()
    n_events = n_events or int(cfg["stats"]["dev_events"])
    seeds = _seeds(seeds, cfg)

    if method == "finite-diff":
        return _sensitivity_fd(name, wrt, dp, n_events, seeds, ctrl)

    # need per-layer means to evaluate the Jacobian / explicit partial
    mean_arr, n_total, _ = _sim.forward_profile_multiseed(dp, wrt, n_events, seeds, ctrl=ctrl)
    if mean_arr is None:
        return Sensitivity(name, wrt, float("nan"), float("inf"),
                           reliability="untrusted", method=method, design_point=dp)
    m = mean_arr[:, 0]
    gm = _grad_wrt_m(name, m, dp)                 # ∂f/∂m_l
    explicit = _explicit_partial(name, m, dp, wrt)

    if method == "forward-AD":
        dm = mean_arr[:, 2]                       # dm_l/dθ (exact AD)
        var_dm = mean_arr[:, 3]
        val = float(np.dot(gm, dm) + explicit)
        se = float(np.sqrt(np.sum((gm ** 2) * (var_dm / n_total))))
        return Sensitivity(name, wrt, val, se, method="forward-AD", design_point=dp)

    if method == "reverse-AD":
        grad_d, se_d, n_tot = _sim.reverse_gradient_multiseed(dp, gm, n_events, seeds, ctrl=ctrl)
        if grad_d is None:
            return Sensitivity(name, wrt, float("nan"), float("inf"),
                               reliability="untrusted", method="reverse-AD", design_point=dp)
        val = float(grad_d[wrt] + explicit)
        return Sensitivity(name, wrt, val, float(se_d[wrt]),
                           method="reverse-AD", design_point=dp)

    raise ValueError(f"unknown method {method!r}")


def _sensitivity_fd(name, wrt, dp, n_events, seeds, ctrl) -> Sensitivity:
    cfg = _sim.load_config()
    h = float(cfg["reliability"]["fd_step"][wrt])
    base = dp.get(wrt)
    o_plus = observe(name, dp.with_param(wrt, base + h), n_events, seeds, ctrl)
    o_minus = observe(name, dp.with_param(wrt, base - h), n_events, seeds, ctrl)
    val = (o_plus.value - o_minus.value) / (2 * h)
    se = np.sqrt(o_plus.stderr ** 2 + o_minus.stderr ** 2) / (2 * h)
    return Sensitivity(name, wrt, float(val), float(se),
                       method="finite-diff", design_point=dp)


def profile(wrt: str, dp: DesignPoint, n_events: Optional[int] = None,
            seeds=None, ctrl: Optional[CtrlFlags] = None) -> Profile:
    """Per-layer forward sensitivity profile d(edep_l)/dθ."""
    cfg = _sim.load_config()
    n_events = n_events or int(cfg["stats"]["dev_events"])
    seeds = _seeds(seeds, cfg)
    mean_arr, n_total, _ = _sim.forward_profile_multiseed(dp, wrt, n_events, seeds, ctrl=ctrl)
    if mean_arr is None:
        z = np.full(dp.n_layers, np.nan)
        return Profile("edep_per_layer", wrt, z, z, reliability="untrusted", design_point=dp)
    dm = mean_arr[:, 2]
    se = np.sqrt(mean_arr[:, 3] / n_total)
    return Profile("edep_per_layer", wrt, dm, se, design_point=dp)


def loss_gradient(adjoints: np.ndarray, dp: DesignPoint, loss_spec: str = "custom",
                  n_events: Optional[int] = None, seeds=None,
                  ctrl: Optional[CtrlFlags] = None) -> LossGradient:
    """Reverse-mode gradient of Σ_l w_l·edep_l w.r.t. (a, g, E) in one pass."""
    cfg = _sim.load_config()
    n_events = n_events or int(cfg["stats"]["dev_events"])
    seeds = _seeds(seeds, cfg)
    grad_d, se_d, n_total = _sim.reverse_gradient_multiseed(dp, adjoints, n_events, seeds, ctrl=ctrl)
    if grad_d is None:
        nan = {k: float("nan") for k in DIFFERENTIABLE_PARAMS}
        return LossGradient(loss_spec, nan, nan, reliability="untrusted", design_point=dp)
    return LossGradient(loss_spec, grad_d, se_d, design_point=dp)

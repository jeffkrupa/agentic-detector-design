"""E2 inner loop: region-aware AD tuning of per-region thicknesses.

Given a fixed structural ``DesignRepresentation`` (a tiling of layers into
regions) and a scalar target on a derived observable, this module tunes the
PER-REGION absorber/gap thicknesses by exact reverse-mode AD to minimize the
squared-residual loss::

    L = 0.5 * (O - target)**2

The structure (region boundaries / count) is held FIXED; only the continuous
``absorber_mm`` / ``gap_mm`` of each region are moved. The outer (structural)
loop -- which proposes Split/Merge/SplitEqual edits -- lives elsewhere
(``agent/structural_moves.py``); this is its inner workhorse.

Gradient pipeline (exact AD, one reverse pass per iteration)
-----------------------------------------------------------
1. Measure ``O = observe(obs, dp).value`` and ``residual = O - target``.
2. Per-layer adjoints for the loss::

       w_l = dL/d(edep_l) = residual * dO/d(edep_l)

   where ``dO/d(edep_l)`` is the observables Jacobian helper ``_grad_wrt_m``.
3. One reverse pass with those adjoints (``run_reverse_per_layer``) yields the
   per-LAYER gradient of L w.r.t. every layer's absorber/gap thickness.
4. ``region_gradients`` sums member-layer gradients into per-REGION gradients
   ``dL/d(absorber_mm[region])`` and ``dL/d(gap_mm[region])``.
5. Gradient-descent step per region, clipped to a trust region, projected back
   into the parameter box implied by the constraints, written back into the
   representation via ``SetRegionThickness`` moves.

Why this is exact: the chain rule ``dL/dt = sum_l (dL/d edep_l)(d edep_l/dt)``
is realized by seeding the reverse pass with ``w_l`` (step 2) -- the simulator
returns exactly ``sum_l w_l * d edep_l/dt`` per thickness handle, so no finite
differencing of the simulator is ever needed.

Reliability (SNR) gate
----------------------
When ``len(seeds) > 1`` we additionally run the multiseed reverse helper to get
per-layer stderrs, combine member-layer stderrs in quadrature per region, and
SKIP any region/direction whose ``|grad| / stderr < snr_floor`` -- i.e. a step
we cannot tell apart from Monte-Carlo noise. With a single seed there are no
stderrs, so the gate is disabled (every direction is taken).

Step-size choice (defaults)
---------------------------
For a squared loss the region gradients scale like ``residual * dO/dedep *
d edep/dt`` and can be very large (~1e2-1e3 MeV^2/mm). A raw ``lr * grad`` step
is therefore impossible to size robustly across observables. Instead we
NORMALIZE the full per-region step vector to the trust region:

    raw_step_i  = -lr * grad_i
    if ||raw_step||_inf > trust_region:  scale all steps by
                                         trust_region / ||raw_step||_inf

so the largest single thickness change per iteration is at most
``trust_region`` mm. Defaults: ``trust_region = 0.05`` mm/step. We deliberately
keep this small because ``total_edep`` is steep in thickness (~5e2 MeV/mm), so a
0.2 mm step overshoots the target and the residual oscillates in sign; 0.05 mm
keeps the local linearization honest and gives clean, near-monotone convergence
(verified empirically -- a 10% total_edep reduction converges to tol in ~4
iterations). ``lr = 1e-3`` (the normalization caps the step anyway, so ``lr``
mainly sets the *direction-preserving* relative scale of regions whose gradient
is below the cap). ``tol = 1e-3`` (relative residual). These are documented
module defaults and may be overridden per call (raise ``trust_region`` for
shallower observables).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

from .schemas import DesignPoint, CtrlFlags
from . import sim as _sim
from . import observables as _obs
from . import constraints as _con
from agent.structural_moves import (
    DesignRepresentation, SetRegionThickness, apply_move,
)

# ----- documented defaults ------------------------------------------------- #
_DEFAULT_LR = 1e-3            # base learning rate (step is trust-region capped)
_DEFAULT_TRUST_REGION = 0.05  # mm: max single-thickness change per iteration
_DEFAULT_TOL = 1e-3          # relative-residual convergence tolerance
_DEFAULT_SNR_FLOOR = 2.0     # min |grad|/stderr to trust a direction (multiseed)


@dataclass
class InnerResult:
    """Outcome of one inner (fixed-representation) AD optimization run.

    Fields
    ------
    final_rep : DesignRepresentation
        The tuned representation (same structure, updated region thicknesses).
    history : list[dict]
        One dict per measured iteration with keys: ``iter``, ``objective``
        (the observable value O), ``residual`` (O - target), ``loss``
        (0.5*residual**2), ``region_grads`` (list of ``{"absorber","gap"}``
        per region), and ``step`` (list of ``{"absorber","gap"}`` thickness
        deltas actually applied to reach the NEXT iterate; the final entry has
        an all-zero step).
    final_objective : float
        Observable value at the returned ``final_rep`` (last measured O).
    final_residual : float
        ``final_objective - target_value``.
    converged : bool
        True if ``|residual| < tol*max(1,|target|)`` was reached.
    n_events, seeds : the measurement budget actually used.
    predicted_vs_realized : list[dict]
        For each step that was applied, the predicted change in loss from the
        first-order model (``dL ~ grad . step``) vs the realized change measured
        at the next iteration. Keys: ``iter``, ``predicted_dloss``,
        ``realized_dloss``. Used by the orchestrator's autopsy / honesty metric.
    """
    final_rep: DesignRepresentation
    history: list
    final_objective: float
    final_residual: float
    n_events: int
    seeds: tuple
    converged: bool
    predicted_vs_realized: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Constraint helpers (per-region analog of clip_to_bounds)
# --------------------------------------------------------------------------- #
def _param_box(constraints) -> dict:
    """Collect tightest (lo, hi) box bounds per parameter from constraints.

    Returns ``{param: (lo_or_None, hi_or_None)}`` for any of ``a``/``g`` that
    appear. ``constraints`` may be a list of strings, parsed Constraint
    objects, or ``None``.
    """
    if not constraints:
        return {}
    parsed = []
    for c in constraints:
        if isinstance(c, str):
            parsed.append(_con.parse_constraint(c))
        else:
            parsed.append(c)
    lo: dict = {}
    hi: dict = {}

    def tighten_lo(name, v):
        lo[name] = v if name not in lo else max(lo[name], v)

    def tighten_hi(name, v):
        hi[name] = v if name not in hi else min(hi[name], v)

    for c in parsed:
        if c.is_observable or c.subject not in ("a", "g"):
            continue
        if c.two_sided:
            tighten_lo(c.subject, c.lo)
            tighten_hi(c.subject, c.hi)
        elif c.op in ("<", "<="):
            tighten_hi(c.subject, c.rhs)
        elif c.op in (">", ">="):
            tighten_lo(c.subject, c.rhs)
        elif c.op == "==":
            tighten_lo(c.subject, c.rhs)
            tighten_hi(c.subject, c.rhs)
    box = {}
    for name in ("a", "g"):
        if name in lo or name in hi:
            box[name] = (lo.get(name), hi.get(name))
    return box


def _clip(value: float, bound) -> float:
    """Clamp ``value`` into the (lo, hi) bound (either side may be None)."""
    if bound is None:
        return value
    blo, bhi = bound
    if blo is not None:
        value = max(value, blo)
    if bhi is not None:
        value = min(value, bhi)
    return value


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def optimize_inner(
    rep: DesignRepresentation,
    target_observable: str,
    target_value: float,
    constraints=None,
    max_iters: int = 12,
    n_events: int = 1000,
    seeds: Sequence[int] = (1,),
    ctrl: Optional[CtrlFlags] = None,
    lr: float = _DEFAULT_LR,
    trust_region: float = _DEFAULT_TRUST_REGION,
    snr_floor: float = _DEFAULT_SNR_FLOOR,
    tol: float = _DEFAULT_TOL,
) -> InnerResult:
    """Tune per-region thicknesses of ``rep`` toward ``target_value``.

    See the module docstring for the gradient pipeline, the trust-region step
    rule, and the SNR gate. Returns an :class:`InnerResult`.
    """
    seeds = tuple(int(s) for s in seeds)
    box = _param_box(constraints)
    abs_bound = box.get("a")
    gap_bound = box.get("g")
    use_snr = len(seeds) > 1

    history: list = []
    pred_vs_real: list = []

    cur = rep
    conv_tol = tol * max(1.0, abs(float(target_value)))
    converged = False

    last_loss = None
    last_pred_dloss = None
    last_iter_idx = None

    for it in range(max_iters):
        dp = cur.to_design_point()
        obs = _obs.observe(target_observable, dp, n_events=n_events,
                           seeds=list(seeds), ctrl=ctrl)
        O = float(obs.value)
        residual = O - float(target_value)
        loss = 0.5 * residual * residual

        # Honesty bookkeeping: realized change vs the prediction made last step.
        if last_loss is not None:
            realized = loss - last_loss
            pred_vs_real.append({
                "iter": last_iter_idx,
                "predicted_dloss": float(last_pred_dloss),
                "realized_dloss": float(realized),
            })

        hist_entry = {
            "iter": it,
            "objective": O,
            "residual": residual,
            "loss": loss,
            "region_grads": None,
            "step": None,
        }

        # Convergence on residual -> stop (record a zero-step terminal entry).
        if abs(residual) < conv_tol:
            converged = True
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break

        # ----- per-layer adjoints w_l = dL/d(edep_l) ----------------------- #
        if np.isnan(O):
            # measurement failed; bail out, recording a zero step.
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break
        m = _value_means(dp, n_events, seeds, ctrl)
        if m is None:
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break
        dO_dm = _obs._grad_wrt_m(target_observable, m, dp)   # dO/d(edep_l)
        w = residual * dO_dm                                  # dL/d(edep_l)

        # ----- reverse pass -> per-region gradients ------------------------ #
        per_layer = _sim.run_reverse_per_layer(
            dp, adjoints=w, n_events=n_events, seed=seeds[0], ctrl=ctrl)
        if per_layer is None:
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break
        rgrad = _sim.region_gradients(cur.regions, per_layer, cur.n_layers)

        # ----- optional SNR gating (multiseed) ----------------------------- #
        region_snr_abs = None
        region_snr_gap = None
        if use_snr:
            ms = _sim.reverse_per_layer_gradients_multiseed(
                dp, adjoints=w, n_events=n_events, seeds=list(seeds), ctrl=ctrl)
            if ms is not None:
                abs_se = np.asarray(ms["absorber_stderr"], dtype=float)
                gap_se = np.asarray(ms["gap_stderr"], dtype=float)
                region_snr_abs = {}
                region_snr_gap = {}
                for ridx, r in enumerate(cur.regions):
                    s, e = int(r.start), int(r.end)
                    # member stderrs combine in quadrature (independent layers)
                    se_a = float(np.sqrt(np.sum(abs_se[s:e] ** 2)))
                    se_g = float(np.sqrt(np.sum(gap_se[s:e] ** 2)))
                    ga = rgrad[ridx]["absorber"]
                    gg = rgrad[ridx]["gap"]
                    region_snr_abs[ridx] = abs(ga) / se_a if se_a > 0 else np.inf
                    region_snr_gap[ridx] = abs(gg) / se_g if se_g > 0 else np.inf

        # ----- build raw GD step per region, gate by SNR ------------------- #
        raw_abs = np.zeros(cur.n_regions)
        raw_gap = np.zeros(cur.n_regions)
        for ridx in range(cur.n_regions):
            ga = rgrad[ridx]["absorber"]
            gg = rgrad[ridx]["gap"]
            take_a = True
            take_g = True
            if region_snr_abs is not None:
                take_a = region_snr_abs[ridx] >= snr_floor
                take_g = region_snr_gap[ridx] >= snr_floor
            raw_abs[ridx] = (-lr * ga) if take_a else 0.0
            raw_gap[ridx] = (-lr * gg) if take_g else 0.0

        # ----- trust-region normalization (inf-norm cap) ------------------- #
        max_mag = max(float(np.max(np.abs(raw_abs))) if raw_abs.size else 0.0,
                      float(np.max(np.abs(raw_gap))) if raw_gap.size else 0.0)
        if max_mag > trust_region and max_mag > 0:
            scale = trust_region / max_mag
            raw_abs = raw_abs * scale
            raw_gap = raw_gap * scale

        # No reliable improving direction -> stop.
        if max_mag == 0.0:
            hist_entry["region_grads"] = [
                {"absorber": rgrad[i]["absorber"], "gap": rgrad[i]["gap"]}
                for i in range(cur.n_regions)]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break

        # ----- apply step + project to feasibility ------------------------- #
        applied_step = []
        new_rep = cur
        grad_vec = []
        step_vec = []
        for ridx in range(cur.n_regions):
            r = cur.regions[ridx]
            new_a = _clip(r.absorber_mm + raw_abs[ridx], abs_bound)
            new_g = _clip(r.gap_mm + raw_gap[ridx], gap_bound)
            d_a = new_a - r.absorber_mm
            d_g = new_g - r.gap_mm
            applied_step.append({"absorber": d_a, "gap": d_g})
            grad_vec.append(rgrad[ridx]["absorber"])
            grad_vec.append(rgrad[ridx]["gap"])
            step_vec.append(d_a)
            step_vec.append(d_g)
            new_rep = apply_move(new_rep, SetRegionThickness(
                region_index=ridx, absorber_mm=new_a, gap_mm=new_g,
                justification="inner-AD step"))

        # first-order predicted change in loss for the honesty metric
        predicted_dloss = float(np.dot(np.array(grad_vec), np.array(step_vec)))

        hist_entry["region_grads"] = [
            {"absorber": rgrad[i]["absorber"], "gap": rgrad[i]["gap"]}
            for i in range(cur.n_regions)]
        hist_entry["step"] = applied_step
        history.append(hist_entry)

        # carry over for next iteration's realized-vs-predicted comparison
        last_loss = loss
        last_pred_dloss = predicted_dloss
        last_iter_idx = it
        cur = new_rep

    # Ensure we always have a measured final objective at the returned rep.
    if history:
        final_objective = history[-1]["objective"]
        final_residual = history[-1]["residual"]
    else:  # pragma: no cover -- max_iters==0
        dp = cur.to_design_point()
        obs = _obs.observe(target_observable, dp, n_events=n_events,
                           seeds=list(seeds), ctrl=ctrl)
        final_objective = float(obs.value)
        final_residual = final_objective - float(target_value)

    return InnerResult(
        final_rep=cur,
        history=history,
        final_objective=float(final_objective),
        final_residual=float(final_residual),
        n_events=int(n_events),
        seeds=seeds,
        converged=bool(converged),
        predicted_vs_realized=pred_vs_real,
    )


def optimize_inner_profile(
    rep: DesignRepresentation,
    target_profile,
    constraints=None,
    max_iters: int = 12,
    n_events: int = 1000,
    seeds: Sequence[int] = (1,),
    ctrl: Optional[CtrlFlags] = None,
    lr: float = _DEFAULT_LR,
    trust_region: float = 0.05,
    snr_floor: float = _DEFAULT_SNR_FLOOR,
    tol: float = 1e-4,
) -> InnerResult:
    """Tune per-region thicknesses of ``rep`` to match a per-LAYER profile.

    Mirrors :func:`optimize_inner` but the loss is the mean-squared profile
    mismatch ``L = (1/N) sum_l (E_l - T_l)^2`` directly on the per-layer mean
    edep profile ``E`` (no scalar-observable Jacobian in the chain). The
    reverse-AD adjoints are ``w_l = (2/N)(E_l - T_l)`` (see ``profile_target``);
    those seed ``run_reverse_per_layer``, and ``region_gradients`` sums member
    layers into per-region gradients. A trust-region-capped GD step moves each
    region's absorber AND gap, projected into the constraint box and written
    back via ``SetRegionThickness``.

    The per-region complexity penalty is NOT part of this inner loss (it is
    constant in the thicknesses) and is intentionally ignored here.

    Returns an :class:`InnerResult`; ``final_objective`` carries the final loss
    and ``final_residual`` is the same loss (there is no scalar target value).
    History entries store ``loss`` and mirror it into ``objective``.
    """
    from .profile_target import profile_loss, profile_adjoints

    seeds = tuple(int(s) for s in seeds)
    target = np.asarray(target_profile, dtype=float)
    box = _param_box(constraints)
    abs_bound = box.get("a")
    gap_bound = box.get("g")
    use_snr = len(seeds) > 1

    history: list = []
    pred_vs_real: list = []

    cur = rep
    converged = False

    last_loss = None
    last_pred_dloss = None
    last_iter_idx = None

    for it in range(max_iters):
        dp = cur.to_design_point()

        # ----- measure the per-layer profile E and the loss ---------------- #
        E = _value_means(dp, n_events, seeds, ctrl)
        if E is None or np.isnan(np.asarray(E)).any():
            # measurement failed; bail out recording a zero step.
            hist_entry = {
                "iter": it, "objective": float("nan"),
                "residual": float("nan"), "loss": float("nan"),
                "region_grads": [{"absorber": 0.0, "gap": 0.0} for _ in cur.regions],
                "step": [{"absorber": 0.0, "gap": 0.0} for _ in cur.regions],
            }
            history.append(hist_entry)
            break
        E = np.asarray(E, dtype=float)
        loss = profile_loss(E, target)

        # Honesty bookkeeping: realized change vs the prediction made last step.
        if last_loss is not None:
            realized = loss - last_loss
            pred_vs_real.append({
                "iter": last_iter_idx,
                "predicted_dloss": float(last_pred_dloss),
                "realized_dloss": float(realized),
            })

        hist_entry = {
            "iter": it,
            "objective": loss,      # mirror loss into objective for InnerResult
            "residual": loss,
            "loss": loss,
            "region_grads": None,
            "step": None,
        }

        # Convergence on the loss itself -> stop (zero-step terminal entry).
        if loss < tol:
            converged = True
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break

        # ----- per-layer adjoints w_l = (2/N)(E_l - T_l) ------------------- #
        w = profile_adjoints(E, target)

        # ----- reverse pass -> per-region gradients ------------------------ #
        per_layer = _sim.run_reverse_per_layer(
            dp, adjoints=w, n_events=n_events, seed=seeds[0], ctrl=ctrl)
        if per_layer is None or bool(np.isnan(per_layer).any()):
            hist_entry["region_grads"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break
        rgrad = _sim.region_gradients(cur.regions, per_layer, cur.n_layers)

        # ----- optional SNR gating (multiseed) ----------------------------- #
        region_snr_abs = None
        region_snr_gap = None
        if use_snr:
            ms = _sim.reverse_per_layer_gradients_multiseed(
                dp, adjoints=w, n_events=n_events, seeds=list(seeds), ctrl=ctrl)
            if ms is not None:
                abs_se = np.asarray(ms["absorber_stderr"], dtype=float)
                gap_se = np.asarray(ms["gap_stderr"], dtype=float)
                region_snr_abs = {}
                region_snr_gap = {}
                for ridx, r in enumerate(cur.regions):
                    s, e = int(r.start), int(r.end)
                    se_a = float(np.sqrt(np.sum(abs_se[s:e] ** 2)))
                    se_g = float(np.sqrt(np.sum(gap_se[s:e] ** 2)))
                    ga = rgrad[ridx]["absorber"]
                    gg = rgrad[ridx]["gap"]
                    region_snr_abs[ridx] = abs(ga) / se_a if se_a > 0 else np.inf
                    region_snr_gap[ridx] = abs(gg) / se_g if se_g > 0 else np.inf

        # ----- build raw GD step per region, gate by SNR ------------------- #
        raw_abs = np.zeros(cur.n_regions)
        raw_gap = np.zeros(cur.n_regions)
        for ridx in range(cur.n_regions):
            ga = rgrad[ridx]["absorber"]
            gg = rgrad[ridx]["gap"]
            take_a = True
            take_g = True
            if region_snr_abs is not None:
                take_a = region_snr_abs[ridx] >= snr_floor
                take_g = region_snr_gap[ridx] >= snr_floor
            raw_abs[ridx] = (-lr * ga) if take_a else 0.0
            raw_gap[ridx] = (-lr * gg) if take_g else 0.0

        # ----- trust-region normalization (inf-norm cap) ------------------- #
        max_mag = max(float(np.max(np.abs(raw_abs))) if raw_abs.size else 0.0,
                      float(np.max(np.abs(raw_gap))) if raw_gap.size else 0.0)
        if max_mag > trust_region and max_mag > 0:
            scale = trust_region / max_mag
            raw_abs = raw_abs * scale
            raw_gap = raw_gap * scale

        # No reliable improving direction -> stop.
        if max_mag == 0.0:
            hist_entry["region_grads"] = [
                {"absorber": rgrad[i]["absorber"], "gap": rgrad[i]["gap"]}
                for i in range(cur.n_regions)]
            hist_entry["step"] = [
                {"absorber": 0.0, "gap": 0.0} for _ in cur.regions]
            history.append(hist_entry)
            break

        # ----- apply step + project to feasibility ------------------------- #
        applied_step = []
        new_rep = cur
        grad_vec = []
        step_vec = []
        for ridx in range(cur.n_regions):
            r = cur.regions[ridx]
            new_a = _clip(r.absorber_mm + raw_abs[ridx], abs_bound)
            new_g = _clip(r.gap_mm + raw_gap[ridx], gap_bound)
            d_a = new_a - r.absorber_mm
            d_g = new_g - r.gap_mm
            applied_step.append({"absorber": d_a, "gap": d_g})
            grad_vec.append(rgrad[ridx]["absorber"])
            grad_vec.append(rgrad[ridx]["gap"])
            step_vec.append(d_a)
            step_vec.append(d_g)
            new_rep = apply_move(new_rep, SetRegionThickness(
                region_index=ridx, absorber_mm=new_a, gap_mm=new_g,
                justification="inner-AD profile step"))

        predicted_dloss = float(np.dot(np.array(grad_vec), np.array(step_vec)))

        hist_entry["region_grads"] = [
            {"absorber": rgrad[i]["absorber"], "gap": rgrad[i]["gap"]}
            for i in range(cur.n_regions)]
        hist_entry["step"] = applied_step
        history.append(hist_entry)

        last_loss = loss
        last_pred_dloss = predicted_dloss
        last_iter_idx = it
        cur = new_rep

    if history:
        final_objective = history[-1]["objective"]
        final_residual = history[-1]["residual"]
    else:  # pragma: no cover -- max_iters==0
        dp = cur.to_design_point()
        E = _value_means(dp, n_events, seeds, ctrl)
        final_objective = profile_loss(np.asarray(E, dtype=float), target)
        final_residual = final_objective

    return InnerResult(
        final_rep=cur,
        history=history,
        final_objective=float(final_objective),
        final_residual=float(final_residual),
        n_events=int(n_events),
        seeds=seeds,
        converged=bool(converged),
        predicted_vs_realized=pred_vs_real,
    )


def _value_means(dp: DesignPoint, n_events, seeds, ctrl):
    """Per-layer edep mean vector (column 0) for a (possibly profile) design.

    Profile-safe: seeds 'energy' when a profile is set (sim.py forbids seeding
    'a'/'g' under a profile), otherwise 'a'. Column 0 (the value) is identical
    regardless of which input carries the forward dot.
    """
    seed_param = "a"
    if dp.abs_profile is not None or dp.gap_profile is not None:
        seed_param = "energy"
    mean_arr, _n, _runs = _sim.forward_profile_multiseed(
        dp, seed_param, n_events, list(seeds), ctrl=ctrl)
    if mean_arr is None:
        return None
    return mean_arr[:, 0]


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("== optimizer inner-loop self-validation ==")

    rep = DesignRepresentation.uniform(
        n_layers=20, n_regions=2, absorber_mm=2.3, gap_mm=5.7,
        energy=10000, particle="e-")

    constraints = ["1.0 <= a <= 3.5", "3.0 <= g <= 9.0"]
    n_events = 500
    seeds = (1,)

    # initial observable -> target at ~90% so there is a real residual.
    dp0 = rep.to_design_point()
    O0 = _obs.observe("total_edep", dp0, n_events=n_events, seeds=list(seeds)).value
    target = 0.9 * float(O0)
    print(f"[init] total_edep={O0:.2f} MeV   target={target:.2f} MeV "
          f"(residual={O0 - target:+.2f})")

    res = optimize_inner(
        rep, "total_edep", target, constraints=constraints,
        n_events=n_events, seeds=seeds, max_iters=8)

    print("\n[history]  iter   objective(MeV)      residual          loss")
    for h in res.history:
        print(f"   {h['iter']:>3d}   {h['objective']:>14.3f}   "
              f"{h['residual']:>14.3f}   {h['loss']:>14.3f}")

    loss0 = res.history[0]["loss"]
    lossF = res.history[-1]["loss"]
    print(f"\n[loss]  iter0={loss0:.3f}  final={lossF:.3f}  "
          f"(converged={res.converged})")

    # feasibility check: every region thickness inside the box.
    in_box = True
    for r in res.final_rep.regions:
        if not (1.0 - 1e-9 <= r.absorber_mm <= 3.5 + 1e-9):
            in_box = False
        if not (3.0 - 1e-9 <= r.gap_mm <= 9.0 + 1e-9):
            in_box = False
    print(f"[box]   all region thicknesses within a[1,3.5]/g[3,9]: {in_box}")
    print("        final regions: " + ", ".join(
        f"a={r.absorber_mm:.4f},g={r.gap_mm:.4f}" for r in res.final_rep.regions))

    print("\n[predicted_vs_realized]")
    for pr in res.predicted_vs_realized:
        print(f"   iter {pr['iter']:>3d}  predicted_dloss={pr['predicted_dloss']:>14.3f}  "
              f"realized_dloss={pr['realized_dloss']:>14.3f}")

    moved = lossF < loss0
    ok = moved and in_box
    print(f"\n[assert] loss decreased (final < iter0): {moved}")
    print(f"[assert] thicknesses feasible:           {in_box}")
    print("\nRESULT: " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)

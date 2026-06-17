"""The optimization autopsy: structured derivative evidence for the agent.

Per ``HANDOFF_HYPOTHESIS_TEST.md`` (section 4), after the AD inner-loop optimizer
runs within a fixed structural representation, the agent must receive structured
derivative evidence -- not just the final score -- so it can choose the next
STRUCTURAL move (split/merge regions, regrade absorber, reallocate sampling,
coarsen readout). This module summarizes the post-optimization gradient landscape
into an :class:`OptimizationAutopsy`:

* the final objective value and its residual to target;
* per-layer absorber/gap sensitivity heatmaps;
* per-region sensitivities with signal-to-noise ratios (|grad|/stderr);
* binding / violated constraints (via :mod:`tools.constraints`);
* short heuristic notes flagging flat/stuck landscapes and where sensitivity
  concentrates.

The structural-analyst subagent reads ``to_text()`` as its per-iteration evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import DesignPoint, CtrlFlags
from . import sim as _sim
from . import constraints as _constraints

# Region SNR below this is treated as "not statistically distinguishable from a
# flat landscape" for note generation.
_SNR_FLOOR = 2.0


@dataclass
class RegionSensitivity:
    """Post-optimization sensitivity summary for one structural region.

    ``absorber_grad``/``gap_grad`` are the SUM of the member layers' per-layer
    gradient means. ``absorber_snr``/``gap_snr`` are ``|grad| / stderr`` where the
    region stderr combines member-layer stderrs in quadrature; they are ``None``
    when stderr is unavailable or zero (e.g. a single seed).
    """
    region_index: int
    absorber_grad: float
    gap_grad: float
    absorber_snr: Optional[float]
    gap_snr: Optional[float]
    n_layers_in_region: int


@dataclass
class OptimizationAutopsy:
    objective_value: float
    residual: float
    region_sensitivities: list = field(default_factory=list)   # list[RegionSensitivity]
    per_layer_abs_grad: list = field(default_factory=list)      # heatmap
    per_layer_gap_grad: list = field(default_factory=list)      # heatmap
    binding_constraints: list = field(default_factory=list)     # list[ConstraintViolation]
    notes: list = field(default_factory=list)                   # list[str]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "objective_value": self.objective_value,
            "residual": self.residual,
            "region_sensitivities": [
                {
                    "region_index": r.region_index,
                    "absorber_grad": r.absorber_grad,
                    "gap_grad": r.gap_grad,
                    "absorber_snr": r.absorber_snr,
                    "gap_snr": r.gap_snr,
                    "n_layers_in_region": r.n_layers_in_region,
                }
                for r in self.region_sensitivities
            ],
            "per_layer_abs_grad": list(self.per_layer_abs_grad),
            "per_layer_gap_grad": list(self.per_layer_gap_grad),
            "binding_constraints": [
                {
                    "raw": v.constraint.raw,
                    "subject": v.constraint.subject,
                    "value": v.value,
                    "amount": v.amount,
                }
                for v in self.binding_constraints
            ],
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        """Compact, LLM-readable summary of the gradient landscape."""
        lines = []
        lines.append("=== OPTIMIZATION AUTOPSY ===")
        lines.append(
            f"objective = {self.objective_value:.6g}   "
            f"residual = {self.residual:+.6g} "
            f"({'above' if self.residual > 0 else 'below' if self.residual < 0 else 'at'} target)")

        # top-3 regions by |absorber_grad|
        regs = sorted(self.region_sensitivities,
                      key=lambda r: abs(r.absorber_grad), reverse=True)
        lines.append("top regions by |d(obj)/d(absorber)|:")
        if not regs:
            lines.append("   (no regions)")
        for r in regs[:3]:
            snr = f"{r.absorber_snr:.2g}" if r.absorber_snr is not None else "n/a"
            sign = "+" if r.absorber_grad >= 0 else "-"
            lines.append(
                f"   region {r.region_index} ({r.n_layers_in_region} layers): "
                f"abs_grad={sign}{abs(r.absorber_grad):.4g} (SNR {snr}), "
                f"gap_grad={r.gap_grad:+.4g}")

        # high vs low sensitivity classification (by SNR)
        hi, lo = [], []
        for r in self.region_sensitivities:
            snr = r.absorber_snr
            if snr is None:
                continue
            (hi if snr >= _SNR_FLOOR else lo).append(r.region_index)
        if hi or lo:
            lines.append(f"high-sensitivity regions (SNR>={_SNR_FLOOR:g}): "
                         f"{hi if hi else 'none'}")
            lines.append(f"low-sensitivity regions  (SNR<{_SNR_FLOOR:g}): "
                         f"{lo if lo else 'none'}")

        # constraints
        if self.binding_constraints:
            lines.append("binding/violated constraints:")
            for v in self.binding_constraints:
                lines.append(f"   {v.constraint.raw!r}: value={v.value:.4g} "
                             f"amount={v.amount:+.4g}")
        else:
            lines.append("binding/violated constraints: none")

        # notes
        if self.notes:
            lines.append("notes:")
            for n in self.notes:
                lines.append(f"   - {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _snr(grad: float, stderr: Optional[float]) -> Optional[float]:
    if stderr is None:
        return None
    if not np.isfinite(stderr) or stderr <= 0.0:
        return None
    return float(abs(grad) / stderr)


def build_autopsy(representation_regions, n_layers: int, dp: DesignPoint,
                  adjoints, objective_value: float, target_value: float,
                  n_events: int, seeds, ctrl: Optional[CtrlFlags] = None,
                  constraints=None, measurements=None) -> OptimizationAutopsy:
    """Assemble the optimization autopsy for the current representation.

    Calls :func:`tools.sim.reverse_per_layer_gradients_multiseed` to obtain
    per-layer gradient means + standard errors, builds the per-layer heatmaps and
    per-region sensitivities (summing member-layer grads; combining stderrs in
    quadrature for the region SNR), computes the residual to ``target_value``,
    evaluates any constraints against ``measurements``, and emits short heuristic
    notes. Robust to ``seeds=[1]`` (stderr may be 0 -> SNR reported as ``None``).
    """
    N = int(n_layers)
    residual = float(objective_value) - float(target_value)

    multiseed = _sim.reverse_per_layer_gradients_multiseed(
        dp, adjoints, n_events, seeds, ctrl=ctrl)

    notes: list = []

    if multiseed is None:
        notes.append("reverse per-layer gradient run FAILED (no successful seed) "
                     "-> gradient landscape unavailable.")
        autopsy = OptimizationAutopsy(
            objective_value=float(objective_value), residual=residual,
            region_sensitivities=[], per_layer_abs_grad=[], per_layer_gap_grad=[],
            binding_constraints=[], notes=notes)
        _append_constraint_notes(autopsy, constraints, dp, measurements, notes)
        return autopsy

    abs_grad = np.asarray(multiseed["absorber_grad"], dtype=float)
    gap_grad = np.asarray(multiseed["gap_grad"], dtype=float)
    abs_se = np.asarray(multiseed["absorber_stderr"], dtype=float)
    gap_se = np.asarray(multiseed["gap_stderr"], dtype=float)

    region_sens = []
    for idx, r in enumerate(representation_regions):
        s, e = int(r.start), int(r.end)
        n_in = e - s
        a_g = float(abs_grad[s:e].sum())
        g_g = float(gap_grad[s:e].sum())
        # combine member-layer stderrs in quadrature
        a_se = float(np.sqrt(np.sum(abs_se[s:e] ** 2)))
        g_se = float(np.sqrt(np.sum(gap_se[s:e] ** 2)))
        region_sens.append(RegionSensitivity(
            region_index=idx,
            absorber_grad=a_g,
            gap_grad=g_g,
            absorber_snr=_snr(a_g, a_se),
            gap_snr=_snr(g_g, g_se),
            n_layers_in_region=n_in,
        ))

    autopsy = OptimizationAutopsy(
        objective_value=float(objective_value),
        residual=residual,
        region_sensitivities=region_sens,
        per_layer_abs_grad=[float(x) for x in abs_grad],
        per_layer_gap_grad=[float(x) for x in gap_grad],
        binding_constraints=[],
        notes=notes,
    )

    # --- heuristic notes ------------------------------------------------- #
    if region_sens:
        # residual sign
        if residual > 0:
            notes.append("residual sign positive -> objective above target.")
        elif residual < 0:
            notes.append("residual sign negative -> objective below target.")
        else:
            notes.append("objective at target (zero residual).")

        # highest-sensitivity region by |absorber_grad|
        top = max(region_sens, key=lambda r: abs(r.absorber_grad))
        top_r = representation_regions[top.region_index]
        notes.append(
            f"highest sensitivity in region {top.region_index} "
            f"(layers {int(top_r.start)}-{int(top_r.end) - 1}, "
            f"|abs_grad|={abs(top.absorber_grad):.4g}).")

        # flat / stuck check: every region SNR below the floor (or unavailable)
        snrs = [r.absorber_snr for r in region_sens if r.absorber_snr is not None]
        if not snrs:
            notes.append("region SNRs unavailable (single seed?) -> cannot "
                         "distinguish landscape from flat; add seeds to assess.")
        elif all(s < _SNR_FLOOR for s in snrs):
            notes.append(f"all region |grad| below SNR {_SNR_FLOOR:g} -> "
                         f"flat/stuck landscape; structural move likely needed.")
        else:
            hi = [r.region_index for r in region_sens
                  if r.absorber_snr is not None and r.absorber_snr >= _SNR_FLOOR]
            notes.append(f"statistically resolved sensitivity in region(s) {hi}.")

    _append_constraint_notes(autopsy, constraints, dp, measurements, notes)
    return autopsy


def _append_constraint_notes(autopsy, constraints, dp, measurements, notes):
    """Evaluate constraints (if provided) and record violations + a note."""
    if not constraints:
        return
    if measurements is None:
        measurements = {}
    viols = _constraints.check_constraints(constraints, dp, measurements)
    autopsy.binding_constraints = viols
    if viols:
        subjects = sorted({v.constraint.subject for v in viols})
        notes.append(f"{len(viols)} constraint(s) violated: {subjects}.")
    else:
        notes.append("all provided constraints feasible.")


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .schemas import Region, regions_to_profiles
    from . import observables as _obs

    n_layers = 20
    regions = [Region(0, 10, absorber_mm=2.3, gap_mm=5.7),
               Region(10, 20, absorber_mm=2.3, gap_mm=5.7)]
    abs_t, gap_t = regions_to_profiles(regions, n_layers)
    dp = DesignPoint(a=2.3, g=5.7, energy=10000.0, n_layers=n_layers,
                     abs_profile=abs_t, gap_profile=gap_t)

    adjoints = np.ones(n_layers)
    n_events, seeds = 500, [1]

    print("[autopsy] measuring objective (total_edep) ...")
    # `observe` forward-seeds 'a' internally, which conflicts with abs_profile.
    # Both regions here share a=2.3/g=5.7, so the scalar point is geometrically
    # identical; use it just to read the objective value.
    dp_scalar = DesignPoint(a=2.3, g=5.7, energy=10000.0, n_layers=n_layers)
    obj = _obs.observe("total_edep", dp_scalar, n_events=n_events, seeds=seeds)
    objective_value = obj.value
    target_value = objective_value * 0.9
    print(f"[autopsy] objective={objective_value:.6g} target={target_value:.6g}")

    cons = _constraints.parse_constraints(
        ["1.0 <= a <= 3.5", "visible_fraction > 0.03"])
    measurements = {"visible_fraction": objective_value / dp.energy}

    autopsy = build_autopsy(
        representation_regions=regions, n_layers=n_layers, dp=dp,
        adjoints=adjoints, objective_value=objective_value,
        target_value=target_value, n_events=n_events, seeds=seeds,
        constraints=cons, measurements=measurements)

    assert len(autopsy.region_sensitivities) == 2, autopsy.region_sensitivities
    assert len(autopsy.per_layer_abs_grad) == n_layers
    assert len(autopsy.per_layer_gap_grad) == n_layers
    for r in autopsy.region_sensitivities:
        assert r.n_layers_in_region == 10

    print()
    print(autopsy.to_text())
    print()
    # round-trip to_dict
    d = autopsy.to_dict()
    assert d["residual"] == autopsy.residual
    assert len(d["region_sensitivities"]) == 2
    print("[autopsy] SELF-VALIDATION PASS")

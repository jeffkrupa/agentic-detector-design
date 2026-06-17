"""Shared dataclass contract for the tool layer, agent layer, and benchmark.

Every tool returns one of these dataclasses (never a bare dict) so that the
orchestrator, subagents, and benchmark all speak the same language. See
docs/AGENTS.md (message/data contract) and docs/SIMULATION_INTERFACE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

# Continuous, differentiable design parameters (names match sim flags).
DIFFERENTIABLE_PARAMS = ("a", "g", "energy")  # absorber mm, gap mm, beam MeV

# Reliability flag values, ordered worst->best for comparisons.
RELIABILITY_LEVELS = ("untrusted", "marginal", "ok")


@dataclass(frozen=True)
class DesignPoint:
    """A point in the (continuous + discrete) design space.

    ``a``/``g`` are the scalar (broadcast-to-all-layers) absorber/gap
    thicknesses. Optionally, ``abs_profile``/``gap_profile`` carry an explicit
    per-layer thickness vector (length == ``n_layers``). When a profile is set
    it overrides the corresponding scalar at the simulation layer (``a``/``g``
    are still kept for provenance / nominal value but are not used to build the
    geometry flags). Profiles are stored as immutable tuples of floats so the
    dataclass stays hashable/frozen.
    """
    a: float            # absorber thickness [mm] (scalar / nominal)
    g: float            # gap thickness [mm] (scalar / nominal)
    energy: float       # beam energy [MeV]
    n_layers: int = 50
    transverse: float = 400.0
    particle: str = "e-"
    abs_profile: Optional[tuple] = None   # per-layer absorber thickness [mm]
    gap_profile: Optional[tuple] = None   # per-layer gap thickness [mm]

    def __post_init__(self):
        for name in ("abs_profile", "gap_profile"):
            val = getattr(self, name)
            if val is None:
                continue
            coerced = tuple(float(x) for x in val)
            if len(coerced) != int(self.n_layers):
                raise ValueError(
                    f"{name} length {len(coerced)} != n_layers {self.n_layers}")
            object.__setattr__(self, name, coerced)

    def with_param(self, name: str, value: float) -> "DesignPoint":
        """Return a copy with one differentiable parameter changed."""
        if name not in DIFFERENTIABLE_PARAMS:
            raise ValueError(f"{name!r} is not a differentiable parameter {DIFFERENTIABLE_PARAMS}")
        return DesignPoint(**{**asdict(self), name: float(value)})

    def get(self, name: str) -> float:
        return float(getattr(self, name))


@dataclass(frozen=True)
class CtrlFlags:
    """Differentiation-control flags (stop-grad policy + regularizers).

    Field names are unambiguous; the trailing comment is the actual CLI short
    flag (verified against Simulation/include/InputParameters.hh). Use
    ``to_cli_args()`` to render the command line. Set a value to ``None`` to omit
    the flag (uses the binary's own default / disabled).
    """
    stop_grad_mode: Optional[int] = 2          # -x / --stop-grad-mode (0,1,2)
    grazing_stop_track: Optional[int] = 1      # -y
    backward_boundary_stop: Optional[int] = 1  # -B
    grazing_threshold: Optional[float] = 0.2   # -f  (|vx| grazing threshold)
    conversion_reg_eps: Optional[str] = "1e-3" # -N  (CRE; 0/None disables)
    gamma_mfp_cap: Optional[float] = 1000.0    # -C  (gmc; 0/None disables)
    numia_mfp_floor: Optional[str] = None      # -A  (off by default)

    # field -> CLI short flag (only fields that map directly to a value flag)
    _FLAG_MAP = {
        "stop_grad_mode": "-x",
        "grazing_stop_track": "-y",
        "backward_boundary_stop": "-B",
        "grazing_threshold": "-f",
        "conversion_reg_eps": "-N",
        "gamma_mfp_cap": "-C",
        "numia_mfp_floor": "-A",
    }

    def to_cli_args(self) -> list:
        args = []
        for field_name, flag in self._FLAG_MAP.items():
            val = getattr(self, field_name)
            if val is None:
                continue
            args += [flag, str(val)]
        return args


@dataclass
class RawRun:
    """Parsed output of a single binary invocation."""
    edeps: Optional[np.ndarray]      # shape (nlayers, 4): [mean_E, var_E, mean_dE, var_dE]
    bar_inputs: Optional[np.ndarray] # shape (3, 2): rows [a, g, energy], cols [mean, var]
    n_events: int
    seed: int
    mode: str                        # "forward" | "reverse"
    seeded_param: Optional[str]      # forward mode: which input carried the :1 dot
    returncode: int = 0
    nan: bool = False


@dataclass
class Observation:
    """A scalar observable measured at a design point (no derivative)."""
    observable: str
    value: float
    stderr: float
    design_point: DesignPoint
    n_total: int = 0                 # total events behind the estimate


@dataclass
class Sensitivity:
    """Exact derivative dO/dtheta with stats and a reliability assessment."""
    observable: str
    wrt: str                         # one of DIFFERENTIABLE_PARAMS
    value: float
    stderr: float
    reliability: str = "ok"          # one of RELIABILITY_LEVELS
    method: str = "forward-AD"       # "forward-AD" | "reverse-AD" | "finite-diff"
    design_point: Optional[DesignPoint] = None

    @property
    def sign(self) -> str:
        if abs(self.value) <= self.stderr:
            return "~0"
        return "+" if self.value > 0 else "-"

    @property
    def log10_abs(self) -> Optional[float]:
        v = abs(self.value)
        return float(np.log10(v)) if v > 0 else None


@dataclass
class Profile:
    """Per-layer forward-mode sensitivity profile of an observable density."""
    observable: str                  # typically "edep_per_layer"
    wrt: str
    per_layer_value: np.ndarray      # shape (nlayers,)
    per_layer_stderr: np.ndarray
    reliability: str = "ok"
    design_point: Optional[DesignPoint] = None


@dataclass
class LossGradient:
    """Reverse-mode gradient of a scalar loss w.r.t. the full design vector."""
    loss_spec: str
    grad: dict                       # {"a": float, "g": float, "energy": float}
    stderr: dict                     # same keys
    reliability: str = "ok"
    method: str = "reverse-AD"
    design_point: Optional[DesignPoint] = None


@dataclass
class CriticCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CriticVerdict:
    passed: bool
    reason: str = ""
    checks: list = field(default_factory=list)   # list[CriticCheck]


def worst_reliability(*levels: str) -> str:
    """Return the most pessimistic reliability level among the arguments."""
    idx = min(RELIABILITY_LEVELS.index(l) for l in levels)
    return RELIABILITY_LEVELS[idx]


# --------------------------------------------------------------------------- #
# Per-region / per-layer design helpers
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A contiguous block of layers sharing one absorber/gap thickness.

    The range is **half-open**: ``[start, end)`` covers layer indices
    ``start, start+1, ..., end-1`` (``end`` itself is NOT included).
    """
    start: int
    end: int
    absorber_mm: float
    gap_mm: float


def regions_to_profiles(regions, n_layers):
    """Expand a list of ``Region`` into ``(abs_tuple, gap_tuple)`` of len n_layers.

    The regions must exactly tile ``[0, n_layers)``: sorted by ``start``,
    contiguous (each region's ``start`` == previous region's ``end``), no gaps,
    no overlaps, first ``start == 0`` and last ``end == n_layers``. Otherwise a
    ``ValueError`` is raised. Each region broadcasts its ``absorber_mm`` /
    ``gap_mm`` across its member layers.
    """
    n_layers = int(n_layers)
    if not regions:
        raise ValueError("regions must be a non-empty list")
    ordered = sorted(regions, key=lambda r: r.start)
    abs_vals = [0.0] * n_layers
    gap_vals = [0.0] * n_layers
    cursor = 0
    for r in ordered:
        s, e = int(r.start), int(r.end)
        if s != cursor:
            raise ValueError(
                f"regions do not tile [0,{n_layers}) contiguously: expected "
                f"start={cursor}, got region [{s},{e})")
        if e <= s:
            raise ValueError(f"region [{s},{e}) is empty or reversed")
        if e > n_layers:
            raise ValueError(f"region [{s},{e}) extends past n_layers={n_layers}")
        for i in range(s, e):
            abs_vals[i] = float(r.absorber_mm)
            gap_vals[i] = float(r.gap_mm)
        cursor = e
    if cursor != n_layers:
        raise ValueError(
            f"regions tile only [0,{cursor}) but n_layers={n_layers}")
    return tuple(abs_vals), tuple(gap_vals)


def profiles_from_design(dp, n_layers):
    """Return ``(abs_tuple, gap_tuple)`` for a DesignPoint.

    Uses ``dp.abs_profile``/``dp.gap_profile`` when set, otherwise broadcasts
    the scalar ``dp.a``/``dp.g`` across all ``n_layers``.
    """
    n_layers = int(n_layers)
    abs_t = dp.abs_profile if dp.abs_profile is not None else tuple([float(dp.a)] * n_layers)
    gap_t = dp.gap_profile if dp.gap_profile is not None else tuple([float(dp.g)] * n_layers)
    return abs_t, gap_t

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
    """A point in the (continuous + discrete) design space."""
    a: float            # absorber thickness [mm]
    g: float            # gap thickness [mm]
    energy: float       # beam energy [MeV]
    n_layers: int = 50
    transverse: float = 400.0
    particle: str = "e-"

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

"""Tool layer: deterministic functions driving the differentiable HepEmShow sim.

Public surface (import from agent / benchmark code):

    from tools import sim, observables, reliability
    from tools.schemas import DesignPoint, Sensitivity, ...

See docs/SIMULATION_INTERFACE.md for the exact simulation contract.
"""
from . import schemas       # noqa: F401
from . import sim           # noqa: F401
from . import observables   # noqa: F401
from . import reliability   # noqa: F401

__all__ = ["schemas", "sim", "observables", "reliability"]

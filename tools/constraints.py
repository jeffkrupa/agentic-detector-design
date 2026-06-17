"""Constraint parsing + checking for the bilevel design loop (E2).

Human-written natural-language constraints are parsed into structured
:class:`Constraint` objects and checked against a :class:`DesignPoint` (for
parameter bounds) and an injected ``measurements`` dict (for observable
constraints). This module performs NO simulation calls -- all observable values
are injected by the caller, so it stays cheap and side-effect free.

Supported constraint forms (one per string; surrounding whitespace and a
trailing ``# comment`` are stripped):

* parameter bounds on ``a``, ``g``, ``energy``::

      "1.0 <= a <= 3.5"     # two-sided box bound
      "a <= 3.5"            # one-sided
      "g >= 3.0"
      "energy == 10000"

* observable constraints (operators ``<``, ``<=``, ``>``, ``>=``, ``==``)::

      "visible_fraction > 0.03"
      "shower_max_depth < 12"
      "total_edep >= 9000"

The orchestrator/optimizer consumes this module for feasibility checking
(``check_constraints``) and for projecting a candidate design back into the
parameter box (``clip_to_bounds``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .schemas import DesignPoint, DIFFERENTIABLE_PARAMS

# Subjects that are design parameters (read off the DesignPoint). Anything else
# is treated as an observable (read off the injected measurements dict).
_PARAM_SUBJECTS = set(DIFFERENTIABLE_PARAMS) | {"transverse"}

_OPS = ("<=", ">=", "==", "<", ">")  # longest-match-first ordering matters


@dataclass
class Constraint:
    """A single parsed constraint.

    For one-sided constraints, ``op``/``rhs`` carry the comparison and
    ``lo``/``hi`` are ``None``. For two-sided box bounds (``lo <= x <= hi``),
    ``lo`` and ``hi`` are both set, ``op`` is ``"<=<="`` (a sentinel), and
    ``rhs`` mirrors ``hi`` for convenience.
    """
    subject: str
    op: str
    rhs: float
    raw: str
    lo: Optional[float] = None
    hi: Optional[float] = None

    @property
    def is_observable(self) -> bool:
        return self.subject not in _PARAM_SUBJECTS

    @property
    def two_sided(self) -> bool:
        return self.lo is not None and self.hi is not None


@dataclass
class ConstraintViolation:
    """A violated (or evaluated) constraint.

    ``amount`` is the signed distance *outside* the feasible region: ``> 0``
    means the constraint is violated by that much; ``<= 0`` means feasible (the
    magnitude is the slack/margin inside the region).
    """
    constraint: Constraint
    value: float
    amount: float


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

# two-sided:  lo <= subject <= hi   (only <= / < allowed on the outer bounds)
_TWO_SIDED = re.compile(
    rf"^\s*({_NUM})\s*(<=|<)\s*({_IDENT})\s*(<=|<)\s*({_NUM})\s*$")
# one-sided:  subject OP rhs   or   rhs OP subject
_ONE_SIDED_LR = re.compile(rf"^\s*({_IDENT})\s*(<=|>=|==|<|>)\s*({_NUM})\s*$")
_ONE_SIDED_RL = re.compile(rf"^\s*({_NUM})\s*(<=|>=|==|<|>)\s*({_IDENT})\s*$")

# flip an operator when the subject is on the right-hand side
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "=="}


def _strip(s: str) -> str:
    # drop trailing comment, then surrounding whitespace
    if "#" in s:
        s = s.split("#", 1)[0]
    return s.strip()


def parse_constraint(s: str) -> Constraint:
    """Parse a single constraint string into a :class:`Constraint`.

    Raises ``ValueError`` if the string cannot be parsed.
    """
    raw = s
    text = _strip(s)
    if not text:
        raise ValueError(f"empty constraint: {raw!r}")

    m = _TWO_SIDED.match(text)
    if m:
        lo, _op1, subject, _op2, hi = m.groups()
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError(f"two-sided bound has lo > hi: {raw!r}")
        return Constraint(subject=subject, op="<=<=", rhs=hi, raw=raw,
                          lo=lo, hi=hi)

    m = _ONE_SIDED_LR.match(text)
    if m:
        subject, op, rhs = m.groups()
        return Constraint(subject=subject, op=op, rhs=float(rhs), raw=raw)

    m = _ONE_SIDED_RL.match(text)
    if m:
        rhs, op, subject = m.groups()
        return Constraint(subject=subject, op=_FLIP[op], rhs=float(rhs), raw=raw)

    raise ValueError(f"unparseable constraint: {raw!r}")


def parse_constraints(list_of_str) -> list:
    """Parse a list of constraint strings, skipping blank/comment-only lines."""
    out = []
    for s in list_of_str:
        if not _strip(s):
            continue
        out.append(parse_constraint(s))
    return out


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #
def _violation_amount(op: str, value: float, rhs: float) -> float:
    """Signed distance outside the feasible region (>0 == violated).

    For strict operators we use the same numeric distance as the non-strict
    form; an exactly-on-the-boundary value reports amount 0.0 (treated as
    feasible / on the margin).
    """
    if op in ("<", "<="):
        return value - rhs          # feasible when value <= rhs
    if op in (">", ">="):
        return rhs - value          # feasible when value >= rhs
    if op == "==":
        return abs(value - rhs)     # feasible only when equal
    raise ValueError(f"unknown operator {op!r}")


def _read_value(constraint: Constraint, dp: DesignPoint, measurements: dict) -> float:
    if constraint.is_observable:
        if measurements is None or constraint.subject not in measurements:
            raise KeyError(
                f"measurement for observable {constraint.subject!r} not provided "
                f"(constraint {constraint.raw!r}); pass it in `measurements`")
        return float(measurements[constraint.subject])
    return float(dp.get(constraint.subject))


def check_constraints(constraints, dp: DesignPoint, measurements: dict) -> list:
    """Return the list of violated constraints (empty == feasible).

    Parameter constraints read their value from ``dp``; observable constraints
    read from ``measurements[name]`` (raises ``KeyError`` with a clear message
    if the measurement is missing). Two-sided box bounds are checked on both
    sides and reported once (with the worst side's amount).
    """
    violations = []
    for c in constraints:
        value = _read_value(c, dp, measurements)
        if c.two_sided:
            amt_lo = c.lo - value    # violated if value < lo
            amt_hi = value - c.hi    # violated if value > hi
            amount = max(amt_lo, amt_hi)
        else:
            amount = _violation_amount(c.op, value, c.rhs)
        if amount > 0.0:
            violations.append(ConstraintViolation(constraint=c, value=value,
                                                  amount=float(amount)))
    return violations


# --------------------------------------------------------------------------- #
# Projection (box bounds only)
# --------------------------------------------------------------------------- #
def clip_to_bounds(dp: DesignPoint, constraints) -> DesignPoint:
    """Clip ``a``/``g``/``energy`` into the feasible box and return a new point.

    Only PARAMETER bound constraints are honoured; observable constraints are
    ignored here (they cannot be satisfied by clipping a scalar). This handles
    box bounds only -- it collects, per parameter, the tightest lower and upper
    bound implied by the constraints and clamps the value into ``[lo, hi]``.
    ``==`` constraints pin the parameter to the exact value.
    """
    lo_bound: dict = {}
    hi_bound: dict = {}

    def _tighten_lo(name, v):
        lo_bound[name] = v if name not in lo_bound else max(lo_bound[name], v)

    def _tighten_hi(name, v):
        hi_bound[name] = v if name not in hi_bound else min(hi_bound[name], v)

    for c in constraints:
        if c.is_observable or c.subject not in DIFFERENTIABLE_PARAMS:
            continue
        if c.two_sided:
            _tighten_lo(c.subject, c.lo)
            _tighten_hi(c.subject, c.hi)
        elif c.op in ("<", "<="):
            _tighten_hi(c.subject, c.rhs)
        elif c.op in (">", ">="):
            _tighten_lo(c.subject, c.rhs)
        elif c.op == "==":
            _tighten_lo(c.subject, c.rhs)
            _tighten_hi(c.subject, c.rhs)

    new_dp = dp
    for name in DIFFERENTIABLE_PARAMS:
        val = float(new_dp.get(name))
        clipped = val
        if name in lo_bound:
            clipped = max(clipped, lo_bound[name])
        if name in hi_bound:
            clipped = min(clipped, hi_bound[name])
        if clipped != val:
            new_dp = new_dp.with_param(name, clipped)
    return new_dp


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    specs = [
        "1.0 <= a <= 3.5",
        "g >= 3.0",
        "visible_fraction > 0.03   # readout floor",
        "shower_max_depth < 12",
    ]
    cons = parse_constraints(specs)
    print("[parse] parsed", len(cons), "constraints:")
    for c in cons:
        kind = "observable" if c.is_observable else "param"
        print(f"   {c.raw!r:40s} -> subject={c.subject} op={c.op} "
              f"rhs={c.rhs} lo={c.lo} hi={c.hi} ({kind})")

    # a is out of range (5.0 > 3.5); g is fine (5.7 >= 3.0)
    dp = DesignPoint(a=5.0, g=5.7, energy=10000.0, n_layers=20)
    # visible_fraction below its floor -> violated; shower_max_depth fine
    measurements = {"visible_fraction": 0.01, "shower_max_depth": 8.0}

    viols = check_constraints(cons, dp, measurements)
    print("\n[check] violations:")
    for v in viols:
        print(f"   {v.constraint.raw!r} value={v.value} amount={v.amount:+.4g}")

    subjects = {v.constraint.subject for v in viols}
    assert subjects == {"a", "visible_fraction"}, subjects
    # g and shower_max_depth must NOT appear
    assert "g" not in subjects and "shower_max_depth" not in subjects
    print("[check] OK: exactly {a, visible_fraction} violated as expected.")

    clipped = clip_to_bounds(dp, cons)
    print(f"\n[clip] a: {dp.a} -> {clipped.a} (g: {dp.g} -> {clipped.g})")
    assert 1.0 <= clipped.a <= 3.5, clipped.a
    assert clipped.a == 3.5, clipped.a   # clamped to upper bound
    assert clipped.g == dp.g             # g already feasible, untouched
    # re-check param constraints after clipping: a no longer violated
    viols2 = check_constraints(cons, clipped, measurements)
    subjects2 = {v.constraint.subject for v in viols2}
    assert subjects2 == {"visible_fraction"}, subjects2
    print("[clip] OK: a back in range; only observable constraint remains.")
    print("\n[constraints] SELF-VALIDATION PASS")

"""Discrete structural design space for Experiment E2 (the bilevel loop).

This module is the agent's OUTER-loop representation: an ordered tuple of
``Region`` that exactly tiles ``[0, n_layers)``, plus a small algebra of
*structural moves* (split / merge / set-thickness / split-equal) that transform
one representation into a new one. It is pure logic + validation: there are NO
simulation calls here. The inner (AD) optimizer differentiates the continuous
thicknesses *within* a fixed representation; this module only changes the
*structure* (how layers are grouped into regions).

See ``HANDOFF_HYPOTHESIS_TEST.md``:
  Agent proposes a structural representation
        -> AD optimizer tunes continuous params within it (inner loop)
        -> full sim evaluates -> autopsy -> agent proposes next structural edit.

Public API
----------
- ``DesignRepresentation`` (frozen dataclass) with ``validate``,
  ``to_design_point``, ``n_regions``, ``region_of_layer`` and the
  ``uniform(...)`` constructor helper.
- Move dataclasses: ``Split``, ``Merge``, ``SetRegionThickness``,
  ``SplitEqual`` (all subclasses of ``Move``, frozen, printable, carrying an
  optional ``justification``).
- ``apply_move(rep, move) -> DesignRepresentation`` (full legality checks).
- ``legal_moves(rep) -> list[Move]`` (enumerate currently legal moves).
- ``IllegalMoveError`` (subclass of ``ValueError``).

Build-on API (from ``tools.schemas``): ``Region``, ``regions_to_profiles``,
``DesignPoint``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tools.schemas import Region, regions_to_profiles, DesignPoint


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class IllegalMoveError(ValueError):
    """Raised when a structural move cannot be legally applied to a rep."""


# --------------------------------------------------------------------------- #
# Representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DesignRepresentation:
    """An ordered tuple of ``Region`` that exactly tiles ``[0, n_layers)``.

    This is the discrete/structural state the outer-loop agent edits. The
    continuous thicknesses (``absorber_mm`` / ``gap_mm`` on each region) are the
    handles the inner AD loop tunes; the agent edits *which* layers share a
    thickness (the region boundaries / count).

    Attributes
    ----------
    regions : tuple[Region, ...]
        Sorted, contiguous, non-overlapping; first ``start == 0`` and last
        ``end == n_layers``.
    n_layers, energy, particle : the fixed beam/geometry context.
    """

    regions: Tuple[Region, ...]
    n_layers: int
    energy: float
    particle: str = "e-"

    def __post_init__(self):
        # Normalize to a tuple so the dataclass stays hashable / frozen and so
        # callers may pass a list.
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "n_layers", int(self.n_layers))
        object.__setattr__(self, "energy", float(self.energy))
        self.validate()

    # -- validation -------------------------------------------------------- #
    def validate(self) -> None:
        """Raise ``ValueError`` unless the regions exactly tile ``[0, n_layers)``.

        Reuses ``regions_to_profiles`` for the tiling check (sorted, contiguous,
        no gaps/overlaps, covers exactly ``[0, n_layers)``).
        """
        if not self.regions:
            raise ValueError("DesignRepresentation requires at least one region")
        # regions_to_profiles raises ValueError on any tiling violation.
        regions_to_profiles(self.regions, self.n_layers)

    # -- properties / helpers --------------------------------------------- #
    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def region_of_layer(self, layer: int) -> int:
        """Return the index of the region containing absolute ``layer``."""
        layer = int(layer)
        if not (0 <= layer < self.n_layers):
            raise ValueError(
                f"layer {layer} out of range [0,{self.n_layers})")
        for i, r in enumerate(self.regions):
            if r.start <= layer < r.end:
                return i
        # Unreachable if validate() passed.
        raise ValueError(f"no region contains layer {layer}")

    # -- conversion to the continuous design point ------------------------ #
    def to_design_point(self) -> DesignPoint:
        """Expand the regions into per-layer profiles and build a DesignPoint.

        The scalar ``a`` / ``g`` on the returned ``DesignPoint`` are taken from
        the *first* region (``regions[0]``) as a nominal value for provenance;
        the actual geometry is carried by ``abs_profile`` / ``gap_profile``
        (length ``n_layers``), which override the scalars at the sim layer.
        """
        abs_t, gap_t = regions_to_profiles(self.regions, self.n_layers)
        nominal = self.regions[0]
        return DesignPoint(
            a=float(nominal.absorber_mm),
            g=float(nominal.gap_mm),
            energy=float(self.energy),
            n_layers=int(self.n_layers),
            particle=self.particle,
            abs_profile=abs_t,
            gap_profile=gap_t,
        )

    # -- constructor helper ----------------------------------------------- #
    @staticmethod
    def uniform(
        n_layers: int,
        n_regions: int,
        absorber_mm: float,
        gap_mm: float,
        energy: float,
        particle: str = "e-",
    ) -> "DesignRepresentation":
        """Partition ``[0, n_layers)`` into ``n_regions`` near-equal regions.

        All regions get the same ``absorber_mm`` / ``gap_mm``. The partition is
        the canonical near-equal contiguous split: the first ``n_layers %
        n_regions`` regions are one layer larger than the rest.
        """
        n_layers = int(n_layers)
        n_regions = int(n_regions)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not (1 <= n_regions <= n_layers):
            raise ValueError(
                f"n_regions must be in [1, n_layers]={n_layers}, got {n_regions}")
        bounds = _near_equal_bounds(0, n_layers, n_regions)
        regions = tuple(
            Region(start=s, end=e, absorber_mm=float(absorber_mm),
                   gap_mm=float(gap_mm))
            for (s, e) in bounds
        )
        return DesignRepresentation(
            regions=regions, n_layers=n_layers, energy=energy,
            particle=particle)

    def __repr__(self) -> str:
        body = ", ".join(
            f"[{r.start},{r.end}) a={r.absorber_mm:g} g={r.gap_mm:g}"
            for r in self.regions
        )
        return (f"DesignRepresentation(n_layers={self.n_layers}, "
                f"energy={self.energy:g}, particle={self.particle!r}, "
                f"regions=[{body}])")


def _near_equal_bounds(start: int, end: int, k: int) -> List[Tuple[int, int]]:
    """Split ``[start, end)`` into ``k`` near-equal contiguous half-open spans.

    The first ``(end-start) % k`` spans are one unit longer. Requires
    ``end - start >= k``.
    """
    total = end - start
    if k < 1:
        raise ValueError("k must be >= 1")
    if total < k:
        raise ValueError(
            f"cannot split span of size {total} into {k} non-empty parts")
    base, extra = divmod(total, k)
    bounds = []
    cur = start
    for i in range(k):
        size = base + (1 if i < extra else 0)
        bounds.append((cur, cur + size))
        cur += size
    return bounds


# --------------------------------------------------------------------------- #
# Structural moves
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Move:
    """Base class for structural moves (a tagged union via subclassing).

    Every concrete move carries an optional human-readable ``justification``
    (why the agent proposed it) as its trailing field, preserved for the
    optimization autopsy / log. ``Move`` itself declares no fields so that
    subclasses can add required positional fields under Python 3.9's dataclass
    inheritance rules (a base-class default field would force every subclass
    field to have a default).
    """


@dataclass(frozen=True)
class Split(Move):
    """Split region ``region_index`` into two at absolute layer ``at_layer``.

    Both children inherit the parent's thicknesses, so the *geometry is
    unchanged* at the instant of the split (it only increases the number of
    independently tunable regions). ``at_layer`` must be strictly inside the
    target region and not equal to its ``start`` (so both children are
    non-empty); the new boundary is ``[start, at_layer)`` and
    ``[at_layer, end)``.
    """

    region_index: int
    at_layer: int
    justification: str = ""

    def __repr__(self) -> str:
        return (f"Split(region_index={self.region_index}, "
                f"at_layer={self.at_layer}, justification={self.justification!r})")


@dataclass(frozen=True)
class Merge(Move):
    """Merge region ``region_index`` with its RIGHT neighbor.

    Merge-thickness rule: layer-count-weighted average of the two regions'
    thicknesses, i.e.
        merged = (n_left*t_left + n_right*t_right) / (n_left + n_right)
    applied independently to ``absorber_mm`` and ``gap_mm``. This conserves the
    total (sum over layers) absorber/gap material across the merge. Requires a
    right neighbor to exist (``region_index < n_regions - 1``).
    """

    region_index: int
    justification: str = ""

    def __repr__(self) -> str:
        return (f"Merge(region_index={self.region_index}, "
                f"justification={self.justification!r})")


@dataclass(frozen=True)
class SetRegionThickness(Move):
    """Set one region's thickness; only the provided fields change.

    Used by the inner optimizer to *write back* tuned continuous values into the
    structural representation. ``absorber_mm`` and/or ``gap_mm`` may be ``None``
    to leave that field untouched. The region boundaries are unchanged.
    """

    region_index: int
    absorber_mm: Optional[float] = None
    gap_mm: Optional[float] = None
    justification: str = ""

    def __repr__(self) -> str:
        return (f"SetRegionThickness(region_index={self.region_index}, "
                f"absorber_mm={self.absorber_mm}, gap_mm={self.gap_mm}, "
                f"justification={self.justification!r})")


@dataclass(frozen=True)
class SplitEqual(Move):
    """Split region ``region_index`` into ``k`` near-equal contiguous children.

    A granularity-increase move. All children inherit the parent's thicknesses
    (geometry unchanged at the instant of the split). Requires ``k >= 2`` and the
    region to contain at least ``k`` layers.
    """

    region_index: int
    k: int
    justification: str = ""

    def __repr__(self) -> str:
        return (f"SplitEqual(region_index={self.region_index}, k={self.k}, "
                f"justification={self.justification!r})")


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def apply_move(rep: DesignRepresentation, move: Move) -> DesignRepresentation:
    """Return a NEW ``DesignRepresentation`` with ``move`` applied.

    Performs full legality checks and raises ``IllegalMoveError`` (a subclass of
    ``ValueError``) on any illegal move. The returned representation is
    constructed through ``DesignRepresentation`` and therefore re-validates the
    tiling.
    """
    if not isinstance(move, Move):
        raise IllegalMoveError(f"not a Move: {move!r}")

    if isinstance(move, Split):
        new_regions = _apply_split(rep, move.region_index, move.at_layer)
    elif isinstance(move, SplitEqual):
        new_regions = _apply_split_equal(rep, move.region_index, move.k)
    elif isinstance(move, Merge):
        new_regions = _apply_merge(rep, move.region_index)
    elif isinstance(move, SetRegionThickness):
        new_regions = _apply_set_thickness(
            rep, move.region_index, move.absorber_mm, move.gap_mm)
    else:
        raise IllegalMoveError(f"unknown move type: {type(move).__name__}")

    return DesignRepresentation(
        regions=new_regions,
        n_layers=rep.n_layers,
        energy=rep.energy,
        particle=rep.particle,
    )


def _check_index(rep: DesignRepresentation, idx: int) -> None:
    if not (0 <= idx < rep.n_regions):
        raise IllegalMoveError(
            f"region_index {idx} out of range [0,{rep.n_regions})")


def _apply_split(rep, region_index, at_layer):
    _check_index(rep, region_index)
    at_layer = int(at_layer)
    r = rep.regions[region_index]
    if not (r.start < at_layer < r.end):
        raise IllegalMoveError(
            f"Split.at_layer={at_layer} must be strictly inside region "
            f"[{r.start},{r.end}) (not equal to start and not at/past end)")
    left = Region(start=r.start, end=at_layer,
                  absorber_mm=r.absorber_mm, gap_mm=r.gap_mm)
    right = Region(start=at_layer, end=r.end,
                   absorber_mm=r.absorber_mm, gap_mm=r.gap_mm)
    regions = list(rep.regions)
    regions[region_index:region_index + 1] = [left, right]
    return tuple(regions)


def _apply_split_equal(rep, region_index, k):
    _check_index(rep, region_index)
    k = int(k)
    if k < 2:
        raise IllegalMoveError(f"SplitEqual.k must be >= 2, got {k}")
    r = rep.regions[region_index]
    size = r.end - r.start
    if size < k:
        raise IllegalMoveError(
            f"SplitEqual: region [{r.start},{r.end}) has {size} layers, "
            f"cannot split into {k} non-empty children")
    bounds = _near_equal_bounds(r.start, r.end, k)
    children = [
        Region(start=s, end=e, absorber_mm=r.absorber_mm, gap_mm=r.gap_mm)
        for (s, e) in bounds
    ]
    regions = list(rep.regions)
    regions[region_index:region_index + 1] = children
    return tuple(regions)


def _apply_merge(rep, region_index):
    _check_index(rep, region_index)
    if region_index >= rep.n_regions - 1:
        raise IllegalMoveError(
            f"Merge: region {region_index} has no right neighbor "
            f"(n_regions={rep.n_regions})")
    left = rep.regions[region_index]
    right = rep.regions[region_index + 1]
    n_left = left.end - left.start
    n_right = right.end - right.start
    total = n_left + n_right
    merged_abs = (n_left * left.absorber_mm + n_right * right.absorber_mm) / total
    merged_gap = (n_left * left.gap_mm + n_right * right.gap_mm) / total
    merged = Region(start=left.start, end=right.end,
                    absorber_mm=merged_abs, gap_mm=merged_gap)
    regions = list(rep.regions)
    regions[region_index:region_index + 2] = [merged]
    return tuple(regions)


def _apply_set_thickness(rep, region_index, absorber_mm, gap_mm):
    _check_index(rep, region_index)
    r = rep.regions[region_index]
    new_abs = r.absorber_mm if absorber_mm is None else float(absorber_mm)
    new_gap = r.gap_mm if gap_mm is None else float(gap_mm)
    updated = Region(start=r.start, end=r.end,
                     absorber_mm=new_abs, gap_mm=new_gap)
    regions = list(rep.regions)
    regions[region_index] = updated
    return tuple(regions)


# --------------------------------------------------------------------------- #
# Legal-move enumeration (for a scripted baseline)
# --------------------------------------------------------------------------- #
def legal_moves(rep: DesignRepresentation) -> List[Move]:
    """Enumerate currently-legal structural moves for ``rep``.

    Includes every valid ``Split`` point (each interior layer of each region)
    and every valid ``Merge`` (each region that has a right neighbor). Keeps it
    simple: ``SplitEqual`` and ``SetRegionThickness`` are parameterized by
    continuous/integer values the caller chooses, so they are not enumerated
    here.
    """
    moves: List[Move] = []
    for ri, r in enumerate(rep.regions):
        for at in range(r.start + 1, r.end):  # strictly interior boundaries
            moves.append(Split(region_index=ri, at_layer=at,
                               justification="enumerated split"))
    for ri in range(rep.n_regions - 1):
        moves.append(Merge(region_index=ri, justification="enumerated merge"))
    return moves


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    failures = []

    def check(cond, msg):
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    print("== structural_moves self-validation ==")

    # 1. uniform with 1 region
    rep0 = DesignRepresentation.uniform(
        n_layers=20, n_regions=1, absorber_mm=2.3, gap_mm=5.7,
        energy=10000, particle="e-")
    check(rep0.n_regions == 1, "uniform(1) -> 1 region")
    check(rep0.regions[0].start == 0 and rep0.regions[0].end == 20,
          "uniform(1) tiles [0,20)")
    rep0.validate()

    # 2. Split at layer 10 -> 2 regions, still tiles, thicknesses inherited
    rep_s = apply_move(rep0, Split(region_index=0, at_layer=10,
                                   justification="front/rear split"))
    check(rep_s.n_regions == 2, "Split -> 2 regions")
    check(rep_s.regions[0].start == 0 and rep_s.regions[0].end == 10
          and rep_s.regions[1].start == 10 and rep_s.regions[1].end == 20,
          "Split tiles [0,10)+[10,20)")
    check(rep_s.regions[0].absorber_mm == 2.3 and rep_s.regions[1].absorber_mm == 2.3
          and rep_s.regions[0].gap_mm == 5.7 and rep_s.regions[1].gap_mm == 5.7,
          "Split children inherit parent thicknesses")
    rep_s.validate()

    # 3. SplitEqual(0,4) on fresh uniform-1 -> 4 near-equal regions tiling [0,20)
    rep_se = apply_move(rep0, SplitEqual(region_index=0, k=4))
    check(rep_se.n_regions == 4, "SplitEqual(0,4) -> 4 regions")
    sizes = [r.end - r.start for r in rep_se.regions]
    check(sizes == [5, 5, 5, 5], f"SplitEqual(0,4) near-equal sizes == {sizes}")
    abs_t, gap_t = regions_to_profiles(rep_se.regions, 20)
    check(len(abs_t) == 20 and len(gap_t) == 20,
          "SplitEqual still tiles [0,20)")

    # uneven case: 20 into 3 -> [7,7,6]
    rep_se3 = apply_move(rep0, SplitEqual(region_index=0, k=3))
    sizes3 = [r.end - r.start for r in rep_se3.regions]
    check(sizes3 == [7, 7, 6], f"SplitEqual(0,3) near-equal sizes == {sizes3}")

    # 4. Merge two regions back -> fewer regions, valid tiling
    #    Differentiate the two children first so we can check the weighted rule.
    rep_diff = apply_move(rep_se,
                          SetRegionThickness(region_index=0, absorber_mm=1.0))
    rep_diff = apply_move(rep_diff,
                          SetRegionThickness(region_index=1, absorber_mm=3.0))
    rep_m = apply_move(rep_diff, Merge(region_index=0,
                                       justification="merge low-value"))
    check(rep_m.n_regions == rep_diff.n_regions - 1, "Merge -> fewer regions")
    # both children had 5 layers => simple average 2.0
    check(abs(rep_m.regions[0].absorber_mm - 2.0) < 1e-12,
          f"Merge weighted-average absorber == {rep_m.regions[0].absorber_mm}")
    rep_m.validate()

    # 5. SetRegionThickness changes only target region; profiles correct
    rep_t = apply_move(rep_s,
                       SetRegionThickness(region_index=1, absorber_mm=9.9,
                                          gap_mm=None))
    check(rep_t.regions[0].absorber_mm == 2.3, "SetRegionThickness leaves region 0")
    check(rep_t.regions[1].absorber_mm == 9.9, "SetRegionThickness sets region 1 abs")
    check(rep_t.regions[1].gap_mm == 5.7,
          "SetRegionThickness gap_mm=None untouched")
    dp = rep_t.to_design_point()
    check(dp.abs_profile is not None and len(dp.abs_profile) == 20,
          "to_design_point abs_profile length 20")
    check(dp.gap_profile is not None and len(dp.gap_profile) == 20,
          "to_design_point gap_profile length 20")
    expected_abs = tuple([2.3] * 10 + [9.9] * 10)
    check(dp.abs_profile == expected_abs,
          "to_design_point abs_profile per-region values correct")
    check(dp.gap_profile == tuple([5.7] * 20),
          "to_design_point gap_profile per-region values correct")
    check(dp.a == 2.3 and dp.g == 5.7,
          "to_design_point nominal a/g from region[0] (provenance)")

    # region_of_layer
    check(rep_t.region_of_layer(3) == 0 and rep_t.region_of_layer(15) == 1,
          "region_of_layer maps correctly")

    # 6. Illegal moves raise
    def expect_illegal(fn, label):
        try:
            fn()
        except ValueError:
            print(f"  PASS: illegal {label} raised ValueError")
        else:
            print(f"  FAIL: illegal {label} did NOT raise")
            failures.append(f"illegal {label} did not raise")

    expect_illegal(lambda: apply_move(rep0, Split(region_index=0, at_layer=0)),
                   "Split at region start")
    expect_illegal(lambda: apply_move(rep0, Split(region_index=0, at_layer=20)),
                   "Split at region end")
    expect_illegal(lambda: apply_move(rep0, Split(region_index=5, at_layer=3)),
                   "Split region_index out of range")
    expect_illegal(lambda: apply_move(rep_s, Merge(region_index=1)),
                   "Merge with no right neighbor")
    expect_illegal(lambda: apply_move(rep0, SplitEqual(region_index=0, k=1)),
                   "SplitEqual k<2")
    expect_illegal(
        lambda: apply_move(
            DesignRepresentation.uniform(3, 1, 1.0, 1.0, 100.0),
            SplitEqual(region_index=0, k=5)),
        "SplitEqual region too small")

    # 7. legal_moves sanity
    lm = legal_moves(rep_s)
    n_splits = sum(1 for m in lm if isinstance(m, Split))
    n_merges = sum(1 for m in lm if isinstance(m, Merge))
    # region [0,10): interior splits 1..9 = 9; region [10,20): 11..19 = 9; merges=1
    check(n_splits == 18 and n_merges == 1,
          f"legal_moves enumerates {n_splits} splits + {n_merges} merge")
    for m in lm:
        apply_move(rep_s, m)  # every enumerated move must be applicable

    print()
    if failures:
        print(f"SUMMARY: FAIL ({len(failures)} failing checks)")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("SUMMARY: PASS (all checks passed)")

"""Functional wrapper around the differentiable HepEmShow binaries.

Responsibilities
----------------
* build correct command lines for the forward and reverse AD builds,
* run each invocation in an isolated temp dir (each process writes edeps_* /
  barInputs into its CWD),
* parse the outputs into numpy arrays,
* cache parsed results on disk keyed by a hash of the full command,
* expose high-level helpers used by observables.py / reliability.py / the agent.

All paths and the canonical control-flag set come from config.yaml (falling back
to config.example.yaml). Nothing here is hard-coded.

CLI
---
    python -m tools.sim --selftest
    python -m tools.sim --observable total_edep --wrt a --a 2.3 --energy 10000 -n 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .schemas import DesignPoint, CtrlFlags, RawRun, DIFFERENTIABLE_PARAMS

_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_CONFIG_CACHE: Optional[dict] = None


def load_config() -> dict:
    """Load config.yaml, falling back to config.example.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")
    for name in ("config.yaml", "config.example.yaml"):
        p = _AGENTIC / name
        if p.exists():
            with open(p) as fh:
                _CONFIG_CACHE = yaml.safe_load(fh)
            _CONFIG_CACHE["_source"] = str(p)
            return _CONFIG_CACHE
    raise FileNotFoundError("No config.yaml or config.example.yaml found in agentic/")


def ctrl_flags_from_config(cfg: Optional[dict] = None) -> CtrlFlags:
    cfg = cfg or load_config()
    c = dict(cfg.get("ctrl_flags", {}))
    return CtrlFlags(
        stop_grad_mode=c.get("stop_grad_mode", 2),
        grazing_stop_track=c.get("grazing_stop_track", 1),
        backward_boundary_stop=c.get("backward_boundary_stop", 1),
        grazing_threshold=c.get("grazing_threshold", 0.2),
        conversion_reg_eps=c.get("conversion_reg_eps", "1e-3"),
        gamma_mfp_cap=c.get("gamma_mfp_cap", 1000.0),
        numia_mfp_floor=c.get("numia_mfp_floor", None),
    )


def default_design_point(cfg: Optional[dict] = None) -> DesignPoint:
    cfg = cfg or load_config()
    d = cfg.get("defaults", {})
    return DesignPoint(
        a=float(d.get("absorber_mm", 2.3)),
        g=float(d.get("gap_mm", 5.7)),
        energy=float(d.get("energy_mev", 10000.0)),
        n_layers=int(d.get("n_layers", 50)),
        transverse=float(d.get("transverse_mm", 400.0)),
        particle=str(d.get("particle", "e-")),
    )


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def _common_args(dp: DesignPoint, ctrl: CtrlFlags, n_events: int, seed: int,
                 cfg: dict, seeded_param: Optional[str]) -> list:
    """Geometry/primary/run args shared by forward and reverse builds.

    ``seeded_param`` (forward mode only) appends ':1' to that input to seed the
    forward-AD dot value. Exactly one of {a, g, energy} may be seeded.
    """
    data = cfg["paths"]["hepem_data"]

    def maybe_seed(name: str, value) -> str:
        return f"{value}:1" if name == seeded_param else f"{value}"

    # A per-layer profile cannot also be forward-seeded as a scalar input.
    if dp.abs_profile is not None and seeded_param == "a":
        raise ValueError("cannot seed scalar 'a' when abs_profile is set")
    if dp.gap_profile is not None and seeded_param == "g":
        raise ValueError("cannot seed scalar 'g' when gap_profile is set")

    args = [
        "-d", data,
        "-n", str(int(n_events)),
        "-s", str(int(seed)),
        "-p", dp.particle,
        "-l", str(int(dp.n_layers)),
        "-t", str(dp.transverse),
        "-e", maybe_seed("energy", dp.energy),
    ]
    # Absorber geometry: per-layer profile overrides scalar.
    if dp.abs_profile is not None:
        args += ["--abs-profile", ":".join(repr(float(x)) for x in dp.abs_profile)]
    else:
        args += ["-a", maybe_seed("a", dp.a)]
    # Gap geometry: per-layer profile overrides scalar.
    if dp.gap_profile is not None:
        args += ["--gap-profile", ":".join(repr(float(x)) for x in dp.gap_profile)]
    else:
        args += ["-g", maybe_seed("g", dp.g)]
    args += ["-v", "0"]
    args += ctrl.to_cli_args()
    return args


def _flag_hash(binary: str, args: list) -> str:
    payload = json.dumps([binary, args], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_dir(cfg: dict) -> Path:
    d = _AGENTIC / cfg["paths"].get("cache_dir", ".sim_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_load(cfg: dict, key: str) -> Optional[dict]:
    p = _cache_dir(cfg) / f"{key}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _cache_store(cfg: dict, key: str, **arrays) -> None:
    p = _cache_dir(cfg) / f"{key}.npz"
    np.savez(p, **arrays)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _parse_edeps(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 4:
        raise ValueError(f"{path}: expected 4 columns, got {arr.shape}")
    return arr


def _parse_bar_inputs(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    # rows: [absorber, gap, energy]; cols: [mean, var]
    return arr


def _parse_bar_inputs_per_layer(path: Path) -> np.ndarray:
    """Parse the per-layer reverse output file (``barInputsPerLayer``).

    Returns shape ``(2N+1, 2)`` with columns ``[mean, var]``:
      * rows ``0 .. N-1``   : d(objective)/d(abs_thick[i])
      * rows ``N .. 2N-1``  : d(objective)/d(gap_thick[i])
      * row  ``2N``         : d(objective)/d(energy)
    The ``#`` header line (if present) is skipped by ``np.loadtxt``.
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


def run_forward_gap(dp: DesignPoint, seeded_param: str, n_events: int, seed: int,
                    ctrl: Optional[CtrlFlags] = None, use_cache: bool = True) -> np.ndarray:
    """Forward-mode run returning the **per-layer GAP energy** (sampled signal).

    Same invocation as :func:`run_forward` but parses the ``edeps_gap_<seed>``
    file (identical 4-column layout ``[mean_E, var_E, mean_dE, var_dE]``) that
    the binary writes alongside the combined ``edeps_<seed>``. The absorber
    energy is ``combined - gap`` (no separate output). Returns the ``(nlayers, 4)``
    array, or ``None`` if the run failed. Uses a cache namespace distinct from
    :func:`run_forward` via a ``"__gap__"`` marker baked into the key.
    """
    if seeded_param not in DIFFERENTIABLE_PARAMS:
        raise ValueError(f"seeded_param must be one of {DIFFERENTIABLE_PARAMS}")
    cfg = load_config()
    ctrl = ctrl or ctrl_flags_from_config(cfg)
    binary = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=seeded_param)
    key = _flag_hash(binary, ["__gap__", *args])

    if use_cache:
        cached = _cache_load(cfg, key)
        if cached is not None:
            arr = cached.get("edeps_gap")
            return arr if arr is not None else None

    arr, rc, nan = _execute(binary, args, expect="edeps_gap", seed=seed)
    if use_cache and rc == 0 and arr is not None:
        _cache_store(cfg, key, edeps_gap=arr, rc=rc, nan=nan)
    return arr


def run_forward(dp: DesignPoint, seeded_param: str, n_events: int, seed: int,
                ctrl: Optional[CtrlFlags] = None, use_cache: bool = True) -> RawRun:
    """Forward-mode run seeding exactly one differentiable input."""
    if seeded_param not in DIFFERENTIABLE_PARAMS:
        raise ValueError(f"seeded_param must be one of {DIFFERENTIABLE_PARAMS}")
    cfg = load_config()
    ctrl = ctrl or ctrl_flags_from_config(cfg)
    binary = cfg["paths"]["forward_bin"]
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=seeded_param)
    key = _flag_hash(binary, args)

    if use_cache:
        cached = _cache_load(cfg, key)
        if cached is not None:
            return RawRun(edeps=cached["edeps"], bar_inputs=None, n_events=n_events,
                          seed=seed, mode="forward", seeded_param=seeded_param,
                          returncode=int(cached.get("rc", 0)),
                          nan=bool(cached.get("nan", False)))

    edeps, rc, nan = _execute(binary, args, expect="edeps", seed=seed)
    if use_cache and rc == 0 and edeps is not None:
        _cache_store(cfg, key, edeps=edeps, rc=rc, nan=nan)
    return RawRun(edeps=edeps, bar_inputs=None, n_events=n_events, seed=seed,
                  mode="forward", seeded_param=seeded_param, returncode=rc, nan=nan)


def run_reverse(dp: DesignPoint, adjoints: np.ndarray, n_events: int, seed: int,
                ctrl: Optional[CtrlFlags] = None, use_cache: bool = True,
                gap_adjoints: Optional[np.ndarray] = None) -> RawRun:
    """Reverse-mode run; ``adjoints`` are per-layer output bars (len = n_layers).

    ``gap_adjoints`` (optional, len = n_layers) additionally seeds the per-layer
    **GAP** energy outputs via ``--bar-gap``. The returned ``barInputs`` then
    reflect d/d(design) of ``sum_l(adjoints[l]*E_l + gap_adjoints[l]*E_gap,l)``.
    Pass ``adjoints`` all-zero with ``gap_adjoints`` set to differentiate a pure
    gap-signal objective. When ``gap_adjoints`` is None the behaviour and command
    line are unchanged (combined-energy path only).
    """
    cfg = load_config()
    ctrl = ctrl or ctrl_flags_from_config(cfg)
    binary = cfg["paths"]["reverse_bin"]
    adj = np.asarray(adjoints, dtype=float).ravel()
    if adj.size != dp.n_layers:
        raise ValueError(f"adjoints length {adj.size} != n_layers {dp.n_layers}")
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    args += ["-b", ":".join(repr(float(x)) for x in adj)]
    if gap_adjoints is not None:
        gadj = np.asarray(gap_adjoints, dtype=float).ravel()
        if gadj.size != dp.n_layers:
            raise ValueError(f"gap_adjoints length {gadj.size} != n_layers {dp.n_layers}")
        args += ["--bar-gap", ":".join(repr(float(x)) for x in gadj)]
    key = _flag_hash(binary, args)

    if use_cache:
        cached = _cache_load(cfg, key)
        if cached is not None:
            return RawRun(edeps=cached.get("edeps"), bar_inputs=cached["bar_inputs"],
                          n_events=n_events, seed=seed, mode="reverse",
                          seeded_param=None, returncode=int(cached.get("rc", 0)),
                          nan=bool(cached.get("nan", False)))

    bar, rc, nan = _execute(binary, args, expect="barInputs", seed=seed)
    if use_cache and rc == 0 and bar is not None:
        _cache_store(cfg, key, bar_inputs=bar, rc=rc, nan=nan)
    return RawRun(edeps=None, bar_inputs=bar, n_events=n_events, seed=seed,
                  mode="reverse", seeded_param=None, returncode=rc, nan=nan)


def run_reverse_per_layer(dp: DesignPoint, adjoints: np.ndarray, n_events: int,
                          seed: int, ctrl: Optional[CtrlFlags] = None,
                          use_cache: bool = True,
                          gap_adjoints: Optional[np.ndarray] = None) -> np.ndarray:
    """Reverse-mode run returning the **per-layer** gradient array.

    Same invocation as :func:`run_reverse` (``adjoints`` are per-layer output
    bars, len == n_layers) but parses the ``barInputsPerLayer`` file instead of
    the legacy 3-row ``barInputs``.

    ``gap_adjoints`` (optional, len = n_layers) additionally seeds the per-layer
    **GAP** energy outputs via ``--bar-gap`` (mirrors :func:`run_reverse`). The
    returned per-layer ``barInputsPerLayer`` then reflects d/d(design) of
    ``sum_l(adjoints[l]*E_l + gap_adjoints[l]*E_gap,l)``. Pass ``adjoints``
    all-zero with ``gap_adjoints`` set to differentiate a pure gap-signal
    objective. When ``gap_adjoints`` is None the command line is unchanged.

    Returns the ``(2N+1, 2)`` array with columns ``[mean, var]``:
      * rows ``0 .. N-1``   : d(objective)/d(abs_thick[i])
      * rows ``N .. 2N-1``  : d(objective)/d(gap_thick[i])
      * row  ``2N``         : d(objective)/d(energy)

    Returns ``None`` if the run failed. Uses a cache namespace distinct from
    :func:`run_reverse` (via a ``"__perlayer__"`` marker baked into the key) so
    the two never collide on disk.
    """
    cfg = load_config()
    ctrl = ctrl or ctrl_flags_from_config(cfg)
    binary = cfg["paths"]["reverse_bin"]
    adj = np.asarray(adjoints, dtype=float).ravel()
    if adj.size != dp.n_layers:
        raise ValueError(f"adjoints length {adj.size} != n_layers {dp.n_layers}")
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param=None)
    args += ["-b", ":".join(repr(float(x)) for x in adj)]
    if gap_adjoints is not None:
        gadj = np.asarray(gap_adjoints, dtype=float).ravel()
        if gadj.size != dp.n_layers:
            raise ValueError(f"gap_adjoints length {gadj.size} != n_layers {dp.n_layers}")
        args += ["--bar-gap", ":".join(repr(float(x)) for x in gadj)]
    # Distinct cache namespace so per-layer results never collide with the
    # legacy 3-row run that shares the identical command line.
    key = _flag_hash(binary, ["__perlayer__", *args])

    if use_cache:
        cached = _cache_load(cfg, key)
        if cached is not None:
            arr = cached.get("bar_inputs_per_layer")
            return arr if arr is not None else None

    arr, rc, nan = _execute(binary, args, expect="barInputsPerLayer", seed=seed)
    if use_cache and rc == 0 and arr is not None:
        _cache_store(cfg, key, bar_inputs_per_layer=arr, rc=rc, nan=nan)
    return arr


def _subprocess_timeout_s(cfg: Optional[dict] = None) -> Optional[float]:
    """Per-invocation wall-clock cap for the sim binary (seconds).

    Read from ``sim.subprocess_timeout_s`` in config (default 300s). A finite
    cap turns a stalled/degenerate sim into a low-reliability FAILED run instead
    of an unbounded hang. Set to null/<=0 to disable (legacy behaviour)."""
    try:
        cfg = cfg or load_config()
    except Exception:  # pragma: no cover - config always present in practice
        return 300.0
    val = cfg.get("sim", {}).get("subprocess_timeout_s", 300.0)
    if val is None:
        return None
    val = float(val)
    return val if val > 0 else None


def _execute(binary: str, args: list, expect: str, seed: int):
    """Run the binary in an isolated temp dir and parse the expected output."""
    if not Path(binary).exists():
        raise FileNotFoundError(f"binary not found: {binary} (check config.yaml paths)")
    timeout_s = _subprocess_timeout_s()
    with tempfile.TemporaryDirectory(prefix="hepemshow_agentic_") as wd:
        try:
            proc = subprocess.run([binary, *args], cwd=wd,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # A stalled / degenerate-geometry run: subprocess.run already killed
            # (SIGKILL) the child on timeout. Surface this as the SAME failure
            # shape a bad run uses (None value, nonzero rc, nan=True) so callers
            # treat it as a low-reliability / failed run instead of hanging.
            sys.stderr.write(
                f"[sim] subprocess timed out after {timeout_s}s; "
                f"treating as failed run: {binary}\n")
            return None, -1, True
        rc = proc.returncode
        if expect == "edeps":
            out = Path(wd) / f"edeps_{seed}"
            if not out.exists():
                # some builds name it without seed suffix; fall back to glob
                cand = list(Path(wd).glob("edeps_*"))
                out = cand[0] if cand else out
            if rc != 0 or not out.exists():
                return None, rc, True
            arr = _parse_edeps(out)
            return arr, rc, bool(np.isnan(arr).any())
        elif expect == "edeps_gap":
            out = Path(wd) / f"edeps_gap_{seed}"
            if not out.exists():
                cand = list(Path(wd).glob("edeps_gap_*"))
                out = cand[0] if cand else out
            if rc != 0 or not out.exists():
                return None, rc, True
            arr = _parse_edeps(out)
            return arr, rc, bool(np.isnan(arr).any())
        elif expect == "barInputsPerLayer":
            out = Path(wd) / "barInputsPerLayer"
            if rc != 0 or not out.exists():
                return None, rc, True
            arr = _parse_bar_inputs_per_layer(out)
            return arr, rc, bool(np.isnan(arr).any())
        else:
            out = Path(wd) / "barInputs"
            if rc != 0 or not out.exists():
                return None, rc, True
            arr = _parse_bar_inputs(out)
            return arr, rc, bool(np.isnan(arr).any())


# --------------------------------------------------------------------------- #
# Multi-seed aggregation helpers
# --------------------------------------------------------------------------- #
def forward_profile_multiseed(dp: DesignPoint, seeded_param: str, n_events: int,
                              seeds, ctrl: Optional[CtrlFlags] = None):
    """Average edeps arrays over seeds. Returns (mean_arr, n_total, reliability_inputs).

    mean_arr has shape (n_layers, 4); n_total = n_events * n_ok_seeds.
    """
    runs = [run_forward(dp, seeded_param, n_events, s, ctrl=ctrl) for s in seeds]
    ok = [r for r in runs if r.returncode == 0 and r.edeps is not None and not r.nan]
    if not ok:
        return None, 0, runs
    stack = np.stack([r.edeps for r in ok], axis=0)   # (n_ok, n_layers, 4)
    mean_arr = stack.mean(axis=0)
    n_total = n_events * len(ok)
    return mean_arr, n_total, runs


def reverse_gradient_multiseed(dp: DesignPoint, adjoints: np.ndarray, n_events: int,
                               seeds, ctrl: Optional[CtrlFlags] = None):
    """Average barInputs over seeds. Returns (grad_dict, stderr_dict, n_total)."""
    runs = [run_reverse(dp, adjoints, n_events, s, ctrl=ctrl) for s in seeds]
    ok = [r for r in runs if r.returncode == 0 and r.bar_inputs is not None and not r.nan]
    if not ok:
        return None, None, 0
    means = np.stack([r.bar_inputs[:, 0] for r in ok], axis=0)  # (n_ok, 3)
    vars_ = np.stack([r.bar_inputs[:, 1] for r in ok], axis=0)
    n_total = n_events * len(ok)
    grad = means.mean(axis=0)
    # combine per-event variance across seeds -> SE of the mean
    se = np.sqrt(vars_.mean(axis=0) / n_total)
    keys = ("a", "g", "energy")
    grad_d = {k: float(grad[i]) for i, k in enumerate(keys)}
    se_d = {k: float(se[i]) for i, k in enumerate(keys)}
    return grad_d, se_d, n_total


def reverse_per_layer_gradients_multiseed(dp: DesignPoint, adjoints: np.ndarray,
                                          n_events: int, seeds,
                                          ctrl: Optional[CtrlFlags] = None,
                                          gap_adjoints: Optional[np.ndarray] = None):
    """Average the per-layer reverse gradient over seeds.

    Analogous to :func:`reverse_gradient_multiseed` but for the per-layer
    ``barInputsPerLayer`` output. Returns a dict with per-layer means and
    standard errors (of the mean), or ``None`` if no seed succeeded::

        {
          "absorber_grad":   np.ndarray (N,),   # mean d(obj)/d(abs_thick[i])
          "absorber_stderr": np.ndarray (N,),
          "gap_grad":        np.ndarray (N,),   # mean d(obj)/d(gap_thick[i])
          "gap_stderr":      np.ndarray (N,),
          "energy_grad":     float,             # mean d(obj)/d(energy)
          "energy_stderr":   float,
          "n_total":         int,               # n_events * n_ok_seeds
          "n_layers":        int,
        }
    """
    runs = [run_reverse_per_layer(dp, adjoints, n_events, s, ctrl=ctrl,
                                  gap_adjoints=gap_adjoints) for s in seeds]
    ok = [r for r in runs if r is not None and not bool(np.isnan(r).any())]
    if not ok:
        return None
    N = int(dp.n_layers)
    stack = np.stack(ok, axis=0)            # (n_ok, 2N+1, 2)
    means = stack[:, :, 0]                  # (n_ok, 2N+1)
    vars_ = stack[:, :, 1]                  # (n_ok, 2N+1)
    n_total = n_events * len(ok)
    grad = means.mean(axis=0)              # (2N+1,)
    se = np.sqrt(vars_.mean(axis=0) / n_total)
    return {
        "absorber_grad": grad[0:N].copy(),
        "absorber_stderr": se[0:N].copy(),
        "gap_grad": grad[N:2 * N].copy(),
        "gap_stderr": se[N:2 * N].copy(),
        "energy_grad": float(grad[2 * N]),
        "energy_stderr": float(se[2 * N]),
        "n_total": n_total,
        "n_layers": N,
    }


def region_gradients(regions, per_layer_grads: np.ndarray, n_layers: int) -> dict:
    """Aggregate per-layer gradients into per-region gradients.

    ``per_layer_grads`` is the ``(2N+1, 2)`` array returned by
    :func:`run_reverse_per_layer` (column 0 = mean). Each region's gradient is
    the **sum** of its member layers' per-layer gradient means (half-open
    ``[start, end)``).

    Returns ``{region_index: {"absorber": float, "gap": float}}``.
    """
    N = int(n_layers)
    arr = np.asarray(per_layer_grads, dtype=float)
    abs_g = arr[0:N, 0]
    gap_g = arr[N:2 * N, 0]
    out = {}
    for idx, r in enumerate(regions):
        s, e = int(r.start), int(r.end)
        out[idx] = {
            "absorber": float(abs_g[s:e].sum()),
            "gap": float(gap_g[s:e].sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# CLI / self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    cfg = load_config()
    print(f"[config] {cfg['_source']}")
    dp = default_design_point(cfg)
    ctrl = ctrl_flags_from_config(cfg)
    n = int(cfg["stats"]["dev_events"])
    seed = int(cfg["stats"]["dev_seeds"][0])
    print(f"[design] {dp}")
    print(f"[ctrl]   forward args: {ctrl.to_cli_args()}")

    # IMPORTANT: forward and reverse must use the SAME seed so they operate on
    # identical shower realizations. Both are exact AD of the same quantity
    # (sum_l edep_l), so summed over the same events they must agree to ~machine
    # precision. Using different seeds only agrees in expectation and is swamped
    # by the (very large) event-to-event derivative variance -- that comparison
    # is meaningless at small N and must NOT be used to validate the plumbing.
    print(f"[forward] seeding 'a' (absorber thickness), seed={seed} ...")
    fr = run_forward(dp, "a", n_events=n, seed=seed, ctrl=ctrl, use_cache=False)
    if fr.returncode != 0 or fr.edeps is None:
        print(f"  FAIL rc={fr.returncode}")
        return 1
    total = fr.edeps[:, 0].sum()
    dtotal_da_fwd = fr.edeps[:, 2].sum()
    print(f"  OK  total_edep={total:.1f} MeV   d(total)/da={dtotal_da_fwd:.6g} MeV/mm")

    print(f"[reverse] adjoints = ones (gradient of total_edep), seed={seed} ...")
    adj = np.ones(dp.n_layers)
    rr = run_reverse(dp, adj, n_events=n, seed=seed, ctrl=ctrl, use_cache=False)
    if rr.returncode != 0 or rr.bar_inputs is None:
        print(f"  FAIL rc={rr.returncode}")
        return 1
    ba, bg, be = rr.bar_inputs[:, 0]
    print(f"  OK  d(total)/d[a,g,E] = [{ba:.6g}, {bg:.6g}, {be:.6g}]")

    # Real validation: forward (sum of per-layer dE/da) == reverse barThicknessAbsorber.
    denom = max(abs(dtotal_da_fwd), abs(ba), 1e-12)
    rel = abs(dtotal_da_fwd - ba) / denom
    tol = 1e-3  # generous; same-seed exact AD should match far tighter
    print(f"  cross-check (same seed): forward d(total)/da={dtotal_da_fwd:.6g} vs "
          f"reverse {ba:.6g}  -> rel.diff={rel:.2e} (tol={tol:g})")
    if not np.isfinite(rel) or rel > tol:
        print("[selftest] FAIL: forward/reverse AD disagree beyond tolerance.")
        print("  (If both runs succeeded, check that the two builds share physics "
              "config and that seeds match.)")
        return 1
    print("[selftest] PASS")
    return 0


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HepEmShow differentiable-sim tool layer")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--observable", default=None)
    ap.add_argument("--wrt", default="a", choices=DIFFERENTIABLE_PARAMS)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--energy", type=float, default=None)
    ap.add_argument("-n", "--n-events", type=int, default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.observable:
        from . import observables  # local import to avoid cycle at module load
        cfg = load_config()
        dp = default_design_point(cfg)
        if args.a is not None:
            dp = dp.with_param("a", args.a)
        if args.g is not None:
            dp = dp.with_param("g", args.g)
        if args.energy is not None:
            dp = dp.with_param("energy", args.energy)
        n = args.n_events or int(cfg["stats"]["dev_events"])
        sens = observables.sensitivity(args.observable, args.wrt, dp, n_events=n,
                                       seeds=cfg["stats"]["dev_seeds"])
        print(json.dumps({**asdict(sens), "sign": sens.sign,
                          "log10_abs": sens.log10_abs}, default=str, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""Wave-2 condor validation: prefix-only stop-grad re-anchoring knob.

Validates the knob branch `knob/prefix-anchor` @ 61c7bb5 (hepemshow, built in
build_agent_fwd/ -- do NOT rebuild: the working tree is back on the baseline
branch) against the standing 1M-event FD truth
(experiments/perlayer_adfd_1M_summary.json), the knob-off AD from the same
file, and the wave-1 references (fidelity/wave1_validation_summary.json).
Uniform default design (-l 50 -a 2.3 -g 5.7 -e 10000 -t 400, canonical ctrl
flags from config.yaml), forward mode.

Two configurations, 10 seeds x 20000 events each (400k events total):

    gap_ad_prefix   gap seed (-g 5.7:1),      --stopgrad-prefix-anchor 1
    abs_ad_prefix   absorber seed (-a 2.3:1), --stopgrad-prefix-anchor 1

Every sim call goes through tools.sim.run_forward (temp-dir isolation +
provenance-keyed .sim_cache -> re-running a unit is idempotent); the knob
binary and the 10800 s timeout (wave-1 lesson: 7200 s clipped the slow tail)
are set by in-process config overrides (the load_config() dict is cached, so
mutating it affects THIS process only -- same trick as wave 1; global
config.yaml untouched). Wall time per unit is recorded for the runtime test;
cache-hit reruns produce near-zero wall times and are excluded from runtime
stats in --analyze.

Run one unit (one condor job):
    python -u -m fidelity.wave2_validation --config gap_ad_prefix --seed 1

    Appends one JSONL row (per-layer mean_E/var_E/mean_dE/var_dE, flags,
    binary provenance, wall time) to fidelity/wave2_runs/<config>_s<seed>.jsonl;
    skips the sim if an ok row with the same n_events already exists.

Aggregate after all 20 jobs finish:
    python -u -m fidelity.wave2_validation --analyze
        -> fidelity/wave2_validation_summary.json
    (With zero unit rows present it reports and exits WITHOUT writing a
    placeholder summary file.)

Pre-registered tests (LEDGER.md wave 2; predictions written before the edit):
  1. gap core-window (L5-18) summed AD/FD in [0.8, 1.5] -- strictly between
     knob-1's 0.566 and knob-off's 3.02, and closer to 1 than both.
  2. L49 collapse retained: gap L49 AD/FD within ~3x of 1 (vs knob-off 49).
  3. absorber core-window summed AD/FD in [0.60, 0.80]; report where
     (consistent with knob-off ~0.77 vs lower).
  4. primal spot check: knob-on mean_E per layer vs knob-off 1M mean_E
     (z-scores).
  5. runtime: mean wall time per config vs baseline references
     (gap 4262 s, absorber 5848 s at 20k events) -- within ~15%?
  6. variance: seed-averaged var_dE ratio (on/off) over core L5-18 and the
     L40+ tail -- tail <= 0.3x, gap core in [0.5, 1.2]x (knob-off per-unit
     var_dE from experiments/perlayer_adfd/{gap_ad,absorber_ad}_s*.jsonl,
     failed rows excluded; absorber core reported informationally).

All knob-on errors are seed-scatter SEs (the sim is not bit-reproducible at
fixed seed; PROTOCOL.md). Failed / NaN runs are recorded as failed rows and
surfaced in the summary, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       run_forward, binary_provenance)
from fidelity.wave1_validation import (_load_unit_rows, _seed_scatter,
                                       _window_sum, _ratio, _fd_window,
                                       _knoboff_window_ratio, _knoboff_primal)

# Wave-specific agent build (PROTOCOL.md: shared build/ never moves off
# baseline; agent builds live in build_agent_fwd/). Deliberately not in
# config.yaml; injected via the in-process config override in _wave_config().
# Provenance (path + mtime + size) is recorded per row and is part of the
# sim cache key.
KNOB_BINARY = "/eos/user/j/jeffkrup/agentic/hepemshow/build_agent_fwd/HepEmShow"
KNOB_BRANCH = "knob/prefix-anchor"
KNOB_SHA = "61c7bb5"  # working tree is back on baseline, so the provenance
                      # git rev reads the baseline SHA; this is the truth.

CONFIGS = {
    # name: (param, seeded_param, extra_args)
    "gap_ad_prefix": ("g", "g", ["--stopgrad-prefix-anchor", "1"]),
    "abs_ad_prefix": ("a", "a", ["--stopgrad-prefix-anchor", "1"]),
}

DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_N_EVENTS = 20000
SIM_TIMEOUT_OVERRIDE_S = 10800.0  # wave-1 lesson: 7200 s clipped the slow tail

CORE_WINDOW = (5, 18)   # inclusive layer window for the summed core estimator
TAIL_WINDOW = (40, 49)  # inclusive L40+ tail window for the variance test
L_ANOMALY = 49          # the last-layer gap anomaly

# Baseline knob-off mean wall times at 20k events (LEDGER.md wave-1 addendum).
RUNTIME_REF_S = {"gap_ad_prefix": 4262.0, "abs_ad_prefix": 5848.0}
RUNTIME_MIN_VALID_S = 60.0  # rows faster than this are sim-cache hits

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
OUT_DIR = _HERE / "wave2_runs"
SUMMARY_PATH = _HERE / "wave2_validation_summary.json"
FD_TRUTH_PATH = _REPO / "experiments" / "perlayer_adfd_1M_summary.json"
WAVE1_SUMMARY_PATH = _HERE / "wave1_validation_summary.json"
KNOBOFF_UNIT_GLOBS = {
    "gap_ad_prefix": str(_REPO / "experiments" / "perlayer_adfd" / "gap_ad_s*.jsonl"),
    "abs_ad_prefix": str(_REPO / "experiments" / "perlayer_adfd" / "absorber_ad_s*.jsonl"),
}


def _wave_config() -> dict:
    cfg = load_config()
    cfg.setdefault("sim", {})["subprocess_timeout_s"] = SIM_TIMEOUT_OVERRIDE_S
    cfg["paths"]["forward_bin"] = KNOB_BINARY
    return cfg


def unit_out_path(config: str, seed: int) -> Path:
    return OUT_DIR / f"{config}_s{seed}.jsonl"


def row_exists(path: Path, config: str, seed: int, n_events: int) -> bool:
    if not path.exists():
        return False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("config") == config and r.get("seed") == seed
                    and r.get("n_events") == n_events and r.get("ok")):
                return True
    return False


def run_unit(config: str, seed: int, n_events: int) -> int:
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    param, seeded, extra = CONFIGS[config]
    cfg = _wave_config()
    dp = default_design_point(cfg)
    ctrl = ctrl_flags_from_config(cfg)

    out_path = unit_out_path(config, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if row_exists(out_path, config, seed, n_events):
        print(f"[unit] row exists for ({config}, s{seed}, n={n_events}) -> skip")
        return 0

    print(f"[unit] config={config} seed={seed} n={n_events} "
          f"a={dp.a} g={dp.g} seeded={seeded} extra={extra}")
    t0 = time.monotonic()
    rr = run_forward(dp, seeded, n_events=n_events, seed=seed, ctrl=ctrl,
                     extra_args=extra)
    wall_s = time.monotonic() - t0
    ok = (rr.returncode == 0 and rr.edeps is not None and not rr.nan)
    row = {
        "config": config,
        "param": param,
        "seed": seed,
        "n_events": n_events,
        "a": dp.a,
        "g": dp.g,
        "n_layers": dp.n_layers,
        "seeded_param": seeded,
        "flags": ctrl.to_cli_args(),
        "extra_args": extra,
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "provenance": rr.provenance or binary_provenance(KNOB_BINARY),
        "ok": ok,
        "returncode": rr.returncode,
        "nan": bool(rr.nan),
        "wall_time_s": wall_s,
    }
    if rr.edeps is not None:
        row["mean_E"] = rr.edeps[:, 0].tolist()
        row["var_E"] = rr.edeps[:, 1].tolist()
        row["mean_dE"] = rr.edeps[:, 2].tolist()
        row["var_dE"] = rr.edeps[:, 3].tolist()
    with open(out_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if not ok:
        print(f"[unit] FAILED run (rc={rr.returncode}, nan={rr.nan}); "
              f"recorded as failed row", file=sys.stderr)
        return 1
    d = rr.edeps
    print(f"[unit] OK  total_E={d[:, 0].sum():.2f} MeV"
          f"  total_dE={d[:, 2].sum():.4f}  wall={wall_s:.1f}s -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _knoboff_var_window(pattern: str, n_events: int, lo: int, hi: int):
    """Seed-averaged per-layer var_dE from the standing knob-off unit files,
    summed over [lo, hi]. Failed rows excluded. Returns (window_sum, n_seeds)."""
    rows = [r for r in _load_unit_rows(pattern, n_events)
            if r.get("ok") and "var_dE" in r]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], r)
    if len(by_seed) < 2:
        return None, len(by_seed)
    stack = np.stack([np.asarray(r["var_dE"], dtype=float)
                      for r in by_seed.values()])
    mean_l = stack.mean(axis=0)
    return float(mean_l[lo:hi + 1].sum()), len(by_seed)


def analyze(n_events: int, out_json: Path) -> int:
    if not FD_TRUTH_PATH.exists():
        print(f"[analyze] FATAL: standing FD truth missing: {FD_TRUTH_PATH}",
              file=sys.stderr)
        return 1
    with open(FD_TRUTH_PATH) as fh:
        truth = json.load(fh)
    wave1 = None
    if WAVE1_SUMMARY_PATH.exists():
        with open(WAVE1_SUMMARY_PATH) as fh:
            wave1 = json.load(fh)
    else:
        print(f"[analyze] WARNING: wave-1 summary missing: {WAVE1_SUMMARY_PATH}",
              file=sys.stderr)
    lo, hi = CORE_WINDOW
    tlo, thi = TAIL_WINDOW
    rows = _load_unit_rows(str(OUT_DIR / "*.jsonl"), n_events)
    if not rows:
        print(f"[analyze] no unit rows with n_events={n_events} under "
              f"{OUT_DIR}; nothing to score -- NOT writing a summary file")
        return 0

    summary = {
        "wave": 2,
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "binary": binary_provenance(KNOB_BINARY) if Path(KNOB_BINARY).exists()
                  else {"binary_path": KNOB_BINARY, "missing": True},
        "n_events_per_unit": n_events,
        "expected_seeds": DEFAULT_SEEDS,
        "core_window_layers": [lo, hi],
        "tail_window_layers": [tlo, thi],
        "fd_truth": str(FD_TRUTH_PATH),
        "wave1_reference": str(WAVE1_SUMMARY_PATH) if wave1 else None,
        "note_errors": ("knob-on values: seed-scatter SEs; window sums keep "
                        "inter-layer covariance via per-seed sums. FD window "
                        "sums add per-layer SEs in quadrature (the standing "
                        "summary stores no inter-layer covariance)."),
        "configs": {},
        "tests": {},
    }

    per_cfg = {}
    complete = True
    for cfg_name in CONFIGS:
        mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, cfg_name, "mean_dE")
        missing = sorted(set(DEFAULT_SEEDS) - set(seeds))
        block = {
            "n_ok_seeds": len(seeds),
            "ok_seeds": seeds,
            "missing_seeds": missing,
            "n_failed_rows": n_fail,
        }
        if mean_l is None:
            print(f"[analyze] {cfg_name}: only {len(seeds)} ok unit(s) "
                  f"(need >= 2); missing seeds {missing} -> skipping tests "
                  f"for this config", file=sys.stderr)
            complete = False
            per_cfg[cfg_name] = None
        else:
            if missing:
                print(f"[analyze] {cfg_name}: WARNING missing seeds {missing} "
                      f"(proceeding with {len(seeds)})", file=sys.stderr)
                complete = False
            truth_param = truth["params"]["gap" if cfg_name.startswith("gap") else "absorber"]
            fd = np.asarray(truth_param["per_layer"]["fd"], dtype=float)
            fd_se = np.asarray(truth_param["per_layer"]["fd_se"], dtype=float)
            ratio_l = np.where(fd != 0, mean_l / np.where(fd != 0, fd, 1.0), np.nan)
            rel = np.sqrt(np.where(mean_l != 0,
                                   (se_l / np.where(mean_l != 0, mean_l, 1.0)) ** 2, np.inf)
                          + np.where(fd != 0,
                                     (fd_se / np.where(fd != 0, fd, 1.0)) ** 2, np.inf))
            block["per_layer"] = {
                "ad": mean_l.tolist(),
                "ad_se": se_l.tolist(),
                "ratio_ad_over_fd": ratio_l.tolist(),
                "ratio_se": (np.abs(ratio_l) * rel).tolist(),
            }
            per_cfg[cfg_name] = (mean_l, se_l, stack)
        summary["configs"][cfg_name] = block

    gap_truth = truth["params"]["gap"]
    abs_truth = truth["params"]["absorber"]
    w1_tests = (wave1 or {}).get("tests", {})
    knob1_gap_core = w1_tests.get("1_gap_core_window", {}).get("knob_on_ratio")
    knob1_abs_core = w1_tests.get("3_absorber_core_window", {}).get("knob_on_ratio")

    # -- Test 1: gap core-window summed AD/FD ------------------------------- #
    t1 = {"description": f"gap core-window L{lo}-{hi} summed AD/FD; "
                         f"pre-registered: in [0.8, 1.5], strictly between "
                         f"knob-1 0.566 and knob-off 3.02, closer to 1 than both"}
    off_r, off_r_se = _knoboff_window_ratio(gap_truth, lo, hi)
    t1["knob_off_ratio"] = off_r
    t1["knob_off_ratio_se"] = off_r_se
    t1["wave1_knob1_ratio"] = knob1_gap_core
    if per_cfg.get("gap_ad_prefix") is not None:
        _, _, stack = per_cfg["gap_ad_prefix"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        fd_w, fd_w_se = _fd_window(gap_truth, lo, hi)
        r, r_se = _ratio(ad_w, ad_w_se, fd_w, fd_w_se)
        t1.update({"ad_window": ad_w, "ad_window_se": ad_w_se,
                   "fd_window": fd_w, "fd_window_se": fd_w_se,
                   "knob_on_ratio": r, "knob_on_ratio_se": r_se,
                   "sigma_from_unity": (r - 1.0) / r_se if r_se else None})
        if r is not None:
            t1["in_band_0.8_1.5"] = bool(0.8 <= r <= 1.5)
            if knob1_gap_core is not None and off_r is not None:
                lo_ref, hi_ref = sorted((knob1_gap_core, off_r))
                t1["strictly_between_knob1_and_off"] = bool(lo_ref < r < hi_ref)
                t1["closer_to_1_than_both"] = bool(
                    abs(r - 1.0) < abs(knob1_gap_core - 1.0)
                    and abs(r - 1.0) < abs(off_r - 1.0))
            t1["pass"] = bool(t1.get("in_band_0.8_1.5")
                              and t1.get("strictly_between_knob1_and_off", True)
                              and t1.get("closer_to_1_than_both", True))
    else:
        t1["status"] = "MISSING DATA"
    summary["tests"]["1_gap_core_window"] = t1

    # -- Test 2: L49 gap anomaly -------------------------------------------- #
    t2 = {"description": f"L{L_ANOMALY} gap AD/FD; pre-registered: within ~3x "
                         f"of 1 (vs knob-off 49 +- 11)"}
    t2["knob_off_ratio"] = gap_truth["per_layer"]["ratio_ad_over_fd"][L_ANOMALY]
    t2["knob_off_ratio_se"] = gap_truth["per_layer"]["ratio_se"][L_ANOMALY]
    fd49 = gap_truth["per_layer"]["fd"][L_ANOMALY]
    fd49_se = gap_truth["per_layer"]["fd_se"][L_ANOMALY]
    if per_cfg.get("gap_ad_prefix") is not None:
        mean_l, se_l, _ = per_cfg["gap_ad_prefix"]
        r, r_se = _ratio(float(mean_l[L_ANOMALY]), float(se_l[L_ANOMALY]),
                         fd49, fd49_se)
        t2.update({"ad_L49": float(mean_l[L_ANOMALY]),
                   "ad_L49_se": float(se_l[L_ANOMALY]),
                   "ratio": r, "ratio_se": r_se})
        if r is not None:
            off49 = t2["knob_off_ratio"]
            if r != 0 and off49:
                t2["collapse_factor"] = abs(off49 / r)
            # "within ~3x of 1": |ratio| in [1/3, 3], with the 1-sigma band
            # allowed to reach it (the L49 tail is known heavy).
            t2["within_3x_of_1"] = bool(1.0 / 3.0 <= abs(r) <= 3.0)
            t2["within_3x_of_1_at_1sigma"] = bool(
                abs(r) - (r_se or 0.0) <= 3.0
                and (abs(r) + (r_se or 0.0)) >= 1.0 / 3.0)
            t2["pass"] = t2["within_3x_of_1"]
    else:
        t2["status"] = "MISSING DATA"
    summary["tests"]["2_gap_L49"] = t2

    # -- Test 3: absorber core-window summed AD/FD -------------------------- #
    t3 = {"description": f"absorber core-window L{lo}-{hi} summed AD/FD; "
                         f"pre-registered: in [0.60, 0.80]; report whether "
                         f"consistent with knob-off ~0.77 or lower"}
    off_r, off_r_se = _knoboff_window_ratio(abs_truth, lo, hi)
    t3["knob_off_ratio"] = off_r
    t3["knob_off_ratio_se"] = off_r_se
    t3["wave1_knob1_ratio"] = knob1_abs_core
    if per_cfg.get("abs_ad_prefix") is not None:
        _, _, stack = per_cfg["abs_ad_prefix"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        fd_w, fd_w_se = _fd_window(abs_truth, lo, hi)
        r, r_se = _ratio(ad_w, ad_w_se, fd_w, fd_w_se)
        t3.update({"ad_window": ad_w, "ad_window_se": ad_w_se,
                   "fd_window": fd_w, "fd_window_se": fd_w_se,
                   "knob_on_ratio": r, "knob_on_ratio_se": r_se})
        if r is not None and off_r:
            shift_sigma = (r - off_r) / math.sqrt(r_se ** 2 + off_r_se ** 2)
            t3["shift_sigma_vs_knob_off"] = shift_sigma
            t3["in_band_0.60_0.80"] = bool(0.60 <= r <= 0.80)
            t3["where"] = ("consistent_with_knob_off_0.77"
                           if abs(shift_sigma) < 2.0 else
                           ("below_knob_off" if r < off_r else "above_knob_off"))
            t3["pass"] = t3["in_band_0.60_0.80"]
    else:
        t3["status"] = "MISSING DATA"
    summary["tests"]["3_absorber_core_window"] = t3

    # -- Test 4: primal spot check ------------------------------------------ #
    t4 = {"description": "knob-on mean_E per layer vs knob-off 1M mean_E "
                         "(seed-scatter SEs both sides); |z| summary per config"}
    off_m, off_se, n_off = _knoboff_primal(n_events)
    if off_m is None:
        t4["status"] = "MISSING knob-off primal unit files"
    else:
        t4["n_knob_off_units"] = n_off
        for cfg_name in CONFIGS:
            mean_l, se_l, _, seeds, _ = _seed_scatter(rows, cfg_name, "mean_E")
            if mean_l is None:
                t4[cfg_name] = {"status": "MISSING DATA"}
                continue
            z = (mean_l - off_m) / np.sqrt(se_l ** 2 + off_se ** 2)
            t4[cfg_name] = {
                "max_abs_z": float(np.max(np.abs(z))),
                "argmax_layer": int(np.argmax(np.abs(z))),
                "n_layers_absz_gt3": int(np.sum(np.abs(z) > 3)),
                "mean_abs_z": float(np.mean(np.abs(z))),
            }
    summary["tests"]["4_primal_spot_check"] = t4

    # -- Test 5: runtime ----------------------------------------------------- #
    t5 = {"description": f"mean wall time per config vs baseline references "
                         f"(gap {RUNTIME_REF_S['gap_ad_prefix']:.0f} s, abs "
                         f"{RUNTIME_REF_S['abs_ad_prefix']:.0f} s at 20k); "
                         f"within ~15%? rows with wall < "
                         f"{RUNTIME_MIN_VALID_S:.0f} s excluded (cache hits)"}
    for cfg_name in CONFIGS:
        walls = [r["wall_time_s"] for r in rows
                 if r.get("config") == cfg_name and r.get("ok")
                 and isinstance(r.get("wall_time_s"), (int, float))
                 and r["wall_time_s"] >= RUNTIME_MIN_VALID_S]
        ref = RUNTIME_REF_S[cfg_name]
        entry = {"n_timed_units": len(walls), "reference_s": ref}
        if walls:
            mean_w = float(np.mean(walls))
            entry.update({
                "mean_wall_s": mean_w,
                "se_wall_s": (float(np.std(walls, ddof=1) / math.sqrt(len(walls)))
                              if len(walls) > 1 else None),
                "min_wall_s": float(np.min(walls)),
                "max_wall_s": float(np.max(walls)),
                "ratio_vs_reference": mean_w / ref,
                "within_15pct": bool(abs(mean_w / ref - 1.0) <= 0.15),
            })
        else:
            entry["status"] = "NO TIMED UNITS"
        t5[cfg_name] = entry
    summary["tests"]["5_runtime"] = t5

    # -- Test 6: variance ---------------------------------------------------- #
    t6 = {"description": f"seed-averaged var_dE ratio (knob on/off), core "
                         f"L{lo}-{hi} and tail L{tlo}+; pre-registered: tail "
                         f"<= 0.3x (both configs), gap core in [0.5, 1.2]x; "
                         f"absorber core informational"}
    for cfg_name in CONFIGS:
        entry = {}
        if per_cfg.get(cfg_name) is None:
            entry["status"] = "MISSING DATA"
            t6[cfg_name] = entry
            continue
        on_rows = [r for r in rows if r.get("config") == cfg_name
                   and r.get("ok") and "var_dE" in r]
        by_seed = {}
        for r in on_rows:
            by_seed.setdefault(r["seed"], r)
        on_stack = np.stack([np.asarray(r["var_dE"], dtype=float)
                             for r in by_seed.values()])
        on_l = on_stack.mean(axis=0)
        entry["n_on_seeds"] = len(by_seed)
        for wname, (wlo, whi) in (("core", (lo, hi)), ("tail", (tlo, thi))):
            off_w, n_off_seeds = _knoboff_var_window(
                KNOBOFF_UNIT_GLOBS[cfg_name], n_events, wlo, whi)
            on_w = float(on_l[wlo:whi + 1].sum())
            w = {"on_var_window": on_w, "off_var_window": off_w,
                 "n_off_seeds": n_off_seeds}
            if off_w:
                ratio = on_w / off_w
                w["ratio_on_over_off"] = ratio
                if wname == "tail":
                    w["pass_le_0.3"] = bool(ratio <= 0.3)
                elif cfg_name == "gap_ad_prefix":
                    w["pass_in_0.5_1.2"] = bool(0.5 <= ratio <= 1.2)
                else:
                    w["note"] = "informational (no pre-registered bound)"
            else:
                w["status"] = "MISSING knob-off var_dE"
            entry[wname] = w
        t6[cfg_name] = entry
    summary["tests"]["6_variance"] = t6

    summary["complete"] = complete
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[analyze] units: " + ", ".join(
        f"{c}={summary['configs'][c]['n_ok_seeds']}/{len(DEFAULT_SEEDS)}"
        for c in CONFIGS))
    for name, t in summary["tests"].items():
        print(f"[analyze] {name}: " + json.dumps(
            {k: v for k, v in t.items() if k != "description"}, default=str)[:400])
    print(f"[analyze] complete={complete} -> wrote {out_json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", choices=sorted(CONFIGS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--out-summary", default=str(SUMMARY_PATH))
    args = ap.parse_args(argv)

    if args.analyze:
        return analyze(args.n_events, Path(args.out_summary))
    if args.config is None or args.seed is None:
        ap.error("run mode needs --config and --seed (or use --analyze)")
    return run_unit(args.config, args.seed, args.n_events)


if __name__ == "__main__":
    sys.exit(main())

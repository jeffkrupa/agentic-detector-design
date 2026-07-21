"""Wave-1 condor validation: local-frame stop-grad re-anchoring knob.

Validates the knob branch `knob/local-frame-anchor` @ e9ed89b (hepemshow,
built in build_agent_fwd/ -- do NOT rebuild: the working tree is back on the
baseline branch) against the standing 1M-event FD truth
(experiments/perlayer_adfd_1M_summary.json) and the knob-off AD from the same
file. Uniform default design (-l 50 -a 2.3 -g 5.7 -e 10000 -t 400, canonical
ctrl flags from config.yaml), forward mode.

Three configurations, 10 seeds x 20000 events each (600k events total):

    gap_ad_knob1   gap seed (-g 5.7:1),      --stopgrad-local-frame 1
    abs_ad_knob1   absorber seed (-a 2.3:1), --stopgrad-local-frame 1
    gap_ad_knob2   gap seed (-g 5.7:1),      --last-layer-local 1 (knob 2 ONLY)

Every sim call goes through tools.sim.run_forward (temp-dir isolation +
provenance-keyed .sim_cache -> re-running a unit is idempotent); the knob
binary and the 7200 s timeout are set by in-process config overrides (the
load_config() dict is cached, so mutating it affects THIS process only --
same trick as experiments/perlayer_adfd_1M.py; global config.yaml untouched).

Run one unit (one condor job):
    python -u -m fidelity.wave1_validation --config gap_ad_knob1 --seed 1

    Appends one JSONL row (per-layer mean_E/var_E/mean_dE/var_dE, flags,
    binary provenance) to fidelity/wave1_runs/<config>_s<seed>.jsonl; skips
    the sim if an ok row with the same n_events already exists.

Aggregate after all 30 jobs finish:
    python -u -m fidelity.wave1_validation --analyze
        -> fidelity/wave1_validation_summary.json

Pre-registered tests (LEDGER.md wave 1; predictions written before the edit):
  1. gap core-window (L5-18) summed AD/FD: knob-off ~2.7 -> consistent with 1?
  2. L49 gap AD/FD: knob-off 49 +- 11 -> collapse by >= 10x (knob 1 and
     knob-2-only both)?
  3. absorber core-window summed AD/FD: knob-off 0.754 +- 0.007 -> changes
     little?
  4. primal spot check: knob-on mean_E per layer vs knob-off 1M mean_E.

All knob-on errors are seed-scatter SEs (the sim is not bit-reproducible at
fixed seed; PROTOCOL.md). Failed / NaN runs are recorded as failed rows and
surfaced in the summary, never silently dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       run_forward, binary_provenance)

# Wave-specific agent build (PROTOCOL.md: shared build/ never moves off
# baseline; agent builds live in build_agent_fwd/). Deliberately not in
# config.yaml -- config points at the shared baseline binaries and this
# binary exists only for this wave. Injected via the in-process config
# override in _wave_config(); provenance (path + mtime + size) is recorded
# per row and is part of the sim cache key.
KNOB_BINARY = "/eos/user/j/jeffkrup/agentic/hepemshow/build_agent_fwd/HepEmShow"
KNOB_BRANCH = "knob/local-frame-anchor"
KNOB_SHA = "e9ed89b"  # working tree is back on baseline, so the provenance
                      # git rev reads the baseline SHA; this is the truth.

CONFIGS = {
    # name: (param, seeded_param, extra_args)
    "gap_ad_knob1": ("g", "g", ["--stopgrad-local-frame", "1"]),
    "abs_ad_knob1": ("a", "a", ["--stopgrad-local-frame", "1"]),
    "gap_ad_knob2": ("g", "g", ["--last-layer-local", "1"]),
}

DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_N_EVENTS = 20000
SIM_TIMEOUT_OVERRIDE_S = 7200.0  # in-process only; see module docstring

CORE_WINDOW = (5, 18)   # inclusive layer window for the summed core estimator
L_ANOMALY = 49          # the last-layer gap anomaly

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
OUT_DIR = _HERE / "wave1_runs"
SUMMARY_PATH = _HERE / "wave1_validation_summary.json"
FD_TRUTH_PATH = _REPO / "experiments" / "perlayer_adfd_1M_summary.json"
KNOBOFF_UNIT_GLOB = str(_REPO / "experiments" / "perlayer_adfd" / "*_ad_s*.jsonl")


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
    rr = run_forward(dp, seeded, n_events=n_events, seed=seed, ctrl=ctrl,
                     extra_args=extra)
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
          f"  total_dE={d[:, 2].sum():.4f} -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _load_unit_rows(pattern: str, n_events: int) -> list:
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[analyze] WARNING: bad JSON line in {p}", file=sys.stderr)
                    continue
                if r.get("n_events") == n_events:
                    rows.append(r)
                else:
                    print(f"[analyze] ignoring row with n_events="
                          f"{r.get('n_events')} in {p} (expected {n_events})",
                          file=sys.stderr)
    return rows


def _seed_scatter(rows: list, config: str, key: str):
    """Per-layer mean +- SE over seeds -> (mean_l, se_l, seed_stack, seeds, n_fail).

    SE is the seed-to-seed scatter of the per-seed means (ddof=1), per
    PROTOCOL.md (the sim is not bit-reproducible, so scatter is the honest
    error). Duplicate seed rows keep the first ok row.
    """
    by_seed, n_fail = {}, 0
    for r in rows:
        if r.get("config") != config:
            continue
        if not r.get("ok"):
            n_fail += 1
            continue
        by_seed.setdefault(r["seed"], r)
    if len(by_seed) < 2:
        return None, None, None, sorted(by_seed), n_fail
    seeds = sorted(by_seed)
    stack = np.stack([np.asarray(by_seed[s][key], dtype=float) for s in seeds])
    mean_l = stack.mean(axis=0)
    se_l = stack.std(axis=0, ddof=1) / math.sqrt(len(seeds))
    return mean_l, se_l, stack, seeds, n_fail


def _window_sum(stack: np.ndarray, lo: int, hi: int):
    """Summed window value +- seed-scatter SE (keeps inter-layer covariance)."""
    tot = stack[:, lo:hi + 1].sum(axis=1)
    return float(tot.mean()), float(tot.std(ddof=1) / math.sqrt(len(tot)))


def _ratio(a, a_se, b, b_se):
    if a is None or b is None or b == 0:
        return None, None
    r = a / b
    se = abs(r) * math.sqrt((a_se / a) ** 2 + (b_se / b) ** 2) if a != 0 else abs(b_se / b)
    return float(r), float(se)


def _fd_window(truth_param: dict, lo: int, hi: int):
    """Summed FD over a layer window from the standing truth. The summary has
    per-layer fd/fd_se only, so SEs add in quadrature (inter-layer FD
    covariance not recoverable from the summary; noted in the output)."""
    fd = np.asarray(truth_param["per_layer"]["fd"], dtype=float)
    fd_se = np.asarray(truth_param["per_layer"]["fd_se"], dtype=float)
    return float(fd[lo:hi + 1].sum()), float(np.sqrt((fd_se[lo:hi + 1] ** 2).sum()))


def _knoboff_window_ratio(truth_param: dict, lo: int, hi: int):
    ad = np.asarray(truth_param["per_layer"]["ad"], dtype=float)
    ad_se = np.asarray(truth_param["per_layer"]["ad_se"], dtype=float)
    ad_w = float(ad[lo:hi + 1].sum())
    ad_w_se = float(np.sqrt((ad_se[lo:hi + 1] ** 2).sum()))
    fd_w, fd_w_se = _fd_window(truth_param, lo, hi)
    return _ratio(ad_w, ad_w_se, fd_w, fd_w_se)


def _knoboff_primal(n_events: int):
    """Per-layer knob-off mean_E +- seed-scatter SE from the standing 1M unit
    files (the summary JSON does not store the primal)."""
    rows = _load_unit_rows(KNOBOFF_UNIT_GLOB, n_events)
    by_seed = {}
    for r in rows:
        if r.get("ok") and "mean_E" in r:
            by_seed.setdefault((r["config"], r["seed"]), r)
    if len(by_seed) < 2:
        return None, None, 0
    stack = np.stack([np.asarray(r["mean_E"], dtype=float)
                      for r in by_seed.values()])
    mean_l = stack.mean(axis=0)
    se_l = stack.std(axis=0, ddof=1) / math.sqrt(stack.shape[0])
    return mean_l, se_l, stack.shape[0]


def analyze(n_events: int, out_json: Path) -> int:
    if not FD_TRUTH_PATH.exists():
        print(f"[analyze] FATAL: standing FD truth missing: {FD_TRUTH_PATH}",
              file=sys.stderr)
        return 1
    with open(FD_TRUTH_PATH) as fh:
        truth = json.load(fh)
    lo, hi = CORE_WINDOW
    rows = _load_unit_rows(str(OUT_DIR / "*.jsonl"), n_events)

    summary = {
        "wave": 1,
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "binary": binary_provenance(KNOB_BINARY) if Path(KNOB_BINARY).exists()
                  else {"binary_path": KNOB_BINARY, "missing": True},
        "n_events_per_unit": n_events,
        "expected_seeds": DEFAULT_SEEDS,
        "core_window_layers": [lo, hi],
        "fd_truth": str(FD_TRUTH_PATH),
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

    # -- Test 1: gap core-window summed AD/FD (knob 1) ---------------------- #
    t1 = {"description": f"gap core-window L{lo}-{hi} summed AD/FD, knob 1 on; "
                         f"pre-registered: knob-off ~2.7 -> toward 1"}
    off_r, off_r_se = _knoboff_window_ratio(gap_truth, lo, hi)
    t1["knob_off_ratio"] = off_r
    t1["knob_off_ratio_se"] = off_r_se
    if per_cfg.get("gap_ad_knob1") is not None:
        _, _, stack = per_cfg["gap_ad_knob1"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        fd_w, fd_w_se = _fd_window(gap_truth, lo, hi)
        r, r_se = _ratio(ad_w, ad_w_se, fd_w, fd_w_se)
        t1.update({"ad_window": ad_w, "ad_window_se": ad_w_se,
                   "fd_window": fd_w, "fd_window_se": fd_w_se,
                   "knob_on_ratio": r, "knob_on_ratio_se": r_se,
                   "sigma_from_unity": (r - 1.0) / r_se if r_se else None})
    else:
        t1["status"] = "MISSING DATA"
    summary["tests"]["1_gap_core_window"] = t1

    # -- Test 2: L49 gap anomaly ------------------------------------------- #
    t2 = {"description": f"L{L_ANOMALY} gap AD/FD; pre-registered: knob-off "
                         f"49 +- 11 collapses by >= 10x (knob 1 and knob-2-only)"}
    off49 = gap_truth["per_layer"]["ratio_ad_over_fd"][L_ANOMALY]
    off49_se = gap_truth["per_layer"]["ratio_se"][L_ANOMALY]
    fd49 = gap_truth["per_layer"]["fd"][L_ANOMALY]
    fd49_se = gap_truth["per_layer"]["fd_se"][L_ANOMALY]
    t2["knob_off_ratio"] = off49
    t2["knob_off_ratio_se"] = off49_se
    for cfg_name in ("gap_ad_knob1", "gap_ad_knob2"):
        if per_cfg.get(cfg_name) is None:
            t2[cfg_name] = {"status": "MISSING DATA"}
            continue
        mean_l, se_l, _ = per_cfg[cfg_name]
        r, r_se = _ratio(float(mean_l[L_ANOMALY]), float(se_l[L_ANOMALY]),
                         fd49, fd49_se)
        entry = {"ad_L49": float(mean_l[L_ANOMALY]),
                 "ad_L49_se": float(se_l[L_ANOMALY]),
                 "ratio": r, "ratio_se": r_se}
        if r is not None and r != 0 and off49:
            entry["collapse_factor"] = abs(off49 / r)
            entry["collapsed_10x"] = bool(abs(off49 / r) >= 10.0)
        t2[cfg_name] = entry
    summary["tests"]["2_gap_L49"] = t2

    # -- Test 3: absorber core-window summed AD/FD -------------------------- #
    t3 = {"description": f"absorber core-window L{lo}-{hi} summed AD/FD, knob 1 "
                         f"on; pre-registered: knob-off 0.754 +- 0.007 changes "
                         f"little"}
    off_r, off_r_se = _knoboff_window_ratio(abs_truth, lo, hi)
    t3["knob_off_ratio"] = off_r
    t3["knob_off_ratio_se"] = off_r_se
    if per_cfg.get("abs_ad_knob1") is not None:
        _, _, stack = per_cfg["abs_ad_knob1"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        fd_w, fd_w_se = _fd_window(abs_truth, lo, hi)
        r, r_se = _ratio(ad_w, ad_w_se, fd_w, fd_w_se)
        t3.update({"ad_window": ad_w, "ad_window_se": ad_w_se,
                   "fd_window": fd_w, "fd_window_se": fd_w_se,
                   "knob_on_ratio": r, "knob_on_ratio_se": r_se})
        if r is not None and off_r:
            t3["shift_sigma"] = ((r - off_r) / math.sqrt(r_se ** 2 + off_r_se ** 2))
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

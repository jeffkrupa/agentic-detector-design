"""Wave-9 condor validation: --boundary-dot-cap 80 as the gap-channel severing
replacement + the unsevered absorber MEAN question (ledger wave-10
"validation" items; caps implemented on knob/dot-caps @ fc388aa).

Validates the knob branch `knob/dot-caps` @ fc388aa (hepemshow, built in
build_agent_fwd/ -- do NOT rebuild: the working tree is back on the baseline
branch) against the standing 1M-event FD truth
(experiments/perlayer_adfd_1M_summary.json) and, off-design, against its OWN
large-h FD arms per PROTOCOL.md (unpaired estimator, seed-scatter errors).
All forward mode, lineage-off (--rng-lineage never passed; binary default 0).

Five configurations, 60 jobs / 1.2M events total, 20000 events each:

    gap_b80_val         s1-10  baseline design, unsevered -x 0 -y 0 -B 0,
                               --boundary-dot-cap 80, gap seed (-g 5.7:1)
    abs_unsevered_mean  s1-20  baseline design, unsevered, NO caps,
                               absorber seed (-a 2.3:1)  [heavy tail expected:
                               huge var_dE is the measurement, not a bug]
    gap_b80_od          s1-10  off-design g=3.0, unsevered + b80, -g 3.0:1
    gap_fd_od_plus      s1-10  g=3.1, CANONICAL severed flags (primal-only FD
    gap_fd_od_minus     s1-10  g=2.9,  arm; dot slot energy-seeded per repo
                               convention -- experiments/perlayer_adfd_1M.py)

Every sim call goes through tools.sim.run_forward (temp-dir isolation +
provenance-keyed .sim_cache -> re-running a unit is idempotent); the knob
binary and the 10800 s timeout are set by in-process config overrides (the
load_config() dict is cached, so mutating it affects THIS process only --
same trick as waves 1-2; global config.yaml untouched).

Per-event capture (AD configs only): the run sets HEPEMSHOW_EVENT_DUMP to a
local temp path (env-gated telemetry on the knob branch; primal-identical,
gate G1/G2), then reduces the dump to per-event core-window sums W_e and
layer-sum totals T_e saved as fidelity/wave9val_runs/<config>_s<seed>_events.npz.
This is what makes the pooled SE / median / trimmed-mean comparison against
the wave-8 per-event locators possible (per-seed medians would not be).
A cache-hit rerun cannot regenerate the dump; the row then records
per_event=null and --analyze proceeds on the seeds that have it.

Run one unit (one condor job):
    python -u -m fidelity.wave9_validation --config gap_b80_val --seed 1

    Appends one JSONL row (per-layer mean_E/var_E/mean_dE/var_dE, flags,
    binary provenance, per-event summary, wall time) to
    fidelity/wave9val_runs/<config>_s<seed>.jsonl; skips the sim if an ok
    row with the same n_events already exists.

Aggregate after all 60 jobs finish:
    python -u -m fidelity.wave9_validation --analyze
        -> fidelity/wave9_validation_summary.json
    (With zero unit rows present it reports and exits WITHOUT writing a
    placeholder summary file.)

Pre-registered tests (LEDGER.md wave 9/10; ledger.jsonl wave-10
predicted_signature "validation" items, written before this batch):
  1. gap_b80_val core-window (L5-18) mean vs 1M FD truth 195.8 +- 9.4:
     within 2 sigma?  Also pooled per-event median / trimmed means, the
     variance ratio vs the canonical severed reference (seed-averaged
     per-event var_dE from experiments/perlayer_adfd/gap_ad_s*.jsonl, same
     covariance-free window construction on both sides), and the per-layer
     profile vs truth.
  2. abs_unsevered_mean core mean +- SE, seed-scatter AND pooled per-event
     (both reported), vs truth 2233.7 +- 14.8 -- the decisive
     structural-vs-spike question. Median / tm5% / tm1% reported next to
     the wave-8 locators (median 1430, tm5 1580, tm1 1876, mean 994+-3032).
  3. gap_b80_od core mean vs its own FD: per-layer FD = (E+ - E-)/0.2 from
     the od arms, UNPAIRED, seed-scatter errors (PROTOCOL: CRN pairing is
     worthless here) -- within 2 sigma?
  4. Layer-sum totals for ALL configs (known cap-bias caveat: preview b80
     ensemble total ~27 vs on-design FD-truth total 84.2 +- 14.2 -- the cap
     truncates real net-derivative mass; reported, not hidden).

All errors are seed-scatter SEs unless labelled pooled (the sim is not
bit-reproducible at fixed seed; PROTOCOL.md). Failed / NaN runs are recorded
as failed rows and surfaced in the summary, never silently dropped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       run_forward, binary_provenance)
from fidelity.wave1_validation import (_load_unit_rows, _seed_scatter,
                                       _window_sum, _ratio, _fd_window)

# Wave-specific agent build (PROTOCOL.md: shared build/ never moves off
# baseline; agent builds live in build_agent_fwd/). Deliberately not in
# config.yaml; injected via the in-process config override in _wave_config().
# The working tree is back on the baseline branch (9ceda1e), so the
# provenance git rev reads the baseline SHA; the branch/SHA below are the
# truth for the binary (verified via strings + smoke before submission).
KNOB_BINARY = "/eos/user/j/jeffkrup/agentic/hepemshow/build_agent_fwd/HepEmShow"
KNOB_BRANCH = "knob/dot-caps"
KNOB_SHA = "fc388aa"

# name: (seeded_param, gap_mm_override, severed, extra_args, seeds, per_event)
CONFIGS = {
    "gap_b80_val":        ("g", None, False, ["--boundary-dot-cap", "80"],
                           list(range(1, 11)), True),
    "abs_unsevered_mean": ("a", None, False, [],
                           list(range(1, 21)), True),
    "gap_b80_od":         ("g", 3.0, False, ["--boundary-dot-cap", "80"],
                           list(range(1, 11)), True),
    "gap_fd_od_plus":     ("energy", 3.1, True, [],
                           list(range(1, 11)), False),
    "gap_fd_od_minus":    ("energy", 2.9, True, [],
                           list(range(1, 11)), False),
}
FD_OD_DENOM_MM = 0.2  # (g=3.1) - (g=2.9), PROTOCOL large-h methodology

DEFAULT_N_EVENTS = 20000
SIM_TIMEOUT_OVERRIDE_S = 10800.0  # wave-1 lesson: 7200 s clipped the slow tail

CORE_WINDOW = (5, 18)  # inclusive layer window for the summed core estimator

# Pre-registered reference numbers (recorded here so the summary is
# self-contained; the truth windows are also recomputed from the truth file).
TRUTH_GAP_CORE = (195.8, 9.4)        # 1M FD, L5-18, on-design
TRUTH_ABS_CORE = (2233.7, 14.8)      # 1M FD, L5-18, on-design
TRUTH_GAP_TOTAL = (84.2, 14.2)       # 1M FD, sum L0-49, on-design
WAVE8_ABS_LOCATORS = {               # per-event, LEDGER wave-8 census
    "median": 1430.0, "tm5": 1580.0, "tm1": 1876.0,
    "raw_mean": 994.0, "raw_mean_se": 3032.0,
}
PREVIEW_B80 = {                      # n=2000x2 preview, LEDGER wave-9
    "pooled_core_mean": 207.9, "pooled_core_se": 6.0,
    "median_s1": 184.6, "median_s2": 183.9,
    "canonical_varWe_s1": 2.4e6, "canonical_varWe_s2": 4.5e5,
}

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
OUT_DIR = _HERE / "wave9val_runs"
SUMMARY_PATH = _HERE / "wave9_validation_summary.json"
FD_TRUTH_PATH = _REPO / "experiments" / "perlayer_adfd_1M_summary.json"
CANONICAL_GAP_UNIT_GLOB = str(_REPO / "experiments" / "perlayer_adfd"
                              / "gap_ad_s*.jsonl")


def _wave_config() -> dict:
    cfg = load_config()
    cfg.setdefault("sim", {})["subprocess_timeout_s"] = SIM_TIMEOUT_OVERRIDE_S
    cfg["paths"]["forward_bin"] = KNOB_BINARY
    return cfg


def _wave_ctrl(cfg: dict, severed: bool):
    ctrl = ctrl_flags_from_config(cfg)
    if not severed:
        # Canonical non-severing configuration: -x 0 -y 0 -B 0 with the
        # non-severing regularizers kept (-f 0.2 -N 1e-3 -C 1000).
        ctrl = dataclasses.replace(ctrl, stop_grad_mode=0, grazing_stop_track=0,
                                   backward_boundary_stop=0)
    return ctrl


def unit_out_path(config: str, seed: int) -> Path:
    return OUT_DIR / f"{config}_s{seed}.jsonl"


def events_npz_path(config: str, seed: int) -> Path:
    return OUT_DIR / f"{config}_s{seed}_events.npz"


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


def _parse_event_dump(path: Path, n_layers: int):
    """Wave-3-format event dump -> (W_e core-window dot sums, T_e dot totals).

    One line per event: eventID, sig, tracks, steps, n_layers edeps,
    n_layers dots (same parser as fidelity/wave9_analyze.py)."""
    lo, hi = CORE_WINDOW
    W, T = [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4 + 2 * n_layers:
                continue
            dots = np.asarray(p[4 + n_layers:4 + 2 * n_layers], dtype=float)
            W.append(float(dots[lo:hi + 1].sum()))
            T.append(float(dots.sum()))
    return np.asarray(W), np.asarray(T)


def _per_event_stats(W: np.ndarray, T: np.ndarray) -> dict:
    n = len(W)
    if n < 2:
        return {"n": n}
    s = np.sort(W)
    return {
        "n": n,
        "mean_W": float(W.mean()),
        "se_W": float(W.std(ddof=1) / math.sqrt(n)),
        "median_W": float(np.median(W)),
        "tm5_W": float(s[int(0.05 * n):n - int(0.05 * n)].mean()),
        "tm1_W": float(s[int(0.01 * n):n - int(0.01 * n)].mean()),
        "var_W": float(W.var(ddof=1)),
        "mean_T": float(T.mean()),
        "se_T": float(T.std(ddof=1) / math.sqrt(n)),
        "nonfinite": int((~np.isfinite(W)).sum() + (~np.isfinite(T)).sum()),
    }


def run_unit(config: str, seed: int, n_events: int) -> int:
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    seeded, g_override, severed, extra, _, want_events = CONFIGS[config]
    cfg = _wave_config()
    dp = default_design_point(cfg)
    if g_override is not None:
        dp = dataclasses.replace(dp, g=float(g_override))
    ctrl = _wave_ctrl(cfg, severed)

    out_path = unit_out_path(config, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if row_exists(out_path, config, seed, n_events):
        print(f"[unit] row exists for ({config}, s{seed}, n={n_events}) -> skip")
        return 0

    print(f"[unit] config={config} seed={seed} n={n_events} "
          f"a={dp.a} g={dp.g} seeded={seeded} severed={severed} extra={extra}")
    dump_dir = tempfile.mkdtemp(prefix="wave9val_dump_") if want_events else None
    dump_path = Path(dump_dir) / "dump.txt" if dump_dir else None
    if dump_path is not None:
        os.environ["HEPEMSHOW_EVENT_DUMP"] = str(dump_path)
    try:
        t0 = time.monotonic()
        rr = run_forward(dp, seeded, n_events=n_events, seed=seed, ctrl=ctrl,
                         extra_args=extra)
        wall_s = time.monotonic() - t0
    finally:
        os.environ.pop("HEPEMSHOW_EVENT_DUMP", None)
    ok = (rr.returncode == 0 and rr.edeps is not None and not rr.nan)

    per_event = None
    if want_events and ok:
        if dump_path is not None and dump_path.exists():
            W, T = _parse_event_dump(dump_path, dp.n_layers)
            if len(W) == n_events:
                per_event = _per_event_stats(W, T)
                np.savez_compressed(events_npz_path(config, seed), W=W, T=T)
                per_event["npz"] = events_npz_path(config, seed).name
            else:
                per_event = {"error": f"dump has {len(W)} events, expected "
                                      f"{n_events}; per-event stats dropped"}
        else:
            per_event = {"error": "no dump written (sim-cache hit?); "
                                  "per-event stats unavailable for this unit"}
    if dump_path is not None and dump_path.exists():
        try:
            dump_path.unlink()
            os.rmdir(dump_dir)
        except OSError:
            pass

    row = {
        "config": config,
        "seed": seed,
        "n_events": n_events,
        "a": dp.a,
        "g": dp.g,
        "n_layers": dp.n_layers,
        "seeded_param": seeded,
        "flags": ctrl.to_cli_args(),
        "extra_args": extra,
        "rng_lineage": "off (flag never passed; binary default 0)",
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "provenance": rr.provenance or binary_provenance(KNOB_BINARY),
        "ok": ok,
        "returncode": rr.returncode,
        "nan": bool(rr.nan),
        "wall_time_s": wall_s,
        "per_event": per_event,
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
def _canonical_var_window(n_events: int, lo: int, hi: int):
    """Seed-averaged per-event var_dE from the canonical severed gap units,
    summed over [lo, hi] (covariance-free construction; the knob-on side of
    the ratio uses the same construction). Returns (window_sum, n_seeds)."""
    rows = [r for r in _load_unit_rows(CANONICAL_GAP_UNIT_GLOB, n_events)
            if r.get("ok") and "var_dE" in r]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], r)
    if len(by_seed) < 2:
        return None, len(by_seed)
    stack = np.stack([np.asarray(r["var_dE"], dtype=float)
                      for r in by_seed.values()])
    return float(stack.mean(axis=0)[lo:hi + 1].sum()), len(by_seed)


def _pooled_events(rows: list, config: str):
    """Concatenated per-event (W, T) over ok seeds with an events npz.
    Returns (W, T, seeds_with_events, seeds_without)."""
    with_ev, without = [], []
    Ws, Ts = [], []
    seen = set()
    for r in rows:
        if r.get("config") != config or not r.get("ok"):
            continue
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        p = events_npz_path(config, r["seed"])
        if p.exists():
            with np.load(p) as z:
                Ws.append(z["W"])
                Ts.append(z["T"])
            with_ev.append(r["seed"])
        else:
            without.append(r["seed"])
    if not Ws:
        return None, None, with_ev, sorted(without)
    return np.concatenate(Ws), np.concatenate(Ts), sorted(with_ev), sorted(without)


def _totals_block(rows: list, config: str, key: str):
    """Per-seed layer-sum totals of `key` -> mean +- seed-scatter SE."""
    mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, config, key)
    if mean_l is None:
        return None
    tot = stack.sum(axis=1)
    return {"key": key, "n_seeds": len(seeds),
            "mean": float(tot.mean()),
            "se": float(tot.std(ddof=1) / math.sqrt(len(tot)))}


def _z(a, a_se, b, b_se):
    den = math.sqrt(a_se ** 2 + b_se ** 2)
    return (a - b) / den if den > 0 else None


def analyze(n_events: int, out_json: Path) -> int:
    if not FD_TRUTH_PATH.exists():
        print(f"[analyze] FATAL: standing FD truth missing: {FD_TRUTH_PATH}",
              file=sys.stderr)
        return 1
    with open(FD_TRUTH_PATH) as fh:
        truth = json.load(fh)
    lo, hi = CORE_WINDOW
    rows = _load_unit_rows(str(OUT_DIR / "*.jsonl"), n_events)
    if not rows:
        print(f"[analyze] no unit rows with n_events={n_events} under "
              f"{OUT_DIR}; nothing to score -- NOT writing a summary file")
        return 0

    summary = {
        "wave": 9,
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "binary": binary_provenance(KNOB_BINARY) if Path(KNOB_BINARY).exists()
                  else {"binary_path": KNOB_BINARY, "missing": True},
        "n_events_per_unit": n_events,
        "core_window_layers": [lo, hi],
        "fd_truth": str(FD_TRUTH_PATH),
        "note_errors": ("seed-scatter SEs unless labelled pooled (per-event); "
                        "window sums keep inter-layer covariance via per-seed "
                        "sums; FD truth window SEs add per-layer SEs in "
                        "quadrature. var_dE window ratios use the "
                        "covariance-free per-layer sum on BOTH sides."),
        "configs": {},
        "tests": {},
    }

    per_cfg = {}
    complete = True
    for cfg_name, (_, g_ovr, severed, extra, exp_seeds, want_ev) in CONFIGS.items():
        mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, cfg_name, "mean_dE")
        missing = sorted(set(exp_seeds) - set(seeds))
        walls = [r["wall_time_s"] for r in rows
                 if r.get("config") == cfg_name and r.get("ok")
                 and isinstance(r.get("wall_time_s"), (int, float))
                 and r["wall_time_s"] >= 60.0]
        block = {
            "g_mm": g_ovr if g_ovr is not None else 5.7,
            "severed": severed,
            "extra_args": extra,
            "n_ok_seeds": len(seeds),
            "ok_seeds": seeds,
            "missing_seeds": missing,
            "n_failed_rows": n_fail,
            "mean_wall_s": float(np.mean(walls)) if walls else None,
        }
        if mean_l is None:
            print(f"[analyze] {cfg_name}: only {len(seeds)} ok unit(s) "
                  f"(need >= 2); missing {missing} -> config skipped in tests",
                  file=sys.stderr)
            complete = False
            per_cfg[cfg_name] = None
        else:
            if missing:
                print(f"[analyze] {cfg_name}: WARNING missing seeds {missing} "
                      f"(proceeding with {len(seeds)})", file=sys.stderr)
                complete = False
            per_cfg[cfg_name] = (mean_l, se_l, stack)
        summary["configs"][cfg_name] = block

    gap_truth = truth["params"]["gap"]
    abs_truth = truth["params"]["absorber"]

    # -- Test 1: gap_b80_val core window vs 1M FD truth ---------------------- #
    t1 = {"description": f"gap_b80_val core L{lo}-{hi} mean vs 1M FD truth "
                         f"{TRUTH_GAP_CORE[0]} +- {TRUTH_GAP_CORE[1]}; "
                         f"pre-registered: within 2 sigma. Plus pooled "
                         f"per-event median/trimmed, var ratio vs canonical "
                         f"severed (covariance-free var_dE window, both "
                         f"sides), per-layer profile.",
          "preview_reference": PREVIEW_B80}
    fd_w, fd_w_se = _fd_window(gap_truth, lo, hi)
    t1["fd_truth_window"] = [fd_w, fd_w_se]
    if per_cfg.get("gap_b80_val") is not None:
        mean_l, se_l, stack = per_cfg["gap_b80_val"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        z = _z(ad_w, ad_w_se, fd_w, fd_w_se)
        t1.update({"ad_core_mean": ad_w, "ad_core_se": ad_w_se,
                   "z_vs_truth": z, "pass_within_2sigma": bool(abs(z) <= 2.0),
                   "per_seed_core_median": float(np.median(stack[:, lo:hi + 1].sum(axis=1)))})
        W, T, with_ev, without = _pooled_events(rows, "gap_b80_val")
        if W is not None:
            pe = _per_event_stats(W, T)
            pe["seeds_with_events"] = with_ev
            pe["seeds_without_events"] = without
            pe["z_pooled_vs_truth"] = _z(pe["mean_W"], pe["se_W"], fd_w, fd_w_se)
            t1["pooled_per_event"] = pe
        else:
            t1["pooled_per_event"] = {"status": "NO EVENT NPZ FILES",
                                      "seeds_without_events": without}
        # variance ratio vs canonical severed, same construction both sides
        on_rows = [r for r in rows if r.get("config") == "gap_b80_val"
                   and r.get("ok") and "var_dE" in r]
        by_seed = {}
        for r in on_rows:
            by_seed.setdefault(r["seed"], r)
        on_var = float(np.stack([np.asarray(r["var_dE"], dtype=float)
                                 for r in by_seed.values()]
                                ).mean(axis=0)[lo:hi + 1].sum())
        off_var, n_off = _canonical_var_window(n_events, lo, hi)
        t1["var_window"] = {"on": on_var, "off_canonical": off_var,
                            "n_canonical_seeds": n_off,
                            "ratio_on_over_off": (on_var / off_var)
                                                 if off_var else None}
        fd = np.asarray(gap_truth["per_layer"]["fd"], dtype=float)
        fd_se = np.asarray(gap_truth["per_layer"]["fd_se"], dtype=float)
        ratio_l = np.where(fd != 0, mean_l / np.where(fd != 0, fd, 1.0), np.nan)
        t1["per_layer"] = {"ad": mean_l.tolist(), "ad_se": se_l.tolist(),
                           "fd_truth": fd.tolist(), "fd_truth_se": fd_se.tolist(),
                           "ratio_ad_over_fd": ratio_l.tolist()}
    else:
        t1["status"] = "MISSING DATA"
    summary["tests"]["1_gap_b80_core_vs_truth"] = t1

    # -- Test 2: unsevered absorber MEAN ------------------------------------- #
    t2 = {"description": f"abs_unsevered_mean core L{lo}-{hi} mean vs truth "
                         f"{TRUTH_ABS_CORE[0]} +- {TRUTH_ABS_CORE[1]}; "
                         f"seed-scatter AND pooled SEs both reported; "
                         f"median/trimmed vs the wave-8 per-event locators. "
                         f"Decides structural-missing-mass vs spike-taming.",
          "wave8_locators": WAVE8_ABS_LOCATORS}
    fd_w_a, fd_w_a_se = _fd_window(abs_truth, lo, hi)
    t2["fd_truth_window"] = [fd_w_a, fd_w_a_se]
    if per_cfg.get("abs_unsevered_mean") is not None:
        _, _, stack = per_cfg["abs_unsevered_mean"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        z = _z(ad_w, ad_w_se, fd_w_a, fd_w_a_se)
        t2.update({"core_mean_seed_scatter": ad_w, "core_se_seed_scatter": ad_w_se,
                   "z_seed_scatter_vs_truth": z,
                   "per_seed_core_means": stack[:, lo:hi + 1].sum(axis=1).tolist()})
        W, T, with_ev, without = _pooled_events(rows, "abs_unsevered_mean")
        if W is not None:
            pe = _per_event_stats(W, T)
            pe["seeds_with_events"] = with_ev
            pe["seeds_without_events"] = without
            zp = _z(pe["mean_W"], pe["se_W"], fd_w_a, fd_w_a_se)
            pe["z_pooled_vs_truth"] = zp
            t2["pooled_per_event"] = pe
            zdec = zp if zp is not None else z
        else:
            t2["pooled_per_event"] = {"status": "NO EVENT NPZ FILES",
                                      "seeds_without_events": without}
            zdec = z
        if zdec is not None:
            t2["verdict"] = ("mean_on_truth_spike_taming_problem"
                             if abs(zdec) <= 2.0 else
                             ("significantly_low_structural_missing_mass"
                              if zdec < 0 else "significantly_high"))
    else:
        t2["status"] = "MISSING DATA"
    summary["tests"]["2_abs_unsevered_mean"] = t2

    # -- Test 3: off-design gap b80 vs its own FD ---------------------------- #
    t3 = {"description": f"gap_b80_od (g=3.0) core L{lo}-{hi} mean vs own FD "
                         f"(E+ - E-)/{FD_OD_DENOM_MM} from g=3.1/2.9 arms, "
                         f"UNPAIRED, seed-scatter errors; pre-registered: "
                         f"within 2 sigma."}
    have_arms = all(per_cfg.get(c) is not None
                    for c in ("gap_fd_od_plus", "gap_fd_od_minus"))
    if have_arms:
        parts = {}
        for c in ("gap_fd_od_plus", "gap_fd_od_minus"):
            m_l, s_l, stk, seeds, _ = _seed_scatter(rows, c, "mean_E")
            w, w_se = _window_sum(stk, lo, hi)
            tot = stk.sum(axis=1)
            parts[c] = (m_l, s_l, w, w_se,
                        float(tot.mean()),
                        float(tot.std(ddof=1) / math.sqrt(len(tot))))
        (mp, sp, wp, wp_se, tp, tp_se) = parts["gap_fd_od_plus"]
        (mm, sm, wm, wm_se, tm, tm_se) = parts["gap_fd_od_minus"]
        fd_core = (wp - wm) / FD_OD_DENOM_MM
        fd_core_se = math.sqrt(wp_se ** 2 + wm_se ** 2) / FD_OD_DENOM_MM
        fd_l = (mp - mm) / FD_OD_DENOM_MM
        fd_l_se = np.sqrt(sp ** 2 + sm ** 2) / FD_OD_DENOM_MM
        t3["fd_core"] = fd_core
        t3["fd_core_se"] = fd_core_se
        t3["fd_total"] = (tp - tm) / FD_OD_DENOM_MM
        t3["fd_total_se"] = math.sqrt(tp_se ** 2 + tm_se ** 2) / FD_OD_DENOM_MM
        t3["fd_per_layer"] = fd_l.tolist()
        t3["fd_per_layer_se"] = fd_l_se.tolist()
    else:
        t3["fd_status"] = "MISSING FD ARM DATA"
    if per_cfg.get("gap_b80_od") is not None and have_arms:
        mean_l, se_l, stack = per_cfg["gap_b80_od"]
        ad_w, ad_w_se = _window_sum(stack, lo, hi)
        z = _z(ad_w, ad_w_se, fd_core, fd_core_se)
        ratio_l = np.where(fd_l != 0, mean_l / np.where(fd_l != 0, fd_l, 1.0),
                           np.nan)
        t3.update({"ad_core_mean": ad_w, "ad_core_se": ad_w_se,
                   "z_vs_own_fd": z, "pass_within_2sigma": bool(abs(z) <= 2.0),
                   "per_layer_ad": mean_l.tolist(),
                   "per_layer_ad_se": se_l.tolist(),
                   "per_layer_ratio_ad_over_fd": ratio_l.tolist()})
        W, T, with_ev, without = _pooled_events(rows, "gap_b80_od")
        if W is not None:
            pe = _per_event_stats(W, T)
            pe["seeds_with_events"] = with_ev
            pe["seeds_without_events"] = without
            t3["pooled_per_event"] = pe
    elif per_cfg.get("gap_b80_od") is None:
        t3["status"] = "MISSING AD DATA"
    summary["tests"]["3_gap_b80_offdesign_vs_own_fd"] = t3

    # -- Test 4: layer-sum totals (cap-bias caveat) -------------------------- #
    t4 = {"description": "layer-sum (L0-49) totals for all configs; known "
                         "cap-bias caveat: the boundary cap truncates real "
                         "net-derivative mass (preview b80 total ~27 vs "
                         "on-design FD-truth total "
                         f"{TRUTH_GAP_TOTAL[0]} +- {TRUTH_GAP_TOTAL[1]}). "
                         "Reported, not headlined (PROTOCOL: totals are "
                         "cancellation-/tail-dominated).",
          "fd_truth_gap_total_on_design": list(TRUTH_GAP_TOTAL)}
    for cfg_name in CONFIGS:
        key = "mean_E" if cfg_name.startswith("gap_fd_od") else "mean_dE"
        t4[cfg_name] = _totals_block(rows, cfg_name, key) or {"status": "MISSING"}
    if "fd_total" in t3:
        t4["fd_od_total_derivative"] = [t3["fd_total"], t3["fd_total_se"]]
    summary["tests"]["4_layer_sum_totals"] = t4

    summary["complete"] = complete
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[analyze] units: " + ", ".join(
        f"{c}={summary['configs'][c]['n_ok_seeds']}/{len(CONFIGS[c][4])}"
        for c in CONFIGS))
    for name, t in summary["tests"].items():
        print(f"[analyze] {name}: " + json.dumps(
            {k: v for k, v in t.items()
             if k not in ("description", "per_layer", "fd_per_layer",
                          "fd_per_layer_se", "per_layer_ad", "per_layer_ad_se",
                          "per_layer_ratio_ad_over_fd")},
            default=str)[:500])
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

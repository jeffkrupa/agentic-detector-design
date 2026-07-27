"""Wave-13 condor validation: absorber E_cap ~ a scaling-rule CONFIRMATION.

Wave-12 (LOCAL, n=2000) found the wave-11 off-design governed-absorber excess
(a=3.0, `--track-dot-cap 1000:50`: core L5-18 = 2213.83 +- 47.18 vs own-FD
truth 2018.05 +- 15.36, z = 3.95, +9.7%) behaves like a clamp-scale mismatch,
with the E clamp the controlling axis (E-axis median monotone, 5.2 sigma
slope) and a measured exponent k ~= 1, i.e. the rule **E_cap = 50 * a / 2.3 at
fixed P = 1000** (E propto a). The mean-level confirmation was beyond local
reach (pooled SEs 300-730 MeV at 8k events vs the ~60 needed).

This wave is the pre-registered confirming batch (ledger.jsonl wave-13; the
wave-12 LEDGER recommendation): three governed geometries at their scaled caps
plus a fresh a=2.0 FD bracket, at condor scale (15/15/10 seeds x 20k), against
the reused a=3.0 FD truth and a fresh a=2.0 FD truth. All forward mode,
lineage-off (`--rng-lineage` never passed; binary default 0). Same gov binary
`knob/dot-governor` @ 3b6d45f (build_gov_fwd in the -gov worktree, verified by
strings + a 10-event smoke with `--track-dot-cap 1000:65.2`; mtime/size
re-checked before submission -- never rebuilt).

Five configurations, 60 jobs / 1.2M events total, 20000 events each:

    abs_a30_scaled    s1-15  a=3.00 g=5.70, unsevered -x 0 -y 0 -B 0
                             (-f 0.2 -N 1e-3 -C 1000), --track-dot-cap
                             1000:65.2 (= 50*3.0/2.3), absorber seed -a 3.0:1
    abs_a20_scaled    s1-15  a=2.00 g=5.70, unsevered + governor
                             --track-dot-cap 1000:43.5 (= 50*2.0/2.3),
                             absorber seed -a 2.0:1
    abs_a20_ctrl      s1-10  a=2.00, unsevered + UNSCALED control governor
                             --track-dot-cap 1000:50, absorber seed -a 2.0:1
    abs_fd_a20_plus   s1-10  a=2.1, CANONICAL severed flags (primal-only FD
    abs_fd_a20_minus  s1-10  a=1.9,  arm; dot slot energy-seeded per repo
                             convention -- experiments/perlayer_adfd_1M.py;
                             h=0.1 each side => denom (a=2.1)-(a=1.9)=0.2)

The a=3.0 FD truth (2018.05 +- 15.36) already exists from the wave-11
abs_fd_od arms; the a=2.3 on-design 1.00 anchor already exists (wave-11
abs_gov_val). Neither is rerun here.

Every sim call goes through tools.sim.run_forward (temp-dir isolation +
provenance-keyed .sim_cache -> re-running a unit is idempotent); the gov
binary and the 10800 s timeout are set by in-process config overrides (the
load_config() dict is cached, so mutating it affects THIS process only; global
config.yaml untouched). Per-event capture (governed AD configs only) mirrors
wave-11: HEPEMSHOW_EVENT_DUMP -> local temp path, reduced to per-event
core-window sums W_e and layer-sum totals T_e in
fidelity/wave13_runs/<config>_s<seed>_events.npz. A cache-hit rerun cannot
regenerate the dump; the row then records per_event=null and --analyze
proceeds on the seeds that have it.

Run one unit (one condor job):
    python -u -m fidelity.wave13_scaleval --config abs_a30_scaled --seed 1

Aggregate after all jobs finish:
    python -u -m fidelity.wave13_scaleval --analyze
        -> fidelity/wave13_scaleval_summary.json
    (With zero unit rows present it reports and exits WITHOUT writing a
    placeholder summary file.)

Pre-registered tests (ledger.jsonl wave-13 predicted_signature):
  1. abs_a30_scaled core L5-18 mean vs the a=3.0 FD truth 2018.05 +- 15.36:
     within 2 sigma? (unscaled E50 was +9.7%, z = 3.95). Seed-scatter AND
     pooled per-event SEs; median / tm5.
  2. abs_a20_scaled core mean vs its own FD ((E+ - E-)/0.2 from the a=2.1/1.9
     arms, UNPAIRED, seed-scatter): within 2 sigma? The a=2.0 FD truth is
     itself a reported result.
  3. abs_a20_ctrl (UNSCALED E50) core mean vs the same a=2.0 FD: is it OFF
     (rule load-bearing, not a null)? Report the scaled-vs-ctrl delta.
  4. Layer-sum totals for all configs + the known governor totals-bias.

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
                                       _window_sum, _ratio)
from fidelity.wave11_absval import (_parse_event_dump, _per_event_stats)

# Wave-specific gov-worktree build (PROTOCOL.md: shared build/ never moves off
# baseline; the governor build lives in the -gov worktree). Injected via the
# in-process config override in _wave_config(); branch/SHA verified by strings
# + a 10-event smoke with --track-dot-cap 1000:65.2 before submission.
KNOB_BINARY = ("/eos/user/j/jeffkrup/agentic/hepemshow-gov/"
               "build_gov_fwd/HepEmShow")
KNOB_BRANCH = "knob/dot-governor"
KNOB_SHA = "3b6d45f"

# Scaling rule under test: E_cap = 50 * a / 2.3 at fixed P = 1000 (E propto a).
# String forms match the pre-registered ledger spec exactly.
CAP_A30 = ["--track-dot-cap", "1000:65.2"]   # 50*3.0/2.3 = 65.217
CAP_A20 = ["--track-dot-cap", "1000:43.5"]   # 50*2.0/2.3 = 43.478
CAP_E50 = ["--track-dot-cap", "1000:50"]     # unscaled control

# name: (seeded_param, a_mm_override, severed, extra_args, seeds, per_event)
CONFIGS = {
    "abs_a30_scaled":   ("a", 3.0, False, CAP_A30, list(range(1, 16)), True),
    "abs_a20_scaled":   ("a", 2.0, False, CAP_A20, list(range(1, 16)), True),
    "abs_a20_ctrl":     ("a", 2.0, False, CAP_E50, list(range(1, 11)), True),
    "abs_fd_a20_plus":  ("energy", 2.1, True, [],   list(range(1, 11)), False),
    "abs_fd_a20_minus": ("energy", 1.9, True, [],   list(range(1, 11)), False),
}
FD_A20_DENOM_MM = 0.2  # (a=2.1) - (a=1.9), PROTOCOL large-h methodology

DEFAULT_N_EVENTS = 20000
SIM_TIMEOUT_OVERRIDE_S = 10800.0  # wave-1 lesson: 7200 s clipped the slow tail

CORE_WINDOW = (5, 18)  # inclusive layer window for the summed core estimator

# Reused a=3.0 FD truth (wave-11 abs_fd_od arms; wave-12 od_fd_truth_core) and
# the unscaled a=3.0 reference (wave-11/12 P1000:E50 at a=3.0) for context.
TRUTH_A30_CORE = (2018.0495457018242, 15.362877569225724)
UNSCALED_A30_REF = {"core_mean": 2213.8297082818126, "core_se": 47.1822602449957,
                    "z_vs_truth": 3.95, "pct_excess": 9.7,
                    "note": "wave-11/12 a=3.0 --track-dot-cap 1000:50 (E50, unscaled)"}
TRUTH_ABS_TOTAL_ON_DESIGN = (1050.8, 20.2)  # 1M FD, sum L0-49, a=2.3

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
OUT_DIR = _HERE / "wave13_runs"
SUMMARY_PATH = _HERE / "wave13_scaleval_summary.json"


def _wave_config() -> dict:
    cfg = load_config()
    cfg.setdefault("sim", {})["subprocess_timeout_s"] = SIM_TIMEOUT_OVERRIDE_S
    cfg["paths"]["forward_bin"] = KNOB_BINARY
    return cfg


def _wave_ctrl(cfg: dict, severed: bool):
    ctrl = ctrl_flags_from_config(cfg)
    if not severed:
        # Unsevered governed configuration: -x 0 -y 0 -B 0 with the
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


def run_unit(config: str, seed: int, n_events: int) -> int:
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    seeded, a_override, severed, extra, _, want_events = CONFIGS[config]
    cfg = _wave_config()
    dp = default_design_point(cfg)
    if a_override is not None:
        dp = dataclasses.replace(dp, a=float(a_override))
    ctrl = _wave_ctrl(cfg, severed)

    out_path = unit_out_path(config, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if row_exists(out_path, config, seed, n_events):
        print(f"[unit] row exists for ({config}, s{seed}, n={n_events}) -> skip")
        return 0

    print(f"[unit] config={config} seed={seed} n={n_events} "
          f"a={dp.a} g={dp.g} seeded={seeded} severed={severed} extra={extra}")
    dump_dir = tempfile.mkdtemp(prefix="wave13sv_dump_") if want_events else None
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


def _fd_a20(rows: list, lo: int, hi: int):
    """Fresh a=2.0 FD truth from the a=2.1/1.9 primal arms (unpaired,
    seed-scatter). Returns dict with core/total central FD and per-layer FD,
    or None if either arm has < 2 ok seeds."""
    parts = {}
    for c in ("abs_fd_a20_plus", "abs_fd_a20_minus"):
        m_l, s_l, stk, seeds, _ = _seed_scatter(rows, c, "mean_E")
        if m_l is None:
            return None
        w, w_se = _window_sum(stk, lo, hi)
        tot = stk.sum(axis=1)
        parts[c] = (m_l, s_l, w, w_se, float(tot.mean()),
                    float(tot.std(ddof=1) / math.sqrt(len(tot))), seeds)
    (mp, sp, wp, wp_se, tp, tp_se, sp_seeds) = parts["abs_fd_a20_plus"]
    (mm, sm, wm, wm_se, tm, tm_se, sm_seeds) = parts["abs_fd_a20_minus"]
    fd_core = (wp - wm) / FD_A20_DENOM_MM
    fd_core_se = math.sqrt(wp_se ** 2 + wm_se ** 2) / FD_A20_DENOM_MM
    fd_total = (tp - tm) / FD_A20_DENOM_MM
    fd_total_se = math.sqrt(tp_se ** 2 + tm_se ** 2) / FD_A20_DENOM_MM
    fd_l = (mp - mm) / FD_A20_DENOM_MM
    fd_l_se = np.sqrt(sp ** 2 + sm ** 2) / FD_A20_DENOM_MM
    return {"fd_core": fd_core, "fd_core_se": fd_core_se,
            "fd_total": fd_total, "fd_total_se": fd_total_se,
            "fd_per_layer": fd_l.tolist(), "fd_per_layer_se": fd_l_se.tolist(),
            "denom_mm": FD_A20_DENOM_MM,
            "plus_seeds": sp_seeds, "minus_seeds": sm_seeds,
            "plus_core": [wp, wp_se], "minus_core": [wm, wm_se]}


def _governed_core(rows: list, config: str, lo: int, hi: int):
    """Seed-scatter and pooled per-event core-window estimators for a governed
    config. Returns (block dict, seed-scatter (mean, se), pooled (mean, se))."""
    mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, config, "mean_dE")
    if mean_l is None:
        return None, None, None
    ad_w, ad_w_se = _window_sum(stack, lo, hi)
    block = {"core_mean_seed_scatter": ad_w, "core_se_seed_scatter": ad_w_se,
             "per_seed_core_means": stack[:, lo:hi + 1].sum(axis=1).tolist(),
             "n_ok_seeds": len(seeds)}
    pooled = None
    W, T, with_ev, without = _pooled_events(rows, config)
    if W is not None:
        pe = _per_event_stats(W, T)
        pe["seeds_with_events"] = with_ev
        pe["seeds_without_events"] = without
        block["pooled_per_event"] = pe
        pooled = (pe["mean_W"], pe["se_W"])
    else:
        block["pooled_per_event"] = {"status": "NO EVENT NPZ FILES",
                                     "seeds_without_events": without}
    return block, (ad_w, ad_w_se), pooled


def analyze(n_events: int, out_json: Path) -> int:
    lo, hi = CORE_WINDOW
    rows = _load_unit_rows(str(OUT_DIR / "*.jsonl"), n_events)
    if not rows:
        print(f"[analyze] no unit rows with n_events={n_events} under "
              f"{OUT_DIR}; nothing to score -- NOT writing a summary file")
        return 0

    summary = {
        "wave": 13,
        "item": "absorber E_cap ~ a scaling-rule confirmation (3-geometry batch)",
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "rule": "E_cap = 50 * a / 2.3 at fixed P = 1000 (E propto a)",
        "caps": {"abs_a30_scaled": " ".join(CAP_A30),
                 "abs_a20_scaled": " ".join(CAP_A20),
                 "abs_a20_ctrl": " ".join(CAP_E50)},
        "binary": binary_provenance(KNOB_BINARY) if Path(KNOB_BINARY).exists()
                  else {"binary_path": KNOB_BINARY, "missing": True},
        "n_events_per_unit": n_events,
        "core_window_layers": [lo, hi],
        "truth_a30_core": list(TRUTH_A30_CORE),
        "unscaled_a30_reference": UNSCALED_A30_REF,
        "note_errors": ("seed-scatter SEs unless labelled pooled (per-event); "
                        "window sums keep inter-layer covariance via per-seed "
                        "sums; a=2.0 FD is unpaired (E+ - E-)/0.2 from the "
                        "a=2.1/1.9 primal arms, seed-scatter errors."),
        "configs": {},
        "tests": {},
    }

    complete = True
    for cfg_name, (_, a_ovr, severed, extra, exp_seeds, _) in CONFIGS.items():
        mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, cfg_name, "mean_dE"
                                                           if not severed else "mean_E")
        missing = sorted(set(exp_seeds) - set(seeds))
        walls = [r["wall_time_s"] for r in rows
                 if r.get("config") == cfg_name and r.get("ok")
                 and isinstance(r.get("wall_time_s"), (int, float))
                 and r["wall_time_s"] >= 60.0]
        summary["configs"][cfg_name] = {
            "a_mm": a_ovr if a_ovr is not None else 2.3,
            "severed": severed,
            "extra_args": extra,
            "n_ok_seeds": len(seeds),
            "ok_seeds": seeds,
            "missing_seeds": missing,
            "n_failed_rows": n_fail,
            "mean_wall_s": float(np.mean(walls)) if walls else None,
        }
        if missing or len(seeds) < 2:
            complete = False

    # -- a=2.0 FD truth from the fresh arms ---------------------------------- #
    fd20 = _fd_a20(rows, lo, hi)

    # -- Test 1: abs_a30_scaled core vs the reused a=3.0 FD truth ------------- #
    t1 = {"description": f"abs_a30_scaled (a=3.0, cap {' '.join(CAP_A30)}) core "
                         f"L{lo}-{hi} mean vs a=3.0 FD truth "
                         f"{TRUTH_A30_CORE[0]:.2f} +- {TRUTH_A30_CORE[1]:.2f}; "
                         f"pre-registered: within 2 sigma (unscaled E50 was "
                         f"+9.7%, z=3.95). Seed-scatter AND pooled per-event.",
          "truth": list(TRUTH_A30_CORE),
          "unscaled_reference": UNSCALED_A30_REF}
    blk, ss, pooled = _governed_core(rows, "abs_a30_scaled", lo, hi)
    if blk is not None:
        t1.update(blk)
        z_ss = _z(ss[0], ss[1], TRUTH_A30_CORE[0], TRUTH_A30_CORE[1])
        t1["z_seed_scatter_vs_truth"] = z_ss
        t1["pass_within_2sigma_seed_scatter"] = bool(abs(z_ss) <= 2.0)
        if pooled is not None:
            z_p = _z(pooled[0], pooled[1], TRUTH_A30_CORE[0], TRUTH_A30_CORE[1])
            t1["z_pooled_vs_truth"] = z_p
            t1["pass_within_2sigma_pooled"] = bool(abs(z_p) <= 2.0)
    else:
        t1["status"] = "MISSING DATA"
    summary["tests"]["1_abs_a30_scaled_vs_truth"] = t1

    # -- Test 2: abs_a20_scaled core vs its own fresh FD --------------------- #
    t2 = {"description": f"abs_a20_scaled (a=2.0, cap {' '.join(CAP_A20)}) core "
                         f"L{lo}-{hi} mean vs its own FD ((E+ - E-)/"
                         f"{FD_A20_DENOM_MM} from a=2.1/1.9 arms, UNPAIRED, "
                         f"seed-scatter); pre-registered: within 2 sigma. The "
                         f"a=2.0 FD truth is itself a reported result."}
    if fd20 is not None:
        t2["fd_a20_core"] = [fd20["fd_core"], fd20["fd_core_se"]]
        t2["fd_a20_arms"] = {"plus_seeds": fd20["plus_seeds"],
                             "minus_seeds": fd20["minus_seeds"],
                             "plus_core": fd20["plus_core"],
                             "minus_core": fd20["minus_core"]}
    else:
        t2["fd_status"] = "MISSING FD ARM DATA"
    blk2, ss2, pooled2 = _governed_core(rows, "abs_a20_scaled", lo, hi)
    if blk2 is not None:
        t2.update(blk2)
        if fd20 is not None:
            z_ss = _z(ss2[0], ss2[1], fd20["fd_core"], fd20["fd_core_se"])
            t2["z_seed_scatter_vs_fd"] = z_ss
            t2["pass_within_2sigma_seed_scatter"] = bool(abs(z_ss) <= 2.0)
            if pooled2 is not None:
                z_p = _z(pooled2[0], pooled2[1], fd20["fd_core"], fd20["fd_core_se"])
                t2["z_pooled_vs_fd"] = z_p
                t2["pass_within_2sigma_pooled"] = bool(abs(z_p) <= 2.0)
    else:
        t2["status"] = "MISSING AD DATA"
    summary["tests"]["2_abs_a20_scaled_vs_own_fd"] = t2

    # -- Test 3: abs_a20_ctrl (unscaled E50) vs the SAME a=2.0 FD ------------- #
    t3 = {"description": f"abs_a20_ctrl (a=2.0, UNSCALED cap {' '.join(CAP_E50)}) "
                         f"core L{lo}-{hi} mean vs the SAME a=2.0 FD; "
                         f"pre-registered: OFF (>2 sigma) => the E_cap~a rule "
                         f"is load-bearing, not a null. Reports the "
                         f"scaled-vs-ctrl delta."}
    blk3, ss3, pooled3 = _governed_core(rows, "abs_a20_ctrl", lo, hi)
    if blk3 is not None:
        t3.update(blk3)
        if fd20 is not None:
            z_ss = _z(ss3[0], ss3[1], fd20["fd_core"], fd20["fd_core_se"])
            t3["z_seed_scatter_vs_fd"] = z_ss
            t3["off_beyond_2sigma_seed_scatter"] = bool(abs(z_ss) > 2.0)
            if pooled3 is not None:
                z_p = _z(pooled3[0], pooled3[1], fd20["fd_core"], fd20["fd_core_se"])
                t3["z_pooled_vs_fd"] = z_p
                t3["off_beyond_2sigma_pooled"] = bool(abs(z_p) > 2.0)
        # scaled-vs-ctrl delta (ctrl - scaled), pooled where available else ss
        if pooled3 is not None and pooled2 is not None:
            d = pooled3[0] - pooled2[0]
            d_se = math.sqrt(pooled3[1] ** 2 + pooled2[1] ** 2)
            t3["delta_ctrl_minus_scaled_pooled"] = [d, d_se,
                                                    (d / d_se if d_se else None)]
        if ss3 is not None and ss2 is not None:
            d = ss3[0] - ss2[0]
            d_se = math.sqrt(ss3[1] ** 2 + ss2[1] ** 2)
            t3["delta_ctrl_minus_scaled_seed_scatter"] = [d, d_se,
                                                          (d / d_se if d_se else None)]
    else:
        t3["status"] = "MISSING DATA"
    summary["tests"]["3_abs_a20_ctrl_off"] = t3

    # -- Test 4: layer-sum totals + governor totals-bias --------------------- #
    t4 = {"description": "layer-sum (L0-49) totals for all configs. The "
                         "governor totals-bias is a measurement (it clamps "
                         "state carriers, not crossing generators). Reported, "
                         "not headlined (PROTOCOL: absorber totals are "
                         "tail-noise-dominated).",
          "fd_truth_abs_total_on_design_a2.3": list(TRUTH_ABS_TOTAL_ON_DESIGN)}
    for cfg_name in CONFIGS:
        key = "mean_E" if cfg_name.startswith("abs_fd") else "mean_dE"
        t4[cfg_name] = _totals_block(rows, cfg_name, key) or {"status": "MISSING"}
    if fd20 is not None:
        t4["fd_a20_total_derivative"] = [fd20["fd_total"], fd20["fd_total_se"]]
        for cfg_name in ("abs_a20_scaled", "abs_a20_ctrl"):
            tb = t4.get(cfg_name)
            if tb and "mean" in tb and fd20["fd_total"]:
                r, r_se = _ratio(tb["mean"], tb["se"],
                                 fd20["fd_total"], fd20["fd_total_se"])
                t4[f"{cfg_name}_total_over_fd_a20"] = [r, r_se]
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
             if k not in ("description", "per_seed_core_means", "pooled_per_event",
                          "fd_per_layer", "fd_per_layer_se")},
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

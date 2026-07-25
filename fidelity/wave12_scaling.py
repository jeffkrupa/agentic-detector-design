"""Wave-12 absorber-governor scaling-rule bracket scan (LOCAL, pre-registered).

Wave-11 measured the governed absorber (`--track-dot-cap 1000:50`, wave-10
knob branch `knob/dot-governor` @ 3b6d45f) MATCHED on-design (a=2.30: core
L5-18 2233.68 +- 52.29 vs 1M FD truth 2233.68 +- 14.75, z = 0.0001) but ran
+9.7% HOT off-design (a=3.00: 2213.83 +- 47.18 vs own-FD truth
2018.05 +- 15.36, z = 3.95).  Wave-11 item A showed the gap boundary cap's
analogous off-design failure was a clamp-scale mismatch cured by cap ~ g.
This wave brackets the governor's two clamps separately at a=3.0 to identify
which clamp controls the mean, fit the local slope, interpolate the matched
clamp value, and infer the scaling variable.

Bracket variants (all a=3.00 g=5.70, unsevered -x 0 -y 0 -B 0 -f 0.2
-N 1e-3 -C 1000, absorber seed -a 3.0:1, n=2000, seeds 1-4, event dumps on;
the P1000:E50 reference at a=3.0 exists at 200k events from the wave-11
condor batch and is REUSED, never rerun):

    P-axis (E=50):  od_P500_E50, od_P700_E50, od_P1300_E50
    E-axis (P=1000): od_P1000_E35, od_P1000_E70

The bracket points sit on candidate scaling rules evaluated at a=3.0 from
the matched (P=1000, E=50) at a=2.3:
    P ~ 1/a -> P = 1000*(2.3/3.0) = 767   (~P700)
    P ~ a   -> P = 1000*(3.0/2.3) = 1304  (~P1300)
    E ~ a   -> E = 50*(3.0/2.3)  = 65.2   (~E70)
    E ~ 1/a -> E = 50*(2.3/3.0)  = 38.3   (~E35)
P500 widens the P bracket.  Wave-10 ON-design slope context: lower P lowers
the mean (P250:E50 -> 0.851x truth vs P1000:E50 -> 1.082x) and LOOSER E
lowers the mean (P1000:E200 -> 0.868x); the od excess is +9.7%, so the
mean-lowering directions are P-down and E-up.

Every sim call goes through tools.sim.run_forward (temp-dir isolation +
provenance-keyed cache); the gov binary and a 3600 s timeout are injected by
the in-process config override (load_config() dict is cached; global
config.yaml untouched) -- same trick as waves 1-11.

Run one unit:
    python -u -m fidelity.wave12_scaling --config od_P500_E50 --seed 1
Aggregate (partial data allowed; used for the pre-registered mid-scan
flatness check as well as the final summary):
    python -u -m fidelity.wave12_scaling --analyze
        -> fidelity/wave12_scaling_summary.json

Pre-registered predictions (ledger.jsonl wave-12, written before any run):
  1. some (P,E) at a=3.0 lands within 2 sigma of 2018.05 +- 15.36;
  2. the response is monotone in the controlling clamp with a fittable
     slope;
  3. the inferred rule, applied backward, is consistent with P1000:E50
     being matched at a=2.3.
Also recorded: per-variant per-event variance (a rule must not buy bias
transport with a variance blowup).

Errors: seed-scatter SEs AND pooled per-event SEs are both reported; the
slope fits weight by the pooled per-event SE (stable at 4 seeds; wave-11
showed the two agree, 52.3 vs 56.7 at 200k).  Failed / NaN runs are
recorded as failed rows and surfaced, never silently dropped.
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
                                       _window_sum)
from fidelity.wave11_absval import (_parse_event_dump, _per_event_stats,
                                    row_exists)

KNOB_BINARY = ("/eos/user/j/jeffkrup/agentic/hepemshow-gov/"
               "build_gov_fwd/HepEmShow")
KNOB_BRANCH = "knob/dot-governor"
KNOB_SHA = "3b6d45f"

A_OD = 3.0            # off-design absorber thickness [mm]
A_ON = 2.3            # on-design (where P1000:E50 is matched)
DEFAULT_N_EVENTS = 2000
SIM_TIMEOUT_OVERRIDE_S = 3600.0
CORE_WINDOW = (5, 18)
SEEDS = [1, 2, 3, 4]

# name: (P, E).  od_P1000_E65 was ADDED mid-scan under the pre-registered
# adaptive rule (fine-bracket the live axis): after the full first bracket
# the E clamp controls the response (median swing ~430 MeV over E 35->70,
# monotone) while the P axis is subdominant and non-monotone (~60 MeV over
# P 500->1300); E=65 sits on the E ~ a candidate rule (50*3.0/2.3 = 65.2)
# and seeds 5-8 were added to od_P1000_E70 to tighten the lower bracket.
CONFIGS = {
    "od_P500_E50":  (500.0, 50.0),
    "od_P700_E50":  (700.0, 50.0),
    "od_P1300_E50": (1300.0, 50.0),
    "od_P1000_E35": (1000.0, 35.0),
    "od_P1000_E65": (1000.0, 65.0),
    "od_P1000_E70": (1000.0, 70.0),
}

# Reference / truth numbers (recomputed from the wave-11 summary at analyze
# time when present; these literals make the module self-contained).
OD_FD_TRUTH = (2018.0495457018242, 15.362877569225724)   # own-FD, L5-18, a=3.0
REF_P, REF_E = 1000.0, 50.0
REF_OD = {                       # wave-11 abs_gov_od, 200k events, a=3.0
    "core_mean_seed_scatter": 2213.8297082818126,
    "core_se_seed_scatter": 47.1822602449957,
    "pooled_mean_W": 2213.829708281818,
    "pooled_se_W": 56.65903497575112,
    "pooled_var_W": 642049248.8766778,
    "pooled_median_W": 2117.118960875339,
    "n_events": 200000,
}
ON_MATCHED = {                   # wave-11 abs_gov_val, 200k events, a=2.3
    "core_mean_seed_scatter": 2233.68150912244,
    "core_se_seed_scatter": 52.28707676643073,
    "fd_truth": (2233.677839040897, 14.75417071763103),
}

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
OUT_DIR = _HERE / "wave12_runs"
SUMMARY_PATH = _HERE / "wave12_scaling_summary.json"
WAVE11_SUMMARY = _HERE / "wave11_absval_summary.json"


def _wave_config() -> dict:
    cfg = load_config()
    cfg.setdefault("sim", {})["subprocess_timeout_s"] = SIM_TIMEOUT_OVERRIDE_S
    cfg["paths"]["forward_bin"] = KNOB_BINARY
    return cfg


def _wave_ctrl(cfg: dict):
    ctrl = ctrl_flags_from_config(cfg)
    return dataclasses.replace(ctrl, stop_grad_mode=0, grazing_stop_track=0,
                               backward_boundary_stop=0)


def _gov_args(config: str) -> list:
    P, E = CONFIGS[config]
    return ["--track-dot-cap", f"{P:g}:{E:g}"]


def unit_out_path(config: str, seed: int) -> Path:
    return OUT_DIR / f"{config}_s{seed}.jsonl"


def events_npz_path(config: str, seed: int) -> Path:
    return OUT_DIR / f"{config}_s{seed}_events.npz"


def run_unit(config: str, seed: int, n_events: int) -> int:
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    extra = _gov_args(config)
    cfg = _wave_config()
    dp = dataclasses.replace(default_design_point(cfg), a=A_OD)
    ctrl = _wave_ctrl(cfg)

    out_path = unit_out_path(config, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if row_exists(out_path, config, seed, n_events):
        print(f"[unit] row exists for ({config}, s{seed}, n={n_events}) -> skip")
        return 0

    print(f"[unit] config={config} seed={seed} n={n_events} "
          f"a={dp.a} g={dp.g} extra={extra}")
    dump_dir = tempfile.mkdtemp(prefix="wave12_dump_")
    dump_path = Path(dump_dir) / "dump.txt"
    os.environ["HEPEMSHOW_EVENT_DUMP"] = str(dump_path)
    try:
        t0 = time.monotonic()
        rr = run_forward(dp, "a", n_events=n_events, seed=seed, ctrl=ctrl,
                         extra_args=extra)
        wall_s = time.monotonic() - t0
    finally:
        os.environ.pop("HEPEMSHOW_EVENT_DUMP", None)
    ok = (rr.returncode == 0 and rr.edeps is not None and not rr.nan)

    per_event = None
    if ok:
        if dump_path.exists():
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
    if dump_path.exists():
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
        "seeded_param": "a",
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


def _z(a, a_se, b, b_se):
    den = math.sqrt(a_se ** 2 + b_se ** 2)
    return (a - b) / den if den > 0 else None


def _wls_line(x, y, se):
    """Weighted least-squares line y = m*x + c -> (m, m_se, c, c_se)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(se, dtype=float) ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    if D <= 0:
        return None, None, None, None
    m = (S * Sxy - Sx * Sy) / D
    c = (Sxx * Sy - Sx * Sxy) / D
    return float(m), float(math.sqrt(S / D)), float(c), float(math.sqrt(Sxx / D))


def _load_wave11_refs():
    """Prefer live numbers from the wave-11 summary over the literals."""
    od_fd, ref_od, on_matched = OD_FD_TRUTH, dict(REF_OD), dict(ON_MATCHED)
    if WAVE11_SUMMARY.exists():
        with open(WAVE11_SUMMARY) as fh:
            s11 = json.load(fh)
        t2 = s11["tests"]["2_abs_gov_offdesign_vs_own_fd"]
        od_fd = (t2["fd_core"], t2["fd_core_se"])
        ref_od.update({
            "core_mean_seed_scatter": t2["ad_core_mean"],
            "core_se_seed_scatter": t2["ad_core_se"],
            "pooled_mean_W": t2["pooled_per_event"]["mean_W"],
            "pooled_se_W": t2["pooled_per_event"]["se_W"],
            "pooled_var_W": t2["pooled_per_event"]["var_W"],
            "pooled_median_W": t2["pooled_per_event"]["median_W"],
            "n_events": s11["n_events_per_unit"] * len(t2["pooled_per_event"]
                                                       ["seeds_with_events"]),
        })
        t1 = s11["tests"]["1_abs_gov_core_vs_truth"]
        on_matched.update({
            "core_mean_seed_scatter": t1["ad_core_mean_seed_scatter"],
            "core_se_seed_scatter": t1["ad_core_se_seed_scatter"],
            "fd_truth": tuple(t1["fd_truth_window"]),
        })
    return od_fd, ref_od, on_matched


def analyze(n_events: int, out_json: Path) -> int:
    lo, hi = CORE_WINDOW
    od_fd, ref_od, on_matched = _load_wave11_refs()
    truth, truth_se = od_fd
    rows = _load_unit_rows(str(OUT_DIR / "*.jsonl"), n_events)
    if not rows:
        print(f"[analyze] no unit rows with n_events={n_events} under "
              f"{OUT_DIR}; nothing to score -- NOT writing a summary file")
        return 0

    summary = {
        "wave": 12,
        "item": "absorber governor scaling-rule bracket scan (a=3.0)",
        "knob_branch": KNOB_BRANCH,
        "knob_sha": KNOB_SHA,
        "binary": binary_provenance(KNOB_BINARY) if Path(KNOB_BINARY).exists()
                  else {"binary_path": KNOB_BINARY, "missing": True},
        "n_events_per_unit": n_events,
        "seeds_per_variant": SEEDS,
        "core_window_layers": [lo, hi],
        "od_fd_truth_core": [truth, truth_se],
        "reference_P1000_E50_at_a3.0": ref_od,
        "on_design_matched_P1000_E50": on_matched,
        "note_errors": ("seed-scatter SEs AND pooled per-event SEs both "
                        "reported; window sums keep inter-layer covariance "
                        "via per-seed sums; slope fits weight by the pooled "
                        "per-event SE (stable at 4 seeds)."),
        "variants": {},
        "tests": {},
    }

    # -- per-variant bracket table ------------------------------------------- #
    variants = {}
    for cfg_name, (P, E) in CONFIGS.items():
        mean_l, se_l, stack, seeds, n_fail = _seed_scatter(rows, cfg_name,
                                                           "mean_dE")
        block = {"P": P, "E": E,
                 "ok_seeds": seeds, "n_failed_rows": n_fail,
                 "missing_seeds": sorted(set(SEEDS) - set(seeds))}
        if mean_l is not None:
            w, w_se = _window_sum(stack, lo, hi)
            block.update({
                "core_mean_seed_scatter": w,
                "core_se_seed_scatter": w_se,
                "per_seed_core_means": stack[:, lo:hi + 1].sum(axis=1).tolist(),
                "z_vs_od_truth_seed_scatter": _z(w, w_se, truth, truth_se),
            })
            Wv, Tv, with_ev, without = _pooled_events(rows, cfg_name)
            if Wv is not None:
                pe = _per_event_stats(Wv, Tv)
                pe["seeds_with_events"] = with_ev
                pe["seeds_without_events"] = without
                pe["z_pooled_vs_od_truth"] = _z(pe["mean_W"], pe["se_W"],
                                                truth, truth_se)
                pe["var_ratio_vs_ref_P1000_E50"] = (pe["var_W"]
                                                    / ref_od["pooled_var_W"])
                block["pooled_per_event"] = pe
        else:
            block["status"] = "INSUFFICIENT DATA (<2 ok seeds)"
        variants[cfg_name] = block
    summary["variants"] = variants

    def _pt(name):
        b = variants.get(name, {})
        if "pooled_per_event" in b:
            return (b["pooled_per_event"]["mean_W"],
                    b["pooled_per_event"]["se_W"])
        if "core_mean_seed_scatter" in b:
            return (b["core_mean_seed_scatter"], b["core_se_seed_scatter"])
        return None

    ref_pt = (ref_od["pooled_mean_W"], ref_od["pooled_se_W"])

    # -- Test 1: controlling clamp + local slopes ---------------------------- #
    p_axis = [(CONFIGS[n][0], _pt(n)) for n in
              ("od_P500_E50", "od_P700_E50", "od_P1300_E50")]
    p_axis.append((REF_P, ref_pt))
    p_axis = sorted([(x, v) for x, v in p_axis if v is not None])
    e_axis = [(CONFIGS[n][1], _pt(n)) for n in
              ("od_P1000_E35", "od_P1000_E65", "od_P1000_E70")]
    e_axis.append((REF_E, ref_pt))
    e_axis = sorted([(x, v) for x, v in e_axis if v is not None])

    t1 = {"description": f"core L{lo}-{hi} pooled mean vs clamp value on each "
                         f"axis (reference P1000:E50 from wave-11 200k "
                         f"included); WLS slope d(mean)/d(clamp); the "
                         f"controlling clamp is the axis with the larger "
                         f"|slope|*bracket-width significance."}
    axes = {}
    for ax_name, pts, width in (("P", p_axis, 800.0), ("E", e_axis, 35.0)):
        xs = [x for x, _ in pts]
        ys = [v[0] for _, v in pts]
        ses = [v[1] for _, v in pts]
        m, m_se, c, c_se = (_wls_line(xs, ys, ses) if len(pts) >= 2
                            else (None, None, None, None))
        mono = None
        if len(ys) >= 2:
            d = np.diff(ys)
            mono = bool((d > 0).all() or (d < 0).all())
        axes[ax_name] = {
            "points": [{"clamp": x, "mean": v[0], "se": v[1]} for x, v in pts],
            "slope_MeV_per_unit": m, "slope_se": m_se,
            "intercept": c, "intercept_se": c_se,
            "slope_significance": (abs(m / m_se) if m is not None and m_se
                                   else None),
            "move_over_bracket_MeV": (m * width if m is not None else None),
            "monotone": mono,
        }
    t1["axes"] = axes
    ctrl_axis = None
    sig = {a: (axes[a]["slope_significance"] or 0.0) *
              (abs(axes[a]["move_over_bracket_MeV"] or 0.0) > 0)
           for a in axes}
    moves = {a: abs(axes[a]["move_over_bracket_MeV"] or 0.0) for a in axes}
    if any(v is not None for v in (axes["P"]["slope_MeV_per_unit"],
                                   axes["E"]["slope_MeV_per_unit"])):
        # controlling = larger bracket-move weighted by slope significance
        score = {a: moves[a] * min(1.0, sig[a] / 2.0) for a in axes}
        ctrl_axis = max(score, key=score.get)
    t1["controlling_axis"] = ctrl_axis
    summary["tests"]["1_controlling_clamp_and_slope"] = t1

    # -- Test 2: matched clamp + scaling-rule candidates --------------------- #
    t2 = {"description": f"interpolate the clamp value where the fitted line "
                         f"crosses the od FD truth {truth:.2f} +- "
                         f"{truth_se:.2f}; exponent k from (matched at "
                         f"a={A_OD}) vs (matched {REF_P:g}:{REF_E:g} at "
                         f"a={A_ON}): clamp ~ a^k; nearest simple rules "
                         f"quoted. Backward consistency = the rule evaluated "
                         f"at a={A_ON} returns the wave-11-matched clamp by "
                         f"construction (anchor); the substantive check is "
                         f"whether a SIMPLE rule (k in {{-1, 0, 1}}) is "
                         f"consistent with the matched value within its "
                         f"interpolation error."}
    matched = {}
    for ax_name, x0 in (("P", REF_P), ("E", REF_E)):
        ax = axes[ax_name]
        m, m_se, c = (ax["slope_MeV_per_unit"], ax["slope_se"],
                      ax["intercept"])
        if m in (None, 0.0):
            matched[ax_name] = {"status": "no usable slope"}
            continue
        x_star = (truth - c) / m
        # error: propagate truth SE and slope SE about the reference point
        dx_truth = truth_se / abs(m)
        dx_slope = (abs((truth - c) / m) * (m_se / abs(m))
                    if m_se is not None else 0.0)
        x_star_se = math.sqrt(dx_truth ** 2 + dx_slope ** 2)
        k = (math.log(x_star / x0) / math.log(A_OD / A_ON)
             if x_star > 0 else None)
        entry = {"matched_clamp": float(x_star),
                 "matched_clamp_se": float(x_star_se),
                 "anchor_at_a2.3": x0,
                 "exponent_k": k,
                 "simple_rules": {}}
        for kk, label in ((-1.0, "clamp ~ 1/a"), (0.0, "clamp const"),
                          (1.0, "clamp ~ a")):
            pred = x0 * (A_OD / A_ON) ** kk
            entry["simple_rules"][label] = {
                "predicted_clamp_at_a3.0": pred,
                "consistent_within_matched_se": bool(
                    abs(pred - x_star) <= 2.0 * x_star_se),
            }
        matched[ax_name] = entry
    t2["matched"] = matched
    summary["tests"]["2_matched_clamp_and_rule"] = t2

    # -- Test 3: verdict vs pre-registration --------------------------------- #
    within = {n: (b.get("pooled_per_event", {}).get("z_pooled_vs_od_truth")
                  if "pooled_per_event" in b
                  else b.get("z_vs_od_truth_seed_scatter"))
              for n, b in variants.items()}
    within2s = {n: (abs(z) <= 2.0) for n, z in within.items()
                if z is not None}
    t3 = {
        "description": "pre-registered: (1) some variant within 2 sigma of "
                       "the od FD truth; (2) monotone controlling axis with "
                       "a fittable slope; (3) rule backward-consistent with "
                       "the on-design match.",
        "variant_z_vs_truth": within,
        "pred1_some_variant_within_2sigma": bool(any(within2s.values()))
                                            if within2s else None,
        "pred2_controlling_axis_monotone": (axes[ctrl_axis]["monotone"]
                                            if ctrl_axis else None),
        "pred2_slope_significance": (axes[ctrl_axis]["slope_significance"]
                                     if ctrl_axis else None),
    }
    summary["tests"]["3_verdict_vs_preregistration"] = t3

    # -- Test 4: per-variant variance (bias transport must not blow up var) -- #
    t4 = {"description": "pooled per-event var(W_e) per variant, ratio vs the "
                         "wave-11 P1000:E50 reference at a=3.0 "
                         f"({ref_od['pooled_var_W']:.4g}).",
          "ref_var_W": ref_od["pooled_var_W"],
          "per_variant": {n: {"var_W": b.get("pooled_per_event", {}).get("var_W"),
                              "ratio_vs_ref": b.get("pooled_per_event", {})
                              .get("var_ratio_vs_ref_P1000_E50")}
                          for n, b in variants.items()}}
    summary["tests"]["4_variance_transport"] = t4

    # -- Test 5 (POST-HOC, not pre-registered): robust-estimator curves ------ #
    # The pre-registered mean estimator is variance-limited at n=2000 x 4
    # (pooled SEs 300-730 MeV vs the ~60 needed); the per-event median /
    # trimmed means are far tighter on the same dumps (wave-11 B precedent:
    # medians as diagnostics, labelled post-hoc).  Bootstrap SEs (10k
    # resamples); the wave-11 reference recomputed from its own event npz
    # files with the same estimators so both sides are like-for-like.
    rng = np.random.default_rng(20260725)

    def _boot(W, stat, n_boot=None):
        n = len(W)
        # chunked (memory-bounded) resampling; fewer resamples for very
        # large pools (SE-of-SE stays < ~2%)
        if n_boot is None:
            n_boot = int(min(10000, max(2000, 4e8 // n)))
        chunk = max(1, int(2e7 // n))
        vals = []
        done = 0
        while done < n_boot:
            b = min(chunk, n_boot - done)
            idx = rng.integers(0, n, size=(b, n))
            vals.append(np.atleast_1d(stat(W[idx])))
            done += b
        return float(np.concatenate(vals).std(ddof=1))

    def _tm(Wm, frac):
        s = np.sort(Wm, axis=-1)
        k = int(frac * s.shape[-1])
        return s[..., k:s.shape[-1] - k].mean(axis=-1)

    def _robust_block(W):
        return {
            "median": float(np.median(W)),
            "median_se_boot": _boot(W, lambda w: np.median(w, axis=-1)),
            "tm5": float(_tm(W, 0.05)),
            "tm5_se_boot": _boot(W, lambda w: _tm(w, 0.05)),
        }

    t5 = {"description": "POST-HOC (not pre-registered): per-event median and "
                         "5%-trimmed-mean vs clamp value, bootstrap SEs (10k "
                         "resamples); wave-11 P1000:E50 reference recomputed "
                         "from its own event npz with the same estimators. "
                         "Median-anchored matched clamp assumes a locally "
                         "clamp-independent median/mean ratio (taken at the "
                         "reference) -- stated assumption, condor-scale mean "
                         "confirmation still required.",
          "post_hoc": True}
    ref_W = []
    for p in sorted((_HERE / "wave11_runs").glob("abs_gov_od_s*_events.npz")):
        with np.load(p) as z:
            ref_W.append(z["W"])
    robust = {}
    if ref_W:
        ref_W = np.concatenate(ref_W)
        rb = _robust_block(ref_W)
        rb["mean"] = float(ref_W.mean())
        rb["n_events"] = int(len(ref_W))
        robust["REF_P1000_E50"] = rb
    for cfg_name in CONFIGS:
        Wv, Tv, _, _ = _pooled_events(rows, cfg_name)
        if Wv is not None and len(Wv) >= 100:
            robust[cfg_name] = _robust_block(Wv)
            robust[cfg_name]["n_events"] = int(len(Wv))
    t5["estimators"] = robust
    if "REF_P1000_E50" in robust:
        for est in ("median", "tm5"):
            e_pts = []
            for n2, (P2, E2) in CONFIGS.items():
                if P2 == REF_P and n2 in robust:
                    e_pts.append((E2, robust[n2][est],
                                  robust[n2][f"{est}_se_boot"]))
            e_pts.append((REF_E, robust["REF_P1000_E50"][est],
                          robust["REF_P1000_E50"][f"{est}_se_boot"]))
            e_pts.sort()
            xs = [p[0] for p in e_pts]
            ys = [p[1] for p in e_pts]
            ses = [p[2] for p in e_pts]
            m, m_se, c, c_se = _wls_line(xs, ys, ses)
            d = np.diff(ys)
            blk = {"points": [{"E": x, est: y, "se_boot": s}
                              for x, y, s in e_pts],
                   "monotone": bool((d > 0).all() or (d < 0).all()),
                   "slope_MeV_per_unit": m, "slope_se": m_se,
                   "slope_significance": (abs(m / m_se)
                                          if m and m_se else None)}
            # anchor: target = truth * (ref_est / ref_mean)
            ratio = (robust["REF_P1000_E50"][est]
                     / robust["REF_P1000_E50"]["mean"])
            target = truth * ratio
            if m not in (None, 0.0):
                x_star = (target - c) / m
                x_star_se = math.sqrt((truth_se * ratio / m) ** 2
                                      + ((target - c) / m) ** 2
                                        * (m_se / m) ** 2)
                k = (math.log(x_star / REF_E) / math.log(A_OD / A_ON)
                     if x_star > 0 else None)
                blk.update({"anchor_target": target,
                            "matched_E": float(x_star),
                            "matched_E_se": float(abs(x_star_se)),
                            "exponent_k": k,
                            "E_prop_a_prediction": REF_E * A_OD / A_ON})
            t5[f"e_axis_{est}"] = blk
        # P-axis robust curve (context: expected subdominant / non-monotone)
        p_pts = []
        for n2, (P2, E2) in CONFIGS.items():
            if E2 == REF_E and n2 in robust:
                p_pts.append((P2, robust[n2]["median"],
                              robust[n2]["median_se_boot"]))
        p_pts.append((REF_P, robust["REF_P1000_E50"]["median"],
                      robust["REF_P1000_E50"]["median_se_boot"]))
        p_pts.sort()
        d = np.diff([p[1] for p in p_pts])
        t5["p_axis_median"] = {
            "points": [{"P": x, "median": y, "se_boot": s}
                       for x, y, s in p_pts],
            "monotone": bool((d > 0).all() or (d < 0).all()),
        }
    summary["tests"]["5_posthoc_robust_estimators"] = t5

    complete = all("core_mean_seed_scatter" in b and not b["missing_seeds"]
                   for b in variants.values())
    summary["complete"] = complete
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[analyze] bracket table (pooled per-event mean +- SE; z vs "
          f"od truth {truth:.2f} +- {truth_se:.2f}):")
    for n, b in variants.items():
        pe = b.get("pooled_per_event")
        if pe:
            print(f"  {n:16s} P={b['P']:6g} E={b['E']:4g}  "
                  f"{pe['mean_W']:8.2f} +- {pe['se_W']:6.2f}  "
                  f"z={pe['z_pooled_vs_od_truth']:+.2f}  "
                  f"var_x={pe['var_ratio_vs_ref_P1000_E50']:.2f}")
        else:
            print(f"  {n:16s} {b.get('status', 'no pooled events')}")
    print(f"  {'REF P1000:E50':16s} P=  1000 E=  50  "
          f"{ref_pt[0]:8.2f} +- {ref_pt[1]:6.2f}  (wave-11, 200k)")
    for a in ("P", "E"):
        ax = axes[a]
        if ax["slope_MeV_per_unit"] is not None:
            print(f"[analyze] {a}-axis slope {ax['slope_MeV_per_unit']:+.4f} "
                  f"+- {ax['slope_se']:.4f} MeV/unit  "
                  f"(signif {ax['slope_significance']:.1f}, "
                  f"monotone={ax['monotone']})")
    print(f"[analyze] controlling axis: {ctrl_axis}")
    for a, mres in matched.items():
        if "matched_clamp" in mres:
            print(f"[analyze] {a} matched at a={A_OD}: "
                  f"{mres['matched_clamp']:.1f} +- "
                  f"{mres['matched_clamp_se']:.1f} (k={mres['exponent_k']:+.2f})")
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

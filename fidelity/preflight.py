"""WAVE 15 pre-flight for the Fisher-info sampling optimization.

Verifies the two load-bearing assumptions of the campaign
(fidelity/OPTIMIZATION_PLAN.md §3) BEFORE the ~155 CPU-h search:

  Check 1 -- ENERGY-GRADIENT EXACTNESS off the uniform baseline.
    The objective g_i = dmu_i/dE must stay exact (AD/FD_E ~ 1) across the
    displaced gap profiles the optimizer will explore, not just at uniform.
    Four gap profiles at fixed total gap Sigma g_i = 50*5.70 = 285 mm, uniform
    absorber a=2.30, 10 GeV e-, t=400, 50 layers:
      P0 uniform      : g_i = 5.70 all layers.
      P1 front-loaded : 8.55 (L0-24) / 2.85 (L25-49).
      P2 back-loaded  : 2.85 (L0-24) / 8.55 (L25-49).
      P3 peaked       : 9.00 (L8-24, shower-max zone) / 4.00 (elsewhere).
    AD energy derivative (forward, -e 10000:1, gap profile set, rng-lineage OFF)
    vs CRN-paired central FD in E (primals at E=9950 / 10050 MeV, h=100,
    --rng-lineage 1 for per-event pairing, wave-4). FD_i = (mu_i(E+)-mu_i(E-))/100.
    ACCEPTANCE: core-window (L5-18) AD/FD_E within ~2% at every profile; any
    layer/profile deviating >5% (where FD is significant) is FLAGGED.

  Check 2 -- COVARIANCE CONDITIONING at n=20k (P0, P3).
    From the per-event per-layer E_i dumps (HEPEMSHOW_EVENT_DUMP) build the
    50x50 covariance Sigma; report cond(Sigma), I = g^T Sigma^-1 g (g = mean_dE),
    I_diag = sum g_i^2/var_i, and whether I is stable to <~10% between two
    independent 20k runs (seeds 1 vs 2). Also the 6x6 ZONE-level covariance and
    I_zone/cond as the documented fallback, plus a Ledoit-Wolf-shrunk I.
    ACCEPTANCE: EITHER the 50x50 I is stable <10% run-to-run (use full Sigma)
    OR recommend the 6x6 zone-Fisher fallback (report which, with numbers).

Binary: the governor forward build (has --gap-profile, --rng-lineage, the
HEPEMSHOW_EVENT_DUMP per-event dump, all knobs default-off => baseline physics).
AD runs carry NO lineage; FD runs carry --rng-lineage 1. The energy channel is
lineage-insensitive in expectation (OPTIMIZATION_PLAN §4 / wave-4), so AD (no
lineage) and CRN-paired FD (lineage on) are compared consistently.

Run one unit (one condor job):
    python -u -m fidelity.preflight --profile P0 --role ad --seed 1 --n-events 20000
      -> appends one JSONL row to fidelity/preflight_runs/<profile>_<role>_s<seed>.jsonl
      -> P0/P3 AD units also write fidelity/preflight_runs/dumps/<...>.evdump
    Idempotent: skips the sim if the ok row (and any expected dump) already exist.

Analyze after all units finish:
    python -u -m fidelity.preflight --analyze
      -> fidelity/preflight_summary.json (both checks)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tools.sim import (load_config, default_design_point, ctrl_flags_from_config,
                       binary_provenance, _common_args, _parse_edeps)
from tools.schemas import DesignPoint

# --------------------------------------------------------------------------- #
# Constants / registry
# --------------------------------------------------------------------------- #
GOV_BIN = "/eos/user/j/jeffkrup/agentic/hepemshow-gov/build_gov_fwd/HepEmShow"
N_LAYERS = 50
TOTAL_GAP_MM = 50 * 5.70               # 285.0 mm, fixed sensitive volume
ABSORBER_MM = 2.30
BEAM_MEV = 10000.0
FD_H_MEV = 50.0                        # +-50 MeV displacement; central denom = 100
CORE_WINDOW = (5, 19)                  # L5..L18 inclusive -> half-open [5,19)
ZONE_EDGES = (0, 8, 16, 24, 32, 40, 50)  # 6 zones of ~8 layers
SIM_TIMEOUT_S = 10800.0                # in-process wall cap per unit
DEFAULT_N_EVENTS = 20000

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "preflight_runs"
DUMP_DIR = OUT_DIR / "dumps"
SUMMARY_PATH = _HERE / "preflight_summary.json"


def _uniform():
    return tuple([5.70] * N_LAYERS)


def _front():
    return tuple([8.55] * 25 + [2.85] * 25)


def _back():
    return tuple([2.85] * 25 + [8.55] * 25)


def _peaked():
    return tuple([9.0 if 8 <= i <= 24 else 4.0 for i in range(N_LAYERS)])


PROFILES = {
    "P0": _uniform(),
    "P1": _front(),
    "P2": _back(),
    "P3": _peaked(),
}

# role -> (energy_MeV, rng_lineage)
ROLES = {
    "ad":      (BEAM_MEV, 0),
    "fdplus":  (BEAM_MEV + FD_H_MEV, 1),
    "fdminus": (BEAM_MEV - FD_H_MEV, 1),
}

# Which units write the per-event dump for the covariance check (P0/P3 AD only).
_DUMP_PROFILES = ("P0", "P3")

# The pre-registered unit matrix (28 units, ~28 CPU-h; the pre-flight ~20-30 job
# budget). AD: 4 profiles x 3 seeds (seeds 1/2 = the two independent covariance
# runs for P0/P3, seed 3 a robustness third). FD: 4 profiles x 2 CRN-paired
# seeds per side.
AD_SEEDS = (1, 2, 3)
FD_SEEDS = (1, 2)


def unit_matrix():
    units = []
    for prof in PROFILES:
        for s in AD_SEEDS:
            units.append((prof, "ad", s))
        for role in ("fdplus", "fdminus"):
            for s in FD_SEEDS:
                units.append((prof, role, s))
    return units


# --------------------------------------------------------------------------- #
# Paths / idempotency
# --------------------------------------------------------------------------- #
def unit_out_path(profile: str, role: str, seed: int) -> Path:
    return OUT_DIR / f"{profile}_{role}_s{seed}.jsonl"


def dump_path(profile: str, role: str, seed: int) -> Path:
    return DUMP_DIR / f"{profile}_{role}_s{seed}.evdump"


def expects_dump(profile: str, role: str) -> bool:
    return role == "ad" and profile in _DUMP_PROFILES


def row_exists(path: Path, profile: str, role: str, seed: int, n_events: int) -> bool:
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
            if (r.get("profile") == profile and r.get("role") == role
                    and r.get("seed") == seed and r.get("n_events") == n_events
                    and r.get("ok")):
                return True
    return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_unit(profile: str, role: str, seed: int, n_events: int) -> int:
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    if role not in ROLES:
        raise SystemExit(f"unknown role {role!r}; choose from {sorted(ROLES)}")
    energy, rng_lineage = ROLES[role]
    gap_profile = PROFILES[profile]

    out_path = unit_out_path(profile, role, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    want_dump = expects_dump(profile, role)
    dp_dump = dump_path(profile, role, seed)
    if want_dump:
        dp_dump.parent.mkdir(parents=True, exist_ok=True)

    have_row = row_exists(out_path, profile, role, seed, n_events)
    have_dump = (not want_dump) or dp_dump.exists()
    if have_row and have_dump:
        print(f"[unit] complete for ({profile},{role},s{seed},n={n_events}) -> skip")
        return 0

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)   # canonical severed flags (paper defaults)
    dp = DesignPoint(a=ABSORBER_MM, g=5.70, energy=float(energy),
                     n_layers=N_LAYERS, transverse=400.0, particle="e-",
                     gap_profile=gap_profile)
    # Reuse tools.sim's arg construction for exact consistency with the rest of
    # the repo (gap-profile + energy-dot + canonical ctrl flags). Seed the
    # ENERGY dot: primal mean_E is dot-independent, and AD units read mean_dE.
    args = _common_args(dp, ctrl, n_events, seed, cfg, seeded_param="energy")
    if rng_lineage:
        args += ["--rng-lineage", "1"]

    prov = binary_provenance(GOV_BIN)
    print(f"[unit] profile={profile} role={role} seed={seed} n={n_events} "
          f"E={energy} rng_lineage={rng_lineage} dump={want_dump}")
    print(f"[unit] binary={prov['binary_path']} rev={prov['binary_git_rev']}")

    edeps, rc, nan, dump_written = _execute_gov(args, seed, dp_dump if want_dump else None)
    ok = (rc == 0 and edeps is not None and not nan
          and ((not want_dump) or dump_written))

    row = {
        "profile": profile,
        "role": role,
        "seed": seed,
        "n_events": n_events,
        "energy_mev": float(energy),
        "rng_lineage": int(rng_lineage),
        "a": ABSORBER_MM,
        "n_layers": N_LAYERS,
        "gap_profile": list(gap_profile),
        "total_gap_mm": float(sum(gap_profile)),
        "flags": ctrl.to_cli_args(),
        "binary": prov["binary_path"],
        "binary_git_rev": prov["binary_git_rev"],
        "binary_mtime_ns": prov["binary_mtime_ns"],
        "binary_size": prov["binary_size"],
        "fd_h_mev": FD_H_MEV,
        "dump_path": str(dp_dump) if (want_dump and dump_written) else None,
        "ok": bool(ok),
        "returncode": int(rc),
        "nan": bool(nan),
    }
    if edeps is not None:
        row["mean_E"] = edeps[:, 0].tolist()
        row["var_E"] = edeps[:, 1].tolist()
        if role == "ad":
            row["mean_dE"] = edeps[:, 2].tolist()   # g_i = dmu_i/dE
            row["var_dE"] = edeps[:, 3].tolist()
    with open(out_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    if not ok:
        print(f"[unit] FAILED (rc={rc}, nan={nan}, dump_written={dump_written}); "
              f"recorded as failed row", file=sys.stderr)
        return 1
    tot = edeps[:, 0].sum()
    msg = f"[unit] OK total_E={tot:.2f} MeV"
    if role == "ad":
        msg += f"  sum_dE={edeps[:, 2].sum():.5f}"
    if want_dump:
        msg += f"  dump={dp_dump}"
    print(msg + f" -> {out_path}")
    return 0


def _execute_gov(args, seed, dump_target):
    """Run the gov binary in an isolated temp dir; parse edeps_<seed>; copy the
    per-event dump to ``dump_target`` if requested. Returns
    (edeps|None, rc, nan, dump_written)."""
    if not Path(GOV_BIN).exists():
        raise FileNotFoundError(f"gov binary not found: {GOV_BIN}")
    with tempfile.TemporaryDirectory(prefix="preflight_gov_") as wd:
        env = dict(os.environ)
        dump_tmp = None
        if dump_target is not None:
            dump_tmp = str(Path(wd) / "event.dump")
            env["HEPEMSHOW_EVENT_DUMP"] = dump_tmp
        else:
            env.pop("HEPEMSHOW_EVENT_DUMP", None)
        try:
            proc = subprocess.run([GOV_BIN, *args], cwd=wd, env=env,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE, timeout=SIM_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[sim] timed out after {SIM_TIMEOUT_S}s: {GOV_BIN}\n")
            return None, -1, True, False
        rc = proc.returncode
        out = Path(wd) / f"edeps_{seed}"
        if not out.exists():
            cand = list(Path(wd).glob("edeps_*"))
            cand = [c for c in cand if "gap" not in c.name]
            out = cand[0] if cand else out
        if rc != 0 or not out.exists():
            sys.stderr.write(proc.stderr.decode()[-2000:] if proc.stderr else "")
            return None, rc, True, False
        arr = _parse_edeps(out)
        nan = bool(np.isnan(arr).any())
        dump_written = False
        if dump_target is not None and dump_tmp is not None and Path(dump_tmp).exists():
            dump_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dump_tmp, dump_target)
            dump_written = True
        return arr, rc, nan, dump_written


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def _load_rows(profile=None, role=None):
    rows = []
    for p in sorted(glob.glob(str(OUT_DIR / "*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if profile and r.get("profile") != profile:
                    continue
                if role and r.get("role") != role:
                    continue
                rows.append(r)
    return rows


def _ok_by_seed(rows, profile, role):
    by = {}
    for r in rows:
        if r.get("profile") == profile and r.get("role") == role and r.get("ok"):
            by.setdefault(r["seed"], r)
    return by


def _ad_aggregate(rows, profile):
    """AD g_i aggregated over seeds. Returns
    (g_l, g_se_pooled_l, g_se_scatter_l, per_seed_g, n_total, seeds)."""
    by = _ok_by_seed(rows, profile, "ad")
    if not by:
        return None
    means, vars_, ns = [], [], []
    for s in sorted(by):
        r = by[s]
        means.append(np.asarray(r["mean_dE"], float))
        vars_.append(np.asarray(r["var_dE"], float))
        ns.append(int(r["n_events"]))
    means = np.stack(means)
    vars_ = np.stack(vars_)
    ns = np.asarray(ns, float)[:, None]
    n_total = float(ns.sum())
    g_l = (means * ns).sum(0) / n_total
    var_pooled = (vars_ * ns).sum(0) / n_total
    se_pooled = np.sqrt(var_pooled / n_total)
    if means.shape[0] >= 2:
        se_scatter = means.std(0, ddof=1) / math.sqrt(means.shape[0])
    else:
        se_scatter = np.full_like(g_l, np.nan)
    return g_l, se_pooled, se_scatter, means, n_total, sorted(by)


def _paired_fd(rows, profile):
    """CRN central FD per matched seed -> mean over seeds, SE from seed scatter.
    FD_i = (mu_i(E+) - mu_i(E-)) / (2*FD_H). Returns (fd_l, fd_se_l, seeds, n_ev)."""
    bp = _ok_by_seed(rows, profile, "fdplus")
    bm = _ok_by_seed(rows, profile, "fdminus")
    seeds = sorted(set(bp) & set(bm))
    if len(seeds) < 2:
        return None
    diffs, used, n_ev = [], [], 0
    for s in seeds:
        rp, rm = bp[s], bm[s]
        if rp["n_events"] != rm["n_events"]:
            continue
        d = (np.asarray(rp["mean_E"], float) - np.asarray(rm["mean_E"], float)) / (2 * FD_H_MEV)
        diffs.append(d)
        used.append(s)
        n_ev += int(rp["n_events"])
    if len(diffs) < 2:
        return None
    diffs = np.stack(diffs)
    fd_l = diffs.mean(0)
    fd_se_l = diffs.std(0, ddof=1) / math.sqrt(diffs.shape[0])
    return fd_l, fd_se_l, used, n_ev


def _ratio_err(ad, ad_se, fd, fd_se):
    ratio = ad / fd
    rel = np.sqrt((ad_se / ad) ** 2 + (fd_se / fd) ** 2)
    return ratio, np.abs(ratio) * rel


def _load_dump_E(path: Path):
    """Load per-event per-layer E_i (columns 4..4+N-1) from an event dump."""
    E = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4 + N_LAYERS:
                continue
            E.append([float(x) for x in parts[4:4 + N_LAYERS]])
    return np.asarray(E, float)


def _ledoit_wolf(X):
    """Ledoit-Wolf shrinkage of the sample covariance toward a scaled identity.
    X is (n_samples, p). Returns (Sigma_shrunk, shrinkage_delta)."""
    n, p = X.shape
    Xc = X - X.mean(0, keepdims=True)
    S = (Xc.T @ Xc) / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.sum((S - F) ** 2) / p
    b_bar2 = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        Si = xi @ xi.T
        b_bar2 += np.sum((Si - S) ** 2)
    b_bar2 = b_bar2 / (n ** 2) / p
    b2 = min(b_bar2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return (1 - delta) * S + delta * F, float(delta)


def _fisher_from_cov(Sigma, g):
    """I = g^T Sigma^-1 g via a solve (SPD-safe)."""
    try:
        x = np.linalg.solve(Sigma, g)
        return float(g @ x)
    except np.linalg.LinAlgError:
        return float(g @ (np.linalg.pinv(Sigma) @ g))


def _zone_reduce(E):
    """(n_events, 50) per-layer E -> (n_events, 6) per-zone E (sums)."""
    Z = []
    for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:]):
        Z.append(E[:, a:b].sum(1))
    return np.stack(Z, axis=1)


def _cov_block(E, g):
    """Given per-event E (n,50) and g (50,), compute the full covariance report."""
    Sigma = np.cov(E, rowvar=False)
    var_i = np.diag(Sigma)
    cond = float(np.linalg.cond(Sigma))
    I_full = _fisher_from_cov(Sigma, g)
    I_diag = float(np.sum(g ** 2 / var_i))
    Sig_lw, delta = _ledoit_wolf(E)
    I_lw = _fisher_from_cov(Sig_lw, g)
    cond_lw = float(np.linalg.cond(Sig_lw))
    return {
        "n_events": int(E.shape[0]),
        "cond_Sigma": cond,
        "I_full": I_full,
        "I_diag": I_diag,
        "I_lw": I_lw,
        "cond_Sigma_lw": cond_lw,
        "lw_shrinkage_delta": delta,
    }


def _zone_block(E, g):
    Ez = _zone_reduce(E)
    gz = np.asarray([g[a:b].sum() for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:])])
    Sz = np.cov(Ez, rowvar=False)
    cond = float(np.linalg.cond(Sz))
    I_zone = _fisher_from_cov(Sz, gz)
    I_zone_diag = float(np.sum(gz ** 2 / np.diag(Sz)))
    return {
        "n_zones": len(gz),
        "zone_edges": list(ZONE_EDGES),
        "cond_Sigma_zone": cond,
        "I_zone": I_zone,
        "I_zone_diag": I_zone_diag,
        "g_zone": gz.tolist(),
    }


def _rel_spread(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    return float((max(vals) - min(vals)) / np.mean(vals))


# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #
def analyze() -> int:
    rows = _load_rows()
    if not rows:
        print(f"[analyze] no rows under {OUT_DIR}", file=sys.stderr)
        return 1
    lo, hi = CORE_WINDOW
    summary = {
        "checks": ["energy_gradient_exactness", "covariance_conditioning"],
        "profiles": {k: list(v) for k, v in PROFILES.items()},
        "core_window": [lo, hi],
        "fd_h_mev": FD_H_MEV,
        "fd_denominator_mev": 2 * FD_H_MEV,
        "binary": GOV_BIN,
        "acceptance": {
            "check1_core_within_pct": 2.0,
            "check1_flag_layer_pct": 5.0,
            "check2_I_stable_pct": 10.0,
        },
        "fallbacks": {},
        "check1_energy_exactness": {},
        "check2_covariance": {},
    }

    # ---- Check 1 -----------------------------------------------------------
    for prof in PROFILES:
        for role in ("ad", "fdplus", "fdminus"):
            by = _ok_by_seed(rows, prof, role)
            n_fail = sum(1 for r in rows if r.get("profile") == prof
                         and r.get("role") == role and not r.get("ok"))
            summary["fallbacks"][f"{prof}_{role}"] = {
                "n_ok_seeds": len(by), "n_failed_rows": n_fail}
        ad = _ad_aggregate(rows, prof)
        fd = _paired_fd(rows, prof)
        if ad is None or fd is None:
            summary["check1_energy_exactness"][prof] = {
                "status": "incomplete",
                "have_ad": ad is not None, "have_fd": fd is not None}
            print(f"[check1] {prof}: incomplete (ad={ad is not None}, fd={fd is not None})")
            continue
        g_l, g_se_pool, g_se_scat, per_seed_g, n_ad, ad_seeds = ad
        fd_l, fd_se_l, fd_seeds, fd_nev = fd
        ad_se = g_se_pool
        ratio_l, ratio_se_l = _ratio_err(g_l, ad_se, fd_l, fd_se_l)

        # core-window sums (keeps inter-layer covariance via seed scatter for AD)
        ad_core_seed = per_seed_g[:, lo:hi].sum(1)
        ad_core = float(ad_core_seed.mean())
        ad_core_se = (float(ad_core_seed.std(ddof=1) / math.sqrt(len(ad_core_seed)))
                      if len(ad_core_seed) >= 2 else float("nan"))
        # FD core from per-seed paired differences
        bp = _ok_by_seed(rows, prof, "fdplus")
        bm = _ok_by_seed(rows, prof, "fdminus")
        fd_core_seed = []
        for s in fd_seeds:
            d = (np.asarray(bp[s]["mean_E"], float) - np.asarray(bm[s]["mean_E"], float)) / (2 * FD_H_MEV)
            fd_core_seed.append(d[lo:hi].sum())
        fd_core_seed = np.asarray(fd_core_seed)
        fd_core = float(fd_core_seed.mean())
        fd_core_se = float(fd_core_seed.std(ddof=1) / math.sqrt(len(fd_core_seed)))
        core_ratio = ad_core / fd_core
        core_ratio_se = abs(core_ratio) * math.sqrt(
            (ad_core_se / ad_core) ** 2 + (fd_core_se / fd_core) ** 2)

        # per-layer flags: FD-significant layers deviating >5%
        fd_signif = np.abs(fd_l) > 3 * fd_se_l
        dev = np.abs(ratio_l - 1.0)
        flagged = [int(i) for i in range(N_LAYERS)
                   if fd_signif[i] and np.isfinite(ratio_l[i]) and dev[i] > 0.05]

        core_within_2pct = abs(core_ratio - 1.0) <= 0.02
        summary["check1_energy_exactness"][prof] = {
            "n_ad_seeds": len(ad_seeds), "ad_seeds": ad_seeds, "n_ad_events": int(n_ad),
            "n_fd_seeds": len(fd_seeds), "fd_seeds": fd_seeds, "n_fd_events_per_side": fd_nev,
            "core_window": [lo, hi],
            "core_ad": ad_core, "core_ad_se": ad_core_se,
            "core_fd": fd_core, "core_fd_se": fd_core_se,
            "core_ratio_ad_over_fd": core_ratio, "core_ratio_se": core_ratio_se,
            "core_within_2pct": bool(core_within_2pct),
            "n_fd_significant_layers": int(fd_signif.sum()),
            "n_flagged_layers_gt5pct": len(flagged),
            "flagged_layers": flagged,
            "per_layer": {
                "ad": g_l.tolist(), "ad_se": ad_se.tolist(),
                "fd": fd_l.tolist(), "fd_se": fd_se_l.tolist(),
                "ratio_ad_over_fd": ratio_l.tolist(), "ratio_se": ratio_se_l.tolist(),
            },
        }
        print(f"[check1] {prof}: core L{lo}-{hi-1} AD/FD = {core_ratio:.4f} "
              f"+- {core_ratio_se:.4f}  within2%={core_within_2pct}  "
              f"flagged(>5%)={len(flagged)}")

    # ---- Check 2 -----------------------------------------------------------
    for prof in _DUMP_PROFILES:
        by = _ok_by_seed(rows, prof, "ad")
        per_run = {}
        for s in sorted(by):
            r = by[s]
            dpth = r.get("dump_path")
            if not dpth or not Path(dpth).exists():
                continue
            E = _load_dump_E(Path(dpth))
            if E.shape[0] < N_LAYERS + 2:
                continue
            g = np.asarray(r["mean_dE"], float)
            # cross-check: dump per-layer mean vs edeps mean_E
            edeps_mean = np.asarray(r["mean_E"], float)
            dump_mean = E.mean(0)
            mean_rel = float(np.max(np.abs(dump_mean - edeps_mean)
                                    / (np.abs(edeps_mean) + 1e-9)))
            full = _cov_block(E, g)
            zone = _zone_block(E, g)
            per_run[s] = {"dump_mean_vs_edeps_maxrel": mean_rel,
                          "full": full, "zone": zone}
            print(f"[check2] {prof} s{s}: n_ev={full['n_events']} "
                  f"cond(Sigma)={full['cond_Sigma']:.3e} I_full={full['I_full']:.4g} "
                  f"I_diag={full['I_diag']:.4g} I_lw={full['I_lw']:.4g} "
                  f"I_zone={zone['I_zone']:.4g} cond_zone={zone['cond_Sigma_zone']:.3e}")

        # run-to-run stability (seeds 1 vs 2 preferred)
        seeds_avail = sorted(per_run)
        I_full_vals = [per_run[s]["full"]["I_full"] for s in seeds_avail]
        I_lw_vals = [per_run[s]["full"]["I_lw"] for s in seeds_avail]
        I_zone_vals = [per_run[s]["zone"]["I_zone"] for s in seeds_avail]
        spread_full = _rel_spread(I_full_vals)
        spread_lw = _rel_spread(I_lw_vals)
        spread_zone = _rel_spread(I_zone_vals)
        full_stable = (spread_full is not None and spread_full < 0.10)
        zone_stable = (spread_zone is not None and spread_zone < 0.10)
        recommendation = ("full_50x50" if full_stable
                          else ("zone_6x6" if zone_stable else "neither_stable"))
        summary["check2_covariance"][prof] = {
            "seeds": seeds_avail,
            "per_run": per_run,
            "I_full_run_to_run_relspread": spread_full,
            "I_lw_run_to_run_relspread": spread_lw,
            "I_zone_run_to_run_relspread": spread_zone,
            "full_stable_lt10pct": bool(full_stable),
            "zone_stable_lt10pct": bool(zone_stable),
            "recommendation": recommendation,
        }
        print(f"[check2] {prof}: I_full relspread={spread_full} "
              f"I_zone relspread={spread_zone} -> recommend {recommendation}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[analyze] wrote {SUMMARY_PATH}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None)
    ap.add_argument("--role", choices=sorted(ROLES), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--list-units", action="store_true")
    args = ap.parse_args(argv)

    if args.list_units:
        for prof, role, seed in unit_matrix():
            print(f"{prof},{role},{seed}")
        return 0
    if args.analyze:
        return analyze()
    if args.profile is None or args.role is None or args.seed is None:
        ap.error("run mode needs --profile, --role and --seed (or use --analyze)")
    return run_unit(args.profile, args.role, args.seed, args.n_events)


if __name__ == "__main__":
    sys.exit(main())

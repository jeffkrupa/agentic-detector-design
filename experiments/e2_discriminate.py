"""E2 DISCRIMINATION GATE.

Question: does adding geometric structure (more regions K) measurably improve a
longitudinal-profile-shaping objective? We target a FLAT plateau profile (total-
matched, forcing redistribution rather than rescaling) and, for a fixed inner-
iteration budget, optimize per-region absorber/gap thicknesses for K in
{1,2,3,4,6}. More regions = more knobs at the SAME iteration budget -> a fair
test of whether structure helps.

VERDICT: DISCRIMINATES if final profile_loss decreases clearly and roughly
monotonically with K, in particular K=4 loss < 0.5 * K=1 loss.

A robustness check repeats K=1 vs K=4 at extra layer counts (``--robust-layers``).

This script is INCREMENTAL + RESUMABLE: each unit ``(n_layers, K)`` is
checkpointed to a JSONL file the moment its optimization finishes, so a killed
background run loses nothing and a rerun skips completed units.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

from agent.structural_moves import DesignRepresentation
from tools import sim as _sim
from tools.optimizer import optimize_inner_profile
from tools.profile_target import natural_profile, make_plateau_target, profile_loss


# ---- experiment configuration (fixed physics) ---------------------------- #
ENERGY = 10000.0          # 10 GeV e-
PARTICLE = "e-"
ABS0, GAP0 = 2.3, 5.7     # uniform starting thicknesses
CONSTRAINTS = ["1.0 <= a <= 3.5", "3.0 <= g <= 9.0"]
TRUST_REGION = 0.05

_HERE = Path(__file__).resolve().parent
_SUMMARY_JSON = _HERE / "e2_discriminate_summary.json"


def _parse_int_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="E2 discrimination gate (incremental + resumable).")
    p.add_argument("--k", type=str, default="1,2,4",
                   help="comma list of region counts K (default 1,2,4)")
    p.add_argument("--n-layers", type=int, default=20,
                   help="primary number of layers (default 20)")
    p.add_argument("--n-events", type=int, default=500,
                   help="events per profile measurement (default 500)")
    p.add_argument("--max-iters", type=int, default=10,
                   help="inner-optimizer iteration budget (default 10)")
    p.add_argument("--seeds", type=str, default="1",
                   help="comma list of seeds (default 1)")
    p.add_argument("--out-jsonl", type=str,
                   default=str(_HERE / "e2_discriminate_results.jsonl"),
                   help="JSONL checkpoint file (default "
                        "experiments/e2_discriminate_results.jsonl)")
    p.add_argument("--robust-layers", type=str, default="",
                   help="comma list of extra layer counts; for each, also run "
                        "K=1 and K=4 as extra units (default empty)")
    return p.parse_args(argv)


def build_units(n_layers, k_list, robust_layers):
    """Ordered list of (n_layers, K) units, de-duplicated, order preserved."""
    units = []
    seen = set()

    def add(nl, K):
        key = (int(nl), int(K))
        if key not in seen:
            seen.add(key)
            units.append(key)

    for K in k_list:
        add(n_layers, K)
    for rl in robust_layers:
        add(rl, 1)
        add(rl, 4)
    return units


def load_completed(out_path: Path):
    """Set of completed (n_layers, K) keys read from an existing JSONL."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add((int(row["n_layers"]), int(row["K"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def natural_and_target(n_layers, n_events, seeds, ctrl):
    """Natural profile + flat (total-matched) target for a uniform-1 base.

    Depends only on ``n_layers`` (the uniform init design), so it is computed
    once per distinct layer count and reused across that layer's K units.
    """
    base = DesignRepresentation.uniform(
        n_layers, 1, ABS0, GAP0, ENERGY, PARTICLE)
    natural = natural_profile(base, n_events, seeds, ctrl=ctrl)
    target = make_plateau_target(natural, total_match=True)
    return natural, target


def run_unit(n_layers, K, target, natural, n_events, max_iters, seeds, ctrl):
    """Optimize one (n_layers, K) unit; return a result-row dict."""
    rep = DesignRepresentation.uniform(
        n_layers, K, ABS0, GAP0, ENERGY, PARTICLE)

    init_E = natural_profile(rep, n_events, seeds, ctrl=ctrl)
    init_loss = float(profile_loss(init_E, target))

    res = optimize_inner_profile(
        rep, target, constraints=CONSTRAINTS, max_iters=max_iters,
        n_events=n_events, seeds=seeds, ctrl=ctrl,
        trust_region=TRUST_REGION)

    final_loss = float(res.final_objective)
    return {
        "n_layers": int(n_layers),
        "K": int(K),
        "n_regions": int(res.final_rep.n_regions),
        "final_loss": final_loss,
        "initial_uniform_loss": init_loss,
        "n_events": int(n_events),
        "max_iters": int(max_iters),
        "seeds": list(seeds),
        "natural_total": float(np.asarray(natural).sum()),
        "target_level": float(np.asarray(target)[0]),
        "converged": bool(res.converged),
        "n_iters": int(len(res.history)),
    }


def append_row(out_path: Path, row: dict):
    """Append one JSON line, flush + fsync so a kill can't lose it."""
    with open(out_path, "a", buffering=1) as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(out_path: Path):
    rows = []
    if not out_path.exists():
        return rows
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _verdict_for_layer(rows_by_k, k_list):
    """Return (table, discriminates) for one layer count's K rows."""
    table = []
    ks = [K for K in k_list if K in rows_by_k]
    base = rows_by_k[ks[0]]["final_loss"] if ks else None
    for K in ks:
        fl = rows_by_k[K]["final_loss"]
        ratio = (fl / base) if (base and base > 0) else float("nan")
        table.append({"K": K, "final_loss": fl, "ratio": ratio})
    fl = [rows_by_k[K]["final_loss"] for K in ks]
    mostly_monotone = all(
        fl[i + 1] <= fl[i] * 1.10 for i in range(len(fl) - 1))
    k4_strong = (1 in rows_by_k and 4 in rows_by_k and
                 rows_by_k[4]["final_loss"] < 0.5 * rows_by_k[1]["final_loss"])
    discriminates = bool(k4_strong and mostly_monotone)
    return table, discriminates, mostly_monotone


def final_verdict(out_path, n_layers, k_list, robust_layers):
    """Read JSONL back, print verdict table(s), write summary JSON."""
    rows = read_rows(out_path)
    by_layer = {}
    for r in rows:
        by_layer.setdefault(int(r["n_layers"]), {})[int(r["K"])] = r

    summary = {"primary": {}, "robust": {}}

    prim = by_layer.get(n_layers, {})
    table, discriminates, mostly_monotone = _verdict_for_layer(prim, k_list)
    print(f"\n=== PRIMARY n_layers={n_layers} ===")
    print(f"{'K':>4} {'final_loss':>14} {'ratio_vs_K1':>14}")
    for t in table:
        print(f"{t['K']:>4} {t['final_loss']:>14.6f} {t['ratio']:>14.4f}")
    if 1 in prim and 4 in prim:
        l1, l4 = prim[1]["final_loss"], prim[4]["final_loss"]
        print(f"K=4 vs K=1: {l4:.6f} vs {l1:.6f} "
              f"(ratio={l4/l1:.4f}; threshold < 0.5)")
    print(f"mostly-monotone in K: {mostly_monotone}")
    print(f"DISCRIMINATES: {'YES' if discriminates else 'NO'}")
    summary["primary"] = {
        "n_layers": n_layers, "k_list": list(k_list),
        "table": table, "mostly_monotone": mostly_monotone,
        "discriminates": discriminates,
    }

    for rl in robust_layers:
        rrows = by_layer.get(rl, {})
        rtable, rdisc, _ = _verdict_for_layer(rrows, [1, 4])
        print(f"\n=== ROBUST n_layers={rl} (K=1 vs K=4) ===")
        for t in rtable:
            print(f"{t['K']:>4} {t['final_loss']:>14.6f} "
                  f"{t['ratio']:>14.4f}")
        if 1 in rrows and 4 in rrows:
            l1, l4 = rrows[1]["final_loss"], rrows[4]["final_loss"]
            print(f"[n_layers={rl}] K=4 vs K=1 {l4:.6f} vs {l1:.6f} "
                  f"(ratio={l4/l1:.4f}) DISCRIMINATES: "
                  f"{'YES' if rdisc else 'NO'}")
        summary["robust"][str(rl)] = {
            "n_layers": rl, "table": rtable, "discriminates": rdisc}

    with open(_SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsaved summary -> {_SUMMARY_JSON}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    k_list = _parse_int_list(args.k)
    seeds = tuple(_parse_int_list(args.seeds))
    robust_layers = _parse_int_list(args.robust_layers)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)

    units = build_units(args.n_layers, k_list, robust_layers)
    completed = load_completed(out_path)

    to_run = [u for u in units if u not in completed]
    skipped = [u for u in units if u in completed]
    print(f"[resume] out={out_path}")
    print(f"[resume] units total={len(units)} "
          f"skip={len(skipped)} run={len(to_run)}")
    if skipped:
        print(f"[resume] SKIPPED (already done): "
              f"{', '.join(f'(nl={nl},K={K})' for nl, K in skipped)}")
    if to_run:
        print(f"[resume] TO RUN: "
              f"{', '.join(f'(nl={nl},K={K})' for nl, K in to_run)}")
    sys.stdout.flush()

    # group units to run by layer count so natural/target is computed once
    nat_cache = {}
    for nl, K in to_run:
        if nl not in nat_cache:
            nat_cache[nl] = natural_and_target(
                nl, args.n_events, seeds, ctrl)
        natural, target = nat_cache[nl]

        row = run_unit(nl, K, target, natural, args.n_events,
                       args.max_iters, seeds, ctrl)
        append_row(out_path, row)
        print(f"[done] nl={nl} K={K} n_regions={row['n_regions']} "
              f"final_loss={row['final_loss']:.6f} "
              f"init_loss={row['initial_uniform_loss']:.6f} "
              f"-> appended to {out_path}")
        sys.stdout.flush()

    # final verdict only when every requested unit is present
    completed_after = load_completed(out_path)
    all_done = all(u in completed_after for u in units)
    if all_done:
        final_verdict(out_path, args.n_layers, k_list, robust_layers)
    else:
        missing = [u for u in units if u not in completed_after]
        print(f"[verdict] skipped: {len(missing)} unit(s) still missing "
              f"-> recompute later from {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

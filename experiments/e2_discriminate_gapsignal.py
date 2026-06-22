"""E2 DISCRIMINATION GATE -- minimum-length GAP-SIGNAL (sampled-energy) objective.

Question: does adding geometric structure (more regions K) measurably improve a
minimum-length objective defined on the SAMPLED (gap) signal? Phase A made the
per-layer GAP energy a differentiable output; this gate tests whether an
objective on ``Evis = sum_l E_gap,l`` (the energy actually read out of the active
gaps) finally rewards longitudinal structure -- the previous combined-energy
objectives did not.

We require the sampled energy ``Evis >= evis_target`` while paying a per-layer
LENGTH cost on GAP thickness only, and for a fixed inner-iteration budget we
optimize per-region absorber/gap thicknesses for K in {1,2,4,...}. More regions =
more knobs at the SAME iteration budget -> a fair test of whether structure
helps. Because the length cost is paid per layer while the gap-signal gradient is
concentrated in the front (shower) layers, the optimum is NON-UNIFORM (thick
front, thin tail), which structure can exploit and a single uniform region
cannot.

VERDICT: DISCRIMINATES if final gap-signal loss decreases clearly and roughly
monotonically with K, in particular K=max loss < 0.5 * K=1 loss.

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
from tools.optimizer import optimize_inner_gapsignal
from tools.gap_signal_target import total_gap_signal, gap_signal_loss


# ---- experiment configuration (fixed physics) ---------------------------- #
ENERGY = 10000.0          # 10 GeV e-
PARTICLE = "e-"
ABS0, GAP0 = 2.3, 5.7     # uniform starting thicknesses
CONSTRAINTS = ["1.0 <= a <= 3.5", "3.0 <= g <= 9.0"]
TRUST_REGION = 0.05

_HERE = Path(__file__).resolve().parent
_SUMMARY_JSON = _HERE / "e2_discriminate_gapsignal_summary.json"


def _parse_int_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="E2 gap-signal discrimination gate (incremental + resumable).")
    p.add_argument("--k", type=str, default="1,2,4",
                   help="comma list of region counts K (default 1,2,4)")
    p.add_argument("--n-layers", type=int, default=20,
                   help="primary number of layers (default 20)")
    p.add_argument("--n-events", type=int, default=500,
                   help="events per gap-signal measurement (default 500)")
    p.add_argument("--max-iters", type=int, default=12,
                   help="inner-optimizer iteration budget (default 12)")
    p.add_argument("--seeds", type=str, default="1",
                   help="comma list of seeds (default 1)")
    p.add_argument("--evis-target", type=float, default=None,
                   help="target sampled energy Evis = sum_l E_gap,l [MeV]. If "
                        "unset, set PER UNIT at runtime to 1.3x the uniform-init "
                        "Evis (a reachable stretch that makes the relu active).")
    p.add_argument("--lam", type=float, default=1e4,
                   help="gap-signal penalty weight lambda (default 1e4)")
    p.add_argument("--mu", type=float, default=1.0,
                   help="length-cost weight mu (default 1.0)")
    p.add_argument("--trust-region", type=float, default=TRUST_REGION,
                   help="max single-thickness change per inner iteration [mm] "
                        f"(default {TRUST_REGION}); raise to take bigger steps "
                        "with fewer iterations")
    p.add_argument("--lr", type=float, default=None,
                   help="base learning rate for the inner GD step (default: "
                        "optimizer module default; step is trust-region capped)")
    p.add_argument("--out-jsonl", type=str,
                   default=str(_HERE / "e2_discriminate_gapsignal_results.jsonl"),
                   help="JSONL checkpoint file (default "
                        "experiments/e2_discriminate_gapsignal_results.jsonl)")
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


def run_unit(n_layers, K, evis_target_cli, lam, mu,
             n_events, max_iters, seeds, ctrl,
             trust_region=TRUST_REGION, lr=None):
    """Optimize one (n_layers, K) unit; return a result-row dict.

    If ``evis_target_cli`` is None, the target is set PER UNIT to 1.3x the
    uniform-init Evis so the relu is active and the stretch is reachable.
    """
    rep = DesignRepresentation.uniform(
        n_layers, K, ABS0, GAP0, ENERGY, PARTICLE)

    init_evis = total_gap_signal(rep.to_design_point(), n_events, seeds, ctrl=ctrl)
    if evis_target_cli is None:
        evis_target = 1.3 * float(init_evis)
    else:
        evis_target = float(evis_target_cli)
    init_loss = float(gap_signal_loss(init_evis, evis_target, lam, mu, rep.regions))

    opt_kwargs = dict(
        constraints=CONSTRAINTS, max_iters=max_iters, n_events=n_events,
        seeds=seeds, ctrl=ctrl, trust_region=trust_region)
    if lr is not None:
        opt_kwargs["lr"] = lr
    res = optimize_inner_gapsignal(rep, evis_target, lam, mu, **opt_kwargs)

    final_loss = float(res.final_objective)
    final_evis = float("nan")
    if res.history:
        final_evis = float(res.history[-1].get("evis", float("nan")))

    final_abs = [float(r.absorber_mm) for r in res.final_rep.regions]
    final_gap = [float(r.gap_mm) for r in res.final_rep.regions]
    return {
        "n_layers": int(n_layers),
        "K": int(K),
        "n_regions": int(res.final_rep.n_regions),
        "final_loss": final_loss,
        "final_evis": final_evis,
        "final_absorber_mm": final_abs,
        "final_gap_mm": final_gap,
        "initial_uniform_loss": init_loss,
        "initial_evis": float(init_evis),
        "evis_target": float(evis_target),
        "lam": float(lam),
        "mu": float(mu),
        "n_events": int(n_events),
        "max_iters": int(max_iters),
        "trust_region": float(trust_region),
        "seeds": list(seeds),
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
    """Return (table, discriminates, mostly_monotone) for one layer's K rows."""
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
    # K=max strong vs K=1: final loss at the LARGEST available K beats half of K=1
    kmax = ks[-1] if ks else None
    kmax_strong = (ks and 1 in rows_by_k and kmax != 1 and
                   rows_by_k[kmax]["final_loss"] < 0.5 * rows_by_k[1]["final_loss"])
    discriminates = bool(kmax_strong and mostly_monotone)
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
    print(f"{'K':>4} {'final_loss':>14} {'ratio_vs_K1':>14} {'evis':>12}")
    for t in table:
        ev = prim[t["K"]].get("final_evis", float("nan"))
        print(f"{t['K']:>4} {t['final_loss']:>14.6f} {t['ratio']:>14.4f} "
              f"{ev:>12.2f}")
    ks = [K for K in k_list if K in prim]
    kmax = ks[-1] if ks else None
    if ks and 1 in prim and kmax != 1:
        l1, lk = prim[1]["final_loss"], prim[kmax]["final_loss"]
        ratio = (lk / l1) if l1 else float("nan")
        print(f"K={kmax} vs K=1: {lk:.6f} vs {l1:.6f} "
              f"(ratio={ratio:.4f}; threshold < 0.5)")
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

    for nl, K in to_run:
        row = run_unit(nl, K, args.evis_target, args.lam, args.mu,
                       args.n_events, args.max_iters, seeds, ctrl,
                       trust_region=args.trust_region, lr=args.lr)
        append_row(out_path, row)
        print(f"[done] nl={nl} K={K} n_regions={row['n_regions']} "
              f"final_loss={row['final_loss']:.6f} "
              f"final_evis={row['final_evis']:.2f} "
              f"evis_target={row['evis_target']:.2f} "
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

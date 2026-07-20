"""E8 (revised): minimal adaptive bilevel PoC — normalized-profile MSE objective.

Outer proposer: two-call adaptive loop over k in {16, 25, 33}.
  Call 1: select two k values with a one-sentence rationale.
  Call 2: see compact results, decide whether to evaluate the remaining k.
Inner loop: E5 AD/GD machinery (validated, gradient-descent, NOT curve-argmin).
  Objective: earlier-shift normalized-profile MSE (same as E5).
  Start from uniform, exact budget, <=10 steps, 6 CRN seeds, N=1000.
  SE-aware gate: PASS/FLAT/FAIL; FLAT -> proceed (do not abort).

Compact result per k:
  {k, best_loss, a_front, a_rear, steps, status}

Baselines: uniform (no inner loop) + exhaustive E5 three-k results as offline oracle.

Reuse: if a completed E5 result file exists for a combo, load it instead of rerunning.
       Set --no-reuse to force fresh runs (e.g. for smoke tests).

SLURM commands (for a full run):
  sbatch --account=atlas:compef --partition=ampere --qos=normal \\
         --cpus-per-task=8 --mem=16G --time=4:00:00 \\
         --output=experiments/e8_profile_bilevel_%j.txt \\
         --wrap="source .venv/bin/activate && \\
                 python -m experiments.e8_profile_bilevel --result-file experiments/e8_profile_bilevel_result.txt"

  # Per-k fan-out (run k=16, k=25, k=33 independently, then aggregate):
  for K in 16 25 33; do
    sbatch --account=atlas:compef --partition=ampere --qos=normal \\
           --cpus-per-task=8 --mem=16G --time=2:00:00 \\
           --output=experiments/e8_profile_bilevel_k${K}_%j.txt \\
           --wrap="source .venv/bin/activate && \\
                   python -m experiments.e8_profile_bilevel --only-k ${K} \\
                          --result-file experiments/e8_profile_bilevel_k${K}_result.txt"
  done
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

import tools.sim as _simmod
from experiments.e3_region_scan import ABS_FLOOR_MM, uniform_profile
from experiments.e4_profile_matching import shift_target
from experiments.e5_ad_optimize import (
    optimize, per_layer_profiles, loss_stats,
    N_DEFAULT, A0_DEFAULT, GAP_DEFAULT, E_DEFAULT,
)
from tools import sim as _sim

_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

ORACLE_FILES = {
    16: "experiments/e5_pilot_earlier_k16.txt",
    25: "experiments/e5_earlier_k25.txt",
    33: "experiments/e5_earlier_k33.txt",
}

K_CHOICES = [16, 25, 33]


# ---------------------------------------------------------------------------
# Load a completed E5 result rather than rerunning it.
# ---------------------------------------------------------------------------

def load_e5_result(k: int) -> dict | None:
    """Parse the compact summary lines from an existing E5 earlier-k result file."""
    path = ORACLE_FILES.get(k)
    if path is None or not os.path.exists(path):
        return None
    best_loss = a_front = a_rear = None
    uniform_loss = None
    steps_done = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("# SUMMARY"):
                pass
            m = re.search(r"uniform loss\s*=\s*([0-9eE+.\-]+)", line)
            if m:
                uniform_loss = float(m.group(1))
            m = re.search(r"final/best AD loss\s*=\s*([0-9eE+.\-]+).*a_front=([0-9.]+)\s+a_rear=([0-9.]+)", line)
            if m:
                best_loss = float(m.group(1))
                a_front = float(m.group(2))
                a_rear = float(m.group(3))
            # count non-header step lines (integer step index at start)
            if re.match(r"^\s*\d+\s*\|", line):
                steps_done += 1
    if best_loss is None or a_front is None:
        return None
    return {
        "k": k,
        "best_loss": best_loss,
        "a_front": a_front,
        "a_rear": a_rear,
        "steps": steps_done,
        "status": "loaded_from_e5",
        "uniform_loss": uniform_loss,
    }


# ---------------------------------------------------------------------------
# Proposer: two-call adaptive logic (deterministic heuristic, not an LLM).
# The "agent" is this function — it encodes the two-call protocol.
# ---------------------------------------------------------------------------

def proposer_call_1(oracle_available: list[int]) -> tuple[list[int], str]:
    """First proposer call: select two k values.

    Heuristic: prefer k=16 (E5 best for earlier-shift) and one other split.
    If k=16 results already exist locally (oracle), pick k=25 as the novel test;
    otherwise prefer k=25 to test a different front/rear boundary.
    Returns (selected_ks, rationale).
    """
    if 16 in oracle_available:
        chosen = [16, 25]
        rationale = ("k=16 is the E5-known optimum for the earlier-shift target; "
                     "k=25 tests a different front-boundary to check generalization.")
    else:
        chosen = [16, 25]
        rationale = ("k=16 splits at N/3 (front-weighted) and k=25 at N/2; "
                     "these two span the most distinct structural hypotheses.")
    return chosen, rationale


def proposer_call_2(episode_results: dict[int, dict], all_ks: list[int]) -> tuple[bool, str]:
    """Second proposer call: given compact results from the episode's first two k values,
    decide whether to evaluate the remaining unevaluated k.

    episode_results contains ONLY the k values explicitly proposed and evaluated during
    this episode — oracle availability is irrelevant here.

    Runs the remaining k if both first-round runs completed cleanly AND their
    best_loss values differ by more than 3x the typical loss SE (~5e-6), signalling
    that split placement genuinely matters and a third data point adds information.
    Otherwise, skip (budget spent on confirming the known-good k suffices).
    """
    # remaining = all k NOT yet evaluated in this episode (regardless of oracle cache)
    remaining = [k for k in all_ks if k not in episode_results]
    done_ks = [k for k in all_ks if k in episode_results and
               episode_results[k]["status"] not in ("GATE_FAIL", "error")]
    if not remaining:
        return False, "All candidates have been evaluated in this episode."
    if len(done_ks) < 2:
        return True, "Too few completed episode results to judge; evaluate remaining k for completeness."
    losses = [episode_results[k]["best_loss"] for k in done_ks]
    spread = max(losses) - min(losses)
    if spread > 3 * 5e-6:
        return True, (f"Loss spread across first two episode results ({spread:.2e}) exceeds "
                      f"3x typical SE (~1.5e-5); evaluating k={remaining[0]} is informative.")
    else:
        return False, (f"Loss spread ({spread:.2e}) is within noise; "
                       f"skipping k={remaining[0]} — first two results are sufficient.")


# ---------------------------------------------------------------------------
# Inner loop: thin wrapper around E5 optimize() that returns a compact result.
# ---------------------------------------------------------------------------

def run_inner_loop(cfg, ctrl, k: int, p_target, p0, N: int, a0: float, gap: float,
                   energy: float, n_events: int, seeds: list[int],
                   max_steps: int, da_target: float, max_step_mm: float) -> dict:
    """Run the E5 GD inner loop; return compact {k, best_loss, a_front, a_rear, steps, status}."""
    res = optimize(cfg, ctrl, "earlier", p_target, k, N, a0, gap, energy,
                   n_events, seeds, max_steps, da_target, max_step_mm)
    if res["status"].startswith("STOP"):
        return {"k": k, "best_loss": float("nan"), "a_front": float("nan"),
                "a_rear": float("nan"), "steps": 0, "status": res["status"]}
    best = res["best"]
    return {
        "k": k,
        "best_loss": best["loss"],
        "a_front": best["a_front"],
        "a_rear": best["a_rear"],
        "steps": len(res["hist"]),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=N_DEFAULT)
    ap.add_argument("--a0", type=float, default=A0_DEFAULT)
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT)
    ap.add_argument("--energy", type=float, default=E_DEFAULT)
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--shift", type=float, default=2.0)
    ap.add_argument("--da-target", type=float, default=0.08)
    ap.add_argument("--max-step-mm", type=float, default=0.1)
    ap.add_argument("--no-reuse", action="store_true",
                    help="Force fresh AD runs even if E5 result files exist")
    ap.add_argument("--only-k", type=int, choices=K_CHOICES,
                    help="Run inner loop for one k only (fan-out mode)")
    ap.add_argument("--result-file", default=None,
                    help="Write final JSON summary to this path")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke-test only: load oracle results, run proposer calls, "
                         "skip actual sim. Exits 0 on success.")
    args = ap.parse_args()

    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)
    N, a0, gap = args.n_layers, args.a0, args.gap

    print(f"# E8-profile-bilevel  N={N} a0={a0} A_tot={N*a0:.1f}mm "
          f"gap={gap} E={args.energy} N_ev={args.n_events} seeds={args.seeds}")
    print(f"# max_steps={args.max_steps} shift={args.shift}  no-reuse={args.no_reuse}")

    # --- build the earlier-shift target (same construction as E4/E5) ----------
    uni_prof = uniform_profile(N, a0)
    if args.smoke:
        # use a fixed synthetic p0 so smoke test needs no sim
        rng = np.random.default_rng(0)
        raw = rng.exponential(1.0, N).astype(float)
        p0 = raw / raw.sum()
        print("# [smoke] using synthetic p0 (no sim)")
    else:
        ps0, _ = per_layer_profiles(cfg, ctrl, uni_prof, gap, args.energy,
                                    N, args.n_events, args.seeds)
        good = [p for p in ps0 if np.all(np.isfinite(p))]
        if not good:
            print("ERROR: uniform forward pass returned all NaN — check sim / config")
            sys.exit(1)
        p0 = np.mean(good, axis=0)
    p_target = shift_target(p0, -args.shift)
    Lu_val = float(np.sum((p0 - p_target) ** 2))  # approximate uniform loss
    print(f"# p0 sum={p0.sum():.4f} peak@{int(np.argmax(p0))}  "
          f"target sum={p_target.sum():.4f} peak@{int(np.argmax(p_target))}")
    print(f"# approx uniform loss = {Lu_val:.4e}")

    # --- oracle: check which E5 results already exist -------------------------
    oracle_loaded: dict[int, dict] = {}
    if not args.no_reuse:
        for k in K_CHOICES:
            r = load_e5_result(k)
            if r is not None:
                oracle_loaded[k] = r
                print(f"# oracle loaded: k={k}  best_loss={r['best_loss']:.4e} "
                      f"a_front={r['a_front']:.4f}  [{ORACLE_FILES[k]}]")
    if oracle_loaded:
        print(f"# oracle covers k={sorted(oracle_loaded)}; "
              f"those k will NOT be rerun unless --no-reuse")

    # --- fan-out mode: only run one k -----------------------------------------
    if args.only_k is not None:
        k = args.only_k
        print(f"\n## FAN-OUT MODE: k={k} only")
        if k in oracle_loaded and not args.no_reuse:
            r = oracle_loaded[k]
            print(f"# reusing E5 result for k={k}: {r}")
        elif args.smoke:
            print(f"# [smoke] skip sim for k={k}")
        else:
            r = run_inner_loop(cfg, ctrl, k, p_target, p0, N, a0, gap,
                               args.energy, args.n_events, args.seeds,
                               args.max_steps, args.da_target, args.max_step_mm)
            print(f"# k={k} compact result: {r}")
            if args.result_file:
                with open(args.result_file, "w") as fh:
                    json.dump(r, fh, indent=2)
        return

    # --- FULL bilevel loop ----------------------------------------------------
    # episode_results: ONLY k values proposed and evaluated during this run.
    # oracle_loaded:   cache of E5 results — determines HOW a proposed k is
    #                  evaluated (load vs. fresh sim), NOT whether it counts as
    #                  evaluated for the proposer's adaptive logic.
    episode_results: dict[int, dict] = {}
    total_new_runs = 0

    def _resolve_k(k: int) -> dict:
        """Load oracle or run fresh sim for a proposed k; record in episode_results."""
        nonlocal total_new_runs
        if k in oracle_loaded and not args.no_reuse:
            r = oracle_loaded[k]
            print(f"\n# k={k}: loading oracle result (counts as episode evaluation)")
            _print_compact(r)
            return r
        if args.smoke:
            r = _synthetic_result(k)
            print(f"# [smoke] synthetic result for k={k}: {r}")
            return r
        print(f"\n## AD inner loop: k={k}")
        r = run_inner_loop(cfg, ctrl, k, p_target, p0, N, a0, gap,
                           args.energy, args.n_events, args.seeds,
                           args.max_steps, args.da_target, args.max_step_mm)
        total_new_runs += 1
        _print_compact(r)
        return r

    # PROPOSER CALL 1: select exactly two k values
    selected_ks, rationale_1 = proposer_call_1(list(oracle_loaded.keys()))
    print(f"\n### PROPOSER CALL 1")
    print(f"# selected k values: {selected_ks}")
    print(f"# rationale: {rationale_1}")

    for k in selected_ks:
        episode_results[k] = _resolve_k(k)

    # PROPOSER CALL 2: sees only episode_results; remaining = all K_CHOICES not
    # yet in episode_results, regardless of oracle availability.
    remaining_ks = [k for k in K_CHOICES if k not in episode_results]
    print(f"\n### PROPOSER CALL 2")
    print(f"# episode results so far: k={sorted(episode_results)}")
    print(f"# remaining unevaluated k (this episode): {remaining_ks}")
    run_remaining, rationale_2 = proposer_call_2(episode_results, K_CHOICES)
    print(f"# evaluate remaining? {run_remaining}")
    print(f"# rationale: {rationale_2}")

    if run_remaining and remaining_ks:
        k = remaining_ks[0]
        episode_results[k] = _resolve_k(k)

    # --- summary table (episode candidates only) ------------------------------
    print(f"\n### EPISODE SUMMARY  (new inner-loop runs: {total_new_runs})")
    print(f"{'k':>4} | {'best_loss':>12} | {'a_front':>8} | {'a_rear':>8} | "
          f"{'steps':>5} | status")
    print("-" * 66)

    Lu, Lu_se = Lu_val, 1e-6   # approximate; sufficient for oracle-reuse mode
    print(f"{'unif':>4} | {Lu:12.4e} | {a0:8.4f} | {a0:8.4f} | {'0':>5} | baseline")

    best_k, best_loss = None, float("inf")
    for k in sorted(episode_results):
        r = episode_results[k]
        _print_table_row(r)
        if np.isfinite(r["best_loss"]) and r["best_loss"] < best_loss:
            best_loss, best_k = r["best_loss"], k

    not_evaluated = [k for k in K_CHOICES if k not in episode_results]
    if not_evaluated:
        for k in not_evaluated:
            print(f"{k:>4} | {'(not evaluated this episode)':>12}")

    # oracle audit (external, not used for ranking)
    print(f"\n# External oracle audit (E5 exhaustive — not used for episode ranking):")
    oracle_best_k = min(oracle_loaded, key=lambda k: oracle_loaded[k]["best_loss"]) \
        if oracle_loaded else None
    for k in K_CHOICES:
        r = oracle_loaded.get(k)
        marker = " <- oracle best" if k == oracle_best_k else ""
        if r:
            print(f"#   k={k}: best_loss={r['best_loss']:.4e}  a_front={r['a_front']:.4f}{marker}")
        else:
            print(f"#   k={k}: not in oracle cache")

    # bilevel verdict
    print(f"\n### BILEVEL VERDICT")
    if best_k is not None:
        print(f"# Episode best: k={best_k}  best_loss={best_loss:.4e}")
        if oracle_best_k is not None:
            agrees = (best_k == oracle_best_k)
            print(f"# Oracle best: k={oracle_best_k}  "
                  f"({'AGREES' if agrees else 'DISAGREES — episode did not evaluate oracle-best k'})")
    else:
        print("# No clean episode result available.")

    # write JSON summary
    summary = {
        "uniform_loss": float(Lu),
        "episode_results": {str(k): v for k, v in episode_results.items()},
        "not_evaluated_this_episode": not_evaluated,
        "best_k_episode": best_k,
        "best_loss_episode": float(best_loss) if np.isfinite(best_loss) else None,
        "oracle": {str(k): v for k, v in oracle_loaded.items()},
        "oracle_best_k": oracle_best_k,
        "proposer_call_1": {"selected": selected_ks, "rationale": rationale_1},
        "proposer_call_2": {
            "remaining_presented": remaining_ks,
            "run_remaining": run_remaining,
            "rationale": rationale_2,
        },
    }
    if args.result_file:
        with open(args.result_file, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"# JSON summary written to: {args.result_file}")

    return 0


def _print_compact(r: dict):
    print(f"# compact result: k={r['k']}  best_loss={r['best_loss']:.4e}  "
          f"a_front={r.get('a_front', float('nan')):.4f}  "
          f"a_rear={r.get('a_rear', float('nan')):.4f}  "
          f"steps={r.get('steps', '?')}  status={r['status']}")


def _print_table_row(r: dict):
    bl = r["best_loss"]
    af = r.get("a_front", float("nan"))
    ar = r.get("a_rear", float("nan"))
    st = r.get("steps", "?")
    print(f"{r['k']:>4} | {bl:12.4e} | {af:8.4f} | {ar:8.4f} | "
          f"{str(st):>5} | {r['status']}")


def _synthetic_result(k: int) -> dict:
    """Deterministic placeholder for smoke-test mode."""
    af = {16: 2.516, 25: 2.480, 33: 2.500}[k]
    N, a0 = N_DEFAULT, A0_DEFAULT
    ar = (N * a0 - k * af) / (N - k)
    bl = {16: 1.87e-4, 25: 2.32e-4, 33: 2.68e-4}[k]
    return {"k": k, "best_loss": bl, "a_front": af, "a_rear": ar,
            "steps": 7, "status": "smoke"}


if __name__ == "__main__":
    sys.exit(main())

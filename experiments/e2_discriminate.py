"""E2 DISCRIMINATION GATE.

Question: does adding geometric structure (more regions K) measurably improve a
longitudinal-profile-shaping objective? We target a FLAT plateau profile (total-
matched, forcing redistribution rather than rescaling) and, for a fixed inner-
iteration budget, optimize per-region absorber/gap thicknesses for K in
{1,2,3,4,6}. More regions = more knobs at the SAME iteration budget -> a fair
test of whether structure helps.

VERDICT: DISCRIMINATES if final profile_loss decreases clearly and roughly
monotonically with K, in particular K=4 loss < 0.5 * K=1 loss.

A robustness check repeats K=1 vs K=4 at n_layers=30.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from agent.structural_moves import DesignRepresentation
from tools import sim as _sim
from tools.optimizer import optimize_inner_profile
from tools.profile_target import natural_profile, make_plateau_target, profile_loss


# ---- experiment configuration -------------------------------------------- #
ENERGY = 10000.0          # 10 GeV e-
PARTICLE = "e-"
ABS0, GAP0 = 2.3, 5.7     # uniform starting thicknesses
CONSTRAINTS = ["1.0 <= a <= 3.5", "3.0 <= g <= 9.0"]
N_EVENTS = 800
SEEDS = (1,)
MAX_ITERS = 15
TRUST_REGION = 0.05
K_LIST = [1, 2, 3, 4, 6]

_HERE = Path(__file__).resolve().parent
_RESULT_JSON = _HERE / "e2_discriminate_result.json"


def run_for_n_layers(n_layers: int, k_list, ctrl):
    """Return (natural, target, rows) where rows[k] has init/final losses."""
    # Natural profile + flat (total-matched) plateau target from a uniform-1 base.
    base = DesignRepresentation.uniform(
        n_layers, 1, ABS0, GAP0, ENERGY, PARTICLE)
    natural = natural_profile(base, N_EVENTS, SEEDS, ctrl=ctrl)
    target = make_plateau_target(natural, total_match=True)

    rows = {}
    for K in k_list:
        rep = DesignRepresentation.uniform(
            n_layers, K, ABS0, GAP0, ENERGY, PARTICLE)
        # initial (no-optimization) loss for reference
        init_E = natural_profile(rep, N_EVENTS, SEEDS, ctrl=ctrl)
        init_loss = profile_loss(init_E, target)

        res = optimize_inner_profile(
            rep, target, constraints=CONSTRAINTS, max_iters=MAX_ITERS,
            n_events=N_EVENTS, seeds=SEEDS, ctrl=ctrl,
            trust_region=TRUST_REGION)
        final_loss = float(res.final_objective)
        final_E = natural_profile(res.final_rep, N_EVENTS, SEEDS, ctrl=ctrl)
        rows[K] = {
            "K": K,
            "n_regions": res.final_rep.n_regions,
            "init_loss": float(init_loss),
            "final_loss": final_loss,
            "final_profile": final_E.tolist(),
            "converged": bool(res.converged),
            "n_iters": len(res.history),
        }
    return natural, target, rows


def print_table(n_layers, natural, target, rows, k_list):
    print(f"\n=== n_layers={n_layers}  e- @ {ENERGY/1000:.0f} GeV  "
          f"n_events={N_EVENTS} seeds={SEEDS} max_iters={MAX_ITERS} ===")
    print(f"natural total={natural.sum():.2f} MeV   "
          f"target (flat) const={target[0]:.4f} MeV/layer   "
          f"target total={target.sum():.2f} MeV")
    base = rows[k_list[0]]["final_loss"]
    print(f"{'K':>3} {'n_reg':>6} {'init_loss':>12} {'final_loss':>12} "
          f"{'ratio_vs_K1':>12} {'iters':>6} {'conv':>5}")
    for K in k_list:
        r = rows[K]
        ratio = r["final_loss"] / base if base > 0 else float("nan")
        print(f"{K:>3} {r['n_regions']:>6} {r['init_loss']:>12.5f} "
              f"{r['final_loss']:>12.5f} {ratio:>12.4f} "
              f"{r['n_iters']:>6} {str(r['converged']):>5}")


def main():
    cfg = _sim.load_config()
    ctrl = _sim.ctrl_flags_from_config(cfg)

    # ---- primary gate: n_layers=20 over full K_LIST ---------------------- #
    nat20, tgt20, rows20 = run_for_n_layers(20, K_LIST, ctrl)
    print_table(20, nat20, tgt20, rows20, K_LIST)

    loss_k1 = rows20[1]["final_loss"]
    loss_k4 = rows20[4]["final_loss"]
    # monotone-ish: each step does not increase loss by more than 10%
    fl = [rows20[K]["final_loss"] for K in K_LIST]
    mostly_monotone = all(fl[i + 1] <= fl[i] * 1.10 for i in range(len(fl) - 1))
    k4_strong = (loss_k4 < 0.5 * loss_k1)
    discriminates = bool(k4_strong and mostly_monotone)

    print(f"\nK=4 vs K=1: final_loss {loss_k4:.5f} vs {loss_k1:.5f} "
          f"(ratio={loss_k4/loss_k1:.4f}; threshold < 0.5)")
    print(f"mostly-monotone decrease across K: {mostly_monotone}")
    print(f"DISCRIMINATES: {'YES' if discriminates else 'NO'}")

    # ---- robustness: K=1 vs K=4 at n_layers=30 --------------------------- #
    nat30, tgt30, rows30 = run_for_n_layers(30, [1, 4], ctrl)
    print_table(30, nat30, tgt30, rows30, [1, 4])
    l1_30 = rows30[1]["final_loss"]
    l4_30 = rows30[4]["final_loss"]
    print(f"\n[n_layers=30] K=4 vs K=1 final_loss {l4_30:.5f} vs {l1_30:.5f} "
          f"(ratio={l4_30/l1_30:.4f})")
    robust = (l4_30 < 0.5 * l1_30)
    print(f"[n_layers=30] DISCRIMINATES: {'YES' if robust else 'NO'}")

    # ---- persist for later plotting -------------------------------------- #
    out = {
        "config": {
            "energy": ENERGY, "particle": PARTICLE,
            "abs0": ABS0, "gap0": GAP0, "constraints": CONSTRAINTS,
            "n_events": N_EVENTS, "seeds": list(SEEDS),
            "max_iters": MAX_ITERS, "trust_region": TRUST_REGION,
            "k_list": K_LIST,
        },
        "n_layers_20": {
            "natural": nat20.tolist(),
            "target": tgt20.tolist(),
            "rows": rows20,
            "discriminates": discriminates,
            "loss_k1": loss_k1, "loss_k4": loss_k4,
            "ratio_k4_k1": loss_k4 / loss_k1 if loss_k1 > 0 else None,
            "mostly_monotone": mostly_monotone,
        },
        "n_layers_30": {
            "natural": nat30.tolist(),
            "target": tgt30.tolist(),
            "rows": rows30,
            "loss_k1": l1_30, "loss_k4": l4_30,
            "ratio_k4_k1": l4_30 / l1_30 if l1_30 > 0 else None,
            "discriminates": robust,
        },
    }
    with open(_RESULT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {_RESULT_JSON}")

    return discriminates


if __name__ == "__main__":
    main()

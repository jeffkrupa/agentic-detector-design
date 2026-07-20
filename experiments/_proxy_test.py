"""Differentiable mean-profile PROXY vs real e-/gamma separation ordering.

Uses ONLY mean per-layer energy profiles (and per-layer variance) of the two
classes -- quantities the reverse-AD can exactly differentiate (linear in Ebar_l)
-- and asks whether the proxies rank fine>uniform>coarse like the real CV-AUC.

Proxies:
  (A) D = sum_l (Ebar_l^e - Ebar_l^g)^2
  (B) F = sum_l (Ebar_l^e - Ebar_l^g)^2 / (var_l^e + var_l^g)   [diagonal Fisher]

Per-layer mean/var come from the STANDARD forward output edeps_<seed> cols
[mean_E, var_E, mean_dE, var_dE] (CLAUDE.md sec.2). No boundary_stats needed.
"""
from __future__ import annotations
import json
import sys
import numpy as np

from tools.sim import (load_config, ctrl_flags_from_config, run_forward,
                       run_reverse, run_reverse_per_layer)
from experiments.depth_resolution import (
    ABSORBER_MM, GAP_PER_LAYER_MM, build_scale_profile, profiles_from_scale,
    _design_point,
)

DESIGNS = ("uniform", "fine_at_max", "coarse_at_max")
N_LAYERS = 40
ENERGY = 10000.0
REAL_AUC = {"uniform": 0.8360, "fine_at_max": 0.8825, "coarse_at_max": 0.7896}


def make_dp(cfg, particle, abs_p, gap_p):
    dp = _design_point(cfg, N_LAYERS, abs_p, gap_p)
    return dp.__class__(a=ABSORBER_MM, g=GAP_PER_LAYER_MM, energy=ENERGY,
                        n_layers=N_LAYERS, transverse=400.0, particle=particle,
                        abs_profile=abs_p, gap_profile=gap_p)


def forward_mean_var(cfg, ctrl, particle, abs_p, gap_p, n_events, seed):
    """Return (mean_E[N], var_E[N]) from standard forward edeps_<seed>.

    seeded_param='energy' is required by run_forward but irrelevant for the
    mean_E/var_E columns (0,1); only the derivative cols (2,3) depend on it.
    """
    dp = make_dp(cfg, particle, abs_p, gap_p)
    rr = run_forward(dp, "energy", n_events=n_events, seed=seed, ctrl=ctrl)
    if rr.returncode != 0 or rr.edeps is None:
        raise RuntimeError(f"forward failed rc={rr.returncode} particle={particle}")
    ed = np.asarray(rr.edeps, dtype=float)
    return ed[:, 0], ed[:, 1]


def proxies(mean_e, var_e, mean_g, var_g):
    diff = mean_e - mean_g
    D = float(np.sum(diff ** 2))
    denom = var_e + var_g
    safe = denom > 0
    F = float(np.sum((diff[safe] ** 2) / denom[safe]))
    return D, F


def main():
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    print(f"[cfg] {cfg['_source']}  n_events={n_events} seeds={seeds} "
          f"N={N_LAYERS} E={ENERGY} abs={ABSORBER_MM} gap={GAP_PER_LAYER_MM}")

    results = {}
    profiles = {}  # design -> (mean_e, var_e, mean_g, var_g) averaged over seeds
    for design in DESIGNS:
        w = build_scale_profile(design, N_LAYERS, shower_max_layer=0)
        abs_p, gap_p = profiles_from_scale(w, ABSORBER_MM, GAP_PER_LAYER_MM)
        me_l, ve_l, mg_l, vg_l = [], [], [], []
        for s in seeds:
            me, ve = forward_mean_var(cfg, ctrl, "e-", abs_p, gap_p, n_events, s)
            mg, vg = forward_mean_var(cfg, ctrl, "gamma", abs_p, gap_p, n_events, s)
            me_l.append(me); ve_l.append(ve); mg_l.append(mg); vg_l.append(vg)
        mean_e = np.mean(me_l, axis=0); var_e = np.mean(ve_l, axis=0)
        mean_g = np.mean(mg_l, axis=0); var_g = np.mean(vg_l, axis=0)
        profiles[design] = (mean_e, var_e, mean_g, var_g, abs_p, gap_p)
        D, F = proxies(mean_e, var_e, mean_g, var_g)
        results[design] = {"D": D, "F": F, "real_auc": REAL_AUC[design]}
        print(f"[{design:14s}] D(mean-dist)={D:12.4g}  F(inv-var)={F:10.4f}  "
              f"realAUC={REAL_AUC[design]:.4f}  "
              f"meanTot_e={mean_e.sum():.1f} meanTot_g={mean_g.sum():.1f}")

    def order(key):
        return [d for d, _ in sorted(results.items(), key=lambda kv: -kv[1][key])]
    print("\n=== ORDERING (best->worst) ===")
    print(f"  real AUC : {order('real_auc')}")
    print(f"  proxy A D: {order('D')}")
    print(f"  proxy B F: {order('F')}")
    target = ["fine_at_max", "uniform", "coarse_at_max"]
    print(f"  TARGET   : {target}")
    print(f"  A reproduces target? {order('D') == target}")
    print(f"  B reproduces target? {order('F') == target}")

    print(json.dumps({"results": results,
                      "order_real": order('real_auc'),
                      "order_A": order('D'),
                      "order_B": order('F')}))

    # ---- save uniform profiles for the AD-gradient check ----
    me, ve, mg, vg, abs_p, gap_p = profiles["uniform"]
    np.savez("/sdf/data/atlas/u/jkrupa/agentic/agentic-detector-design/experiments/_proxy_uniform.npz",
             mean_e=me, mean_g=mg, abs_p=np.asarray(abs_p), gap_p=np.asarray(gap_p))
    print("[saved] uniform profiles for gradient check")


if __name__ == "__main__":
    main()

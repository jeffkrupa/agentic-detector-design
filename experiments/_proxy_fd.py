"""Finite-difference cross-check of proxy D gradient (matched seeds).

Compares against the exact-AD gradient computed by _proxy_grad.py:
  AD dD/d[a,g,E] = [-3279.96, 6810.08, 14.5803]  (N_total=6000, seeds 1-3)

D = sum_l (Ebar_l^e - Ebar_l^g)^2 evaluated by re-simulating both classes at the
perturbed scalar geometry; central difference (D(+h)-D(-h))/2h at matched seeds.
The known prior finding: dD/d(gap) AD is ~10x biased vs FD -- we explicitly test.
"""
from __future__ import annotations
import json
import sys
import numpy as np

import tools.sim as _simmod
from tools.sim import load_config, ctrl_flags_from_config, run_forward
from experiments.depth_resolution import ABSORBER_MM, GAP_PER_LAYER_MM

_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_LAYERS, ENERGY = 40, 10000.0
A0, G0 = ABSORBER_MM, GAP_PER_LAYER_MM
AD = {"absorber(a)": -3279.9587621366954, "gap(g)": 6810.076479523355,
      "energy(E)": 14.580255808066603}


def scalar_dp(cfg, particle, a, g, energy):
    from tools.sim import default_design_point
    dp = default_design_point(cfg)
    return dp.__class__(a=float(a), g=float(g), energy=float(energy),
                        n_layers=N_LAYERS, transverse=400.0, particle=particle,
                        abs_profile=None, gap_profile=None)


def mean_profile(cfg, ctrl, particle, a, g, energy, n, seed):
    print(f"    [fwd] {particle} a={a} g={g} E={energy} s={seed}", flush=True)
    dp = scalar_dp(cfg, particle, a, g, energy)
    rr = run_forward(dp, "energy", n_events=n, seed=seed, ctrl=ctrl)
    if rr.returncode != 0 or rr.edeps is None:
        raise RuntimeError("fwd failed")
    return np.asarray(rr.edeps)[:, 0]


def D_of(cfg, ctrl, a, g, e, n, seeds):
    me = np.mean([mean_profile(cfg, ctrl, "e-", a, g, e, n, s) for s in seeds], axis=0)
    mg = np.mean([mean_profile(cfg, ctrl, "gamma", a, g, e, n, s) for s in seeds], axis=0)
    return float(np.sum((me - mg) ** 2))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
    cfg = load_config(); ctrl = ctrl_flags_from_config(cfg)
    rel = cfg["reliability"]
    print(f"[cfg] n={n} seeds={seeds} ctrl={ctrl.to_cli_args()}")
    steps = {"absorber(a)": 0.05, "gap(g)": 0.05, "energy(E)": 50.0}
    out = {}
    for k, h in steps.items():
        if k.startswith("absorber"):
            Dp = D_of(cfg, ctrl, A0 + h, G0, ENERGY, n, seeds)
            Dm = D_of(cfg, ctrl, A0 - h, G0, ENERGY, n, seeds)
        elif k.startswith("gap"):
            Dp = D_of(cfg, ctrl, A0, G0 + h, ENERGY, n, seeds)
            Dm = D_of(cfg, ctrl, A0, G0 - h, ENERGY, n, seeds)
        else:
            Dp = D_of(cfg, ctrl, A0, G0, ENERGY + h, n, seeds)
            Dm = D_of(cfg, ctrl, A0, G0, ENERGY - h, n, seeds)
        fd = (Dp - Dm) / (2 * h)
        ad = AD[k]
        ratio = ad / fd if fd != 0 else float("inf")
        denom = max(abs(ad), abs(fd), 1e-12)
        rel_dis = abs(ad - fd) / denom
        gate = ("ok" if rel_dis <= 0.5 * rel["fd_rel_tol"] else
                "marginal" if rel_dis <= rel["fd_rel_tol"] else "untrusted")
        out[k] = {"AD": ad, "FD": fd, "AD_over_FD": ratio, "rel_dis": rel_dis,
                  "gate": gate, "Dp": Dp, "Dm": Dm}
        print(f"  d/d{k:12s}: AD={ad:13.6g} FD={fd:13.6g} AD/FD={ratio:8.3f} "
              f"rel.dis={rel_dis:.3f} [{gate}]", flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

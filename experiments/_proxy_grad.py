"""Exact-AD gradient of proxy D = sum_l (Ebar_l^e - Ebar_l^g)^2 w.r.t. design.

dD/dtheta = sum_l 2 (Ebar_l^e - Ebar_l^g) (dEbar_l^e/dtheta - dEbar_l^g/dtheta)
         = [reverse pass, e-, adjoints  2*diff_l]
         + [reverse pass, gamma, adjoints -2*diff_l]   (summed component-wise)

The per-layer adjoints diff_l = Ebar_l^e - Ebar_l^g are DETACHED constants
(estimated from a high-stat baseline). Proxy D has no variance denominator;
proxy B's (var^e+var^g) would likewise be detached fixed weights -- we note that
the forward var_dE column exists so differentiating var IS possible in principle,
but we do not (differentiating a noisy 2nd moment is ill-advised).

UNIFORM design == scalar a=3.5,g=5.7 (w_l=1), so we use SCALAR params (no
profile). That lets the 3-row barInputs (d/d scalar a,g,E) be used and a clean
scalar finite-difference cross-check be done.

Reliability per component: SNR=|grad|/SE, SE=sqrt(var/N_total). AD-vs-FD ratio
gates trust. Reports ok/marginal/untrusted per config thresholds.
"""
from __future__ import annotations
import json
import sys
import numpy as np

import tools.sim as _simmod
from tools.sim import (load_config, ctrl_flags_from_config, run_forward,
                       run_reverse)

# The roma node is heavily loaded (~0.2-0.25 s/event for the reverse build), so a
# 2000-event reverse run exceeds the default 300s subprocess cap. Raise the cap
# in-process ONLY (no edit to the shared config.yaml) so the AD passes complete.
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0
from experiments.depth_resolution import ABSORBER_MM, GAP_PER_LAYER_MM

N_LAYERS = 40
ENERGY = 10000.0
A0 = ABSORBER_MM   # 3.5
G0 = GAP_PER_LAYER_MM  # 5.7


def scalar_dp(cfg, particle, a, g, energy):
    from tools.sim import default_design_point
    dp = default_design_point(cfg)
    return dp.__class__(a=float(a), g=float(g), energy=float(energy),
                        n_layers=N_LAYERS, transverse=400.0, particle=particle,
                        abs_profile=None, gap_profile=None)


def mean_profile(cfg, ctrl, particle, a, g, energy, n_events, seed):
    print(f"    [fwd] {particle} a={a} g={g} E={energy} seed={seed}", flush=True)
    dp = scalar_dp(cfg, particle, a, g, energy)
    rr = run_forward(dp, "energy", n_events=n_events, seed=seed, ctrl=ctrl)
    if rr.returncode != 0 or rr.edeps is None:
        raise RuntimeError(f"forward failed {particle}")
    return np.asarray(rr.edeps, dtype=float)[:, 0]


def compute_D(cfg, ctrl, a, g, energy, n_events, seeds):
    """Detached diff and D, averaged over seeds (per-class mean profiles)."""
    me = np.mean([mean_profile(cfg, ctrl, "e-", a, g, energy, n_events, s) for s in seeds], axis=0)
    mg = np.mean([mean_profile(cfg, ctrl, "gamma", a, g, energy, n_events, s) for s in seeds], axis=0)
    diff = me - mg
    return float(np.sum(diff ** 2)), diff, me, mg


def reverse_grad(cfg, ctrl, particle, a, g, energy, adjoints, n_events, seeds):
    """Average 3-row barInputs over seeds. Returns (grad[3], var[3], n_total)."""
    means, vars_ = [], []
    for s in seeds:
        print(f"    [rev] {particle} seed={s}", flush=True)
        dp = scalar_dp(cfg, particle, a, g, energy)
        rr = run_reverse(dp, adjoints, n_events=n_events, seed=s, ctrl=ctrl)
        if rr.returncode != 0 or rr.bar_inputs is None or rr.nan:
            raise RuntimeError(f"reverse failed {particle} seed {s}")
        means.append(rr.bar_inputs[:, 0]); vars_.append(rr.bar_inputs[:, 1])
    n_total = n_events * len(seeds)
    return np.mean(means, axis=0), np.mean(vars_, axis=0), n_total


def main():
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3, 4]
    fd_seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] != "none" else seeds
    do_fd = not (len(sys.argv) > 3 and sys.argv[3] == "none")
    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)
    rel = cfg["reliability"]
    print(f"[cfg] {cfg['_source']} n_events={n_events} seeds={seeds} "
          f"N_total={n_events*len(seeds)}")
    print(f"[ctrl] {ctrl.to_cli_args()}")
    print(f"[design] UNIFORM scalar a={A0} g={G0} E={ENERGY}")

    # --- baseline diff (detached adjoints) ---
    D0, diff, me, mg = compute_D(cfg, ctrl, A0, G0, ENERGY, n_events, seeds)
    print(f"[baseline] D={D0:.6g}  meanTot_e={me.sum():.1f} meanTot_g={mg.sum():.1f}")

    adj_e = 2.0 * diff
    adj_g = -2.0 * diff

    # --- exact AD gradient: two reverse passes summed ---
    ge, ve, nt = reverse_grad(cfg, ctrl, "e-", A0, G0, ENERGY, adj_e, n_events, seeds)
    gg, vg, _ = reverse_grad(cfg, ctrl, "gamma", A0, G0, ENERGY, adj_g, n_events, seeds)
    grad = ge + gg                  # d D / d[a, g, E]
    # SE of summed independent passes: var adds; SE = sqrt((ve+vg)/N)
    se = np.sqrt((ve + vg) / nt)
    keys = ["absorber(a)", "gap(g)", "energy(E)"]
    print("\n=== EXACT-AD GRADIENT of D ===")
    for i, k in enumerate(keys):
        snr = abs(grad[i]) / se[i] if se[i] > 0 else np.inf
        flag = ("ok" if snr >= rel["snr_ok"] else
                "marginal" if snr >= rel["snr_marginal"] else "untrusted")
        print(f"  dD/d{k:12s} = {grad[i]:14.6g}  SE={se[i]:.4g}  SNR={snr:8.2f}  [{flag}]  "
              f"finite={np.isfinite(grad[i])}")

    if not do_fd:
        print(json.dumps({"D0": D0, "grad_ad": grad.tolist(), "se": se.tolist(),
                          "n_total": nt, "keys": keys}))
        return
    # --- finite-difference cross-check (central), matched seeds ---
    print("\n=== AD-vs-FD CROSS-CHECK (central diff, matched seeds) ===")
    steps = {"absorber(a)": 0.05, "gap(g)": 0.05, "energy(E)": 50.0}
    print(f"  (FD seeds={fd_seeds})")
    fd = {}
    for i, k in enumerate(keys):
        h = steps[k]
        if k.startswith("absorber"):
            Dp, *_ = compute_D(cfg, ctrl, A0 + h, G0, ENERGY, n_events, fd_seeds)
            Dm, *_ = compute_D(cfg, ctrl, A0 - h, G0, ENERGY, n_events, fd_seeds)
        elif k.startswith("gap"):
            Dp, *_ = compute_D(cfg, ctrl, A0, G0 + h, ENERGY, n_events, fd_seeds)
            Dm, *_ = compute_D(cfg, ctrl, A0, G0 - h, ENERGY, n_events, fd_seeds)
        else:
            Dp, *_ = compute_D(cfg, ctrl, A0, G0, ENERGY + h, n_events, fd_seeds)
            Dm, *_ = compute_D(cfg, ctrl, A0, G0, ENERGY - h, n_events, fd_seeds)
        fd_val = (Dp - Dm) / (2 * h)
        fd[k] = fd_val
        ratio = grad[i] / fd_val if fd_val != 0 else np.inf
        denom = max(abs(grad[i]), abs(fd_val), 1e-12)
        rel_dis = abs(grad[i] - fd_val) / denom
        gate = ("ok" if rel_dis <= 0.5 * rel["fd_rel_tol"] else
                "marginal" if rel_dis <= rel["fd_rel_tol"] else "untrusted")
        print(f"  d/d{k:12s}: AD={grad[i]:13.6g}  FD={fd_val:13.6g}  "
              f"AD/FD={ratio:8.3f}  rel.dis={rel_dis:.3f}  [{gate}]")

    print(json.dumps({"D0": D0, "grad_ad": grad.tolist(), "se": se.tolist(),
                      "fd": fd, "n_total": nt, "keys": keys}))


if __name__ == "__main__":
    main()

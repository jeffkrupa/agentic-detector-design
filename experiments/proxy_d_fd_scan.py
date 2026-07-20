"""Re-diagnose the proxy-D AD-vs-FD discrepancy with a resolved epsilon scan + CRN.

Observable: D = sum_l (Ebar_l^e - Ebar_l^g)^2  (squared per-layer profile distance,
a 2nd-order quantity -> FD is noise-prone). Old finding: dD/d(gap) AD/FD ~10x at a
single fixed step h=0.05mm. E0 showed small-fixed-eps central differences on this
MC sim are noise-dominated (FD_SE ~ 1/eps). Here we scan eps and use common random
numbers (per-seed matched +/-eps, then average across seeds) to see whether
proxy-D AD/FD plateaus near 1 in a resolved window (=> old discrepancy was FD noise).

Small by design: forward-mode, N<=1000, dev seeds, thread-pooled subprocesses.
Reuses the AD path from _proxy_grad.py (two reverse passes with detached adjoints).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tools.sim as _simmod
from tools.sim import (load_config, ctrl_flags_from_config, run_forward,
                       run_reverse, default_design_point)

# Forward N=1000 ~185s; keep margin above the shared 300s cap (in-process only).
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_LAYERS, ENERGY = 40, 10000.0
A0, G0 = 3.5, 5.7          # baseline scalar geometry (== _proxy_grad UNIFORM)
EPS_GAP = [0.05, 0.1, 0.25, 0.5, 1.0]


def _dp(cfg, particle, a, g, energy):
    dp = default_design_point(cfg)
    return dp.__class__(a=float(a), g=float(g), energy=float(energy),
                        n_layers=N_LAYERS, transverse=400.0, particle=particle,
                        abs_profile=None, gap_profile=None)


def _mean_profile(cfg, ctrl, particle, a, g, energy, n, seed):
    rr = run_forward(_dp(cfg, particle, a, g, energy), "energy",
                     n_events=n, seed=seed, ctrl=ctrl)
    if rr.returncode != 0 or rr.edeps is None:
        raise RuntimeError(f"fwd failed {particle} a={a} g={g} s={seed}")
    return np.asarray(rr.edeps, dtype=float)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-events", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    n, seeds, workers = args.n_events, args.seeds, args.workers

    cfg = load_config()
    ctrl = ctrl_flags_from_config(cfg)

    # ---- all forward jobs needed: baseline + gap +/-eps, both classes, all seeds.
    # tag -> (particle, a, g, energy, seed); dispatch concurrently, cache by tag.
    fwd_jobs = {}
    def add_pt(a, g, key):
        for p in ("e-", "gamma"):
            for s in seeds:
                fwd_jobs[(key, p, s)] = (p, a, g, ENERGY, s)
    add_pt(A0, G0, "base")
    for e in EPS_GAP:
        add_pt(A0, G0 + e, f"g+{e}")
        add_pt(A0, G0 - e, f"g-{e}")

    prof = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def _run(item):
            tag, (p, a, g, energy, s) = item
            return tag, _mean_profile(cfg, ctrl, p, a, g, energy, n, s)
        for tag, arr in ex.map(_run, list(fwd_jobs.items())):
            prof[tag] = arr

    def D_per_seed(key, s):
        return float(np.sum((prof[(key, "e-", s)] - prof[(key, "gamma", s)]) ** 2))

    # ---- AD gradient (reverse) at the SAME seeds, detached adjoints = 2*diff.
    me = np.mean([prof[("base", "e-", s)] for s in seeds], axis=0)
    mg = np.mean([prof[("base", "gamma", s)] for s in seeds], axis=0)
    diff = me - mg
    adj_e, adj_g = 2.0 * diff, -2.0 * diff

    def _rev(particle, adj):
        outs = []
        for s in seeds:
            rr = run_reverse(_dp(cfg, particle, A0, G0, ENERGY), adj,
                             n_events=n, seed=s, ctrl=ctrl)
            if rr.returncode != 0 or rr.bar_inputs is None or rr.nan:
                raise RuntimeError(f"reverse failed {particle} s={s}")
            outs.append(rr.bar_inputs[:, 0])
        return np.mean(outs, axis=0)          # [d/da, d/dg, d/dE]
    ge = _rev("e-", adj_e)
    gg = _rev("gamma", adj_g)
    grad = ge + gg
    ad_gap = float(grad[1])                   # dD/d(gap)

    # ---- FD epsilon scan for gap: per-seed matched CRN, average across seeds.
    print(f"\n# proxy D = sum_l (Ebar_e - Ebar_g)^2   N={n} seeds={seeds}")
    print(f"# AD dD/d[a,g,E] = [{grad[0]:.4g}, {grad[1]:.4g}, {grad[2]:.4g}]  (matched seeds)")
    print("observable | parameter | epsilon | AD | FD | AD/FD | FD_noise | notes")
    for e in EPS_GAP:
        fd_s = [(D_per_seed(f"g+{e}", s) - D_per_seed(f"g-{e}", s)) / (2 * e)
                for s in seeds]
        fd = float(np.mean(fd_s))
        noise = float(np.std(fd_s) / np.sqrt(len(fd_s))) if len(fd_s) > 1 else float("nan")
        ratio = ad_gap / fd if fd != 0 else float("nan")
        note = "noise-dominated" if (len(fd_s) > 1 and abs(fd) < 3 * noise) else ""
        print(f"proxy_D | gap(g) | {e:>5g} | {ad_gap:+.5g} | {fd:+.5g} | "
              f"{ratio:+.3f} | {noise:.3g} | {note}")


if __name__ == "__main__":
    main()

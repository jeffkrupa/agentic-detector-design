"""FD audit driver: validate FD machinery on linear control + diagnose proxy D.

Usage:
  python -m experiments._fd_audit control   N seedlist   # Task B (total_edep)
  python -m experiments._fd_audit proxy     N seedlist   # Task C (proxy D)

All runs use the canonical ctrl flags and the UNIFORM N=40, 10 GeV design.
Common random numbers: the SAME seed list is used on the + and - sides for FD,
and (separately) per-particle seeds are reused on both sides. total_edep control
uses a single particle (e-) so CRN is unambiguous.
"""
from __future__ import annotations
import json
import sys
import numpy as np

import tools.sim as _simmod
from tools.sim import load_config, ctrl_flags_from_config, run_forward, run_reverse

_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0

N_LAYERS, ENERGY = 40, 10000.0
A0, G0 = 3.5, 5.7


def scalar_dp(cfg, particle, a, g, energy):
    from tools.sim import default_design_point
    dp = default_design_point(cfg)
    return dp.__class__(a=float(a), g=float(g), energy=float(energy),
                        n_layers=N_LAYERS, transverse=400.0, particle=particle,
                        abs_profile=None, gap_profile=None)


def mean_arr(cfg, ctrl, particle, a, g, energy, n, seed, seeded="a"):
    """Return full (nlayers,4) edeps mean array averaged trivially for one seed."""
    dp = scalar_dp(cfg, particle, a, g, energy)
    rr = run_forward(dp, seeded, n_events=n, seed=seed, ctrl=ctrl)
    if rr.returncode != 0 or rr.edeps is None:
        raise RuntimeError(f"fwd failed {particle} a={a} g={g} E={energy} s={seed}")
    return np.asarray(rr.edeps, dtype=float)


def edep_profile_multiseed(cfg, ctrl, particle, a, g, energy, n, seeds, seeded="a"):
    """Per-layer mean energy (col0) averaged over seeds, plus per-layer var (col1)."""
    arrs = [mean_arr(cfg, ctrl, particle, a, g, energy, n, s, seeded) for s in seeds]
    stack = np.stack(arrs, axis=0)          # (nseed, nlayers, 4)
    m = stack[:, :, 0].mean(axis=0)         # mean energy per layer
    v = stack[:, :, 1].mean(axis=0)         # per-event variance per layer
    return m, v


# --------------------------------------------------------------------------- #
# Task B: linear control total_edep, single particle e-
# --------------------------------------------------------------------------- #
def task_control(n, seeds):
    cfg = load_config(); ctrl = ctrl_flags_from_config(cfg)
    print(f"[control] total_edep e- uniform a={A0} g={G0} E={ENERGY} N={n} seeds={seeds}")

    # --- AD reverse gradient of total_edep (adjoints=ones) ---
    grads = {"a": [], "g": [], "energy": []}
    vars_ = {"a": [], "g": [], "energy": []}
    for s in seeds:
        dp = scalar_dp(cfg, "e-", A0, G0, ENERGY)
        rr = run_reverse(dp, np.ones(N_LAYERS), n_events=n, seed=s, ctrl=ctrl)
        if rr.returncode != 0 or rr.bar_inputs is None or rr.nan:
            raise RuntimeError(f"reverse failed s={s}")
        for i, k in enumerate(("a", "g", "energy")):
            grads[k].append(rr.bar_inputs[i, 0]); vars_[k].append(rr.bar_inputs[i, 1])
    nt = n * len(seeds)
    ad = {k: float(np.mean(grads[k])) for k in grads}
    ad_se = {k: float(np.sqrt(np.mean(vars_[k]) / nt)) for k in grads}
    print(f"[AD] d(total_edep)/da     = {ad['a']:.6g}  SE={ad_se['a']:.4g}")
    print(f"[AD] d(total_edep)/dE     = {ad['energy']:.6g}  SE={ad_se['energy']:.4g}")

    # --- total_edep value at a point (sum over layers of col0) ---
    def total_edep(a, g, energy):
        m, _ = edep_profile_multiseed(cfg, ctrl, "e-", a, g, energy, n, seeds)
        return float(m.sum())

    # per-event variance of TOTAL edep for noise floor: need var of sum per event.
    # Approx: SE(total) ~ sqrt(sum_l var_l / nt) (ignores inter-layer cov -> lower bound)
    def total_edep_with_se(a, g, energy):
        m, v = edep_profile_multiseed(cfg, ctrl, "e-", a, g, energy, n, seeds)
        return float(m.sum()), float(np.sqrt(v.sum() / nt))

    out = {"AD": ad, "AD_SE": ad_se, "N": n, "seeds": seeds, "scans": {}}

    eps_a = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    print("\n  --- eps scan: d(total_edep)/da (CRN) ---")
    print(f"  {'eps':>7} {'FD':>12} {'dD_se':>10} {'AD/FD':>9} {'SNR_dD':>8}")
    rows_a = []
    for h in eps_a:
        Tp, sp = total_edep_with_se(A0 + h, G0, ENERGY)
        Tm, sm = total_edep_with_se(A0 - h, G0, ENERGY)
        fd = (Tp - Tm) / (2 * h)
        # SE of the difference. With CRN the diff is correlated; this is an upper
        # bound treating sides as independent.
        d_se = np.sqrt(sp ** 2 + sm ** 2)
        snr = abs(Tp - Tm) / d_se if d_se > 0 else np.inf
        ratio = ad["a"] / fd if fd != 0 else np.inf
        rows_a.append({"eps": h, "FD": fd, "AD_over_FD": ratio,
                       "dDelta": Tp - Tm, "delta_se": d_se, "snr": snr})
        print(f"  {h:7.3f} {fd:12.6g} {d_se:10.4g} {ratio:9.3f} {snr:8.1f}")
    out["scans"]["a"] = rows_a

    eps_E = [5.0, 10.0, 20.0, 50.0, 100.0]
    print("\n  --- eps scan: d(total_edep)/dE (CRN) ---")
    print(f"  {'eps':>7} {'FD':>12} {'dD_se':>10} {'AD/FD':>9} {'SNR_dD':>8}")
    rows_E = []
    for h in eps_E:
        Tp, sp = total_edep_with_se(A0, G0, ENERGY + h)
        Tm, sm = total_edep_with_se(A0, G0, ENERGY - h)
        fd = (Tp - Tm) / (2 * h)
        d_se = np.sqrt(sp ** 2 + sm ** 2)
        snr = abs(Tp - Tm) / d_se if d_se > 0 else np.inf
        ratio = ad["energy"] / fd if fd != 0 else np.inf
        rows_E.append({"eps": h, "FD": fd, "AD_over_FD": ratio,
                       "dDelta": Tp - Tm, "delta_se": d_se, "snr": snr})
        print(f"  {h:7.1f} {fd:12.6g} {d_se:10.4g} {ratio:9.3f} {snr:8.1f}")
    out["scans"]["energy"] = rows_E

    print("RESULT_JSON " + json.dumps(out))


# --------------------------------------------------------------------------- #
# Task C: proxy D = sum_l (Ebar_l^e - Ebar_l^g)^2
# --------------------------------------------------------------------------- #
def D_of(cfg, ctrl, a, g, e, n, seeds):
    """D and the per-layer diff, plus a bootstrap noise estimate on D.

    CRN: e- uses `seeds`, gamma uses `seeds` (same list reused on +/- sides).
    Noise floor: estimate Var(D) by treating per-layer means as Gaussian with
    SE^2 = var_l/nt, propagated through D = sum (me-mg)^2.
    """
    me, ve = edep_profile_multiseed(cfg, ctrl, "e-", a, g, e, n, seeds)
    mg, vg = edep_profile_multiseed(cfg, ctrl, "gamma", a, g, e, n, seeds)
    diff = me - mg
    D = float(np.sum(diff ** 2))
    nt = n * len(seeds)
    # SE of each per-layer diff (e- and gamma independent):
    se_diff = np.sqrt(ve / nt + vg / nt)
    # Var(D) via delta method: dD/d(diff_l) = 2 diff_l ; plus the +2*sum(se^2)
    # bias of E[sum diff^2] = sum(true_diff^2)+sum(se^2). Report both pieces.
    varD_delta = float(np.sum((2 * diff) ** 2 * se_diff ** 2))
    bias_D = float(np.sum(se_diff ** 2))   # positive bias in E[D] from noise
    return D, diff, np.sqrt(varD_delta), bias_D


def task_proxy(n, seeds):
    cfg = load_config(); ctrl = ctrl_flags_from_config(cfg)
    print(f"[proxy] D=sum(Ebar_e-Ebar_g)^2 uniform a={A0} g={G0} E={ENERGY} N={n} seeds={seeds}")
    nt = n * len(seeds)

    D0, diff0, seD0, biasD0 = D_of(cfg, ctrl, A0, G0, ENERGY, n, seeds)
    print(f"[baseline] D0={D0:.6g}  SE(D0)~{seD0:.4g}  noise-bias(D)~{biasD0:.4g}  nt={nt}")

    # --- AD reverse gradient of D (two reverse passes, detached adjoints) ---
    adj_e = 2.0 * diff0
    adj_g = -2.0 * diff0
    ge = np.zeros(3); gg = np.zeros(3); ve = np.zeros(3); vg = np.zeros(3)
    for s in seeds:
        dpe = scalar_dp(cfg, "e-", A0, G0, ENERGY)
        rre = run_reverse(dpe, adj_e, n_events=n, seed=s, ctrl=ctrl)
        dpg = scalar_dp(cfg, "gamma", A0, G0, ENERGY)
        rrg = run_reverse(dpg, adj_g, n_events=n, seed=s, ctrl=ctrl)
        if rre.returncode or rrg.returncode or rre.bar_inputs is None or rrg.bar_inputs is None:
            raise RuntimeError("reverse failed")
        ge += rre.bar_inputs[:, 0]; ve += rre.bar_inputs[:, 1]
        gg += rrg.bar_inputs[:, 0]; vg += rrg.bar_inputs[:, 1]
    ge /= len(seeds); gg /= len(seeds); ve /= len(seeds); vg /= len(seeds)
    grad = ge + gg
    se = np.sqrt((ve + vg) / nt)
    keys = ["a", "g", "energy"]
    ad = {k: float(grad[i]) for i, k in enumerate(keys)}
    ad_se = {k: float(se[i]) for i, k in enumerate(keys)}
    print("[AD] dD/d:", {k: f"{ad[k]:.6g}+-{ad_se[k]:.3g}" for k in keys})

    out = {"D0": D0, "seD0": seD0, "biasD0": biasD0, "AD": ad, "AD_SE": ad_se,
           "N": n, "seeds": seeds, "scans": {}}

    eps_map = {"a": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
               "g": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
               "energy": [5.0, 10.0, 20.0, 50.0, 100.0]}
    for param, eps_list in eps_map.items():
        print(f"\n  --- eps scan: dD/d{param} (CRN, same seeds both sides) ---")
        print(f"  {'eps':>7} {'FD':>13} {'dDelta':>12} {'sig_dD':>11} {'SNR':>7} {'AD/FD':>9}")
        rows = []
        for h in eps_list:
            if param == "a":
                Dp, _, sp, _ = D_of(cfg, ctrl, A0 + h, G0, ENERGY, n, seeds)
                Dm, _, sm, _ = D_of(cfg, ctrl, A0 - h, G0, ENERGY, n, seeds)
            elif param == "g":
                Dp, _, sp, _ = D_of(cfg, ctrl, A0, G0 + h, ENERGY, n, seeds)
                Dm, _, sm, _ = D_of(cfg, ctrl, A0, G0 - h, ENERGY, n, seeds)
            else:
                Dp, _, sp, _ = D_of(cfg, ctrl, A0, G0, ENERGY + h, n, seeds)
                Dm, _, sm, _ = D_of(cfg, ctrl, A0, G0, ENERGY - h, n, seeds)
            fd = (Dp - Dm) / (2 * h)
            dDelta = Dp - Dm
            sig_dD = np.sqrt(sp ** 2 + sm ** 2)   # noise on Delta D (independent-side bound)
            snr = abs(dDelta) / sig_dD if sig_dD > 0 else np.inf
            ratio = ad[param] / fd if fd != 0 else np.inf
            rows.append({"eps": h, "FD": fd, "dDelta": dDelta, "sig_dD": sig_dD,
                         "snr": snr, "AD_over_FD": ratio})
            print(f"  {h:7.3f} {fd:13.6g} {dDelta:12.6g} {sig_dD:11.5g} {snr:7.2f} {ratio:9.3f}")
        out["scans"][param] = rows

    print("RESULT_JSON " + json.dumps(out))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "control"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
    if mode == "control":
        task_control(n, seeds)
    elif mode == "proxy":
        task_proxy(n, seeds)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

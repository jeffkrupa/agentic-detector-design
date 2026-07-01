"""Pure-numpy e-/gamma separation metrics for the EM-calorimeter testbed.

This module is the metric core of the particle-ID experiment (e- vs gamma
separation as a function of FRONT longitudinal granularity). It has NO scikit-
learn dependency (the LCG-view python on the workers ships numpy but not sklearn)
and NO simulation dependency -- it operates on already-extracted per-event
per-layer profile matrices, so it is fully unit-testable via ``--self-test``.

What is provided
----------------
* ``fisher_separation(x_e, x_g)`` -- the 1-D Fisher discriminant ratio
  ``S = (mu_e - mu_g)^2 / (var_e + var_g)`` for a single scalar feature per class.
  Class-separation power of one feature; sign-agnostic, >= 0, scale-free.
* ``roc_auc(scores_e, scores_g)`` -- ROC AUC computed from the Mann-Whitney U
  rank statistic (handles ties via average ranks). No sorting-by-threshold sweep
  needed; ``AUC = U / (n_e * n_g)`` where U counts (e>g) pairs.
* ``ridge_cv_auc(Xe, Xg, ...)`` -- a ridge-regularized LINEAR classifier with
  K-FOLD cross-validation on two ``(n_events, n_features)`` profile matrices,
  returning the CV-mean held-out AUC and its std across folds. The classifier is
  ridge-regression-to-labels (closed form, numpy-only); features are standardized
  on the TRAIN fold only. The regularization + held-out evaluation are essential:
  without them a finer (higher-dimensional) feature space can win purely by
  overfitting the training set, which would be a dimensionality artifact rather
  than a real physics effect.
* ``gridfair_auc`` / ``gridfair_fisher`` (+ ``rebin_to_common_axis``) -- the
  GRID-FAIR control. Every design has a different number of layers AT DIFFERENT
  physical depths, so a native-grid comparison conflates "more/finer features"
  with "the physics of where energy lands". The control REBINS each event's
  per-layer profile onto ONE common physical-depth axis (fixed 1-mm bins from 0
  to the total length) shared by all designs, so every design is scored at the
  SAME feature dimensionality. A real onset-granularity effect (e- deposits
  earlier than gamma) survives this rebinning; a pure dimensionality artifact
  vanishes. We then report the AUC of a single physical onset feature
  (energy-fraction deposited in the first ``onset_mm`` mm) and/or the Fisher S of
  that feature.

Self-test (``python -m tools.separation``)
------------------------------------------
Fabricates two separable gaussian feature populations (NO sim) and asserts
``roc_auc > 0.9``, ``fisher_separation > 0``, and the ridge CV-AUC recovers the
separation on a multi-feature version. Also checks the rebinning is energy-
conserving and that the grid-fair AUC tracks an injected onset shift.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 1-D scalar-feature metrics
# --------------------------------------------------------------------------- #
def fisher_separation(x_e: np.ndarray, x_g: np.ndarray) -> float:
    """Fisher discriminant ratio S = (mu_e - mu_g)^2 / (var_e + var_g).

    ``x_e``, ``x_g`` are 1-D arrays of one scalar feature, one entry per event,
    for class e- and class gamma. Larger S = better-separated classes. Returns
    0.0 if the pooled variance is non-positive (degenerate / constant feature).
    """
    x_e = np.asarray(x_e, dtype=float).ravel()
    x_g = np.asarray(x_g, dtype=float).ravel()
    if x_e.size < 1 or x_g.size < 1:
        raise ValueError("need >=1 sample per class for Fisher separation")
    var = float(x_e.var(ddof=1) if x_e.size > 1 else 0.0) + \
        float(x_g.var(ddof=1) if x_g.size > 1 else 0.0)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    return float((x_e.mean() - x_g.mean()) ** 2 / var)


def roc_auc(scores_e: np.ndarray, scores_g: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U rank statistic (numpy, ties-aware).

    Interprets a HIGHER score as more e--like. AUC = P(score_e > score_g) +
    0.5 P(tie), computed as ``U / (n_e n_g)`` with ``U = R_e - n_e(n_e+1)/2``
    where ``R_e`` is the sum of average ranks of the e- scores in the pooled
    sample. Returns 0.5 for empty/degenerate input.
    """
    scores_e = np.asarray(scores_e, dtype=float).ravel()
    scores_g = np.asarray(scores_g, dtype=float).ravel()
    n_e, n_g = scores_e.size, scores_g.size
    if n_e == 0 or n_g == 0:
        return 0.5
    pooled = np.concatenate([scores_e, scores_g])
    # Average ranks (1-based), so tied scores share the mean of their ranks.
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty(pooled.size, dtype=float)
    ranks[order] = np.arange(1, pooled.size + 1, dtype=float)
    # resolve ties to average rank
    sorted_vals = pooled[order]
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    r_e = ranks[:n_e].sum()
    u_e = r_e - n_e * (n_e + 1) / 2.0
    return float(u_e / (n_e * n_g))


# --------------------------------------------------------------------------- #
# Ridge-regularized linear classifier with K-fold cross-validation
# --------------------------------------------------------------------------- #
def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression to labels y; returns weight vector w.

    Solves (X^T X + lam I) w = X^T y. A column of ones is assumed to already be
    appended by the caller for the intercept; that intercept column is left
    UNREGULARIZED by zeroing its diagonal penalty entry.
    """
    n_feat = X.shape[1]
    reg = lam * np.eye(n_feat)
    reg[-1, -1] = 0.0  # do not regularize the intercept (last column = ones)
    A = X.T @ X + reg
    b = X.T @ y
    return np.linalg.solve(A, b)


def ridge_cv_auc(Xe: np.ndarray, Xg: np.ndarray, k: int = 5,
                 lam: float = 1.0, seed: int = 0):
    """K-fold CV held-out AUC of a ridge-regression-to-labels linear classifier.

    ``Xe``, ``Xg`` are ``(n_events, n_features)`` per-event feature matrices for
    class e- (label +1) and gamma (label 0). For each of ``k`` folds the
    classifier is FIT on the train split (features standardized using TRAIN
    statistics only, ridge penalty ``lam``) and scored on the held-out split;
    the fold AUC is computed by :func:`roc_auc`. Returns
    ``(mean_auc, std_auc)`` across folds.

    Why CV + regularization: a finer feature space (more layers) can separate the
    TRAIN set perfectly by overfitting. Held-out evaluation + the ridge penalty
    keep the metric honest, so a finer design only wins if the extra resolution
    carries GENERALIZING separation power, not noise it memorized.
    """
    Xe = np.asarray(Xe, dtype=float)
    Xg = np.asarray(Xg, dtype=float)
    if Xe.ndim == 1:
        Xe = Xe.reshape(-1, 1)
    if Xg.ndim == 1:
        Xg = Xg.reshape(-1, 1)
    if Xe.shape[1] != Xg.shape[1]:
        raise ValueError(f"feature dim mismatch: {Xe.shape[1]} vs {Xg.shape[1]}")
    rng = np.random.default_rng(seed)
    # Per-class fold assignments so every fold sees both classes in train+test.
    folds_e = rng.permutation(Xe.shape[0]) % k
    folds_g = rng.permutation(Xg.shape[0]) % k
    aucs = []
    for f in range(k):
        tr_e, te_e = Xe[folds_e != f], Xe[folds_e == f]
        tr_g, te_g = Xg[folds_g != f], Xg[folds_g == f]
        if te_e.shape[0] == 0 or te_g.shape[0] == 0 or \
           tr_e.shape[0] == 0 or tr_g.shape[0] == 0:
            continue
        # Standardize on the TRAIN fold only.
        train = np.vstack([tr_e, tr_g])
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd[sd <= 0] = 1.0

        def _std(M):
            return (M - mu) / sd

        Xtr = np.hstack([_std(train), np.ones((train.shape[0], 1))])
        ytr = np.concatenate([np.ones(tr_e.shape[0]), np.zeros(tr_g.shape[0])])
        w = _ridge_fit(Xtr, ytr, lam)
        se = np.hstack([_std(te_e), np.ones((te_e.shape[0], 1))]) @ w
        sg = np.hstack([_std(te_g), np.ones((te_g.shape[0], 1))]) @ w
        aucs.append(roc_auc(se, sg))
    if not aucs:
        raise RuntimeError("no usable CV folds (too few events per class?)")
    aucs = np.asarray(aucs, dtype=float)
    return float(aucs.mean()), float(aucs.std(ddof=1) if aucs.size > 1 else 0.0)


# --------------------------------------------------------------------------- #
# Grid-fair control: rebin per-layer profiles onto a common physical-depth axis
# --------------------------------------------------------------------------- #
def rebin_to_common_axis(profiles: np.ndarray, layer_lo: np.ndarray,
                         layer_hi: np.ndarray, total_length: float,
                         bin_mm: float = 1.0) -> np.ndarray:
    """Rebin per-event per-layer edep onto common ``bin_mm`` physical-depth bins.

    ``profiles`` is ``(n_events, n_layers)``; ``layer_lo``/``layer_hi`` are the
    physical depth (mm) edges of each layer (length n_layers). Energy in a layer
    is distributed to the common-axis bins in proportion to the overlap length
    between the layer span and each bin (uniform-deposit-within-layer
    assumption), which is energy-conserving: the rebinned row sums to the layer
    row sum (modulo any energy beyond ``total_length``). All designs share the
    SAME common axis (0..total_length in ``bin_mm`` steps), so the rebinned
    matrix has ONE feature dimensionality regardless of n_layers.
    """
    profiles = np.asarray(profiles, dtype=float)
    layer_lo = np.asarray(layer_lo, dtype=float)
    layer_hi = np.asarray(layer_hi, dtype=float)
    n_bins = int(np.ceil(total_length / bin_mm))
    bin_edges = np.arange(n_bins + 1, dtype=float) * bin_mm
    out = np.zeros((profiles.shape[0], n_bins), dtype=float)
    for l in range(profiles.shape[1]):
        lo, hi = layer_lo[l], layer_hi[l]
        width = hi - lo
        if width <= 0:
            continue
        # First and last bin this layer touches.
        b0 = max(0, int(np.floor(lo / bin_mm)))
        b1 = min(n_bins, int(np.ceil(hi / bin_mm)))
        for b in range(b0, b1):
            overlap = min(hi, bin_edges[b + 1]) - max(lo, bin_edges[b])
            if overlap <= 0:
                continue
            out[:, b] += profiles[:, l] * (overlap / width)
    return out


def _onset_fraction(common: np.ndarray, onset_mm: float, bin_mm: float = 1.0):
    """Fraction of each event's energy deposited in the first ``onset_mm`` mm."""
    n_onset = max(1, int(round(onset_mm / bin_mm)))
    total = common.sum(axis=1)
    total[total <= 0] = np.nan  # drop zero-energy events from the fraction
    return common[:, :n_onset].sum(axis=1) / total


def gridfair_features(profiles_e, profiles_g, layer_lo, layer_hi, total_length,
                      onset_mm: float = 30.0, bin_mm: float = 1.0):
    """Return (frac_e, frac_g): grid-fair onset energy-fraction per event/class.

    Rebins both classes onto the common physical-depth axis and computes, per
    event, the energy fraction in the first ``onset_mm`` mm -- a SINGLE physical
    onset feature shared across all designs. NaN (zero-energy) events dropped.
    """
    ce = rebin_to_common_axis(profiles_e, layer_lo, layer_hi, total_length, bin_mm)
    cg = rebin_to_common_axis(profiles_g, layer_lo, layer_hi, total_length, bin_mm)
    fe = _onset_fraction(ce, onset_mm, bin_mm)
    fg = _onset_fraction(cg, onset_mm, bin_mm)
    return fe[np.isfinite(fe)], fg[np.isfinite(fg)]


def gridfair_auc(profiles_e, profiles_g, layer_lo, layer_hi, total_length,
                 onset_mm: float = 30.0, bin_mm: float = 1.0) -> float:
    """Grid-fair AUC: AUC of the common-axis onset energy-fraction feature.

    The artifact control. e- showers start earlier than gamma (which must first
    convert), so e- has MORE energy in the first ``onset_mm`` mm -> higher
    onset fraction. Because every design is scored on the SAME physical axis at
    the SAME dimensionality, a separation that survives here is a genuine onset-
    granularity / physics effect, not a feature-count artifact.
    """
    fe, fg = gridfair_features(profiles_e, profiles_g, layer_lo, layer_hi,
                               total_length, onset_mm, bin_mm)
    return roc_auc(fe, fg)


def gridfair_fisher(profiles_e, profiles_g, layer_lo, layer_hi, total_length,
                    onset_mm: float = 30.0, bin_mm: float = 1.0) -> float:
    """Grid-fair Fisher S of the common-axis onset energy-fraction feature."""
    fe, fg = gridfair_features(profiles_e, profiles_g, layer_lo, layer_hi,
                               total_length, onset_mm, bin_mm)
    return fisher_separation(fe, fg)


# --------------------------------------------------------------------------- #
# Front-onset single features on the NATIVE grid (used by the driver too)
# --------------------------------------------------------------------------- #
def frac_in_first_layers(profiles: np.ndarray, n_first: int) -> np.ndarray:
    """Per-event fraction of energy in the first ``n_first`` layers (native grid).

    Zero-energy events map to NaN so the caller can drop them.
    """
    profiles = np.asarray(profiles, dtype=float)
    total = profiles.sum(axis=1)
    total = np.where(total > 0, total, np.nan)
    return profiles[:, :n_first].sum(axis=1) / total


# --------------------------------------------------------------------------- #
# Self-test (no sim): fabricate separable populations, validate the metrics.
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    print("[self-test] separation metrics on fabricated data (no sim)")
    rng = np.random.default_rng(0)

    # (1) Two clearly separable gaussian 1-D populations.
    n = 4000
    x_e = rng.normal(1.0, 1.0, n)   # e- shifted up
    x_g = rng.normal(-1.0, 1.0, n)  # gamma shifted down
    auc = roc_auc(x_e, x_g)
    S = fisher_separation(x_e, x_g)
    print(f"[self-test] 1-D gaussians: AUC={auc:.4f} Fisher_S={S:.4f}")
    assert auc > 0.9, f"AUC {auc} not > 0.9 on well-separated data"
    assert S > 0, f"Fisher S {S} not > 0"

    # AUC sanity bounds: identical populations -> ~0.5; flipped sign symmetric.
    same = roc_auc(x_e, x_e.copy())
    assert abs(same - 0.5) < 0.05, f"AUC of identical samples {same} != ~0.5"
    flipped = roc_auc(x_g, x_e)
    assert abs((auc + flipped) - 1.0) < 1e-9, "AUC not antisymmetric under swap"
    # Ties handled: all-equal scores -> 0.5 exactly.
    assert roc_auc(np.zeros(10), np.zeros(7)) == 0.5

    # (2) Multi-feature ridge CV-AUC recovers the separation; CV is finite.
    d = 8
    mean_e = np.full(d, 0.75)
    Xe = rng.normal(0, 1, (n, d)) + mean_e
    Xg = rng.normal(0, 1, (n, d))
    cv_mean, cv_std = ridge_cv_auc(Xe, Xg, k=5, lam=1.0, seed=1)
    print(f"[self-test] ridge CV-AUC (8 informative feats): "
          f"{cv_mean:.4f} +/- {cv_std:.4f}")
    assert cv_mean > 0.9, f"CV-AUC {cv_mean} not > 0.9 on separable multi-feat"
    assert np.isfinite(cv_std)

    # (2b) Pure-noise features must NOT separate on held-out folds (no overfit):
    #      CV-AUC stays near 0.5 even with many features and few events.
    Xe_noise = rng.normal(0, 1, (200, 50))
    Xg_noise = rng.normal(0, 1, (200, 50))
    noise_mean, _ = ridge_cv_auc(Xe_noise, Xg_noise, k=5, lam=1.0, seed=2)
    print(f"[self-test] ridge CV-AUC (50 NOISE feats, 200 ev): {noise_mean:.4f} "
          f"(should be ~0.5 -- CV defeats overfitting)")
    assert abs(noise_mean - 0.5) < 0.12, \
        f"CV-AUC {noise_mean} strayed from 0.5 on pure noise (overfit leak)"

    # (3) Grid-fair rebinning is energy-conserving and tracks an onset shift.
    n_layers = 40
    layer_t = 9.2  # mm per layer (uniform)
    total_length = n_layers * layer_t
    edges = np.arange(n_layers + 1) * layer_t
    layer_lo, layer_hi = edges[:-1], edges[1:]
    centers = 0.5 * (layer_lo + layer_hi)

    # e- deposits earlier (onset shallow), gamma later (onset deep).
    prof_e, prof_g = [], []
    for _ in range(1500):
        xe = 70.0 + rng.normal(0, 15)   # shallow peak
        xg = 110.0 + rng.normal(0, 15)  # deep peak
        prof_e.append(np.exp(-0.5 * ((centers - xe) / 25.0) ** 2))
        prof_g.append(np.exp(-0.5 * ((centers - xg) / 25.0) ** 2))
    prof_e = np.asarray(prof_e)
    prof_g = np.asarray(prof_g)

    reb = rebin_to_common_axis(prof_e[:5], layer_lo, layer_hi, total_length, 1.0)
    # Energy conservation: rebinned row sum == native row sum (uniform layers).
    err = np.max(np.abs(reb.sum(axis=1) - prof_e[:5].sum(axis=1)))
    print(f"[self-test] rebin energy-conservation max abs err={err:.3e}")
    assert err < 1e-6, "rebinning is not energy-conserving"

    gf_auc = gridfair_auc(prof_e, prof_g, layer_lo, layer_hi, total_length,
                          onset_mm=30.0, bin_mm=1.0)
    gf_S = gridfair_fisher(prof_e, prof_g, layer_lo, layer_hi, total_length,
                           onset_mm=30.0, bin_mm=1.0)
    print(f"[self-test] grid-fair onset AUC={gf_auc:.4f} Fisher_S={gf_S:.4f}")
    assert gf_auc > 0.9, f"grid-fair AUC {gf_auc} not > 0.9 on onset-shifted data"
    assert gf_S > 0

    # Native-grid single onset feature (frac in first 3 layers) also separates.
    fe = frac_in_first_layers(prof_e, 3)
    fg = frac_in_first_layers(prof_g, 3)
    front_auc = roc_auc(fe[np.isfinite(fe)], fg[np.isfinite(fg)])
    print(f"[self-test] native frac-first-3-layers AUC={front_auc:.4f}")
    assert front_auc > 0.9

    print("[self-test] PASS: AUC>0.9 and Fisher S>0 on fabricated data")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

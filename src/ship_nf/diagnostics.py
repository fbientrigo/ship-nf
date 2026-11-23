"""Diagnostics for tilted sampling and for generative models in the tails.

Three questions, one function each:

* How many independent draws is a weighted sample worth?  :func:`kish_ess`.
* Can those weights be trusted at all, or is the estimate driven by a few huge weights?  :func:`psis_khat`.
* Does a generative model put samples where the real tail points are?  :func:`coverage`, :func:`tail_mass_ratio`.

ESS alone is not enough: it summarises the weights you *saw*, so a proposal that never visits part of the target
can show a high ESS and still be wrong. The Pareto ``k`` estimates how heavy the tail of the weight distribution is,
which is what decides whether the importance-sampling estimate converges at a usable rate (Vehtari et al. 2024).

References
----------
Kish, L. (1965). *Survey Sampling*. Wiley.
    ESS = (sum w)^2 / sum w^2.
Elvira, V., Martino, L., & Robert, C. P. (2022). Rethinking the effective sample size.
    *International Statistical Review*, 90(3), 525-550.
    Why Kish ESS is only an approximation and can mislead when the proposal misses the target.
Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling.
    *Journal of Machine Learning Research*, 25(72), 1-58. arXiv:1507.02646.
    k-hat diagnostic: k < 0.5 good, 0.5-0.7 usable with care, > 0.7 unreliable (thresholds for large S).
Zhang, J., & Stephens, M. A. (2009). A new and efficient estimation method for the generalized Pareto distribution.
    *Technometrics*, 51(3), 316-325.
    The estimator used for k-hat (as in PSIS and ArviZ).
Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but did it work?: Evaluating variational inference.
    *ICML 2018*.
    k-hat of p/q as a diagnostic of how well a fitted density q covers a target p (used in the self-consuming study).
Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., & Yoo, J. (2020). Reliable fidelity and diversity metrics for generative
    models. *ICML 2020*. arXiv:2002.09797.
    Coverage: fraction of real points whose k-NN ball (among real points) contains at least one generated sample.

All references were written from memory on 2026-11-19 and must be checked before they go into the thesis.
"""

from __future__ import annotations

import numpy as np


def kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size of non-negative weights ``w``."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _gpd_fit(x: np.ndarray) -> float:
    """Shape ``k`` of a generalised Pareto fitted to sorted positive exceedances ``x`` (Zhang & Stephens 2009),
    with the weak prior towards 0.5 used by PSIS."""
    n = len(x)
    m = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3 * x[int(n / 4 + 0.5) - 1]) + 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ll = n * (np.log(-b / k) - k - 1)
    w = 1 / np.exp(ll - ll[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep] / w[keep].sum(), b[keep]
    k = np.log1p(-(b * w).sum() * x).mean()
    return float((n * k + 10 * 0.5) / (n + 10))


def psis_khat(log_w: np.ndarray) -> float:
    """Pareto ``k`` of the upper tail of importance weights given as ``log_w`` (Vehtari et al. 2024).

    Uses the largest ``min(S / 5, 3 sqrt(S))`` weights. Returns ``inf`` with fewer than 25 weights (too few to fit a
    tail) and ``nan`` when the largest weights are all equal (no tail to fit, e.g. proposal identical to target).
    """
    lw = np.sort(np.asarray(log_w, dtype=float))
    s = len(lw)
    m = int(min(s / 5, 3 * np.sqrt(s)))
    if m < 5:
        return float("inf")
    cut = lw[-m - 1]
    x = np.exp(lw[-m:] - lw[-1]) - np.exp(cut - lw[-1])  # scaled exceedances, same shape k
    x = x[x > 0]
    return _gpd_fit(np.sort(x)) if len(x) >= 5 else float("nan")


def coverage(real: np.ndarray, fake: np.ndarray, k: int = 5) -> float:
    """Naeem et al. (2020) coverage of ``real`` points by ``fake`` samples, in the space the arrays are given in.

    Each real point gets a ball of radius = distance to its ``k``-th nearest *real* neighbour; coverage is the
    fraction of balls that contain at least one fake sample. 1 = every real point has a model sample nearby.
    """
    from scipy.spatial import cKDTree

    r = cKDTree(real).query(real, k + 1)[0][:, -1]
    d = cKDTree(fake).query(real, 1)[0]
    return float((d <= r).mean())


def tail_mass_ratio(in_region_model: np.ndarray, in_region_data: np.ndarray) -> float:
    """``P_model(region) / P_data(region)`` from boolean membership of model samples and data points.

    1 means the model puts the right mass in the region; below 1 it under-covers it (typical in tails).
    """
    pd = float(np.mean(in_region_data))
    return float(np.mean(in_region_model) / pd) if pd > 0 else float("nan")


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_proj: int = 500, seed: int = 0) -> float:
    """Sliced Wasserstein-1 distance between two equal-size samples ``a``, ``b`` of shape ``(n, d)``.

    Both samples are projected on the same ``n_proj`` random unit directions of the full ``d``-D space (fixed by
    ``seed``); the result is the mean over directions of the exact 1-D W1 of the projections, which for equal sizes is
    the mean absolute difference of the sorted values. Because every direction mixes all coordinates, a change in the
    dependence between coordinates is seen even when every marginal is unchanged. Units are those of the inputs.
    Equal sizes are required so that the finite-sample bias is the same for every comparison made with it.
    """
    if a.shape != b.shape:
        raise ValueError(f"equal-size samples required, got {a.shape} and {b.shape}")
    th = np.random.default_rng(seed).standard_normal((n_proj, a.shape[1]))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    return float(np.abs(np.sort(a @ th.T, 0) - np.sort(b @ th.T, 0)).mean())


def knn_two_sample_accuracy(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Leave-one-out ``k``-NN classifier accuracy at telling equal-size samples ``a`` and ``b`` apart (a classifier
    two-sample test). 0.5 = indistinguishable, 1 = fully separated. Ties count as wrong."""
    from scipy.spatial import cKDTree

    x, y = np.vstack([a, b]), np.r_[np.zeros(len(a)), np.ones(len(b))]
    nb = cKDTree(x).query(x, k + 1)[1][:, 1:]
    return float(((y[nb].mean(1) > 0.5) == (y == 1)).mean() * 0.5 + ((y[nb].mean(1) < 0.5) == (y == 0)).mean() * 0.5)


def error_vs_distance(real, gen, d_real, d_gen, edges, metrics, splits=200, seed=0):
    """Distribution error of any generator per distance band. For each ``[lo, hi)`` band of ``edges`` and each
    ``metrics[name](x, y)``: the metric between half of the real rows and an equal-size random subset of the generated
    rows of that band (``name``), and between the two real halves (``name_floor``, the finite-sample floor). Median and
    16-84 % over ``splits``. A band with fewer generated rows than half its real rows gets no metric values."""
    rng, out = np.random.default_rng(seed), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r, g = real[(d_real >= lo) & (d_real < hi)], gen[(d_gen >= lo) & (d_gen < hi)]
        n, row = len(r) // 2, {"lo": float(lo), "hi": float(hi), "real": len(r), "gen": len(g)}
        if len(g) >= n > 0:
            v = {k: ([], []) for k in metrics}
            for _ in range(splits):
                p = rng.permutation(len(r))
                A, B, G = r[p[:n]], r[p[n:2 * n]], g[rng.choice(len(g), n, replace=False)]
                for k, m in metrics.items():
                    v[k][0].append(m(A, G))
                    v[k][1].append(m(A, B))
            for k, (e, f) in v.items():
                row[k], row[k + "_floor"] = np.quantile(e, [0.16, 0.5, 0.84]).tolist(), np.quantile(f, [0.16, 0.5, 0.84]).tolist()
        out.append(row)
    return out

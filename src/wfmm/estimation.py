"""Score, entropy, and moment-parameter estimators.

Bandwidth is an explicit argument. Silverman's rule is one default, not unique.
Directional cosine cannot identify the overall scale of (a, b, beta); moment
laws recover the timescale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


def _as_1d(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).ravel()


def silverman_bandwidth(X: np.ndarray) -> float:
    X = _as_1d(X)
    n = X.size
    sigma = float(np.std(X))
    return float(sigma * n ** (-1.0 / 5.0) + 1e-12)


def score_kde(
    X: np.ndarray,
    query: np.ndarray | None = None,
    bandwidth: float | None = None,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Gaussian-KDE mean-shift estimate of grad log rho.

    grad log rho(x) = (1/h^2) (sum_j w_j(x) x_j - x)
    """
    X = _as_1d(X)
    Q = X if query is None else _as_1d(query)
    h = (silverman_bandwidth(X) if bandwidth is None else float(bandwidth)) * bandwidth_scale
    h2 = h * h
    sq = (Q[:, None] - X[None, :]) ** 2
    logw = -sq / (2.0 * h2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    weighted_mean = w @ X
    return (weighted_mean - Q) / h2


def differential_entropy(X: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko k-NN estimate of differential entropy (nats)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    k = min(k, n - 1)
    tree = cKDTree(X)
    dist, _ = tree.query(X, k=k + 1)
    r = np.maximum(dist[:, k], 1e-12)
    log_vol = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    return float(digamma(n) - digamma(k) + log_vol + (d / n) * np.sum(np.log(r)))


@dataclass
class MomentFit:
    a: float
    a_plus_b: float
    sigma2_inf: float
    b: float | None
    n_mean_pairs: int
    n_var_pairs: int
    mean_r2: float
    var_r2: float


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_moment_laws(
    means: np.ndarray,
    variances: np.ndarray,
    dt: float,
    mean_floor: float = 0.05,
) -> MomentFit:
    """Fit a and (a+b) from consecutive empirical moments.

    mu_{t+dt} ≈ e^{-a dt} mu_t
    Sigma_{t+dt} ≈ sigma_inf + e^{-2(a+b)dt} (Sigma_t - sigma_inf)
                = alpha + gamma Sigma_t
    """
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    mu_t, mu_n = means[:-1], means[1:]
    mask = np.abs(mu_t) > mean_floor
    if np.any(mask):
        y = mu_n[mask]
        x = mu_t[mask]
        gamma_mu = float(np.dot(x, y) / (np.dot(x, x) + 1e-18))
        gamma_mu = float(np.clip(gamma_mu, 1e-8, 1.0 - 1e-8))
        a = -np.log(gamma_mu) / dt
        mean_r2 = _r2(y, gamma_mu * x)
        n_mean = int(mask.sum())
    else:
        a = float("nan")
        mean_r2 = float("nan")
        n_mean = 0

    s_t, s_n = variances[:-1], variances[1:]
    A = np.column_stack([np.ones_like(s_t), s_t])
    coef, *_ = np.linalg.lstsq(A, s_n, rcond=None)
    alpha, gamma = float(coef[0]), float(coef[1])
    gamma = float(np.clip(gamma, 1e-8, 1.0 - 1e-8))
    a_plus_b = -np.log(gamma) / (2.0 * dt)
    sigma2_inf = alpha / (1.0 - gamma)
    var_r2 = _r2(s_n, alpha + gamma * s_t)
    b = (a_plus_b - a) if np.isfinite(a) else None
    return MomentFit(
        a=float(a),
        a_plus_b=float(a_plus_b),
        sigma2_inf=float(sigma2_inf),
        b=None if b is None else float(b),
        n_mean_pairs=n_mean,
        n_var_pairs=int(s_t.size),
        mean_r2=float(mean_r2),
        var_r2=float(var_r2),
    )


def predict_next_moments(
    mu: float, var: float, dt: float, a: float, a_plus_b: float, sigma2_inf: float
) -> tuple[float, float]:
    mu_hat = mu * np.exp(-a * dt)
    var_hat = sigma2_inf + np.exp(-2.0 * a_plus_b * dt) * (var - sigma2_inf)
    return float(mu_hat), float(max(var_hat, 1e-12))

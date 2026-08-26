"""One-dimensional optimal transport: monotone maps and W2."""

from __future__ import annotations

import numpy as np


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def ot_map_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brenier / monotone rearrangement T with T_# empirical(x) = empirical(y).

    This is the minimum-cost coupling of the two *marginals*. It is not the
    trajectory of labeled agents unless identities are observed.
    """
    x = _as_1d(x)
    y = np.sort(_as_1d(y))
    n, m = x.size, y.size
    order = np.argsort(x)
    u = np.empty(n)
    u[order] = (np.arange(n) + 0.5) / n
    yq = (np.arange(m) + 0.5) / m
    return np.interp(u, yq, y)


def displacement_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = _as_1d(x)
    return ot_map_1d(x, y) - x


def w2_samples(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(_as_1d(x))
    y = np.sort(_as_1d(y))
    n = max(x.size, y.size)
    u = (np.arange(n) + 0.5) / n
    qx = np.interp(u, (np.arange(x.size) + 0.5) / x.size, x)
    qy = np.interp(u, (np.arange(y.size) + 0.5) / y.size, y)
    return float(np.sqrt(np.mean((qx - qy) ** 2)))


def w2_densities(m_a: np.ndarray, m_b: np.ndarray, x: np.ndarray, dx: float) -> float:
    """W2 via quantile functions of two grid densities."""
    m_a = np.clip(np.asarray(m_a, dtype=float), 0.0, None)
    m_b = np.clip(np.asarray(m_b, dtype=float), 0.0, None)
    cdf_a = np.cumsum(m_a) * dx
    cdf_b = np.cumsum(m_b) * dx
    if cdf_a[-1] <= 0 or cdf_b[-1] <= 0:
        return float("nan")
    cdf_a = np.clip(cdf_a / cdf_a[-1], 0.0, 1.0)
    cdf_b = np.clip(cdf_b / cdf_b[-1], 0.0, 1.0)
    u = np.linspace(0.0, 1.0, x.size)
    qa = np.interp(u, cdf_a, x)
    qb = np.interp(u, cdf_b, x)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def density_to_samples(m: np.ndarray, x: np.ndarray, dx: float, n: int, rng: np.random.Generator) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype=float), 0.0, None)
    p = m / m.sum()
    return rng.choice(x, size=n, replace=True, p=p) + rng.uniform(-0.5 * dx, 0.5 * dx, size=n)

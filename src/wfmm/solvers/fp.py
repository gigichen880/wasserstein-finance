"""Eulerian Fokker-Planck solvers (conservative upwind + central diffusion, no-flux)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from wfmm.model import Model, free_energy_grid, gaussian_pdf, moments_grid, predicted_drift


def make_grid(L: float = 6.0, n: int = 481) -> tuple[np.ndarray, float]:
    x = np.linspace(-L, L, n)
    return x, float(x[1] - x[0])


def normalize(m: np.ndarray, dx: float) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype=float), 0.0, None)
    s = m.sum() * dx
    if s <= 0:
        return m
    return m / s


def _flux_matrix(x: np.ndarray, dx: float, beta: float, velocity: np.ndarray) -> sparse.csr_matrix:
    n = x.size
    rows, cols, data = [], [], []

    def add(i, j, val):
        rows.append(i)
        cols.append(j)
        data.append(val)

    for i in range(n - 1):
        uf = 0.5 * (velocity[i] + velocity[i + 1])
        adv_i, adv_ip = (uf, 0.0) if uf >= 0 else (0.0, uf)
        d = beta / dx
        add(i, i, -adv_i / dx - d / dx)
        add(i, i + 1, -adv_ip / dx + d / dx)
        add(i + 1, i, adv_i / dx + d / dx)
        add(i + 1, i + 1, adv_ip / dx - d / dx)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def fokker_planck(
    m0: np.ndarray,
    x: np.ndarray,
    dx: float,
    model: Model,
    t_final: float,
    dt: float,
    scheme: str = "implicit",
    snapshot_times: tuple[float, ...] | None = None,
    tilt=None,
    renormalize: bool = True,
) -> tuple[np.ndarray, list[dict], dict]:
    """Evolve ∂t m = ∂x( m [(a+b)x - b μ - a c_t] ) + β ∂xx m.

    ``tilt(t)`` returns the potential center c_t (0 if unforced). Mass reported
    in the history is *before* optional renormalization.
    """
    m = normalize(m0.copy(), dx)
    n_steps = int(round(t_final / dt))
    hist = []
    snapshots = {}
    pending = sorted(snapshot_times or ())
    diverged = False

    for n in range(n_steps):
        t = n * dt
        c = 0.0 if tilt is None else float(tilt(t))
        mu = float(np.sum(x * m) * dx)
        vel = predicted_drift(x, mu, model) + model.a * c
        L = _flux_matrix(x, dx, model.beta, vel)
        if scheme == "explicit":
            m_new = m + dt * (L @ m)
        elif scheme == "implicit":
            A = sparse.eye(x.size, format="csr") - dt * L
            m_new = spsolve(A, m)
        else:
            raise ValueError(scheme)
        mass = float(np.sum(m_new) * dx)
        min_mass = float(np.min(m_new))
        neg_mass = float(np.sum(np.minimum(m_new, 0.0)) * dx)
        if not np.isfinite(m_new).all() or abs(mass) > 1e3:
            diverged = True
        t_next = (n + 1) * dt
        m_clip = np.clip(m_new, 0.0, None)
        mu_n, var_n = moments_grid(m_clip, x, dx)
        m_for_e = normalize(m_new, dx) if np.isfinite(m_new).all() else m_new
        energy = free_energy_grid(m_for_e, x, dx, model) if not diverged else float("inf")
        hist.append(dict(
            t=t_next, energy=energy, mean=mu_n, var=var_n,
            min_mass=min_mass, neg_mass=neg_mass, mass=mass, diverged=diverged,
        ))
        while pending and t_next >= pending[0] - 1e-12:
            snapshots[pending[0]] = normalize(m_new.copy(), dx) if not diverged else m_new.copy()
            pending.pop(0)
        if diverged:
            return m_new, hist, snapshots
        m = normalize(m_new, dx) if (renormalize and scheme == "implicit") else m_new
    return m, hist, snapshots


def cfl_dt(dx: float, beta: float) -> float:
    return dx ** 2 / (2.0 * beta)


def equilibrium_pdf(x: np.ndarray, model: Model) -> np.ndarray:
    return gaussian_pdf(x, 0.0, model.sigma2_inf)

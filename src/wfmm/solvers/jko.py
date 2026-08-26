"""1D JKO scheme in quantile coordinates (paper Appendix B)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from wfmm.model import Model


class JKO1D:
    """Quantile JKO: each step minimizes F[Q] + (1/(2 tau)) ||Q - Qk||^2_{L2(0,1)}.

    Monotonicity is Q_i = z_1 + sum_{j<=i} exp(z_j). Mass and positivity are
    exact by construction.
    """

    def __init__(self, model: Model, m: int = 240):
        self.model = model
        self.M = m
        self.du = 1.0 / m
        self.u = (np.arange(m) + 0.5) / m

    @staticmethod
    def z_to_Q(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inc = np.empty_like(z)
        inc[0] = 0.0
        inc[1:] = np.exp(z[1:])
        Q = np.empty_like(z)
        Q[0] = z[0]
        Q[1:] = z[0] + np.cumsum(inc[1:])
        return Q, inc

    def Q_to_z(self, Q: np.ndarray) -> np.ndarray:
        z = np.empty_like(Q)
        z[0] = Q[0]
        z[1:] = np.log(np.clip(np.diff(Q), 1e-9, None))
        return z

    def energy(self, Q: np.ndarray) -> float:
        a, b, beta = self.model.a, self.model.b, self.model.beta
        du = self.du
        m1 = np.sum(Q) * du
        m2 = np.sum(Q ** 2) * du
        pot = 0.5 * a * m2
        inter = 0.5 * b * (m2 - m1 ** 2)
        slope = np.diff(Q) / du
        ent = -beta * np.sum(np.log(np.clip(slope, 1e-12, None))) * du
        return float(pot + inter + ent)

    def _objective(self, z, Qk, tau):
        a, b, beta = self.model.a, self.model.b, self.model.beta
        du = self.du
        Q, inc = self.z_to_Q(z)
        m1 = np.sum(Q) * du
        m2 = np.sum(Q ** 2) * du
        pot = 0.5 * a * m2
        inter = 0.5 * b * (m2 - m1 ** 2)
        slope = np.diff(Q) / du
        ent = -beta * np.sum(np.log(np.clip(slope, 1e-12, None))) * du
        prox = (0.5 / tau) * np.sum((Q - Qk) ** 2) * du
        J = pot + inter + ent + prox
        Gpot = a * Q * du
        Ginter = b * (Q - m1) * du
        Gprox = (1.0 / tau) * (Q - Qk) * du
        G = Gpot + Ginter + Gprox
        grad = np.empty_like(z)
        suffix = np.cumsum(G[::-1])[::-1]
        grad[0] = np.sum(G)
        grad[1:] = inc[1:] * suffix[1:]
        grad[1:] += -beta * du
        return J, grad

    def step(self, Qk: np.ndarray, tau: float) -> np.ndarray:
        z0 = self.Q_to_z(Qk)
        res = minimize(
            lambda z: self._objective(z, Qk, tau),
            z0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=400, ftol=1e-12, gtol=1e-8),
        )
        Q, _ = self.z_to_Q(res.x)
        return Q

    def flow(self, Q0: np.ndarray, tau: float, n_steps: int) -> list[np.ndarray]:
        Qs = [Q0.copy()]
        Q = Q0.copy()
        for _ in range(n_steps):
            Q = self.step(Q, tau)
            Qs.append(Q.copy())
        return Qs

    def quantile_of_gaussian(self, mu: float, var: float) -> np.ndarray:
        return mu + np.sqrt(var) * norm.ppf(self.u)

    def quantile_of_mixture(self, comps, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = 150000
        ws = np.array([w for w, _, _ in comps], dtype=float)
        ws = ws / ws.sum()
        idx = rng.choice(len(comps), size=n, p=ws)
        samp = np.array([rng.normal(comps[i][1], comps[i][2]) for i in idx])
        return np.quantile(samp, self.u)

    def quantile_of_samples(self, x: np.ndarray) -> np.ndarray:
        return np.quantile(np.asarray(x, dtype=float).ravel(), self.u)

    @staticmethod
    def moments(Q: np.ndarray, du: float | None = None) -> tuple[float, float]:
        du = 1.0 / Q.size if du is None else du
        mu = float(np.sum(Q) * du)
        var = float(np.sum(Q ** 2) * du - mu ** 2)
        return mu, var

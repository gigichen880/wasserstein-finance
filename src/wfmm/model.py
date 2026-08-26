"""Quadratic free energy, first variation, and closed-form moment laws.

F(rho) = (a/2) int x^2 rho dx + (b/2) Var(rho) + beta int rho log rho dx

The interaction W(x,y) = (b/2)(x-y)^2 with b>0 penalizes *dispersion* and
synchronizes inventories (it does not repel similar dealers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from wfmm.estimation import differential_entropy


@dataclass(frozen=True)
class Model:
    a: float = 1.0
    b: float = 0.5
    beta: float = 0.5

    @property
    def sigma2_inf(self) -> float:
        return self.beta / (self.a + self.b)

    @property
    def mean_rate(self) -> float:
        return self.a

    @property
    def variance_rate(self) -> float:
        return 2.0 * (self.a + self.b)

    def mean_law(self, mu0: float, t: np.ndarray | float) -> np.ndarray | float:
        return mu0 * np.exp(-self.a * t)

    def var_law(self, sigma0: float, t: np.ndarray | float) -> np.ndarray | float:
        s = self.sigma2_inf
        return s + (sigma0 - s) * np.exp(-self.variance_rate * t)

    def w2_gaussian_to_eq(self, mu: float, var: float) -> float:
        """W2 between N(mu, var) and N(0, sigma2_inf)."""
        return float(np.sqrt(mu ** 2 + (np.sqrt(max(var, 0.0)) - np.sqrt(self.sigma2_inf)) ** 2))

    def w2_contraction_bound(self, w2_0: float, t: np.ndarray | float) -> np.ndarray | float:
        return w2_0 * np.exp(-self.a * t)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sigma2_inf"] = self.sigma2_inf
        return d


def gaussian_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(float(var), 1e-18)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def bimodal_pdf(x: np.ndarray, left: float = -2.0, right: float = 2.0, sd: float = 0.2) -> np.ndarray:
    return 0.5 * gaussian_pdf(x, left, sd ** 2) + 0.5 * gaussian_pdf(x, right, sd ** 2)


def predicted_drift(x: np.ndarray, mu: float, model: Model) -> np.ndarray:
    """Deterministic McKean-Vlasov drift: v(x) = -(a+b)x + b mu.

    This is the probability-current velocity without the entropic score term;
    diffusion beta Delta rho is written separately in the Fokker-Planck equation.
    """
    return -(model.a + model.b) * x + model.b * mu


def predicted_velocity(x: np.ndarray, mu: float, score: np.ndarray, model: Model) -> np.ndarray:
    """Otto / Wasserstein velocity: -grad (delta F / delta rho)."""
    return predicted_drift(x, mu, model) - model.beta * score


def free_energy_grid(m: np.ndarray, x: np.ndarray, dx: float, model: Model) -> float:
    m = np.asarray(m, dtype=float)
    mu = float(np.sum(x * m) * dx)
    var = float(np.sum((x - mu) ** 2 * m) * dx)
    pot = 0.5 * model.a * float(np.sum(x ** 2 * m) * dx)
    inter = 0.5 * model.b * var
    mpos = np.clip(m, 1e-300, None)
    ent = model.beta * float(np.sum(mpos * np.log(mpos)) * dx)
    return pot + inter + ent


def free_energy_samples(X: np.ndarray, model: Model, k: int = 5) -> float:
    X = np.asarray(X, dtype=float).ravel()
    mu = float(X.mean())
    second = float(np.mean(X ** 2))
    var = float(np.mean((X - mu) ** 2))
    h = differential_entropy(X, k=k)
    return 0.5 * model.a * second + 0.5 * model.b * var - model.beta * h


def moments_grid(m: np.ndarray, x: np.ndarray, dx: float) -> tuple[float, float]:
    mass = float(np.sum(m) * dx)
    if mass <= 0:
        return float("nan"), float("nan")
    mu = float(np.sum(x * m) * dx) / mass
    var = float(np.sum((x - mu) ** 2 * m) * dx) / mass
    return mu, var

"""Particle / McKean-Vlasov generators for synthetic trajectories."""

from __future__ import annotations

import numpy as np

from wfmm.model import Model, predicted_drift


def bimodal_samples(n: int, rng: np.random.Generator, left: float = -2.0,
                    right: float = 2.0, sd: float = 0.2) -> np.ndarray:
    n_left = n // 2
    return np.concatenate([
        rng.normal(left, sd, n_left),
        rng.normal(right, sd, n - n_left),
    ])


def step_mckean_vlasov(X: np.ndarray, model: Model, dt: float, rng: np.random.Generator,
                       substeps: int = 4) -> np.ndarray:
    X = np.asarray(X, dtype=float).ravel().copy()
    h = dt / substeps
    for _ in range(substeps):
        mu = float(X.mean())
        X = X + predicted_drift(X, mu, model) * h + np.sqrt(2.0 * model.beta * h) * rng.standard_normal(X.size)
    return X


def step_wrong_params(X: np.ndarray, true_model: Model, dt: float, rng: np.random.Generator,
                      substeps: int = 4) -> np.ndarray:
    return step_mckean_vlasov(X, true_model, dt, rng, substeps=substeps)


def step_tanh(X: np.ndarray, a: float, beta: float, dt: float, rng: np.random.Generator,
              substeps: int = 4) -> np.ndarray:
    """Nonlinear mean reversion: dX = -a tanh(X) dt + sqrt(2 beta) dW."""
    X = np.asarray(X, dtype=float).ravel().copy()
    h = dt / substeps
    for _ in range(substeps):
        X = X - a * np.tanh(X) * h + np.sqrt(2.0 * beta * h) * rng.standard_normal(X.size)
    return X


def step_state_dep_diffusion(X: np.ndarray, model: Model, dt: float, rng: np.random.Generator,
                             substeps: int = 4) -> np.ndarray:
    """Correct drift, diffusion sqrt(2 beta (1 + 0.5 x^2))."""
    X = np.asarray(X, dtype=float).ravel().copy()
    h = dt / substeps
    for _ in range(substeps):
        mu = float(X.mean())
        vol = np.sqrt(2.0 * model.beta * (1.0 + 0.5 * X ** 2) * h)
        X = X + predicted_drift(X, mu, model) * h + vol * rng.standard_normal(X.size)
    return X


def step_forced(X: np.ndarray, model: Model, dt: float, rng: np.random.Generator,
                force: float = 0.8, substeps: int = 4) -> np.ndarray:
    """True McKean-Vlasov plus a constant omitted force."""
    X = np.asarray(X, dtype=float).ravel().copy()
    h = dt / substeps
    for _ in range(substeps):
        mu = float(X.mean())
        X = X + (predicted_drift(X, mu, model) + force) * h + np.sqrt(2.0 * model.beta * h) * rng.standard_normal(X.size)
    return X


def step_antigradient(X: np.ndarray, model: Model, dt: float, rng: np.random.Generator,
                      substeps: int = 4) -> np.ndarray:
    X = np.asarray(X, dtype=float).ravel().copy()
    h = dt / substeps
    for _ in range(substeps):
        mu = float(X.mean())
        X = X - predicted_drift(X, mu, model) * h + np.sqrt(2.0 * model.beta * h) * rng.standard_normal(X.size)
    return X


def step_translate(X: np.ndarray, shift: float) -> np.ndarray:
    return np.asarray(X, dtype=float).ravel() + shift


def simulate_snapshots(
    X0: np.ndarray,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    stepper,
) -> list[np.ndarray]:
    snaps = [np.asarray(X0, dtype=float).ravel().copy()]
    X = snaps[0].copy()
    for _ in range(n_steps):
        X = stepper(X)
        snaps.append(np.asarray(X, dtype=float).ravel().copy())
    return snaps

"""Unit tests for reusable model/solver/diagnostic components."""

from __future__ import annotations

import numpy as np
import pytest

from wfmm.diagnostics import alignment
from wfmm.estimation import fit_moment_laws, score_kde, silverman_bandwidth
from wfmm.model import Model, predicted_drift, predicted_velocity
from wfmm.solvers.jko import JKO1D
from wfmm.transport import ot_map_1d, w2_samples


def test_sigma2_inf():
    m = Model(a=1.0, b=0.5, beta=0.5)
    assert m.sigma2_inf == pytest.approx(1.0 / 3.0)


def test_mean_and_variance_laws():
    m = Model()
    t = np.linspace(0, 2, 20)
    assert np.allclose(m.mean_law(1.2, t), 1.2 * np.exp(-m.a * t))
    s0 = 0.9
    expected = m.sigma2_inf + (s0 - m.sigma2_inf) * np.exp(-2 * (m.a + m.b) * t)
    assert np.allclose(m.var_law(s0, t), expected)


def test_predicted_drift_mean_reversion():
    x = np.array([-2.0, 0.0, 2.0])
    v = predicted_drift(x, 0.0, Model())
    assert v[0] > 0 and v[2] < 0


def test_ot_map_monotone():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = rng.normal(loc=1.0, size=80)
    t = ot_map_1d(x, y)
    assert np.all(np.diff(t[np.argsort(x)]) >= -1e-12)


def test_w2_translation():
    x = np.linspace(-1, 1, 40)
    assert w2_samples(x, x + 2.0) == pytest.approx(2.0, abs=1e-6)


def test_deterministic_alignment_near_one():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 0.2, 200), rng.normal(2, 0.2, 200)])
    model = Model()
    score = score_kde(x)
    v = predicted_velocity(x, float(x.mean()), score, model)
    y = x + 0.05 * v
    r = alignment(x, y, model, bandwidth=silverman_bandwidth(x))
    assert r.cos > 0.95
    assert r.residual_frac < 0.1
    assert abs(r.cos2 + r.residual_frac - 1.0) < 1e-10


def test_mean_zero_translation_nearly_orthogonal():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 0.2, 200), rng.normal(2, 0.2, 200)])
    r = alignment(x, x + 1.0, Model(), bandwidth=silverman_bandwidth(x))
    assert abs(r.cos) < 0.25


def test_jko_energy_decreases():
    model = Model()
    jko = JKO1D(model, m=60)
    Q0 = jko.quantile_of_gaussian(0.8, 1.0)
    Q1 = jko.step(Q0, tau=0.05)
    assert jko.energy(Q1) <= jko.energy(Q0) + 1e-8


def test_moment_fit_recovers_a_on_exact_law():
    model = Model()
    t = np.arange(0, 1.2, 0.05)
    mu = model.mean_law(1.0, t)
    var = model.var_law(1.0, t)
    fit = fit_moment_laws(mu, var, 0.05, mean_floor=0.05)
    assert fit.a == pytest.approx(model.a, rel=0.05)
    assert fit.a_plus_b == pytest.approx(model.a + model.b, rel=0.08)
    assert fit.sigma2_inf == pytest.approx(model.sigma2_inf, rel=0.08)

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
    assert fit.beta == pytest.approx(model.beta, rel=0.10)


def test_tilt_mean_law():
    m = Model(a=1.0)
    t = np.array([0.0, 1.0, 2.0])
    got = m.mean_law_tilt(0.0, t, c=2.0)
    assert np.allclose(got, 2.0 * (1.0 - np.exp(-t)))


def test_explicit_fp_conserves_signed_mass_below_cfl():
    from wfmm.model import gaussian_pdf
    from wfmm.solvers.fp import cfl_dt, fokker_planck, make_grid, normalize

    model = Model()
    x, dx = make_grid(6.0, 81)
    m0 = normalize(gaussian_pdf(x, 0.8, 0.9), dx)
    dt = 0.4 * cfl_dt(dx, model.beta)
    _, hist, _ = fokker_planck(m0, x, dx, model, t_final=0.2, dt=dt,
                               scheme="explicit", renormalize=False)
    mass = np.array([h["signed_mass"] for h in hist])
    assert np.max(np.abs(mass - 1.0)) < 1e-10
    assert np.min([h["min_mass"] for h in hist]) >= -1e-8


def test_exact_mixture_recovers_moment_laws():
    from wfmm.model import evolve_gaussian_mixture

    model = Model()
    comps0 = [(0.5, -2.0, 0.2 ** 2), (0.5, 2.0, 0.2 ** 2)]
    mu0 = 0.0
    var0 = 0.5 * ((-2.0) ** 2 + 0.04) + 0.5 * (2.0 ** 2 + 0.04)
    t = 0.7
    comps = evolve_gaussian_mixture(comps0, t, model)
    w, m, v = zip(*comps)
    mu = float(np.dot(w, m))
    var = float(np.dot(w, np.array(v) + np.array(m) ** 2) - mu ** 2)
    assert mu == pytest.approx(model.mean_law(mu0, t), abs=1e-12)
    assert var == pytest.approx(model.var_law(var0, t), rel=1e-10)
    comps_id = evolve_gaussian_mixture(comps0, 0.0, model)
    assert comps_id[0][1] == pytest.approx(-2.0)
    assert comps_id[0][2] == pytest.approx(0.04)


def test_jko_mixture_quantile_is_monotone():
    jko = JKO1D(Model(), m=80)
    Q = jko.quantile_of_gaussian_mixture([(0.5, -2.0, 0.04), (0.5, 2.0, 0.04)])
    assert np.all(np.diff(Q) > 0)
    assert jko.w2_to_mixture(Q, [(0.5, -2.0, 0.04), (0.5, 2.0, 0.04)]) < 1e-10


def test_jko_w2_to_exact_gaussian_is_small_after_steps():
    model = Model()
    jko = JKO1D(model, m=80)
    mu0, var0 = 1.0, 1.0
    Q = jko.quantile_of_gaussian(mu0, var0)
    tau = 0.05
    n = 8
    for _ in range(n):
        Q = jko.step(Q, tau)
    t = n * tau
    mu_t, var_t = model.mean_law(mu0, t), model.var_law(var0, t)
    assert jko.w2_to_gaussian(Q, mu_t, var_t) < 0.05

"""Analytic Fisher formulas and snapshot-design invariances."""

from __future__ import annotations

import numpy as np
import pytest

from wfmm.identifiability import (
    constant_variance_optimal_delta,
    constant_variance_spacing_root,
    expected_gaussian_nll_psi,
    fisher_psi,
    gauge_direction_abb,
    lambda_min_mean_leading,
    mean_efficient_information,
    mean_unit_information,
    plugin_three_equal_spacing,
    profiled_fisher_abb,
    schur_complement,
    two_snapshot_avar_a,
    variance_efficient_information,
    variance_kernel_direction_abb,
    variance_unit_information,
)
from wfmm.model import Model


def _fd_hessian(fn, x, step=1e-5):
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n))
    eye = np.eye(n)
    for i in range(n):
        for j in range(i, n):
            di = step * max(1.0, abs(x[i]))
            dj = step * max(1.0, abs(x[j]))
            fpp = fn(x + eye[i] * di + eye[j] * dj)
            fpm = fn(x + eye[i] * di - eye[j] * dj)
            fmp = fn(x - eye[i] * di + eye[j] * dj)
            fmm = fn(x - eye[i] * di - eye[j] * dj)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * di * dj)
            hess[j, i] = hess[i, j]
    return hess


def test_spacing_root_solves_the_corollary_equation():
    x = constant_variance_spacing_root()
    assert x == pytest.approx(1.0 + np.exp(-2.0 * x), rel=1e-10)
    assert 1.10 < x < 1.12
    assert constant_variance_optimal_delta(2.0) == pytest.approx(x / 2.0)


def test_centered_population_has_zero_mean_information_and_singular_abb():
    model = Model(a=1.0, b=0.5, beta=0.5)
    times = np.array([0.0, 0.5, 1.0])
    n = 200
    info = profiled_fisher_abb(model, mu0=0.0, var0=1.5, times=times, n_per_time=n)
    vals = np.linalg.eigvalsh(info)
    assert vals[0] == pytest.approx(0.0, abs=1e-10)
    ia = mean_efficient_information(model.a, 0.0, model.a + model.b, model.beta, 1.5, times, n)
    assert ia == pytest.approx(0.0)


def test_equilibrium_variance_has_zero_k_information():
    model = Model()
    q = model.sigma2_inf
    times = np.array([0.0, 0.5, 1.0])
    ik = variance_efficient_information(model.a + model.b, q, q, times, 400)
    assert ik == pytest.approx(0.0)
    info = profiled_fisher_abb(model, mu0=0.8, var0=q, times=times, n_per_time=400)
    assert np.linalg.eigvalsh(info)[0] == pytest.approx(0.0, abs=1e-8)


def test_mean_schur_matches_efficient_information():
    a, k, q, mu0, var0 = 1.0, 1.5, 1.0 / 3.0, 0.8, 1.5
    times = np.array([0.0, 0.5, 1.0])
    n = 300
    info = fisher_psi(a, k, q, mu0, var0, times, n)
    ia = float(schur_complement(info, keep=[1], nuisance=[0])[0, 0])
    ia_lib = mean_efficient_information(a, mu0, k, k * q, var0, times, n)
    assert ia == pytest.approx(ia_lib, rel=1e-10)
    assert ia / mu0 ** 2 == pytest.approx(
        mean_efficient_information(a, 1.0, k, k * q, var0, times, n), rel=1e-10,
    )


def test_variance_schur_matches_efficient_information():
    a, k, q, mu0, var0 = 1.0, 1.5, 1.0 / 3.0, 0.8, 1.5
    times = np.array([0.0, 0.5, 1.0])
    n = 300
    info = fisher_psi(a, k, q, mu0, var0, times, n)
    ik = float(schur_complement(info, keep=[3], nuisance=[2, 4])[0, 0])
    ik_lib = variance_efficient_information(k, q, var0, times, n)
    assert ik == pytest.approx(ik_lib, rel=1e-8)


def test_two_times_cannot_identify_k_with_unknown_var0_and_q():
    ik = variance_efficient_information(1.5, 1.0 / 3.0, 1.5, np.array([0.0, 1.0]), 500)
    assert ik == pytest.approx(0.0, abs=1e-10)


def test_two_snapshot_avar_and_interior_minimum():
    a, k, q, mu0, var0 = 1.0, 1.5, 1.0 / 3.0, 0.8, 1.5
    deltas = np.linspace(0.15, 3.0, 80)
    vals = [two_snapshot_avar_a(a, k, q, mu0, var0, d) for d in deltas]
    imin = int(np.argmin(vals))
    assert 0 < imin < len(deltas) - 1
    frozen = [two_snapshot_avar_a(a, k, q, mu0, q, d) for d in deltas]
    dstar = constant_variance_optimal_delta(a)
    assert deltas[int(np.argmin(frozen))] == pytest.approx(dstar, abs=0.05)


def test_plugin_recovers_population_moments():
    model = Model()
    mu0, var0, delta = 0.8, 1.5, 0.5
    times = np.array([0.0, delta, 2 * delta])
    mus = np.asarray(model.mean_law(mu0, times), dtype=float)
    var = np.asarray(model.var_law(var0, times), dtype=float)
    hat = plugin_three_equal_spacing(mus, var, delta)
    assert hat is not None
    assert hat["a"] == pytest.approx(model.a, rel=1e-10)
    assert hat["b"] == pytest.approx(model.b, rel=1e-10)
    assert hat["beta"] == pytest.approx(model.beta, rel=1e-10)
    assert plugin_three_equal_spacing(np.zeros(3), var, delta) is None


def test_fisher_matches_expected_nll_hessian():
    psi = np.array([0.8, 1.0, 1.5, 1.5, 1.0 / 3.0])
    times = np.array([0.0, 0.5, 1.0])
    n = 120
    mu0, a, var0, k, q = psi
    info = fisher_psi(a, k, q, mu0, var0, times, n)

    def nll(z):
        return expected_gaussian_nll_psi(z, psi, times, n)

    hess = _fd_hessian(nll, psi, step=3e-5)
    assert np.allclose(info, hess, rtol=2e-3, atol=5e-3)


def test_mean_information_is_exactly_quadratic_in_mu0():
    a, k, beta, var0 = 1.0, 1.5, 0.5, 1.5
    times = np.array([0.0, 0.5, 1.0])
    unit = mean_unit_information(a, k, beta, var0, times, 400)
    for mu0 in (0.0, 0.03, 0.2, 1.7):
        ia = mean_efficient_information(a, mu0, k, beta, var0, times, 400)
        assert ia == pytest.approx(mu0 ** 2 * unit, rel=0, abs=1e-12)


def test_variance_information_is_eps_squared_times_c_eps():
    k, q = 1.5, 1.0 / 3.0
    times = np.array([0.0, 0.5, 1.0])
    n = 400
    c0 = variance_unit_information(k, q, q, times, n)
    assert c0 > 0.0
    for eps in (1e-4, 0.05, 0.4, 1.2):
        c = variance_unit_information(k, q, q + eps, times, n)
        ik = variance_efficient_information(k, q, q + eps, times, n)
        assert ik == pytest.approx(eps ** 2 * c, rel=1e-12)
    assert variance_unit_information(k, q, q + 1e-4, times, n) == pytest.approx(c0, rel=0.02)


def test_lambda_min_collapses_as_half_unit_mean_information():
    model = Model()
    times = np.array([0.0, 0.5, 1.0])
    n = 400
    var0 = 1.5
    lead = lambda_min_mean_leading(model.a, model.a + model.b, model.beta, var0, times, n)
    mu0 = 1e-3
    info = profiled_fisher_abb(model, mu0, var0, times, n)
    lam = float(np.linalg.eigvalsh(info)[0])
    assert lam / mu0 ** 2 == pytest.approx(lead, rel=1e-3)
    g = gauge_direction_abb()
    ia = mean_efficient_information(
        model.a, mu0, model.a + model.b, model.beta, var0, times, n,
    )
    assert g @ info @ g == pytest.approx(ia / 2.0, rel=1e-8)


def test_variance_kernel_quadratic_form_vanishes_at_equilibrium():
    model = Model()
    times = np.array([0.0, 0.5, 1.0])
    info = profiled_fisher_abb(model, 0.8, model.sigma2_inf, times, 400)
    v = variance_kernel_direction_abb(model.sigma2_inf)
    assert v @ info @ v == pytest.approx(0.0, abs=1e-8)


def test_equal_budget_lambda_min_monotone_in_abs_mean():
    model = Model()
    times = np.array([0.0, 0.5, 1.0])
    n = 400
    lams = []
    for mu0 in (0.0, 0.2, 0.8):
        info = profiled_fisher_abb(model, mu0, 1.5, times, n)
        lams.append(float(np.linalg.eigvalsh(info)[0]))
    assert lams[0] == pytest.approx(0.0, abs=1e-10)
    assert lams[1] < lams[2]

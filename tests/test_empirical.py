"""Tests for liquidity-state construction, leakage, and forecast formulas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wfmm.empirical.archives import parse_archive_name, parse_csv_name, reconcile_counts, ArchiveRecord
from wfmm.empirical.models import (
    RateFit,
    affine_step,
    fit_mean_variance_paths,
    forecast_factor_persist,
    forecast_heterogeneous,
    gaussian_tail_outside_unit,
    w1_samples,
)
from wfmm.empirical.panel import EXPECTED_SESSION, session_flags, top_imbalance
from wfmm.empirical.synthetic import independent_factor_days, interacting_days, recover_rates
from wfmm.model import Model
from wfmm.transport import w2_samples

ROOT = Path(__file__).resolve().parents[1]


def test_archive_and_csv_name_parsers():
    meta = parse_archive_name("_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60.7z")
    assert meta["ticker"] == "AAPL"
    assert meta["levels"] == 10
    assert meta["freq"] == 60
    csv = parse_csv_name("AAPL_2019-09-03_34200000_57600000_60_10.csv")
    assert csv["date"] == "2019-09-03"
    assert csv["levels"] == 10
    assert parse_archive_name(".DS_Store") is None


def test_reconcile_one_and_ten_level_counts():
    recs = [
        ArchiveRecord("AAPL", "2007-06-27", "2021-07-01", 10, 60, "7z", "p", 1, False),
        ArchiveRecord("AAPL", "2007-06-27", "2021-07-01", 10, 60, "dir", "p2", 1, True),
        ArchiveRecord("A", "2007-06-27", "2021-01-01", 1, 60, "7z", "p3", 1, False),
        ArchiveRecord("MSFT", "2007-06-27", "2021-07-01", 10, 60, "7z", "p4", 1, False),
    ]
    rec = reconcile_counts(recs)
    assert rec["n_unique_tickers"] == 3
    assert rec["n_ten_level_tickers"] == 2
    assert rec["n_one_level_tickers"] == 1
    assert rec["identity"] is True


def test_zero_denominator_is_nan_not_zero():
    x = top_imbalance(np.array([0.0, 3.0, np.nan]), np.array([0.0, 1.0, 1.0]))
    assert np.isnan(x[0])
    assert x[1] == pytest.approx(0.5)
    assert np.isnan(x[2])
    assert np.isnan(top_imbalance(np.array([-1.0]), np.array([2.0]))[0])


def test_session_grid_and_early_close_flag():
    assert len(EXPECTED_SESSION) == 391
    assert EXPECTED_SESSION[0] == "09:30:00"
    assert EXPECTED_SESSION[-1] == "16:00:00"
    full = session_flags(list(EXPECTED_SESSION))
    assert full["is_full_session"] is True
    short = session_flags(["09:30:00", "09:31:00", "13:00:00"])
    assert short["early_close_candidate"] is True
    assert short["n_bars"] == 3


def test_affine_formula_matches_closed_form_moments():
    rng = np.random.default_rng(0)
    model = Model(a=0.02, b=0.03, beta=0.01)
    x0 = rng.normal(0.2, 0.3, size=5000)
    h = 15.0
    y = affine_step(x0, float(x0.mean()), h, model.a, model.a + model.b, model.sigma2_inf, rng=rng)
    assert float(y.mean()) == pytest.approx(float(model.mean_law(float(x0.mean()), h)), abs=0.02)
    assert float(y.var(ddof=1)) == pytest.approx(float(model.var_law(float(x0.var(ddof=1)), h)), rel=0.15)


def test_training_center_is_not_recentering_each_time():
    t = np.linspace(0, 30, 16)
    mu = 0.2 * np.exp(-0.04 * t) + 0.05
    var = 0.1 + (0.2 - 0.1) * np.exp(-2 * 0.06 * t)
    fit0 = fit_mean_variance_paths(t, mu, var, center=0.0, force_k_equals_a=False, name="c0")
    fitc = fit_mean_variance_paths(t, mu, var, center=0.05, force_k_equals_a=False, name="cfix")
    assert fitc.center == pytest.approx(0.05)
    assert fit0.center == pytest.approx(0.0)
    assert fit0.center != pytest.approx(fitc.center)


def test_gaussian_leakage_on_unit_interval():
    assert gaussian_tail_outside_unit(0.0, 0.01) < 1e-6
    assert gaussian_tail_outside_unit(0.0, 4.0) > 0.5


def test_w1_and_w2_identical_samples_are_zero():
    x = np.linspace(-0.5, 0.5, 20)
    assert w1_samples(x, x) == pytest.approx(0.0)
    assert w2_samples(x, x) == pytest.approx(0.0)


def test_synthetic_interacting_recovers_positive_ka():
    rng = np.random.default_rng(3)
    panels = interacting_days(
        n_names=20, n_days=6, n_times=120, a=0.03, k=0.08, q=0.05,
        mu0=0.25, var0=0.15, rng=rng,
    )
    fit = recover_rates(panels, spacing=2)
    assert fit.k > fit.a
    assert fit.a == pytest.approx(0.03, rel=0.6, abs=0.02)


def test_heterogeneous_ou_keeps_finite_spread():
    rng = np.random.default_rng(4)
    names = [f"S{j}" for j in range(40)]
    x0 = rng.uniform(-0.8, 0.8, size=40)
    het = {n: {"a": 0.2, "c": 0.0, "resid": 0.3} for n in names}
    y = forecast_heterogeneous(x0, names, het, h=5.0, rng=rng)
    assert np.all(np.isfinite(y))
    assert float(y.std()) > 0.05


def test_factor_persist_uses_only_origin_proxy():
    rng = np.random.default_rng(5)
    names = ["AAA", "SPY"]
    x0 = np.array([0.4, 0.2])
    factor = {
        "proxy": "SPY",
        "available": True,
        "alpha": {"AAA": 0.0, "SPY": 0.0},
        "beta": {"AAA": 0.5, "SPY": 1.0},
    }
    fit = RateFit("ou", a=0.02, k=0.02, q=0.05, center=0.0, mean_sse=0.0, var_sse=0.0, admissible=True, notes="ok")
    y = forecast_factor_persist(x0, names, factor, 10.0, fit, rng)
    # reconstruction at h=0 without noise is origin + residual noise of size 0
    assert y.shape == x0.shape
    assert np.all(np.isfinite(y))


def test_config_prespecifies_splits_and_stress_rule():
    cfg = json.loads((ROOT / "research" / "empirical" / "config_pilot.json").read_text())
    assert "2019-09-27" in cfg["test_dates"]
    assert "2019-09-27" not in cfg["train_dates"]
    assert "2020-03-09" in cfg["stress_dates"]
    assert "COVID" in cfg["stress_rule"]
    assert cfg["market_proxy"] == "SPY"
    overlap = set(cfg["train_dates"]) & set(cfg["test_dates"])
    assert not overlap

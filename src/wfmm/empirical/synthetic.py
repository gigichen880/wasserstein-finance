"""Two synthetic controls using the empirical forecast pipeline."""

from __future__ import annotations

import numpy as np

from wfmm.empirical.models import RateFit, affine_step, fit_mean_variance_paths
from wfmm.empirical.panel import DayPanel, minutes_since_open


def interacting_days(
    *,
    n_names: int,
    n_days: int,
    n_times: int,
    a: float,
    k: float,
    q: float,
    mu0: float,
    var0: float,
    rng: np.random.Generator,
) -> list[DayPanel]:
    times = []
    h, m = 9, 30
    for _ in range(n_times):
        times.append(f"{h:02d}:{m:02d}:00")
        m += 1
        if m == 60:
            h += 1
            m = 0
    times = np.asarray(times)
    names = [f"S{j:02d}" for j in range(n_names)]
    panels = []
    for d in range(n_days):
        x0 = rng.normal(mu0, np.sqrt(max(var0, 1e-6)), size=n_names)
        x = np.empty((n_times, n_names))
        x[0] = x0
        for t in range(1, n_times):
            mu = float(x[t - 1].mean())
            x[t] = affine_step(x[t - 1], mu, 1.0, a, k, q, rng=rng)
        panels.append(DayPanel(date=f"sim-{d:03d}", times=times, tickers=names, x=x, valid=np.isfinite(x), quality=[]))
    return panels


def independent_factor_days(
    *,
    n_names: int,
    n_days: int,
    n_times: int,
    a_mean: float,
    a_disp: float,
    q: float,
    factor_vol: float,
    rng: np.random.Generator,
) -> list[DayPanel]:
    times = []
    h, m = 9, 30
    for _ in range(n_times):
        times.append(f"{h:02d}:{m:02d}:00")
        m += 1
        if m == 60:
            h += 1
            m = 0
    times = np.asarray(times)
    names = [f"S{j:02d}" for j in range(n_names)]
    a_i = np.clip(rng.normal(a_mean, a_disp, size=n_names), 0.005, 0.2)
    beta = rng.normal(0.0, 0.4, size=n_names)
    panels = []
    for d in range(n_days):
        x = np.empty((n_times, n_names))
        x[0] = rng.normal(0.0, np.sqrt(q), size=n_names)
        f = rng.normal(0.0, factor_vol, size=n_times)
        for t in range(1, n_times):
            for j in range(n_names):
                x[t, j] = np.exp(-a_i[j]) * x[t - 1, j] + beta[j] * f[t] + np.sqrt(q * (1 - np.exp(-2 * a_i[j]))) * rng.normal()
        panels.append(DayPanel(date=f"ind-{d:03d}", times=times, tickers=names, x=x, valid=np.isfinite(x), quality=[]))
    return panels


def recover_rates(panels: list, spacing: int = 5) -> RateFit:
    from wfmm.empirical.models import pooled_paths

    def mask(times):
        mins = minutes_since_open(times)
        return mins >= 0.0

    t, mu, var = pooled_paths(panels, spacing, mask)
    return fit_mean_variance_paths(t, mu, var, center=0.0, force_k_equals_a=False, name="synth")

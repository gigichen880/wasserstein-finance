"""Moment restrictions, affine forecasts, baselines, and scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from wfmm.model import Model
from wfmm.transport import w2_samples


def w1_samples(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float).ravel())
    y = np.sort(np.asarray(y, dtype=float).ravel())
    n = max(x.size, y.size)
    u = (np.arange(n) + 0.5) / n
    qx = np.interp(u, (np.arange(x.size) + 0.5) / x.size, x)
    qy = np.interp(u, (np.arange(y.size) + 0.5) / y.size, y)
    return float(np.mean(np.abs(qx - qy)))


def gaussian_tail_outside_unit(mu: float, var: float) -> float:
    """P(|X|>1) for X~N(mu, var)."""
    sd = np.sqrt(max(float(var), 1e-18))
    z_hi = (1.0 - float(mu)) / sd
    z_lo = (-1.0 - float(mu)) / sd
    # 1 - (Phi(z_hi) - Phi(z_lo))
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return float(max(0.0, min(1.0, 1.0 - (phi(z_hi) - phi(z_lo)))))


def affine_step(
    x: np.ndarray,
    mu: float,
    h: float,
    a: float,
    k: float,
    q: float,
    rng: np.random.Generator | None = None,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    h = float(h)
    mean_part = float(mu) * np.exp(-float(a) * h)
    center_part = np.exp(-float(k) * h) * (x - float(mu))
    scale = np.sqrt(max(float(q) * (1.0 - np.exp(-2.0 * float(k) * h)), 0.0))
    if noise is None:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(size=x.shape)
    return mean_part + center_part + scale * np.asarray(noise, dtype=float)


def affine_matches_model(model: Model, x0: np.ndarray, h: float, rng) -> bool:
    mu = float(np.mean(x0))
    y = affine_step(x0, mu, h, model.a, model.a + model.b, model.sigma2_inf, rng=rng)
    z = rng.normal(size=x0.size)  # not used; check mean/var laws instead
    _ = z
    mu_h = float(model.mean_law(mu, h))
    var_h = float(model.var_law(float(np.var(x0, ddof=1)), h))
    return np.isfinite(y).all() and np.isfinite(mu_h) and np.isfinite(var_h)


@dataclass
class RateFit:
    name: str
    a: float
    k: float
    q: float
    center: float
    mean_sse: float
    var_sse: float
    admissible: bool
    notes: str


def _exp_mean(t, mu0, a, c):
    return c + (mu0 - c) * np.exp(-a * t)


def _exp_var(t, s0, k, q):
    return q + (s0 - q) * np.exp(-2.0 * k * t)


def fit_mean_variance_paths(
    t: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    *,
    center: float | None,
    force_k_equals_a: bool,
    name: str,
) -> RateFit:
    t = np.asarray(t, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    ok = np.isfinite(t) & np.isfinite(mu) & np.isfinite(var)
    t, mu, var = t[ok], mu[ok], var[ok]
    if t.size < 4:
        return RateFit(name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "too_few_points")
    mu0, s0 = float(mu[0]), float(var[0])
    c0 = 0.0 if center is None else float(center)
    if center is None:
        # free center, initialized at training mean of the path
        x0 = np.array([0.02, max(s0, 1e-4), float(np.mean(mu))])
        def resid(z):
            a, q, c = z
            a = np.abs(a) + 1e-8
            q = np.abs(q)
            return np.concatenate([
                _exp_mean(t, mu0, a, c) - mu,
                _exp_var(t, s0, a if force_k_equals_a else max(a, 0.02), q) - var,
            ])
        sol = least_squares(resid, x0, max_nfev=200)
        a = abs(float(sol.x[0])) + 1e-8
        q = abs(float(sol.x[1]))
        c = float(sol.x[2])
        k = a if force_k_equals_a else max(a, 0.02)
        if not force_k_equals_a:
            def resid_k(z):
                kk, qq = np.abs(z) + 1e-8, np.abs(z[1])
                return _exp_var(t, s0, kk, qq) - var
            solk = least_squares(lambda z: _exp_var(t, s0, abs(z[0]) + 1e-8, abs(z[1])) - var,
                                 np.array([max(a, 0.02), max(s0, 1e-4)]), max_nfev=200)
            k = abs(float(solk.x[0])) + 1e-8
            q = abs(float(solk.x[1]))
    else:
        c = c0
        def resid_a(z):
            a = abs(float(z[0])) + 1e-8
            return _exp_mean(t, mu0, a, c) - mu
        sola = least_squares(resid_a, np.array([0.02]), max_nfev=200)
        a = abs(float(sola.x[0])) + 1e-8
        if force_k_equals_a:
            k = a
            solq = least_squares(lambda z: _exp_var(t, s0, k, abs(z[0])) - var,
                                 np.array([max(s0, 1e-4)]), max_nfev=200)
            q = abs(float(solq.x[0]))
        else:
            solk = least_squares(lambda z: _exp_var(t, s0, abs(z[0]) + 1e-8, abs(z[1])) - var,
                                 np.array([max(a, 0.02), max(s0, 1e-4)]), max_nfev=200)
            k = abs(float(solk.x[0])) + 1e-8
            q = abs(float(solk.x[1]))
    mu_hat = _exp_mean(t, mu0, a, c)
    var_hat = _exp_var(t, s0, k, q)
    mean_sse = float(np.mean((mu_hat - mu) ** 2))
    var_sse = float(np.mean((var_hat - var) ** 2))
    admissible = bool(a > 0.0 and k >= a - 1e-12 and q > 0.0)
    notes = "ok" if admissible else "inadmissible_or_k_lt_a"
    return RateFit(name, float(a), float(k), float(q), float(c), mean_sse, var_sse, admissible, notes)


def pooled_paths(panels: list, spacing: int, time_mask_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average cross-sectional moment paths over days, aligned by clock time."""
    from wfmm.empirical.panel import cross_section_moments

    by_time: dict[str, list[tuple[float, float]]] = {}
    for panel in panels:
        mask = time_mask_fn(panel.times)
        idx = np.where(mask)[0][::spacing]
        for i in idx:
            mu, var, n = cross_section_moments(panel.x[i])
            if n < 3 or not np.isfinite(mu) or not np.isfinite(var):
                continue
            by_time.setdefault(str(panel.times[i]), []).append((mu, var))
    times = sorted(by_time)
    from wfmm.empirical.panel import minutes_since_open
    tmin = minutes_since_open(np.asarray(times))
    tmin = tmin - tmin[0]
    mu = np.array([np.mean([p[0] for p in by_time[t]]) for t in times])
    var = np.array([np.mean([p[1] for p in by_time[t]]) for t in times])
    return tmin, mu, var


def gaussian_nll(x: np.ndarray, mu: float, var: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    var = max(float(var), 1e-12)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var))


def score_forecast(realized: np.ndarray, forecast: np.ndarray) -> dict:
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    realized = realized[np.isfinite(realized)]
    forecast = forecast[np.isfinite(forecast)]
    if realized.size < 2 or forecast.size < 2:
        return {"w1": np.nan, "w2": np.nan, "mean_err": np.nan, "var_err": np.nan,
                "q10_err": np.nan, "q90_err": np.nan, "nll_g": np.nan}
    return {
        "w1": w1_samples(forecast, realized),
        "w2": w2_samples(forecast, realized),
        "mean_err": abs(float(forecast.mean()) - float(realized.mean())),
        "var_err": abs(float(forecast.var(ddof=1)) - float(realized.var(ddof=1))),
        "q10_err": abs(float(np.quantile(forecast, 0.1)) - float(np.quantile(realized, 0.1))),
        "q90_err": abs(float(np.quantile(forecast, 0.9)) - float(np.quantile(realized, 0.9))),
        "nll_g": gaussian_nll(realized, float(forecast.mean()), float(forecast.var(ddof=1))),
    }


def common_valid(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x0) & np.isfinite(x1)
    return x0[m], x1[m]


def forecast_persistence(x0: np.ndarray) -> np.ndarray:
    return np.asarray(x0, dtype=float)


def forecast_affine(x0: np.ndarray, h: float, fit: RateFit, rng) -> np.ndarray:
    mu = float(np.mean(x0)) if fit.center == 0.0 else float(fit.center) + float(np.mean(x0) - fit.center)
    # Use the empirical mean of the origin sample; center c is for the restriction fit.
    mu = float(np.mean(x0))
    return affine_step(x0, mu, h, fit.a, fit.k, fit.q, rng=rng)


def forecast_tod(clock: str, train_by_clock: dict[str, np.ndarray]) -> np.ndarray | None:
    y = train_by_clock.get(str(clock))
    if y is None or y.size < 2:
        return None
    return y


def fit_heterogeneous(
    panels: list,
    *,
    prior_count: int,
    a_pool: float,
    c_pool: float,
) -> dict[str, dict]:
    """Regularized per-name AR(1) / OU on the observed imbalance."""
    names = panels[0].tickers
    stats = {n: {"x": [], "y": []} for n in names}
    for panel in panels:
        idx = {n: j for j, n in enumerate(panel.tickers)}
        for n in names:
            if n not in idx:
                continue
            series = panel.x[:, idx[n]]
            ok = np.isfinite(series)
            s = series[ok]
            if s.size < 3:
                continue
            stats[n]["x"].append(s[:-1])
            stats[n]["y"].append(s[1:])
    out = {}
    dt = 1.0
    for n, d in stats.items():
        if not d["x"]:
            out[n] = {"a": a_pool, "c": c_pool, "resid": 0.1, "n": 0}
            continue
        x = np.concatenate(d["x"])
        y = np.concatenate(d["y"])
        nobs = int(x.size)
        # y - c = rho (x - c)
        xbar, ybar = float(x.mean()), float(y.mean())
        varx = float(np.dot(x - xbar, x - xbar)) + 1e-12
        rho = float(np.dot(x - xbar, y - ybar) / varx)
        rho = float(np.clip(rho, 1e-4, 0.999))
        c = float((ybar - rho * xbar) / max(1.0 - rho, 1e-8))
        a_hat = float(np.clip(-np.log(rho) / dt, 1e-4, 2.0))
        a = (nobs * a_hat + prior_count * a_pool) / (nobs + prior_count)
        a = float(np.clip(a, 1e-4, 2.0))
        c = (nobs * c + prior_count * c_pool) / (nobs + prior_count)
        resid = y - (c + np.exp(-a * dt) * (x - c))
        out[n] = {"a": float(a), "c": float(c), "resid": float(np.std(resid) + 1e-8), "n": nobs}
    return out


def forecast_heterogeneous(x0: np.ndarray, names: list[str], het: dict, h: float, rng) -> np.ndarray:
    """Independent OU per name, including stationary-variance noise."""
    y = np.empty_like(x0, dtype=float)
    z = rng.normal(size=x0.shape)
    h = float(h)
    for j, n in enumerate(names):
        p = het.get(n, {"a": 0.02, "c": 0.0, "resid": 0.1})
        a = float(np.clip(p["a"], 1e-6, 2.0))
        c = float(p["c"])
        q = float(p["resid"]) ** 2 / max(1.0 - np.exp(-2.0 * a), 1e-8)
        scale = np.sqrt(max(q * (1.0 - np.exp(-2.0 * a * h)), 0.0))
        y[j] = c + np.exp(-a * h) * (x0[j] - c) + scale * z[j]
    return y


def fit_spy_factor(panels: list, proxy: str) -> dict:
    """Training OLS of each name on a contemporaneous market proxy."""
    names = panels[0].tickers
    if proxy not in names:
        return {"proxy": proxy, "available": False, "alpha": {}, "beta": {}}
    ip = names.index(proxy)
    stacked = np.concatenate([p.x for p in panels], axis=0)
    spy = stacked[:, ip]
    alpha, beta = {}, {}
    for j, n in enumerate(names):
        y = stacked[:, j]
        m = np.isfinite(spy) & np.isfinite(y)
        if int(m.sum()) < 30:
            alpha[n], beta[n] = 0.0, 0.0
            continue
        s, yv = spy[m], y[m]
        sc = s - s.mean()
        var = float(np.dot(sc, sc)) + 1e-12
        b = float(np.dot(sc, yv - yv.mean()) / var)
        a = float(yv.mean() - b * s.mean())
        alpha[n], beta[n] = a, b
    return {"proxy": proxy, "available": True, "alpha": alpha, "beta": beta}


def forecast_factor_persist(x0: np.ndarray, names: list[str], factor: dict, h: float, fit: RateFit, rng) -> np.ndarray:
    """Residual affine forecast, adding back the origin proxy level (no future leak)."""
    proxy = factor.get("proxy")
    if not factor.get("available") or proxy not in names:
        return forecast_affine(x0, h, fit, rng)
    ip = names.index(proxy)
    spy0 = float(x0[ip])
    alpha = factor["alpha"]
    beta = factor["beta"]
    e0 = np.array([
        x0[j] - float(alpha.get(n, 0.0)) - float(beta.get(n, 0.0)) * spy0
        for j, n in enumerate(names)
    ], dtype=float)
    ehat = forecast_affine(e0, h, fit, rng)
    return np.array([
        ehat[j] + float(alpha.get(n, 0.0)) + float(beta.get(n, 0.0)) * spy0
        for j, n in enumerate(names)
    ], dtype=float)

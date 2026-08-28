"""Alignment, moment-check, and next-distribution forecasting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wfmm.estimation import predict_next_moments, score_kde
from wfmm.model import Model, predicted_velocity
from wfmm.solvers.jko import JKO1D
from wfmm.transport import displacement_1d, w2_samples


def l2_inner(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    return float(np.mean(u * v))


@dataclass
class AlignmentReport:
    cos: float
    cos2: float
    residual_frac: float
    lambda_hat: float
    norm_u: float
    norm_v: float
    bandwidth: float
    n: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def alignment(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    bandwidth: float | None = None,
    bandwidth_scale: float = 1.0,
) -> AlignmentReport:
    """Otto projection of the Brenier displacement onto v_pred.

    residual_frac = |u - λ v_pred|^2 / |u|^2 is the unexplained Wasserstein
    displacement of the two marginals, not identifiable microscopic circulation.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    u = displacement_1d(x, y)
    score = score_kde(x, x, bandwidth=bandwidth, bandwidth_scale=bandwidth_scale)
    v = predicted_velocity(x, float(x.mean()), score, model)
    num = l2_inner(u, v)
    nu = np.sqrt(l2_inner(u, u))
    nv = np.sqrt(l2_inner(v, v))
    cos = float(np.clip(num / (nu * nv + 1e-30), -1.0, 1.0))
    lam = float(num / (nv ** 2 + 1e-30))
    return AlignmentReport(
        cos=cos,
        cos2=cos ** 2,
        residual_frac=1.0 - cos ** 2,
        lambda_hat=lam,
        norm_u=float(nu),
        norm_v=float(nv),
        bandwidth=float(bandwidth) if bandwidth is not None else float("nan"),
        n=int(x.size),
    )


def moment_step_errors(
    mu: np.ndarray, var: np.ndarray, dt: float, a: float, a_plus_b: float, sigma2_inf: float
) -> dict:
    mu_hat = []
    var_hat = []
    for m, s in zip(mu[:-1], var[:-1]):
        mh, sh = predict_next_moments(m, s, dt, a, a_plus_b, sigma2_inf)
        mu_hat.append(mh)
        var_hat.append(sh)
    mu_hat = np.array(mu_hat)
    var_hat = np.array(var_hat)
    mu_true = mu[1:]
    var_true = var[1:]

    def _metrics(y, yhat, prefix):
        err = yhat - y
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float("nan") if ss_tot < 1e-18 else 1.0 - float(np.sum(err ** 2)) / ss_tot
        return {
            f"{prefix}_mae": float(np.mean(np.abs(err))),
            f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))),
            f"{prefix}_r2": r2,
        }

    out = {}
    out.update(_metrics(mu_true, mu_hat, "mean"))
    out.update(_metrics(var_true, var_hat, "var"))
    return out


def forecast_w2(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    dt: float,
    rng: np.random.Generator,
    a: float | None = None,
    a_plus_b: float | None = None,
    sigma2_inf: float | None = None,
    jko_m: int = 120,
    with_jko: bool = True,
) -> dict:
    """One-step distributional forecasts vs the realized next cloud y."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    a = model.a if a is None else a
    a_plus_b = (model.a + model.b) if a_plus_b is None else a_plus_b
    sigma2_inf = model.sigma2_inf if sigma2_inf is None else sigma2_inf
    mu, var = float(x.mean()), float(x.var())
    mu_h, var_h = predict_next_moments(mu, var, dt, a, a_plus_b, sigma2_inf)

    persist = w2_samples(x, y)
    scale = np.sqrt(var_h / max(var, 1e-12))
    gauss_cloud = mu_h + scale * (x - mu)
    gauss = w2_samples(gauss_cloud, y)

    mu0, var0 = predict_next_moments(mu, var, dt, a, a, model.beta / max(a, 1e-12))
    scale0 = np.sqrt(var0 / max(var, 1e-12))
    no_int = w2_samples(mu0 + scale0 * (x - mu), y)

    out = {
        "w2_persistence": persist,
        "w2_gaussian_moments": gauss,
        "w2_no_interaction": no_int,
        "w2_jko": float("nan"),
    }
    if with_jko:
        jko = JKO1D(Model(a=a, b=max(a_plus_b - a, 0.0), beta=model.beta), m=jko_m)
        Q0 = jko.quantile_of_samples(x)
        Q1 = jko.step(Q0, tau=dt)
        out["w2_jko"] = w2_samples(Q1, y)
    return out

"""Exact formulas for confinement--interaction identifiability experiments.

All KL and information formulas in this module use independent Gaussian
snapshots. Exact KL compares fully specified laws. Efficient information
projects the relevant score over explicitly named nuisance parameters.
"""

from __future__ import annotations

import numpy as np

from wfmm.model import Model


def ridge_model(base: Model, h: float) -> Model:
    """Move quadratic mass from interaction to confinement.

    Returns (a+h, b-h, beta), preserving k=a+b and beta. The scalar model class
    requires a+h>0 and b-h>=0.
    """
    model = Model(a=base.a + float(h), b=base.b - float(h), beta=base.beta)
    if model.a <= 0.0 or model.b < 0.0:
        raise ValueError("inadmissible ridge perturbation")
    return model


def gaussian_moments(
    model: Model, mu0: float, var0: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    return (
        np.asarray(model.mean_law(float(mu0), times), dtype=float),
        np.asarray(model.var_law(float(var0), times), dtype=float),
    )


def gaussian_nll(x: np.ndarray, mean: float, var: float) -> float:
    """Mean negative log likelihood under N(mean, var)."""
    x = np.asarray(x, dtype=float)
    var = max(float(var), 1e-15)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var))


def gaussian_w2_same_variance(mean0: float, mean1: float) -> float:
    return float(abs(float(mean0) - float(mean1)))


def gaussian_w2_1d(
    mean0: float | np.ndarray,
    var0: float | np.ndarray,
    mean1: float | np.ndarray,
    var1: float | np.ndarray,
) -> np.ndarray:
    """W2 distance between univariate Gaussian laws."""
    mean0 = np.asarray(mean0, dtype=float)
    mean1 = np.asarray(mean1, dtype=float)
    var0 = np.asarray(var0, dtype=float)
    var1 = np.asarray(var1, dtype=float)
    if np.any(var0 < 0.0) or np.any(var1 < 0.0):
        raise ValueError("Gaussian variances must be nonnegative")
    return np.sqrt((mean0 - mean1) ** 2 + (np.sqrt(var0) - np.sqrt(var1)) ** 2)


def known_design_mean_information(
    a: float,
    initial_means: np.ndarray,
    counts: np.ndarray,
    times: np.ndarray,
    variances: np.ndarray,
) -> float:
    """Mean-block Fisher information for a with known design means.

    ``counts`` may contain one count per population or a population-by-time
    matrix. With common observation times and variance paths, this factors into
    a time term times ``sum_r N_r mu0_r**2``.
    """
    initial_means = np.asarray(initial_means, dtype=float)
    times = np.asarray(times, dtype=float)
    variances = np.asarray(variances, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        if counts.shape != initial_means.shape:
            raise ValueError("population counts must match initial means")
        counts = np.broadcast_to(counts[:, None], (initial_means.size, times.size))
    if counts.shape != (initial_means.size, times.size):
        raise ValueError("counts must have shape (populations, times)")
    if variances.shape != times.shape or np.any(variances <= 0.0):
        raise ValueError("variances must be positive and match times")
    derivative = (
        -initial_means[:, None] * times[None, :] * np.exp(-float(a) * times[None, :])
    )
    return float(np.sum(counts * derivative ** 2 / variances[None, :]))


def _counts(n_per_time: int | np.ndarray, size: int) -> np.ndarray:
    counts = np.asarray(n_per_time, dtype=float)
    if counts.ndim == 0:
        counts = np.full(size, float(counts))
    if counts.shape != (size,) or np.any(counts <= 0):
        raise ValueError("n_per_time must be positive and match times")
    return counts


def snapshot_variances(
    k: float, q: float, var0: float, times: np.ndarray
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return q + (float(var0) - q) * np.exp(-2.0 * float(k) * times)


def mean_kl_known_mu(
    a0: float,
    a1: float,
    mu0: float,
    k: float,
    beta: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """KL(P_a0 || P_a1) at fixed k,beta and known common initial mean."""
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    variances = snapshot_variances(k, beta / k, var0, times)
    delta_mean = mu0 * (np.exp(-a0 * times) - np.exp(-a1 * times))
    return float(0.5 * np.sum(counts * delta_mean ** 2 / variances))


def mean_kl_profile_mu(
    a0: float,
    a1: float,
    mu0: float,
    k: float,
    beta: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> tuple[float, float]:
    """Exact KL minimized over the alternative initial mean.

    Returns (profiled KL, least-distinguishable alternative mu0).
    """
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    variances = snapshot_variances(k, beta / k, var0, times)
    weights = counts / variances
    f = np.exp(-a0 * times)
    g = np.exp(-a1 * times)
    mu1 = float(mu0 * np.sum(weights * f * g) / np.sum(weights * g ** 2))
    kl = 0.5 * np.sum(weights * (mu0 * f - mu1 * g) ** 2)
    return float(kl), mu1


def mean_efficient_information(
    a: float,
    mu0: float,
    k: float,
    beta: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """Information for a after projecting over unknown mu0."""
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    variances = snapshot_variances(k, beta / k, var0, times)
    weights = counts * np.exp(-2.0 * a * times) / variances
    tbar = float(np.sum(weights * times) / np.sum(weights))
    return float(mu0 ** 2 * np.sum(weights * (times - tbar) ** 2))


def variance_kl(
    k0: float,
    k1: float,
    q: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """KL between centered Gaussian snapshots at fixed equilibrium variance q."""
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    s0 = snapshot_variances(k0, q, var0, times)
    s1 = snapshot_variances(k1, q, var0, times)
    ratio = s0 / s1
    return float(0.5 * np.sum(counts * (ratio - 1.0 - np.log(ratio))))


def mean_unit_information(
    a: float,
    k: float,
    beta: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """I_a^eff / mu0^2. Independent of mu0 because S_j does not depend on the mean."""
    return mean_efficient_information(a, 1.0, k, beta, var0, times, n_per_time)


def variance_unit_information(
    k: float,
    q: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """C(eps) in I_k^eff = eps^2 C(eps), including the eps -> 0 limit at S_j = q."""
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    k = float(k)
    q = float(q)
    eps = float(var0) - q
    r = np.exp(-2.0 * k * times)
    if abs(eps) < 1e-15:
        variances = np.full(times.shape, q)
    else:
        variances = q + eps * r
    if np.any(variances <= 0.0):
        raise ValueError("snapshot variances must be positive")
    target = -2.0 * times * r
    nuisance = np.column_stack([r, 1.0 - r])
    sqrt_w = np.sqrt(counts / (2.0 * variances ** 2))
    design = sqrt_w[:, None] * nuisance
    response = sqrt_w * target
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coef
    return float(residual @ residual)


def variance_efficient_information(
    k: float,
    q: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """Information for k after projecting over nuisance (var0, q)."""
    eps = float(var0) - float(q)
    return float(eps ** 2) * variance_unit_information(k, q, var0, times, n_per_time)


def gauge_direction_abb() -> np.ndarray:
    """Unit gauge (1, -1, 0)/sqrt(2) in (a, b, beta)."""
    v = np.array([1.0, -1.0, 0.0])
    return v / np.linalg.norm(v)


def variance_kernel_direction_abb(q: float) -> np.ndarray:
    """Unit direction (0, 1, q) that preserves a and q = beta/(a+b)."""
    v = np.array([0.0, 1.0, float(q)])
    return v / np.linalg.norm(v)


def lambda_min_mean_leading(
    a: float,
    k: float,
    beta: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """Leading coefficient: lambda_min(I_abb)/mu0^2 -> this as mu0->0 at fixed eps.

    Equals I_a^{(1)}/2 because the unit gauge has a-component 1/sqrt(2).
    """
    return 0.5 * mean_unit_information(a, k, beta, var0, times, n_per_time)


def lecam_probability_bound(kl: float) -> float:
    """Pinsker/Le Cam lower bound on max two-point error probability."""
    return float(max(0.0, 0.5 * (1.0 - np.sqrt(max(float(kl), 0.0) / 2.0))))


def lecam_mse_bound(delta: float, kl: float) -> float:
    return float((float(delta) ** 2 / 4.0) * lecam_probability_bound(kl))


# ---------------------------------------------------------------------------
# Full Gaussian snapshot Fisher information and two-snapshot design formulas
# ---------------------------------------------------------------------------

PSI_NAMES = ("mu0", "a", "var0", "k", "q")
ABB_NAMES = ("a", "b", "beta")


def snapshot_mean_variance(
    a: float, k: float, q: float, mu0: float, var0: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    mu = float(mu0) * np.exp(-float(a) * times)
    var = float(q) + (float(var0) - float(q)) * np.exp(-2.0 * float(k) * times)
    return mu, var


def fisher_psi(
    a: float,
    k: float,
    q: float,
    mu0: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> np.ndarray:
    """5x5 Fisher for (mu0, a, var0, k, q) from independent Gaussian snapshots.

    Block diagonal between the mean parameters (mu0, a) and the variance
    parameters (var0, k, q), matching the paper's local asymptotic theorem.
    """
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    if float(q) <= 0.0:
        raise ValueError("equilibrium variance q must be positive")
    mu, var = snapshot_mean_variance(a, k, q, mu0, var0, times)
    if np.any(var <= 0.0):
        raise ValueError("snapshot variances must be positive")
    r = np.exp(-2.0 * float(k) * times)
    eps = float(var0) - float(q)
    dmu = np.zeros((times.size, 5))
    dmu[:, 0] = np.exp(-float(a) * times)
    dmu[:, 1] = -float(mu0) * times * np.exp(-float(a) * times)
    ds = np.zeros((times.size, 5))
    ds[:, 2] = r
    ds[:, 3] = -2.0 * times * eps * r
    ds[:, 4] = 1.0 - r
    info = np.zeros((5, 5))
    for j in range(times.size):
        info += (counts[j] / var[j]) * np.outer(dmu[j], dmu[j])
        info += (counts[j] / (2.0 * var[j] ** 2)) * np.outer(ds[j], ds[j])
    return info


def schur_complement(info: np.ndarray, keep: list[int], nuisance: list[int]) -> np.ndarray:
    info = np.asarray(info, dtype=float)
    i11 = info[np.ix_(keep, keep)]
    if not nuisance:
        return i11
    i12 = info[np.ix_(keep, nuisance)]
    i22 = info[np.ix_(nuisance, nuisance)]
    return i11 - i12 @ np.linalg.pinv(i22) @ i12.T


def jacobian_psi_from_abb(a: float, b: float, beta: float) -> np.ndarray:
    """d(mu0, a, var0, k, q) / d(mu0, a, var0, b, beta)."""
    k = float(a) + float(b)
    if k <= 0.0:
        raise ValueError("k=a+b must be positive")
    q = float(beta) / k
    jac = np.zeros((5, 5))
    jac[0, 0] = 1.0
    jac[1, 1] = 1.0
    jac[2, 2] = 1.0
    jac[3, 1] = 1.0
    jac[3, 3] = 1.0
    jac[4, 1] = -q / k
    jac[4, 3] = -q / k
    jac[4, 4] = 1.0 / k
    return jac


def fisher_abb(
    model: Model,
    mu0: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> np.ndarray:
    """5x5 Fisher for (mu0, a, var0, b, beta)."""
    info_psi = fisher_psi(
        model.a, model.a + model.b, model.sigma2_inf, mu0, var0, times, n_per_time,
    )
    jac = jacobian_psi_from_abb(model.a, model.b, model.beta)
    return jac.T @ info_psi @ jac


def profiled_fisher_abb(
    model: Model,
    mu0: float,
    var0: float,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> np.ndarray:
    """3x3 Fisher for (a, b, beta) after projecting out (mu0, var0)."""
    info = fisher_abb(model, mu0, var0, times, n_per_time)
    return schur_complement(info, keep=[1, 3, 4], nuisance=[0, 2])


def information_spectrum(info: np.ndarray, eig_floor: float = 1e-18) -> dict:
    vals = np.linalg.eigvalsh(np.asarray(info, dtype=float))
    vals = np.clip(vals, 0.0, None)
    lam_min = float(vals[0])
    lam_max = float(vals[-1])
    cond = float(lam_max / max(lam_min, eig_floor)) if lam_max > 0 else float("inf")
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(info)), 0.0, None))
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": cond,
        "eigenvalues": vals,
        "asymptotic_se": se,
        "rank_tol": eig_floor,
        "numerical_rank": int(np.sum(vals > 1e-10 * max(lam_max, 1e-18))),
    }


def two_snapshot_avar_a(
    a: float, k: float, q: float, mu0: float, var0: float, delta: float
) -> float:
    """Asymptotic variance of sqrt(N)(a-hat - a) for the two-snapshot plug-in.

    N samples at t=0 and N at t=Δ. Paper Theorem est / eq. avar.
    """
    delta = float(delta)
    mu0 = float(mu0)
    if delta <= 0.0 or mu0 == 0.0:
        return float("inf")
    _, sig = snapshot_mean_variance(a, k, q, mu0, var0, np.array([0.0, delta]))
    return (1.0 / delta ** 2) * (
        sig[0] / mu0 ** 2 + sig[1] / (mu0 ** 2 * np.exp(-2.0 * float(a) * delta))
    )


def constant_variance_spacing_root() -> float:
    """Unique positive root of x = 1 + exp(-2x)."""
    lo, hi = 0.1, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid - 1.0 - np.exp(-2.0 * mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def constant_variance_optimal_delta(a: float) -> float:
    return constant_variance_spacing_root() / float(a)


def expected_gaussian_nll_psi(
    psi: np.ndarray,
    psi_true: np.ndarray,
    times: np.ndarray,
    n_per_time: int | np.ndarray,
) -> float:
    """Expected NLL of independent Gaussian snapshots (nats, total sample)."""
    mu0, a, var0, k, q = np.asarray(psi, dtype=float)
    mu0_t, a_t, var0_t, k_t, q_t = np.asarray(psi_true, dtype=float)
    times = np.asarray(times, dtype=float)
    counts = _counts(n_per_time, times.size)
    mu, var = snapshot_mean_variance(a, k, q, mu0, var0, times)
    mu_t, var_t = snapshot_mean_variance(a_t, k_t, q_t, mu0_t, var0_t, times)
    if np.any(var <= 0.0):
        return float("inf")
    return float(0.5 * np.sum(
        counts * (np.log(2.0 * np.pi * var) + (var_t + (mu_t - mu) ** 2) / var)
    ))


def plugin_three_equal_spacing(
    means: np.ndarray, variances: np.ndarray, delta: float
) -> dict | None:
    """Plug-in of paper eq. g from three equally spaced snapshot moments."""
    means = np.asarray(means, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()
    delta = float(delta)
    if means.size < 3 or variances.size < 3 or delta <= 0.0:
        return None
    if not np.isfinite(means[:3]).all() or not np.isfinite(variances[:3]).all():
        return None
    if means[0] == 0.0 or means[1] / means[0] <= 0.0:
        return None
    d1 = variances[1] - variances[0]
    d2 = variances[2] - variances[1]
    if d1 == 0.0:
        return None
    ratio = d2 / d1
    if not (0.0 < ratio < 1.0):
        return None
    a_hat = -np.log(means[1] / means[0]) / delta
    k_hat = -np.log(ratio) / (2.0 * delta)
    q_hat = variances[0] - d1 / (ratio - 1.0)
    if not np.isfinite([a_hat, k_hat, q_hat]).all() or a_hat <= 0.0 or q_hat <= 0.0:
        return None
    b_hat = k_hat - a_hat
    beta_hat = k_hat * q_hat
    return {
        "a": float(a_hat),
        "b": float(b_hat),
        "beta": float(beta_hat),
        "k": float(k_hat),
        "q": float(q_hat),
        "admissible": bool(k_hat >= a_hat - 1e-12 and b_hat >= -1e-8),
    }

"""Experiment 13 — separate weak mean and weak variance excitation.

The experiment uses independent Gaussian snapshots and reports:
  * exact two-point KL and Le Cam bounds for fully specified alternatives;
  * nuisance-adjusted efficient information;
  * invalid plug-in rates and conditional RMSE;
  * bounded, everywhere-defined constrained-estimator loss and coverage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import (
    lecam_mse_bound,
    mean_efficient_information,
    mean_kl_profile_mu,
    snapshot_variances,
    variance_efficient_information,
    variance_kl,
)
from wfmm.io_util import save_csv, save_json
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


A = 1.0
K = 1.5
Q = 1.0 / 3.0
BETA = K * Q
TIMES = np.array([0.0, 0.5, 1.0])


def _rmse_se(errors: np.ndarray) -> tuple[float, float]:
    sq = np.asarray(errors, dtype=float) ** 2
    mse = float(np.mean(sq))
    rmse = float(np.sqrt(mse))
    if rmse == 0.0 or sq.size < 2:
        return rmse, 0.0
    mse_se = float(np.std(sq, ddof=1) / np.sqrt(sq.size))
    return rmse, mse_se / (2.0 * rmse)


def _mean_profile_estimator(
    sample_means: np.ndarray, variances: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained profile MLE for a with unknown mu0 and known variances."""
    grid = np.linspace(0.05, K - 0.05, 281)
    basis = np.exp(-grid[:, None] * TIMES[None, :])
    weights = n / variances
    denom = np.sum(weights[None, :] * basis ** 2, axis=1)
    numer = sample_means @ (weights[None, :] * basis).T
    mu_hat = numer / denom[None, :]
    residual = sample_means[:, None, :] - mu_hat[:, :, None] * basis[None, :, :]
    profile = np.sum(weights[None, None, :] * residual ** 2, axis=2)
    index = np.argmin(profile, axis=1)
    a_hat = grid[index]

    truth_basis = np.exp(-A * TIMES)
    truth_mu = (
        np.sum(weights[None, :] * sample_means * truth_basis[None, :], axis=1)
        / np.sum(weights * truth_basis ** 2)
    )
    truth_residual = sample_means - truth_mu[:, None] * truth_basis[None, :]
    truth_profile = np.sum(weights[None, :] * truth_residual ** 2, axis=1)
    lr_truth = truth_profile - profile[np.arange(sample_means.shape[0]), index]
    return a_hat, lr_truth


def _mean_rows(rng: np.random.Generator, reps: int) -> list[dict]:
    rows = []
    ns = (100, 400, 1600)
    scaled_signals = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    var0 = 1.5
    delta_alt = 0.25
    variances = snapshot_variances(K, Q, var0, TIMES)
    for n in ns:
        for scaled in scaled_signals:
            mu0 = scaled / np.sqrt(n)
            means = mu0 * np.exp(-A * TIMES)
            sample_means = rng.normal(
                means[None, :], np.sqrt(variances[None, :] / n), size=(reps, TIMES.size)
            )
            a_hat, lr_truth = _mean_profile_estimator(sample_means, variances, n)
            error = a_hat - A
            rmse, rmse_se = _rmse_se(error)
            coverage = float(np.mean(lr_truth <= 3.841458820694124))

            ratio = sample_means[:, 1] / sample_means[:, 0]
            plugin_valid = np.isfinite(ratio) & (ratio > 0.0)
            plugin = np.full(reps, np.nan)
            plugin[plugin_valid] = -np.log(ratio[plugin_valid]) / TIMES[1]
            plugin_valid &= (plugin > 0.0) & (plugin < K)
            conditional_rmse = (
                float(np.sqrt(np.mean((plugin[plugin_valid] - A) ** 2)))
                if np.any(plugin_valid) else float("nan")
            )
            invalid_rate = float(1.0 - np.mean(plugin_valid))

            information = mean_efficient_information(
                A, mu0, K, BETA, var0, TIMES, n
            )
            kl, alternative_mu = mean_kl_profile_mu(
                A, A + delta_alt, mu0, K, BETA, var0, TIMES, n
            )
            lower = lecam_mse_bound(delta_alt, kl)
            bounded_mse = float(np.mean(np.minimum(error ** 2, 1.0)))
            rows.append(dict(
                mechanism="mean",
                n=n,
                signal=mu0,
                candidate_scaling=n * mu0 ** 2,
                efficient_information=information,
                exact_profiled_kl=kl,
                alternative_delta=delta_alt,
                least_favorable_alternative_mu0=alternative_mu,
                lecam_mse_bound=lower,
                constrained_rmse=rmse,
                constrained_rmse_se=rmse_se,
                unconditional_bounded_mse=bounded_mse,
                plugin_invalid_rate=invalid_rate,
                plugin_invalid_se=np.sqrt(invalid_rate * (1.0 - invalid_rate) / reps),
                plugin_conditional_rmse=conditional_rmse,
                profile_lr_coverage_95=coverage,
                coverage_se=np.sqrt(coverage * (1.0 - coverage) / reps),
            ))
    return rows


def _variance_rows(rng: np.random.Generator, reps: int) -> list[dict]:
    rows = []
    ns = (100, 400, 1600)
    scaled_signals = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    delta_alt = 0.20
    dt = TIMES[1] - TIMES[0]
    k_min = A + 0.01
    k_max = 3.0
    r_min = np.exp(-2.0 * k_max * dt)
    r_max = np.exp(-2.0 * k_min * dt)
    for n in ns:
        for scaled in scaled_signals:
            epsilon = scaled / np.sqrt(n)
            var0 = Q + epsilon
            variances = snapshot_variances(K, Q, var0, TIMES)
            sample_vars = (
                variances[None, :]
                * rng.chisquare(n - 1, size=(reps, TIMES.size))
                / (n - 1)
            )
            d1 = sample_vars[:, 1] - sample_vars[:, 0]
            d2 = sample_vars[:, 2] - sample_vars[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_r = d2 / d1
                raw_k = -np.log(raw_r) / (2.0 * dt)
                raw_q = sample_vars[:, 0] - d1 / (raw_r - 1.0)
            valid = (
                np.isfinite(raw_r) & (raw_r > 0.0) & (raw_r < 1.0)
                & np.isfinite(raw_k) & (raw_k > A) & (raw_k < k_max)
                & np.isfinite(raw_q) & (raw_q > 0.0)
            )
            conditional_rmse = (
                float(np.sqrt(np.mean((raw_k[valid] - K) ** 2)))
                if np.any(valid) else float("nan")
            )
            invalid_rate = float(1.0 - np.mean(valid))

            # Defined on every sample and bounded by construction. It is not
            # called efficient; it supplies an unconditional performance metric.
            safe_r = np.nan_to_num(raw_r, nan=r_max, posinf=r_max, neginf=r_min)
            clipped_r = np.clip(safe_r, r_min, r_max)
            clipped_k = -np.log(clipped_r) / (2.0 * dt)
            error = clipped_k - K
            rmse, rmse_se = _rmse_se(error)

            information = variance_efficient_information(K, Q, var0, TIMES, n)
            half_width = (
                1.959963984540054 / np.sqrt(information)
                if information > 0.0 else float("inf")
            )
            coverage = float(np.mean(np.abs(error) <= half_width))
            kl = variance_kl(K, K + delta_alt, Q, var0, TIMES, n)
            lower = lecam_mse_bound(delta_alt, kl)
            bounded_mse = float(np.mean(np.minimum(error ** 2, 4.0)))
            rows.append(dict(
                mechanism="variance",
                n=n,
                signal=epsilon,
                candidate_scaling=n * epsilon ** 2,
                efficient_information=information,
                exact_kl=kl,
                alternative_delta=delta_alt,
                lecam_mse_bound=lower,
                clipped_rmse=rmse,
                clipped_rmse_se=rmse_se,
                unconditional_bounded_mse=bounded_mse,
                plugin_invalid_rate=invalid_rate,
                plugin_invalid_se=np.sqrt(invalid_rate * (1.0 - invalid_rate) / reps),
                plugin_conditional_rmse=conditional_rmse,
                oracle_information_coverage_95=coverage,
                coverage_se=np.sqrt(coverage * (1.0 - coverage) / reps),
            ))
    return rows


def run(seed: int = 13, reps: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    mean_rows = _mean_rows(rng, reps)
    variance_rows = _variance_rows(rng, reps)
    rows = mean_rows + variance_rows

    def ratio_range(part):
        ratios = [
            r["efficient_information"] / r["candidate_scaling"]
            for r in part if r["candidate_scaling"] > 0
        ]
        return min(ratios), max(ratios)

    mean_ratio = ratio_range(mean_rows)
    variance_ratio = ratio_range(variance_rows)
    metrics = dict(
        claim="weak_identification_separate_mechanisms",
        seed=seed,
        replications=reps,
        observation_times=TIMES,
        base=dict(a=A, k=K, b=K - A, q=Q, beta=BETA),
        mean_information_per_candidate_scaling=mean_ratio,
        variance_information_per_candidate_scaling=variance_ratio,
        max_mean_invalid_rate=max(r["plugin_invalid_rate"] for r in mean_rows),
        max_variance_invalid_rate=max(r["plugin_invalid_rate"] for r in variance_rows),
        mean_coverage_range=(
            min(r["profile_lr_coverage_95"] for r in mean_rows),
            max(r["profile_lr_coverage_95"] for r in mean_rows),
        ),
        variance_coverage_range=(
            min(r["oracle_information_coverage_95"] for r in variance_rows),
            max(r["oracle_information_coverage_95"] for r in variance_rows),
        ),
        rows=rows,
        notes=(
            "Mean alternatives vary a at fixed k,beta. Variance alternatives vary k "
            "at fixed q=beta/k. Exact KL is fully specified; efficient information "
            "projects named nuisance parameters. Conditional plugin RMSE is always "
            "reported with invalid-event rates. Bounded constrained/clipped losses "
            "are defined on every replicate."
        ),
    )
    return dict(rows=rows, metrics=metrics)


def figure(data: dict, path: Path) -> None:
    rows = data["rows"]
    mean_rows = [r for r in rows if r["mechanism"] == "mean"]
    var_rows = [r for r in rows if r["mechanism"] == "variance"]
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.2))

    for n in ns:
        m = sorted([r for r in mean_rows if r["n"] == n], key=lambda r: r["efficient_information"])
        x = np.array([r["efficient_information"] for r in m])
        axes[0, 0].plot(x, [r["constrained_rmse"] for r in m], "o-", label=f"N={n}")
        axes[0, 1].plot(x, [r["plugin_invalid_rate"] for r in m], "o-")
        axes[0, 2].plot(x, [r["profile_lr_coverage_95"] for r in m], "o-")

        v = sorted([r for r in var_rows if r["n"] == n], key=lambda r: r["efficient_information"])
        x = np.array([r["efficient_information"] for r in v])
        axes[1, 0].plot(x, [r["clipped_rmse"] for r in v], "o-", label=f"N={n}")
        axes[1, 1].plot(x, [r["plugin_invalid_rate"] for r in v], "o-")
        axes[1, 2].plot(x, [r["oracle_information_coverage_95"] for r in v], "o-")

    titles = (
        "constrained RMSE", "plug-in invalid rate", "95% coverage",
        "clipped RMSE", "plug-in invalid rate", "95% coverage",
    )
    for i, ax in enumerate(axes.flat):
        ax.set_xscale("log")
        ax.set_xlabel("nuisance-adjusted information")
        ax.set_title(titles[i], fontsize=9)
        panel_label(ax, f"({chr(ord('a') + i)})")
    axes[0, 0].set_ylabel("weak mean excitation")
    axes[1, 0].set_ylabel("weak variance excitation")
    axes[0, 2].axhline(0.95, color="0.4", ls="--", lw=1)
    axes[1, 2].axhline(0.95, color="0.4", ls="--", lw=1)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    save_fig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run(reps=200 if args.quick else 1000)
    figure(data, FIGS_DIR / "exp13_weak_identification.png")
    metrics = dict(data["metrics"])
    metrics["runtime_sec"] = time.time() - t0
    metrics["command"] = "python experiments/13_weak_identification.py"
    save_json(RESULTS_DIR / "exp13_weak_identification.json", metrics)
    save_csv(RESULTS_DIR / "exp13_weak_identification.csv", data["rows"])

    paper_figs = Path(__file__).resolve().parents[1] / "paper" / "figs"
    paper_figs.mkdir(exist_ok=True)
    (paper_figs / "exp13_weak_identification.png").write_bytes(
        (FIGS_DIR / "exp13_weak_identification.png").read_bytes()
    )
    print(
        "exp13 max invalid rates",
        metrics["max_mean_invalid_rate"],
        metrics["max_variance_invalid_rate"],
        "coverage ranges",
        metrics["mean_coverage_range"],
        metrics["variance_coverage_range"],
    )


if __name__ == "__main__":
    main()

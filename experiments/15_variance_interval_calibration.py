"""Experiment 15 — interval calibration under weak variance excitation.

The clipped ratio estimator from Experiment 13 is compared with an exact
Gaussian variance MLE, likelihood-ratio profile sets, and basic parametric
bootstrap sets.  Confidence sets are stored as components; interval hulls are
reported separately and full-domain sets are not collapsed to finite-looking
error bars.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import variance_efficient_information
from wfmm.io_util import save_csv, save_json
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


K_TRUE = 1.5
Q_TRUE = 1.0 / 3.0
A_REFERENCE = 1.0
TIMES = np.array([0.0, 0.5, 1.0])
NS = (100, 400, 1600)
SCALED_SIGNALS = (0.5, 2.0, 8.0, 16.0)
K_BOUNDS = (1.01, 3.00)
Q_BOUNDS = (0.02, 2.00)
S0_BOUNDS = (0.02, 2.50)
K_GRID = np.linspace(K_BOUNDS[0], K_BOUNDS[1], 51)
LR95 = 3.841458820694124


def variance_path(k: float, q: float, s0: float) -> np.ndarray:
    return q + (s0 - q) * np.exp(-2.0 * k * TIMES)


def variance_nll(sample_vars: np.ndarray, n: int, k: float, q: float, s0: float) -> float:
    v = variance_path(k, q, s0)
    if np.any(v <= 0.0):
        return float("inf")
    df = n - 1
    return float(0.5 * df * np.sum(np.log(v) + sample_vars / v))


def variance_nll_gradient(
    sample_vars: np.ndarray, n: int, k: float, q: float, s0: float
) -> np.ndarray:
    r = np.exp(-2.0 * k * TIMES)
    v = q + (s0 - q) * r
    multiplier = 0.5 * (n - 1) * (v - sample_vars) / v ** 2
    derivatives = np.column_stack([
        -2.0 * TIMES * (s0 - q) * r,
        1.0 - r,
        r,
    ])
    return multiplier @ derivatives


def fit_variance(sample_vars: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    fit = minimize(
        lambda x: variance_nll(sample_vars, n, x[0], x[1], x[2]),
        jac=lambda x: variance_nll_gradient(sample_vars, n, x[0], x[1], x[2]),
        x0=np.array([K_TRUE, Q_TRUE, max(float(sample_vars[0]), Q_BOUNDS[0])]),
        method="L-BFGS-B",
        bounds=[K_BOUNDS, Q_BOUNDS, S0_BOUNDS],
        options={"ftol": 1e-11, "maxiter": 250},
    )
    if not fit.success:
        fit = minimize(
            lambda x: variance_nll(sample_vars, n, x[0], x[1], x[2]),
            x0=np.array([K_TRUE, Q_TRUE, max(float(sample_vars[0]), Q_BOUNDS[0])]),
            method="Powell",
            bounds=[K_BOUNDS, Q_BOUNDS, S0_BOUNDS],
            options={"ftol": 1e-10, "maxiter": 600},
        )
    if not fit.success:
        raise RuntimeError(f"variance MLE failed: {fit.message}")
    return np.asarray(fit.x), float(fit.fun)


def profile_curve(sample_vars: np.ndarray, n: int, minimum: float) -> np.ndarray:
    values = np.empty(K_GRID.size)
    previous = np.array([Q_TRUE, np.clip(sample_vars[0], *S0_BOUNDS)])
    for i, k in enumerate(K_GRID):
        fit = minimize(
            lambda x: variance_nll(sample_vars, n, float(k), x[0], x[1]),
            jac=lambda x: variance_nll_gradient(
                sample_vars, n, float(k), x[0], x[1]
            )[1:],
            x0=previous,
            method="L-BFGS-B",
            bounds=[Q_BOUNDS, S0_BOUNDS],
            options={"ftol": 1e-11, "maxiter": 150},
        )
        if not fit.success:
            fit = minimize(
                lambda x: variance_nll(sample_vars, n, float(k), x[0], x[1]),
                x0=previous,
                method="Powell",
                bounds=[Q_BOUNDS, S0_BOUNDS],
                options={"ftol": 1e-10, "maxiter": 400},
            )
        if not fit.success:
            raise RuntimeError(f"variance profile failed: {fit.message}")
        previous = np.asarray(fit.x)
        values[i] = 2.0 * (float(fit.fun) - minimum)
    return np.maximum(values, 0.0)


def set_components(grid: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [[float(grid[g[0]]), float(grid[g[-1]])] for g in np.split(indices, split)]


def grid_covers_truth(selected: np.ndarray) -> bool:
    nearest = int(np.argmin(np.abs(K_GRID - K_TRUE)))
    return bool(selected[nearest])


def clipped_ratio(sample_vars: np.ndarray) -> float:
    dt = TIMES[1] - TIMES[0]
    raw_r = (sample_vars[2] - sample_vars[1]) / (sample_vars[1] - sample_vars[0])
    r_min = np.exp(-2.0 * K_BOUNDS[1] * dt)
    r_max = np.exp(-2.0 * K_BOUNDS[0] * dt)
    safe = np.nan_to_num(raw_r, nan=r_max, posinf=r_max, neginf=r_min)
    return float(-np.log(np.clip(safe, r_min, r_max)) / (2.0 * dt))


def bootstrap_set(
    rng: np.random.Generator,
    fitted: np.ndarray,
    n: int,
    draws: int,
) -> tuple[list[list[float]], bool]:
    k_hat, q_hat, s0_hat = fitted
    v_hat = variance_path(k_hat, q_hat, s0_hat)
    boot = np.empty(draws)
    for j in range(draws):
        sample = v_hat * rng.chisquare(n - 1, size=TIMES.size) / (n - 1)
        boot[j] = fit_variance(sample, n)[0][0]
    error_lo, error_hi = np.quantile(boot - k_hat, [0.025, 0.975])
    lo = max(K_BOUNDS[0], float(k_hat - error_hi))
    hi = min(K_BOUNDS[1], float(k_hat - error_lo))
    components = [[lo, hi]] if lo <= hi else []
    covers = bool(lo <= K_TRUE <= hi)
    return components, covers


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(values) - K_TRUE) ** 2)))


def run(
    seed: int = 15,
    reps: int = 250,
    bootstrap_outer: int = 60,
    bootstrap_draws: int = 99,
) -> dict:
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict] = []

    for n in NS:
        for scaled in SCALED_SIGNALS:
            epsilon = scaled / np.sqrt(n)
            s0 = Q_TRUE + epsilon
            truth_v = variance_path(K_TRUE, Q_TRUE, s0)
            information = variance_efficient_information(
                K_TRUE, Q_TRUE, s0, TIMES, n
            )
            oracle_half_width = (
                1.959963984540054 / np.sqrt(information)
                if information > 0.0 else float("inf")
            )
            for rep in range(reps):
                sample_vars = truth_v * rng.chisquare(n - 1, size=TIMES.size) / (n - 1)
                clipped = clipped_ratio(sample_vars)
                fitted, minimum = fit_variance(sample_vars, n)
                profile = profile_curve(sample_vars, n, minimum)
                selected = profile <= LR95
                components = set_components(K_GRID, selected)
                hull_width = (
                    float(K_GRID[selected][-1] - K_GRID[selected][0])
                    if np.any(selected) else float("nan")
                )
                if rep < bootstrap_outer:
                    boot_components, boot_covers = bootstrap_set(
                        rng, fitted, n, bootstrap_draws
                    )
                    boot_width = (
                        boot_components[0][1] - boot_components[0][0]
                        if boot_components else float("nan")
                    )
                    boot_full = bool(
                        boot_components
                        and boot_components[0][0] == K_BOUNDS[0]
                        and boot_components[0][1] == K_BOUNDS[1]
                    )
                else:
                    boot_components, boot_covers, boot_width, boot_full = [], None, np.nan, False

                artificial_contact = bool(
                    min(
                        abs(fitted[0] - K_BOUNDS[0]), abs(fitted[0] - K_BOUNDS[1]),
                        abs(fitted[1] - Q_BOUNDS[0]), abs(fitted[1] - Q_BOUNDS[1]),
                        abs(fitted[2] - S0_BOUNDS[0]), abs(fitted[2] - S0_BOUNDS[1]),
                    ) < 1e-4
                )
                replicate_rows.append({
                    "n": n,
                    "scaled_signal": scaled,
                    "signal": epsilon,
                    "candidate_scaling": n * epsilon ** 2,
                    "efficient_information": information,
                    "replicate": rep,
                    "clipped_k": clipped,
                    "clipped_oracle_wald_covers": float(abs(clipped - K_TRUE) <= oracle_half_width),
                    "mle_k": float(fitted[0]),
                    "mle_q": float(fitted[1]),
                    "mle_s0": float(fitted[2]),
                    "mle_artificial_bound_contact": float(artificial_contact),
                    "weak_surface_distance": float(abs(fitted[2] - fitted[1])),
                    "profile_components": repr(components),
                    "profile_component_count": len(components),
                    "profile_covers": float(grid_covers_truth(selected)),
                    "profile_full_domain": float(np.all(selected)),
                    "profile_hull_width": hull_width,
                    "bootstrap_components": repr(boot_components),
                    "bootstrap_covers": (
                        float(boot_covers) if boot_covers is not None else float("nan")
                    ),
                    "bootstrap_full_domain": (
                        float(boot_full) if boot_covers is not None else float("nan")
                    ),
                    "bootstrap_hull_width": boot_width,
                })

    summary_rows: list[dict] = []
    for n in NS:
        for scaled in SCALED_SIGNALS:
            part = [
                r for r in replicate_rows
                if r["n"] == n and r["scaled_signal"] == scaled
            ]
            boot = [r for r in part if np.isfinite(r["bootstrap_covers"])]
            summary_rows.append({
                "n": n,
                "scaled_signal": scaled,
                "signal": part[0]["signal"],
                "candidate_scaling": part[0]["candidate_scaling"],
                "efficient_information": part[0]["efficient_information"],
                "replications": reps,
                "bootstrap_replications": len(boot),
                "bootstrap_draws": bootstrap_draws,
                "clipped_rmse": _rmse([r["clipped_k"] for r in part]),
                "clipped_oracle_wald_coverage": float(np.mean(
                    [r["clipped_oracle_wald_covers"] for r in part]
                )),
                "mle_rmse": _rmse([r["mle_k"] for r in part]),
                "mle_artificial_bound_rate": float(np.mean(
                    [r["mle_artificial_bound_contact"] for r in part]
                )),
                "profile_coverage": float(np.mean([r["profile_covers"] for r in part])),
                "profile_full_domain_rate": float(np.mean(
                    [r["profile_full_domain"] for r in part]
                )),
                "profile_disconnected_rate": float(np.mean(
                    [r["profile_component_count"] > 1 for r in part]
                )),
                "profile_mean_hull_width": float(np.nanmean(
                    [r["profile_hull_width"] for r in part]
                )),
                "bootstrap_coverage": float(np.mean(
                    [r["bootstrap_covers"] for r in boot]
                )),
                "bootstrap_full_domain_rate": float(np.mean(
                    [r["bootstrap_full_domain"] for r in boot]
                )),
                "bootstrap_mean_hull_width": float(np.nanmean(
                    [r["bootstrap_hull_width"] for r in boot]
                )),
            })

    metrics = {
        "claim": "variance_interval_methods_require_empirical_calibration",
        "seed": seed,
        "times": TIMES,
        "true_k": K_TRUE,
        "true_q": Q_TRUE,
        "replications": reps,
        "bootstrap_outer_replications": bootstrap_outer,
        "bootstrap_draws": bootstrap_draws,
        "profile_lr_cutoff": LR95,
        "scientific_domain": "k>a_reference, q>0, sigma0>0",
        "a_reference": A_REFERENCE,
        "compact_numerical_bounds": {
            "k": K_BOUNDS, "q": Q_BOUNDS, "sigma0": S0_BOUNDS,
        },
        "bootstrap_construction": (
            "Basic parametric bootstrap: fit constrained Gaussian variance MLE; "
            "simulate independent sample variances from the fitted path; refit each "
            "draw; invert the 2.5% and 97.5% quantiles of k_boot-k_hat and intersect "
            "with the reported compact domain."
        ),
        "profile_reporting": (
            "Grid confidence sets are retained as connected components. Full-domain "
            "sets and interval-hull widths are reported separately."
        ),
        "notes": (
            "The clipped estimator's oracle-information Wald width is not its own "
            "sampling standard error. Ratio noise, clipping, skewness, and point "
            "masses at compact limits are estimator-specific; low efficient "
            "information and full-domain profile sets are identification effects. "
            "Bootstrap coverage is evaluated, not assumed."
        ),
        "summary_rows": summary_rows,
    }
    return {"replicate_rows": replicate_rows, "summary_rows": summary_rows, "metrics": metrics}


def figure(data: dict, path: Path) -> None:
    rows = data["summary_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.2))
    for n in NS:
        part = sorted(
            [r for r in rows if r["n"] == n],
            key=lambda r: r["efficient_information"],
        )
        x = np.array([r["efficient_information"] for r in part])
        axes[0].plot(x, [r["clipped_oracle_wald_coverage"] for r in part],
                     "o--", label=f"clipped, N={n}")
        axes[0].plot(x, [r["profile_coverage"] for r in part], "s-",
                     label=f"profile, N={n}")
        axes[1].plot(x, [r["bootstrap_coverage"] for r in part], "o-",
                     label=f"N={n}")
        axes[2].plot(x, [r["profile_full_domain_rate"] for r in part], "o-",
                     label=f"N={n}")

    axes[0].set_title("Wald versus profile", fontsize=9)
    axes[1].set_title("parametric bootstrap", fontsize=9)
    axes[2].set_title("uninformative profile sets", fontsize=9)
    axes[0].set_ylabel("empirical 95% coverage")
    axes[1].set_ylabel("empirical 95% coverage")
    axes[2].set_ylabel("full-domain set rate")
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlabel("nuisance-adjusted information")
        panel_label(ax, f"({chr(ord('a') + i)})")
    axes[0].axhline(0.95, color="0.4", ls=":", lw=1)
    axes[1].axhline(0.95, color="0.4", ls=":", lw=1)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    save_fig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    if args.quick:
        data = run(reps=12, bootstrap_outer=3, bootstrap_draws=9)
    else:
        data = run(reps=180, bootstrap_outer=40, bootstrap_draws=79)
    figure(data, FIGS_DIR / "exp15_variance_interval_calibration.png")
    metrics = dict(data["metrics"])
    metrics["runtime_sec"] = time.time() - t0
    metrics["command"] = "python experiments/15_variance_interval_calibration.py"
    save_json(RESULTS_DIR / "exp15_variance_interval_calibration.json", metrics)
    save_csv(RESULTS_DIR / "exp15_variance_interval_calibration.csv", data["summary_rows"])
    save_csv(RESULTS_DIR / "exp15_variance_interval_calibration_replicates.csv",
             data["replicate_rows"])

    paper_figs = Path(__file__).resolve().parents[1] / "paper" / "figs"
    paper_figs.mkdir(exist_ok=True)
    (paper_figs / "exp15_variance_interval_calibration.png").write_bytes(
        (FIGS_DIR / "exp15_variance_interval_calibration.png").read_bytes()
    )
    print(
        "exp15 coverage ranges",
        min(r["clipped_oracle_wald_coverage"] for r in data["summary_rows"]),
        max(r["clipped_oracle_wald_coverage"] for r in data["summary_rows"]),
        min(r["profile_coverage"] for r in data["summary_rows"]),
        max(r["profile_coverage"] for r in data["summary_rows"]),
    )


if __name__ == "__main__":
    main()

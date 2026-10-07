"""Experiment 14 — equal-budget recovery from shifted populations.

Three designs receive the same total number of observations at every time:
one centered population, one shifted population, or three populations whose
per-time budget is split equally.  Initial means are known design inputs while
the common initial variance is estimated in every arm.  Evaluation uses an
independent shifted population and both population and sampled-test losses.
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

from wfmm.identifiability import gaussian_w2_1d, known_design_mean_information
from wfmm.io_util import save_csv, save_json
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


A_TRUE = 1.0
K_TRUE = 1.5
Q_TRUE = 1.0 / 3.0
S0_TRUE = 1.5
TEST_MU0 = 1.2
TIMES = np.array([0.25, 0.50, 1.00])
TEST_TIMES = np.array([0.25, 0.50, 1.00, 1.50])
BUDGETS = (300, 1200, 4800)
DESIGNS = {
    "centered": np.array([0.0]),
    "shifted_only": np.array([0.8]),
    "mixed": np.array([-0.8, 0.0, 0.8]),
}

# Compact limits are numerical devices, not scientific parameter boundaries.
A_BOUNDS = (0.05, 2.50)
K_BOUNDS = (0.10, 3.00)
Q_BOUNDS = (0.05, 2.50)
S0_BOUNDS = (0.10, 3.00)
LR95 = 3.841458820694124


def variance_path(k: float, q: float, s0: float, times: np.ndarray) -> np.ndarray:
    return q + (s0 - q) * np.exp(-2.0 * k * times)


def simulate_stats(
    rng: np.random.Generator,
    mu0s: np.ndarray,
    times: np.ndarray,
    counts: np.ndarray,
) -> dict:
    """Generate exact sufficient statistics without storing observations."""
    variances = variance_path(K_TRUE, Q_TRUE, S0_TRUE, times)
    means = mu0s[:, None] * np.exp(-A_TRUE * times[None, :])
    n = np.broadcast_to(counts[None, :], means.shape).astype(int)
    xbar = rng.normal(means, np.sqrt(variances[None, :] / n))
    sse = variances[None, :] * rng.chisquare(n - 1)
    return {"mu0": mu0s, "times": times, "n": n, "xbar": xbar, "sse": sse}


def gaussian_nll_stats(
    stats: dict, a: float, k: float, q: float, s0: float
) -> float:
    if not (a > 0.0 and k > 0.0 and q > 0.0 and s0 > 0.0):
        return float("inf")
    v = variance_path(k, q, s0, stats["times"])
    if np.any(v <= 0.0):
        return float("inf")
    mean = stats["mu0"][:, None] * np.exp(-a * stats["times"][None, :])
    rss = stats["sse"] + stats["n"] * (stats["xbar"] - mean) ** 2
    return float(0.5 * np.sum(stats["n"] * np.log(2.0 * np.pi * v) + rss / v))


def variance_nll_stats(stats: dict, k: float, q: float, s0: float) -> float:
    # At the centered mean, the mean likelihood contains no decomposition parameter.
    return gaussian_nll_stats(stats, min(0.5 * k, A_TRUE), k, q, s0)


def fit_centered(stats: dict) -> tuple[np.ndarray, float]:
    def objective(x: np.ndarray) -> float:
        return variance_nll_stats(stats, x[0], x[1], x[2])

    fit = minimize(
        objective,
        x0=np.array([K_TRUE, Q_TRUE, S0_TRUE]),
        method="L-BFGS-B",
        bounds=[K_BOUNDS, Q_BOUNDS, S0_BOUNDS],
    )
    if not fit.success:
        raise RuntimeError(f"centered variance fit failed: {fit.message}")
    return np.asarray(fit.x), float(fit.fun)


def fit_joint(stats: dict) -> tuple[np.ndarray, float]:
    scale = float(np.sum(stats["n"]))

    def objective(x: np.ndarray) -> float:
        violation = max(float(x[0] - x[1]), 0.0)
        return gaussian_nll_stats(stats, *x) / scale + 1e4 * violation ** 2

    fit = minimize(
        objective,
        x0=np.array([A_TRUE, K_TRUE, Q_TRUE, S0_TRUE]),
        method="L-BFGS-B",
        bounds=[A_BOUNDS, K_BOUNDS, Q_BOUNDS, S0_BOUNDS],
        options={"ftol": 1e-12, "maxiter": 300},
    )
    if not fit.success or fit.x[1] + 1e-5 < fit.x[0]:
        fit = minimize(
            objective,
            x0=np.array([A_TRUE, K_TRUE, Q_TRUE, S0_TRUE]),
            method="Powell",
            bounds=[A_BOUNDS, K_BOUNDS, Q_BOUNDS, S0_BOUNDS],
            options={"ftol": 1e-10, "maxiter": 1000},
        )
    if not fit.success or fit.x[1] + 1e-4 < fit.x[0]:
        raise RuntimeError(f"joint fit failed: {fit.message}")
    params = np.asarray(fit.x)
    params[1] = max(params[1], params[0])
    return params, gaussian_nll_stats(stats, *params)


def profile_a(stats: dict, minimum: float, grid: np.ndarray) -> np.ndarray:
    """Exact constrained likelihood profile over a, including k >= a."""
    out = np.empty(grid.size)
    previous = np.array([K_TRUE, Q_TRUE, S0_TRUE])
    scale = float(np.sum(stats["n"]))
    for i, a in enumerate(grid):
        lower_k = max(K_BOUNDS[0], float(a))
        if lower_k > K_BOUNDS[1]:
            out[i] = np.inf
            continue

        def objective(x: np.ndarray) -> float:
            return gaussian_nll_stats(stats, float(a), x[0], x[1], x[2]) / scale

        previous[0] = np.clip(previous[0], lower_k, K_BOUNDS[1])
        fit = minimize(
            objective,
            x0=previous,
            method="L-BFGS-B",
            bounds=[(lower_k, K_BOUNDS[1]), Q_BOUNDS, S0_BOUNDS],
            options={"ftol": 1e-9, "maxiter": 200},
        )
        if not fit.success:
            raise RuntimeError(f"a-profile fit failed: {fit.message}")
        previous = np.asarray(fit.x)
        profiled = gaussian_nll_stats(
            stats, float(a), previous[0], previous[1], previous[2]
        )
        out[i] = 2.0 * (profiled - minimum)
    return np.maximum(out, 0.0)


def set_components(grid: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [[float(grid[g[0]]), float(grid[g[-1]])] for g in np.split(indices, split)]


def evaluate_point(params: np.ndarray, test_stats: dict) -> dict:
    a, k, q, s0 = params
    mean = TEST_MU0 * np.exp(-a * TEST_TIMES)
    var = variance_path(k, q, s0, TEST_TIMES)
    truth_mean = TEST_MU0 * np.exp(-A_TRUE * TEST_TIMES)
    truth_var = variance_path(K_TRUE, Q_TRUE, S0_TRUE, TEST_TIMES)
    w2 = gaussian_w2_1d(mean, var, truth_mean, truth_var)
    nll = gaussian_nll_stats(test_stats, a, k, q, s0) / np.sum(test_stats["n"])
    return {
        "mean_abs_error": float(np.mean(np.abs(mean - truth_mean))),
        "variance_abs_error": float(np.mean(np.abs(var - truth_var))),
        "mean_w2": float(np.mean(w2)),
        "final_w2": float(w2[-1]),
        "sampled_test_nll": float(nll),
    }


def evaluate_centered_ridge(
    variance_params: np.ndarray, test_stats: dict
) -> dict:
    k, q, s0 = variance_params
    # This grid represents the compact numerical display of the genuine ridge
    # 0 < a <= k. It is never tuned using test outcomes.
    upper = min(float(k), A_BOUNDS[1])
    a_grid = np.linspace(A_BOUNDS[0], upper, 121)
    evaluations = [
        evaluate_point(np.array([a, k, q, s0]), test_stats) for a in a_grid
    ]
    return {
        "ridge_a_display_min": float(a_grid[0]),
        "ridge_a_display_max": float(a_grid[-1]),
        "ridge_is_full_genuine_domain": bool(k <= A_BOUNDS[1]),
        "mean_w2_min": min(e["mean_w2"] for e in evaluations),
        "mean_w2_max": max(e["mean_w2"] for e in evaluations),
        "final_w2_min": min(e["final_w2"] for e in evaluations),
        "final_w2_max": max(e["final_w2"] for e in evaluations),
        "sampled_test_nll_min": min(e["sampled_test_nll"] for e in evaluations),
        "sampled_test_nll_max": max(e["sampled_test_nll"] for e in evaluations),
    }


def _summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def run(seed: int = 14, reps: int = 300, test_n: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict] = []
    profile_grid = np.linspace(A_BOUNDS[0], A_BOUNDS[1], 41)
    test_sets = [
        simulate_stats(
            rng, np.array([TEST_MU0]), TEST_TIMES,
            np.full(TEST_TIMES.size, test_n, dtype=int),
        )
        for _ in range(reps)
    ]

    for budget in BUDGETS:
        for design, mu0s in DESIGNS.items():
            if budget % mu0s.size:
                raise ValueError("per-time budget must divide equally across populations")
            per_population = budget // mu0s.size
            counts = np.full(TIMES.size, per_population, dtype=int)
            for rep in range(reps):
                train = simulate_stats(rng, mu0s, TIMES, counts)
                test = test_sets[rep]
                base = {
                    "design": design,
                    "budget_per_time": budget,
                    "samples_per_population_time": per_population,
                    "replicate": rep,
                    "total_training_samples": budget * TIMES.size,
                }
                if design == "centered":
                    variance_params, nll = fit_centered(train)
                    ridge = evaluate_centered_ridge(variance_params, test)
                    k, q, s0 = variance_params
                    replicate_rows.append({
                        **base,
                        "a_hat": float("nan"),
                        "k_hat": float(k),
                        "q_hat": float(q),
                        "s0_hat": float(s0),
                        "b_hat": float("nan"),
                        "beta_hat": float(k * q),
                        "genuine_b0_contact": float("nan"),
                        "artificial_bound_contact": float(
                            min(abs(k - K_BOUNDS[0]), abs(k - K_BOUNDS[1]),
                                abs(q - Q_BOUNDS[0]), abs(q - Q_BOUNDS[1]),
                                abs(s0 - S0_BOUNDS[0]), abs(s0 - S0_BOUNDS[1])) < 1e-4
                        ),
                        "profile_components": f"(0,{k:.8g}]",
                        "profile_full_domain": 1.0,
                        "profile_hull_width": float(k),
                        "train_nll_per_sample": nll / (budget * TIMES.size),
                        **ridge,
                    })
                else:
                    params, nll = fit_joint(train)
                    a, k, q, s0 = params
                    profile = profile_a(train, nll, profile_grid)
                    truth_lr = float(profile_a(train, nll, np.array([A_TRUE]))[0])
                    selected = profile <= LR95
                    components = set_components(profile_grid, selected)
                    evaluation = evaluate_point(params, test)
                    distances = (
                        abs(a - A_BOUNDS[0]), abs(a - A_BOUNDS[1]),
                        abs(k - K_BOUNDS[0]), abs(k - K_BOUNDS[1]),
                        abs(q - Q_BOUNDS[0]), abs(q - Q_BOUNDS[1]),
                        abs(s0 - S0_BOUNDS[0]), abs(s0 - S0_BOUNDS[1]),
                    )
                    replicate_rows.append({
                        **base,
                        "a_hat": float(a),
                        "k_hat": float(k),
                        "q_hat": float(q),
                        "s0_hat": float(s0),
                        "b_hat": float(k - a),
                        "beta_hat": float(k * q),
                        "genuine_b0_contact": float(abs(k - a) < 1e-4),
                        "artificial_bound_contact": float(min(distances) < 1e-4),
                        "profile_components": repr(components),
                        "profile_full_domain": float(np.all(selected)),
                        "profile_hull_width": (
                            float(profile_grid[selected][-1] - profile_grid[selected][0])
                            if np.any(selected) else float("nan")
                        ),
                        "profile_truth_lr": truth_lr,
                        "profile_covers_a": float(truth_lr <= LR95),
                        "train_nll_per_sample": nll / (budget * TIMES.size),
                        **evaluation,
                    })

    summary_rows: list[dict] = []
    for budget in BUDGETS:
        for design in DESIGNS:
            part = [
                r for r in replicate_rows
                if r["budget_per_time"] == budget and r["design"] == design
            ]
            row = {
                "design": design,
                "budget_per_time": budget,
                "samples_per_population_time": part[0]["samples_per_population_time"],
                "replications": reps,
                "total_training_samples": part[0]["total_training_samples"],
                "k_rmse": float(np.sqrt(np.mean([(r["k_hat"] - K_TRUE) ** 2 for r in part]))),
                "q_rmse": float(np.sqrt(np.mean([(r["q_hat"] - Q_TRUE) ** 2 for r in part]))),
                "s0_rmse": float(np.sqrt(np.mean([(r["s0_hat"] - S0_TRUE) ** 2 for r in part]))),
                "artificial_bound_rate": float(np.mean([r["artificial_bound_contact"] for r in part])),
            }
            if design == "centered":
                for key in (
                    "mean_w2_min", "mean_w2_max", "final_w2_min", "final_w2_max",
                    "sampled_test_nll_min", "sampled_test_nll_max",
                ):
                    row[key] = float(np.mean([r[key] for r in part]))
                row.update({
                    "a_rmse": float("nan"),
                    "profile_coverage": 1.0,
                    "profile_full_domain_rate": 1.0,
                    "genuine_b0_contact_rate": float("nan"),
                })
            else:
                for key in (
                    "mean_w2", "final_w2", "sampled_test_nll",
                    "mean_abs_error", "variance_abs_error",
                ):
                    value, se = _summary(np.array([r[key] for r in part]))
                    row[key] = value
                    row[f"{key}_se"] = se
                row.update({
                    "a_rmse": float(np.sqrt(np.mean([(r["a_hat"] - A_TRUE) ** 2 for r in part]))),
                    "profile_coverage": float(np.mean([r["profile_covers_a"] for r in part])),
                    "profile_full_domain_rate": float(np.mean([r["profile_full_domain"] for r in part])),
                    "genuine_b0_contact_rate": float(np.mean([r["genuine_b0_contact"] for r in part])),
                    "mean_profile_hull_width": float(np.nanmean([r["profile_hull_width"] for r in part])),
                })
            summary_rows.append(row)

    common_variances = variance_path(K_TRUE, Q_TRUE, S0_TRUE, TIMES)
    shifted_information = known_design_mean_information(
        A_TRUE, DESIGNS["shifted_only"], np.array([1.0]), TIMES, common_variances
    )
    mixed_information = known_design_mean_information(
        A_TRUE, DESIGNS["mixed"], np.full(3, 1.0 / 3.0), TIMES, common_variances
    )
    metrics = {
        "claim": "equal_budget_shifted_training_recovery",
        "seed": seed,
        "replications": reps,
        "training_times": TIMES,
        "test_times": TEST_TIMES,
        "test_samples_per_time": test_n,
        "same_independent_test_sample_used_for_all_arms_and_budgets": True,
        "budgets_are_total_samples_per_time": True,
        "initial_means_are_known_design_inputs": True,
        "initial_variance_is_unknown_common_nuisance": True,
        "true_parameters": {
            "a": A_TRUE, "k": K_TRUE, "b": K_TRUE - A_TRUE,
            "q": Q_TRUE, "beta": K_TRUE * Q_TRUE, "sigma0": S0_TRUE,
        },
        "constraints": {
            "scientific": "a>0, k>=a, q>0; b=k-a, beta=kq",
            "compact_numerical": {
                "a": A_BOUNDS, "k": K_BOUNDS, "q": Q_BOUNDS, "sigma0": S0_BOUNDS,
            },
        },
        "mean_block_information_ratio_mixed_to_shifted_only": (
            mixed_information / shifted_information
        ),
        "centered_reporting": (
            "Set-valued ridge and forecast/NLL ranges; no point decomposition and "
            "no test-data selection."
        ),
        "summary_rows": summary_rows,
    }
    return {"replicate_rows": replicate_rows, "summary_rows": summary_rows, "metrics": metrics}


def figure(data: dict, path: Path) -> None:
    rows = data["summary_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.2))
    colors = {"centered": "C3", "shifted_only": "C0", "mixed": "C2"}
    labels = {"centered": "centered ridge", "shifted_only": "shifted only", "mixed": "mixed shifts"}

    for design in DESIGNS:
        part = [r for r in rows if r["design"] == design]
        x = np.array([r["budget_per_time"] for r in part])
        if design == "centered":
            low = np.array([r["mean_w2_min"] for r in part])
            high = np.array([r["mean_w2_max"] for r in part])
            axes[0].fill_between(x, low, high, alpha=0.25, color=colors[design],
                                 label=labels[design])
            nll_low = np.array([r["sampled_test_nll_min"] for r in part])
            nll_high = np.array([r["sampled_test_nll_max"] for r in part])
            axes[2].fill_between(x, nll_low, nll_high, alpha=0.25,
                                 color=colors[design], label=labels[design])
        else:
            axes[0].errorbar(
                x, [r["mean_w2"] for r in part],
                yerr=[1.96 * r["mean_w2_se"] for r in part],
                marker="o", color=colors[design], label=labels[design],
            )
            axes[1].plot(x, [r["a_rmse"] for r in part], "o-",
                         color=colors[design], label=labels[design])
            axes[2].errorbar(
                x, [r["sampled_test_nll"] for r in part],
                yerr=[1.96 * r["sampled_test_nll_se"] for r in part],
                marker="o", color=colors[design], label=labels[design],
            )

    titles = ("exact intervention error", "confinement recovery", "sampled test loss")
    ylabels = (r"mean Gaussian $W_2$", r"RMSE of $\hat a$", "held-out NLL")
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlabel("total training samples per time")
        ax.set_title(titles[i], fontsize=9)
        ax.set_ylabel(ylabels[i])
        panel_label(ax, f"({chr(ord('a') + i)})")
    axes[0].legend(frameon=False, fontsize=7)
    save_fig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run(reps=12 if args.quick else 180, test_n=1000 if args.quick else 4000)
    figure(data, FIGS_DIR / "exp14_shifted_training_recovery.png")
    metrics = dict(data["metrics"])
    metrics["runtime_sec"] = time.time() - t0
    metrics["command"] = "python experiments/14_shifted_training_recovery.py"
    save_json(RESULTS_DIR / "exp14_shifted_training_recovery.json", metrics)
    save_csv(RESULTS_DIR / "exp14_shifted_training_recovery.csv", data["summary_rows"])
    save_csv(RESULTS_DIR / "exp14_shifted_training_recovery_replicates.csv",
             data["replicate_rows"])

    paper_figs = Path(__file__).resolve().parents[1] / "paper" / "figs"
    paper_figs.mkdir(exist_ok=True)
    (paper_figs / "exp14_shifted_training_recovery.png").write_bytes(
        (FIGS_DIR / "exp14_shifted_training_recovery.png").read_bytes()
    )
    print("exp14 information ratio", metrics["mean_block_information_ratio_mixed_to_shifted_only"])
    for row in data["summary_rows"]:
        if row["budget_per_time"] == max(BUDGETS):
            value = (
                row["mean_w2"] if "mean_w2" in row
                else (row["mean_w2_min"], row["mean_w2_max"])
            )
            print(row["design"], "mean W2", value)


if __name__ == "__main__":
    main()

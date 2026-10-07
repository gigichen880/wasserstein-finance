"""Experiment 12 — exact ambiguity and shifted-population intervention.

All snapshots are sampled independently from exact Gaussian marginal laws.
Models on (a+h,b-h,beta) agree on centered training populations and disagree
after a physical mean-shift intervention under the same fixed energy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import gaussian_moments, gaussian_nll, ridge_model
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


def run(seed: int = 12, n: int = 400, reps: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    base = Model(a=1.0, b=0.5, beta=0.5)
    hs = np.array([-0.75, -0.50, -0.25, 0.0, 0.20, 0.40])
    times = np.array([0.25, 0.50, 1.00])
    var0 = 1.5
    shift = 0.8

    models = [ridge_model(base, h) for h in hs]
    centered = [gaussian_moments(m, 0.0, var0, times) for m in models]
    shifted = [gaussian_moments(m, shift, var0, times) for m in models]
    true_center_mean, true_vars = centered[hs.tolist().index(0.0)]
    true_shift_mean, _ = shifted[hs.tolist().index(0.0)]

    max_population_centered_mean_diff = float(
        max(np.max(np.abs(mu - true_center_mean)) for mu, _ in centered)
    )
    max_population_centered_var_diff = float(
        max(np.max(np.abs(var - true_vars)) for _, var in centered)
    )

    train_losses = np.empty((reps, len(hs)))
    holdout_losses = np.empty((reps, len(hs)))
    for r in range(reps):
        centered_samples = [
            rng.normal(true_center_mean[j], np.sqrt(true_vars[j]), size=n)
            for j in range(times.size)
        ]
        shifted_samples = [
            rng.normal(true_shift_mean[j], np.sqrt(true_vars[j]), size=n)
            for j in range(times.size)
        ]
        for i, ((candidate_center_mean, candidate_var),
                (candidate_shift_mean, _)) in enumerate(zip(centered, shifted)):
            train_losses[r, i] = np.mean([
                gaussian_nll(centered_samples[j], candidate_center_mean[j], candidate_var[j])
                for j in range(times.size)
            ])
            holdout_losses[r, i] = np.mean([
                gaussian_nll(shifted_samples[j], candidate_shift_mean[j], candidate_var[j])
                for j in range(times.size)
            ])

    base_idx = hs.tolist().index(0.0)
    train_excess = train_losses - train_losses[:, [base_idx]]
    holdout_excess = holdout_losses - holdout_losses[:, [base_idx]]
    train_mean = train_excess.mean(axis=0)
    train_se = train_excess.std(axis=0, ddof=1) / np.sqrt(reps)
    holdout_mean = holdout_excess.mean(axis=0)
    holdout_se = holdout_excess.std(axis=0, ddof=1) / np.sqrt(reps)

    # Expected per-observation excess NLL is the Gaussian KL because candidate
    # and truth have the same variance at every time.
    population_holdout_excess = np.array([
        np.mean(0.5 * (candidate_mean - true_shift_mean) ** 2 / true_vars)
        for candidate_mean, _ in shifted
    ])
    final_w2 = np.array([
        abs(candidate_mean[-1] - true_shift_mean[-1])
        for candidate_mean, _ in shifted
    ])

    rows = []
    for i, h in enumerate(hs):
        rows.append(dict(
            h=float(h),
            a=float(models[i].a),
            b=float(models[i].b),
            beta=float(models[i].beta),
            centered_excess_nll_mean=float(train_mean[i]),
            centered_excess_nll_se=float(train_se[i]),
            shifted_excess_nll_mean=float(holdout_mean[i]),
            shifted_excess_nll_se=float(holdout_se[i]),
            shifted_population_excess_nll=float(population_holdout_excess[i]),
            shifted_final_w2=float(final_w2[i]),
        ))

    metrics = dict(
        claim="centered_ambiguity_shifted_intervention",
        seed=seed,
        n_per_snapshot=n,
        replications=reps,
        times=times,
        shift=shift,
        var0=var0,
        base_model=base.as_dict(),
        h_grid=hs,
        max_population_centered_mean_diff=max_population_centered_mean_diff,
        max_population_centered_var_diff=max_population_centered_var_diff,
        max_sample_centered_excess_loss=float(np.max(np.abs(train_excess))),
        max_shifted_final_w2=float(np.max(final_w2)),
        rows=rows,
        supports_ambiguity=(
            max_population_centered_mean_diff < 1e-14
            and max_population_centered_var_diff < 1e-14
            and np.max(np.abs(train_excess)) < 1e-12
        ),
        supports_intervention_disagreement=float(np.max(final_w2)) > 0.1,
        notes=(
            "Centered training snapshots are independent across times and replications. "
            "The shifted holdout is a physical new initial population under the same energy. "
            "It reveals this ridge but does not prove global nonlinear identifiability."
        ),
    )
    return dict(rows=rows, metrics=metrics)


def figure(data: dict, path: Path) -> None:
    rows = data["rows"]
    h = np.array([r["h"] for r in rows])
    train = np.array([r["centered_excess_nll_mean"] for r in rows])
    train_se = np.array([r["centered_excess_nll_se"] for r in rows])
    holdout = np.array([r["shifted_excess_nll_mean"] for r in rows])
    holdout_se = np.array([r["shifted_excess_nll_se"] for r in rows])
    holdout_truth = np.array([r["shifted_population_excess_nll"] for r in rows])
    w2 = np.array([r["shifted_final_w2"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    ax = axes[0]
    ax.errorbar(h, train, yerr=1.96 * train_se, fmt="o", color="C0", capsize=2)
    ax.axhline(0.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(r"ridge shift $h$")
    ax.set_ylabel("centered excess NLL")
    ax.set_title("independent training snapshots", fontsize=9)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.errorbar(h, holdout, yerr=1.96 * holdout_se, fmt="o", color="C1",
                capsize=2, label="Monte Carlo")
    ax.plot(h, holdout_truth, color="C0", label="population KL")
    ax.set_xlabel(r"ridge shift $h$")
    ax.set_ylabel("shifted excess NLL")
    ax.set_title("physical shifted holdout", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "(b)")

    ax = axes[2]
    ax.plot(h, w2, "o-", color="C2")
    ax.set_xlabel(r"ridge shift $h$")
    ax.set_ylabel(r"$W_2$ disagreement at $T=1$")
    ax.set_title("intervention forecast", fontsize=9)
    panel_label(ax, "(c)")
    save_fig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run(n=200 if args.quick else 400, reps=100 if args.quick else 500)
    figure(data, FIGS_DIR / "exp12_ambiguity_intervention.png")
    metrics = dict(data["metrics"])
    metrics["runtime_sec"] = time.time() - t0
    metrics["command"] = "python experiments/12_ambiguity_intervention.py"
    save_json(RESULTS_DIR / "exp12_ambiguity_intervention.json", metrics)
    save_csv(RESULTS_DIR / "exp12_ambiguity_intervention.csv", data["rows"])

    paper_figs = Path(__file__).resolve().parents[1] / "paper" / "figs"
    paper_figs.mkdir(exist_ok=True)
    (paper_figs / "exp12_ambiguity_intervention.png").write_bytes(
        (FIGS_DIR / "exp12_ambiguity_intervention.png").read_bytes()
    )
    print(
        "exp12 centered max excess",
        metrics["max_sample_centered_excess_loss"],
        "shifted max W2",
        metrics["max_shifted_final_w2"],
    )


if __name__ == "__main__":
    main()

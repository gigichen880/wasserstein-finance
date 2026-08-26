"""Experiment G — next-distribution forecasting vs baselines."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.diagnostics import forecast_w2
from wfmm.estimation import fit_moment_laws
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.particles import bimodal_samples, simulate_snapshots, step_mckean_vlasov
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


def run(seed: int = 0, n: int = 700, n_steps: int = 16, dt: float = 0.05) -> dict:
    model = Model()
    rng = np.random.default_rng(seed)
    X0 = bimodal_samples(n, rng, left=-1.2, right=2.6)
    snaps = simulate_snapshots(
        X0, n_steps, dt, rng, lambda X: step_mckean_vlasov(X, model, dt, rng, substeps=4)
    )
    mu = np.array([s.mean() for s in snaps])
    var = np.array([s.var() for s in snaps])
    split = 9
    fit = fit_moment_laws(mu[:split], var[:split], dt, mean_floor=0.05)
    rows = []
    for i in range(split - 1, n_steps):
        fc = forecast_w2(
            snaps[i], snaps[i + 1], model, dt, rng,
            a=fit.a if np.isfinite(fit.a) else model.a,
            a_plus_b=fit.a_plus_b,
            sigma2_inf=fit.sigma2_inf,
            jko_m=80,
        )
        fc["window"] = i
        fc["t"] = (i + 1) * dt
        rows.append(fc)

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    metrics = dict(
        claim="model_validation_synthetic",
        seed=seed, n=n, n_steps=n_steps, dt=dt, train_end_index=split,
        fitted_a=fit.a, fitted_a_plus_b=fit.a_plus_b, fitted_sigma2_inf=fit.sigma2_inf,
        mean_w2_persistence=_mean("w2_persistence"),
        mean_w2_gaussian=_mean("w2_gaussian_moments"),
        mean_w2_no_interaction=_mean("w2_no_interaction"),
        mean_w2_jko=_mean("w2_jko"),
        rows=rows,
        notes=(
            "Parameters from training windows only. Persistence is the no-skill baseline. "
            "JKO uses fitted (a,b) with tau=dt and M=80."
        ),
    )
    gauss_beats_persist = metrics["mean_w2_gaussian"] < metrics["mean_w2_persistence"]
    jko_beats_persist = metrics["mean_w2_jko"] < metrics["mean_w2_persistence"]
    metrics["supports_claim"] = jko_beats_persist
    metrics["gaussian_beats_persistence"] = gauss_beats_persist
    return dict(rows=rows, metrics=metrics)


def figure(data, path) -> None:
    rows = data["rows"]
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(t, [r["w2_persistence"] for r in rows], lw=1.4, label="persistence")
    ax.plot(t, [r["w2_gaussian_moments"] for r in rows], lw=1.4, label="Gaussian moment model")
    ax.plot(t, [r["w2_no_interaction"] for r in rows], lw=1.4, label="no interaction (b=0)")
    ax.plot(t, [r["w2_jko"] for r in rows], lw=1.4, label="JKO one step (fitted)")
    ax.set_xlabel("window end time")
    ax.set_ylabel(r"$W_2(\widehat D_{t+\Delta t}, D_{t+\Delta t})$")
    ax.set_title("one-step distributional forecast error (test windows)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "(a)")
    save_fig(fig, path)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp07_distribution_forecast.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/07_distribution_forecast.py"
    save_json(RESULTS_DIR / "exp07_distribution_forecast.json", meta)
    save_csv(RESULTS_DIR / "exp07_distribution_forecast.csv", data["rows"])
    print("exp07 persist", meta["mean_w2_persistence"], "gauss", meta["mean_w2_gaussian"],
          "jko", meta["mean_w2_jko"])


if __name__ == "__main__":
    main()

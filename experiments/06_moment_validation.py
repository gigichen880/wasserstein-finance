"""Experiment F — moment restrictions, train then test (first empirical falsification)."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.diagnostics import moment_step_errors
from wfmm.estimation import fit_moment_laws, predict_next_moments
from wfmm.io_util import save_json
from wfmm.model import Model
from wfmm.particles import bimodal_samples, simulate_snapshots, step_mckean_vlasov
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


def _bootstrap_mae(y, yhat, rng, n_boot=200):
    err = np.abs(np.asarray(yhat) - np.asarray(y))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, err.size, size=err.size)
        stats.append(float(err[idx].mean()))
    lo, hi = np.quantile(stats, [0.05, 0.95])
    return float(np.mean(stats)), float(lo), float(hi)


def run(seed: int = 0, n: int = 1200, n_steps: int = 24, dt: float = 0.05) -> dict:
    model = Model()
    rng = np.random.default_rng(seed)
    X0 = bimodal_samples(n, rng, left=-1.2, right=2.8)
    snaps = simulate_snapshots(
        X0, n_steps, dt, rng, lambda X: step_mckean_vlasov(X, model, dt, rng, substeps=5)
    )
    mu = np.array([s.mean() for s in snaps])
    var = np.array([s.var() for s in snaps])
    t = np.arange(len(snaps)) * dt
    split = 13  # train on first 12 steps (indices 0..12 inclusive => 12 pairs)
    fit = fit_moment_laws(mu[:split], var[:split], dt, mean_floor=0.05)
    train_err = moment_step_errors(
        mu[:split], var[:split], dt, fit.a, fit.a_plus_b, fit.sigma2_inf
    )
    test_err = moment_step_errors(
        mu[split - 1 :], var[split - 1 :], dt, fit.a, fit.a_plus_b, fit.sigma2_inf
    )
    # oracle (true params) on test
    oracle = moment_step_errors(
        mu[split - 1 :], var[split - 1 :], dt, model.a, model.a + model.b, model.sigma2_inf
    )
    mu_hat_test, var_hat_test = [], []
    for m, s in zip(mu[split - 1:-1], var[split - 1:-1]):
        mh, sh = predict_next_moments(m, s, dt, fit.a, fit.a_plus_b, fit.sigma2_inf)
        mu_hat_test.append(mh)
        var_hat_test.append(sh)
    mu_hat_test = np.array(mu_hat_test)
    var_hat_test = np.array(var_hat_test)
    mae_mu, lo_mu, hi_mu = _bootstrap_mae(mu[split:], mu_hat_test, rng)
    mae_s, lo_s, hi_s = _bootstrap_mae(var[split:], var_hat_test, rng)

    identifiable = np.isfinite(fit.a) and fit.n_mean_pairs >= 3
    metrics = dict(
        claim="model_validation_synthetic",
        seed=seed, n=n, n_steps=n_steps, dt=dt, train_end_index=split,
        true=model.as_dict(),
        fitted_a=fit.a, fitted_a_plus_b=fit.a_plus_b, fitted_b=fit.b,
        fitted_sigma2_inf=fit.sigma2_inf,
        n_mean_pairs=fit.n_mean_pairs, mean_fit_r2=fit.mean_r2, var_fit_r2=fit.var_r2,
        train=train_err, test=test_err, oracle_test=oracle,
        test_mean_mae=mae_mu, test_mean_mae_ci=(lo_mu, hi_mu),
        test_var_mae=mae_s, test_var_mae_ci=(lo_s, hi_s),
        scale_not_identified_by_cosine=True,
        supports_claim=identifiable and test_err["mean_mae"] < 0.15 and test_err["var_mae"] < 0.15,
        notes=(
            "Parameters fitted only on the training segment. Cosine alignment is not "
            "used here; (a,b,beta) ~ c(a,b,beta) is identified via moment timescales."
        ),
    )
    return dict(
        t=t, mu=mu, var=var, split=split, mu_hat_test=mu_hat_test, var_hat_test=var_hat_test,
        metrics=metrics, model=model, fit=fit,
    )


def figure(data, path) -> None:
    t, split, model = data["t"], data["split"], data["model"]
    t_split = t[split]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.plot(t, data["mu"], lw=1.5, label="empirical mean")
    ax.plot(t, model.mean_law(data["mu"][0], t), ":", lw=1.6, label="true law")
    ax.axvline(t_split, color="0.5", ls="--", lw=1, label="train | test")
    ax.set_xlabel("time")
    ax.set_ylabel("mean")
    ax.set_title("mean restriction, out of sample", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.plot(t, data["var"], lw=1.5, label="empirical variance")
    ax.plot(t, model.var_law(data["var"][0], t), ":", lw=1.6, label="true law")
    ax.axhline(model.sigma2_inf, color="k", ls="--", lw=1, label=r"$\beta/(a+b)$")
    ax.axvline(t_split, color="0.5", ls="--", lw=1)
    ax.set_xlabel("time")
    ax.set_ylabel("variance")
    ax.set_title("variance restriction, out of sample", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, path, bottom_pad=0.02)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp06_moment_validation.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/06_moment_validation.py"
    save_json(RESULTS_DIR / "exp06_moment_validation.json", meta)
    m = meta
    print("exp06 fitted a", m["fitted_a"], "true", 1.0, "test mean MAE", m["test"]["mean_mae"])


if __name__ == "__main__":
    main()

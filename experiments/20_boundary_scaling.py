"""Experiment 20 — approach to the non-identifiable boundary.

Sweep mu0 -> 0 at fixed variance excitation, and eps = var0-q -> 0 at fixed
mean excitation. Fit lambda_min ~ C * signal^p. Independent Gaussian snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import (
    information_spectrum,
    mean_efficient_information,
    plugin_three_equal_spacing,
    profiled_fisher_abb,
    snapshot_mean_variance,
    variance_efficient_information,
)
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig

MODEL = Model(a=1.0, b=0.5, beta=0.5)
A = MODEL.a
K = MODEL.a + MODEL.b
Q = MODEL.sigma2_inf
TIMES = np.array([0.0, 0.5, 1.0])
DELTA = 0.5
N_PER = 400
SEED = 20
N_REP = 400
VAR0_MEAN_SWEEP = 1.5
MU0_VAR_SWEEP = 0.8


def _fit_log_slope(x, y, x_max):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & (x <= x_max)
    if int(m.sum()) < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    p, logc = np.polyfit(lx, ly, 1)
    return float(p), float(np.exp(logc))


def _rmse_a(mu0, var0, rng, n_rep):
    mu, var = snapshot_mean_variance(A, K, Q, mu0, var0, TIMES)
    hats = []
    n_valid = 0
    for _ in range(n_rep):
        mus = np.empty(3)
        sigs = np.empty(3)
        for j in range(3):
            x = rng.normal(mu[j], np.sqrt(max(var[j], 1e-12)), size=N_PER)
            mus[j] = x.mean()
            sigs[j] = x.var(ddof=1)
        hat = plugin_three_equal_spacing(mus, sigs, DELTA)
        if hat is None:
            continue
        n_valid += 1
        if hat["admissible"]:
            hats.append(hat["a"])
    hats = np.asarray(hats, dtype=float)
    if hats.size < 5:
        return float("nan"), n_valid / n_rep, float("nan")
    return float(np.sqrt(np.mean((hats - A) ** 2))), n_valid / n_rep, float(np.var(hats, ddof=1))


def main():
    ensure_output_dirs()
    rng = np.random.default_rng(SEED)
    signals = np.array([0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6, 0.9])

    mean_rows = []
    for mu0 in signals:
        spec = information_spectrum(
            profiled_fisher_abb(MODEL, float(mu0), VAR0_MEAN_SWEEP, TIMES, N_PER)
        )
        ia = mean_efficient_information(A, float(mu0), K, MODEL.beta, VAR0_MEAN_SWEEP, TIMES, N_PER)
        rmse, valid, var_hat = _rmse_a(float(mu0), VAR0_MEAN_SWEEP, rng, N_REP)
        mean_rows.append({
            "signal": float(mu0),
            "mechanism": "mean",
            "lambda_min": spec["lambda_min"],
            "I_eff": ia,
            "cond": spec["cond"],
            "rmse_a": rmse,
            "plugin_valid_rate": valid,
            "var_a_hat": var_hat,
        })

    var_rows = []
    for eps in signals:
        var0 = Q + float(eps)
        spec = information_spectrum(profiled_fisher_abb(MODEL, MU0_VAR_SWEEP, var0, TIMES, N_PER))
        ik = variance_efficient_information(K, Q, var0, TIMES, N_PER)
        rmse, valid, var_hat = _rmse_a(MU0_VAR_SWEEP, var0, rng, N_REP)
        var_rows.append({
            "signal": float(eps),
            "mechanism": "variance",
            "lambda_min": spec["lambda_min"],
            "I_eff": ik,
            "cond": spec["cond"],
            "rmse_a": rmse,
            "plugin_valid_rate": valid,
            "var_a_hat": var_hat,
        })

    p_mean, c_mean = _fit_log_slope(
        [r["signal"] for r in mean_rows], [r["lambda_min"] for r in mean_rows], 0.25,
    )
    p_var, c_var = _fit_log_slope(
        [r["signal"] for r in var_rows], [r["lambda_min"] for r in var_rows], 0.25,
    )
    p_ia, _ = _fit_log_slope(
        [r["signal"] for r in mean_rows], [r["I_eff"] for r in mean_rows], 0.25,
    )
    p_ik, _ = _fit_log_slope(
        [r["signal"] for r in var_rows], [r["I_eff"] for r in var_rows], 0.25,
    )

    payload = {
        "claim": "boundary_scaling",
        "seed": SEED,
        "model": MODEL.as_dict(),
        "times": TIMES.tolist(),
        "n_per_time": N_PER,
        "replications": N_REP,
        "predicted_exponent": 2.0,
        "lambda_min_mean_exponent": p_mean,
        "lambda_min_variance_exponent": p_var,
        "I_a_exponent": p_ia,
        "I_k_exponent": p_ik,
        "mean_prefactor": c_mean,
        "variance_prefactor": c_var,
        "mean_rows": mean_rows,
        "variance_rows": var_rows,
        "notes": (
            "Analytic I_a^eff is exactly proportional to mu0^2. I_k^eff is "
            "epsilon^2 C(epsilon); C is not universal. Fitted lambda_min "
            "exponents should be near 2 on a small-signal window."
        ),
    }
    save_json(RESULTS_DIR / "exp20_boundary_scaling.json", payload)
    save_csv(RESULTS_DIR / "exp20_boundary_scaling.csv", mean_rows + var_rows)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85))
    ax = axes[0]
    sx = [r["signal"] for r in mean_rows]
    ax.loglog(sx, [r["lambda_min"] for r in mean_rows], "o-", color="black", ms=4, label=r"$\lambda_{\min}$")
    ax.loglog(sx, [r["I_eff"] for r in mean_rows], "s--", color="#4c6a92", ms=4, label=r"$I_a^{\mathrm{eff}}$")
    xref = np.array(sx)
    ax.loglog(xref, c_mean * xref ** 2, color="0.5", lw=0.8, label=r"$\propto \mu_0^2$")
    ax.set_xlabel(r"$\mu_0$")
    ax.set_ylabel("information")
    ax.set_title(rf"mean boundary, $\hat p={p_mean:.2f}$", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "a")

    ax = axes[1]
    sx = [r["signal"] for r in var_rows]
    ax.loglog(sx, [r["lambda_min"] for r in var_rows], "o-", color="black", ms=4, label=r"$\lambda_{\min}$")
    ax.loglog(sx, [r["I_eff"] for r in var_rows], "s--", color="#c44e52", ms=4, label=r"$I_k^{\mathrm{eff}}$")
    ax.loglog(np.array(sx), c_var * np.array(sx) ** 2, color="0.5", lw=0.8, label=r"$\propto\varepsilon^2$")
    ax.set_xlabel(r"$|\Sigma_0-q|$")
    ax.set_ylabel("information")
    ax.set_title(rf"variance boundary, $\hat p={p_var:.2f}$", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "b")

    ax = axes[2]
    ax.loglog(
        [r["signal"] for r in mean_rows], [r["rmse_a"] for r in mean_rows],
        "o-", color="black", ms=4, label=r"RMSE of $\hat a$, $\mu_0\to 0$",
    )
    ax.loglog(
        [r["signal"] for r in var_rows], [r["rmse_a"] for r in var_rows],
        "s--", color="#4c6a92", ms=4, label=r"RMSE of $\hat a$, $\varepsilon\to 0$",
    )
    ax.set_xlabel("excitation")
    ax.set_ylabel(r"RMSE of $\hat a$")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "c")
    fig.suptitle("Approach to the non-identifiable boundary", fontsize=10)
    save_fig(fig, FIGS_DIR / "exp20_boundary_scaling.png", bottom_pad=0.02)

    print(
        f"p_lambda_mean={p_mean:.3f} p_lambda_var={p_var:.3f} "
        f"p_Ia={p_ia:.3f} p_Ik={p_ik:.3f}"
    )


if __name__ == "__main__":
    main()

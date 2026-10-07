"""Experiment 19 — identifiability phase diagram at fixed sample budget.

Grid over |mu0| and |var0 - q|. Analytic Fisher spectrum plus Monte Carlo
three-snapshot plug-in recovery. Independent Gaussian snapshots only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import (
    information_spectrum,
    plugin_three_equal_spacing,
    profiled_fisher_abb,
    snapshot_mean_variance,
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
N_TOT = int(N_PER * TIMES.size)
SEED = 19
N_REP = 250
TRUTH = np.array([A, MODEL.b, MODEL.beta])


def _recover(mu0, var0, rng, n_rep):
    mu, var = snapshot_mean_variance(A, K, Q, mu0, var0, TIMES)
    errors = []
    n_valid = 0
    n_admissible = 0
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
            n_admissible += 1
            errors.append([hat["a"] - A, hat["b"] - MODEL.b, hat["beta"] - MODEL.beta])
    errors = np.asarray(errors, dtype=float)
    if errors.size:
        med = np.median(np.linalg.norm(errors, axis=1))
        rmse_a = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
        p_success = float(np.mean(np.linalg.norm(errors, axis=1) < 0.5)) * (n_admissible / n_rep)
    else:
        med = float("nan")
        rmse_a = float("nan")
        p_success = 0.0
    return {
        "plugin_valid_rate": n_valid / n_rep,
        "admissible_rate": n_admissible / n_rep,
        "median_param_error": float(med),
        "rmse_a": rmse_a,
        "p_recover": float(p_success),
    }


def main():
    ensure_output_dirs()
    rng = np.random.default_rng(SEED)
    mu_grid = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
    eps_grid = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    rows = []
    lam = np.zeros((eps_grid.size, mu_grid.size))
    cond = np.zeros_like(lam)
    rec = np.zeros_like(lam)
    med = np.zeros_like(lam)
    for i, eps in enumerate(eps_grid):
        var0 = Q + float(eps)
        for j, mu0 in enumerate(mu_grid):
            spec = information_spectrum(profiled_fisher_abb(MODEL, float(mu0), var0, TIMES, N_PER))
            mc = _recover(float(mu0), var0, rng, N_REP)
            row = {
                "mu0": float(mu0),
                "eps": float(eps),
                "var0": var0,
                "lambda_min": spec["lambda_min"],
                "cond": spec["cond"],
                "numerical_rank": spec["numerical_rank"],
                "n_total": N_TOT,
                **mc,
            }
            rows.append(row)
            lam[i, j] = spec["lambda_min"]
            cond[i, j] = spec["cond"]
            rec[i, j] = mc["p_recover"]
            med[i, j] = mc["median_param_error"]
            print(
                f"mu0={mu0:.2f} eps={eps:.2f} lmin={spec['lambda_min']:.2e} "
                f"recover={mc['p_recover']:.2f} med={mc['median_param_error']}"
            )

    payload = {
        "claim": "identifiability_phase_diagram",
        "seed": SEED,
        "model": MODEL.as_dict(),
        "times": TIMES.tolist(),
        "n_per_time": N_PER,
        "n_total": N_TOT,
        "replications": N_REP,
        "mu_grid": mu_grid.tolist(),
        "eps_grid": eps_grid.tolist(),
        "rows": rows,
        "notes": (
            "Fixed total budget. Structural null at mu0=0 (ridge) and at eps=0 "
            "(k unidentified for Gaussians). Weak cells have tiny lambda_min "
            "but nonzero rank."
        ),
    }
    save_json(RESULTS_DIR / "exp19_identifiability_phase.json", payload)
    save_csv(RESULTS_DIR / "exp19_identifiability_phase.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))
    extent = [-0.5, mu_grid.size - 0.5, -0.5, eps_grid.size - 0.5]

    def _heat(ax, data, title, log=False, cmap="viridis"):
        z = np.array(data, dtype=float)
        if log:
            z = np.log10(np.clip(z, 1e-16, None))
        im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(mu_grid.size))
        ax.set_xticklabels([f"{v:g}" for v in mu_grid], fontsize=7)
        ax.set_yticks(np.arange(eps_grid.size))
        ax.set_yticklabels([f"{v:g}" for v in eps_grid], fontsize=7)
        ax.set_xlabel(r"$|\mu_0|$")
        ax.set_ylabel(r"$|\Sigma_0-q|$")
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    _heat(axes[0], lam, r"$\log_{10}\lambda_{\min}(I_{a,b,\beta})$", log=True)
    panel_label(axes[0], "a")
    _heat(axes[1], rec, r"$P(\|\hat\theta-\theta\|<0.5)$", log=False, cmap="cividis")
    panel_label(axes[1], "b")
    _heat(axes[2], med, r"median $\|\hat\theta-\theta\|$ (valid)", log=False, cmap="magma_r")
    panel_label(axes[2], "c")
    fig.suptitle("Identifiability phase diagram (fixed budget, independent snapshots)", fontsize=10)
    save_fig(fig, FIGS_DIR / "exp19_identifiability_phase.png", bottom_pad=0.02)
    _ = extent
    print("wrote exp19")


if __name__ == "__main__":
    main()

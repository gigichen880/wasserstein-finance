"""Experiment C — comparative statics in a, b, and beta."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.estimation import fit_moment_laws
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model, gaussian_pdf
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.solvers.fp import equilibrium_pdf, fokker_planck, make_grid, normalize


def _one_run(model: Model, m0, x, dx, t_final, dt):
    _, hist, _ = fokker_planck(m0, x, dx, model, t_final, dt, scheme="implicit")
    t = np.array([h["t"] for h in hist])
    mean = np.array([h["mean"] for h in hist])
    var = np.array([h["var"] for h in hist])
    fit = fit_moment_laws(mean, var, dt, mean_floor=0.02)
    return dict(
        terminal_var=float(var[-1]),
        theory_var=model.sigma2_inf,
        abs_err=abs(float(var[-1]) - model.sigma2_inf),
        fitted_a=fit.a,
        fitted_a_plus_b=fit.a_plus_b,
        true_a=model.a,
        true_a_plus_b=model.a + model.b,
        mean_rate_err=abs(fit.a - model.a) if np.isfinite(fit.a) else float("nan"),
        var_rate_err=abs(fit.a_plus_b - (model.a + model.b)),
    )


def run(t_final: float = 5.0, dt: float = 3e-3) -> dict:
    x, dx = make_grid(6.0, 321)
    m0 = gaussian_pdf(x, 1.0, 1.0)  # nonzero mean so a is identifiable
    m0 = m0 / (m0.sum() * dx)

    b_vals = np.linspace(0.0, 3.0, 7)
    a_vals = np.array([0.5, 1.0, 1.5, 2.0])
    beta_vals = np.array([0.25, 0.5, 0.75, 1.0])

    b_rows = []
    for b in b_vals:
        mdl = Model(a=1.0, b=float(b), beta=0.5)
        r = _one_run(mdl, m0, x, dx, t_final, dt)
        r.update(b=float(b), a=1.0, beta=0.5)
        b_rows.append(r)

    a_rows = []
    for a in a_vals:
        mdl = Model(a=float(a), b=0.5, beta=0.5)
        r = _one_run(mdl, m0, x, dx, t_final, dt)
        r.update(a=float(a), b=0.5, beta=0.5)
        a_rows.append(r)

    beta_rows = []
    for beta in beta_vals:
        mdl = Model(a=1.0, b=0.5, beta=float(beta))
        r = _one_run(mdl, m0, x, dx, t_final, dt)
        r.update(a=1.0, b=0.5, beta=float(beta))
        beta_rows.append(r)

    b_abs = [r["abs_err"] for r in b_rows]
    metrics = dict(
        claim="theory_implementation_sanity",
        t_final=t_final, dt=dt,
        b_max_abs_err=float(np.max(b_abs)),
        b_monotone_var_decrease=bool(np.all(np.diff([r["terminal_var"] for r in b_rows]) <= 1e-4)),
        a_mean_rate_mae=float(np.nanmean([r["mean_rate_err"] for r in a_rows])),
        b_var_rate_mae=float(np.nanmean([r["var_rate_err"] for r in b_rows])),
        beta_var_mae=float(np.nanmean([r["abs_err"] for r in beta_rows])),
        supports_claim=float(np.max(b_abs)) < 0.04,
        notes="b>0 is a dispersion penalty / synchronizing interaction, not anti-crowding repulsion.",
        b_rows=b_rows, a_rows=a_rows, beta_rows=beta_rows,
    )
    return dict(x=x, m0=m0, metrics=metrics, b_rows=b_rows, a_rows=a_rows, beta_rows=beta_rows,
                t_final=t_final)


def figure(data, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    b_rows = data["b_rows"]
    bs = [r["b"] for r in b_rows]
    ax = axes[0]
    ax.plot(bs, [r["terminal_var"] for r in b_rows], "o", label="numerical eq. var")
    ax.plot(bs, [r["theory_var"] for r in b_rows], "-", label=r"$\beta/(a+b)$")
    ax.set_xlabel("interaction $b$")
    ax.set_ylabel("equilibrium variance")
    ax.set_title("larger $b$ compresses dispersion", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    a_rows = data["a_rows"]
    ax.plot([r["a"] for r in a_rows], [r["true_a"] for r in a_rows], "k--", label="true $a$")
    ax.plot([r["a"] for r in a_rows], [r["fitted_a"] for r in a_rows], "o", label="fitted from mean law")
    ax.set_xlabel("true $a$")
    ax.set_ylabel("estimated mean rate")
    ax.set_title("mean relaxation rate", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    beta_rows = data["beta_rows"]
    ax.plot([r["beta"] for r in beta_rows], [r["theory_var"] for r in beta_rows], "-", label=r"$\beta/(a+b)$")
    ax.plot([r["beta"] for r in beta_rows], [r["terminal_var"] for r in beta_rows], "o", label="numerical")
    ax.set_xlabel(r"entropy $\beta$")
    ax.set_ylabel("equilibrium variance")
    ax.set_title("larger $\\beta$ widens equilibrium", fontsize=9)
    panel_label(ax, "(c)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, path, bottom_pad=0.02)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp03_parameter_sweep.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/03_parameter_sweep.py"
    save_json(RESULTS_DIR / "exp03_parameter_sweep.json", meta)
    save_csv(RESULTS_DIR / "exp03_parameter_sweep.csv", data["b_rows"])
    print("exp03 max |var err|", meta["b_max_abs_err"], "a-rate MAE", meta["a_mean_rate_mae"])


if __name__ == "__main__":
    main()

"""Experiment E — synthetic falsification of the quadratic gradient-flow model."""

from __future__ import annotations

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.diagnostics import alignment, moment_step_errors
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.particles import (
    bimodal_samples,
    simulate_snapshots,
    step_antigradient,
    step_forced,
    step_mckean_vlasov,
    step_quartic,
    step_state_dep_diffusion,
    step_tanh,
    step_translate,
)
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.estimation import silverman_bandwidth


def _moments(snaps):
    mu = np.array([s.mean() for s in snaps])
    var = np.array([s.var() for s in snaps])
    return mu, var


def _align_seq(snaps, model, bw):
    rows = [alignment(a, b, model, bandwidth=bw) for a, b in zip(snaps[:-1], snaps[1:])]
    return dict(
        mean_cos=float(np.mean([r.cos for r in rows])),
        mean_cos2=float(np.mean([r.cos2 for r in rows])),
        mean_residual=float(np.mean([r.residual_frac for r in rows])),
        mean_lambda=float(np.mean([r.lambda_hat for r in rows])),
    )


def run(seed: int = 0, n: int = 800, n_steps: int = 12, dt: float = 0.05) -> dict:
    model = Model()
    wrong = Model(a=2.0, b=2.0, beta=0.2)
    rng = np.random.default_rng(seed)
    X0 = bimodal_samples(n, rng, left=-1.5, right=2.2)
    bw = silverman_bandwidth(X0)
    gens = {
        "true_mv": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_mckean_vlasov(X, model, dt, rng)),
        "eval_wrong_params": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_mckean_vlasov(X, model, dt, rng)),
        "tanh_drift": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_tanh(X, model.a, model.beta, dt, rng)),
        "state_dep_diffusion": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_state_dep_diffusion(X, model, dt, rng)),
        "omitted_force": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_forced(X, model, dt, rng, force=0.8)),
        "anti_gradient": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_antigradient(X, model, dt, rng)),
        "translation": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_translate(X, 0.25)),
        "quartic_potential": lambda rng=rng: simulate_snapshots(
            X0, n_steps, dt, rng, lambda X: step_quartic(X, model, dt, rng, gamma=0.3)),
    }
    eval_model = {k: model for k in gens}
    eval_model["eval_wrong_params"] = wrong

    rows = []
    for name, gen in gens.items():
        snaps = gen()
        mdl = eval_model[name]
        al = _align_seq(snaps, mdl, bw)
        mu, var = _moments(snaps)
        mom = moment_step_errors(mu, var, dt, mdl.a, mdl.a + mdl.b, mdl.sigma2_inf)
        row = dict(dgp=name, eval_a=mdl.a, eval_b=mdl.b, eval_beta=mdl.beta, **al, **mom)
        rows.append(row)

    true = next(r for r in rows if r["dgp"] == "true_mv")
    obvious = [r for r in rows if r["dgp"] in ("anti_gradient", "translation", "tanh_drift")]
    worse_align = all(r["mean_cos"] < true["mean_cos"] - 0.05 for r in obvious)
    wrong_scale = next(r for r in rows if r["dgp"] == "eval_wrong_params")
    quartic = next(r for r in rows if r["dgp"] == "quartic_potential")
    metrics = dict(
        claim="model_validation_synthetic",
        seed=seed, n=n, n_steps=n_steps, dt=dt, bandwidth=bw,
        model=model.as_dict(), rows=rows,
        true_mean_cos=true["mean_cos"],
        true_mean_residual=true["mean_residual"],
        true_var_mae=true["var_mae"],
        wrong_scale_cos=wrong_scale["mean_cos"],
        wrong_scale_var_mae=wrong_scale["var_mae"],
        quartic_cos=quartic["mean_cos"],
        quartic_var_mae=quartic["var_mae"],
        obvious_wrong_lower_cosine=worse_align,
        scale_misspec_caught_by_variance=wrong_scale["var_mae"] > max(2.0 * true["var_mae"], 0.05),
        supports_claim=true["mean_cos"] > 0.5 and worse_align,
        notes=(
            "Positive DGP is the quadratic McKean-Vlasov. Remaining rows are "
            "misspecified dynamics or misspecified evaluation parameters. "
            "quartic_potential is a nearby nonquadratic V=(a/2)x^2+(γ/4)x^4. "
            "Cosine is scale-blind; variance MAE is the scale-sensitive check."
        ),
    )
    return dict(rows=rows, metrics=metrics)


DGP_LABELS = {
    "true_mv": "true quadratic",
    "eval_wrong_params": "wrong parameter scale",
    "tanh_drift": "nonlinear tanh drift",
    "state_dep_diffusion": "state-dependent diffusion",
    "omitted_force": "omitted force",
    "anti_gradient": "anti-gradient",
    "translation": "rigid translation",
    "quartic_potential": "quartic potential",
}


def figure(data, path) -> None:
    rows = list(data["rows"])
    names = [DGP_LABELS[r["dgp"]] for r in rows]
    if any("_" in n for n in names):
        raise RuntimeError(f"code-style labels leaked into figure: {names}")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    ypos = np.arange(len(names))

    def _bars(ax, values, color):
        ax.barh(ypos, values, color=color, height=0.7)
        ax.set_yticks(ypos, labels=names, fontsize=9)
        ax.tick_params(axis="y", pad=4)
        ax.invert_yaxis()
        ax.margins(y=0.04)

    ax = axes[0, 0]
    _bars(ax, [r["mean_cos"] for r in rows], "C0")
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"mean $\cos\theta$")
    ax.set_title("alignment (scale-free)", fontsize=9)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    _bars(ax, [r["var_mae"] for r in rows], "C1")
    ax.set_xlabel("variance-law MAE")
    ax.set_title("next-step variance error (scale-sensitive)", fontsize=9)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    _bars(ax, [r["mean_mae"] for r in rows], "C2")
    ax.set_xlabel("mean-law MAE")
    ax.set_title("next-step mean error", fontsize=9)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    _bars(ax, [r["mean_residual"] for r in rows], "0.45")
    ax.set_xlabel("unexplained displacement fraction")
    ax.set_title(r"$|u-\hat\lambda v_{pred}|^2/|u|^2$ of the marginals", fontsize=9)
    panel_label(ax, "(d)")
    fig.subplots_adjust(left=0.26, wspace=0.45, hspace=0.38)
    save_fig(fig, path, bottom_pad=0.02)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp05_synthetic_falsification.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/05_synthetic_falsification.py"
    save_json(RESULTS_DIR / "exp05_synthetic_falsification.json", meta)
    save_csv(RESULTS_DIR / "exp05_synthetic_falsification.csv", data["rows"])
    print("exp05 true cos", meta["true_mean_cos"], "wrong-scale var MAE",
          meta["wrong_scale_var_mae"], "quartic var MAE", meta["quartic_var_mae"])


if __name__ == "__main__":
    main()

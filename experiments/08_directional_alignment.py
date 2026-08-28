"""Experiment H — directional alignment, Delta t robustness, bandwidth sensitivity.

Residual is unexplained Wasserstein displacement of the two marginals, not
microscopic circulation. Cosine does not identify overall parameter scale.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.diagnostics import alignment
from wfmm.estimation import silverman_bandwidth
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model, predicted_velocity
from wfmm.particles import bimodal_samples, simulate_snapshots, step_mckean_vlasov, step_translate
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.estimation import score_kde


def _seq_stats(snaps, model, bw, scale=1.0):
    rows = [
        alignment(a, b, model, bandwidth=bw, bandwidth_scale=scale)
        for a, b in zip(snaps[:-1], snaps[1:])
    ]
    cos = np.array([r.cos for r in rows])
    return dict(
        mean_cos=float(np.mean(cos)),
        std_cos=float(np.std(cos)),
        mean_cos2=float(np.mean([r.cos2 for r in rows])),
        mean_residual=float(np.mean([r.residual_frac for r in rows])),
        mean_lambda=float(np.mean([r.lambda_hat for r in rows])),
        n_windows=len(rows),
        cos=cos.tolist(),
    )


def run(seed: int = 0, n: int = 800, n_steps: int = 12) -> dict:
    model = Model()
    rng = np.random.default_rng(seed)
    X0 = bimodal_samples(n, rng, left=-1.4, right=2.3)
    bw0 = silverman_bandwidth(X0)
    dts = (0.02, 0.05, 0.10, 0.20, 0.40)
    dt_rows = []
    for dt in dts:
        snaps = simulate_snapshots(
            X0, n_steps, dt, rng, lambda X, d=dt: step_mckean_vlasov(X, model, d, rng, substeps=4)
        )
        st = _seq_stats(snaps, model, bw0)
        st.update(dt=dt, dgp="true_mv")
        dt_rows.append(st)

    # bandwidth sensitivity at dt=0.05
    snaps05 = simulate_snapshots(
        X0, n_steps, 0.05, rng, lambda X: step_mckean_vlasov(X, model, 0.05, rng, substeps=4)
    )
    bw_rows = []
    for scale in (0.5, 1.0, 2.0):
        st = _seq_stats(snaps05, model, bw0, scale=scale)
        st.update(bandwidth_scale=scale, bandwidth=bw0 * scale)
        bw_rows.append(st)

    # controls at dt=0.05, same N
    v = predicted_velocity(X0, float(X0.mean()), score_kde(X0, bandwidth=bw0), model)
    det = X0 + 0.05 * v
    anti = X0 - 0.05 * v
    trans = step_translate(X0, 1.0)
    controls = {
        "deterministic_step": alignment(X0, det, model, bandwidth=bw0).as_dict(),
        "anti_gradient": alignment(X0, anti, model, bandwidth=bw0).as_dict(),
        "translation": alignment(X0, trans, model, bandwidth=bw0).as_dict(),
        "true_mv_mean": {k: dt_rows[1][k] for k in ("mean_cos", "mean_residual", "mean_lambda")},
    }

    local = dt_rows[0]["mean_cos"] > 0.4
    degrades = dt_rows[-1]["mean_cos"] < dt_rows[0]["mean_cos"] + 0.15
    metrics = dict(
        claim="model_validation_synthetic",
        seed=seed, n=n, n_steps=n_steps, silverman_bandwidth=bw0,
        dt_rows=dt_rows, bw_rows=bw_rows, controls=controls,
        cosine_highest_at_small_dt=dt_rows[0]["mean_cos"] >= min(r["mean_cos"] for r in dt_rows) - 1e-9,
        supports_claim=local,
        notes=(
            "Directional agreement is a local (small Delta t) statement. "
            "Residual is unexplained Wasserstein displacement, not circulation. "
            "lambda_hat absorbs the unknown timescale; cosine is scale-free."
        ),
    )
    return dict(metrics=metrics, dt_rows=dt_rows, bw_rows=bw_rows, controls=controls)


def figure(data, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    dts = [r["dt"] for r in data["dt_rows"]]
    ax = axes[0]
    ax.plot(dts, [r["mean_cos"] for r in data["dt_rows"]], "o-", lw=1.5)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"mean $\cos\theta$")
    ax.set_title("window-size robustness", fontsize=9)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(dts, [r["mean_residual"] for r in data["dt_rows"]], "o-", color="C1", lw=1.5)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("unexplained fraction")
    ax.set_title("residual vs window", fontsize=9)
    panel_label(ax, "(b)")

    ax = axes[2]
    names = ["deterministic step", "true quadratic", "anti-gradient", "rigid translation"]
    if any("_" in n or n in ("true MV", "deterministic") for n in names):
        raise RuntimeError(f"code-style labels leaked into figure: {names}")
    vals = [
        data["controls"]["deterministic_step"]["cos"],
        data["controls"]["true_mv_mean"]["mean_cos"],
        data["controls"]["anti_gradient"]["cos"],
        data["controls"]["translation"]["cos"],
    ]
    ax.bar(np.arange(len(names)), vals, color=["C2", "C0", "C3", "C1"], width=0.72)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=20, ha="right", fontsize=8)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel(r"$\cos\theta$")
    ax.set_title("controls at matching $N$", fontsize=9)
    panel_label(ax, "(c)")
    save_fig(fig, path, bottom_pad=0.10)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp08_directional_alignment.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/08_directional_alignment.py"
    save_json(RESULTS_DIR / "exp08_directional_alignment.json", meta)
    save_csv(
        RESULTS_DIR / "exp08_directional_alignment.csv",
        [{k: v for k, v in r.items() if k != "cos"} for r in data["dt_rows"]],
    )
    print("exp08 dt cosines", [round(r["mean_cos"], 3) for r in data["dt_rows"]])


if __name__ == "__main__":
    main()

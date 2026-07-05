"""
Generate paper figures and result artifacts for the mean-field market-making model.

Three main model experiments plus one solver-comparison ablation:
  1. Inventory relaxation
  2. Liquidity shock
  3. Crowding sweep (comparative statics)
  4. Solver comparison (JKO vs finite-difference)

Run:
    python make_figures.py --all
    python make_figures.py --exp 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import mfmm
from mfmm import (
    FIGS_DIR,
    RESULTS_DIR,
    Model,
    ensure_output_dirs,
    equilibrium_pdf,
    free_energy_grid,
    run_exp1,
    run_exp2,
    run_exp3,
    run_exp4,
)

FIG_EXT = "png"
FIG_DPI = 150
SHOCK_COLOR = "#f4a261"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "figure.dpi": FIG_DPI,
})

EXP1_SNAP_LABELS = {
    0.0: r"initial: bimodal split ($t=0$)",
    0.5: r"merging ($t=0.5$)",
    1.0: r"$t=1$",
    2.0: r"$t=2$",
    4.0: r"near equilibrium ($t=4$)",
}


def _panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left")


def _shock_span(ax, window):
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)


def _legend_below(ax, ncol: int = 2, fontsize: float = 6.5):
    ax.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.30), ncol=ncol, fontsize=fontsize,
    )


def _legend_upper_left(ax, fontsize: float = 7):
    ax.legend(frameon=False, loc="upper left", fontsize=fontsize)


def _save_fig(fig, path: Path, bottom_pad: float = 0.0, use_tight: bool = True) -> None:
    if use_tight:
        if bottom_pad:
            fig.tight_layout(rect=[0, bottom_pad, 1, 1])
        else:
            fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


def _save_json(path: Path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _snapshot_colors(n: int):
    cmap = plt.cm.viridis
    return [cmap(v) for v in np.linspace(0.15, 0.85, max(n, 1))]


def figure_exp1(data, path: Path) -> None:
    x = data["x"]
    model = data["model"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    ax = axes[0]
    snap_items = sorted(data["snaps"].items())
    colors = _snapshot_colors(len(snap_items))
    for (t, m), col in zip(snap_items, colors):
        lbl = EXP1_SNAP_LABELS.get(t, f"$t={t:g}$")
        ax.plot(x, m, color=col, lw=1.4, label=lbl)
    ax.plot(x, data["m_inf"], "k--", lw=1.6,
            label=rf"target $m_\infty=\mathcal{{N}}(0,\beta/(a+b))$")
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel(r"density $m_t(x)$")
    ax.set_title("distribution contracts onto equilibrium", fontsize=9, pad=6)
    ax.set_ylim(bottom=0)
    _panel_label(ax, "(a)")
    _legend_below(ax, ncol=2)

    ax = axes[1]
    ax.plot(data["t_arr"], data["energy"], color="C0", lw=1.5, label=r"$\mathcal{F}(m_t)$")
    ax.axhline(data["f_inf"], color="k", ls="--", lw=1, label=r"$\mathcal{F}(m_\infty)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"free energy $\mathcal{F}(m_t)$")
    ax.set_title("monotone energy dissipation", fontsize=9, pad=6)
    _panel_label(ax, "(b)")
    _legend_upper_left(ax)

    ax = axes[2]
    ax.plot(data["t_arr"], data["var_arr"], label="numerical", color="C0", lw=1.5)
    ax.plot(data["t_arr"], data["var_theory"], ":", color="C1", lw=1.8,
            label=r"analytic $\Sigma_t$ law")
    ax.axhline(model.sigma2_inf, color="k", ls="--", lw=1, label=r"target $\beta/(a+b)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"variance $\Sigma_t$")
    ax.set_title("variance relaxes to closed form", fontsize=9, pad=6)
    _panel_label(ax, "(c)")
    _legend_upper_left(ax)

    _save_fig(fig, path, bottom_pad=0.06)


def figure_exp2(data, path: Path) -> None:
    hist = data["hist_arr"]
    window = data["window"]
    c = data["shock_center"]
    x = data["x"]
    snap_labels = data.get("snap_labels", {})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # (a) density first — same logic as exp1 (a)
    ax = axes[0]
    plot_times = [t for t in (1.0, 3.0, 6.0) if t in data["snaps"]]
    colors = _snapshot_colors(len(plot_times))
    for t, col in zip(plot_times, colors):
        lbl = snap_labels.get(t, f"$t={t:g}$")
        ax.plot(x, data["snaps"][t], color=col, lw=1.4, label=lbl)
    ax.plot(x, data["m_eq"], "k--", lw=1.6,
            label=rf"baseline $m_\infty=\mathcal{{N}}(0,\beta/(a+b))$")
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel(r"density $m_t(x)$")
    ax.set_title("distribution shifts under tilt, then recenters", fontsize=9, pad=6)
    ax.set_ylim(bottom=0)
    _panel_label(ax, "(a)")
    _legend_below(ax, ncol=2)

    # (b) mean
    ax = axes[1]
    ax.plot(hist[:, 0], hist[:, 1], color="C0", lw=1.5, label=r"mean $\mu_t$")
    _shock_span(ax, window)
    ax.axhline(c, color=SHOCK_COLOR, ls="--", lw=1.2, label=rf"tilt target $c={c:g}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"mean inventory $\mu_t$")
    ax.set_title("mean driven toward tilt, then relaxes at rate $a$", fontsize=9, pad=6)
    _panel_label(ax, "(b)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="C0", lw=1.5, label=r"mean $\mu_t$"),
        Patch(facecolor=SHOCK_COLOR, alpha=0.18,
              label=rf"shock $t\in[{window[0]:g},{window[1]:g}]$"),
        plt.Line2D([0], [0], color=SHOCK_COLOR, ls="--", lw=1.2,
                   label=rf"tilt target $c={c:g}$"),
    ], frameon=False, fontsize=7, loc="upper left")

    # (c) W2 distance
    ax = axes[2]
    ax.plot(hist[:, 0], hist[:, 2], color="C2", lw=1.5,
            label=r"$W_2(m_t, m_\infty)$")
    _shock_span(ax, window)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$W_2(m_t, m_\infty)$")
    ax.set_title("departure from and return to equilibrium", fontsize=9, pad=6)
    _panel_label(ax, "(c)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="C2", lw=1.5, label=r"$W_2(m_t, m_\infty)$"),
        Patch(facecolor=SHOCK_COLOR, alpha=0.18,
              label=rf"shock $t\in[{window[0]:g},{window[1]:g}]$"),
    ], frameon=False, fontsize=7, loc="upper left")

    _save_fig(fig, path, bottom_pad=0.06)


def figure_exp3(data, path: Path) -> None:
    table = data["table"]
    x = data["x"]
    T = data["T"]
    mid_t = T / 2.0
    relaxation = data["rep_relaxation"]

    fig = plt.figure(figsize=(12, 6.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.1], hspace=0.45, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(table[:, 0], table[:, 1], "o", label="numerical equilibrium", color="C0", ms=5)
    ax_a.plot(table[:, 0], table[:, 2], "-", label=r"theory $\beta/(a+b)$", color="C1", lw=1.5)
    ax_a.set_xlabel("interaction strength $b$")
    ax_a.set_ylabel("equilibrium variance")
    ax_a.set_title(r"larger $b$ compresses equilibrium dispersion", fontsize=9, pad=6)
    _panel_label(ax_a, "(a)")
    ax_a.legend(frameon=False, loc="upper right")

    palette = {0.0: "C0", 1.0: "C1", 3.0: "C2"}
    for i, b in enumerate((0.0, 1.0, 3.0)):
        ax = fig.add_subplot(gs[1, i])
        r = relaxation[b]
        col = palette[b]
        ax.plot(x, r["m0"], color="0.55", ls=":", lw=1.2, label=rf"initial $\mathcal{{N}}(0,1)$")
        if r["mid"] is not None:
            ax.plot(x, r["mid"], color=col, ls="-.", lw=1.2, alpha=0.75,
                    label=rf"$t={mid_t:g}$ (compressing)")
        ax.plot(x, r["final"], color=col, lw=1.6, label=rf"$t={T:g}$ (numerical eq.)")
        ax.plot(x, r["m_inf"], "k--", lw=1.2, label=r"theory $m_\infty$")
        ax.set_xlabel("inventory $x$")
        if i == 0:
            ax.set_ylabel(r"density $m_t(x)$")
        ax.set_title(rf"relaxation for $b={b:g}$", fontsize=9, pad=4)
        ax.set_ylim(bottom=0)
        if i == 0:
            _panel_label(ax, "(b)")
        _legend_below(ax, ncol=1, fontsize=6)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.14, hspace=0.50, wspace=0.28)
    _save_fig(fig, path, use_tight=False)


def figure_exp4(data, path: Path) -> None:
    traces = data["traces"]
    model = data["model"]
    x, dx = data["x"], data["dx"]
    terminal = data["terminal"]
    f_inf = free_energy_grid(equilibrium_pdf(x, model), x, dx, model)
    labels = {"explicit": "explicit FD", "implicit": "implicit FD", "jko": "JKO (quantile)"}
    colors = {"explicit": "C3", "implicit": "C0", "jko": "C2"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))

    ax = axes[0, 0]
    for k in ("explicit", "implicit", "jko"):
        tr = traces[k]
        ax.plot(tr["t"], tr["energy"], label=labels[k], color=colors[k], lw=1.4)
    ax.axhline(f_inf, color="k", ls="--", lw=1, label=r"$\mathcal{F}(m_\infty)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\mathcal{F}(m_t)$")
    ax.set_title(r"energy at step $1.8\times$ explicit CFL limit", fontsize=9, pad=6)
    _panel_label(ax, "(a)")
    _legend_upper_left(ax)

    ax = axes[0, 1]
    for k in ("explicit", "implicit", "jko"):
        tr = traces[k]
        y = np.maximum(np.abs(tr["neg_mass"]), 1e-300)
        ax.semilogy(tr["t"], y, label=labels[k], color=colors[k], lw=1.4)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\int (m_t)_-\,dx$")
    ax.set_title("positivity: explicit blows up", fontsize=9, pad=6)
    _panel_label(ax, "(b)")
    _legend_upper_left(ax)

    ax = axes[1, 0]
    for k in ("explicit", "implicit", "jko"):
        tr = traces[k]
        y = np.maximum(tr["mass_err"], 1e-300)
        ax.semilogy(tr["t"], y, label=labels[k], color=colors[k], lw=1.4)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$|\int m_t - 1|$")
    ax.set_title("mass conservation", fontsize=9, pad=6)
    _panel_label(ax, "(c)")
    _legend_upper_left(ax)

    ax = axes[1, 1]
    m_inf = equilibrium_pdf(x, model)
    ax.plot(x, m_inf, "k--", lw=1.2, label=r"target $m_\infty$")
    for k in ("implicit", "jko"):
        ax.plot(x, terminal[k], label=labels[k], color=colors[k], lw=1.4)
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel(r"terminal density")
    ax.set_title("stable schemes reach near-equilibrium profile", fontsize=9, pad=6)
    ax.set_ylim(bottom=0)
    _panel_label(ax, "(d)")
    _legend_below(ax, ncol=1)

    _save_fig(fig, path, bottom_pad=0.04)


def save_exp1(data) -> None:
    figure_exp1(data, FIGS_DIR / f"exp1_relaxation.{FIG_EXT}")
    _save_json(RESULTS_DIR / "exp1_relaxation.json", data["metrics"])


def save_exp2(data) -> None:
    figure_exp2(data, FIGS_DIR / f"exp2_shock.{FIG_EXT}")
    _save_json(RESULTS_DIR / "exp2_shock.json", data["metrics"])


def save_exp3(data) -> None:
    figure_exp3(data, FIGS_DIR / f"exp3_crowding.{FIG_EXT}")
    table = data["table"]
    csv_path = RESULTS_DIR / "exp3_crowding_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["b", "numerical_variance", "theoretical_variance", "absolute_error", "relative_error"])
        for row in table:
            w.writerow([f"{v:.10g}" for v in row])
    _save_json(RESULTS_DIR / "exp3_crowding_sweep.json", data["metrics"])


def save_exp4(data) -> None:
    figure_exp4(data, FIGS_DIR / f"exp4_solvers.{FIG_EXT}")
    summary = data["summary"]
    csv_path = RESULTS_DIR / "exp4_solver_comparison.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "stable", "energy_increase_steps", "mass_error",
                     "min_density_or_negative_mass", "equilibrium_variance"])
        for method in ("explicit", "implicit", "jko"):
            d = summary[method]
            w.writerow([
                method,
                d["stable"],
                d["energy_increases"],
                d["mass_err"],
                d["min_mass"],
                d["eq_var"],
            ])
    json_metrics = {k: v for k, v in summary.items()}
    json_metrics["meta"] = data["meta"]
    _save_json(RESULTS_DIR / "exp4_solver_comparison.json", json_metrics)


def generate_all(model: Model | None = None) -> None:
    ensure_output_dirs()
    model = model or Model()
    print("Running experiment 1: inventory relaxation ...")
    save_exp1(run_exp1(model))
    print("Running experiment 2: liquidity shock ...")
    save_exp2(run_exp2(model))
    print("Running experiment 3: crowding sweep ...")
    save_exp3(run_exp3(model))
    print("Running experiment 4: solver comparison (JKO ablation) ...")
    save_exp4(run_exp4(model))
    print(f"\nFigures saved to {FIGS_DIR}/ (*.{FIG_EXT})")
    print(f"Results saved to {RESULTS_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Generate paper figures and result tables")
    p.add_argument("--all", action="store_true", help="run all four experiments")
    p.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="run a single experiment")
    args = p.parse_args()
    ensure_output_dirs()
    model = Model()
    runners = {1: save_exp1, 2: save_exp2, 3: save_exp3, 4: save_exp4}
    data_fns = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4}
    if args.exp:
        print(f"Running experiment {args.exp} ...")
        runners[args.exp](data_fns[args.exp](model))
    else:
        generate_all(model)


if __name__ == "__main__":
    main()

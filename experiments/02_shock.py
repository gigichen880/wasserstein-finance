"""Experiment B — transient forcing / liquidity shock.

The unadjusted energy F is not required to decrease during the forcing window.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_json
from wfmm.model import Model, free_energy_grid
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import SHOCK_COLOR, panel_label, save_fig
from wfmm.solvers.fp import equilibrium_pdf, fokker_planck, make_grid, normalize
from wfmm.transport import w2_densities


def run(model: Model | None = None, t_final: float = 8.0, dt: float = 2e-3,
        c: float = 2.0, window=(2.0, 4.0)) -> dict:
    model = model or Model()
    x, dx = make_grid(6.0, 481)
    m0 = normalize(equilibrium_pdf(x, model), dx)
    m_eq = m0.copy()
    f_eq = free_energy_grid(m_eq, x, dx, model)

    def tilt(t, w=window, cc=c):
        return cc if (w[0] <= t < w[1]) else 0.0

    snap_times = (1.0, 3.0, 6.0)
    _, hist, snaps = fokker_planck(
        m0, x, dx, model, t_final, dt, scheme="implicit",
        snapshot_times=snap_times, tilt=tilt,
    )
    snaps[0.0] = m_eq.copy()
    t = np.array([h["t"] for h in hist])
    mean = np.array([h["mean"] for h in hist])
    energy = np.array([h["energy"] for h in hist])
    w2 = np.array([model.w2_gaussian_to_eq(h["mean"], h["var"]) for h in hist])
    snap_w2 = {ts: w2_densities(snaps[ts], m_eq, x, dx) for ts in snaps}

    idx_start = int(np.argmin(np.abs(t - window[0])))
    idx_end = int(np.argmin(np.abs(t - window[1])))
    forced = (t >= window[0]) & (t <= window[1])
    after = t > window[1]
    energy_up_forced = int(np.sum(np.diff(energy[forced]) > 1e-9)) if np.any(forced) else 0
    energy_up_after = int(np.sum(np.diff(energy[after]) > 1e-9)) if np.sum(after) > 1 else 0
    quasi = c * (1.0 - np.exp(-model.a * (window[1] - window[0])))
    metrics = dict(
        claim="theory_implementation_sanity",
        model=model.as_dict(),
        shock_center=c, shock_window=list(window), dt=dt,
        mean_at_shock_start=float(mean[idx_start]),
        mean_at_shock_end=float(mean[idx_end]),
        quasi_static_prediction=float(quasi),
        shock_end_abs_error=abs(float(mean[idx_end]) - quasi),
        final_mean=float(mean[-1]),
        final_w2=float(w2[-1]),
        energy_increase_steps_during_shock=energy_up_forced,
        energy_increase_steps_after_shock=energy_up_after,
        snap_w2=snap_w2,
        supports_claim=(
            float(mean[idx_end]) > float(mean[idx_start]) + 0.5
            and abs(float(mean[idx_end]) - quasi) < 0.15
            and abs(float(mean[-1])) < abs(float(mean[idx_end]))
        ),
        notes=(
            "Unadjusted F may increase during the tilt: forcing injects energy. "
            "Monotonicity is required only on unforced segments."
        ),
    )
    return dict(
        x=x, model=model, snaps=snaps, m_eq=m_eq, f_eq=f_eq, window=window, c=c,
        t=t, mean=mean, energy=energy, w2=w2, metrics=metrics,
    )


def figure(data, path) -> None:
    x, window, c = data["x"], data["window"], data["c"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    ax = axes[0, 0]
    for t, col, lbl in zip((1.0, 3.0, 6.0), ("C0", "C1", "C2"),
                           ("before", "during", "after")):
        if t in data["snaps"]:
            ax.plot(x, data["snaps"][t], color=col, lw=1.4, label=f"{lbl} t={t:g}")
    ax.plot(x, data["m_eq"], "k--", lw=1.4, label=r"baseline $m_\infty$")
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.set_title("distribution under tilt then recenters", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[0, 1]
    ax.plot(data["t"], data["mean"], color="C0", lw=1.5, label=r"$\mu_t$")
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)
    ax.axhline(c, color=SHOCK_COLOR, ls="--", lw=1.1, label=rf"$c={c:g}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("mean inventory")
    ax.set_title("mean driven toward tilt, then rate-$a$ recovery", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="C0", lw=1.5, label=r"$\mu_t$"),
        Patch(facecolor=SHOCK_COLOR, alpha=0.18, label=rf"shock $[{window[0]:g},{window[1]:g}]$"),
        plt.Line2D([0], [0], color=SHOCK_COLOR, ls="--", lw=1.1, label=rf"tilt $c={c:g}$"),
    ], frameon=False, fontsize=7)

    ax = axes[1, 0]
    ax.plot(data["t"], data["w2"], color="C2", lw=1.5)
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$W_2$ (Gaussian-moment)")
    ax.set_title("departure from and return to equilibrium", fontsize=9)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.plot(data["t"], data["energy"], color="C0", lw=1.5, label=r"unadjusted $\mathcal{F}(m_t)$")
    ax.axhline(data["f_eq"], color="k", ls="--", lw=1, label=r"$\mathcal{F}(m_\infty)$")
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("free energy")
    ax.set_title("forcing can inject energy; do not require decrease", fontsize=9)
    panel_label(ax, "(d)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, path, bottom_pad=0.02)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp02_shock.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/02_shock.py"
    save_json(RESULTS_DIR / "exp02_shock.json", meta)
    print("exp02 mean_end", meta["mean_at_shock_end"], "pred", meta["quasi_static_prediction"],
          "E_up_forced", meta["energy_increase_steps_during_shock"])


if __name__ == "__main__":
    main()

"""Experiment B — piecewise-constant liquidity shock.

The tilt V_t = (a/2)(x-c_t)^2 is piecewise constant, so the mean ODE is exact
on each interval: μ(t) = c + (μ(t0)-c) e^{-a(t-t0)}. Baseline F_0 may rise
while the instantaneous tilted energy F_c dissipates after the switch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_csv, save_json
from wfmm.model import Model, free_energy_grid
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import SHOCK_COLOR, panel_label, save_fig
from wfmm.solvers.fp import equilibrium_pdf, fokker_planck, make_grid, normalize


def piecewise_mean(t: np.ndarray, model: Model, c: float, t0: float, t1: float,
                   mu0: float = 0.0) -> np.ndarray:
    """Exact mean for a single rectangular shock on [t0, t1)."""
    out = np.empty_like(t, dtype=float)
    before = t < t0
    during = (t >= t0) & (t < t1)
    after = t >= t1
    out[before] = model.mean_law(mu0, t[before])
    out[during] = model.mean_law_tilt(mu0, t[during] - t0, c)
    mu_end = model.mean_law_tilt(mu0, t1 - t0, c)
    out[after] = model.mean_law(mu_end, t[after] - t1)
    return out


def _run_one(model, x, dx, m0, c, window, t_final, dt):
    def tilt(t, w=window, cc=c):
        return cc if (w[0] <= t < w[1]) else 0.0

    _, hist, snaps = fokker_planck(
        m0, x, dx, model, t_final, dt, scheme="implicit",
        snapshot_times=(max(0.5, window[0] - 1.0), 0.5 * (window[0] + window[1]), window[1] + 2.0),
        tilt=tilt,
    )
    t = np.array([h["t"] for h in hist])
    mean = np.array([h["mean"] for h in hist])
    energy0 = np.array([h["energy"] for h in hist])
    energy_c = np.array([h["energy_tilt"] for h in hist])
    mu_exact = piecewise_mean(t, model, c, window[0], window[1], mu0=0.0)
    idx_end = int(np.argmin(np.abs(t - window[1])))
    pred_end = float(model.mean_law_tilt(0.0, window[1] - window[0], c))
    forced = (t >= window[0]) & (t < window[1])
    after = t >= window[1]
    e0_up_forced = int(np.sum(np.diff(energy0[forced]) > 1e-9)) if np.sum(forced) > 1 else 0
    e0_up_after = int(np.sum(np.diff(energy0[after]) > 1e-9)) if np.sum(after) > 1 else 0
    # skip the first forced step: F_c jumps when c switches, then should fall
    ec_forced = energy_c[forced]
    ec_up_forced = int(np.sum(np.diff(ec_forced[1:]) > 1e-9)) if ec_forced.size > 2 else 0
    return dict(
        t=t, mean=mean, energy0=energy0, energy_c=energy_c, mu_exact=mu_exact,
        snaps=snaps, hist=hist,
        mean_at_shock_end=float(mean[idx_end]),
        exact_shock_end=pred_end,
        shock_end_abs_error=abs(float(mean[idx_end]) - pred_end),
        mean_law_rmse=float(np.sqrt(np.mean((mean - mu_exact) ** 2))),
        energy0_increase_steps_during=e0_up_forced,
        energy0_increase_steps_after=e0_up_after,
        energy_tilt_increase_steps_during_after_switch=ec_up_forced,
        c=c, window=list(window),
    )


def run(model: Model | None = None, t_final: float = 8.0, dt: float = 2e-3,
        c: float = 2.0, window=(2.0, 4.0)) -> dict:
    model = model or Model()
    x, dx = make_grid(6.0, 481)
    m0 = normalize(equilibrium_pdf(x, model), dx)
    m_eq = m0.copy()
    f_eq = free_energy_grid(m_eq, x, dx, model, center=0.0)

    main = _run_one(model, x, dx, m0, c, window, t_final, dt)
    main["snaps"][0.0] = m_eq.copy()

    sweep_rows = []
    for cc in (1.0, 2.0, 3.0):
        for dur in (0.5, 1.0, 2.0):
            w = (2.0, 2.0 + dur)
            tf = w[1] + 2.0
            r = _run_one(model, x, dx, m0, cc, w, tf, dt)
            sweep_rows.append(dict(
                c=cc, duration=dur, window=w,
                observed_end=r["mean_at_shock_end"],
                exact_end=r["exact_shock_end"],
                abs_error=r["shock_end_abs_error"],
                mean_law_rmse=r["mean_law_rmse"],
            ))
    sweep_err = [row["abs_error"] for row in sweep_rows]
    metrics = dict(
        claim="theory_implementation_sanity",
        model=model.as_dict(),
        shock_center=c, shock_window=list(window), dt=dt,
        mean_at_shock_end=main["mean_at_shock_end"],
        exact_shock_end=main["exact_shock_end"],
        shock_end_abs_error=main["shock_end_abs_error"],
        mean_law_rmse=main["mean_law_rmse"],
        final_mean=float(main["mean"][-1]),
        energy0_increase_steps_during=main["energy0_increase_steps_during"],
        energy0_increase_steps_after=main["energy0_increase_steps_after"],
        energy_tilt_increase_steps_during_after_switch=(
            main["energy_tilt_increase_steps_during_after_switch"]
        ),
        sweep_max_abs_error=float(np.max(sweep_err)),
        sweep_mae=float(np.mean(sweep_err)),
        supports_claim=(
            main["shock_end_abs_error"] < 0.05
            and main["energy0_increase_steps_during"] > 0
            and main["energy0_increase_steps_after"] == 0
        ),
        notes=(
            "Piecewise-constant tilt: mean law is exact, not quasi-static. "
            "Baseline F_0 can rise during the shock; instantaneous F_c should "
            "dissipate after the switch. Lyapunov decrease is not claimed across "
            "the jump in c."
        ),
    )
    return dict(
        x=x, model=model, m_eq=m_eq, f_eq=f_eq, window=window, c=c,
        main=main, sweep_rows=sweep_rows, metrics=metrics,
    )


def figure(data, path) -> None:
    x, window, c = data["x"], data["window"], data["c"]
    main = data["main"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8))

    ax = axes[0, 0]
    for t, col, lbl in zip((1.0, 3.0, 6.0), ("C0", "C1", "C2"),
                           ("before", "during", "after")):
        if t in main["snaps"]:
            ax.plot(x, main["snaps"][t], color=col, lw=1.4, label=f"{lbl} t={t:g}")
    ax.plot(x, data["m_eq"], "k--", lw=1.4, label=r"baseline $m_\infty$")
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.set_title("distribution under piecewise-constant tilt", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[0, 1]
    ax.plot(main["t"], main["mean"], color="C0", lw=1.6, label=r"numerical $\mu_t$")
    ax.plot(main["t"], main["mu_exact"], color="k", ls="--", lw=1.2,
            label=r"exact $c+(\mu_0-c)e^{-a\Delta t}$")
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)
    ax.axhline(c, color=SHOCK_COLOR, ls=":", lw=1.0, label=rf"$c={c:g}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("mean inventory")
    ax.set_title("exact mean ODE on each constant-$c$ interval", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="C0", lw=1.6, label=r"numerical $\mu_t$"),
        plt.Line2D([0], [0], color="k", ls="--", lw=1.2, label="exact piecewise mean"),
        Patch(facecolor=SHOCK_COLOR, alpha=0.18, label=rf"shock $[{window[0]:g},{window[1]:g})$"),
    ], frameon=False, fontsize=7)

    ax = axes[1, 0]
    ax.plot(main["t"], main["energy0"], color="C0", lw=1.5, label=r"baseline $\mathcal{F}_0(m_t)$")
    ax.plot(main["t"], main["energy_c"], color="C3", lw=1.3,
            label=r"instantaneous $\mathcal{F}_{c_t}(m_t)$")
    ax.axhline(data["f_eq"], color="k", ls="--", lw=1, label=r"$\mathcal{F}_0(m_\infty)$")
    ax.axvspan(window[0], window[1], alpha=0.18, color=SHOCK_COLOR, zorder=0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("free energy")
    ax.set_title(r"$\mathcal{F}_0$ may rise; $\mathcal{F}_{c_t}$ jumps at switches", fontsize=9)
    panel_label(ax, "(c)")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 1]
    rows = data["sweep_rows"]
    pred = [r["exact_end"] for r in rows]
    obs = [r["observed_end"] for r in rows]
    ax.scatter(pred, obs, c="C0", s=28, zorder=3)
    lo, hi = min(pred + obs), max(pred + obs)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel(r"exact $c(1-e^{-a\Delta T})$")
    ax.set_ylabel("numerical mean at shock end")
    ax.set_title(r"amplitude/duration sweep ($c\in\{1,2,3\}$, $\Delta T\in\{0.5,1,2\}$)", fontsize=9)
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
    save_csv(RESULTS_DIR / "exp02_shock_sweep.csv", data["sweep_rows"])
    print("exp02 mean_end", meta["mean_at_shock_end"], "exact", meta["exact_shock_end"],
          "sweep mae", meta["sweep_mae"], "F0_up_forced", meta["energy0_increase_steps_during"])


if __name__ == "__main__":
    main()

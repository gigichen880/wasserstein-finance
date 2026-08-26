"""Experiment A — non-Gaussian relaxation (theory / implementation sanity)."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_json
from wfmm.model import Model, bimodal_pdf, free_energy_grid, gaussian_pdf
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.solvers.fp import equilibrium_pdf, fokker_planck, make_grid, normalize
from wfmm.transport import w2_densities


def _rmse(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def run(model: Model | None = None, t_final: float = 4.0, dt: float = 2e-3) -> dict:
    model = model or Model()
    x, dx = make_grid(6.0, 481)
    m0 = bimodal_pdf(x)
    snap_times = (0.0, 0.5, 1.0, 2.0, 4.0)
    _, hist, snaps = fokker_planck(
        m0, x, dx, model, t_final, dt, scheme="implicit", snapshot_times=snap_times
    )
    snaps[0.0] = normalize(m0.copy(), dx)
    m_inf = equilibrium_pdf(x, model)
    f_inf = free_energy_grid(m_inf, x, dx, model)
    t = np.array([h["t"] for h in hist])
    mean = np.array([h["mean"] for h in hist])
    var = np.array([h["var"] for h in hist])
    energy = np.array([h["energy"] for h in hist])
    mass = np.array([h["mass"] for h in hist])
    mu0 = float(np.sum(x * snaps[0.0]) * dx)
    var0 = float(np.sum((x - mu0) ** 2 * snaps[0.0]) * dx)
    var_th = model.var_law(var0, t)
    mean_th = model.mean_law(mu0, t)

    w2_gauss = np.array([model.w2_gaussian_to_eq(m, v) for m, v in zip(mean, var)])
    w2_0 = w2_densities(snaps[0.0], m_inf, x, dx)
    w2_bound = model.w2_contraction_bound(w2_0, t)
    w2_times, w2_vals = [], []
    for ts in snap_times:
        if ts in snaps:
            w2_times.append(ts)
            w2_vals.append(w2_densities(snaps[ts], m_inf, x, dx))
    last_t = max(snaps)
    w2_T = w2_densities(snaps[last_t], m_inf, x, dx)
    m0_shift = 0.5 * gaussian_pdf(x, -1.2, 0.2 ** 2) + 0.5 * gaussian_pdf(x, 2.8, 0.2 ** 2)
    _, hist_s, _ = fokker_planck(m0_shift, x, dx, model, t_final, dt, scheme="implicit")
    t_s = np.array([h["t"] for h in hist_s])
    mean_s = np.array([h["mean"] for h in hist_s])
    mu0_s = mean_s[0] if t_s[0] > 0 else float(np.sum(x * normalize(m0_shift, dx)) * dx)
    # hist starts at dt; reconstruct mu0 from initial
    mu0_s = float(np.sum(x * normalize(m0_shift, dx)) * dx)
    mean_s_th = model.mean_law(mu0_s, t_s)

    metrics = dict(
        claim="theory_implementation_sanity",
        model=model.as_dict(),
        dt=dt, t_final=t_final, seed=None,
        target_variance=model.sigma2_inf,
        terminal_variance=float(var[-1]),
        variance_abs_error=abs(float(var[-1]) - model.sigma2_inf),
        variance_law_rmse=_rmse(var, var_th),
        mean_bimodal_rmse=_rmse(mean, mean_th),
        mean_shifted_rmse=_rmse(mean_s, mean_s_th),
        energy_increase_steps=int(np.sum(np.diff(energy) > 1e-9)),
        mass_error_max=float(np.max(np.abs(mass - 1.0))),
        min_density=float(np.min([h["min_mass"] for h in hist])),
        terminal_w2=float(w2_T),
        initial_w2=float(w2_0),
        w2_snapshot_times=w2_times,
        w2_snapshots=w2_vals,
        supports_claim=(
            abs(float(var[-1]) - model.sigma2_inf) < 0.03
            and int(np.sum(np.diff(energy) > 1e-9)) == 0
            and _rmse(var, var_th) < 0.03
            and _rmse(mean_s, mean_s_th) < 0.05
        ),
        notes=(
            "Paper bimodal is mean-zero so the mean law is nearly vacuous there; "
            "shifted bimodal companion tests mu_t = mu0 e^{-a t}."
        ),
    )
    return dict(
        x=x, dx=dx, model=model, snaps=snaps, m_inf=m_inf, f_inf=f_inf,
        t=t, mean=mean, var=var, energy=energy, var_th=var_th, mean_th=mean_th,
        w2_gauss=w2_gauss, w2_bound=w2_bound, w2_times=np.array(w2_times),
        w2_vals=np.array(w2_vals), t_s=t_s, mean_s=mean_s, mean_s_th=mean_s_th,
        metrics=metrics,
    )


def figure(data, path) -> None:
    x = data["x"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    ax = axes[0, 0]
    cmap = plt.cm.viridis
    items = sorted(data["snaps"].items())
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, len(items))]
    for (t, m), col in zip(items, colors):
        ax.plot(x, m, color=col, lw=1.3, label=rf"$t={t:g}$")
    ax.plot(x, data["m_inf"], "k--", lw=1.5, label=r"$m_\infty$")
    ax.set_xlabel("inventory $x$")
    ax.set_ylabel(r"density")
    ax.set_title("bimodal contracts onto $N(0,\\beta/(a+b))$", fontsize=9)
    ax.set_ylim(bottom=0)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[0, 1]
    ax.plot(data["t"], data["energy"], lw=1.5, label=r"$\mathcal{F}(m_t)$")
    ax.axhline(data["f_inf"], color="k", ls="--", lw=1, label=r"$\mathcal{F}(m_\infty)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"free energy")
    ax.set_title("monotone energy decrease (unforced)", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    ax.plot(data["t"], data["var"], lw=1.5, label="numerical")
    ax.plot(data["t"], data["var_th"], ":", lw=1.8, label=r"analytic $\Sigma_t$")
    ax.axhline(data["model"].sigma2_inf, color="k", ls="--", lw=1, label=r"$\beta/(a+b)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("variance")
    ax.set_title("variance law, rate $2(a+b)$", fontsize=9)
    panel_label(ax, "(c)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    ax.plot(data["t_s"], data["mean_s"], lw=1.5, label="numerical (shifted bimodal)")
    ax.plot(data["t_s"], data["mean_s_th"], ":", lw=1.8, label=r"$\mu_0 e^{-a t}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("mean")
    ax.set_title("mean law, rate $a$ (shifted initial)", fontsize=9)
    panel_label(ax, "(d)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.plot(data["w2_times"], data["w2_vals"], "o-", lw=1.5, label=r"$W_2(m_t,m_\infty)$")
    ax.plot(data["t"], data["w2_bound"], "--", lw=1.2, color="0.3",
            label=r"$e^{-a t} W_2(m_0,m_\infty)$")
    ax.plot(data["t"], data["w2_gauss"], lw=1.0, alpha=0.7, label="Gaussian-moment $W_2$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$W_2$")
    ax.set_title("distance to equilibrium", fontsize=9)
    panel_label(ax, "(e)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1, 2]
    ax.axis("off")
    m = data["metrics"]
    txt = (
        f"terminal var {m['terminal_variance']:.4f} vs {m['target_variance']:.4f}\n"
        f"variance-law RMSE {m['variance_law_rmse']:.4f}\n"
        f"shifted-mean RMSE {m['mean_shifted_rmse']:.4f}\n"
        f"energy increases {m['energy_increase_steps']}\n"
        f"max |mass-1| {m['mass_error_max']:.1e}\n"
        f"terminal W2 {m['terminal_w2']:.4f}"
    )
    ax.text(0.0, 0.5, txt, va="center", fontsize=9, family="monospace")
    panel_label(ax, "(f)")
    save_fig(fig, path, bottom_pad=0.02)


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp01_relaxation.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/01_relaxation.py"
    save_json(RESULTS_DIR / "exp01_relaxation.json", meta)
    print("exp01", meta["terminal_variance"], "vs", meta["target_variance"],
          "var_rmse", meta["variance_law_rmse"], "mean_rmse", meta["mean_shifted_rmse"])


if __name__ == "__main__":
    main()

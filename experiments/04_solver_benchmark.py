"""Experiment D — controlled JKO vs PDE solver benchmark.

Does not claim JKO is more accurate at equal resolution unless the table
supports a fair (runtime- or dof-controlled) comparison.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_csv, save_json
from wfmm.model import Model, bimodal_pdf
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.solvers.fp import cfl_dt, equilibrium_pdf, fokker_planck, make_grid, normalize
from wfmm.solvers.jko import JKO1D
from wfmm.transport import w2_densities, w2_samples


def _fd_run(model, n, dt_factor, t_final, scheme):
    x, dx = make_grid(6.0, n)
    dt_cfl = cfl_dt(dx, model.beta)
    dt = dt_factor * dt_cfl
    m0 = bimodal_pdf(x)
    t0 = time.perf_counter()
    mT, hist, _ = fokker_planck(m0, x, dx, model, t_final, dt, scheme=scheme, renormalize=False)
    runtime = time.perf_counter() - t0
    energy = np.array([h["energy"] for h in hist], dtype=float)
    mass = np.array([h["mass"] for h in hist], dtype=float)
    min_mass = np.array([h["min_mass"] for h in hist], dtype=float)
    means = np.array([h["mean"] for h in hist], dtype=float)
    vars_ = np.array([h["var"] for h in hist], dtype=float)
    t = np.array([h["t"] for h in hist])
    diverged = bool(hist[-1]["diverged"]) or not np.isfinite(energy).all()
    m_inf = equilibrium_pdf(x, model)
    m_end = normalize(np.clip(mT, 0, None), dx) if np.isfinite(mT).all() else mT
    w2 = w2_densities(m_end, m_inf, x, dx) if np.isfinite(m_end).all() else float("inf")
    mu0 = float(np.sum(x * normalize(m0, dx)) * dx)
    var0 = float(np.sum((x - mu0) ** 2 * normalize(m0, dx)) * dx)
    mean_err = float(np.nanmean(np.abs(means - model.mean_law(mu0, t)))) if np.isfinite(means).all() else float("inf")
    var_err = float(np.nanmean(np.abs(vars_ - model.var_law(var0, t)))) if np.isfinite(vars_).all() else float("inf")
    finite_e = energy[np.isfinite(energy)]
    return dict(
        method=scheme, n=n, dx=dx, dt=dt, dt_factor=dt_factor, tau=None, M=None,
        runtime_sec=runtime, stable=not diverged,
        energy_increases=int(np.sum(np.diff(finite_e) > 1e-9)) if finite_e.size > 1 else None,
        mass_err=float(np.nanmax(np.abs(mass - 1.0))) if np.isfinite(mass).all() else float("inf"),
        min_density=float(np.nanmin(min_mass)) if np.isfinite(min_mass).all() else float("nan"),
        mean_mae=mean_err, var_mae=var_err, w2_to_eq=w2,
        terminal_var=float(vars_[-1]) if np.isfinite(vars_[-1]) else float("nan"),
        theory_var=model.sigma2_inf,
    ), hist, x, m_end, m_inf


def _jko_run(model, M, tau, t_final):
    jko = JKO1D(model, m=M)
    Q0 = jko.quantile_of_mixture([(0.5, -2.0, 0.2), (0.5, 2.0, 0.2)])
    n_steps = max(1, int(round(t_final / tau)))
    t0 = time.perf_counter()
    Qs = jko.flow(Q0, tau=tau, n_steps=n_steps)
    runtime = time.perf_counter() - t0
    energies = np.array([jko.energy(Q) for Q in Qs])
    t = np.arange(len(Qs)) * tau
    mus, vars_ = zip(*[jko.moments(Q) for Q in Qs])
    mus, vars_ = np.array(mus), np.array(vars_)
    mu0, var0 = mus[0], vars_[0]
    Q_inf = jko.quantile_of_gaussian(0.0, model.sigma2_inf)
    w2 = w2_samples(Qs[-1], Q_inf)
    return dict(
        method="jko", n=None, dx=None, dt=None, dt_factor=None, tau=tau, M=M,
        runtime_sec=runtime, stable=True,
        energy_increases=int(np.sum(np.diff(energies) > 1e-9)),
        mass_err=0.0, min_density=0.0,
        mean_mae=float(np.mean(np.abs(mus - model.mean_law(mu0, t)))),
        var_mae=float(np.mean(np.abs(vars_ - model.var_law(var0, t)))),
        w2_to_eq=w2, terminal_var=float(vars_[-1]), theory_var=model.sigma2_inf,
    ), t, energies


def run(model: Model | None = None, t_final: float = 1.5) -> dict:
    model = model or Model()
    rows = []
    traces = {}
    # modest grid: enough to see CFL failure and structure preservation
    for n in (81, 161):
        for fac in (0.5, 1.8):
            for scheme in ("explicit", "implicit"):
                row, hist, x, m_end, m_inf = _fd_run(model, n, fac, t_final, scheme)
                rows.append(row)
                key = f"{scheme}_N{n}_f{fac}"
                traces[key] = dict(
                    t=np.array([h["t"] for h in hist]),
                    energy=np.array([h["energy"] for h in hist], dtype=float),
                    method=scheme, n=n, dt_factor=fac,
                )
    jko_traces = {}
    for M in (80, 160):
        for tau in (0.05, 0.1):
            row, t, energies = _jko_run(model, M, tau, t_final)
            rows.append(row)
            jko_traces[f"jko_M{M}_tau{tau}"] = dict(t=t, energy=energies, M=M, tau=tau)

    # representative figure traces: N=161, factor 1.8, JKO M=240 tau=0.1
    metrics = dict(
        claim="numerical_method",
        t_final=t_final, model=model.as_dict(),
        rows=rows,
        notes=(
            "Explicit CFL failure is expected. JKO uses a different discretization "
            "(quantile nodes vs grid points, tau vs dt); do not read terminal variance "
            "as 'more accurate at equal resolution' without matching runtime or DOF."
        ),
    )
    return dict(rows=rows, traces=traces, jko_traces=jko_traces, metrics=metrics, model=model)


def figure(data, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for key, tr in data["traces"].items():
        if "N161_f1.8" in key:
            y = np.array(tr["energy"], dtype=float)
            y = np.where(np.isfinite(y), y, np.nan)
            ax.plot(tr["t"], y, lw=1.3, label=tr["method"])
    for key, tr in data["jko_traces"].items():
        if "M160_tau0.1" in key:
            ax.plot(tr["t"], tr["energy"], lw=1.3, label="jko M=160 tau=0.1")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathcal{F}$")
    ax.set_title("energy at N=161, dt=1.8 CFL (JKO separate tau)", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    rows = data["rows"]
    labels, w2s, runtimes = [], [], []
    for r in rows:
        if r["method"] == "explicit" and r["dt_factor"] == 1.8:
            continue
        lab = r["method"]
        if r["method"] == "jko":
            lab = f"jko M={r['M']} τ={r['tau']}"
        else:
            lab = f"{r['method']} N={r['n']} {r['dt_factor']}CFL"
        labels.append(lab)
        w2s.append(r["w2_to_eq"] if np.isfinite(r["w2_to_eq"]) else np.nan)
        runtimes.append(r["runtime_sec"])
    ax.scatter(runtimes, w2s)
    for lab, rt, w in zip(labels, runtimes, w2s):
        ax.annotate(lab, (rt, w), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("runtime (s)")
    ax.set_ylabel(r"$W_2$ to $m_\infty$ at $T$")
    ax.set_title("accuracy vs cost (stable runs)", fontsize=9)
    panel_label(ax, "(b)")
    save_fig(fig, path, bottom_pad=0.02)


def main():
    argparse.ArgumentParser().parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run()
    figure(data, FIGS_DIR / "exp04_solver_benchmark.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/04_solver_benchmark.py"
    # don't duplicate huge rows twice
    save_json(RESULTS_DIR / "exp04_solver_benchmark.json", meta)
    save_csv(RESULTS_DIR / "exp04_solver_benchmark.csv", data["rows"])
    print("exp04 n_runs", len(data["rows"]))


if __name__ == "__main__":
    main()

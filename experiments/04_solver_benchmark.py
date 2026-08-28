"""Experiment D — structural solver comparison (not an accuracy ranking).

Signed mass of the conservative flux scheme is separate from positivity.
Explicit Euler may keep ∫m dx while developing m < 0 and then diverging.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_csv, save_json
from wfmm.model import Model, bimodal_pdf
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.solvers.fp import cfl_dt, fokker_planck, make_grid, normalize
from wfmm.solvers.jko import JKO1D


def _fd_run(model, n, dt_factor, t_final, scheme):
    x, dx = make_grid(6.0, n)
    dt_cfl = cfl_dt(dx, model.beta)
    dt = dt_factor * dt_cfl
    m0 = bimodal_pdf(x)
    t0 = time.perf_counter()
    mT, hist, _ = fokker_planck(
        m0, x, dx, model, t_final, dt, scheme=scheme, renormalize=False
    )
    runtime = time.perf_counter() - t0
    energy = np.array([h["energy"] for h in hist], dtype=float)
    signed = np.array([h["signed_mass"] for h in hist], dtype=float)
    neg = np.array([h["neg_mass"] for h in hist], dtype=float)
    min_mass = np.array([h["min_mass"] for h in hist], dtype=float)
    diverged = bool(hist[-1]["diverged"]) or not np.isfinite(energy).all()
    finite_e = energy[np.isfinite(energy)]
    return dict(
        method=scheme, n=n, dx=float(dx), dt=float(dt), dt_factor=dt_factor,
        tau=None, M=None, runtime_sec=runtime, stable=not diverged,
        energy_increases=int(np.sum(np.diff(finite_e) > 1e-9)) if finite_e.size > 1 else None,
        signed_mass_err=float(np.nanmax(np.abs(signed - 1.0))) if np.isfinite(signed).all() else float("inf"),
        neg_mass_min=float(np.nanmin(neg)) if np.isfinite(neg).all() else float("nan"),
        min_density=float(np.nanmin(min_mass)) if np.isfinite(min_mass).all() else float("nan"),
        lost_positivity=bool(np.nanmin(min_mass) < -1e-12) if np.isfinite(min_mass).all() else True,
    ), hist, x, mT


def _jko_run(model, M, tau, t_final):
    jko = JKO1D(model, m=M)
    Q0 = jko.quantile_of_mixture([(0.5, -2.0, 0.2), (0.5, 2.0, 0.2)], seed=0)
    n_steps = max(1, int(round(t_final / tau)))
    t0 = time.perf_counter()
    Qs = jko.flow(Q0, tau=tau, n_steps=n_steps)
    runtime = time.perf_counter() - t0
    energies = np.array([jko.energy(Q) for Q in Qs])
    t = np.arange(len(Qs)) * tau
    slopes = [np.min(np.diff(Q)) for Q in Qs]
    return dict(
        method="jko", n=None, dx=None, dt=None, dt_factor=None, tau=tau, M=M,
        runtime_sec=runtime, stable=True,
        energy_increases=int(np.sum(np.diff(energies) > 1e-9)),
        signed_mass_err=0.0, neg_mass_min=0.0, min_density=0.0,
        lost_positivity=False,
        min_quantile_increment=float(np.min(slopes)),
    ), t, energies, hist_mass(Qs, t)


def hist_mass(Qs, t):
    return dict(t=t, signed_mass=np.ones_like(t), neg_mass=np.zeros_like(t),
                min_mass=np.zeros_like(t), energy=None)


def run(model: Model | None = None, t_final: float = 1.5) -> dict:
    model = model or Model()
    rows = []
    traces = {}
    for n in (81, 161):
        for fac in (0.5, 1.8):
            for scheme in ("explicit", "implicit"):
                row, hist, x, mT = _fd_run(model, n, fac, t_final, scheme)
                rows.append(row)
                traces[f"{scheme}_N{n}_f{fac}"] = dict(
                    t=np.array([h["t"] for h in hist]),
                    energy=np.array([h["energy"] for h in hist], dtype=float),
                    signed_mass=np.array([h["signed_mass"] for h in hist], dtype=float),
                    neg_mass=np.array([h["neg_mass"] for h in hist], dtype=float),
                    min_mass=np.array([h["min_mass"] for h in hist], dtype=float),
                    method=scheme, n=n, dt_factor=fac,
                )
    jko_traces = {}
    for M, tau in ((80, 0.1), (160, 0.05)):
        row, t, energies, _ = _jko_run(model, M, tau, t_final)
        rows.append(row)
        jko_traces[f"jko_M{M}_tau{tau}"] = dict(t=t, energy=energies, M=M, tau=tau)

    expl = next(r for r in rows if r["method"] == "explicit" and r["dt_factor"] == 0.5 and r["n"] == 161)
    expl_cfl = next(r for r in rows if r["method"] == "explicit" and r["dt_factor"] == 1.8 and r["n"] == 161)
    impl = next(r for r in rows if r["method"] == "implicit" and r["dt_factor"] == 0.5 and r["n"] == 161)
    jko = next(r for r in rows if r["method"] == "jko" and r["M"] == 160)
    metrics = dict(
        claim="numerical_method",
        t_final=t_final, model=model.as_dict(),
        rows=rows,
        explicit_below_cfl_signed_mass_err=expl["signed_mass_err"],
        explicit_below_cfl_lost_positivity=expl["lost_positivity"],
        explicit_above_cfl_stable=expl_cfl["stable"],
        explicit_above_cfl_lost_positivity=expl_cfl["lost_positivity"],
        implicit_signed_mass_err=impl["signed_mass_err"],
        implicit_lost_positivity=impl["lost_positivity"],
        jko_energy_increases=jko["energy_increases"],
        notes=(
            "Conservative flux: signed mass ∫m dx is separate from positivity. "
            "Explicit Euler below CFL keeps signed mass; above CFL it loses "
            "positivity and can diverge. Do not read this table as an accuracy ranking."
        ),
    )
    return dict(rows=rows, traces=traces, jko_traces=jko_traces, metrics=metrics, model=model)


def figure(data, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    for key, tr in data["traces"].items():
        if "N161_f1.8" in key:
            y = np.where(np.isfinite(tr["energy"]), tr["energy"], np.nan)
            ax.plot(tr["t"], y, lw=1.3, label=tr["method"])
    for key, tr in data["jko_traces"].items():
        if "M160_tau0.05" in key:
            ax.plot(tr["t"], tr["energy"], lw=1.3, label="jko M=160 τ=0.05")
    ax.set_xlabel("time")
    ax.set_ylabel(r"baseline $\mathcal{F}$")
    ax.set_title("energy at N=161, dt=1.8 CFL (JKO separate τ)", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    tr_e = data["traces"]["explicit_N161_f1.8"]
    tr_i = data["traces"]["implicit_N161_f0.5"]
    sm_e = np.array(tr_e["signed_mass"], dtype=float)
    nm = np.maximum(-np.array(tr_e["neg_mass"], dtype=float), 1e-18)
    t_e = np.asarray(tr_e["t"])
    t_i = np.asarray(tr_i["t"])
    ax.plot(t_i, tr_i["signed_mass"], color="C0", lw=1.4,
            label="implicit 0.5 CFL, signed mass")
    mask = np.isfinite(sm_e) & (np.abs(sm_e) < 2.0)
    ax.plot(t_e[mask], sm_e[mask], color="C3", lw=1.2,
            label=r"explicit 1.8 CFL, signed mass (while $|\int m|\leq 2$)")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("signed mass")
    ax.set_ylim(0.0, 2.05)
    ax.set_title("signed mass (stable) vs positivity loss", fontsize=9)
    panel_label(ax, "(b)")
    ax2 = ax.twinx()
    ax2.semilogy(t_e, nm, color="C3", ls=":", lw=1.2, label=r"explicit $\int m^-$")
    ax2.set_ylabel(r"explicit $\int m^-$ (log)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6)
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
    save_json(RESULTS_DIR / "exp04_solver_benchmark.json", meta)
    save_csv(RESULTS_DIR / "exp04_solver_benchmark.csv", data["rows"])
    print("exp04 explicit signed-mass (0.5 CFL)", meta["explicit_below_cfl_signed_mass_err"],
          "above CFL stable?", meta["explicit_above_cfl_stable"],
          "implicit signed-mass", meta["implicit_signed_mass_err"])


if __name__ == "__main__":
    main()

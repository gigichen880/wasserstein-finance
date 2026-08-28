"""Experiment 09 — matched refinement vs analytic truth.

Gaussian IC: m_T is N(μ_T, Σ_T). Bimodal Gaussian-mixture IC: each component
evolves in closed form (same affine McKean–Vlasov map). Compare JKO and FP
without claiming a winner a priori.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_csv, save_json
from wfmm.model import (
    Model, bimodal_pdf, evolve_gaussian_mixture, free_energy_grid, gaussian_pdf,
    mixture_pdf,
)
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig
from wfmm.solvers.fp import cfl_dt, fokker_planck, make_grid, normalize
from wfmm.solvers.jko import JKO1D
from wfmm.transport import w2_densities

MU0 = 1.0
VAR0 = 1.0
BIMODAL0 = [(0.5, -2.0, 0.2 ** 2), (0.5, 2.0, 0.2 ** 2)]
N_WARMUP = 1
N_REPEAT = 10


def _cpu_label() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _exact_gauss(model: Model, t: float) -> tuple[float, float]:
    return float(model.mean_law(MU0, t)), float(model.var_law(VAR0, t))


def _median_time(fn, n_repeat: int, n_warmup: int) -> tuple[object, float, list[float]]:
    for _ in range(n_warmup):
        fn()
    times, last = [], None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return last, float(np.median(times)), times


def _fit_loglog(xs, ys) -> tuple[float, float]:
    x = np.log(np.asarray(xs, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    p, alpha = np.polyfit(x, y, 1)
    return float(p), float(alpha)


def _fp_row(model, n, dt, scheme, t_final, ic="gaussian", dt_factor=None):
    x, dx = make_grid(6.0, n)
    if ic == "gaussian":
        m0 = normalize(gaussian_pdf(x, MU0, VAR0), dx)
    else:
        m0 = normalize(bimodal_pdf(x), dx)

    def _solve():
        return fokker_planck(m0, x, dx, model, t_final, dt, scheme=scheme, renormalize=False)

    t0 = time.perf_counter()
    mT, hist, _ = _solve()
    runtime = time.perf_counter() - t0

    energy = np.array([h["energy"] for h in hist], dtype=float)
    signed = np.array([h["signed_mass"] for h in hist], dtype=float)
    min_mass = np.array([h["min_mass"] for h in hist], dtype=float)
    diverged = bool(hist[-1]["diverged"]) or not np.isfinite(energy).all()
    mu_T, var_T = hist[-1]["mean"], hist[-1]["var"]

    if ic == "gaussian":
        mu_ex, var_ex = _exact_gauss(model, t_final)
        m_ex = normalize(gaussian_pdf(x, mu_ex, var_ex), dx)
    else:
        comps = evolve_gaussian_mixture(BIMODAL0, t_final, model)
        m_ex = normalize(mixture_pdf(x, comps), dx)
        w, m, v = zip(*comps)
        mu_ex = float(np.dot(w, m))
        var_ex = float(np.dot(w, np.array(v) + np.array(m) ** 2) - mu_ex ** 2)

    if diverged or not np.isfinite(mT).all():
        w2 = mean_err = var_err = e_err = float("inf")
    else:
        m_end = normalize(np.clip(mT, 0.0, None), dx)
        w2 = w2_densities(m_end, m_ex, x, dx)
        mean_err = abs(float(mu_T) - mu_ex)
        var_err = abs(float(var_T) - var_ex)
        e_err = abs(free_energy_grid(m_end, x, dx, model) - free_energy_grid(m_ex, x, dx, model))
    finite_e = energy[np.isfinite(energy)]
    return dict(
        method=scheme, ic=ic, n=n, M=None, dx=float(dx), dt=float(dt), tau=None,
        dt_factor=dt_factor, t_final=t_final, runtime_sec=runtime,
        runtime_median_sec=None, n_timing_repeats=None,
        stable=not diverged,
        mean_err=mean_err, var_err=var_err, w2_to_exact=w2, energy_err=e_err,
        signed_mass_err=float(np.nanmax(np.abs(signed - 1.0))) if np.isfinite(signed).all() else float("inf"),
        min_density=float(np.nanmin(min_mass)) if np.isfinite(min_mass).all() else float("nan"),
        energy_increases=int(np.sum(np.diff(finite_e) > 1e-9)) if finite_e.size > 1 else None,
        mu_T=float(mu_T) if np.isfinite(mu_T) else float("nan"),
        var_T=float(var_T) if np.isfinite(var_T) else float("nan"),
        mu_exact=mu_ex, var_exact=var_ex,
    )


def _jko_evolve(model, M, tau, t_final, ic="gaussian"):
    """JKO evolution only; used for both accuracy rows and median timings."""
    jko = JKO1D(model, m=M)
    n_steps = max(1, int(round(t_final / tau)))
    if ic == "gaussian":
        Q0 = jko.quantile_of_gaussian(MU0, VAR0)
    else:
        Q0 = jko.quantile_of_gaussian_mixture(BIMODAL0)

    def _solve():
        Q = Q0.copy()
        for _ in range(n_steps):
            Q = jko.step(Q, tau)
        return Q

    t0 = time.perf_counter()
    Q = _solve()
    runtime = time.perf_counter() - t0
    return jko, Q0, Q, n_steps, runtime


def _jko_row(model, M, tau, t_final, ic="gaussian"):
    jko, Q0, Q, n_steps, runtime = _jko_evolve(model, M, tau, t_final, ic=ic)
    T = n_steps * tau
    e_num = jko.energy(Q)
    e0 = jko.energy(Q0)
    mu_T, var_T = jko.moments(Q)
    if ic == "gaussian":
        mu_ex, var_ex = _exact_gauss(model, T)
        w2 = jko.w2_to_gaussian(Q, mu_ex, var_ex)
        e_ex = jko.energy(jko.quantile_of_gaussian(mu_ex, var_ex))
    else:
        comps = evolve_gaussian_mixture(BIMODAL0, T, model)
        w2 = jko.w2_to_mixture(Q, comps)
        w, m, v = zip(*comps)
        mu_ex = float(np.dot(w, m))
        var_ex = float(np.dot(w, np.array(v) + np.array(m) ** 2) - mu_ex ** 2)
        e_ex = jko.energy(jko.quantile_of_gaussian_mixture(comps))
    return dict(
        method="jko", ic=ic, n=None, M=M, dx=None, dt=None, tau=float(tau),
        dt_factor=None, t_final=T, runtime_sec=runtime,
        runtime_median_sec=None, n_timing_repeats=None, stable=True,
        mean_err=abs(mu_T - mu_ex), var_err=abs(var_T - var_ex),
        w2_to_exact=w2, energy_err=abs(e_num - e_ex),
        signed_mass_err=0.0, min_density=0.0,
        energy_increases=int(e_num > e0 + 1e-9),
        mu_T=mu_T, var_T=var_T, mu_exact=mu_ex, var_exact=var_ex,
    )


def _time_row(row, model, t_final, n_repeat, n_warmup):
    if row["method"] == "jko":
        def fn():
            _jko_evolve(model, row["M"], row["tau"], t_final, ic=row["ic"])
    else:
        def fn():
            return _fp_row(model, row["n"], row["dt"], row["method"], t_final,
                           ic=row["ic"], dt_factor=row["dt_factor"])
    _, med, times = _median_time(fn, n_repeat=n_repeat, n_warmup=n_warmup)
    row["runtime_median_sec"] = med
    row["n_timing_repeats"] = n_repeat
    row["runtime_repeats"] = times
    return row


def run(model: Model | None = None, t_final: float = 1.0, quick: bool = False) -> dict:
    model = model or Model()
    rows = []
    taus = (0.2, 0.1, 0.05, 0.025) if quick else (0.2, 0.1, 0.05, 0.025, 0.0125)
    Ms = (60, 120, 240) if quick else (60, 120, 240, 480)
    Ns = (81, 161, 321) if quick else (81, 161, 321, 641)
    dt_list = (0.2, 0.1, 0.05) if quick else (0.2, 0.1, 0.05, 0.025)
    n_repeat = 3 if quick else N_REPEAT

    print("exp09 JKO Gaussian grid", len(Ms) * len(taus))
    for M in Ms:
        for tau in taus:
            row = _jko_row(model, M, tau, t_final, ic="gaussian")
            rows.append(row)
            print(f"  jko M={M} tau={tau} w2={row['w2_to_exact']:.4g} t={row['runtime_sec']:.2f}s")

    print("exp09 implicit FP Gaussian")
    for n in Ns:
        for dt in dt_list:
            row = _fp_row(model, n, dt, "implicit", t_final, ic="gaussian")
            rows.append(row)

    print("exp09 explicit FP Gaussian")
    for n in Ns:
        x, dx = make_grid(6.0, n)
        dt_cfl = cfl_dt(dx, model.beta)
        for fac in (0.5, 0.25):
            row = _fp_row(model, n, fac * dt_cfl, "explicit", t_final,
                          ic="gaussian", dt_factor=fac)
            rows.append(row)

    bimodal_jko = ((60, 0.05), (240, 0.05), (240, 0.0125)) if quick else (
        (60, 0.2), (60, 0.05), (60, 0.0125),
        (240, 0.2), (240, 0.05), (240, 0.0125),
    )
    print("exp09 bimodal exact-mixture subset")
    for M, tau in bimodal_jko:
        row = _jko_row(model, M, tau, t_final, ic="bimodal")
        rows.append(row)
        print(f"  bimodal jko M={M} tau={tau} w2={row['w2_to_exact']:.4g}")
    for n, dt in ((161, 0.05), (641, 0.025)) if not quick else ((161, 0.05),):
        row = _fp_row(model, n, dt, "implicit", t_final, ic="bimodal")
        rows.append(row)
        print(f"  bimodal implicit N={n} dt={dt} w2={row['w2_to_exact']:.4g}")
    x, dx = make_grid(6.0, 161)
    row = _fp_row(model, 161, 0.5 * cfl_dt(dx, model.beta), "explicit", t_final,
                  ic="bimodal", dt_factor=0.5)
    rows.append(row)
    print(f"  bimodal explicit N=161 0.5CFL w2={row['w2_to_exact']:.4g}")

    gauss = [r for r in rows if r["ic"] == "gaussian" and r["stable"] and np.isfinite(r["w2_to_exact"])]
    jko_g = [r for r in gauss if r["method"] == "jko"]
    impl_g = [r for r in gauss if r["method"] == "implicit"]
    expl_g = [r for r in gauss if r["method"] == "explicit"]
    slopes = {}
    for M in sorted({r["M"] for r in jko_g}):
        sub = sorted([r for r in jko_g if r["M"] == M], key=lambda r: r["tau"])
        p, alpha = _fit_loglog([r["tau"] for r in sub], [r["w2_to_exact"] for r in sub])
        slopes[str(M)] = dict(p=p, alpha=alpha, n=len(sub))
        print(f"  JKO M={M} loglog slope p={p:.3f}")

    timing_keys = [
        ("jko", "gaussian", dict(M=240, tau=0.05)),
        ("jko", "gaussian", dict(M=480, tau=0.0125)),
        ("implicit", "gaussian", dict(n=161, dt=0.05)),
        ("implicit", "gaussian", dict(n=641, dt=0.025)),
        ("explicit", "gaussian", dict(n=161, dt_factor=0.5)),
        ("explicit", "gaussian", dict(n=641, dt_factor=0.5)),
        ("jko", "bimodal", dict(M=240, tau=0.05)),
    ]
    if quick:
        timing_keys = timing_keys[:2]
    print("exp09 median timings", n_repeat, "repeats after", N_WARMUP, "warmup")
    for method, ic, spec in timing_keys:
        match = None
        for r in rows:
            if r["method"] != method or r["ic"] != ic:
                continue
            if all(r.get(k) == v or (k == "dt" and r.get("dt") is not None and abs(r["dt"] - v) < 1e-12)
                   for k, v in spec.items() if k != "dt_factor"):
                if "dt_factor" in spec and r.get("dt_factor") != spec["dt_factor"]:
                    continue
                match = r
                break
        if match is None:
            continue
        _time_row(match, model, t_final, n_repeat=n_repeat, n_warmup=N_WARMUP)
        print(f"  median {method} {ic} {spec} {match['runtime_median_sec']:.4g}s")

    def _best(group):
        return min(group, key=lambda r: r["w2_to_exact"]) if group else None

    bimodal_rows = [r for r in rows if r["ic"] == "bimodal" and r["stable"]]
    metrics = dict(
        claim="numerical_method",
        t_final=t_final, mu0=MU0, var0=VAR0, model=model.as_dict(),
        n_runs=len(rows),
        timing_protocol=dict(
            n_warmup=N_WARMUP, n_repeat=n_repeat,
            timer="time.perf_counter",
            threading="single-threaded NumPy/SciPy",
            platform=platform.platform(),
            processor=_cpu_label(),
        ),
        jko_loglog_slopes=slopes,
        best_jko= _best(jko_g), best_implicit=_best(impl_g), best_explicit=_best(expl_g),
        best_jko_w2=_best(jko_g)["w2_to_exact"] if jko_g else None,
        best_implicit_w2=_best(impl_g)["w2_to_exact"] if impl_g else None,
        best_explicit_w2=_best(expl_g)["w2_to_exact"] if expl_g else None,
        best_jko_bimodal=_best([r for r in bimodal_rows if r["method"] == "jko"]),
        best_implicit_bimodal=_best([r for r in bimodal_rows if r["method"] == "implicit"]),
        notes=(
            "Gaussian truth is N(μ_T, Σ_T). Bimodal truth is the exact Gaussian "
            "mixture obtained by applying the affine McKean–Vlasov map to each "
            "component. Scatter runtimes are single-run; table medians are "
            "n_repeat after one warmup."
        ),
    )
    if "240" in slopes:
        metrics["jko_p_M240"] = slopes["240"]["p"]
    return dict(rows=rows, metrics=metrics, model=model)


def figure(data, path) -> None:
    rows = [r for r in data["rows"] if r["stable"] and np.isfinite(r["w2_to_exact"])]
    gauss = [r for r in rows if r["ic"] == "gaussian"]
    bimod = [r for r in rows if r["ic"] == "bimodal"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    ax = axes[0, 0]
    for M, marker in zip(sorted({r["M"] for r in gauss if r["method"] == "jko"}),
                         ("o", "s", "^", "D")):
        sub = sorted([r for r in gauss if r["method"] == "jko" and r["M"] == M],
                     key=lambda r: r["tau"])
        ax.loglog([r["tau"] for r in sub], [r["w2_to_exact"] for r in sub],
                  marker=marker, lw=1.2, label=f"JKO M={M}")
    if any(r["method"] == "jko" and r["M"] == 240 for r in gauss):
        sub240 = sorted([r for r in gauss if r["method"] == "jko" and r["M"] == 240],
                        key=lambda r: r["tau"])
        tau0, w0 = sub240[-1]["tau"], sub240[-1]["w2_to_exact"]
        taus = np.array([r["tau"] for r in sub240])
        ax.loglog(taus, w0 * (taus / tau0), "k--", lw=0.9, label=r"slope $1$ through $M{=}240$")
    ax.set_xlabel(r"proximal step $\tau$")
    ax.set_ylabel(r"$W_2(m_T^{\mathrm{num}}, m_T^{\mathrm{exact}})$")
    ax.set_title("JKO refinement in τ (Gaussian truth)", fontsize=9)
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[0, 1]
    for n, marker in zip(sorted({r["n"] for r in gauss if r["method"] == "implicit"}),
                         ("o", "s", "^", "D")):
        sub = sorted([r for r in gauss if r["method"] == "implicit" and r["n"] == n],
                     key=lambda r: r["dt"])
        ax.loglog([r["dt"] for r in sub], [r["w2_to_exact"] for r in sub],
                  marker=marker, lw=1.2, label=f"implicit N={n}")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"$W_2(m_T^{\mathrm{num}}, m_T^{\mathrm{exact}})$")
    ax.set_title("implicit FP refinement in Δt", fontsize=9)
    panel_label(ax, "(b)")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 0]
    for method, color, marker in (("jko", "C0", "o"), ("implicit", "C1", "s"),
                                  ("explicit", "C2", "^")):
        sub = [r for r in gauss if r["method"] == method]
        ax.loglog([r["runtime_sec"] for r in sub], [r["w2_to_exact"] for r in sub],
                  marker=marker, ls="none", color=color, label=method, alpha=0.85)
    for method, color, marker in (("jko", "C0", "o"), ("implicit", "C1", "s"),
                                  ("explicit", "C2", "^")):
        sub = [r for r in bimod if r["method"] == method]
        if sub:
            ax.loglog([r["runtime_sec"] for r in sub], [r["w2_to_exact"] for r in sub],
                      marker=marker, ls="none", color=color, mfc="none",
                      label=f"{method} bimodal", alpha=0.9)
    ax.set_xlabel("runtime (s, single run)")
    ax.set_ylabel(r"$W_2$ to exact $m_T$")
    ax.set_title("error vs runtime (open = exact bimodal)", fontsize=9)
    panel_label(ax, "(c)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1, 1]
    for method, color, marker in (("jko", "C0", "o"), ("implicit", "C1", "s"),
                                  ("explicit", "C2", "^")):
        sub = [r for r in gauss if r["method"] == method]
        ax.loglog([r["runtime_sec"] for r in sub], [r["var_err"] for r in sub],
                  marker=marker, ls="none", color=color, label=method, alpha=0.85)
    ax.set_xlabel("runtime (s, single run)")
    ax.set_ylabel(r"$|\Sigma_T^{\mathrm{num}}-\Sigma_T^{\mathrm{exact}}|$")
    ax.set_title("variance error vs runtime (Gaussian IC)", fontsize=9)
    panel_label(ax, "(d)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, path, bottom_pad=0.02)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    data = run(quick=args.quick)
    figure(data, FIGS_DIR / "exp09_convergence.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/09_convergence.py"
    save_json(RESULTS_DIR / "exp09_convergence.json", meta)
    save_csv(RESULTS_DIR / "exp09_convergence.csv", data["rows"])
    print("exp09 best JKO W2", meta["best_jko_w2"], "p_M240", meta.get("jko_p_M240"),
          "best implicit", meta["best_implicit_w2"])


if __name__ == "__main__":
    main()

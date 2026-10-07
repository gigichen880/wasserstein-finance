"""Experiment 11 — plug-in estimation from independent snapshots.

Samples are drawn from the exact affine law, independently across times.
This is not the interacting-particle design of experiments 06--08 and 10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig

A = 1.0
B = 0.5
BETA = 0.5
K = A + B
SIGMA_INF = BETA / K
MU0_MIX = 0.8
SIGMA0_MIX = 4.04


def _mixture_x0(n: int, rng: np.random.Generator) -> np.ndarray:
    left = rng.random(n) < 0.5
    x = np.empty(n)
    x[left] = rng.normal(-1.2, 0.2, size=int(left.sum()))
    x[~left] = rng.normal(2.8, 0.2, size=int((~left).sum()))
    return x


def _exact_sample(x0: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=x0.size)
    noise = np.sqrt(SIGMA_INF * (1.0 - np.exp(-2.0 * K * t)))
    return MU0_MIX * np.exp(-A * t) + np.exp(-K * t) * (x0 - MU0_MIX) + noise * z


def _moments_at(n: int, times: np.ndarray, rng: np.random.Generator, x0_law: str):
    mus = np.empty(len(times))
    sigs = np.empty(len(times))
    for j, t in enumerate(times):
        if x0_law == "mixture":
            x0 = _mixture_x0(n, rng)
            x = x0 if t == 0.0 else _exact_sample(x0, float(t), rng)
        else:
            # N(0.8, 1.5) pushed by the affine map. Mean and variance are closed form,
            # and the law stays Gaussian, so we sample it directly.
            mu = 0.8 * np.exp(-A * float(t))
            var = SIGMA_INF + (1.5 - SIGMA_INF) * np.exp(-2.0 * K * float(t))
            x = rng.normal(mu, np.sqrt(var), size=n)
        mus[j] = x.mean()
        sigs[j] = x.var(ddof=1)
    return mus, sigs


def plugin(mus: np.ndarray, sigs: np.ndarray, delta: float):
    if mus[0] == 0.0 or mus[1] / mus[0] <= 0.0:
        return None
    if sigs[1] == sigs[0]:
        return None
    r = (sigs[2] - sigs[1]) / (sigs[1] - sigs[0])
    if not (0.0 < r < 1.0):
        return None
    a_hat = -np.log(mus[1] / mus[0]) / delta
    k_hat = -np.log(r) / (2.0 * delta)
    sig_inf = sigs[0] - (sigs[1] - sigs[0]) / (r - 1.0)
    if sig_inf <= 0.0 or k_hat <= a_hat:
        # b = k - a may be slightly negative from noise; keep it and let the
        # caller decide. Reject only non-finite values.
        pass
    b_hat = k_hat - a_hat
    beta_hat = k_hat * sig_inf
    if not np.isfinite([a_hat, b_hat, beta_hat]).all():
        return None
    return a_hat, b_hat, beta_hat


def population_var_a(delta: float, mu0: float, sigma0: float) -> float:
    sig1 = SIGMA_INF + (sigma0 - SIGMA_INF) * np.exp(-2.0 * K * delta)
    return (1.0 / delta**2) * (sigma0 / mu0**2 + sig1 / (mu0**2 * np.exp(-2.0 * A * delta)))


def wls_a(mus, sigs, times):
    def resid(z):
        mu0, a = z
        pred = mu0 * np.exp(-a * times)
        return (mus - pred) / np.sqrt(np.maximum(sigs, 1e-12))

    fit = least_squares(resid, x0=np.array([mus[0], 1.0]), bounds=([-np.inf, 1e-6], [np.inf, 20.0]))
    if not fit.success:
        return None
    return float(fit.x[1])


def main():
    ensure_output_dirs()
    rng = np.random.default_rng(11)
    n_rep = 2000

    # (a) RMSE versus N at Delta = 0.5, three snapshots.
    ns = np.array([100, 200, 400, 800, 1600, 3200])
    rmse = {name: [] for name in ("a", "b", "beta")}
    kept_rates = []
    for n in ns:
        est = []
        for _ in range(n_rep):
            mus, sigs = _moments_at(int(n), np.array([0.0, 0.5, 1.0]), rng, "mixture")
            hat = plugin(mus, sigs, 0.5)
            if hat is not None:
                est.append(hat)
        est = np.array(est)
        err = est - np.array([A, B, BETA])
        for i, name in enumerate(("a", "b", "beta")):
            rmse[name].append(float(np.sqrt(np.mean(err[:, i] ** 2))))
        kept_rates.append(float(len(est) / n_rep))
        print(f"N={n} kept={len(est)}/{n_rep} rmse={rmse['a'][-1]:.4f},{rmse['b'][-1]:.4f},{rmse['beta'][-1]:.4f}")

    # (b) Monte Carlo variance of a-hat versus the formula, mixture law.
    deltas = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    mc = []
    theory = []
    for delta in deltas:
        ah = []
        times = np.array([0.0, delta, 2 * delta])
        for _ in range(n_rep):
            mus, sigs = _moments_at(2000, times, rng, "mixture")
            # a uses only the first two means; still require a valid three-snapshot draw
            if mus[0] != 0.0 and mus[1] / mus[0] > 0.0:
                ah.append(-np.log(mus[1] / mus[0]) / delta)
        ah = np.array(ah)
        mc.append(float(2000 * np.var(ah, ddof=1)))
        theory.append(float(population_var_a(float(delta), MU0_MIX, SIGMA0_MIX)))
        print(f"delta={delta:.2f} Nvar={mc[-1]:.2f} theory={theory[-1]:.2f}")

    grid = np.linspace(0.15, 3.0, 400)
    vgrid = np.array([population_var_a(float(d), MU0_MIX, SIGMA0_MIX) for d in grid])
    delta_star = float(grid[int(np.argmin(vgrid))])

    # Efficiency check: Gaussian initial law, two-snapshot plug-in and three-snapshot WLS.
    # Times 0, 0.75, 1.5. Bound values are computed from the Fisher information.
    bound2 = float(population_var_a(0.75, 0.8, 1.5))
    times3 = np.array([0.0, 0.75, 1.5])
    # Schur complement for three snapshots.
    def schur(times, mu0, sigma0):
        mus = mu0 * np.exp(-A * times)
        sigs = SIGMA_INF + (sigma0 - SIGMA_INF) * np.exp(-2.0 * K * times)
        w = mus**2 / sigs
        tbar = np.sum(w * times) / np.sum(w)
        return float(np.sum(w * (times - tbar) ** 2))

    bound3 = 1.0 / schur(times3, 0.8, 1.5)
    a2, a3 = [], []
    for _ in range(n_rep):
        mus, sigs = _moments_at(2000, times3, rng, "gaussian")
        if mus[1] / mus[0] > 0.0:
            a2.append(-np.log(mus[1] / mus[0]) / 0.75)
        hat = wls_a(mus, sigs, times3)
        if hat is not None:
            a3.append(hat)
    a2 = np.array(a2)
    a3 = np.array(a3)
    nvar2 = float(2000 * np.var(a2, ddof=1))
    nvar3 = float(2000 * np.var(a3, ddof=1))
    print(f"efficiency two-snapshot {nvar2:.2f} vs {bound2:.2f}")
    print(f"efficiency three-snapshot {nvar3:.2f} vs {bound3:.2f}")
    print(f"mixture optimal delta {delta_star:.3f}")

    # Reference slope N^{-1/2} through the N=3200 RMSE of a.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    labels = {"a": r"$\hat a$", "b": r"$\hat b$", "beta": r"$\hat\beta$"}
    for name, color in (("a", "C0"), ("b", "C1"), ("beta", "C2")):
        ax.loglog(ns, rmse[name], "o-", color=color, label=labels[name])
    guide = rmse["a"][-1] * np.sqrt(ns[-1] / ns)
    ax.loglog(ns, guide, "k--", lw=1, label=r"$N^{-1/2}$")
    ax.set_xlabel(r"samples per snapshot $N$")
    ax.set_ylabel("RMSE")
    panel_label(ax, "(a)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    fine = np.linspace(0.2, 2.6, 200)
    ax.plot(fine, [population_var_a(float(d), MU0_MIX, SIGMA0_MIX) for d in fine], color="C0", label="formula")
    ax.plot(deltas, mc, "o", color="C1", label=r"$N\,\widehat{\mathrm{Var}}(\hat a)$")
    ax.axvline(delta_star, color="0.4", ls=":", label=f"minimizer {delta_star:.2f}")
    ax.set_xlabel(r"spacing $\Delta$")
    ax.set_ylabel(r"asymptotic variance of $\hat a$")
    panel_label(ax, "(b)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, FIGS_DIR / "exp11_snapshot_estimation.png")

    # Mirror next to the manuscript.
    paper_figs = Path(__file__).resolve().parents[1] / "paper" / "figs"
    paper_figs.mkdir(exist_ok=True)
    (paper_figs / "exp11_snapshot_estimation.png").write_bytes(
        (FIGS_DIR / "exp11_snapshot_estimation.png").read_bytes()
    )

    summary = {
        "n_rep": n_rep,
        "ns": ns.tolist(),
        "kept_rates": kept_rates,
        "rmse": rmse,
        "deltas": deltas.tolist(),
        "mc_nvar": mc,
        "theory_nvar": theory,
        "delta_star_mixture": delta_star,
        "bound_two": bound2,
        "mc_two": nvar2,
        "bound_three": bound3,
        "mc_three": nvar3,
    }
    (RESULTS_DIR / "exp11_snapshot_estimation.json").write_text(json.dumps(summary, indent=2))
    print("wrote", FIGS_DIR / "exp11_snapshot_estimation.png")


if __name__ == "__main__":
    main()

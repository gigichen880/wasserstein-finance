"""Experiment 18 — information geometry and optimal snapshot design.

Independent Gaussian snapshots of the exact affine law. No interacting
particles. Compares analytic Fisher information to Monte Carlo plug-in error
at equal total sample budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.identifiability import (
    constant_variance_optimal_delta,
    fisher_psi,
    information_spectrum,
    mean_efficient_information,
    plugin_three_equal_spacing,
    profiled_fisher_abb,
    snapshot_mean_variance,
    two_snapshot_avar_a,
    variance_efficient_information,
)
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig

MODEL = Model(a=1.0, b=0.5, beta=0.5)
A = MODEL.a
K = MODEL.a + MODEL.b
Q = MODEL.sigma2_inf
MU0 = 0.8
VAR0 = 1.5
N_PER = 400
SEED = 18


def _sample_moments(mu, var, n, rng):
    x = rng.normal(mu, np.sqrt(var), size=n)
    return float(x.mean()), float(x.var(ddof=1))


def main():
    ensure_output_dirs()
    rng = np.random.default_rng(SEED)
    n_rep = 500

    deltas = np.linspace(0.12, 3.0, 48)
    ia_unknown = []
    ia_known = []
    va_moving = []
    va_frozen = []
    ik_three = []
    for d in deltas:
        times2 = np.array([0.0, float(d)])
        ia_unknown.append(mean_efficient_information(A, MU0, K, MODEL.beta, VAR0, times2, N_PER))
        info = fisher_psi(A, K, Q, MU0, VAR0, times2, N_PER)
        ia_known.append(float(info[1, 1]))
        va_moving.append(two_snapshot_avar_a(A, K, Q, MU0, VAR0, float(d)))
        va_frozen.append(two_snapshot_avar_a(A, K, Q, MU0, Q, float(d)))
        times3 = np.array([0.0, 0.5 * float(d), float(d)])
        ik_three.append(variance_efficient_information(K, Q, VAR0, times3, N_PER))

    dstar_frozen = constant_variance_optimal_delta(A)
    dstar_moving = float(deltas[int(np.argmin(va_moving))])
    dstar_ia = float(deltas[int(np.argmax(ia_unknown))])

    # Monte Carlo two-snapshot a-hat at selected spacings.
    mc_rows = []
    mc_deltas = np.array([0.2, 0.5, dstar_moving, 1.5, 2.5])
    for d in mc_deltas:
        hats = []
        mu, var = snapshot_mean_variance(A, K, Q, MU0, VAR0, np.array([0.0, float(d)]))
        for _ in range(n_rep):
            m0, _ = _sample_moments(mu[0], var[0], N_PER, rng)
            m1, _ = _sample_moments(mu[1], var[1], N_PER, rng)
            if m0 != 0.0 and m1 / m0 > 0.0:
                hats.append(-np.log(m1 / m0) / float(d))
        hats = np.asarray(hats, dtype=float)
        nvar = float(N_PER * np.var(hats, ddof=1)) if hats.size > 2 else float("nan")
        theory = two_snapshot_avar_a(A, K, Q, MU0, VAR0, float(d))
        mc_rows.append({
            "delta": float(d),
            "n_kept": int(hats.size),
            "mc_nvar": nvar,
            "theory_avar": theory,
            "rmse": float(np.sqrt(np.mean((hats - A) ** 2))) if hats.size else float("nan"),
        })

    # Too-early / too-late three-snapshot grids: times t, t+Δ, t+2Δ with Δ=0.5.
    origins = np.linspace(0.0, 2.5, 26)
    origin_rows = []
    for t0 in origins:
        times = t0 + np.array([0.0, 0.5, 1.0])
        ia = mean_efficient_information(A, MU0, K, MODEL.beta, VAR0, times, N_PER)
        ik = variance_efficient_information(K, Q, VAR0, times, N_PER)
        spec = information_spectrum(profiled_fisher_abb(MODEL, MU0, VAR0, times, N_PER))
        origin_rows.append({
            "t0": float(t0),
            "I_a_eff": ia,
            "I_k_eff": ik,
            "lambda_min": spec["lambda_min"],
            "cond": spec["cond"],
        })

    # Equal total budget: 3 snapshots, N_tot = 1200.
    budget = 1200
    alloc_rows = []
    for name, counts in (
        ("equal", np.array([400, 400, 400], dtype=float)),
        ("front_loaded", np.array([800, 300, 100], dtype=float)),
        ("back_loaded", np.array([100, 300, 800], dtype=float)),
        ("two_time_0_and_1", np.array([600, 1.0, 599], dtype=float)),
    ):
        times = np.array([0.0, 0.5, 1.0])
        spec = information_spectrum(profiled_fisher_abb(MODEL, MU0, VAR0, times, counts))
        ia = mean_efficient_information(A, MU0, K, MODEL.beta, VAR0, times, counts)
        ik = variance_efficient_information(K, Q, VAR0, times, counts)
        alloc_rows.append({
            "design": name,
            "counts": counts.tolist(),
            "I_a_eff": ia,
            "I_k_eff": ik,
            "lambda_min": spec["lambda_min"],
            "cond": spec["cond"],
            "total": float(counts.sum()),
        })

    # Centered vs shifted at equal budget.
    centered = information_spectrum(profiled_fisher_abb(MODEL, 0.0, VAR0, np.array([0.0, 0.5, 1.0]), N_PER))
    shifted = information_spectrum(profiled_fisher_abb(MODEL, MU0, VAR0, np.array([0.0, 0.5, 1.0]), N_PER))

    # Three-snapshot plug-in RMSE vs N (sanity vs experiment 11, Gaussian law).
    n_grid = np.array([100, 200, 400, 800])
    rmse_n = []
    times = np.array([0.0, 0.5, 1.0])
    mu, var = snapshot_mean_variance(A, K, Q, MU0, VAR0, times)
    for n in n_grid:
        hats = []
        for _ in range(n_rep):
            mus = np.empty(3)
            sigs = np.empty(3)
            for j in range(3):
                mus[j], sigs[j] = _sample_moments(mu[j], var[j], int(n), rng)
            hat = plugin_three_equal_spacing(mus, sigs, 0.5)
            if hat is not None and hat["admissible"]:
                hats.append([hat["a"], hat["b"], hat["beta"]])
        hats = np.asarray(hats, dtype=float)
        err = hats - np.array([A, MODEL.b, MODEL.beta]) if hats.size else np.empty((0, 3))
        rmse_n.append({
            "n": int(n),
            "kept": int(len(hats)),
            "rmse_a": float(np.sqrt(np.mean(err[:, 0] ** 2))) if len(hats) else float("nan"),
            "rmse_b": float(np.sqrt(np.mean(err[:, 1] ** 2))) if len(hats) else float("nan"),
            "rmse_beta": float(np.sqrt(np.mean(err[:, 2] ** 2))) if len(hats) else float("nan"),
        })

    payload = {
        "claim": "snapshot_information_design",
        "seed": SEED,
        "model": MODEL.as_dict(),
        "mu0": MU0,
        "var0": VAR0,
        "n_per_time": N_PER,
        "delta_star_frozen": dstar_frozen,
        "delta_star_moving_grid": dstar_moving,
        "delta_star_I_a_unknown_mu0": dstar_ia,
        "centered_lambda_min": centered["lambda_min"],
        "shifted_lambda_min": shifted["lambda_min"],
        "mc_avar": mc_rows,
        "allocations": alloc_rows,
        "rmse_vs_n": rmse_n,
        "max_I_a_unknown": float(np.max(ia_unknown)),
        "I_a_at_small_delta": float(ia_unknown[0]),
        "I_a_at_large_delta": float(ia_unknown[-1]),
        "notes": (
            "Centered lambda_min is structurally zero. Two-snapshot v_a(Delta) has an "
            "interior minimum: too-close and too-late times are both weak."
        ),
    }
    save_json(RESULTS_DIR / "exp18_information_design.json", payload)
    save_csv(RESULTS_DIR / "exp18_mc_avar.csv", mc_rows)
    save_csv(RESULTS_DIR / "exp18_allocations.csv", alloc_rows)
    save_csv(RESULTS_DIR / "exp18_origin_times.csv", origin_rows)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85))
    ax = axes[0]
    ax.plot(deltas, ia_unknown, color="black", lw=1.4, label=r"$I_a^{\mathrm{eff}}$ (unknown $\mu_0$)")
    ax.plot(deltas, np.array(ia_known) / 8.0, color="#4c6a92", lw=1.1, ls="--", label=r"$I_{aa}/8$ (known $\mu_0$)")
    ax.axvline(dstar_ia, color="0.4", ls=":", lw=0.9)
    ax.set_xlabel(r"spacing $\Delta$ (times $0,\Delta$)")
    ax.set_ylabel("information for $a$")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(deltas, va_moving, color="black", lw=1.4, label=r"moving $\Sigma_t$")
    ax.plot(deltas, va_frozen, color="#c44e52", lw=1.1, ls="--", label=r"frozen $\Sigma=\sigma_\infty^2$")
    ax.axvline(dstar_frozen, color="#c44e52", ls=":", lw=0.8)
    ax.axvline(dstar_moving, color="black", ls=":", lw=0.8)
    mc_d = [r["delta"] for r in mc_rows]
    mc_v = [r["mc_nvar"] for r in mc_rows]
    ax.scatter(mc_d, mc_v, color="#4c6a92", s=18, zorder=3, label=r"MC $N\mathrm{Var}(\hat a)$")
    ax.set_xlabel(r"spacing $\Delta$")
    ax.set_ylabel(r"asymp. var. of $\sqrt{N}(\hat a-a)$")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "b")

    ax = axes[2]
    ax.plot([r["t0"] for r in origin_rows], [r["I_a_eff"] for r in origin_rows],
            color="black", lw=1.4, label=r"$I_a^{\mathrm{eff}}$")
    ax.plot([r["t0"] for r in origin_rows], [r["lambda_min"] for r in origin_rows],
            color="#4c6a92", lw=1.1, label=r"$\lambda_{\min}(I_{a,b,\beta})$")
    ax.set_xlabel(r"window start $t_0$ (times $t_0+\{0,0.5,1\}$)")
    ax.set_ylabel("information")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "c")
    fig.suptitle("Snapshot design: information for the affine Gaussian law", fontsize=10)
    save_fig(fig, FIGS_DIR / "exp18_information_design.png", bottom_pad=0.02)

    print(
        f"d*_frozen={dstar_frozen:.3f} d*_moving={dstar_moving:.3f} "
        f"d*_Ia={dstar_ia:.3f} centered_lmin={centered['lambda_min']:.2e} "
        f"shifted_lmin={shifted['lambda_min']:.3g}"
    )


if __name__ == "__main__":
    main()

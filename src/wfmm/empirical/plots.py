"""Three checkpoint figures."""

from __future__ import annotations

import numpy as np

from wfmm.paths import EMPIRICAL_FIGS_DIR, ensure_empirical_dirs
from wfmm.plotting import panel_label, save_fig
import matplotlib.pyplot as plt


def figure_moments(t, mu, var, fits: dict, leakage, hist_x, path=None):
    ensure_empirical_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    ax = axes[0]
    ax.plot(t, mu, color="black", lw=1.2, label="empirical mean")
    for name, fit in fits.items():
        ax.plot(t, fit.center + (mu[0] - fit.center) * np.exp(-fit.a * t), lw=1, label=name)
    ax.set_xlabel("minutes from window start")
    ax.set_ylabel("cross-sectional mean")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(t, var, color="black", lw=1.2, label="empirical var")
    for name, fit in fits.items():
        ax.plot(t, fit.q + (var[0] - fit.q) * np.exp(-2.0 * fit.k * t), lw=1, label=name)
    ax.set_xlabel("minutes from window start")
    ax.set_ylabel("cross-sectional variance")
    panel_label(ax, "b")
    ax = axes[2]
    ax.hist(hist_x[np.isfinite(hist_x)], bins=20, range=(-1, 1), color="#4c6a92", density=True)
    ax.set_xlabel("top-of-book imbalance")
    ax.set_ylabel("density")
    ax.set_title(f"P(|N|>1)≈{leakage:.3f}", fontsize=8)
    panel_label(ax, "c")
    fig.suptitle("Moment restrictions on liquidity-state imbalance", fontsize=10)
    out = path or (EMPIRICAL_FIGS_DIR / "fig1_moment_restrictions.png")
    save_fig(fig, out)
    return out


def figure_forecasts(rows: list[dict], path=None):
    ensure_empirical_dirs()
    models = sorted({r["model"] for r in rows})
    horizons = sorted({int(r["horizon"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(horizons))
    width = 0.12
    for i, model in enumerate(models):
        means = []
        for h in horizons:
            vals = [r["w2"] for r in rows if r["model"] == model and int(r["horizon"]) == h and np.isfinite(r["w2"])]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (i - len(models) / 2) * width, means, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("horizon (minutes)")
    ax.set_ylabel("mean W2 vs realized cross-section")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.set_title("Paired held-out distribution forecasts")
    out = path or (EMPIRICAL_FIGS_DIR / "fig2_forecasts.png")
    save_fig(fig, out, bottom_pad=0.08)
    return out


def figure_stability(by_group: dict, path=None):
    ensure_empirical_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    keys = list(by_group)
    for ax, key in zip(axes, keys[:3]):
        labels = list(by_group[key])
        vals = [by_group[key][lab] for lab in labels]
        ax.bar(np.arange(len(labels)), vals, color="#4c6a92")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("mean W2")
        ax.set_title(key, fontsize=9)
        panel_label(ax, "abc"[keys.index(key)])
    fig.suptitle("Stability across excitation, groups, and regimes", fontsize=10)
    out = path or (EMPIRICAL_FIGS_DIR / "fig3_stability.png")
    save_fig(fig, out, bottom_pad=0.05)
    return out

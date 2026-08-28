"""Experiment 10 — repeated-seed synthetic parameter recovery.

Fit (a, a+b, σ_∞²) on a training window only, then report bias/MAE across seeds
and out-of-sample moment and W2 forecast errors. Synthetic McKean–Vlasov only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.diagnostics import forecast_w2, moment_step_errors
from wfmm.estimation import fit_moment_laws
from wfmm.io_util import save_csv, save_json
from wfmm.model import Model
from wfmm.particles import bimodal_samples, simulate_snapshots, step_mckean_vlasov
from wfmm.paths import FIGS_DIR, RESULTS_DIR, ensure_output_dirs
from wfmm.plotting import panel_label, save_fig


def _one_seed(model, n, dt, n_train, n_test, seed, with_jko=False):
    rng = np.random.default_rng(seed)
    n_steps = n_train + n_test
    X0 = bimodal_samples(n, rng, left=-1.2, right=2.8)
    snaps = simulate_snapshots(
        X0, n_steps, dt, rng, lambda X: step_mckean_vlasov(X, model, dt, rng, substeps=4)
    )
    mu = np.array([s.mean() for s in snaps])
    var = np.array([s.var() for s in snaps])
    split = n_train + 1  # indices 0..n_train inclusive => n_train pairs
    fit = fit_moment_laws(mu[:split], var[:split], dt, mean_floor=0.05)
    test = moment_step_errors(
        mu[split - 1:], var[split - 1:], dt, fit.a, fit.a_plus_b, fit.sigma2_inf
    )
    w2s = []
    for i in range(split - 1, n_steps):
        fc = forecast_w2(
            snaps[i], snaps[i + 1], model, dt, rng,
            a=fit.a if np.isfinite(fit.a) else model.a,
            a_plus_b=fit.a_plus_b,
            sigma2_inf=fit.sigma2_inf,
            with_jko=with_jko,
        )
        w2s.append(fc)
    return dict(
        seed=seed, n=n, dt=dt, n_train=n_train, n_test=n_test,
        fitted_a=fit.a, fitted_a_plus_b=fit.a_plus_b, fitted_b=fit.b,
        fitted_sigma2_inf=fit.sigma2_inf, fitted_beta=fit.beta,
        test_mean_mae=test["mean_mae"], test_var_mae=test["var_mae"],
        mean_w2_persistence=float(np.mean([w["w2_persistence"] for w in w2s])),
        mean_w2_gaussian=float(np.mean([w["w2_gaussian_moments"] for w in w2s])),
        mean_w2_jko=float("nan"),
        n_mean_pairs=fit.n_mean_pairs,
    )


def _summarize(vals, true=None):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    out = dict(
        n=int(x.size),
        mean=float(x.mean()) if x.size else float("nan"),
        std=float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        mae=float(np.mean(np.abs(x - true))) if true is not None and x.size else float("nan"),
        rmse=float(np.sqrt(np.mean((x - true) ** 2))) if true is not None and x.size else float("nan"),
        bias=float(x.mean() - true) if true is not None and x.size else float("nan"),
        q05=float(np.quantile(x, 0.05)) if x.size else float("nan"),
        q95=float(np.quantile(x, 0.95)) if x.size else float("nan"),
    )
    return out


def run(n_seeds: int = 40, quick: bool = False) -> dict:
    model = Model()
    configs = []
    ns = (200, 800) if quick else (200, 800, 1600)
    dts = (0.05, 0.2)
    n_train = 12
    n_test = 8
    seeds = range(n_seeds)
    rows = []
    for n in ns:
        for dt in dts:
            cfg_rows = []
            for seed in seeds:
                row = _one_seed(model, n, dt, n_train, n_test, seed, with_jko=False)
                rows.append(row)
                cfg_rows.append(row)
            configs.append(dict(
                n=n, dt=dt, n_train=n_train, n_test=n_test, n_seeds=n_seeds,
                a=_summarize([r["fitted_a"] for r in cfg_rows], true=model.a),
                a_plus_b=_summarize([r["fitted_a_plus_b"] for r in cfg_rows], true=model.a + model.b),
                b=_summarize([r["fitted_b"] for r in cfg_rows], true=model.b),
                beta=_summarize([r["fitted_beta"] for r in cfg_rows], true=model.beta),
                sigma2_inf=_summarize([r["fitted_sigma2_inf"] for r in cfg_rows], true=model.sigma2_inf),
                test_mean_mae=_summarize([r["test_mean_mae"] for r in cfg_rows]),
                test_var_mae=_summarize([r["test_var_mae"] for r in cfg_rows]),
                w2_persistence=_summarize([r["mean_w2_persistence"] for r in cfg_rows]),
                w2_gaussian=_summarize([r["mean_w2_gaussian"] for r in cfg_rows]),
                w2_jko=_summarize([r["mean_w2_jko"] for r in cfg_rows]),
            ))
            last = configs[-1]
            print(f"  n={n} dt={dt} a MAE={last['a']['mae']:.3f} "
                  f"var MAE={last['test_var_mae']['mean']:.3f} "
                  f"W2 gauss={last['w2_gaussian']['mean']:.3f}")

    default = next(c for c in configs if c["n"] == 800 and abs(c["dt"] - 0.05) < 1e-12)
    metrics = dict(
        claim="model_validation_synthetic",
        model=model.as_dict(), n_seeds=n_seeds, n_train=n_train, n_test=n_test,
        configs=configs,
        default_a_mae=default["a"]["mae"],
        default_a_bias=default["a"]["bias"],
        default_apb_mae=default["a_plus_b"]["mae"],
        default_test_mean_mae=default["test_mean_mae"]["mean"],
        default_test_var_mae=default["test_var_mae"]["mean"],
        default_w2_persist=default["w2_persistence"]["mean"],
        default_w2_gauss=default["w2_gaussian"]["mean"],
        notes=(
            "Synthetic MV particles only. a from mean decay, a+b from variance "
            "relaxation, β from σ_∞²(a+b). Parameters fit on the training window."
        ),
    )
    return dict(rows=rows, configs=configs, metrics=metrics, model=model)


def figure(data, path) -> None:
    configs = data["configs"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    labels = [f"n={c['n']}\nΔt={c['dt']}" for c in configs]

    ax = axes[0]
    ax.bar(labels, [c["a"]["mae"] for c in configs], color="C0")
    ax.set_ylabel(r"MAE of $\hat a$")
    ax.set_title(r"recovery of $a$ (true $=1$)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.bar(labels, [c["test_var_mae"]["mean"] for c in configs], color="C1")
    ax.set_ylabel("test variance MAE")
    ax.set_title("out-of-sample variance forecast", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    panel_label(ax, "(b)")

    ax = axes[2]
    x = np.arange(len(configs))
    w = 0.35
    ax.bar(x - w / 2, [c["w2_persistence"]["mean"] for c in configs], w, label="persistence")
    ax.bar(x + w / 2, [c["w2_gaussian"]["mean"] for c in configs], w, label="Gaussian moments")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"mean one-step $W_2$")
    ax.set_title("forecast vs persistence", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "(c)")
    save_fig(fig, path, bottom_pad=0.02)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-seeds", type=int, default=None)
    args = p.parse_args()
    n_seeds = args.n_seeds if args.n_seeds is not None else (8 if args.quick else 40)
    ensure_output_dirs()
    t0 = time.time()
    data = run(n_seeds=n_seeds, quick=args.quick)
    figure(data, FIGS_DIR / "exp10_parameter_recovery.png")
    meta = dict(data["metrics"])
    meta["runtime_sec"] = time.time() - t0
    meta["command"] = "python experiments/10_parameter_recovery.py"
    save_json(RESULTS_DIR / "exp10_parameter_recovery.json", meta)
    save_csv(RESULTS_DIR / "exp10_parameter_recovery.csv", data["rows"])
    print("exp10 default a MAE", meta["default_a_mae"], "var MAE", meta["default_test_var_mae"])


if __name__ == "__main__":
    main()

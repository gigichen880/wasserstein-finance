"""
empirical_validation.py  --  paper Section 5 ("Toward empirical validation")
============================================================================

Tests whether a sequence of empirical distributions moves *parallel to the JKO
iterates* of the mean-field market-making free energy F (V=a/2 x^2, W=b/2(x-y)^2,
entropy beta). The step-size-free content of "is this a gradient flow of F" is the
DIRECTION of motion, so we measure, in the Otto / L^2(m_k) inner product,

    cos theta_k = < v_obs , v_pred >        with   v_pred = -grad dF/dm
                  -----------------------            v_obs  = (T_k - id)/dt
                  ||v_obs|| ||v_pred||

  * cos = 1   : observed shift is exactly along the JKO iterate (pure gradient flow)
  * cos = 0   : observed shift is orthogonal to the energy gradient (rotational /
                non-equilibrium circulation)
  * grad_fraction = cos^2 is the share of the motion explained by the energy; the
    rest is the non-gradient residual. Reporting that split is the honest target
    (markets are non-equilibrium steady states, not pure gradient flows).

It runs the diagnostic on three input classes so the number has a yardstick:

  [POS] true JKO iterates of F          -> cos ~ 1     (positive control; built with mfmm.py)
  [NEG] rigid translation away from 0   -> cos ~ -1    (anti-gradient control; energy INCREASES)
  [DATA] your repo's distribution shifts -> the actual measurement

Plus two structure-light cross-checks from Section 5:
  * Lyapunov: fraction of steps with dF <= 0 (a gradient flow needs ~1.0).
  * Gibbs form (Prop. 7): does the stationary distribution match N(0, beta/(a+b))?

Run:
    python empirical_validation.py                 # controls + a repo-style stand-in
    PYTHONPATH=src python empirical_validation.py --case A3   # against the real repo data
    python empirical_validation.py --npz mydists.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from gradient_flow_alignment import (
    EnergyFunctional,
    evaluate_gradient_flow_alignment,
    snapshots_from_npz,
    snapshots_from_repo_case,
)
import mfmm


# --------------------------------------------------------------------------- #
# Energy: identical functional to the paper, in the sample-based representation
# --------------------------------------------------------------------------- #
def energy_from_model(model: mfmm.Model) -> EnergyFunctional:
    # kappa == crowding b ; quadratic potential a ; entropy beta
    return EnergyFunctional(a=model.a, kappa=model.b, beta=model.beta)


# --------------------------------------------------------------------------- #
# Input sequences
# --------------------------------------------------------------------------- #
def jko_snapshots(model: mfmm.Model, tau=5e-2, n_steps=40, M=240):
    """[POS] Genuine JKO iterates of F as 1-D empirical distributions.
    Each quantile vector Q is M equiprobable samples; consecutive Q's are exactly
    the OT-coupled support, so this is the cleanest possible gradient-flow input."""
    jko = mfmm.JKO1D(model, M=M)
    Q0 = jko.quantile_of_mixture([(0.5, -2.0, 0.2), (0.5, 2.0, 0.2)])
    Qs = jko.flow(Q0, tau=tau, n_steps=n_steps)
    return [Q.reshape(-1, 1) for Q in Qs], tau


def translation_control(model: mfmm.Model, n_steps=40, dt=5e-2, speed=1.0, M=240):
    """[NEG] Rigid translation of a fixed Gaussian away from the origin. In 1-D
    there is no rotational (orthogonal) motion, so the sharpest negative control is
    ANTI-gradient: the centre marches uphill in the confining potential, giving
    cos ~ -1 and a Lyapunov rate ~ 0 (energy INCREASES every step). The orthogonal
    cos~0 case is the 2-D rotation demo in gradient_flow_alignment.py."""
    jko = mfmm.JKO1D(model, M=M)
    base = jko.quantile_of_gaussian(0.0, model.sigma2_inf)
    snaps = []
    for k in range(n_steps + 1):
        snaps.append((base + speed * dt * k).reshape(-1, 1))
    return snaps, dt


def regime_shift_stub(n_steps=40, seg=8, M=240, seed=0):
    """Stand-in for the repo's A-case distribution shifts: piecewise-stationary
    samples with abrupt regime jumps (mean/scale), the way changepoint benchmarks
    are built. Used only for the in-container demo; pass --case/--npz for real data.
    Abrupt non-gradient jumps should give LOW alignment and a poor Lyapunov rate."""
    rng = np.random.default_rng(seed)
    regimes = [(-1.5, 0.5), (1.0, 1.2), (0.0, 0.3), (2.0, 0.8), (-0.5, 1.5)]
    snaps = []
    for k in range(n_steps + 1):
        mu, sd = regimes[(k // seg) % len(regimes)]
        snaps.append(rng.normal(mu, sd, size=(M, 1)))
    return snaps


# --------------------------------------------------------------------------- #
# Structure-light cross-checks (Section 5)
# --------------------------------------------------------------------------- #
def gibbs_check(last_snap: np.ndarray, model: mfmm.Model) -> dict:
    """Prop. 7 / Thm. 8: stationary law should be the Gibbs density N(0, beta/(a+b))."""
    x = np.asarray(last_snap).ravel()
    mu, var = float(np.mean(x)), float(np.var(x))
    z = (x - 0.0) / np.sqrt(model.sigma2_inf)        # standardize by predicted eq. scale
    ks = stats.kstest(z, "norm")
    return dict(
        mean=mu, var=var, predicted_var=model.sigma2_inf,
        var_rel_err=abs(var - model.sigma2_inf) / model.sigma2_inf,
        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
    )


def summarize(name, snaps, energy, model, dt):
    rep = evaluate_gradient_flow_alignment(snaps, energy, dt=dt)
    s = rep.summary()
    g = gibbs_check(snaps[-1], model)
    return {
        "sequence": name,
        "n_steps": s["n_steps"],
        "mean_cos_theta": round(s["mean_cos_theta"], 3),
        "grad_fraction": round(s["mean_grad_fraction"], 3),
        "residual_fraction": round(1 - s["mean_grad_fraction"], 3),
        "lyapunov_decrease_rate": round(s["lyapunov_decrease_rate"], 3),
        "stationary_var": round(g["var"], 3),
        "gibbs_var_rel_err": round(g["var_rel_err"], 3),
        "gibbs_ks_p": round(g["ks_p"], 3),
    }


def print_table(rows):
    cols = ["sequence", "n_steps", "mean_cos_theta", "grad_fraction",
            "residual_fraction", "lyapunov_decrease_rate",
            "stationary_var", "gibbs_var_rel_err", "gibbs_ks_p"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--case", type=str, default=None, help="repo synthetic case e.g. A3 (PYTHONPATH=src)")
    p.add_argument("--npz", type=str, default=None, help="path to .npz of empirical distributions")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model = mfmm.Model(a=args.a, b=args.b, beta=args.beta)
    energy = energy_from_model(model)
    print(f"Energy: V=(a/2)x^2, W=(b/2)(x-y)^2, beta-entropy  with a={model.a} b={model.b} "
          f"beta={model.beta}  ->  predicted equilibrium N(0, {model.sigma2_inf:.3f})\n")

    rows = []

    # [POS] true JKO iterates
    pos_snaps, pos_dt = jko_snapshots(model)
    rows.append(summarize("[POS] true JKO flow", pos_snaps, energy, model, pos_dt))

    # [NEG] rigid translation
    neg_snaps, neg_dt = translation_control(model)
    rows.append(summarize("[NEG] translation", neg_snaps, energy, model, neg_dt))

    # [DATA] real repo data if requested, else a repo-style stand-in
    if args.case:
        data_snaps = snapshots_from_repo_case(args.case, args.window, args.stride, seed=0)
        # collapse to 1-D (paper model is 1-D); use first coordinate
        data_snaps = [np.asarray(s)[:, :1] for s in data_snaps]
        rows.append(summarize(f"[DATA] repo {args.case}", data_snaps, energy, model, dt=1.0))
    elif args.npz:
        data_snaps = snapshots_from_npz(args.npz)
        data_snaps = [np.asarray(s)[:, :1] for s in data_snaps]
        rows.append(summarize("[DATA] npz", data_snaps, energy, model, dt=1.0))
    else:
        data_snaps = regime_shift_stub()
        rows.append(summarize("[DATA] regime-shift stub", data_snaps, energy, model, dt=1.0))

    print_table(rows)
    print("\nReading:")
    print("  cos~+1, grad_fraction~1, lyapunov~1  => motion is ALONG the JKO iterate of F (downhill).")
    print("  cos~-1, lyapunov~0                    => ANTI-gradient (uphill); |cos| large but sign +")
    print("                                          Lyapunov reveal it opposes F.")
    print("  cos~0                                 => orthogonal / rotational (non-equilibrium circulation).")
    print("  The [DATA] row's grad_fraction (=cos^2) is 'how gradient-like'; read it WITH the cos sign")
    print("  and lyapunov rate. residual_fraction = 1-cos^2 is the non-gradient part to report.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

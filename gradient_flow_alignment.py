"""
gradient_flow_alignment.py
==========================

Test whether a sequence of empirical distributions moves *parallel to the JKO
iterates* of a specified free-energy functional F.

Why this is the right test (and not one-step JKO matching)
----------------------------------------------------------
A JKO step is  m_{k+1} = argmin_m  F(m) + W2(m, m_k)^2 / (2 tau).
As tau -> 0 the move is a step along the Wasserstein gradient  -grad (dF/dm),
i.e. JKO iterates trace the gradient-flow curve. "Are the data parallel to the
JKO iterates?" therefore reduces to a *direction* question:

    does the observed transport velocity v_obs between m_k and m_{k+1}
    point along the predicted velocity  v_pred(x) = -grad (dF/dm)(x) ?

The cosine of the angle between them is invariant to the step size tau (a
reparametrised gradient flow  xdot = -g(x) grad f  traces the same curve), so
we never have to estimate tau. cos = 1 means "exactly parallel to the JKO
iterate"; cos < 1 quantifies the non-gradient (rotational / non-equilibrium)
part of the motion. This is the honest, continuous diagnostic: report the
gradient fraction, not a binary pass/fail.

The energy functional
----------------------
The repo ships OT distances but NO energy functional, so we specify one here:

    F(m) = ∫ V(x) m(dx)                         (risk / inventory potential)
         + (1/2) ∬ W(x,y) m(dx) m(dy)           (interaction / dispersion compression)
         + beta ∫ m log m  dx                    (entropy / dispersion)

First variation and Wasserstein gradient (velocity field):

    dF/dm (x)        = V(x) + ∫ W(x,y) m(dy) + beta (log m(x) + 1)
    v_pred(x) = -grad dF/dm (x)
              = -gradV(x) - ∫ grad_x W(x,y) m(dy) - beta grad log m(x)

The three terms are estimated from the samples of m_k:
  * -gradV(x_i)                      : closed form from the chosen potential
  * -mean_j grad_x W(x_i, x_j)       : Monte-Carlo over the m_k sample
  * -beta score(x_i)                 : Gaussian-KDE score grad log m_hat(x_i)

Observed velocity
-----------------
From the OT map T_k : m_k -> m_{k+1},  v_obs(x_i) = (T_k(x_i) - x_i) / dt.
T_k is the exact EMD barycentric projection when POT is installed, else the
optimal assignment (scipy linear_sum_assignment) for equal sample sizes.

What it reports, per consecutive pair
-------------------------------------
  cos_theta        : alignment of v_obs with v_pred in L^2(m_k)   (the "parallel" score)
  grad_fraction    : cos_theta**2  = fraction of observed motion explained by the gradient field
  residual_fraction: 1 - cos_theta**2  = non-gradient (rotational) share
  dF               : F(m_{k+1}) - F(m_k); a gradient flow needs dF <= 0 (Lyapunov test)

Usage
-----
    # Self-check: a true gradient flow should score ~1, a pure rotation ~0
    python gradient_flow_alignment.py --demo

    # Against the repo's synthetic A-cases (run from repo root, PYTHONPATH=src)
    PYTHONPATH=src python gradient_flow_alignment.py --case A3 --window 60 --stride 20

    # Against your own sequence of empirical distributions saved as a .npz
    #   snapshots: object array of (n_t, d) arrays, OR a (T, n, d) array
    python gradient_flow_alignment.py --npz my_distributions.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:  # exact OT is nicer when available; repo treats POT as optional too
    import ot  # type: ignore

    _HAVE_POT = True
except Exception:  # pragma: no cover
    ot = None
    _HAVE_POT = False


# --------------------------------------------------------------------------- #
# Energy functional
# --------------------------------------------------------------------------- #
@dataclass
class EnergyFunctional:
    """F(m) = ∫V m + 1/2 ∬W m⊗m + beta ∫ m log m.

    Parameters
    ----------
    a : potential strength.  V(x) = 0.5 * a * ||x||^2,  gradV = a x.  a=0 disables.
    kappa : quadratic interaction strength.  W(x,y)=0.5*kappa*||x-y||^2.
            grad_x W = kappa (x - y); the population term becomes kappa (x - mean).
            kappa>0 penalizes dispersion and compresses equilibrium variance. 0 disables.
    repulse_g, repulse_ell : optional Gaussian repulsion added to W:
            W += g * exp(-||x-y||^2 / (2 ell^2)).  g=0 disables.
    beta : entropy weight (paper convention, eq. 5):  + beta ∫ m log m.
           Larger beta = more entropic dispersion; equilibrium variance is
           beta/(a+b) for the quadratic model. None (or inf) disables the term.
    kde_bandwidth : override KDE bandwidth for the score / entropy estimate.
                    None -> Silverman's rule.
    """

    a: float = 1.0
    kappa: float = 0.0
    repulse_g: float = 0.0
    repulse_ell: float = 1.0
    beta: float | None = None
    kde_bandwidth: float | None = None

    # -- pieces of the free energy (for the Lyapunov / monotone-decrease test) --
    def _silverman_h(self, X: np.ndarray) -> float:
        n, d = X.shape
        std = np.sqrt(np.mean(np.var(X, axis=0)) + 1e-12)
        return float(self.kde_bandwidth or std * n ** (-1.0 / (d + 4)) + 1e-9)

    def value(self, X: np.ndarray) -> float:
        """Monte-Carlo estimate of F(m_hat) on samples X (n, d)."""
        X = np.atleast_2d(X)
        n = X.shape[0]
        f = 0.0
        if self.a:
            f += self.a * 0.5 * float(np.mean(np.sum(X * X, axis=1)))
        if self.kappa or self.repulse_g:
            D2 = cdist(X, X, "sqeuclidean")
            if self.kappa:
                f += 0.5 * self.kappa * 0.5 * float(np.mean(D2))
            if self.repulse_g:
                f += 0.5 * self.repulse_g * float(
                    np.mean(np.exp(-D2 / (2 * self.repulse_ell ** 2)))
                )
        if self.beta not in (None, np.inf):
            # paper eq.(5): + beta ∫ m log m ≈ beta (1/n) Σ log m_hat(x_i)
            f += self.beta * float(np.mean(self._log_density(X, X)))
        return f

    # -- gradient pieces used to build the predicted velocity field -------------
    def _log_density(self, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """log of Gaussian-KDE density of X, evaluated at points Q."""
        h = self._silverman_h(X)
        n, d = X.shape
        D2 = cdist(Q, X, "sqeuclidean")
        logk = -D2 / (2 * h * h) - d * np.log(h) - 0.5 * d * np.log(2 * np.pi)
        # log mean over kernels = logsumexp - log n
        m = np.max(logk, axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.sum(np.exp(logk - m), axis=1))
        return lse - np.log(n)

    def _score(self, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """grad_x log m_hat(x): weighted mean-shift of the Gaussian KDE."""
        h = self._silverman_h(X)
        diff = Q[:, None, :] - X[None, :, :]          # (q, n, d)
        D2 = np.sum(diff * diff, axis=2)              # (q, n)
        logw = -D2 / (2 * h * h)
        logw -= np.max(logw, axis=1, keepdims=True)
        w = np.exp(logw)
        w /= np.sum(w, axis=1, keepdims=True)         # (q, n) softmax weights
        # grad log p = -(1/h^2) Σ w_j (x - x_j)
        return -(1.0 / (h * h)) * np.einsum("qn,qnd->qd", w, diff)

    def predicted_velocity(self, X: np.ndarray) -> np.ndarray:
        """v_pred(x_i) = -grad (dF/dm)(x_i), evaluated on the m_k sample X."""
        X = np.atleast_2d(X)
        v = np.zeros_like(X, dtype=float)
        if self.a:
            v -= self.a * X                                   # -gradV
        if self.kappa:
            v -= self.kappa * (X - np.mean(X, axis=0, keepdims=True))
        if self.repulse_g:
            diff = X[:, None, :] - X[None, :, :]              # (n,n,d)
            D2 = np.sum(diff * diff, axis=2)
            ker = np.exp(-D2 / (2 * self.repulse_ell ** 2))   # (n,n)
            # grad_x W = -g (x-y)/ell^2 * ker  ->  -mean_y grad_x W
            gradW = -(self.repulse_g / self.repulse_ell ** 2) * np.einsum(
                "ij,ijd->id", ker, diff
            ) / X.shape[0]
            v -= gradW
        if self.beta not in (None, np.inf):
            v -= self.beta * self._score(X, X)        # entropy velocity = -beta grad log m
        return v


# --------------------------------------------------------------------------- #
# Observed transport velocity
# --------------------------------------------------------------------------- #
def ot_map(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Barycentric OT map T(x_i) from samples X (n,d) to Y (m,d).

    Uses POT exact EMD if available; otherwise falls back to the optimal
    assignment, which requires equal sample sizes.
    """
    X = np.atleast_2d(X).astype(float)
    Y = np.atleast_2d(Y).astype(float)
    n, m = X.shape[0], Y.shape[0]
    C = cdist(X, Y, "sqeuclidean")
    if _HAVE_POT:
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        G = ot.emd(a, b, C)                       # (n, m) transport plan
        row = G.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return (G @ Y) / row                      # barycentric projection
    if n != m:
        raise RuntimeError(
            "Optimal-assignment fallback needs equal sample sizes "
            f"(got {n} vs {m}); install POT (`pip install pot`) for unequal sizes."
        )
    ri, ci = linear_sum_assignment(C)
    T = np.empty_like(X)
    T[ri] = Y[ci]
    return T


def observed_velocity(X: np.ndarray, Y: np.ndarray, dt: float = 1.0) -> np.ndarray:
    return (ot_map(X, Y) - X) / dt


# --------------------------------------------------------------------------- #
# Alignment diagnostics
# --------------------------------------------------------------------------- #
def l2m_inner(u: np.ndarray, v: np.ndarray) -> float:
    """<u, v>_{L^2(m_hat)} with equal-weight empirical measure on the support."""
    return float(np.mean(np.sum(u * v, axis=1)))


def alignment(v_obs: np.ndarray, v_pred: np.ndarray) -> dict:
    nu = np.sqrt(l2m_inner(v_obs, v_obs))
    nv = np.sqrt(l2m_inner(v_pred, v_pred))
    if nu < 1e-12 or nv < 1e-12:
        cos = float("nan")
    else:
        cos = l2m_inner(v_obs, v_pred) / (nu * nv)
        cos = float(np.clip(cos, -1.0, 1.0))
    grad_frac = cos * cos if np.isfinite(cos) else float("nan")
    return {
        "cos_theta": cos,
        "grad_fraction": grad_frac,
        "residual_fraction": (1.0 - grad_frac) if np.isfinite(grad_frac) else float("nan"),
        "v_obs_norm": nu,
        "v_pred_norm": nv,
    }


@dataclass
class AlignmentReport:
    per_step: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        cos = np.array([s["cos_theta"] for s in self.per_step], float)
        gf = np.array([s["grad_fraction"] for s in self.per_step], float)
        dF = np.array([s["dF"] for s in self.per_step], float)
        cos = cos[np.isfinite(cos)]
        gf = gf[np.isfinite(gf)]
        dF = dF[np.isfinite(dF)]
        return {
            "n_steps": len(self.per_step),
            "mean_cos_theta": float(np.mean(cos)) if cos.size else float("nan"),
            "median_cos_theta": float(np.median(cos)) if cos.size else float("nan"),
            "mean_grad_fraction": float(np.mean(gf)) if gf.size else float("nan"),
            "lyapunov_decrease_rate": float(np.mean(dF <= 0)) if dF.size else float("nan"),
            "mean_dF": float(np.mean(dF)) if dF.size else float("nan"),
            "have_pot": _HAVE_POT,
        }


def evaluate_gradient_flow_alignment(
    snapshots: Sequence[np.ndarray],
    energy: EnergyFunctional,
    dt: float = 1.0,
) -> AlignmentReport:
    """Core entry point: given a time-ordered list of empirical distributions
    (each an (n_t, d) array of samples), test alignment with -grad(dF/dm)."""
    rep = AlignmentReport()
    for k in range(len(snapshots) - 1):
        Xk = np.atleast_2d(snapshots[k]).astype(float)
        Xk1 = np.atleast_2d(snapshots[k + 1]).astype(float)
        v_obs = observed_velocity(Xk, Xk1, dt=dt)
        v_pred = energy.predicted_velocity(Xk)
        row = alignment(v_obs, v_pred)
        row["step"] = k
        row["dF"] = energy.value(Xk1) - energy.value(Xk)
        rep.per_step.append(row)
    return rep


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def snapshots_from_repo_case(case: str, window: int, stride: int, seed: int):
    """Build a distribution sequence from the repo's synthetic generators by
    sliding a window over a generated A-case series. Requires PYTHONPATH=src."""
    from changept_detection.experiments import synthetic as S  # type: ignore

    fn = getattr(S, f"generate_{case.lower()}", None) or getattr(S, f"generate_case_{case.lower()}", None)
    if fn is None:
        # Fall back to a generic case runner if individual generators differ.
        raise RuntimeError(
            f"Could not find a generator for case {case} in experiments.synthetic. "
            "Open that module and pass the produced (T, d) array to "
            "snapshots_from_series() instead."
        )
    out = fn(seed=seed)
    series = out[0] if isinstance(out, tuple) else out
    return snapshots_from_series(np.asarray(series), window=window, stride=stride)


def snapshots_from_series(series: np.ndarray, window: int, stride: int):
    """Slide a window over a (T, d) time series to make empirical distributions."""
    series = np.atleast_2d(series)
    if series.shape[0] < series.shape[1]:
        series = series  # leave as-is; assume rows are time
    T = series.shape[0]
    snaps = []
    s = 0
    while s + window <= T:
        snaps.append(series[s : s + window])
        s += stride
    if len(snaps) < 2:
        raise RuntimeError("Window/stride too large: need at least two snapshots.")
    return snaps


def snapshots_from_npz(path: str):
    data = np.load(path, allow_pickle=True)
    key = "snapshots" if "snapshots" in data else data.files[0]
    arr = data[key]
    if arr.dtype == object:
        return [np.atleast_2d(a).astype(float) for a in arr]
    arr = np.asarray(arr)
    if arr.ndim == 3:  # (T, n, d)
        return [arr[t] for t in range(arr.shape[0])]
    raise RuntimeError("npz must hold an object array of (n,d) arrays or a (T,n,d) array.")


# --------------------------------------------------------------------------- #
# Self-check demo: a true gradient flow vs a pure rotation
# --------------------------------------------------------------------------- #
def _demo(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d, steps = 600, 2, 12
    a, beta, dt = 1.0, 1.0, 0.05

    # (1) DETERMINISTIC gradient flow of F = 0.5 a||x||^2 + beta * entropy.
    #     For an isotropic Gaussian m=N(0,s^2 I), grad log m = -x/s^2, so the
    #     density velocity is v(x) = (-a + beta/s^2) x. Moving each particle by this
    #     velocity keeps m Gaussian and makes particle motion EQUAL the density
    #     velocity. Isolates math + KDE score term: cos ~ 1.
    X = rng.normal(0.0, 3.0, size=(n, d))
    det_snaps = [X.copy()]
    for _ in range(steps):
        s2 = float(np.mean(np.var(X, axis=0)))
        v = (-a + beta / s2) * X
        X = X + v * dt
        det_snaps.append(X.copy())

    # (2) Pure rotation: divergence-free, conserves the energy  =>  cos ~ 0, dF ~ 0.
    theta = 0.25
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = rng.normal(0.0, 2.0, size=(n, d))
    rot_snaps = [X.copy()]
    for _ in range(steps):
        X = X @ R.T
        rot_snaps.append(X.copy())

    # (3) Stochastic Langevin: SAME gradient flow at the density level, but each
    #     trajectory carries Brownian noise. With finite samples and the
    #     assignment-based OT fallback, v_obs is high-variance -> cos is pulled
    #     down even though the energy still decreases. POT's barycentric map and
    #     larger windows recover more of the signal. This is the real-data regime.
    X = rng.normal(0.0, 3.0, size=(n, d))
    lan_snaps = [X.copy()]
    for _ in range(steps):
        X = X - a * X * dt + np.sqrt(2.0 * beta * dt) * rng.normal(size=(n, d))
        lan_snaps.append(X.copy())

    energy = EnergyFunctional(a=a, kappa=0.0, beta=beta)
    print("=== DEMO: diagnostic sanity check ===")
    print(f"POT available: {_HAVE_POT}  (install `pot` to use exact-EMD barycentric maps)")
    print("\n[1 deterministic gradient flow]  expect cos~1, grad_fraction~1, Lyapunov~1.0")
    print(json.dumps(evaluate_gradient_flow_alignment(det_snaps, energy, dt=dt).summary(), indent=2))
    print("\n[2 pure rotation]                expect cos~0, grad_fraction~0, dF~0")
    print(json.dumps(evaluate_gradient_flow_alignment(rot_snaps, energy, dt=dt).summary(), indent=2))
    print("\n[3 stochastic Langevin]          density-level gradient flow seen through")
    print("                                 finite-sample OT noise: cos<1 but dF<0 reliably")
    print(json.dumps(evaluate_gradient_flow_alignment(lan_snaps, energy, dt=dt).summary(), indent=2))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="run the gradient-flow vs rotation sanity check")
    src.add_argument("--case", type=str, help="repo synthetic case, e.g. A3 (needs PYTHONPATH=src)")
    src.add_argument("--npz", type=str, help="path to .npz of empirical distributions")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=1.0)
    # energy params
    p.add_argument("--a", type=float, default=1.0, help="quadratic potential strength")
    p.add_argument("--kappa", type=float, default=0.0, help="quadratic crowding strength")
    p.add_argument("--repulse-g", type=float, default=0.0)
    p.add_argument("--repulse-ell", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=None, help="entropy weight beta; omit to disable")
    p.add_argument("--out", type=str, default=None, help="optional JSON output path")
    args = p.parse_args()

    if args.demo or not (args.case or args.npz):
        _demo(seed=args.seed)
        return

    energy = EnergyFunctional(
        a=args.a, kappa=args.kappa, repulse_g=args.repulse_g,
        repulse_ell=args.repulse_ell, beta=args.beta,
    )
    if args.case:
        snaps = snapshots_from_repo_case(args.case, args.window, args.stride, args.seed)
    else:
        snaps = snapshots_from_npz(args.npz)

    rep = evaluate_gradient_flow_alignment(snaps, energy, dt=args.dt)
    out = {"summary": rep.summary(), "per_step": rep.per_step}
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

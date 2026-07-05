"""
mfmm.py  --  Mean-Field Market Making: numerical core (paper Sections 2-4)
==========================================================================

Implements the one-dimensional inventory model of Chang, "Wasserstein Gradient
Flows for Mean-Field Market Making", with the quadratic specification

    V(x) = (a/2) x^2 ,   W(x,y) = (b/2)(x-y)^2 ,   entropy weight beta > 0,

free energy (paper eq. 12)

    F(rho) = (a/2) ∫ x^2 rho dx + (b/2) Var(rho) + beta ∫ rho log rho dx,

whose W2 gradient flow is the McKean-Vlasov-Fokker-Planck equation (paper eq. 13)

    ∂_t m = ∂_x( m [ (a+b)x - b mu_t ] ) + beta ∂_xx m ,   mu_t = ∫ x m dx,

i.e. transport velocity v(x) = -(a+b)x + b mu  with entropic diffusion beta.

What this module provides
-------------------------
  * Closed-form Gaussian ground truth (paper Theorem 8): equilibrium N(0, beta/(a+b)),
    mean law mu_t = mu_0 e^{-a t}, variance law Sigma_t.
  * Two Eulerian Fokker-Planck solvers on a grid (explicit + implicit, upwind +
    central diffusion, no-flux BCs).
  * The JKO proximal scheme in 1D quantile coordinates (paper Appendix B):
    each step is a strictly convex program over nondecreasing Q, solved by L-BFGS
    with analytic gradients. This is the scheme that yields the *JKO iterates*.
  * The three main model experiments and one solver-comparison ablation:
    against the closed form.

Run:
    python mfmm.py                      # verify against Theorem 8 + run all experiments
    python mfmm.py --make-figures       # generate figs/ + results/ artifacts
    python make_figures.py --all        # same as --make-figures
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parent
FIGS_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------- #
# Model + closed-form ground truth (paper Theorem 8)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    a: float = 1.0      # inventory-risk potential strength
    b: float = 0.5      # quadratic interaction; b>0 penalizes dispersion / compresses variance
    beta: float = 0.5   # entropic dispersion

    @property
    def sigma2_inf(self) -> float:
        return self.beta / (self.a + self.b)

    def mean_law(self, mu0: float, t: np.ndarray) -> np.ndarray:
        return mu0 * np.exp(-self.a * t)

    def var_law(self, Sigma0: float, t: np.ndarray) -> np.ndarray:
        s = self.sigma2_inf
        return s + (Sigma0 - s) * np.exp(-2.0 * (self.a + self.b) * t)


# --------------------------------------------------------------------------- #
# Grid utilities + free energy on a grid
# --------------------------------------------------------------------------- #
def make_grid(L: float = 6.0, N: int = 481):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    return x, dx


def normalize(m: np.ndarray, dx: float) -> np.ndarray:
    m = np.clip(m, 0.0, None)
    return m / (m.sum() * dx)


def gaussian_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def ensure_output_dirs() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def equilibrium_pdf(x: np.ndarray, model: Model) -> np.ndarray:
    return gaussian_pdf(x, 0.0, model.sigma2_inf)


def w2_distance_1d(m_a: np.ndarray, m_b: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Approximate W2 distance via quantile functions on a common u-grid."""
    cdf_a = np.cumsum(m_a) * dx
    cdf_b = np.cumsum(m_b) * dx
    u = np.linspace(0.0, 1.0, x.size)
    qa = np.interp(u, cdf_a, x)
    qb = np.interp(u, cdf_b, x)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def free_energy_grid(m: np.ndarray, x: np.ndarray, dx: float, model: Model) -> float:
    mu = float(np.sum(x * m) * dx)
    var = float(np.sum((x - mu) ** 2 * m) * dx)
    pot = model.a * 0.5 * float(np.sum(x ** 2 * m) * dx)
    inter = model.b * 0.5 * var
    mpos = np.clip(m, 1e-300, None)
    ent = model.beta * float(np.sum(mpos * np.log(mpos)) * dx)
    return pot + inter + ent


# --------------------------------------------------------------------------- #
# Eulerian Fokker-Planck solvers (paper Section 4 "Numerical methods")
# --------------------------------------------------------------------------- #
def _fp_flux_matrix(x, dx, beta, drift_coeff_at_mu, mu, sparse_format=False):
    """Build the spatial operator L (conservative, upwind advection + central
    diffusion, no-flux) such that dm/dt = L @ m, with velocity u(x) = -coeff(x)+...,
    here u(x) = -((a+b)x - b mu). Returns dense (N,N) or CSR sparse matrix."""
    N = x.size
    u = -drift_coeff_at_mu(x, mu)
    rows, cols, data = [], [], []
    L_dense = np.zeros((N, N)) if not sparse_format else None

    def add(i, j, val):
        if sparse_format:
            rows.append(i)
            cols.append(j)
            data.append(val)
        else:
            L_dense[i, j] += val

    for i in range(N - 1):
        uf = 0.5 * (u[i] + u[i + 1])
        if uf >= 0:
            adv_i, adv_ip = uf, 0.0
        else:
            adv_i, adv_ip = 0.0, uf
        d = beta / dx
        add(i, i, -adv_i / dx - d / dx)
        add(i, i + 1, -adv_ip / dx + d / dx)
        add(i + 1, i, adv_i / dx + d / dx)
        add(i + 1, i + 1, adv_ip / dx - d / dx)

    if sparse_format:
        return sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return L_dense


def _drift_coeff(x, mu, model: Model):
    return (model.a + model.b) * x - model.b * mu


def fokker_planck(
    m0,
    x,
    dx,
    model: Model,
    T: float,
    dt: float,
    scheme: str = "implicit",
    snapshot_times: tuple[float, ...] | None = None,
    drift_offset=None,
):
    """Evolve (13) on the grid. scheme in {'explicit','implicit'} (semi-implicit:
    mu frozen at previous step). Returns (m_T, history) with history of (t, energy,
    mean, var, min_mass, mass). Optional snapshot_times stores densities in history."""
    m = normalize(m0.copy(), dx)
    n_steps = int(round(T / dt))
    hist = []
    snapshots = {}
    pending = sorted(snapshot_times or ())
    diverged = False
    drift_fn = drift_offset or (lambda xx, mm: _drift_coeff(xx, mm, model))

    for n in range(n_steps):
        mu = float(np.sum(x * m) * dx)
        use_sparse = scheme == "implicit"
        L = _fp_flux_matrix(x, dx, model.beta, drift_fn, mu, sparse_format=use_sparse)
        if scheme == "explicit":
            m_new = m + dt * (L @ m)
        elif scheme == "implicit":
            A = sparse.eye(x.size, format="csr") - dt * L
            m_new = spsolve(A, m)
        else:
            raise ValueError(scheme)
        mass = float(m_new.sum() * dx)
        min_mass = float(m_new.min())
        neg_mass = float(np.sum(np.minimum(m_new, 0.0)) * dx)
        if not np.isfinite(m_new).all() or abs(mass) > 1e3:
            diverged = True
        t = (n + 1) * dt
        mu_n = float(np.sum(x * np.clip(m_new, 0, None)) * dx) / max(mass, 1e-12)
        var_n = float(np.sum((x - mu_n) ** 2 * np.clip(m_new, 0, None)) * dx) / max(mass, 1e-12)
        energy = free_energy_grid(normalize(m_new, dx), x, dx, model) if not diverged else np.inf
        row = dict(t=t, energy=energy, mean=mu_n, var=var_n, min_mass=min_mass,
                   neg_mass=neg_mass, mass=mass)
        hist.append(row)
        while pending and t >= pending[0] - 1e-12:
            snapshots[pending[0]] = normalize(m_new.copy(), dx) if not diverged else m_new.copy()
            pending.pop(0)
        if diverged:
            return m_new, hist, snapshots
        m = m_new if scheme == "explicit" else normalize(m_new, dx)
    return m, hist, snapshots


# --------------------------------------------------------------------------- #
# JKO scheme in quantile coordinates (paper Appendix B)
# --------------------------------------------------------------------------- #
class JKO1D:
    """One-dimensional JKO solver. State is the quantile function Q on M midpoint
    nodes u_i=(i-0.5)/M. Monotonicity is enforced by Q_i = z_1 + sum_{j<=i} exp(z_j).
    Each step minimizes  F[Q] + (1/2tau) ||Q - Q_k||^2_{L2(0,1)}  by L-BFGS."""

    def __init__(self, model: Model, M: int = 240):
        self.model = model
        self.M = M
        self.du = 1.0 / M
        self.u = (np.arange(M) + 0.5) / M

    # ---- parametrization z -> Q ----
    @staticmethod
    def z_to_Q(z):
        Q = np.empty_like(z)
        inc = np.empty_like(z)
        inc[0] = 0.0
        inc[1:] = np.exp(z[1:])
        Q[0] = z[0]
        Q[1:] = z[0] + np.cumsum(inc[1:])
        return Q, inc

    def Q_to_z(self, Q):
        z = np.empty_like(Q)
        z[0] = Q[0]
        d = np.diff(Q)
        d = np.clip(d, 1e-9, None)
        z[1:] = np.log(d)
        return z

    # ---- energy F[Q] and proximal objective ----
    def energy(self, Q):
        a, b, beta = self.model.a, self.model.b, self.model.beta
        du = self.du
        m1 = np.sum(Q) * du
        m2 = np.sum(Q ** 2) * du
        pot = 0.5 * a * m2
        inter = 0.5 * b * (m2 - m1 ** 2)
        slope = np.diff(Q) / du                       # M-1 slopes
        ent = -beta * np.sum(np.log(np.clip(slope, 1e-12, None))) * du
        return pot + inter + ent

    def _objective(self, z, Qk, tau):
        a, b, beta = self.model.a, self.model.b, self.model.beta
        du = self.du
        Q, inc = self.z_to_Q(z)
        m1 = np.sum(Q) * du
        # value
        m2 = np.sum(Q ** 2) * du
        pot = 0.5 * a * m2
        inter = 0.5 * b * (m2 - m1 ** 2)
        slope = np.diff(Q) / du
        ent = -beta * np.sum(np.log(slope)) * du
        prox = (0.5 / tau) * np.sum((Q - Qk) ** 2) * du
        J = pot + inter + ent + prox
        # gradient wrt Q (potential + interaction + prox); entropy handled in z
        Gpot = a * Q * du
        Ginter = b * (Q - m1) * du
        Gprox = (1.0 / tau) * (Q - Qk) * du
        G = Gpot + Ginter + Gprox                     # dJ/dQ_i  (excl. entropy)
        # chain rule dQ_i/dz_j: z_1 shifts all; z_j (j>=2) scales suffix by exp(z_j)
        grad = np.empty_like(z)
        suffix = np.cumsum(G[::-1])[::-1]             # suffix[i] = sum_{k>=i} G_k
        grad[0] = np.sum(G)                           # dJ/dz_1
        grad[1:] = inc[1:] * suffix[1:]               # dJ/dz_j via Q part
        # entropy gradient wrt z: ent = -beta*du*sum_{j>=1} log(inc[j+1]/du)
        # d/dz_j of -beta*du*log(exp(z_j)/du) = -beta*du, for j>=1 (slopes use inc[1:])
        grad[1:] += -beta * du
        return J, grad

    def step(self, Qk, tau):
        z0 = self.Q_to_z(Qk)
        res = minimize(
            lambda z: self._objective(z, Qk, tau),
            z0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=500, ftol=1e-12, gtol=1e-9),
        )
        Q, _ = self.z_to_Q(res.x)
        return Q

    def flow(self, Q0, tau, n_steps):
        Qs = [Q0.copy()]
        Q = Q0.copy()
        for _ in range(n_steps):
            Q = self.step(Q, tau)
            Qs.append(Q.copy())
        return Qs

    # ---- quantile <-> samples / density helpers ----
    def quantile_of_gaussian(self, mu, var):
        from scipy.stats import norm
        return mu + np.sqrt(var) * norm.ppf(self.u)

    def quantile_of_mixture(self, comps):
        """comps: list of (weight, mu, sigma). Build Q by inverting the mixture CDF
        on the u-grid via sampling + empirical quantiles (simple, robust)."""
        rng = np.random.default_rng(0)
        N = 200000
        ws = np.array([w for w, _, _ in comps]); ws = ws / ws.sum()
        idx = rng.choice(len(comps), size=N, p=ws)
        samp = np.array([rng.normal(comps[i][1], comps[i][2]) for i in idx])
        return np.quantile(samp, self.u)

    @staticmethod
    def moments(Q, du=None):
        du = du if du is not None else 1.0 / Q.size
        mu = np.sum(Q) * du
        var = np.sum(Q ** 2) * du - mu ** 2
        return mu, var

    def samples(self, Q):
        """Return Q itself as M equiprobable samples of the distribution."""
        return Q.copy()


# --------------------------------------------------------------------------- #
# Section 4 experiments
# --------------------------------------------------------------------------- #
def bimodal_initial(x: np.ndarray) -> np.ndarray:
    return 0.5 * gaussian_pdf(x, -2, 0.2 ** 2) + 0.5 * gaussian_pdf(x, 2, 0.2 ** 2)


def exp1_relaxation(model=Model(), T=4.0, dt=2e-3):
    x, dx = make_grid(6.0, 481)
    m0 = bimodal_initial(x)
    _, hist, _ = fokker_planck(m0, x, dx, model, T, dt, scheme="implicit")
    var_T = hist[-1]["var"]
    energy = np.array([h["energy"] for h in hist])
    monotone = bool(np.all(np.diff(energy) <= 1e-9))
    return dict(var_T=var_T, sigma2_inf=model.sigma2_inf,
                energy_monotone=monotone, n_energy_increases=int(np.sum(np.diff(energy) > 1e-9)))


def run_exp1(model=Model(), T=4.0, dt=2e-3):
    x, dx = make_grid(6.0, 481)
    m0 = bimodal_initial(x)
    snap_times = (0.0, 0.5, 1.0, 2.0, 4.0)
    m_T, hist, snaps = fokker_planck(
        m0, x, dx, model, T, dt, scheme="implicit", snapshot_times=snap_times
    )
    snaps[0.0] = normalize(m0.copy(), dx)
    m_inf = equilibrium_pdf(x, model)
    f_inf = free_energy_grid(m_inf, x, dx, model)
    sigma0 = float(np.sum((x - 0.0) ** 2 * snaps[0.0]) * dx)
    t_arr = np.array([h["t"] for h in hist])
    var_arr = np.array([h["var"] for h in hist])
    energy = np.array([h["energy"] for h in hist])
    mass = np.array([h["mass"] for h in hist])
    min_density = np.array([h["min_mass"] for h in hist])
    var_theory = model.var_law(sigma0, t_arr)
    energy_increases = int(np.sum(np.diff(energy) > 1e-9))
    metrics = dict(
        a=model.a, b=model.b, beta=model.beta,
        target_variance=model.sigma2_inf,
        terminal_variance=float(hist[-1]["var"]),
        variance_abs_error=abs(hist[-1]["var"] - model.sigma2_inf),
        energy_increase_steps=energy_increases,
        mass_error_max=float(np.max(np.abs(mass - 1.0))),
        min_density=float(np.min(min_density)),
    )
    return dict(
        x=x, dx=dx, model=model, hist=hist, snaps=snaps, m_inf=m_inf, f_inf=f_inf,
        t_arr=t_arr, var_arr=var_arr, var_theory=var_theory, energy=energy,
        metrics=metrics,
    )


def exp2_shock(model=Model(), T=8.0, dt=2e-3, c=2.0, window=(2.0, 4.0)):
    data = run_exp2(model, T=T, dt=dt, c=c, window=window)
    return dict(
        mean_at_shock_end=data["metrics"]["mean_at_shock_end"],
        quasi_static_pred=data["metrics"]["quasi_static_prediction"],
    )


def run_exp2(model=Model(), T=8.0, dt=2e-3, c=2.0, window=(2.0, 4.0)):
    x, dx = make_grid(6.0, 481)
    m = normalize(equilibrium_pdf(x, model), dx)
    m_eq = m.copy()
    hist = []
    # (time, short label for figure legend)
    snap_targets = ((1.0, "before shock"), (3.0, "during shock"), (6.0, "after shock"))
    snaps = {0.0: m_eq.copy()}
    snap_labels = {0.0: r"start ($t=0$, at equilibrium)"}
    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t = n * dt
        ct = c if (window[0] <= t < window[1]) else 0.0
        mu = float(np.sum(x * m) * dx)
        drift = lambda xx, mm, cc=ct: (model.a + model.b) * xx - model.b * mm - model.a * cc
        L = _fp_flux_matrix(x, dx, model.beta, drift, mu, sparse_format=True)
        A = sparse.eye(x.size, format="csr") - dt * L
        m = normalize(spsolve(A, m), dx)
        t_next = t + dt
        mean = float(np.sum(x * m) * dx)
        w2 = w2_distance_1d(m, m_eq, x, dx)
        hist.append(dict(t=t_next, mean=mean, w2_to_eq=w2))
        for st, lbl in snap_targets:
            if abs(t_next - st) < dt * 0.51:
                snaps[st] = m.copy()
                snap_labels[st] = lbl
    hist_arr = np.array([(h["t"], h["mean"], h["w2_to_eq"]) for h in hist])
    idx_start = int(np.argmin(np.abs(hist_arr[:, 0] - window[0])))
    idx_end = int(np.argmin(np.abs(hist_arr[:, 0] - window[1])))
    quasi_static = c * (1 - np.exp(-model.a * (window[1] - window[0])))
    metrics = dict(
        shock_center=c,
        shock_window=list(window),
        mean_at_shock_start=float(hist_arr[idx_start, 1]),
        mean_at_shock_end=float(hist_arr[idx_end, 1]),
        quasi_static_prediction=float(quasi_static),
        shock_end_abs_error=abs(float(hist_arr[idx_end, 1]) - quasi_static),
        final_mean=float(hist_arr[-1, 1]),
        final_distance_to_equilibrium=float(hist_arr[-1, 2]),
    )
    return dict(
        x=x, dx=dx, model=model, hist=hist, hist_arr=hist_arr, snaps=snaps,
        snap_labels=snap_labels, window=window, shock_center=c, m_eq=m_eq, metrics=metrics,
    )


def exp3_crowding_sweep(model=Model(), bs=np.linspace(0, 3, 13), T=6.0, dt=2e-3):
    data = run_exp3(model, bs=bs, T=T, dt=dt)
    return dict(table=data["table"], max_abs_err=data["metrics"]["max_abs_error"])


def run_exp3(model=Model(), bs=np.linspace(0, 3, 13), T=6.0, dt=2e-3):
    x, dx = make_grid(6.0, 481)
    rows = []
    rep_densities = {}
    rep_relaxation = {}
    m0_raw = gaussian_pdf(x, 0.0, 1.0)
    m0 = normalize(m0_raw, dx)
    mid_t = T / 2.0
    for b in bs:
        mdl = Model(a=model.a, b=float(b), beta=model.beta)
        m_T, hist, _ = fokker_planck(m0_raw, x, dx, mdl, T, dt, scheme="implicit")
        num_var = hist[-1]["var"]
        theo_var = mdl.sigma2_inf
        abs_err = abs(num_var - theo_var)
        rel_err = abs_err / theo_var if theo_var > 0 else abs_err
        rows.append((float(b), num_var, theo_var, abs_err, rel_err))
        if float(b) in (0.0, 1.0, 3.0):
            rep_densities[float(b)] = m_T

    for b in (0.0, 1.0, 3.0):
        mdl = Model(a=model.a, b=b, beta=model.beta)
        _, _, snaps = fokker_planck(
            m0_raw, x, dx, mdl, T, dt, scheme="implicit",
            snapshot_times=(mid_t, T),
        )
        rep_relaxation[b] = dict(
            m0=m0,
            mid=snaps.get(mid_t),
            final=snaps.get(T),
            m_inf=equilibrium_pdf(x, mdl),
        )

    table = np.array(rows)
    abs_errors = table[:, 3]
    rel_errors = table[:, 4]
    num_vars = table[:, 1]
    metrics = dict(
        max_abs_error=float(np.max(abs_errors)),
        max_rel_error=float(np.max(rel_errors)),
        monotone_variance_decrease=bool(np.all(np.diff(num_vars) <= 1e-6)),
    )
    return dict(
        x=x, dx=dx, table=table, rep_densities=rep_densities,
        rep_relaxation=rep_relaxation, T=T, metrics=metrics,
    )


def exp4_solver_comparison(model=Model()):
    data = run_exp4(model)
    out = dict(data["summary"])
    out["_meta"] = data["meta"]
    return out


def run_exp4(model=Model()):
    x, dx = make_grid(6.0, 161)
    cfl = dx ** 2 / (2 * model.beta)
    dt = 1.8 * cfl
    tau = 5e-2
    m0 = bimodal_initial(x)
    T = 2.0
    summary = {}
    traces = {}
    terminal = {}

    for scheme in ("explicit", "implicit"):
        m_T, hist, _ = fokker_planck(m0, x, dx, model, T, dt, scheme=scheme)
        energy = np.array([h["energy"] for h in hist])
        min_mass = np.array([h["min_mass"] for h in hist])
        mass = np.array([h["mass"] for h in hist])
        t_arr = np.array([h["t"] for h in hist])
        diverged = not np.isfinite(energy).all() or np.any(np.abs(mass) > 1e3)
        neg_mass = np.array([h["neg_mass"] for h in hist])
        summary[scheme] = dict(
            stable=not diverged,
            energy_increases=int(np.sum(np.diff(energy[np.isfinite(energy)]) > 1e-9)),
            min_mass=float(np.min(min_mass)),
            mass_err=float(np.max(np.abs(mass - 1.0))) if np.isfinite(mass).all() else np.inf,
            eq_var=float(hist[-1]["var"]) if np.isfinite(energy[-1]) else np.nan,
        )
        traces[scheme] = dict(t=t_arr, energy=energy, neg_mass=neg_mass, mass_err=np.abs(mass - 1.0))
        terminal[scheme] = normalize(m_T, dx) if scheme == "implicit" else m_T

    jko = JKO1D(model, M=240)
    Q0 = jko.quantile_of_mixture([(0.5, -2, 0.2), (0.5, 2, 0.2)])
    n_jko = int(round(T / tau))
    Qs = jko.flow(Q0, tau=tau, n_steps=n_jko)
    energies = np.array([jko.energy(Q) for Q in Qs])
    t_jko = np.arange(len(Qs)) * tau
    _, var_jko = jko.moments(Qs[-1])
    summary["jko"] = dict(
        stable=True,
        energy_increases=int(np.sum(np.diff(energies) > 1e-9)),
        min_mass=0.0,
        mass_err=0.0,
        eq_var=float(var_jko),
    )
    traces["jko"] = dict(
        t=t_jko, energy=energies, neg_mass=np.zeros_like(energies),
        mass_err=np.zeros_like(energies),
    )
    samples = jko.samples(Qs[-1])
    hist_jko, edges = np.histogram(samples, bins=len(x) - 1, range=(x[0], x[-1]), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    terminal["jko"] = np.interp(x, centers, hist_jko, left=0.0, right=0.0)

    meta = dict(dx=dx, dt=dt, cfl=cfl, tau=tau, sigma2_inf=model.sigma2_inf, T=T)
    return dict(x=x, dx=dx, model=model, summary=summary, traces=traces,
                terminal=terminal, meta=meta)


# --------------------------------------------------------------------------- #
def verify():
    m = Model()
    print(f"Model a={m.a} b={m.b} beta={m.beta}  ->  sigma2_inf = beta/(a+b) = {m.sigma2_inf:.6f}")
    print("\n--- Exp 1: inventory relaxation (implicit FD) ---")
    r1 = exp1_relaxation(m)
    print(f"  terminal variance {r1['var_T']:.4f}  vs analytic {r1['sigma2_inf']:.4f}"
          f"  | energy monotone: {r1['energy_monotone']} (increases={r1['n_energy_increases']})")

    print("\n--- Exp 2: liquidity shock ---")
    r2 = exp2_shock(m)
    print(f"  mean at shock end {r2['mean_at_shock_end']:.3f}  vs quasi-static {r2['quasi_static_pred']:.3f}")

    print("\n--- Exp 3: crowding sweep (var vs beta/(a+b)) ---")
    r3 = exp3_crowding_sweep(m)
    print(f"  max |var_num - beta/(a+b)| over b in [0,3]: {r3['max_abs_err']:.4e}")
    for b, vn, va, _, _ in r3["table"][::3]:
        print(f"    b={b:.2f}  num={vn:.4f}  analytic={va:.4f}")

    print("\n--- Exp 4: solver comparison at large step (1.8x CFL) ---")
    r4 = exp4_solver_comparison(m)
    meta = r4.pop("_meta")
    print(f"  dx={meta['dx']:.3f} dt={meta['dt']:.2e} (CFL={meta['cfl']:.2e}) tau={meta['tau']:.2e}"
          f"  analytic var={meta['sigma2_inf']:.3f}")
    hdr = f"  {'method':10s} {'stable':7s} {'E-incr':7s} {'min_mass':>11s} {'mass_err':>11s} {'eq_var':>8s}"
    print(hdr)
    for k in ("explicit", "implicit", "jko"):
        d = r4[k]
        print(f"  {k:10s} {str(d['stable']):7s} {d['energy_increases']:<7d}"
              f" {d['min_mass']:>11.2e} {d['mass_err']:>11.2e} {d['eq_var']:>8.3f}")


def main():
    p = argparse.ArgumentParser(description="Mean-field market-making numerical experiments")
    p.add_argument("--make-figures", action="store_true", help="generate figs/ and results/ artifacts")
    p.add_argument("--verify", action="store_true", help="run quick verification summary")
    args = p.parse_args()
    if args.make_figures:
        from make_figures import generate_all
        generate_all()
    else:
        verify()


if __name__ == "__main__":
    main()

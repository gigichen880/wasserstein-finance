# Wasserstein Gradient Flows for Mean-Field Market Making

## Project overview

This project studies a stylized mean-field market-making model where the object of interest is not an asset price, return, or alpha signal, but the **distribution of dealer inventories** across a population of market makers.

The central question is whether a *stylized* inventory distribution can be
approximated as a Wasserstein gradient flow of an interpretable free energy,
and whether JKO is a stable way to compute that flow. That is three separate
claims: the quadratic model is internally correct; JKO is structure-preserving;
whether real market inventories follow \(F\) is open.

The numerics are a controlled mathematical experiment, not empirical evidence
about dealer inventories in markets. The energy lives on the distribution of
agent states.

---

## Intuition

Imagine many dealers each holding some signed inventory `x`:

- `x < 0`: dealer is short inventory.
- `x = 0`: dealer is flat.
- `x > 0`: dealer is long inventory.

The whole dealer population forms a probability distribution `rho(x)` over inventories.

The paper asks whether this inventory distribution behaves like a probability blob rolling downhill in Wasserstein space. The downhill direction is determined by a free-energy functional with three forces:

1. **Inventory/risk potential**: dealers dislike large absolute inventory and are pulled toward zero.
2. **Interaction / dispersion penalty**: with $W=\frac{b}{2}(x-y)^2$ and $b>0$, the population interaction compresses the spread of inventories (synchronization), it does not repel similar dealers.
3. **Entropy / dispersion**: diffusion prevents collapse and keeps the distribution smooth.

---

## Mathematical model

We use the one-dimensional inventory state `x` and the quadratic specification

```math
V(x) = \frac{a}{2}x^2,
\qquad
W(x,y) = \frac{b}{2}(x-y)^2,
\qquad
\beta > 0.
```

The free energy is

```math
F(\rho)
=
\frac{a}{2}\int x^2\rho(x)\,dx
+
\frac{b}{2}\operatorname{Var}(\rho)
+
\beta\int \rho(x)\log \rho(x)\,dx.
```

The corresponding Wasserstein gradient flow is the nonlinear Fokker--Planck / McKean--Vlasov equation

```math
\partial_t m_t
=
\partial_x\left(m_t\left[(a+b)x-b\mu_t\right]\right)
+
\beta\partial_{xx}m_t,
\qquad
\mu_t = \int x m_t(x)\,dx.
```

Equivalently, the transport velocity is

```math
v_t(x)=-(a+b)x+b\mu_t,
```

with entropy/diffusion controlled by `beta`.

---

## Meaning of parameters

The default model parameters are

```python
a = 1.0
b = 0.5
beta = 0.5
```

They do **not** determine the initial distribution. They determine the energy landscape, the gradient-flow dynamics, and the final equilibrium.

| Parameter | Meaning | Effect |
|---|---|---|
| `a` | inventory-risk strength | Larger `a` pulls mean inventory back to zero faster. |
| `b` | interaction / cross-sectional dispersion penalty | Larger `b` compresses equilibrium variance and speeds variance relaxation. Not a penalty on similar inventories. |
| `beta` | entropy / dispersion strength | Larger `beta` makes the equilibrium distribution wider. |
| `dt` or `tau` | numerical step size | Controls how large each solver/JKO step is; not part of the energy itself. |

Important distinction:

```text
a, b, beta = model knobs inside the energy F(rho)
dt, tau    = numerical step-size knobs
```

---

## Closed-form ground truth

For the quadratic model, the theory predicts a unique Gaussian equilibrium:

```math
m_\infty = N\left(0,\frac{\beta}{a+b}\right).
```

Therefore the target equilibrium variance is

```math
\sigma_\infty^2 = \frac{\beta}{a+b}.
```

With the default parameters,

```math
\frac{\beta}{a+b}
=
\frac{0.5}{1.0+0.5}
=
\frac{1}{3}
\approx 0.333.
```

The closed-form moment laws are

```math
\mu_t = \mu_0 e^{-at},
```

and

```math
\Sigma_t
=
\frac{\beta}{a+b}
+
\left(\Sigma_0-\frac{\beta}{a+b}\right)e^{-2(a+b)t}.
```

So:

- the mean relaxes back to zero at rate `a`;
- the variance relaxes to `beta / (a + b)` at rate `2(a+b)`.

---

## JKO interpretation

The JKO scheme is a Wasserstein-space proximal descent method. One step is

```math
\rho_{k+1}
\in
\arg\min_\rho
\left\{
F(\rho)
+
\frac{1}{2\tau}W_2^2(\rho,\rho_k)
\right\}.
```

Interpretation:

```text
next distribution = lower energy + penalty for moving too far
```

JKO is the distributional analogue of implicit gradient descent. The parameter `tau` is the JKO step size. The model parameters `a`, `b`, and `beta` are inside the energy `F` and decide what “downhill” means.

---

## Current implementation

The library lives in `src/wfmm/`. Experiments are `experiments/01_relaxation.py` through `experiments/10_parameter_recovery.py`. Root scripts `mfmm.py`, `make_figures.py`, `gradient_flow_alignment.py`, and `empirical_validation.py` are shims. Reproduce with `python experiments/run_all.py`.

Claims are kept separate: (1) theory/implementation sanity, (2) JKO structure preservation, (3) empirical relevance — synthetic diagnostics only, no market inventories. Numbers and verdicts: [EXPERIMENTS.md](../EXPERIMENTS.md). Overleaf manuscript: [paper/rewrite.tex](../paper/rewrite.tex).

Figure names are `figs/exp01_*.png` … `figs/exp08_*.png` (not the older `exp1_relaxation` / `exp3_crowding` paths).

---

# Experiment specifications

## Experiment 1: Inventory relaxation

### Goal

Verify that a non-equilibrium bimodal inventory distribution relaxes toward the predicted Gaussian equilibrium.

### Setup

Use default parameters:

```python
a = 1.0
b = 0.5
beta = 0.5
```

The target equilibrium is

```math
N(0,1/3).
```

Initial distribution:

```math
m_0
=
\frac{1}{2}N(-2,0.2^2)
+
\frac{1}{2}N(2,0.2^2).
```

This represents a dealer population split between large short and large long inventories.

### Expected behavior

The two-hump distribution should collapse into one centered Gaussian bump. The variance should follow

```math
\Sigma_t
=
\frac{1}{3}
+
\left(\Sigma_0-\frac{1}{3}\right)e^{-3t}.
```

because `2(a+b) = 3`.

Energy should decrease monotonically.

### Required outputs

Save a figure to:

```text
figs/exp01_relaxation.png
```

Recommended panels:

1. density snapshots over time, with analytic equilibrium overlay;
2. free energy `F(m_t) - F(m_inf)` over time;
3. variance over time, with analytic variance curve and target `1/3`.

Also save metrics to:

```text
results/exp01_relaxation.json
```

Required metrics:

```json
{
  "a": 1.0,
  "b": 0.5,
  "beta": 0.5,
  "target_variance": 0.3333333333,
  "terminal_variance": "...",
  "variance_abs_error": "...",
  "energy_increase_steps": "...",
  "mass_error_max": "...",
  "min_density": "..."
}
```

### Pass criteria

- Terminal variance close to `1/3`.
- Energy increase steps equal to zero or numerically negligible.
- Mass conserved up to numerical tolerance.
- Density remains nonnegative for stable solvers.

---

## Experiment 2: Liquidity shock

### Goal

Verify that a temporary tilt in the potential shifts the mean inventory during the shock and that the distribution returns toward equilibrium after the shock disappears.

### Setup

Start at equilibrium:

```math
m_0=N(0,1/3).
```

Use the time-dependent tilted potential

```math
V_t(x)=\frac{a}{2}(x-c_t)^2,
```

where

```text
c_t = 2 for t in [2,4]
c_t = 0 otherwise
```

### Expected behavior

Before the shock:

```text
mean inventory is approximately 0
```

During the shock:

```text
mean inventory moves toward 2
```

At the end of the shock window, the mean should be near

```math
2(1-e^{-2})\approx 1.73.
```

After the shock:

```text
mean inventory decays back toward 0 at rate a = 1
```

### Required outputs

Save a figure to:

```text
figs/exp02_shock.png
```

Recommended panels:

1. mean inventory over time with shaded shock window `[2,4]` and target line `c=2`;
2. Wasserstein distance or approximate distance to baseline equilibrium over time;
3. optional density snapshots before, during, and after shock.

Save metrics to:

```text
results/exp02_shock.json
```

Required metrics:

```json
{
  "shock_center": 2.0,
  "shock_window": [2.0, 4.0],
  "mean_at_shock_start": "...",
  "mean_at_shock_end": "...",
  "quasi_static_prediction": 1.7293294335,
  "shock_end_abs_error": "...",
  "final_mean": "...",
  "final_distance_to_equilibrium": "..."
}
```

### Pass criteria

- Mean rises during `[2,4]`.
- Mean at shock end is close to `1.73`.
- Mean returns toward zero after `t=4`.

---

## Experiment 3: Interaction-strength sweep

### Goal

Verify the comparative static that increasing `b` compresses the equilibrium distribution and that the numerical variance matches $\beta/(a+b)$. Larger `b` is a dispersion penalty, not a penalty on similar inventories.

```math
\frac{\beta}{a+b}.
```

### Setup

Fix:

```python
a = 1.0
beta = 0.5
```

Sweep:

```python
b_values = np.linspace(0, 3, 13)
```

For each `b`, evolve the system to equilibrium.

### Expected behavior

The numerical equilibrium variance should decrease as `b` increases and match

```math
\sigma_\infty^2(b)=\frac{0.5}{1+b}.
```

Examples:

```text
b = 0.0  -> target variance = 0.500
b = 0.5  -> target variance = 0.333
b = 3.0  -> target variance = 0.125
```

### Required outputs

Save a figure to:

```text
figs/exp03_parameter_sweep.png
```

Recommended panels:

1. numerical equilibrium variance versus theoretical `beta/(a+b)`;
2. equilibrium densities for representative values, e.g. `b in {0, 1, 3}`.

Save table to:

```text
results/exp03_parameter_sweep.csv
```

with columns:

```text
b,numerical_variance,theoretical_variance,absolute_error,relative_error
```

Save metrics to:

```text
results/exp03_parameter_sweep.json
```

Required metrics:

```json
{
  "max_abs_error": "...",
  "max_rel_error": "...",
  "monotone_variance_decrease": true
}
```

### Pass criteria

- Variance decreases monotonically with `b`.
- Numerical variance closely follows `beta/(a+b)`.
- Figure clearly communicates compression as interaction strength increases.

---

## Experiment 4: Solver comparison / JKO ablation

### Goal

Show that JKO is a stable Wasserstein-gradient-descent method and compare it against explicit and implicit finite-difference Fokker--Planck solvers.

### Setup

Use a coarse grid and large time step above the explicit stability limit:

```python
N = 161
L = 6.0
dx = 0.075
dt = 1.8 * dx**2 / (2 * beta)
tau = 5e-2
```

Initial distribution should match Experiment 1:

```math
m_0
=
\frac{1}{2}N(-2,0.2^2)
+
\frac{1}{2}N(2,0.2^2).
```

Compare:

1. explicit finite difference;
2. implicit finite difference;
3. JKO in 1D quantile coordinates.

### Expected behavior

Explicit FD should fail at the large step: negative density, mass issues, exploding energy, or divergence.

Implicit FD should remain stable but may suffer numerical diffusion.

JKO should remain stable, conserve mass by construction, preserve positivity by construction, and monotonically dissipate energy on the *autonomous* problem.

Do **not** claim JKO is more accurate at equal resolution unless runtime or degrees of freedom are matched.

### Required outputs

Save a figure to:

```text
figs/exp04_solver_benchmark.png
```

Recommended panels:

1. energy over time for all solvers;
2. negative mass over time;
3. mass-conservation error over time;
4. optional terminal density comparison.

Save table to:

```text
results/exp04_solver_benchmark.csv
```

with columns:

```text
method,stable,energy_increase_steps,mass_error,min_density_or_negative_mass,equilibrium_variance
```

Save metrics to:

```text
results/exp04_solver_benchmark.json
```

### Pass criteria

- Explicit FD fails or violates positivity at the large step.
- Implicit FD remains stable.
- JKO remains stable, has zero/negligible energy-increase steps, and preserves mass/positivity.
- Explicit conservative flux keeps signed mass while stable; above CFL it loses positivity and diverges.
- On the Gaussian test, report error vs $\tau$/$N$ and vs runtime; do not claim a general equal-resolution theorem.

---

# Section 5: synthetic diagnostics (not market evidence)

These tests are a template for a future empirical study. They are not evidence that dealer inventories follow \(F\).

Run in this order on any candidate market-state distribution:

1. **Moment restrictions** (train, then test out of sample): \(\mu_{t+\Delta t}\approx e^{-a\Delta t}\mu_t\) and the variance law. This is the hardest test for the simple quadratic model to hide from. Cosine cannot identify \((a,b,\beta)\mapsto c(a,b,\beta)\).
2. **Distribution forecasting:** \(W_2(\widehat D_{t+\Delta t}, D_{t+\Delta t})\) against persistence and \(b=0\) / OU baselines.
3. **Local directional alignment:** \(\cos\theta\) and unexplained displacement \(1-\cos^2\theta\) of the Brenier *marginal* map. Not circulation. \(\Delta t\) is testable: agreement is local.

Shock experiments must not require unadjusted \(F\) to decrease during a time-dependent tilt.

See [EXPERIMENTS.md](../EXPERIMENTS.md) and [paper/rewrite.tex](../paper/rewrite.tex) for the numbers and the Overleaf manuscript.

---

# Repository structure

```text
src/wfmm/                 model, solvers, OT, estimation, diagnostics
experiments/01_…10_….py
paper/rewrite.tex         Overleaf manuscript (full article)
figs/exp01_….png … exp08_….png
results/exp01_….json …
tests/test_core.py
EXPERIMENTS.md
```

Reproduce with `python experiments/run_all.py` and `python -m pytest tests/ -v`.

# Wasserstein Gradient Flows for Mean-Field Market Making

## Project overview

This project studies a stylized mean-field market-making model where the object of interest is not an asset price, return, or alpha signal, but the **distribution of dealer inventories** across a population of market makers.

The central question is:

> Can the population distribution of market-maker inventories be modeled as a Wasserstein gradient flow of a free-energy functional, and can a JKO scheme provide a stable and interpretable numerical method for computing the resulting equilibria?

The model should be treated as a controlled mathematical/numerical experiment, not as a direct empirical claim about real market prices. The energy lives on the distribution of agent states, especially dealer inventories.

---

## Intuition

Imagine many dealers each holding some signed inventory `x`:

- `x < 0`: dealer is short inventory.
- `x = 0`: dealer is flat.
- `x > 0`: dealer is long inventory.

The whole dealer population forms a probability distribution `rho(x)` over inventories.

The paper asks whether this inventory distribution behaves like a probability blob rolling downhill in Wasserstein space. The downhill direction is determined by a free-energy functional with three forces:

1. **Inventory/risk potential**: dealers dislike large absolute inventory and are pulled toward zero.
2. **Interaction / crowding-compression term**: the population interaction changes the spread of inventories.
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
| `b` | quadratic interaction / compression strength | Larger `b` makes the equilibrium distribution tighter and speeds variance relaxation. |
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

## Existing scripts

The current project contains three useful starting scripts.

### `mfmm.py`

This is the numerical core. It contains:

- the model class `Model(a,b,beta)`;
- closed-form equilibrium `sigma2_inf = beta / (a+b)`;
- explicit and implicit finite-difference Fokker--Planck solvers;
- a 1D quantile-coordinate JKO solver;
- four experiment functions:
  - `exp1_relaxation`;
  - `exp2_shock`;
  - `exp3_crowding_sweep`;
  - `exp4_solver_comparison`.

This script should be the primary starting point for implementing the paper experiments.

### `gradient_flow_alignment.py`

This script implements a sample-based empirical diagnostic:

- Given consecutive empirical distributions `m_k` and `m_{k+1}`, estimate the observed OT velocity `v_obs`.
- Compute the predicted gradient-flow velocity `v_pred = -grad(delta F / delta m)`.
- Report:
  - `cos_theta`;
  - `grad_fraction = cos_theta**2`;
  - `residual_fraction = 1 - cos_theta**2`;
  - energy change `dF`.

This belongs to an empirical-validation extension, not the main controlled Section 4 numerical experiments.

### `empirical_validation.py`

This script wraps the alignment diagnostic into:

- a positive control using true JKO iterates;
- a negative / anti-gradient translation control;
- a data or regime-shift sequence;
- Lyapunov decrease and Gibbs-form checks.

This should be treated as a Section 5 empirical-validation scaffold.

---

## Important alignment fixes

Please fix the following misalignments before treating the project as final.

### 1. The writeup says “three experiments,” but the code and paper contain four numerical experiments

The clean framing should be:

1. **Inventory relaxation**: model-behavior experiment.
2. **Liquidity shock**: model-behavior experiment.
3. **Crowding sweep**: comparative-statics experiment.
4. **Solver comparison**: numerical-method ablation.

Recommended wording:

> We run three main model experiments and one solver-comparison ablation.

### 2. Missing figure generation

The Overleaf project references figures such as:

```text
figs/exp1_relaxation.png
figs/exp2_shock.png
figs/exp3_crowding.png
figs/exp4_solvers.png
```

but the current scripts mainly compute metrics and print summaries. Implement reproducible figure generation and save outputs under `figs/`.

### 3. Crowding wording must match the formula

The current formula is

```math
W(x,y)=\frac{b}{2}(x-y)^2, \qquad b>0.
```

This penalizes dispersion across dealer inventories and compresses the equilibrium variance. Do **not** describe it as penalizing dealers being too similar. Better wording:

> The quadratic interaction penalizes dispersion across the population and produces a common-inventory compression effect.

### 4. Entropy convention in `gradient_flow_alignment.py`

The top docstring currently describes an entropy coefficient like `beta^{-1}` in places. The paper and code use

```math
+\beta\int \rho\log\rho.
```

Fix docstrings/comments to consistently use the paper convention: entropy weight is `beta`, not `1 / beta`.

### 5. Negative-control wording in `empirical_validation.py`

The top docstring says rigid translation should give `cos ~ 0`, but the implemented 1D translation control is anti-gradient and should give approximately `cos ~ -1` with energy increasing. The later explanation is closer to correct. Fix the docstring to say:

> In 1D, the translation control is anti-gradient rather than orthogonal: expect `cos ~ -1` and poor Lyapunov decrease.

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
figs/exp1_relaxation.png
```

Recommended panels:

1. density snapshots over time, with analytic equilibrium overlay;
2. free energy `F(m_t) - F(m_inf)` over time;
3. variance over time, with analytic variance curve and target `1/3`.

Also save metrics to:

```text
results/exp1_relaxation.json
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
figs/exp2_shock.png
```

Recommended panels:

1. mean inventory over time with shaded shock window `[2,4]` and target line `c=2`;
2. Wasserstein distance or approximate distance to baseline equilibrium over time;
3. optional density snapshots before, during, and after shock.

Save metrics to:

```text
results/exp2_shock.json
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

## Experiment 3: Crowding sweep

### Goal

Verify the comparative static that increasing `b` compresses the equilibrium distribution and that the numerical variance matches

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
figs/exp3_crowding.png
```

Recommended panels:

1. numerical equilibrium variance versus theoretical `beta/(a+b)`;
2. equilibrium densities for representative values, e.g. `b in {0, 1, 3}`.

Save table to:

```text
results/exp3_crowding_sweep.csv
```

with columns:

```text
b,numerical_variance,theoretical_variance,absolute_error,relative_error
```

Save metrics to:

```text
results/exp3_crowding_sweep.json
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

JKO should remain stable, conserve mass by construction, preserve positivity by construction, and monotonically dissipate energy.

### Required outputs

Save a figure to:

```text
figs/exp4_solvers.png
```

Recommended panels:

1. energy over time for all solvers;
2. negative mass over time;
3. mass-conservation error over time;
4. optional terminal density comparison.

Save table to:

```text
results/exp4_solver_comparison.csv
```

with columns:

```text
method,stable,energy_increase_steps,mass_error,min_density_or_negative_mass,equilibrium_variance
```

Save metrics to:

```text
results/exp4_solver_comparison.json
```

### Pass criteria

- Explicit FD fails or violates positivity at the large step.
- Implicit FD remains stable.
- JKO remains stable, has zero/negligible energy-increase steps, preserves mass, and reaches variance near `1/3`.

---

# Optional Section 5: empirical validation diagnostic

This is not necessary for the controlled toy experiments, but the existing scripts already set up a useful extension.

## Goal

Given a sequence of empirical distributions, test whether their observed movement aligns with the predicted Wasserstein-gradient direction.

## Diagnostic

For consecutive distributions `m_k -> m_{k+1}`:

1. Estimate the OT map `T_k`.
2. Compute observed velocity:

```math
v_{obs}(x)=\frac{T_k(x)-x}{\Delta t}.
```

3. Compute predicted velocity:

```math
v_{pred}(x)=-\nabla\frac{\delta F}{\delta m}(x).
```

4. Report the alignment cosine:

```math
\cos\theta
=
\frac{\langle v_{obs},v_{pred}\rangle_{L^2(m_k)}}
{\|v_{obs}\|_{L^2(m_k)}\|v_{pred}\|_{L^2(m_k)}}.
```

Interpretation:

| Value | Meaning |
|---|---|
| `cos_theta ~ 1` | observed movement is along the gradient-flow direction |
| `cos_theta ~ 0` | observed movement is orthogonal / non-gradient |
| `cos_theta ~ -1` | observed movement is anti-gradient / uphill |
| `grad_fraction = cos_theta^2` | share of motion explained by the energy direction |
| `residual_fraction = 1 - cos_theta^2` | non-gradient residual |

## Required fixes before use

- Fix entropy convention in `gradient_flow_alignment.py` docstring.
- Fix negative-control wording in `empirical_validation.py`.
- Make sure beta matches the paper convention.
- If using real data, clearly state that this is only an alignment diagnostic, not proof that markets are pure gradient flows.

---

# Recommended repository structure

```text
.
├── README.md
├── mfmm.py
├── gradient_flow_alignment.py
├── empirical_validation.py
├── figs/
│   ├── exp1_relaxation.png
│   ├── exp2_shock.png
│   ├── exp3_crowding.png
│   └── exp4_solvers.png
├── results/
│   ├── exp1_relaxation.json
│   ├── exp2_shock.json
│   ├── exp3_crowding_sweep.csv
│   ├── exp3_crowding_sweep.json
│   ├── exp4_solver_comparison.csv
│   └── exp4_solver_comparison.json
└── tests/
    ├── test_mfmm_experiments.py
    └── test_alignment_diagnostics.py
```

---

# Implementation checklist for coding agent

## Must implement

- [ ] Add deterministic figure-generation functions to `mfmm.py` or a new `make_figures.py`.
- [ ] Create `figs/` automatically if missing.
- [ ] Create `results/` automatically if missing.
- [ ] Generate `exp1_relaxation.png`.
- [ ] Generate `exp2_shock.png`.
- [ ] Generate `exp3_crowding.png`.
- [ ] Generate `exp4_solvers.png`.
- [ ] Save JSON/CSV result artifacts for all experiments.
- [ ] Add a CLI command such as:

```bash
python make_figures.py --all
```

or

```bash
python mfmm.py --make-figures
```

## Should implement

- [ ] Cache or optimize finite-difference operator construction where possible.
- [ ] Avoid long runtimes from rebuilding dense matrices unnecessarily.
- [ ] Add tests for closed-form laws:
  - `sigma2_inf == beta / (a+b)`;
  - mean law `mu_t = mu0 exp(-a t)`;
  - variance law convergence;
  - JKO energy monotonicity.
- [ ] Add tolerance-based validation tests for each experiment.

## Must fix

- [ ] Correct crowding language in comments/docstrings.
- [ ] Correct entropy convention in `gradient_flow_alignment.py`.
- [ ] Correct negative-control wording in `empirical_validation.py`.
- [ ] Align README/writeup language: “three model experiments + one solver ablation.”

---

# Suggested coding-agent prompt

Use the following prompt to drive implementation:

```text
You are working on a project called "Wasserstein Gradient Flows for Mean-Field Market Making." The project models the distribution of dealer inventories as a Wasserstein gradient flow of a free-energy functional.

The core model is one-dimensional with
V(x) = (a/2)x^2,
W(x,y) = (b/2)(x-y)^2,
and entropy beta * int rho log rho.
The free energy is
F(rho) = (a/2) int x^2 rho dx + (b/2) Var(rho) + beta int rho log rho dx.
The theoretical equilibrium is N(0, beta/(a+b)).

Use the existing scripts mfmm.py, gradient_flow_alignment.py, and empirical_validation.py as starting points.

Primary task: implement reproducible experiment generation for the paper. There should be three main model experiments and one solver-comparison ablation:

1. Experiment 1: inventory relaxation from a bimodal initial distribution 0.5*N(-2,0.2^2)+0.5*N(2,0.2^2). Show density snapshots, energy decrease, and variance convergence to beta/(a+b)=1/3 for a=1,b=0.5,beta=0.5.

2. Experiment 2: liquidity shock. Start from N(0,1/3). Apply tilted potential V_t(x)=a/2*(x-c_t)^2 with c_t=2 on t in [2,4] and 0 otherwise. Show mean inventory moving toward 2 during the shock, reaching approximately 2*(1-exp(-2))=1.73, then decaying back to 0.

3. Experiment 3: crowding sweep. Sweep b in [0,3], compute numerical equilibrium variance, and compare to beta/(a+b). Show variance decreases as b increases and plot representative equilibrium densities.

4. Experiment 4: solver comparison. Compare explicit FD, implicit FD, and 1D quantile JKO at a deliberately large step above the explicit CFL limit. Explicit should fail; implicit and JKO should remain stable; JKO should preserve mass/positivity and dissipate energy.

Add deterministic figure generation and save:
figs/exp1_relaxation.png
figs/exp2_shock.png
figs/exp3_crowding.png
figs/exp4_solvers.png

Also save results JSON/CSV files under results/.

Fix alignment issues:
- Use "three model experiments + one solver ablation" language.
- Correct crowding wording: W=b/2*(x-y)^2 with b>0 penalizes dispersion and compresses variance; do not say it penalizes similar inventories.
- In gradient_flow_alignment.py, fix docstrings so entropy coefficient is beta, not beta^{-1}.
- In empirical_validation.py, fix the negative-control docstring: 1D translation is anti-gradient, so expect cos approximately -1 and poor Lyapunov decrease, not cos approximately 0.

Add tests or validation checks that confirm:
- target variance is beta/(a+b),
- Exp 1 terminal variance is close to target and energy decreases,
- Exp 2 shock-end mean is close to 1.73,
- Exp 3 variance curve matches beta/(a+b),
- Exp 4 JKO has monotone energy, mass preservation, and positivity.
```

---

# Recommended README summary paragraph

This repository implements a stylized mean-field market-making model in which the distribution of dealer inventories evolves as a Wasserstein gradient flow of a free-energy functional. The energy combines inventory risk, quadratic population interaction, and entropy. In the quadratic case, the model has a closed-form Gaussian equilibrium `N(0, beta/(a+b))`, which provides ground truth for numerical experiments. The project tests inventory relaxation, liquidity-shock response, crowding comparative statics, and JKO solver stability against finite-difference baselines. The empirical-validation scripts provide an optional diagnostic for measuring whether observed distributional shifts align with the predicted Wasserstein-gradient direction.

# Wasserstein Gradient Flows for Mean-Field Market Making

This repository implements a stylized mean-field market-making model in which the distribution of dealer inventories evolves as a Wasserstein gradient flow of a free-energy functional. The energy combines inventory risk, quadratic population interaction, and entropy. In the quadratic case, the model has a closed-form Gaussian equilibrium `N(0, beta/(a+b))`, which provides ground truth for numerical experiments. The project tests inventory relaxation, liquidity-shock response, crowding comparative statics, and JKO solver stability against finite-difference baselines. The empirical-validation scripts provide an optional diagnostic for measuring whether observed distributional shifts align with the predicted Wasserstein-gradient direction.

We run **three main model experiments** and **one solver-comparison ablation**.

## Quick start

```bash
pip install -r requirements.txt
python make_figures.py --all      # generate figs/ and results/
python mfmm.py                    # quick verification summary
python mfmm.py --make-figures     # same as make_figures.py --all
```

## Model

```text
V(x) = (a/2) x^2
W(x,y) = (b/2)(x-y)^2
F(rho) = (a/2)∫x²ρ + (b/2)Var(ρ) + beta∫ρ log ρ
Equilibrium: N(0, beta/(a+b))
```

Default parameters: `a=1.0`, `b=0.5`, `beta=0.5` → target variance `1/3`.

The quadratic interaction with `b>0` **penalizes dispersion** across dealer inventories and produces a common-inventory compression effect (it does not penalize dealers being too similar).

## Experiments

| # | Name | Output |
|---|------|--------|
| 1 | Inventory relaxation | `figs/exp1_relaxation.png`, `results/exp1_relaxation.json` |
| 2 | Liquidity shock | `figs/exp2_shock.png`, `results/exp2_shock.json` |
| 3 | Crowding sweep | `figs/exp3_crowding.png`, `results/exp3_crowding_sweep.{csv,json}` |
| 4 | Solver comparison (JKO ablation) | `figs/exp4_solvers.png`, `results/exp4_solver_comparison.{csv,json}` |

## Scripts

- `mfmm.py` — numerical core (Model, Fokker–Planck solvers, JKO, experiment runners)
- `make_figures.py` — figure and artifact generation
- `gradient_flow_alignment.py` — optional Section 5 alignment diagnostic
- `empirical_validation.py` — positive/negative controls for the diagnostic

## Tests

```bash
python -m pytest tests/ -v
```

See [docs/wasserstein_finance.md](docs/wasserstein_finance.md) for the full project specification.

# A Wasserstein Gradient-Flow Model of Mean-Field Dealer Inventories

Stylized 1-D model: the distribution of dealer inventories is the Wasserstein
gradient flow of

```text
F(rho) = (a/2) ∫ x² ρ dx + (b/2) Var(ρ) + β ∫ ρ log ρ dx
```

with \(W(x,y)=\frac{b}{2}(x-y)^2\). For \(b>0\) this **penalizes dispersion**
(synchronizing interaction), it does not repel similar inventories.
Closed form: \(N(0,\beta/(a+b))\), mean rate \(a\), variance rate \(2(a+b)\).

The experiments separate three claims:

1. Theory / implementation sanity
2. JKO as a structure-preserving solver (and, on the Gaussian test, first-order in τ)
3. Model validation (synthetic only — no market inventories)

See [EXPERIMENTS.md](EXPERIMENTS.md) for hypotheses, numbers, verdicts, and
manuscript disagreements. The Overleaf rewrite is [paper/rewrite.tex](paper/rewrite.tex).

## Quick start

```bash
pip install -r requirements.txt
python experiments/run_all.py
python -m pytest tests/ -v
```

Single experiment:

```bash
python experiments/01_relaxation.py
python experiments/04_solver_benchmark.py
python experiments/09_convergence.py
python experiments/10_parameter_recovery.py
python experiments/02_shock.py
```

## Layout

```text
src/wfmm/           reusable model, solvers, OT, estimation, diagnostics
experiments/        01–10 scripts (each writes figs/ and results/)
tests/              unit tests (pytest, pythonpath=src)
Wasserstein_finance.pdf
paper/              Overleaf manuscript (`rewrite.tex`)
EXPERIMENTS.md
```

Default parameters: `a=1`, `b=0.5`, `beta=0.5` → target variance `1/3`.

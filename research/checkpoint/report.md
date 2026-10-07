# Experimental identifiability checkpoint report

Date: October 5, 2026

## Decision

**The checkpoint passes.** The three rewrite gates are met:

1. the confinement--interaction ambiguity is verified analytically and
   numerically;
2. the weak-signal lower bounds are explicit finite-snapshot bounds, with
   exact KL separated from nuisance-adjusted information;
3. the ambiguity produces consequential disagreement on a physically shifted
   held-out population.

A rewrite around ambiguity, weak excitation, and interventions is warranted.
The baseline manuscript has not been structurally rewritten. Only its incorrect
description of the old cosine comparison was corrected.

## Gate 1: verified ambiguity

The proof ledger establishes
\[
\mathcal F_{V_H,W_H,\beta}(\rho)-\mathcal F_{V,W,\beta}(\rho)
=\frac12m(\rho)^\top Hm(\rho)
\]
and the force discrepancy \(Hm(\rho)\). Subject to admissibility, existence,
and uniqueness, a path satisfying \(Hm_t=0\) is common to both models.

In the scalar exact-law experiment, six admissible candidates on
\[
(a,b,\beta)\mapsto(a+h,b-h,\beta)
\]
have:

- maximum centered population mean difference: \(0\);
- maximum centered population variance difference: \(0\);
- maximum excess negative log likelihood on independently sampled centered
  snapshots: \(0\) to numerical precision.

This equality is not merely a similar fit: every centered candidate marginal
density is exactly the same.

## Gate 2: rigorous weak-signal bound

### Mean excitation

Alternatives vary \(a\) while fixing \(k=a+b\) and \(\beta\). The proof ledger
contains both:

- exact Gaussian KL for fully specified alternatives; and
- KL/information after projecting over an unknown initial mean.

Le Cam and Pinsker give an explicit finite-sample probability and MSE lower
bound. For the fixed design \(t=(0,0.5,1)\), efficient information is exactly
proportional to \(N\mu_0^2\), with ratio \(0.237331\) in the experiment.
The boundary \(\mu_0\asymp N^{-1/2}\) therefore leaves bounded information.

Across \(N=100,400,1600\), estimator RMSE, invalid-event rates, and coverage
nearly collapse when plotted against efficient information. At the weakest
signals the direct plug-in is invalid in as many as \(89.6\%\) of replications.
Conditional RMSE is reported only beside that invalid rate. The constrained
profile estimator is defined on every sample and supplies unconditional bounded
loss.

### Variance excitation

Alternatives vary \(k\) while holding \(q=\beta/k\) fixed. The proof ledger
again separates:

- exact KL between fully specified centered Gaussian snapshot laws; and
- information for \(k\) after projecting over nuisance
  \((\Sigma_0,q)\).

The candidate scale \(N(\Sigma_0-q)^2\) is not treated as universal. The ratio
of efficient information to that scale ranges from \(0.0140\) to \(0.0528\)
over the experiment because times, variance level, and nuisance geometry
matter. At the weakest excitation the direct plug-in is invalid in as many as
\(93.3\%\) of replications.

The deliberately simple clipped estimator is defined on every sample, but its
oracle-information Wald coverage ranges from \(0.897\) to \(1.0\). This is a
negative result: clipping is not an efficient regular estimator, and the
checkpoint should not advertise those intervals as calibrated. The exact lower
bound and information calculation remain valid.

## Gate 3: consequential intervention disagreement

Training populations are centered. The holdout is a physically new population
with initial mean \(s=0.8\) under the same fixed energy. For candidate
\(\theta_h=(a+h,b-h,\beta)\),
\[
m_t^{(h)}=s e^{-(a+h)t},
\]
while every candidate has the same variance path.

Over 500 replications with 400 observations per snapshot:

- Monte Carlo shifted-holdout excess NLL agrees with the population Gaussian
  KL curve;
- final-time \(W_2\) disagreement reaches \(0.329\);
- the most separated candidate has shifted excess NLL
  \(0.06296\pm0.00098\) (Monte Carlo 95% error bar), against population KL
  \(0.06281\).

Thus models that are observationally identical on the centered training
population make measurably different intervention forecasts.

This shifted experiment reveals the displayed gauge. It does not prove global
identifiability for arbitrary nonlinear potential and interaction functions.

## Cosine audit

The previous row \((1,0.5,0.5)\mapsto(2,2,0.2)\) was a nonuniform parameter
error and was generated/evaluated in a way that did not test Proposition 10.
It has been relabeled.

A dedicated test now reuses the same particles, score estimates, bandwidth,
OT displacements, weights, and aggregation for
\(c\in\{1/2,1,2\}\). The maximum absolute cosine difference is exactly \(0\)
at machine precision. The theorem and implementation agree.

The nonuniform error remains useful as a separate misspecification:
its variance-law MAE is \(0.388\), while the true-model variance MAE is about
\(0.019\). It is no longer described as a scale test.

## Novelty outcome

The audited closest works do not establish the same result:

- JKOnet is potential-only and has no energy-identifiability theorem.
- JKOnet* fits potential, interaction, and entropy, but assumes an invertible
  stacked Gram matrix rather than characterizing its failure under centered
  populations.
- iJKOnet's recovery theorem is potential-only.
- Guan et al. identify one transient gradient drift and diffusion, not a
  decomposition of a McKean--Vlasov drift.
- Nguyen--Malysheva studies gauges inside one drift field with known diffusion;
  it is the closest conceptual work and narrows the claims we can make.
- Lang--Lu studies an interaction kernel with fixed diffusion from a full
  density path, not joint potential--interaction recovery from sparse
  snapshots.

The defensible contribution is the component-decomposition gauge, its
finite-snapshot weak-signal consequences, and its resolution by shared-energy
population interventions in the model classes actually proved.

## Verification

- `python -m pytest tests/ -v`: **18 passed**.
- `python experiments/run_all.py`: **13 experiments completed**, exit code 0,
  total runtime about 339 seconds.
- `cd paper && latexmk -pdf rewrite.tex`: **compiled**, 19 pages, no LaTeX
  warnings matched by the warning scan.
- IDE lint scan on edited Python and TeX files: **no diagnostics**.

## Recommendation for the rewrite

Proceed with a staged rewrite whose main line is:

1. observationally equivalent energy decompositions;
2. weak identification near the ambiguity;
3. physically shifted populations as interventions;
4. consequential held-out intervention forecasts;
5. experimental design and recovery after informative initializations.

Before an AISTATS submission, the strongest next addition is recovery after
shifted populations are included in training, followed by a correctly
specified JKOnet* linear-dictionary comparison. The matrix extension and neural
models can remain later work. The variance estimator should be upgraded before
making interval-calibration claims; its current undercoverage is documented,
not hidden.

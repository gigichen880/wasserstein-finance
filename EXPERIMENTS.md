# Experiment summary

Default model unless noted: \(a=1\), \(b=0.5\), \(\beta=0.5\), so \(\sigma_\infty^2=\beta/(a+b)=1/3\).
Reproduce everything with:

```bash
python experiments/run_all.py
python -m pytest tests/ -v
```

Claims are separated. A negative or qualified result is recorded as such.

---

## Audit of the previous implementation (what changed)

The old tree mixed three claims in `mfmm.py` + `gradient_flow_alignment.py`. Formulas for \(F\), \(\nabla\delta F/\delta\rho\), \(\sigma_\infty^2\), the mean/variance ODEs, the JKO objective, no-flux FP, and 1D monotone OT already matched `Wasserstein_finance.pdf`. Problems were scientific, not algebraic:

| Issue | Action |
|---|---|
| `Model` and `EnergyModel` duplicated \(F\) | Single `wfmm.model.Model` |
| Exp 1 never tested \(\mu_t=\mu_0 e^{-at}\) (symmetric bimodal) | Added a shifted-bimodal companion |
| Exp 2 never reported that unadjusted \(F\) rises under the tilt | Now counted; not treated as a failure |
| Exp 3 swept only \(b\) | Also sweep \(a\) and \(\beta\), and estimate rates |
| Exp 4 was one CFL-breaking point, then claimed JKO “more accurate at equal resolution” | Resolution/step/runtime grid; that accuracy claim is **not** repeated |
| Alignment residual called “rotational” | Renamed unexplained Wasserstein displacement |
| Alignment was a positive control, not model validation | Added wrong DGPs, train/test moments, forecasts, \(\Delta t\) robustness |
| Silverman bandwidth hardcoded as unique | Configurable `bandwidth_scale` |
| No \(\hat\lambda\) | Reported |

### Manuscript disagreements (not code bugs)

1. **Crowding wording (paper §3 / §4.3).** The quadratic \(W=\frac{b}{2}(x-y)^2\) with \(b>0\) *penalizes dispersion* and synchronizes inventories. The sentence “the more dealers penalize being positioned like their peers, the tighter the population concentrates” describes the opposite interaction. Code/figure labels now say interaction / dispersion penalty.
2. **Solver Table 1.** Paper: explicit energy-increase count 32, JKO terminal variance 0.345 vs implicit 0.381 at \(T=2\). Our controlled grid uses \(T=1.5\) and several \((N,\Delta t,M,\tau)\). Counts differ with \(T\) and how NaNs are tallied. We do **not** claim JKO is more accurate at equal spatial resolution.
3. **“Rotational residual” (paper §5).** In 1D, the Brenier map is a monotone rearrangement of *marginals*. Circulation of labeled dealers is not identifiable. Residual \(=1-\cos^2\theta\) is unexplained displacement.

Paper §4.1–4.2 numbers that we *do* reproduce: terminal variance \(0.345\) vs \(0.333\); shock-end mean \(1.72\) vs \(2(1-e^{-2})\approx 1.73\).

---

## Experiment 01 — non-Gaussian relaxation

**Claim tested:** theory / implementation sanity.

**Hypothesis:** From \(m_0=\frac12 N(-2,0.2^2)+\frac12 N(2,0.2^2)\), implicit FP converges to \(N(0,1/3)\), \(F\) decreases, \(\Sigma_t\) follows the ODE of rate \(2(a+b)\), and \(W_2(m_t,m_\infty)\) contracts. A shifted bimodal tests \(\mu_t=\mu_0 e^{-at}\).

**Method:** Implicit conservative FP, \(N=481\), \(\Delta t=2\times 10^{-3}\), \(T=4\). Companion initial law \(\frac12 N(-1.2,0.2^2)+\frac12 N(2.8,0.2^2)\).

**Result:** Terminal variance \(0.3449\) vs \(1/3\) (abs err \(0.0116\); paper: \(0.345\)). Variance-law RMSE \(0.0166\). Energy-increase steps \(0\). Mass error \(4\times 10^{-16}\). \(W_2\): \(1.545\to 0.0098\). Shifted-mean RMSE \(0.0019\).

**Verdict: supports.** The \(0.012\) terminal-variance bias is numerical diffusion of the Eulerian scheme, as the paper already notes.

**Command:** `python experiments/01_relaxation.py`

---

## Experiment 02 — liquidity shock

**Claim tested:** theory / implementation sanity (forced dynamics).

**Hypothesis:** Tilt \(V_t=\frac{a}{2}(x-c_t)^2\) is piecewise constant, so the mean ODE is exact: \(\mu=c+(\mu_0-c)e^{-a\Delta T}\). Baseline \(F_0\) need not decrease; instantaneous \(F_{c_t}\) should dissipate after the switch.

**Result:** Mean at shock end \(1.720\) vs exact \(c(1-e^{-a\Delta T})=1.729\) (abs err \(0.009\); path RMSE \(0.006\)). Amplitude/duration sweep MAE \(0.009\). Baseline \(F_0\) increased on all forced steps; zero increases after. Instantaneous \(F_{c_t}\) had zero increases after the switch (it jumps at the switch itself).

**Verdict: supports**, including the exact (not quasi-static) mean law and the two-energy distinction.

**Command:** `python experiments/02_shock.py`

---

## Experiment 03 — parameter comparative statics

**Claim tested:** theory / implementation sanity.

**Hypothesis:** \(\Sigma_\infty=\beta/(a+b)\); mean rate \(a\); variance rate \(2(a+b)\). \(b>0\) compresses dispersion.

**Method:** Implicit FP from \(N(1,1)\) (nonzero mean so \(a\) is identifiable). Sweeps \(b\in[0,3]\), \(a\in\{0.5,1,1.5,2\}\), \(\beta\in\{0.25,0.5,0.75,1\}\). Rates estimated from consecutive moments.

**Result:** Max \(|\Sigma_\infty^{\mathrm{num}}-\beta/(a+b)|\) over \(b\): \(0.021\) (paper sweep was \(0.014\) on a finer grid / longer \(T\)). Variance decreases in \(b\). Fitted mean-rate MAE \(0.029\). Fitted \(a+b\) MAE \(0.082\) (harder: variance law is two-parameter).

**Verdict: supports** \(\Sigma_\infty(b,\beta)\) and the mean rate. Variance-rate identification is noisier but the right order.

**Command:** `python experiments/03_parameter_sweep.py`

---

## Experiment 04 — JKO vs PDE solvers

**Claim tested:** numerical-method claim.

**Hypothesis:** JKO is stable and structure-preserving (mass, positivity, monotone \(F\)) at large steps. Explicit FP violates CFL. Accuracy is compared against analytic moment laws and \(W_2(m_T,m_\infty)\), plotted against runtime — not “equal \(N\)”.

**Grid:** \(T=1.5\). FP: \(N\in\{81,161\}\), \(\Delta t\in\{0.5,1.8\}\times\mathrm{CFL}\). JKO: \(M\in\{80,160\}\), \(\tau\in\{0.05,0.1\}\).

**Result:**

- Explicit below CFL: signed mass error \(\sim 10^{-15}\), nonnegative on this test.
- Explicit at \(1.8\times\) CFL: loses positivity and diverges (signed mass then explodes with the instability).
- Implicit: stable above CFL, signed mass \(\sim 10^{-15}\), zero energy increases.
- JKO: mass/positivity exact, zero energy increases.

**Verdict: supports structure preservation.** Do not conflate “negative density” with “mass not conserved.” Accuracy versus analytic truth is Experiment 09.

**Command:** `python experiments/04_solver_benchmark.py`

---

## Experiment 09 — refinement against analytic Gaussian and exact-mixture truth

**Claim tested:** numerical-method claim (accuracy, honestly).

**Hypothesis:** From \(N(1,1)\) the law stays Gaussian, so \(m_T^{\mathrm{exact}}=N(\mu_T,\Sigma_T)\) is known. A bimodal Gaussian mixture remains a mixture under the affine McKean–Vlasov map, so it has an exact non-Gaussian law at every \(t\). Sweep JKO \((\tau,M)\) and FP \((N,\Delta t)\) at common \(T=1\). Do not assume JKO wins.

**Result (Gaussian):** log–log slope \(p=0.95\) at \(M=240\) and \(p=0.96\) at \(M=480\). Best JKO: \(W_2=0.0028\) (\(M=480\), \(\tau=0.0125\), median \(0.387\,\mathrm{s}\) after 1 warm-up + 10 repeats, Apple M2, single-threaded). Best implicit: \(0.018\) (\(N=641\), \(\Delta t=0.025\), \(0.083\,\mathrm{s}\)). Best explicit CFL-stable: \(0.0085\) (\(N=641\), \(9.85\,\mathrm{s}\)). All energy-increase counts \(0\) on stable runs.

**Result (bimodal exact mixture):** JKO \(M=240\), \(\tau=0.0125\) has \(W_2=0.0049\) (versus \(0.0028\) Gaussian). Finest implicit \(N=641\), \(\Delta t=0.025\) has \(W_2=0.024\). Error still tracks \(\tau\); the JKO advantage shrinks but remains.

**Verdict: on these closed-form tests, JKO is more accurate per runtime than Eulerian FP.** Eulerian error is spatially diffusive and explicit is CFL-bound. **Not** a resolution-matched theorem for general data.

**Command:** `python experiments/09_convergence.py`

---

## Experiment 05 — synthetic falsification

**Claim tested:** model validation (synthetic).

**Hypothesis:** The diagnostic should look good on the true McKean–Vlasov DGP and worse on misspecified dynamics / parameters. On a fixed pair of marginals, cosine is exactly invariant under \((a,b,\beta)\mapsto c(a,b,\beta)\).

**Setup:** \(N=800\) particles, 12 windows, \(\Delta t=0.05\), evaluation uses the *proposed* quadratic \(v_{\mathrm{pred}}\) unless noted.

| DGP | mean \(\cos\theta\) | residual | mean MAE | var MAE |
|---|---|---|---|---|
| true MV | \(+0.876\) | \(0.230\) | \(0.004\) | \(0.019\) |
| true MV, nonuniform evaluation error \((2,2,0.2)\) | \(+0.808\) | \(0.344\) | \(0.010\) | \(0.379\) |
| \(\tanh\) drift | \(+0.507\) | \(0.730\) | \(0.012\) | \(0.265\) |
| state-dependent diffusion | \(+0.687\) | \(0.514\) | \(0.009\) | \(0.046\) |
| omitted constant force | \(+0.777\) | \(0.385\) | \(0.040\) | \(0.018\) |
| anti-gradient | \(-0.928\) | \(0.135\) | \(0.047\) | \(2.87\) |
| rigid translation | \(-0.489\) | \(0.722\) | \(0.334\) | \(0.437\) |
| quartic \(V=\frac{a}{2}x^2+\frac{\gamma}{4}x^4\) | \(+0.913\) | \(0.164\) | \(0.014\) | \(0.090\) |

**Verdict: supports as a discriminator, with two qualifications.** The \(0.88\to0.81\) comparison is a nonuniform misspecification, not a scale test. A dedicated check reuses exactly the same snapshots, score, displacement, weights, and bandwidth at \(c\in\{0.5,1,2\}\), and the cosine agrees to numerical precision. A nearby quartic potential has *higher* cosine than the true DGP (\(0.91\)) while variance MAE rises \(0.019\to0.090\): alignment can look better on a plausible misspecification.

**Command:** `python experiments/05_synthetic_falsification.py`

---

## Experiment 06 — moment restrictions, train then test

**Claim tested:** model validation (first falsification test; no score estimator).

**Hypothesis:** Fit \(a\) and \(a+b\) on a training segment only; out-of-sample next-step mean/variance should match.

**Setup:** Shifted bimodal MV particles, \(N=1200\), \(\Delta t=0.05\), 24 steps, train on first 12 pairs.

**Result:** Fitted \(a=0.984\) (true \(1\)), \(a+b=1.455\) (true \(1.5\)), \(\sigma_\infty^2=0.352\) (true \(0.333\)). Test mean MAE \(0.0040\) (bootstrap 5–95% CI \(0.0029\)–\(0.0052\)); test variance MAE \(0.0076\). Oracle (true params) is essentially identical.

**Verdict: supports** on synthetic MV data. This is still not market data.

**Command:** `python experiments/06_moment_validation.py`

---

## Experiment 07 — next-distribution forecast

**Claim tested:** model validation (predictive content, not just direction).

**Hypothesis:** One-step \(\widehat D_{t+\Delta t}\) from fitted parameters beats persistence in \(W_2\).

**Baselines:** persistence; affine Gaussian-moment matching; no-interaction (\(b=0\)); one JKO step (\(\tau=\Delta t\), \(M=80\)). Parameters from training windows only.

**Result (test mean \(W_2\)):** persistence \(0.064\); Gaussian moments \(0.045\); \(b=0\) \(0.048\); JKO \(0.036\).

**Caveat:** Fitted \(\sigma_\infty^2=0.094\) on this short, far-from-equilibrium train is badly biased. Forecasts still beat persistence because they get the *local* mean/variance increment roughly right. Equilibrium identification from a short transient is not reliable.

**Verdict: supports predictive content of a JKO/moment step vs persistence on synthetic MV.** Does not show that the fitted Gibbs variance is recovered from a short window.

**Command:** `python experiments/07_distribution_forecast.py`

---

## Experiment 08 — directional alignment, \(\Delta t\) and bandwidth

**Claim tested:** model validation diagnostic (local, scale-free).

**Hypothesis:** Agreement is local. Cosine should degrade as \(\Delta t\) grows. \(\hat\lambda\) absorbs timescale. Residual is unexplained displacement of the Brenier map, not circulation.

**Result:**

| \(\Delta t\) | mean \(\cos\theta\) | residual | mean \(\hat\lambda\) |
|---|---|---|---|
| \(0.02\) | \(0.822\) | \(0.323\) | \(0.020\) |
| \(0.05\) | \(0.874\) | \(0.232\) | \(0.048\) |
| \(0.10\) | \(0.834\) | \(0.284\) | \(0.075\) |
| \(0.20\) | \(0.603\) | \(0.481\) | \(0.090\) |
| \(0.40\) | \(0.345\) | \(0.638\) | \(0.081\) |

At large \(\Delta t\), later windows even change sign. Bandwidth scale \(0.5/1/2\times\) Silverman: cosine \(0.923/0.872/0.813\). Controls: deterministic \(+1\), anti-gradient \(-1\), translation \(\approx -0.16\) (this \(D_1\) is not mean-zero). \(\hat\lambda\approx\Delta t\) on the true flow, as expected.

**Verdict: supports the local-direction diagnostic on synthetic MV, and the claim that \(\Delta t\) is not a pure nuisance.** Silverman is not unique; report sensitivity.

**Command:** `python experiments/08_directional_alignment.py`

---

## Experiment 10 — repeated-seed parameter recovery

**Claim tested:** model validation (synthetic, statistical).

**Hypothesis:** Across independent MV trajectories, train-window moment fits recover \(a\) and \(a+b\); \(b\) and \(\beta\) are noisier; out-of-sample moment and \(W_2\) errors beat persistence as \(N\) grows.

**Setup:** \(40\) seeds, \(N\in\{200,800,1600\}\), \(\Delta t\in\{0.05,0.2\}\), \(12\) train + \(8\) test steps, shifted bimodal start.

**Result (default \(N=800\), \(\Delta t=0.05\)):** MAE of \(\hat a=0.062\) (bias \(-0.002\)); \(\widehat{a+b}\) MAE \(0.072\); \(b\) MAE \(0.10\); \(\beta\) MAE \(0.14\). Test mean MAE \(0.0065\); variance MAE \(0.013\). One-step \(W_2\): persistence \(0.050\), Gaussian moments \(0.034\). At \(N=1600\), \(\hat a\) MAE \(0.035\), \(W_2\) \(0.026\) vs persistence \(0.044\). At \(\Delta t=0.2\), \(\widehat{a+b}\) is biased.

**Verdict: supports** recovery of \(a\) and local forecasts under sampling noise on synthetic MV. Identification of \(b\) and \(\beta\) is weaker. Not market data.

**Command:** `python experiments/10_parameter_recovery.py`

---

## Experiment 12 — exact ambiguity and shifted intervention

**Claim tested:** centered populations cannot distinguish confinement from
interaction along \((a,b,\beta)\mapsto(a+h,b-h,\beta)\), but a physically
shifted population can distinguish that particular gauge.

**Setup:** Exact independent Gaussian snapshots at
\(t\in\{0.25,0.5,1\}\), \(N=400\) per snapshot, \(500\) replications, and
\(h\in\{-0.75,-0.5,-0.25,0,0.2,0.4\}\). Training populations are centered.
The holdout population has initial mean \(0.8\).

**Result:** Population means and variances on centered training data agree
across the ridge to machine precision. The independently sampled centered
negative log likelihood is exactly flat because every candidate density is the
same. On the shifted holdout, Monte Carlo excess loss matches the population
Gaussian KL, and final-time pairwise \(W_2\) disagreement reaches \(0.329\).

**Verdict: supports.** Perfect agreement on the observed centered population
does not imply agreement after a physical intervention. This removes the
displayed gauge; it does not prove complete nonlinear identifiability.

**Command:** `python experiments/12_ambiguity_intervention.py`

---

## Experiment 13 — separate weak-identification mechanisms

**Claim tested:** small initial mean weakens confinement--interaction
separation, while near-equilibrium variance separately weakens recovery of
dispersion dynamics.

**Setup:** Independent Gaussian snapshots at \(t\in\{0,0.5,1\}\), \(1000\)
replications, and \(N\in\{100,400,1600\}\). Mean alternatives vary \(a\) at
fixed \(k=a+b,\beta\). Variance alternatives vary \(k\) at fixed
\(q=\beta/k\). Exact two-point KL is kept separate from information projected
over nuisance parameters.

**Result:** Curves from all three sample sizes collapse when plotted against
nuisance-adjusted information. At the weakest signals, the direct plug-in is
invalid on up to \(89.6\%\) of mean fits and \(93.3\%\) of variance fits.
Every-dataset constrained/clipped losses remain large in those regimes.
The ratio of mean information to \(N\mu_0^2\) is constant for this design;
the ratio of variance information to \(N(\Sigma_0-q)^2\) varies from
\(0.0140\) to \(0.0528\), confirming that the latter is only a candidate
scaling variable.

**Verdict: supports.** Formal identifiability can be statistically useless
without mean or variance excitation. Conditional RMSE is reported together
with failure rates; unconditional bounded loss is defined on every replicate.

**Command:** `python experiments/13_weak_identification.py`

---

## Experiment 14 — equal-budget shifted training recovery

**Claim tested:** under identical times, initial variance, dynamics, and total
samples per time, shifted training populations restore recovery of \(a\) on an
independent shifted test population, while a centered likelihood remains a
ridge. Mixed training is not presumed to beat shifted-only: it has two thirds
of the mean-block information.

**Setup:** Independent Gaussian snapshots at \(t\in\{0.25,0.5,1\}\). Designs
centered (mean \(0\)), shifted-only (mean \(0.8\)), and mixed (means
\(-0.8,0,0.8\)). Initial means are known; common \(\Sigma_0\) is unknown.
Test population mean \(1.2\). Constraints \(a>0\), \(k\ge a\), \(q>0\).
Centered fits are reported as sets and forecast envelopes. Exact Gaussian
\(W_2\) is separate from sampled test NLL.

**Result:** Mixed/shifted information ratio is exactly \(2/3\). At \(4800\)
samples per time the centered forecast envelope of mean \(W_2\) remains
\((0.008,0.560)\). Shifted-only recovers \(a\) with RMSE \(0.023\) and mean
\(W_2\) \(0.011\); mixed has RMSE \(0.029\) and mean \(W_2\) \(0.013\).

**Verdict: supports, with the information-allocation caveat.** Shifted training
removes this gauge for \(a\). Mixing three means under an equal budget is a
weaker excitation of \(a\) than putting the whole budget on one shifted
population.

**Command:** `python experiments/14_shifted_training_recovery.py`

---

## Experiment 15 — variance-interval calibration

**Claim tested:** clipped-ratio/oracle-Wald, constrained variance MLE,
profile-likelihood sets, and a parametric bootstrap are not automatically
calibrated under weak variance excitation.

**Setup:** Reuses the weak-variance design of Experiment 13, with
\(N\in\{100,400,1600\}\) and scaled signals \(\{0.5,2,8,16\}\). Profile sets
are stored as components, including full-domain sets. Compact numerical bounds
are recorded separately from genuine scientific constraints.

**Result:** Weak-signal profile sets cover because they are often the whole
domain. At the strongest displayed cell, profile coverage is \(0.933\) with
mean hull width \(1.27\) on a domain of width \(1.99\). Bootstrap coverage
ranges from \(0.05\) to \(0.775\). Clipped Wald coverage ranges from \(0.906\)
to \(1\). The constrained MLE hits an artificial bound on \(87.8\%\) of the
weakest large-\(N\) replicates.

**Verdict: supports the diagnosis, not a calibration claim.** Failures split
into vanishing information, compact-bound effects, and estimator-specific
clipping or resampling.

**Command:** `python experiments/15_variance_interval_calibration.py`

---

## Experiment 16 — official JKOnet* attempt

**Claim tested:** none until the official `jkonet-star-linear` pipeline
completes on independent snapshots of the quadratic model without changing its
objective or pairing assumptions.

**Setup:** Pin `antonioterpin/jkonet-star` at
`1741c53ae00da932e0841ce02eca2d842c04b813`. Write independent snapshots in the
official `data.npy` format. Use `--split-population` and
`--solver jkonet-star-linear`. Equal spacing \(\{0,0.5,1\}\) is used because
the official evaluator hard-codes `dt=1`. A quadratic polynomial dictionary is
a labeled feature restriction, not a forked loss.

**Result:** Official `jkonet-star-linear` completed at pinned revision
`1741c53ae00da932e0841ce02eca2d842c04b813`. On equally spaced independent
snapshots with 300 samples per time, official one-step \(W_1\) is \(0.078\)
(centered) and \(0.073\) (shifted-only). Mixed means concatenated into one
cloud become a mean-zero mixture, which is not Experiment 14's
three-population design. Official `test_data` is a split of the same
populations, not an independent shifted intervention. Recorded in
`results/exp16_jkonet_star_comparison.json`.

**Verdict: official solver ran; not a substitute for Experiments 14--15.**

**Command:** `python experiments/16_jkonet_star_comparison.py`

---

## Experiment 18 — information geometry / optimal snapshot design

**Claim tested:** identifiability (synthetic, Gaussian snapshots). Which
initial states and observation times carry information for \((a,b,\beta)\).

**Hypothesis:** Centered populations have \(\lambda_{\min}(I_{a,b,\beta})=0\).
Two-snapshot information for \(a\) has an interior maximum: spacings that are
too small or too late are both weak. Frozen-variance optimal spacing is
\(\Delta^\star=x^\star/a\) with \(x^\star\) the root of \(x=1+e^{-2x}\). Equal
total budget does not make every allocation equivalent.

**Setup:** Default model \(a=1\), \(b=0.5\), \(\beta=0.5\). Independent
Gaussian snapshots. Analytic 5-parameter Fisher in
`wfmm.identifiability`, plus 500 Monte Carlo two-snapshot plug-ins at
\(N=400\) per time.

**Result:** Frozen \(\Delta^\star=1.109\). With moving variance,
\(\Delta^\star\approx 1.41\) maximizes \(I_a^{\mathrm{eff}}\) (peak \(69.0\);
at \(\Delta=0.12\) it is \(1.25\)). Centered \(\lambda_{\min}=2.5\times 10^{-14}\);
shifted \(\mu_0=0.8\) gives \(6.99\). Monte Carlo \(N\mathrm{Var}(\hat a)\)
matches \(v_a(\Delta)\) at moderate \(\Delta\) (at \(0.5\): \(19.42\) vs
\(19.46\)). At equal budget \(N_{\mathrm{tot}}=1200\), equal three-time
allocation has \(\lambda_{\min}=6.99\); a two-time design raises
\(I_a^{\mathrm{eff}}\) to \(87.3\) but drops \(\lambda_{\min}\) to \(0.031\)
because \(I_k^{\mathrm{eff}}\approx 0\). Observing after relaxation
(\(t_0\uparrow\)) collapses both \(I_a^{\mathrm{eff}}\) and \(\lambda_{\min}\).

**Verdict: supports.** Snapshot *design* is first-order. Maximizing information
for \(a\) alone can destroy identification of \((k,q)\).

**Command:** `python experiments/18_information_design.py`

---

## Experiment 19 — identifiability phase diagram

**Claim tested:** identifiability (synthetic). Joint grid of mean and variance
excitation at fixed sample budget.

**Hypothesis:** The \((\lvert\mu_0\rvert,\lvert\Sigma_0-q\rvert)\) plane
separates a structural null (either coordinate zero), a weak/ill-conditioned
interior, and a well-identified corner.

**Setup:** Times \(\{0,0.5,1\}\), \(N=400\) per time (\(N_{\mathrm{tot}}=1200\)
fixed), \(250\) replications. Analytic profiled Fisher for \((a,b,\beta)\) and
three-snapshot plug-in recovery.

**Result:** \(\lambda_{\min}=0\) on both axes \(\mu_0=0\) and \(\Sigma_0=q\).
At \((\mu_0,\varepsilon)=(1.2,2.0)\), \(\lambda_{\min}=10.8\) and
\(P(\|\hat\theta-\theta\|<0.5)=0.82\). At \((0.05,0.05)\),
\(\lambda_{\min}=0.041\) and recovery probability \(0.01\). Plugin validity
on the axes is spurious noise, not identification.

**Verdict: supports.** Three regimes are visible at equal budget.

**Command:** `python experiments/19_identifiability_phase.py`

---

## Experiment 20 — approach to the non-identifiable boundary

**Claim tested:** identifiability (synthetic). Scaling of information as
excitation \(\to 0\).

**Hypothesis:** \(I_a^{\mathrm{eff}}\propto\mu_0^2\) exactly;
\(I_k^{\mathrm{eff}}\propto\varepsilon^2 C(\varepsilon)\) locally, so
\(\lambda_{\min}\sim C\varepsilon^p\) with \(p=2\) for both mechanisms.
RMSE of \(\hat a\) should blow up as \(\mu_0\to 0\) but remain finite as
\(\varepsilon\to 0\), because \(a\) is a mean parameter.

**Setup:** Same three-time design, \(N=400\), \(400\) replications.
Log-log OLS on signals \(\le 0.25\).

**Result:** Fitted exponents \(\hat p_{\lambda}(\mu_0)=1.96\),
\(\hat p_{\lambda}(\varepsilon)=1.92\), \(\hat p(I_a)=2.000\),
\(\hat p(I_k)=1.93\). RMSE of \(\hat a\) is large at small \(\mu_0\)
(\(\approx 0.54\) at \(\mu_0=0.01\)) and stays \(\approx 0.12\) at
\(\varepsilon=0.01\).

**Verdict: supports.** Structural non-identifiability is the \(\varepsilon=0\)
endpoint of a \(\varepsilon^2\) information collapse, not a separate
numerical pathology. The two mechanisms are not interchangeable:
vanishing mean excitation costs \(a\); vanishing variance excitation costs
\((k,b,\beta)\).

**Command:** `python experiments/20_boundary_scaling.py`

Memo: `research/identifiability/memo.md`. Experiments 21–23 were not run.

---

## What is and is not supported

1. **Theory / implementation:** Supported (01–03, shock sweep). Closed-form laws, exact piecewise-constant shock mean, two-energy distinction, comparative statics.
2. **JKO as a numerical method:** Supported for stability, positivity, signed-mass (exact), monotone \(F\), no CFL. On the Gaussian refinement test (09), JKO \(W_2\) has log–log slope \(p=0.95\)–\(0.96\) in \(\tau\) and is smaller per runtime than Eulerian FP. On the exact bimodal-mixture subset the advantage shrinks but remains. **Not** supported as a resolution-matched accuracy theorem for general data. Explicit FP *does* conserve signed mass while stable.
3. **Model validation on observed markets:** **Not tested.** 05–08 and 10 use synthetic McKean–Vlasov (or deliberate wrong) particles. Diagnostics can distinguish wrong dynamics, especially via moments; cosine can look *better* on a nearby quartic. They do not say that dealer inventories in the wild follow \(F\).
4. **Experimental identifiability checkpoint:** Supported in the exact scalar
   model (12--15 and 18--20). Centered ambiguity, weak-signal lower bounds, shifted
   intervention disagreement, and equal-budget recovery after shifted training
   pass. Snapshot *design* has an interior optimal spacing for \(a\); information
   collapses as \(\mu_0^2\) and locally as \((\Sigma_0-q)^2\); a phase diagram at
   fixed budget separates structural null, weak, and well-identified cells.
   Interval methods are diagnosed, not declared calibrated. General
   nonlinear recovery, matrix recovery, and an official JKOnet* comparison are
   not claimed unless Experiment 16 completes faithfully. Experiments 18--20 now
   support the excitation-geometry proposition and joint-design corollary in
   `paper/rewrite.tex`: exact \(\mu_0^2\) and local \(\varepsilon^2\) Fisher
   expansions, a phase diagram, and joint snapshot design.

Population inventories remain unobserved; that limitation in the paper is unchanged.

---

## Manuscript revision

The baseline Overleaf manuscript is [`paper/rewrite.tex`](paper/rewrite.tex)
with [`paper/references.bib`](paper/references.bib). Compile with
`cd paper && latexmk -pdf rewrite.tex`.

The four manuscript changes requested after the experiment audit:

1. Section 4 states that the experiments are numerical verification, not market evidence.
2. \(b\) is interaction strength / cross-sectional dispersion penalty, not a penalty on similar inventories.
3. JKO keeps mass, positivity, and energy dissipation without CFL. Signed mass of explicit FP is conserved while stable. On a Gaussian refinement test, JKO error has log–log slope \(p=0.95\)–\(0.96\) in \(\tau\); an exact bimodal-mixture subset is also reported. Neither is a general equal-resolution theorem.
4. The empirical program is moment restrictions + distribution forecasting + local directional alignment, in that order, plus repeated-seed recovery and a nearby quartic falsification.

---

## Empirical liquidity-state checkpoint (not a paper result)

`python experiments/17_empirical_checkpoint.py` builds a small overlapping
one-minute book panel from `data_by_stocks/` and tests whether the quadratic
flow is a useful **effective** description of cross-sectional **depth imbalance**.
That coordinate is a liquidity state, not dealer inventory. Raw archives stay
out of git. The checkpoint writes `research/empirical/report.md` and
`results/empirical/decision.json`. Do **not** treat it as financial validation
of `paper/rewrite.tex` unless that decision file records a pass.


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

**Hypothesis:** Tilt \(V_t=\frac{a}{2}(x-c_t)^2\), \(c_t=2\) on \([2,4]\), moves the mean toward \(c(1-e^{-a\Delta T})\approx 1.73\), then recovers at rate \(a\). Unadjusted \(F(m_t)\) need not decrease during the tilt.

**Result:** Mean at shock end \(1.720\) vs \(1.729\). Final mean \(0.034\). Energy *increased on all 1000 forced steps*; zero increases after the shock.

**Verdict: supports**, including the caveat that forcing injects energy.

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

- Explicit at \(1.8\times\) CFL: unstable, negative mass, exploding energy (as expected; not a research finding).
- Implicit: stable, mass error \(\sim 10^{-15}\), zero energy increases, but numerical diffusion inflates variance-law MAE (\(0.06\)–\(0.16\)).
- JKO: mass/positivity exact, zero energy increases, variance-law MAE \(0.022\)–\(0.048\), \(W_2\) to equilibrium smaller than implicit at this \(T\). Runtime of JKO (\(M=160,\tau=0.05\)): \(0.21\,\mathrm{s}\) vs implicit \(N=161\), \(0.5\times\) CFL: \(0.39\,\mathrm{s}\).

**Verdict: supports structure preservation and large-step stability.** Does **not** support the paper’s “more accurate at equal resolution” sentence: discretizations are not matched (quantile nodes vs grid, \(\tau\) vs \(\Delta t\)), and \(T=1.5\) is not yet equilibrium.

**Command:** `python experiments/04_solver_benchmark.py`

---

## Experiment 05 — synthetic falsification

**Claim tested:** model validation (synthetic).

**Hypothesis:** The diagnostic should look good on the true McKean–Vlasov DGP and worse on misspecified dynamics / parameters. Cosine alone cannot identify overall scale \((a,b,\beta)\mapsto c(a,b,\beta)\); moments should.

**Setup:** \(N=800\) particles, 12 windows, \(\Delta t=0.05\), evaluation uses the *proposed* quadratic \(v_{\mathrm{pred}}\) unless noted.

| DGP | mean \(\cos\theta\) | residual | mean MAE | var MAE |
|---|---|---|---|---|
| true MV | \(+0.876\) | \(0.230\) | \(0.004\) | \(0.019\) |
| true MV, eval at wrong \((a,b,\beta)\) | \(+0.808\) | \(0.344\) | \(0.010\) | \(0.379\) |
| \(\tanh\) drift | \(+0.507\) | \(0.730\) | \(0.012\) | \(0.265\) |
| state-dependent diffusion | \(+0.687\) | \(0.514\) | \(0.009\) | \(0.046\) |
| omitted constant force | \(+0.777\) | \(0.385\) | \(0.040\) | \(0.018\) |
| anti-gradient | \(-0.928\) | \(0.135\) | \(0.047\) | \(2.87\) |
| rigid translation | \(-0.489\) | \(0.722\) | \(0.334\) | \(0.437\) |

**Verdict: supports as a discriminator, with an important qualification.** Wrong *scale* of parameters barely moves cosine (\(0.88\to 0.81\)) but wrecks the variance law. Alignment without moments would overstate fit. State-dependent diffusion (correct drift, wrong noise) still has cosine \(0.69\): the score term is not a sharp test of \(\beta\).

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

## What is and is not supported

1. **Theory / implementation:** Supported (01–03). Closed-form laws, energy dissipation when unforced, shock mean, comparative statics.
2. **JKO as a numerical method:** Supported for stability, positivity, mass, monotone \(F\). Not supported as a resolution-matched accuracy theorem.
3. **Model validation on observed markets:** **Not tested.** All of 05–08 use synthetic McKean–Vlasov (or deliberate wrong) particles. The diagnostics *work* and *can reject* wrong dynamics, especially when moments are used with cosine. They do not say that dealer inventories in the wild follow \(F\).

Population inventories remain unobserved; that limitation in the paper is unchanged.

---

## Manuscript revision

Drop-in Overleaf text is the single file [`paper/rewrite.tex`](paper/rewrite.tex)
(full article: abstract through appendix). Compile with `cd paper && latexmk -pdf rewrite.tex`.

The four manuscript changes requested after the experiment audit:

1. Section 4 states that the experiments are numerical verification, not market evidence.
2. \(b\) is interaction strength / cross-sectional dispersion penalty, not a penalty on similar inventories.
3. JKO keeps mass, positivity, and energy dissipation without CFL; the equal-resolution accuracy claim is removed.
4. The empirical program is moment restrictions + distribution forecasting + local directional alignment, in that order.


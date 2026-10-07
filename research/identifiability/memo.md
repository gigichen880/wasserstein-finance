# Snapshot identifiability: research memo

This memo inspects the existing affine snapshot theory (`paper/rewrite.tex`
§Learning from snapshots, `src/wfmm/identifiability.py`, experiments 11–16)
before adding experiments 18–20. Experiment 17 remains a negative external
checkpoint and is out of scope. `paper/rewrite.tex` is not modified here.

## 1. Exact moment / snapshot law currently assumed

Independent (not interacting-particle) snapshots from the scalar quadratic
McKean–Vlasov flow. If the initial law is Gaussian \(N(\mu_0,\Sigma_0)\),
every later marginal is Gaussian with

\[
\mu_t=\mu_0 e^{-a t},\qquad
\Sigma_t=q+(\Sigma_0-q)e^{-2kt},\qquad
k=a+b,\quad q=\beta/k.
\]

The library already implements these as `Model.mean_law` / `Model.var_law`
and `identifiability.gaussian_moments` / `snapshot_variances`.

The observation experiment is: times \(t_j\), snapshot \(j\) has \(N_j\) i.i.d.
draws from \(m_{t_j}\), independent across \(j\). No labels across time.

Default numerical truth, matching 01–16: \(a=1\), \(b=0.5\), \(\beta=0.5\),
so \(k=1.5\) and \(q=\beta/k=1/3\).

## 2. Identifiable combinations by design

From Proposition ident (paper) and the gauge \((a,b,\beta)\mapsto(a+h,b-h,\beta)\):

| Design | What snapshots identify |
|---|---|
| \(\mu_0=0\), \(\Sigma_0\neq q\) | \((k,\beta)\) or equivalently \((k,q)\). The ridge in \(a\) vs \(b\) is exact. |
| \(\mu_0\neq 0\), \(\Sigma_0=q\), Gaussian | \((a,q)\). Variance never moves, so \(k\) (hence \(b,\beta\)) is unidentified. |
| \(\mu_0=0\) and \(\Sigma_0=q\) | only \(q\). |
| \(\mu_0\neq 0\) and \(\Sigma_0\neq q\), three times | full \((a,b,\beta)\) from \((\mu_j,\Sigma_j)\). |
| two times, \((\Sigma_0,q)\) both unknown | \(I_k^{\mathrm{eff}}=0\) even away from equilibrium (rank of \(\{r,1-r\}\)). |
| unknown \(\mu_0\) | at least two distinct times are required for \(a\). |

Known design means (several populations, common variance path) give mean-block
information proportional to \(\sum_r N_r\mu_{0,r}^2\). That identity is already
tested (mixed/shifted \(=2/3\)).

Fisher information for Gaussian snapshots is **block diagonal** between
\((\mu_0,a)\) and \((\Sigma_0,k,q)\). This is Theorem LAM in the paper.

## 3. Analytic Fisher / local Hessian that we can use

One observation \(X\sim N(\mu(\theta),S(\theta))\) has

\[
I_{ij}=\frac{(\partial_i\mu)(\partial_j\mu)}{S}+\frac{(\partial_i S)(\partial_j S)}{2S^2}.
\]

Sum \(N_j\) over independent snapshots. Existing scalar formulas:

- Mean, \(\mu_0\) known: \(I_a=\sum_j N_j\mu_0^2 t_j^2 e^{-2at_j}/S_j\).
- Mean, \(\mu_0\) unknown (paper eq. mean-info, already in
  `mean_efficient_information`):
  \(I_a^{\mathrm{eff}}=\mu_0^2\sum_j N_j e^{-2at_j}(t_j-\bar t)^2/S_j\).
- Variance, \((\Sigma_0,q)\) unknown (paper eq. var-info, already in
  `variance_efficient_information`):
  \(I_k^{\mathrm{eff}}=\varepsilon^2 C(\varepsilon;k,q,\{t_j,N_j\})\) with
  \(\varepsilon=\Sigma_0-q\).

Two-snapshot plug-in asymptotic variance of \(\sqrt{N}(\hat a-a)\) (Theorem est):

\[
v_a(\Delta)=\Delta^{-2}\Bigl(\frac{\Sigma_0}{\mu_0^2}+\frac{\Sigma_1}{\mu_0^2 e^{-2a\Delta}}\Bigr).
\]

If \(\Sigma_0=\Sigma_1\), the unique minimizer is \(\Delta^\star=x^\star/a\)
where \(x^\star\) solves \(x=1+e^{-2x}\) (\(x^\star\approx 1.109\)). Close
times are weak through \(\Delta^{-2}\); late times are weak through
\(e^{2a\Delta}\).

**Predicted scaling at the non-identifiable boundary (not assumed in code;
to be fitted and proved):**

- \(I_a^{\mathrm{eff}}\propto N\mu_0^2\), so \(\lambda_{\min}\) of the
  profiled \((a,b,\beta)\) information \(\sim C\mu_0^2\) as \(\mu_0\to 0\)
  with \(\varepsilon\neq 0\) (three times). Exponent \(p=2\).
- \(I_k^{\mathrm{eff}}\propto\varepsilon^2\) locally, so
  \(\lambda_{\min}\sim C\varepsilon^2\) as \(\varepsilon\to 0\) with
  \(\mu_0\neq 0\). Exponent \(p=2\), with \(C(\varepsilon)\to C(0)>0\)
  only if three times make \(\{r,1-r,-2tr\}\) full rank.

The 5-parameter Fisher for \((\mu_0,a,\Sigma_0,k,q)\) and its Jacobian image
for \((a,b,\beta)\) are **not** yet in the library; they are the main code
addition for 18–20.

## 4. What 18–23 would add versus 11–16

| Candidate | Already in 11–16? | New scientific content |
|---|---|---|
| **18 design / information geometry** | 11 checks \(v_a(\Delta)\) on one mixture path; 13–14 evaluate information at fixed times \(\{0,0.5,1\}\). | Map information over times, allocation, and budget; confirm too-early/too-late; locate \(\Delta^\star\); full-matrix \(\lambda_{\min}\) and condition number; equal-budget design comparison. **High value.** |
| **19 phase diagram** | 13 is 1-D slices of \(\mu_0\) or \(\varepsilon\), not a joint grid. | Heat maps that visually separate structural null, weak/ill-conditioned, and well-identified regimes at **fixed budget**. **High value.** |
| **20 approach to the boundary** | 13 uses \(\mu_{0,N}\asymp N^{-1/2}\) (local-power) rather than \(\varepsilon\to 0\) at fixed \(N\). | Continuous scaling \(\lambda_{\min}\sim C\varepsilon^p\) for both mechanisms; distinguishes exact ridge from weak identification. **High value.** |
| 21 structural vs predictive | 12 already: centered ridge is invisible, shifted \(W_2=\lvert s\rvert\lvert e^{-(a+h)t}-e^{-(a+h')t}\rvert\). 14 already: centered forecast envelopes vs shifted recovery. | Incremental unless tied to 19’s weak cells. Defer. |
| 22 misspecification | 05 is a nearby quartic particle DGP, not snapshot Fisher/pseudo-true analysis. | Useful later; not required to sharpen the *identifiability* claim. Defer. |
| 23 adaptive design | None. | Needs 18’s criterion first. Optional after 18–20 stabilize. |

Experiment 16 (JKOnet*) and 17 (LOBSTER imbalance) are not part of this
identifiability program.

## 5. Minimal implementation order

1. Library: 5×5 Gaussian snapshot Fisher, Jacobian to \((a,b,\beta)\),
   profiled \(\lambda_{\min}\)/condition number, two-snapshot \(v_a(\Delta)\),
   \(x^\star\), three-snapshot plug-in. Tests against existing
   `mean_efficient_information` / `variance_efficient_information` and a
   numerical Hessian of expected NLL.
2. **Experiment 18** — information vs time, allocation, budget; analytic
   two-snapshot optimum; MC check of \(v_a\).
3. **Experiment 19** — fixed-budget heat maps over \((\lvert\mu_0\rvert,\lvert\varepsilon\rvert)\).
4. **Experiment 20** — \(\mu_0\to 0\) and \(\varepsilon\to 0\) scaling of
   \(\lambda_{\min}\) and RMSE; fit \(p\); compare to \(p=2\).
5. Inspect results. Only then consider 21 or a manuscript figure.
   Do not edit `rewrite.tex` until a claim is actually supported.

Success criterion: one or two sharp, reproducible identities — expected:
(i) an interior optimal spacing for \(a\); (ii) \(\lambda_{\min}\propto\mu_0^2\)
and \(\propto\varepsilon^2\) at the two boundaries, with a 2-D phase diagram
that makes the three regimes visible.

## Results of 18–20 (not yet in the manuscript)

All three experiments were run. `paper/rewrite.tex` was not edited.

1. Interior optimum: frozen \(\Delta^\star=1.109=x^\star/a\); moving-variance
   peak of \(I_a^{\mathrm{eff}}\) at \(\Delta\approx 1.41\). Too-close and
   too-late times are both weak. Equal three-time allocation beats
   two-time allocation on \(\lambda_{\min}\), even though two times can raise
   \(I_a^{\mathrm{eff}}\) (they leave \(I_k^{\mathrm{eff}}\approx 0\)).
2. Phase diagram: \(\lambda_{\min}=0\) on both axes; recovery probability
   reaches \(0.82\) only when both excitations are large.
3. Scaling: \(\hat p(I_a)=2\), \(\hat p(\lambda_{\min})\approx 1.92\)–\(1.96\).
   RMSE of \(\hat a\) tracks mean excitation, not variance excitation.

These are the figures that could later support a short design/scaling
paragraph in §Learning from snapshots. Experiments 21–23 remain deferred.

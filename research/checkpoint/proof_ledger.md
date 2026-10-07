# Experimental-identifiability proof ledger

This checkpoint separates proved identities from statistical targets. Its
observation model is independent samples within and across snapshot times unless
stated otherwise. A physically shifted initialization is a new population
experiment under the same fixed energy; recentering stored observations is not.

## 1. Potential--interaction gauge

### Statement

Let
\[
\mathcal F_{V,W,\beta}(\rho)
=\int V\,d\rho
+\frac12\iint W(x-y)\,d\rho(x)d\rho(y)
+\beta\operatorname{Ent}(\rho)
\]
on \(\mathcal P_2(\mathbb R^d)\), where \(W\) is even. For a symmetric matrix
\(H\), define
\[
V_H(x)=V(x)+\frac12x^\top Hx,\qquad
W_H(z)=W(z)-\frac12z^\top Hz.
\]
Then, for \(m(\rho)=\int x\,d\rho(x)\),
\[
\mathcal F_{V_H,W_H,\beta}(\rho)-\mathcal F_{V,W,\beta}(\rho)
=\frac12m(\rho)^\top Hm(\rho)
\]
and
\[
\nabla V_H+\nabla W_H*\rho-(\nabla V+\nabla W*\rho)=Hm(\rho).
\]

Suppose both energies belong to the declared admissible class, their
McKean--Vlasov equations exist and are unique from a common initial law, and a
solution of the first model satisfies \(Hm_t=0\) for every \(t\in[0,T]\). That
same path solves the transformed model, hence all of its marginals are
identical under the two decompositions.

This is an explicit sufficient ambiguity mechanism, not a classification of
all possible ambiguities.

### Proof

For independent \(X,Y\sim\rho\),
\[
\mathbb E[(X-Y)^\top H(X-Y)]
=2\mathbb E[X^\top HX]-2m^\top Hm.
\]
The potential correction is
\(\frac12\mathbb E[X^\top HX]\), while the interaction correction is
\(-\frac14\mathbb E[(X-Y)^\top H(X-Y)]\). Their sum is
\(\frac12m^\top Hm\). Differentiation gives
\[
\nabla\!\left(\tfrac12x^\top Hx\right)=Hx,\qquad
\nabla\!\left(-\tfrac12|\cdot|_H^2\right)*\rho=-H(x-m),
\]
so the force difference is \(Hm\). If this vector vanishes along a path, the
two PDE right-hand sides agree there; uniqueness identifies the paths.

### Assumptions and boundaries

- \(H\) is symmetric, so the transformed interaction remains even.
- Convexity, coercivity, growth bounds, and well-posedness must be checked for
  both transformed components. They do not follow for arbitrary \(H\).
- In the scalar quadratic model,
  \((a,b,\beta)\mapsto(a+h,b-h,\beta)\) is admissible when
  \(a+h>0\) and \(b-h\ge0\). A two-sided local perturbation requires \(b>0\).
- Additive constants are a separate trivial gauge and are normalized away.
- If an unknown linear term \(\ell^\top x\) is admitted, the force difference
  becomes \(Hm_t+\ell\); affine rather than linear span controls this gauge.
- A shifted population can expose this gauge. It does not establish complete
  nonlinear identifiability of arbitrary \(V,W\).

### Nonlinear example target

For
\[
V(x)=\frac{\alpha}{4}x^4+\frac a2x^2,\qquad
W(z)=\frac{\eta}{4}z^4+\frac b2z^2,
\]
with an even initial density, symmetry and uniqueness preserve \(m_t=0\).
Admissible small changes \(a\mapsto a+h,\ b\mapsto b-h\) are therefore
invisible even though the transient law need not be Gaussian. The algebra is
proved above; a theorem for this example additionally needs a stated
well-posedness class and admissible range of \(h\).

## 2. Weak mean excitation

Fix \(k=a+b\), \(\beta\), a Gaussian initial variance \(\Sigma_0\), and times
\(t_0,\ldots,t_J\). Snapshot \(j\) contains \(N_j\) independent observations
\[
X_{ij}\sim N(\mu_0e^{-at_j},S_j),\qquad
S_j=q+(\Sigma_0-q)e^{-2kt_j},\quad q=\beta/k.
\]
Compare admissible, fully specified alternatives
\[
(a,b,\beta)\quad\text{and}\quad(a+\delta,b-\delta,\beta).
\]
The alternatives have the same \(k,\beta\), hence the same variances.

### Exact KL with known initial mean

\[
K_{\rm mean}(\delta)
=\frac12\sum_j
\frac{N_j\mu_0^2
\left(e^{-at_j}-e^{-(a+\delta)t_j}\right)^2}{S_j}.
\]
This is a divergence between two fully specified distributions. Its local
information is
\[
I_a^{\rm known}
=\mu_0^2\sum_j\frac{N_jt_j^2e^{-2at_j}}{S_j}.
\]

### Unknown initial mean

Efficient information treats \(\mu_0\) as nuisance. Put
\(w_j=N_j/S_j\), \(f_j=e^{-at_j}\), and
\(g_j=e^{-(a+\delta)t_j}\). The least distinguishable alternative initial
mean is
\[
\mu_0'=\mu_0\frac{\sum_jw_jf_jg_j}{\sum_jw_jg_j^2},
\]
and the nuisance-minimized exact KL is
\[
\inf_{\mu_0'}K
=\frac{\mu_0^2}{2}
\left[
\sum_jw_jf_j^2-
\frac{(\sum_jw_jf_jg_j)^2}{\sum_jw_jg_j^2}
\right].
\]
Its local expansion is \(\delta^2I_a^{\rm eff}/2+o(\delta^2\mu_0^2)\), where
\[
I_a^{\rm eff}
=\mu_0^2\sum_j
\frac{N_je^{-2at_j}}{S_j}(t_j-\bar t)^2,
\quad
\bar t=
\frac{\sum_jN_je^{-2at_j}t_j/S_j}
{\sum_jN_je^{-2at_j}/S_j}.
\]
At least two distinct observation times are needed when \(\mu_0\) is unknown.

### Finite-sample lower bound

For either exact two-point KL \(K\), Pinsker's inequality and Le Cam's
two-point argument imply, for every estimator \(\widehat a\),
\[
\max_iP_i\!\left(|\widehat a-a_i|\ge|\delta|/2\right)
\ge\frac12\left(1-\sqrt{K/2}\right),
\]
and
\[
\max_i\mathbb E_i(\widehat a-a_i)^2
\ge\frac{\delta^2}{8}\left(1-\sqrt{K/2}\right).
\]
Under a fixed nondegenerate design, the information scales as
\(N\mu_{0,N}^2\). Thus \(\mu_{0,N}\asymp N^{-1/2}\) is the weak-signal
boundary, while \(\mu_0=0\) gives exact non-identification along the admissible
ridge.

## 3. Weak variance excitation

This mechanism is separate from weak mean excitation. Let
\(q=\beta/k\), \(\epsilon=\Sigma_0-q\), and compare fully specified,
admissible alternatives
\[
(k,q)\quad\text{and}\quad(k+\delta,q).
\]
Equivalently, at fixed \(a\), change
\((b,\beta)\mapsto(b+\delta,\beta+q\delta)\). Then
\[
S_j(k)=q+\epsilon e^{-2kt_j}.
\]

### Exact KL

For centered Gaussian snapshots,
\[
K_{\rm var}(\delta)
=\frac12\sum_jN_j
\left[
\frac{S_j(k)}{S_j(k+\delta)}
-1-\log\frac{S_j(k)}{S_j(k+\delta)}
\right].
\]
This compares fully specified distributions. For fixed times and variances
bounded away from zero it is
\(O(\delta^2\epsilon^2\sum_jN_j)\). The quantity
\(N\epsilon^2\) is therefore a candidate scaling variable, not a universal
collapse coordinate.

### Nuisance-adjusted information

For parameters \((\Sigma_0,k,q)\), let \(r_j=e^{-2kt_j}\). The derivative
vectors are
\[
\partial_{\Sigma_0}S=r,\qquad
\partial_qS=1-r,\qquad
\partial_kS=-2\epsilon\,t r.
\]
With
\[
\langle u,v\rangle_S=\sum_j\frac{N_j}{2S_j^2}u_jv_j,
\]
the efficient information for \(k\) is
\[
I_k^{\rm eff}
=\left\|
(I-\Pi_{\operatorname{span}\{r,1-r\}})
(-2\epsilon\,tr)
\right\|_S^2.
\]
Hence \(I_k^{\rm eff}=\epsilon^2C\), where \(C\) depends on times, variance
scale, allocation, and nuisance parameters. With three distinct times and
positive weights, \(C\) has a positive limit at \(\epsilon=0\) for a
nondegenerate design. At \(\epsilon=0\), however, the model is singular and the
variance path contains no information about \(k\). With only two variance
observations and both \(\Sigma_0,q\) unknown, the full variance model is also
singular.

### Exact Fisher factorizations (Experiments 18--20)

The mean score is linear in \(\mu_0\) because \(\partial_a\mu_t=-t\mu_0 e^{-at}\).
The \((\mu_0,a)\) Fisher block is therefore homogeneous of degree \((0,1,2)\)
in \(\mu_0\), and the Schur complement is
\[
I_a^{\mathrm{eff}}=\mu_0^2 I_a^{(1)}
\]
exactly, for every \(\mu_0\). The factor \(I_a^{(1)}\) does not depend on
\(\mu_0\) because \(S_j\) does not.

The variance score is linear in \(\varepsilon=\Sigma_0-q\) because
\(\partial_k\Sigma_t=-2t\varepsilon e^{-2kt}\). Hence
\(I_k^{\mathrm{eff}}=\varepsilon^2 C(\varepsilon)\), and \(C(\varepsilon)\to
C(0)\) under the three-time rank condition of Proposition weak-var.

The profiled matrix for \((a,b,\beta)\) splits as
\(I=I_{\mathrm{var}}(\varepsilon)+\mu_0^2 B\) with \(B\) independent of
\(\mu_0\). The variance block has kernel \((1,-1,0)\) when \(\varepsilon\neq 0\).
Along the unit gauge, \(u^\top B u=I_a^{(1)}/2\), so
\[
\lambda_{\min}(I)=\frac{\mu_0^2}{2}I_a^{(1)}+O(\mu_0^4)\qquad(\mu_0\to 0).
\]
As \(\varepsilon\to 0\) at fixed \(\mu_0\neq 0\), the kernel approaches
\((0,1,q)\) and \(\lambda_{\min}=\Theta(\varepsilon^2)\).

Joint design: two times give \(C\equiv 0\), hence \(\lambda_{\min}(I)=0\),
even if \(I_a^{\mathrm{eff}}\) is large. Implemented in
`mean_unit_information`, `variance_unit_information`, and
`lambda_min_mean_leading`.

## 4. Centered training and shifted intervention

Let \(\theta_h=(a+h,b-h,\beta)\), with \(k=a+b\) and \(q=\beta/k\) fixed. For
any scalar initial law,
\[
X_t^{(h)}
=\mu_0e^{-(a+h)t}
+e^{-kt}(X_0-\mu_0)
+\sqrt{q(1-e^{-2kt})}\,Z.
\]

- If \(\mu_0=0\), the complete marginal path is identical for every admissible
  \(h\), including for non-Gaussian initial laws.
- If a physical intervention translates a centered \(Y_0\) by \(s\), then
  \[
  X_t^{(h)}
  =se^{-(a+h)t}+e^{-kt}Y_0+\sqrt{q(1-e^{-2kt})}\,Z.
  \]
  Candidate predictions are translations of one common centered law, so
  \[
  W_2\!\left(\mathcal L(X_t^{(h)}),\mathcal L(X_t^{(h')})\right)
  =|s|\,|e^{-(a+h)t}-e^{-(a+h')t}|.
  \]
- For Gaussian \(Y_0\sim N(0,\Sigma_0)\), every candidate has variance
  \(q+(\Sigma_0-q)e^{-2kt}\).

Centered training can therefore have identical population loss and identical
sample-loss distributions across the ridge while shifted-population forecasts
disagree. If \(s\) is known, one positive-time population mean identifies the
confinement rate within this ridge family. If \(s\) is unknown, two distinct
times are required.

## 5. Status and manuscript cautions

| Item | Status | Remaining condition |
|---|---|---|
| Gauge energy and force identities | proved | transformed model admissibility |
| Path equivalence | proved conditionally | existence and uniqueness |
| Mean exact KL and projection | proved | independent Gaussian snapshots |
| Mean Le Cam bound | proved | admissible two-point alternatives |
| Variance exact KL | proved | independent centered Gaussian snapshots |
| Variance efficient information | proved algebraically | regular nondegenerate design away from \(\epsilon=0\) |
| Centered and shifted exact laws | proved | common fixed energy and physical intervention |
| Quartic example | algebra only | well-posedness and moment assumptions |

The baseline manuscript should also avoid two overstatements in a later
rewrite:

1. A centered law *depends only on* \((a+b,\beta)\), but need not identify both;
   a centered equilibrium Gaussian identifies only \(q=\beta/(a+b)\).
2. Stationary variance plus non-Gaussianity identifies \(k\) through a higher
   cumulant only if a finite, nonzero cumulant of some order \(j\ge3\) is
   available.

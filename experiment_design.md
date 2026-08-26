# Experiment design: testing directional alignment with the JKO gradient flow

## Objective

We want to test, on data, whether an observed shift in the inventory (or
market-state) distribution moves in the *same direction* as one step of the
Wasserstein gradient flow of the proposed free energy

$$
\mathcal{F}(\rho) = \frac{a}{2}\int x^2 \rho\,dx + \frac{b}{2}\operatorname{Var}(\rho) + \beta\int \rho\log\rho\,dx .
$$

The hypothesis is not that markets are an exact gradient flow. Real markets are
non-equilibrium steady states, so we expect motion to be only partially
gradient-like. Alignment is the third diagnostic in a hierarchy:
out-of-sample moment restrictions, then $W_2$ forecasts against persistence,
then local cosine with unexplained displacement $1-\cos^2\theta$. Cosine
cannot identify overall parameter scale.

## Objects and notation

Let $D_1$ and $D_2$ be two consecutive empirical distributions, given as samples,
separated by an elapsed time $\Delta t$. Three objects carry the test.

- The **observed transport map** $T$, the Brenier map with $T_\#\, D_1 = D_2$.
  Alignment is computed on the displacement $u_{\text{obs}}=T-\mathrm{id}$
  (equivalent in cosine to $v_{\text{obs}}=(T-\mathrm{id})/\Delta t$; the
  timescale sits in $\hat\lambda$).
- The **predicted velocity**, the Wasserstein gradient of $\mathcal{F}$ evaluated at
  $D_1$,
  $$
  v_{\text{pred}}(x) = -\nabla\frac{\delta\mathcal{F}}{\delta\rho}(x)
  = -(a+b)\,x + b\,\mu(D_1) - \beta\,\nabla\log\rho(x),
  $$
  where $\mu(D_1)$ is the mean of $D_1$ and $\nabla\log\rho$ is the score of its
  density. For the quadratic model every term is closed-form except the score.
- The **energy gap** $\Delta\mathcal{F} = \mathcal{F}(D_2) - \mathcal{F}(D_1)$, used as a Lyapunov
  cross-check.

## Test statistic

Because a gradient flow and any positive time-reparametrization trace the same
curve, *direction* is the scale-free content of the hypothesis. The cosine
in the $L^2(D_1)$ (Otto) inner product,

$$
\cos\theta = \frac{\langle u_{\text{obs}},\, v_{\text{pred}}\rangle_{L^2(D_1)}}
{\lVert u_{\text{obs}}\rVert_{L^2(D_1)}\;\lVert v_{\text{pred}}\rVert_{L^2(D_1)}},
\qquad
u_{\text{obs}} = T - \mathrm{id},
$$

removes a scalar. It does *not* make $\Delta t$ uninformative: $v_{\text{pred}}$
is an instantaneous tangent while OT between $D_1$ and $D_2$ is a finite
displacement, so directional agreement is only expected locally. We also
report the best-fit scale $\hat\lambda=\langle u_{\text{obs}},v_{\text{pred}}\rangle/\|v_{\text{pred}}\|^2$
and the unexplained Wasserstein displacement $1-\cos^2\theta$. The cosine
cannot identify the overall parameter scale $(a,b,\beta)\mapsto c(a,b,\beta)$;
moment laws recover the timescale. Alignment is an interpretability diagnostic
to be run *after* out-of-sample moment restrictions and $W_2$ forecasts, not
instead of them.

## Why displacements, not maps, are proportional

A natural first guess is to compute the JKO map $\mathcal{T}$ and check whether it is a
constant multiple of $T$. This is not the right comparison. Writing the
displacements $u = T - \mathrm{id}$ and $w = \mathcal{T} - \mathrm{id}$, exact directional
agreement $v_{\text{obs}} = \lambda\, v_{\text{pred}}$ for a single scalar
$\lambda$ gives

$$
\mathcal{T} - \mathrm{id} = c\,(T - \mathrm{id}), \qquad c = \frac{\tau}{\lambda\,\Delta t},
$$

that is, the **displacement fields** are proportional, not the maps. Rearranged,
$\mathcal{T} = (1-c)\,\mathrm{id} + c\,T$, which is the displacement interpolation between
$\mathrm{id}$ and $T$: one gradient step lands a fraction $c$ of the way along the
Wasserstein geodesic from $D_1$ to $D_2$. The relation $\mathcal{T} = c\,T$ holds only
for dilations about the origin (for instance two centred Gaussians) and fails in
general. The constant $c$ is a nuisance scale set by $\tau$ and $\Delta t$; the
cosine removes it, so the procedure never estimates $\mathcal{T}$ or $c$ and compares
velocities directly.

## Procedure

1. **Observed displacement.** Compute the optimal-transport map $T$ from $D_1$ to
   $D_2$. In one dimension this is the monotone rearrangement,
   $T(x) = Q_{D_2}(F_{D_1}(x))$ with $F_{D_1}$ the empirical cumulative
   distribution of $D_1$ and $Q_{D_2}$ the quantile function of $D_2$; it is
   exact for any sample sizes. Set $u_{\text{obs}}(x_i) = T(x_i)-x_i$.
   (Dividing by $\Delta t$ does not change $\cos\theta$; it rescales $\hat\lambda$.)

2. **Predicted velocity.** Evaluate $v_{\text{pred}}$ at the $D_1$ samples using
   the sample mean $\mu(D_1)$ and a score estimate $\nabla\log\hat\rho$.

3. **Alignment.** Form $\cos\theta$, $\hat\lambda$, the unexplained displacement
   $1 - \cos^2\theta$, and the energy gap $\Delta\mathcal{F}$.

4. **Calibration.** Compute the same statistics on the controls below at the
   same sample size and $\Delta t$, and read the data value against them rather
   than against the ideal value $1$.

## Controls

The controls fix the scale of "aligned enough" at a given sample size, where
finite-sample noise in the transport map and the score pulls $\cos\theta$ below
$1$ even for a genuine flow.

- **Positive, deterministic.** Move each sample by the predicted velocity,
  $D_2 = D_1 + \Delta t\, v_{\text{pred}}$. This is a deterministic gradient step
  and returns $\cos\theta \approx 1$ with $\Delta\mathcal{F} \le 0$; it is the ceiling
  the estimator can reach.
- **Positive, stochastic.** Integrate the McKean-Vlasov ensemble
  $dX = -[(a+b)X - b\,\mu_t]\,dt + \sqrt{2\beta}\,dW$ for a short time. This is a
  genuine but noisy gradient flow: $\cos\theta < 1$ with $\Delta\mathcal{F} \le 0$ in
  expectation, the realistic reference for what data can look like.
- **Negative, anti-gradient.** Take one step against the gradient,
  $D_2 = D_1 - \Delta t\, v_{\text{pred}}$; returns $\cos\theta \approx -1$ with
  $\Delta\mathcal{F} > 0$.
- **Negative, rigid translation.** Shift the whole distribution,
  $D_2 = D_1 + s$. For a mean-zero $D_1$ this reads $\cos\theta \approx 0$ yet has
  $\Delta\mathcal{F} > 0$; it demonstrates that the cosine alone can miss an uphill move
  and that the $\Delta\mathcal{F}$ check is what flags it.

## Estimation details

The score $\nabla\log\hat\rho$ is estimated by a Gaussian kernel density
estimate with Silverman bandwidth, whose gradient is the mean-shift vector

$$
\nabla\log\hat\rho(x) = \frac{1}{h^2}\!\left(\sum_j w_j(x)\, x_j - x\right),
\qquad
w_j(x) = \frac{K_h(x - x_j)}{\sum_k K_h(x - x_k)} .
$$

The differential entropy in $\mathcal{F}$, written $\int\rho\log\rho = -h(\rho)$, is
estimated by the Kozachenko-Leonenko $k$-nearest-neighbour estimator, which
avoids binning and extends to several dimensions. Both estimates are noisy in
finite samples, which is the reason the positive control, not the value $1$, is
the reference for a positive result.

## Higher dimensions and learned maps

The one-dimensional monotone rearrangement is exact and cheap. When the market
state is multi-dimensional, replace it with an exact assignment for equal sample
sizes, or an entropic (Sinkhorn) or Earth-mover solver, taking the barycentric
projection to recover a map.

Both maps can also be learned with a neural network, which is the route for
higher-dimensional or noisy states.

- The observed map $T$ can be parametrized as the gradient of an input-convex
  neural network representing the Brenier potential, so the learned map is the
  gradient of a convex function by construction (Makkuva, Taghvaei, Oh, and Lee,
  ICML 2020).
- The gradient-flow map of a *known* energy can be learned the same way by
  discretizing the JKO steps with input-convex networks (Mokrov, Korotin, Li,
  Genevay, Solomon, and Burnaev, NeurIPS 2021; Alvarez-Melis, Schiff, and
  Mroueh, TMLR 2022).
- The inverse problem, recovering the energy $\mathcal{F}_\theta$ that best explains an
  observed sequence of distributions, is addressed by adversarial JKO inverse
  optimization with error bounds for convex potentials. This is the trainable
  form of the alignment target: fit $\mathcal{F}_\theta$ so its gradient direction best
  matches the observed transport, then test out of sample.

## Reporting and acceptance

For each consecutive pair report $\cos\theta$, $\cos^2\theta$,
$\hat\lambda$, $1-\cos^2\theta$, and $\Delta\mathcal{F}$, together with control
values at the matching sample size and $\Delta t$, and together with the same
window's moment-restriction and $W_2$-forecast errors. The residual
$1-\cos^2\theta$ is unexplained Wasserstein displacement of the two
\emph{marginals}, not identifiable microscopic circulation. Alignment is the
third diagnostic, after out-of-sample moments and distributional forecasts.

## Limitations

In one dimension every velocity is a scalar, so pointwise "same direction" only
means same sign; the aggregate cosine with magnitudes carries the content, and
pointwise direction becomes informative only once the state has more than one
dimension. The predicted velocity depends on the quadratic specification of
$\mathcal{F}$; a richer energy need not be a gradient flow at all, as stressed in the
paper. Cosine is invariant to $(a,b,\beta)\mapsto c(a,b,\beta)$. Finally,
inventory distributions are not directly observable at the population level,
so an observable market-state proxy is needed before the test can be run on
real data.

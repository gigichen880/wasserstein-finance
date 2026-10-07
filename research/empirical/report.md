# Empirical checkpoint (liquidity-state population)

This is a **go/no-go research checkpoint**. It is not a manuscript claim.
The theoretical paper and synthetic experiments 01–16 are unchanged.

## Population

Cross-sectional **top-of-book depth imbalance** of listed securities, not
dealer inventories and not reconstructed trader positions. A successful fit
would mean a useful effective description, not a universal financial law.

Primary state: `x = (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)`.
Zero denominators are NaN, not zero. No time-by-time cross-sectional
recentering. Affine Gaussian dynamics on the real line are an **explicit
approximation** of a coordinate supported on [-1, 1].

## Data audit

- Unique tickers in `data_by_stocks/`: **516**
- One-level names: **397**
- Ten-level names: **119**
- Names present at both depths: **0**
- Identity check one+ten−both = unique: **True**
- Extracted directories at scan time: **2**
- Compressed archives: **516**

Reported “119 ten-level + 398 one-level” does not match the archive census:
397 + 119 = 516 unique names, with **zero** names present at both depths.
The extra one-level count is unverified metadata.

Clock times are America/New_York civil session times. Files do not record
venue or timezone offsets. 391 bars is a complete 09:30–16:00 minute grid,
not 391 independent observations. Universe is a prespecified large-cap /
ETF list that survived in the archive; that is a survivorship-conditioned
pilot, not a randomly sampled CRSP panel.

Opened-file sample (extracted `A` and `AAPL` only): 86 files,
86 full 391-bar sessions,
0 with ≥60 trailing identical rows
(possible padded early close), mean stale fraction
0.006231365533691114,
1 file(s) with any crossed book.
Median ask in the opened sample ranges
18.42–616.32
dollars after the 1e-4 price scale. Records can be unchanged
minute-to-minute; they are not guaranteed to be independent snapshots.

Pilot universe (prespecified, 10-level names): `AAPL, MSFT, AMZN, GOOGL, FB, NVDA, JPM, BAC, XOM, JNJ, UNH, PG, V, MA, HD, DIS, INTC, CSCO, BA, WMT, CVX, PFE, NFLX, ADBE, SPY, XLF, XLK, XLE, XLV, XLI`.
30/30 requested names extracted; missing=none.
Train 2019-09-03–2019-09-13,
val 2019-09-16–2019-09-20,
test 2019-09-23–2019-09-27,
stress 2020-03-09, 2020-03-12, 2020-03-16 (external COVID dates, not selected
from imbalance outcomes).

## Moment restrictions (training days, native 1-minute grid after dropping 10 min open/close)

- `quadratic_c0`: a=0.2298/min (half-life 3 min), k=0.3219/min, q=0.2819, k-a=0.09213, admissible=True, mean_sse=0.001165, var_sse=0.0004334
- `independent_ou`: a=0.2298/min (half-life 3 min), k=0.2298/min, q=0.2819, k-a=0, admissible=True, mean_sse=0.001165, var_sse=0.0004335
- `flexible_moments`: a=0.2298/min (half-life 3 min), k=0.3219/min, q=0.2819, k-a=0.09213, admissible=True, mean_sse=0.001165, var_sse=0.0004334
- `fixed_center`: a=0.145/min (half-life 4.8 min), k=0.322/min, q=0.2819, k-a=0.177, admissible=True, mean_sse=0.001094, var_sse=0.0004334

The mean path is a weak, noisy oscillation around zero; the fitted exponential
relaxes in a few minutes. Cross-sectional variance is nearly flat, so `k` is
weakly identified. `k-a > 0` is **not** interpreted as causal interaction.

Observation spacings (same training windows):

- spacing 1 min: a=0.2298, k=0.3219, k-a=0.09213, admissible=True
- spacing 5 min: a=0.07563, k=1.641, k-a=1.565, admissible=True
- spacing 15 min: a=0.08378, k=0.009712, k-a=-0.07407, admissible=False
- spacing 30 min: a=0.46, k=0.46, k-a=1e-08, admissible=True

A single `(a, k)` pair does not describe both moments across spacings. The
15-minute grid is inadmissible (`k < a`).

## Bounded support

Empirical mass outside [-1, 1]: **0** (by construction for a
well-defined imbalance). Gaussian working measure P(|X|>1) ≈ **0.060**.
The affine model is therefore a leaky approximation, not a structurally
correct raw-state law.

## Forecasts (chronological split; day-clustered SE)

- val:factor_spy: mean W2 0.2012 (day-clustered SE 0.002414, 5 days)
- val:flexible_moments: mean W2 0.2012 (day-clustered SE 0.003977, 5 days)
- val:heterogeneous: mean W2 0.2103 (day-clustered SE 0.002656, 5 days)
- val:independent_ou: mean W2 0.2029 (day-clustered SE 0.003462, 5 days)
- val:persistence: mean W2 0.185 (day-clustered SE 0.004174, 5 days)
- val:quadratic_c0: mean W2 0.2085 (day-clustered SE 0.003508, 5 days)
- val:time_of_day: mean W2 0.1398 (day-clustered SE 0.001458, 5 days)
- test:factor_spy: mean W2 0.2079 (day-clustered SE 0.001906, 5 days)
- test:flexible_moments: mean W2 0.2075 (day-clustered SE 0.0009742, 5 days)
- test:heterogeneous: mean W2 0.2191 (day-clustered SE 0.002909, 5 days)
- test:independent_ou: mean W2 0.2119 (day-clustered SE 0.001784, 5 days)
- test:persistence: mean W2 0.2005 (day-clustered SE 0.001422, 5 days)
- test:quadratic_c0: mean W2 0.211 (day-clustered SE 0.00358, 5 days)
- test:time_of_day: mean W2 0.148 (day-clustered SE 0.002403, 5 days)
- stress:factor_spy: mean W2 0.209 (day-clustered SE 0.002955, 3 days)
- stress:flexible_moments: mean W2 0.2221 (day-clustered SE 0.003443, 3 days)
- stress:heterogeneous: mean W2 0.2272 (day-clustered SE 0.003186, 3 days)
- stress:independent_ou: mean W2 0.2092 (day-clustered SE 0.004103, 3 days)
- stress:persistence: mean W2 0.2064 (day-clustered SE 0.008899, 3 days)
- stress:quadratic_c0: mean W2 0.218 (day-clustered SE 0.006571, 3 days)
- stress:time_of_day: mean W2 0.1602 (day-clustered SE 0.001339, 3 days)

Origins and scored names are identical across models. Time-of-day uses the
**target** clock’s training histogram (no test-day values). The SPY factor
model residualizes on training betas and persists the origin proxy; it does
not use future SPY.

## Synthetic controls (pipeline check, not market evidence)

- Interacting DGP recovered k-a positive: **True**
  (a=0.02321, k=0.07556)
- Independent+factor DGP false-positive k-a: **False**
  (a=0.271, k=0.005164)

A false positive would mean `k-a` is observationally ambiguous. A true
negative on one simulated draw does **not** identify interaction in the
real panel.

## Decision gate

**NO-GO for manuscript inclusion**

Test W2 quadratic=0.211, persistence=0.2005, independent OU=0.2119. Gains are not reproducible across meaningful alternatives, or bounded-support / heterogeneity remain first-order. Document the failure; do not add a financial-validation sentence to rewrite.tex.

**Do not expand** the empirical study for the ALT manuscript. Preserve this
negative result as a research note. It does not belong in the main text. A
short appendix is optional only if the authors want to document that
liquidity-state imbalance is a poor coordinate for the quadratic energy;
otherwise neither.

Figures: `figs/empirical/fig1_moment_restrictions.png`,
`fig2_forecasts.png`, `fig3_stability.png`.
Commands: `python experiments/17_empirical_checkpoint.py` and
`python -m pytest tests/test_empirical.py`.

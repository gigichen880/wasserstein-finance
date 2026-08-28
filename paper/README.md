# Overleaf manuscript

The full rewrite is a single compilable file:

**[`rewrite.tex`](rewrite.tex)**

Upload that as the Overleaf main document, together with the figures in a
`figs/` folder next to it:

- `exp01_relaxation.png` … `exp08_directional_alignment.png`
- `exp09_convergence.png`
- `exp10_parameter_recovery.png`

In this repo the figures already live in `../figs/`; `rewrite.tex` looks on
`../figs/`, `figs/`, and `./`.

```bash
cd paper && latexmk -pdf rewrite.tex
```

Relative to `Wasserstein_finance.pdf`, the rewrite:

- studies a stylized potential mean-field inventory model motivated by market
  making, not a derivation from a general MFG;
- reads \(b\) as a cross-sectional dispersion penalty (not “crowding = similar inventories”);
- separates signed mass from positivity in the Eulerian scheme;
- reports a Gaussian refinement study (JKO error first-order in \(\tau\)) and a
  subset against the exact bimodal-mixture law; neither is a general
  equal-resolution theorem;
- treats piecewise-constant shocks as an exact mean ODE, with \(\mathcal{F}_0\) vs \(\mathcal{F}_{c_t}\);
- empirical program = moment restrictions + \(W_2\) forecasts + local alignment,
  including repeated-seed recovery and a nearby quartic falsification.

Theorem numbering matches the original (Remark 1, Definitions 2–3, Theorem 4, …, Theorem 9). Proofs are in the appendix. If you already have an Overleaf `.bib`, you can replace the `thebibliography` block with `\bibliography{...}`.

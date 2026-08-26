# Overleaf manuscript

The full rewrite is a single compilable file:

**[`rewrite.tex`](rewrite.tex)**

Upload that as the Overleaf main document, together with the eight figures
(`exp01_relaxation.png` … `exp08_directional_alignment.png`) in a `figs/`
folder next to it. In this repo the figures already live in `../figs/`;
`rewrite.tex` looks on both paths.

```bash
cd paper && latexmk -pdf rewrite.tex
```

Relative to `Wasserstein_finance.pdf`, the rewrite:

- states that the experiments are numerical verification, not market evidence;
- reads \(b\) as a cross-sectional dispersion penalty (not “crowding = similar inventories”);
- keeps JKO’s mass / positivity / energy dissipation without CFL, and drops the equal-resolution accuracy claim;
- replaces the empirical paragraph with moment restrictions + \(W_2\) forecasts + local alignment.

Theorem numbering matches the original (Remark 1, Definitions 2–3, Theorem 4, …, Theorem 9). Proofs are in the appendix. If you already have an Overleaf `.bib`, you can replace the `thebibliography` block with `\bibliography{...}`.

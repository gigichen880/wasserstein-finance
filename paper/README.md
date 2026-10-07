# Manuscript

ALT 2027 submission: [`rewrite.tex`](rewrite.tex) with [`references.bib`](references.bib), using [`alt2027.cls`](alt2027.cls).

```bash
cd paper && latexmk -pdf rewrite.tex
```

The class option `[anon]` withholds names for review. After acceptance, replace `[anon]` by `[final]`.

There is no page limit, but reviewers may stop after 12 pages of content excluding references. The main argument (gauge, weak identification, intervention, recovery) is in those first 12 pages; proofs and solver checks follow the bibliography.

An AISTATS 2027 cut is kept as [`rewrite_aistats.tex`](rewrite_aistats.tex) and is not the submission file.

Figures are read from `figs/` next to the file or from `../figs/`.

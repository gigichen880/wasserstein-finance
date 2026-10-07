# Empirical checkpoint

Prespecified pilot of a **liquidity-state** population (top-of-book imbalance),
not inventories.

```bash
python experiments/17_empirical_checkpoint.py
python -m pytest tests/test_empirical.py -v
```

Config: `config_pilot.json`. Cache: `cache/` (gitignored). Report: `report.md`.
Raw `data_by_stocks/` archives are licensed data and are gitignored.

Do not add claims to `paper/rewrite.tex` from this folder unless
`results/empirical/decision.json` records a pass.

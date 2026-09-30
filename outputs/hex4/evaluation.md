# Evaluation: 4 km² hexagons

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 152 of 5587 4 km² hexagons (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of 4 km² hexagons | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.508 ± 0.015 | 0.027 ± 0.002 | 0.094 ± 0.024 | 0.073 ± 0.006 |
| baseline_area_population | 0.676 ± 0.004 | 0.044 ± 0.001 | 0.137 ± 0.011 | 0.096 ± 0.006 |
| gradient_boosting | 0.717 ± 0.012 | 0.070 ± 0.007 | 0.322 ± 0.032 | 0.310 ± 0.024 |
| logistic | 0.713 ± 0.012 | 0.071 ± 0.004 | 0.345 ± 0.016 | 0.315 ± 0.022 |
| poisson_rate | 0.659 ± 0.021 | 0.054 ± 0.005 | 0.235 ± 0.026 | 0.203 ± 0.025 |

Selected model (best non-baseline on `capture_top10pct_area`): **logistic**.

## Does it beat size and density alone?

Permutation null for logistic: release labels shuffled 20 times among 4 km² hexagons in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.713 | 0.632 | 0.657 | 0.048 |
| avg_precision | 0.071 | 0.041 | 0.046 | 0.048 |
| capture_top10pct_units | 0.345 | 0.181 | 0.227 | 0.048 |
| capture_top10pct_area | 0.315 | 0.167 | 0.217 | 0.048 |

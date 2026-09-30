# Evaluation: 1 km² hexagons

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 167 of 21774 1 km² hexagons (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of 1 km² hexagons | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.504 ± 0.012 | 0.008 ± 0.000 | 0.101 ± 0.032 | 0.089 ± 0.029 |
| baseline_area_population | 0.657 ± 0.007 | 0.012 ± 0.000 | 0.142 ± 0.018 | 0.116 ± 0.017 |
| gradient_boosting | 0.815 ± 0.009 | 0.048 ± 0.005 | 0.502 ± 0.034 | 0.477 ± 0.036 |
| gradient_boosting+smooth | 0.809 ± 0.007 | 0.043 ± 0.006 | 0.494 ± 0.028 | 0.437 ± 0.031 |
| logistic | 0.835 ± 0.009 | 0.054 ± 0.003 | 0.544 ± 0.022 | 0.528 ± 0.018 |
| logistic+smooth | 0.824 ± 0.009 | 0.048 ± 0.003 | 0.509 ± 0.023 | 0.484 ± 0.024 |
| poisson_rate | 0.771 ± 0.020 | 0.037 ± 0.006 | 0.448 ± 0.041 | 0.411 ± 0.043 |
| poisson_rate+smooth | 0.744 ± 0.023 | 0.029 ± 0.006 | 0.371 ± 0.038 | 0.295 ± 0.042 |

Selected model (best non-baseline on `capture_top10pct_area`): **logistic**.

## Does it beat size and density alone?

Permutation null for logistic: release labels shuffled 20 times among 1 km² hexagons in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.835 | 0.595 | 0.626 | 0.048 |
| avg_precision | 0.054 | 0.010 | 0.012 | 0.048 |
| capture_top10pct_units | 0.544 | 0.128 | 0.170 | 0.048 |
| capture_top10pct_area | 0.528 | 0.130 | 0.160 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 77 1 km² hexagons whose first release was reported on or after it (only 1 km² hexagons without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| logistic | 0.789 | 0.024 | 0.442 | 0.442 |
| baseline_area | 0.482 | 0.003 | 0.104 | 0.104 |
| baseline_area_population | 0.651 | 0.005 | 0.156 | 0.104 |

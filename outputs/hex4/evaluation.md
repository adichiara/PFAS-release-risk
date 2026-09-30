# Evaluation: 4 km² hexagons

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 152 of 5587 4 km² hexagons (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of 4 km² hexagons | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.508 ± 0.015 | 0.027 ± 0.002 | 0.094 ± 0.024 | 0.073 ± 0.006 |
| baseline_area_population | 0.676 ± 0.004 | 0.044 ± 0.001 | 0.137 ± 0.011 | 0.096 ± 0.006 |
| gradient_boosting | 0.777 ± 0.013 | 0.096 ± 0.011 | 0.412 ± 0.037 | 0.395 ± 0.034 |
| gradient_boosting+smooth | 0.779 ± 0.011 | 0.107 ± 0.015 | 0.420 ± 0.030 | 0.388 ± 0.028 |
| logistic | 0.781 ± 0.011 | 0.118 ± 0.007 | 0.443 ± 0.017 | 0.428 ± 0.013 |
| logistic+smooth | 0.780 ± 0.009 | 0.120 ± 0.008 | 0.417 ± 0.021 | 0.410 ± 0.021 |
| poisson_rate | 0.733 ± 0.016 | 0.087 ± 0.014 | 0.338 ± 0.023 | 0.313 ± 0.033 |
| poisson_rate+smooth | 0.713 ± 0.016 | 0.075 ± 0.010 | 0.315 ± 0.027 | 0.260 ± 0.031 |

Selected model (best non-baseline on `capture_top10pct_area`): **logistic**.

## Does it beat size and density alone?

Permutation null for logistic: release labels shuffled 20 times among 4 km² hexagons in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.781 | 0.623 | 0.645 | 0.048 |
| avg_precision | 0.118 | 0.040 | 0.044 | 0.048 |
| capture_top10pct_units | 0.443 | 0.178 | 0.242 | 0.048 |
| capture_top10pct_area | 0.428 | 0.161 | 0.213 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 67 4 km² hexagons whose first release was reported on or after it (only 4 km² hexagons without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| logistic | 0.697 | 0.048 | 0.373 | 0.373 |
| baseline_area | 0.493 | 0.011 | 0.060 | 0.045 |
| baseline_area_population | 0.672 | 0.021 | 0.164 | 0.104 |

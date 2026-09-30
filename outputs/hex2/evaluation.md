# Evaluation: 2 km² hexagons

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 162 of 11024 2 km² hexagons (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of 2 km² hexagons | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.505 ± 0.011 | 0.015 ± 0.001 | 0.080 ± 0.022 | 0.075 ± 0.014 |
| baseline_area_population | 0.661 ± 0.003 | 0.024 ± 0.001 | 0.157 ± 0.017 | 0.106 ± 0.012 |
| gradient_boosting | 0.813 ± 0.008 | 0.068 ± 0.008 | 0.456 ± 0.034 | 0.442 ± 0.041 |
| gradient_boosting+smooth | 0.808 ± 0.009 | 0.070 ± 0.004 | 0.446 ± 0.025 | 0.436 ± 0.023 |
| logistic | 0.819 ± 0.006 | 0.082 ± 0.004 | 0.491 ± 0.026 | 0.482 ± 0.019 |
| logistic+smooth | 0.813 ± 0.007 | 0.082 ± 0.005 | 0.476 ± 0.023 | 0.451 ± 0.024 |
| poisson_rate | 0.762 ± 0.011 | 0.062 ± 0.010 | 0.399 ± 0.021 | 0.371 ± 0.027 |
| poisson_rate+smooth | 0.741 ± 0.013 | 0.052 ± 0.009 | 0.362 ± 0.019 | 0.313 ± 0.029 |

Selected model (best non-baseline on `capture_top10pct_area`): **logistic**.

## Does it beat size and density alone?

Permutation null for logistic: release labels shuffled 20 times among 2 km² hexagons in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.819 | 0.613 | 0.639 | 0.048 |
| avg_precision | 0.082 | 0.022 | 0.027 | 0.048 |
| capture_top10pct_units | 0.491 | 0.168 | 0.206 | 0.048 |
| capture_top10pct_area | 0.482 | 0.161 | 0.197 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 77 2 km² hexagons whose first release was reported on or after it (only 2 km² hexagons without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| logistic | 0.759 | 0.031 | 0.351 | 0.338 |
| baseline_area | 0.501 | 0.007 | 0.078 | 0.078 |
| baseline_area_population | 0.664 | 0.012 | 0.143 | 0.091 |

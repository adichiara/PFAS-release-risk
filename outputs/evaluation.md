# Evaluation

Run 2026-09-29 on `massdep_pfas_releases_2021-11-07.csv`: 51 located releases in 42 of 5109 block groups (1 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over block groups, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of block groups | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.793 ± 0.006 | 0.032 ± 0.006 | 0.396 ± 0.015 | 0.180 ± 0.008 |
| baseline_area_population | 0.786 ± 0.007 | 0.030 ± 0.006 | 0.355 ± 0.017 | 0.190 ± 0.013 |
| gradient_boosting | 0.766 ± 0.016 | 0.052 ± 0.018 | 0.359 ± 0.061 | 0.261 ± 0.033 |
| logistic | 0.747 ± 0.020 | 0.028 ± 0.003 | 0.406 ± 0.063 | 0.233 ± 0.041 |
| poisson_rate | 0.796 ± 0.003 | 0.028 ± 0.002 | 0.400 ± 0.030 | 0.125 ± 0.065 |

Selected model (best non-baseline on `capture_top10pct_area`): **gradient_boosting**.

## Does it beat size and density alone?

Permutation null for gradient_boosting: release labels shuffled 20 times among block groups in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.766 | 0.677 | 0.739 | 0.095 |
| avg_precision | 0.052 | 0.020 | 0.030 | 0.048 |
| capture_top10pct_units | 0.359 | 0.253 | 0.359 | 0.095 |
| capture_top10pct_area | 0.261 | 0.145 | 0.257 | 0.095 |

# Evaluation: block groups

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 140 of 5109 block groups (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of block groups | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.826 ± 0.001 | 0.090 ± 0.001 | 0.416 ± 0.016 | 0.131 ± 0.007 |
| baseline_area_population | 0.823 ± 0.001 | 0.089 ± 0.002 | 0.379 ± 0.011 | 0.157 ± 0.007 |
| gradient_boosting | 0.860 ± 0.006 | 0.171 ± 0.014 | 0.554 ± 0.044 | 0.363 ± 0.026 |
| gradient_boosting+smooth | 0.858 ± 0.006 | 0.146 ± 0.010 | 0.534 ± 0.041 | 0.354 ± 0.028 |
| logistic | 0.870 ± 0.009 | 0.178 ± 0.010 | 0.560 ± 0.023 | 0.351 ± 0.010 |
| logistic+smooth | 0.866 ± 0.009 | 0.137 ± 0.006 | 0.579 ± 0.023 | 0.295 ± 0.015 |
| poisson_rate | 0.834 ± 0.001 | 0.095 ± 0.001 | 0.423 ± 0.016 | 0.353 ± 0.028 |
| poisson_rate+smooth | 0.828 ± 0.001 | 0.092 ± 0.001 | 0.418 ± 0.011 | 0.164 ± 0.010 |

Selected model (best non-baseline on `capture_top10pct_area`): **gradient_boosting**.

## Does it beat size and density alone?

Permutation null for gradient_boosting: release labels shuffled 20 times among block groups in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.860 | 0.772 | 0.789 | 0.048 |
| avg_precision | 0.171 | 0.073 | 0.086 | 0.048 |
| capture_top10pct_units | 0.554 | 0.299 | 0.351 | 0.048 |
| capture_top10pct_area | 0.363 | 0.197 | 0.239 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 61 block groups whose first release was reported on or after it (only block groups without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| gradient_boosting | 0.825 | 0.066 | 0.492 | 0.344 |
| baseline_area | 0.824 | 0.042 | 0.410 | 0.147 |
| baseline_area_population | 0.818 | 0.040 | 0.377 | 0.164 |

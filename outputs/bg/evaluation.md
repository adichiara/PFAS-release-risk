# Evaluation: block groups

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 140 of 5109 block groups (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of block groups | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.826 ± 0.001 | 0.090 ± 0.001 | 0.416 ± 0.016 | 0.131 ± 0.007 |
| baseline_area_population | 0.823 ± 0.001 | 0.089 ± 0.002 | 0.379 ± 0.011 | 0.157 ± 0.007 |
| gradient_boosting | 0.841 ± 0.005 | 0.125 ± 0.010 | 0.451 ± 0.030 | 0.310 ± 0.032 |
| gradient_boosting+smooth | 0.843 ± 0.004 | 0.119 ± 0.009 | 0.438 ± 0.024 | 0.286 ± 0.024 |
| logistic | 0.842 ± 0.007 | 0.116 ± 0.006 | 0.493 ± 0.020 | 0.290 ± 0.018 |
| logistic+smooth | 0.839 ± 0.007 | 0.107 ± 0.005 | 0.457 ± 0.016 | 0.258 ± 0.019 |
| poisson_rate | 0.830 ± 0.001 | 0.091 ± 0.001 | 0.419 ± 0.008 | 0.273 ± 0.033 |
| poisson_rate+smooth | 0.824 ± 0.001 | 0.090 ± 0.001 | 0.403 ± 0.016 | 0.153 ± 0.009 |

Selected model (best non-baseline on `capture_top10pct_area`): **gradient_boosting**.

## Does it beat size and density alone?

Permutation null for gradient_boosting: release labels shuffled 20 times among block groups in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.841 | 0.773 | 0.790 | 0.048 |
| avg_precision | 0.125 | 0.074 | 0.083 | 0.048 |
| capture_top10pct_units | 0.451 | 0.307 | 0.362 | 0.048 |
| capture_top10pct_area | 0.310 | 0.195 | 0.264 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 61 block groups whose first release was reported on or after it (only block groups without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| gradient_boosting | 0.801 | 0.042 | 0.328 | 0.279 |
| baseline_area | 0.824 | 0.042 | 0.410 | 0.147 |
| baseline_area_population | 0.818 | 0.040 | 0.377 | 0.164 |

# Evaluation

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 140 of 5109 block groups (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over block groups, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of block groups | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.826 ± 0.001 | 0.090 ± 0.001 | 0.416 ± 0.016 | 0.131 ± 0.007 |
| baseline_area_population | 0.823 ± 0.001 | 0.089 ± 0.002 | 0.379 ± 0.011 | 0.157 ± 0.007 |
| gradient_boosting | 0.838 ± 0.006 | 0.111 ± 0.004 | 0.436 ± 0.038 | 0.316 ± 0.022 |
| logistic | 0.835 ± 0.005 | 0.100 ± 0.005 | 0.418 ± 0.024 | 0.274 ± 0.015 |
| poisson_rate | 0.827 ± 0.001 | 0.090 ± 0.001 | 0.409 ± 0.006 | 0.199 ± 0.050 |

Selected model (best non-baseline on `capture_top10pct_area`): **gradient_boosting**.

## Does it beat size and density alone?

Permutation null for gradient_boosting: release labels shuffled 20 times among block groups in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.838 | 0.774 | 0.791 | 0.048 |
| avg_precision | 0.111 | 0.074 | 0.087 | 0.048 |
| capture_top10pct_units | 0.436 | 0.302 | 0.377 | 0.048 |
| capture_top10pct_area | 0.316 | 0.196 | 0.268 | 0.048 |

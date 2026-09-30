# Evaluation: 4 km² hexagons

Run 2026-09-30 on `massdep_pfas_releases.csv`: 194 located releases in 152 of 5587 4 km² hexagons (20 could not be located).

Town-grouped 5-fold cross-validation, 10 repeats (mean ± sd). Compare models with the two baselines, not with 10%: releases are not spread evenly over land or over units, so chance capture depends on the budget.

| model | ROC AUC | avg precision | releases in top 10% of 4 km² hexagons | releases in top-risk 10% of land |
|---|---|---|---|---|
| baseline_area | 0.508 ± 0.015 | 0.027 ± 0.002 | 0.094 ± 0.024 | 0.073 ± 0.006 |
| baseline_area_population | 0.676 ± 0.004 | 0.044 ± 0.001 | 0.137 ± 0.011 | 0.096 ± 0.006 |
| gradient_boosting | 0.717 ± 0.012 | 0.070 ± 0.007 | 0.322 ± 0.032 | 0.310 ± 0.024 |
| gradient_boosting+smooth | 0.725 ± 0.010 | 0.080 ± 0.008 | 0.342 ± 0.030 | 0.326 ± 0.032 |
| logistic | 0.713 ± 0.012 | 0.071 ± 0.004 | 0.345 ± 0.016 | 0.315 ± 0.022 |
| logistic+smooth | 0.711 ± 0.011 | 0.077 ± 0.005 | 0.356 ± 0.017 | 0.336 ± 0.018 |
| poisson_rate | 0.659 ± 0.021 | 0.054 ± 0.005 | 0.235 ± 0.026 | 0.203 ± 0.025 |
| poisson_rate+smooth | 0.646 ± 0.021 | 0.050 ± 0.004 | 0.218 ± 0.028 | 0.169 ± 0.024 |

Selected model (best non-baseline on `capture_top10pct_area`): **logistic+smooth**.

## Does it beat size and density alone?

Permutation null for logistic+smooth: release labels shuffled 20 times among 4 km² hexagons in the same land-area x population-density quintile, then the same cross-validation. This keeps the size and density effects and removes everything else.

| metric | observed | null mean | null 95th pct | p |
|---|---|---|---|---|
| roc_auc | 0.711 | 0.639 | 0.666 | 0.048 |
| avg_precision | 0.077 | 0.042 | 0.045 | 0.048 |
| capture_top10pct_units | 0.356 | 0.180 | 0.217 | 0.048 |
| capture_top10pct_area | 0.336 | 0.164 | 0.222 | 0.048 |

## Does it predict new reports?

Trained on the 104 releases reported before 2023-01-01; scored on the 67 4 km² hexagons whose first release was reported on or after it (only 4 km² hexagons without an earlier release are scored).

| model | roc_auc | avg_precision | capture_top10pct_units | capture_top10pct_area |
|---|---|---|---|---|
| logistic+smooth | 0.682 | 0.031 | 0.284 | 0.224 |
| baseline_area | 0.493 | 0.011 | 0.060 | 0.045 |
| baseline_area_population | 0.672 | 0.021 | 0.164 | 0.104 |

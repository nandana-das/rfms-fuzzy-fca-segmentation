# Corrected membership-function ablation smoke results

This report supplements, and does not overwrite, the earlier smoke report in `docs/MEMBERSHIP_FUNCTION_ABLATION_RESULTS.md`. The corrected paired-bootstrap artifacts are under `results/membership_function_ablation/corrected_smoke/`; the pooling-rule rerun is under `results/membership_function_ablation/corrected_pooling_smoke/`.

## Corrections applied

- Quantile cutoffs and scores now explicitly reproduce the frozen baseline's training-only `np.quantile` plus `searchsorted(..., side="left")` rule, including its deterministic rejection of repeated cutpoints. A tied-value regression compares both cutoffs and scores directly with the frozen implementation.
- One customer bootstrap index matrix is generated per cohort and reused by M0, M1, M2, and M3. Metric differences are formed from the resulting paired draws before confidence intervals and p-values are calculated.
- The 365-day Online Retail II primary cohort remains in `per_origin_metrics.csv` and individual comparison rows, but pooled Online Retail II deltas, intervals, bootstrap p-values, and Holm adjustment use only origins explicitly marked `pooled=True`.

## Corrected smoke

The smoke covers Dunnhumby day 347, 2,497 customers, five shared customer folds, and 200 bootstrap draws.

| Method | ROC AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| M0 frozen piecewise | 0.8965 | 0.5499 | 0.6244 |
| M1 normalized Gaussian | 0.8912 | 0.5502 | 0.6187 |
| M2 normalized generalized bell | 0.8911 | 0.5441 | 0.6130 |
| M3 log-coordinate Gaussian | 0.8953 | 0.5470 | 0.6225 |

Corrected paired comparisons against M0:

| Comparison | Metric | Delta | 95% CI | Bootstrap p | Holm p |
|---|---|---:|---|---:|---:|
| M1 − M0 | AUC | −0.0053 | [−0.0136, 0.0027] | 0.16 | 0.72 |
| M1 − M0 | Spend R² | +0.0002 | [−0.0039, 0.0046] | 0.90 | 1.00 |
| M1 − M0 | Invoice R² | −0.0058 | [−0.0104, −0.0008] | 0.04 | 0.32 |
| M2 − M0 | AUC | −0.0053 | [−0.0133, 0.0023] | 0.14 | 0.72 |
| M2 − M0 | Spend R² | −0.0058 | [−0.0110, 0.0000] | 0.06 | 0.42 |
| M2 − M0 | Invoice R² | −0.0115 | [−0.0165, −0.0064] | 0.005 | 0.045 |
| M3 − M0 | AUC | −0.0011 | [−0.0051, 0.0032] | 0.71 | 1.00 |
| M3 − M0 | Spend R² | −0.0029 | [−0.0069, 0.0005] | 0.12 | 0.72 |
| M3 − M0 | Invoice R² | −0.0019 | [−0.0046, 0.0005] | 0.12 | 0.72 |

These are smoke diagnostics with 200 draws and one origin, not full-evaluation evidence. The full evaluation remains blocked pending review.

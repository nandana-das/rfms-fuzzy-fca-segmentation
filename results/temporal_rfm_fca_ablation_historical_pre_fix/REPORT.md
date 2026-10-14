# Temporal RFM-FCA ablation

Three-arm rolling-origin ablation using only origin-safe temporal features. Existing Step 0 outputs were not modified.

## Definitions and leakage rules

- R = days since last observed purchase; F = unique invoices/baskets; M = observed monetary value.
- T = mean interval between consecutive distinct observed purchase dates. T is missing when fewer than two distinct dates exist; missing T is imputed with the training-fold median only.
- `repeat_history` = 1 when at least two distinct observed purchase dates exist, otherwise 0. The T arm therefore includes both interval and repeat-history information.
- L = days from first observed purchase date to the prediction origin.
- Future outcomes use only the subsequent 91-day holdout. Cutpoints, centroids, concept mining, redundancy suppression, imputation, and learners are fit inside each customer training fold.

## Arms

| Arm | Fuzzy attributes |
|---|---|
| Fuzzy RFM | R, F, M; existing M0 builder |
| Fuzzy RFM+T | R, F, M, T, repeat_history |
| Fuzzy RFM+T+L | R, F, M, T, repeat_history, L |

## Pooled means

| dataset | arm | auc | spend_r2 | invoice_r2 |
|---|---|---:|---:|---:|
| Dunnhumby | fuzzy_rfm | 0.8899 | 0.5382 | 0.6227 |
| Dunnhumby | fuzzy_rfm_t | 0.8870 | 0.5347 | 0.6227 |
| Dunnhumby | fuzzy_rfm_t_l | 0.8867 | 0.5290 | 0.6187 |
| Online Retail II | fuzzy_rfm | 0.7799 | 0.3114 | 0.3737 |
| Online Retail II | fuzzy_rfm_t | 0.7875 | 0.3224 | 0.4084 |
| Online Retail II | fuzzy_rfm_t_l | 0.7881 | 0.3232 | 0.4112 |

## Per-origin metrics

The complete per-origin table is also written to `per_origin_metrics.csv`.

| dataset | origin | arm | auc | spend_r2 | invoice_r2 |
|---|---|---|---:|---:|---:|
| Dunnhumby | day 347 | fuzzy_rfm | 0.8965 | 0.5499 | 0.6244 |
| Dunnhumby | day 347 | fuzzy_rfm_t | 0.8919 | 0.5469 | 0.6263 |
| Dunnhumby | day 347 | fuzzy_rfm_t_l | 0.8947 | 0.5422 | 0.6218 |
| Dunnhumby | day 438 | fuzzy_rfm | 0.8892 | 0.5481 | 0.6299 |
| Dunnhumby | day 438 | fuzzy_rfm_t | 0.8904 | 0.5461 | 0.6294 |
| Dunnhumby | day 438 | fuzzy_rfm_t_l | 0.8898 | 0.5379 | 0.6246 |
| Dunnhumby | day 529 | fuzzy_rfm | 0.9005 | 0.5378 | 0.6166 |
| Dunnhumby | day 529 | fuzzy_rfm_t | 0.8969 | 0.5350 | 0.6179 |
| Dunnhumby | day 529 | fuzzy_rfm_t_l | 0.8908 | 0.5326 | 0.6143 |
| Dunnhumby | day 620 | fuzzy_rfm | 0.8736 | 0.5169 | 0.6197 |
| Dunnhumby | day 620 | fuzzy_rfm_t | 0.8688 | 0.5108 | 0.6172 |
| Dunnhumby | day 620 | fuzzy_rfm_t_l | 0.8713 | 0.5032 | 0.6142 |
| Online Retail II | 2010-09-10 | fuzzy_rfm | 0.7064 | 0.2324 | 0.3157 |
| Online Retail II | 2010-09-10 | fuzzy_rfm_t | 0.7282 | 0.2323 | 0.3275 |
| Online Retail II | 2010-09-10 | fuzzy_rfm_t_l | 0.7292 | 0.2337 | 0.3384 |
| Online Retail II | 2010-12-10 | fuzzy_rfm | 0.7958 | 0.3197 | 0.3697 |
| Online Retail II | 2010-12-10 | fuzzy_rfm_t | 0.7980 | 0.3273 | 0.4001 |
| Online Retail II | 2010-12-10 | fuzzy_rfm_t_l | 0.7967 | 0.3271 | 0.4024 |
| Online Retail II | 2011-03-11 | fuzzy_rfm | 0.7857 | 0.3164 | 0.3806 |
| Online Retail II | 2011-03-11 | fuzzy_rfm_t | 0.7890 | 0.3296 | 0.4245 |
| Online Retail II | 2011-03-11 | fuzzy_rfm_t_l | 0.7891 | 0.3309 | 0.4261 |
| Online Retail II | 2011-06-10 | fuzzy_rfm | 0.8181 | 0.3619 | 0.4119 |
| Online Retail II | 2011-06-10 | fuzzy_rfm_t | 0.8236 | 0.3819 | 0.4607 |
| Online Retail II | 2011-06-10 | fuzzy_rfm_t_l | 0.8252 | 0.3815 | 0.4589 |
| Online Retail II | 2011-09-09 | fuzzy_rfm | 0.7933 | 0.3266 | 0.3904 |
| Online Retail II | 2011-09-09 | fuzzy_rfm_t | 0.7986 | 0.3409 | 0.4294 |
| Online Retail II | 2011-09-09 | fuzzy_rfm_t_l | 0.8006 | 0.3426 | 0.4304 |
| Online Retail II | 2010-12-09 (primary, 365d) | fuzzy_rfm | 0.7848 | 0.3519 | 0.4621 |
| Online Retail II | 2010-12-09 (primary, 365d) | fuzzy_rfm_t | 0.7852 | 0.3530 | 0.4802 |
| Online Retail II | 2010-12-09 (primary, 365d) | fuzzy_rfm_t_l | 0.7853 | 0.3511 | 0.4782 |

## Paired comparisons

Delta = later arm minus earlier arm; intervals are paired customer-bootstrap percentile intervals and p-values are Holm-adjusted within dataset across the six incremental metric tests.

| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |
|---|---|---|---:|---:|---:|---:|---|
| Dunnhumby | fuzzy_rfm_t - fuzzy_rfm | auc | -0.0029 | -0.0062 | +0.0003 | 0.3120 | 1/4 |
| Dunnhumby | fuzzy_rfm_t - fuzzy_rfm | spend_r2 | -0.0035 | -0.0053 | -0.0017 | 0.0045 | 0/4 |
| Dunnhumby | fuzzy_rfm_t - fuzzy_rfm | invoice_r2 | +0.0001 | -0.0017 | +0.0018 | 1.0000 | 2/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm_t | auc | -0.0003 | -0.0043 | +0.0035 | 1.0000 | 2/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm_t | spend_r2 | -0.0057 | -0.0085 | -0.0028 | 0.0045 | 0/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm_t | invoice_r2 | -0.0040 | -0.0061 | -0.0019 | 0.0045 | 0/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm | auc | -0.0033 | -0.0076 | +0.0007 | 0.3330 | 1/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm | spend_r2 | -0.0092 | -0.0123 | -0.0059 | 0.0045 | 0/4 |
| Dunnhumby | fuzzy_rfm_t_l - fuzzy_rfm | invoice_r2 | -0.0039 | -0.0067 | -0.0010 | 0.0200 | 0/4 |
| Online Retail II | fuzzy_rfm_t - fuzzy_rfm | auc | +0.0076 | +0.0058 | +0.0095 | 0.0045 | 5/5 |
| Online Retail II | fuzzy_rfm_t - fuzzy_rfm | spend_r2 | +0.0110 | +0.0086 | +0.0136 | 0.0045 | 4/5 |
| Online Retail II | fuzzy_rfm_t - fuzzy_rfm | invoice_r2 | +0.0348 | +0.0302 | +0.0399 | 0.0045 | 5/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm_t | auc | +0.0007 | -0.0004 | +0.0018 | 0.4000 | 4/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm_t | spend_r2 | +0.0008 | -0.0008 | +0.0024 | 0.4000 | 3/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm_t | invoice_r2 | +0.0028 | +0.0007 | +0.0050 | 0.0390 | 4/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm | auc | +0.0083 | +0.0062 | +0.0104 | 0.0045 | 5/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm | spend_r2 | +0.0118 | +0.0089 | +0.0146 | 0.0045 | 5/5 |
| Online Retail II | fuzzy_rfm_t_l - fuzzy_rfm | invoice_r2 | +0.0376 | +0.0325 | +0.0427 | 0.0045 | 5/5 |

## Conclusion

Under this protocol, T has opposite dataset-specific effects: it significantly reduces Dunnhumby Spend R² (Holm p=0.0045) and does not improve the other Dunnhumby outcomes, while it significantly improves all three pooled Online Retail II outcomes (Holm p=0.0045). Adding L after T does not improve Dunnhumby and adds only a small significant Online Retail II Invoice R² increment (Holm p=0.039); its AUC and Spend R² increments are not significant. Therefore T and L do not provide a uniform predictive improvement, and the result is exploratory and limited to these datasets and learners.

## Outputs and command

- Command: `python scripts\temporal_rfm_fca_ablation.py`
- Outputs: `results/temporal_rfm_fca_ablation/`
- Existing Step 0 outputs and manuscript files were not changed.
- Runtime: 26.8 minutes.

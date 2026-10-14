# Non-FCA baselines for the temporal ablation

Exploratory, post hoc follow-up. Tests whether the gain from adding T to fuzzy RFM-FCA also appears when T is added to plain log-RFM and spline-RFM models without FCA.

All arms use the temporal-ablation cohorts, 91-day holdouts, customer folds (seed 1), learners (LogisticRegressionCV / RidgeCV), and paired customer bootstrap (B=2000). T is imputed with the training-fold median and every transform is fit on the training fold only. L is out of scope.

Pooled intervals average per-origin bootstrap draws; origins overlap in customers, so pooled intervals are approximate. Single CV seed.

## Pooled means

| dataset | arm | features | auc | spend_r2 | invoice_r2 |
|---|---|---:|---:|---:|---:|
| Dunnhumby | fuzzy_rfm_t | 170.7 | 0.8856 | 0.5345 | 0.6237 |
| Dunnhumby | spline_log_rfm_t | 29.0 | 0.8948 | 0.5423 | 0.6282 |
| Online Retail II | fuzzy_rfm_t | 161.8 | 0.7869 | 0.3221 | 0.4081 |
| Online Retail II | spline_log_rfm_t | 29.0 | 0.7832 | 0.3123 | 0.4085 |

## Pooled paired comparisons

Delta = first arm minus second arm. Holm within dataset over the 3 pooled tests.

| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |
|---|---|---|---:|---:|---:|---:|---|
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | auc | -0.0092 | -0.0141 | -0.0045 | 0.0015 | 0/4 |
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | spend_r2 | -0.0078 | -0.0123 | -0.0029 | 0.0040 | 1/4 |
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | invoice_r2 | -0.0045 | -0.0084 | -0.0008 | 0.0160 | 1/4 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | auc | +0.0037 | +0.0017 | +0.0056 | 0.0015 | 5/5 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | spend_r2 | +0.0098 | +0.0064 | +0.0131 | 0.0015 | 5/5 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | invoice_r2 | -0.0004 | -0.0084 | +0.0078 | 0.9140 | 3/5 |

Per-origin metrics and comparisons: `per_origin_metrics.csv`, `paired_comparisons.csv`.

Command: `python scripts/temporal_nonfca_baseline.py --cv-seed 1 --arms fuzzy_rfm_t spline_log_rfm_t --pairs fuzzy_rfm_t:spline_log_rfm_t --out results/temporal_second_seed`
Runtime: 11.0 minutes.

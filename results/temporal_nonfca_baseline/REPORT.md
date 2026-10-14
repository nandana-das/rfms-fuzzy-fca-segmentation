# Non-FCA baselines for the temporal ablation

Exploratory, post hoc follow-up. Tests whether the gain from adding T to fuzzy RFM-FCA also appears when T is added to plain log-RFM and spline-RFM models without FCA.

All arms use the temporal-ablation cohorts, 91-day holdouts, customer folds (seed 0), learners (LogisticRegressionCV / RidgeCV), and paired customer bootstrap (B=2000). T is imputed with the training-fold median and every transform is fit on the training fold only. L is out of scope.

Pooled intervals average per-origin bootstrap draws; origins overlap in customers, so pooled intervals are approximate. Single CV seed.

## Pooled means

| dataset | arm | features | auc | spend_r2 | invoice_r2 |
|---|---|---:|---:|---:|---:|
| Dunnhumby | log_rfm | 3.0 | 0.8917 | 0.5195 | 0.6066 |
| Dunnhumby | spline_log_rfm | 21.0 | 0.8950 | 0.5442 | 0.6257 |
| Dunnhumby | log_rfm_t | 5.0 | 0.8931 | 0.5224 | 0.6133 |
| Dunnhumby | spline_log_rfm_t | 29.0 | 0.8949 | 0.5446 | 0.6274 |
| Dunnhumby | fuzzy_rfm | 91.6 | 0.8899 | 0.5382 | 0.6227 |
| Dunnhumby | fuzzy_rfm_t | 171.5 | 0.8870 | 0.5347 | 0.6227 |
| Online Retail II | log_rfm | 3.0 | 0.7809 | 0.2968 | 0.3702 |
| Online Retail II | spline_log_rfm | 21.0 | 0.7805 | 0.3097 | 0.4054 |
| Online Retail II | log_rfm_t | 5.0 | 0.7807 | 0.3020 | 0.3914 |
| Online Retail II | spline_log_rfm_t | 29.0 | 0.7834 | 0.3134 | 0.4094 |
| Online Retail II | fuzzy_rfm | 76.8 | 0.7799 | 0.3114 | 0.3737 |
| Online Retail II | fuzzy_rfm_t | 162.9 | 0.7875 | 0.3224 | 0.4084 |

## Pooled paired comparisons

Delta = first arm minus second arm. Holm within dataset over the 12 pooled tests.

| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |
|---|---|---|---:|---:|---:|---:|---|
| Dunnhumby | fuzzy_rfm_t - log_rfm_t | auc | -0.0061 | -0.0126 | +0.0001 | 0.2950 | 0/4 |
| Dunnhumby | fuzzy_rfm_t - log_rfm_t | spend_r2 | +0.0123 | +0.0056 | +0.0192 | 0.0060 | 4/4 |
| Dunnhumby | fuzzy_rfm_t - log_rfm_t | invoice_r2 | +0.0094 | +0.0041 | +0.0145 | 0.0060 | 4/4 |
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | auc | -0.0079 | -0.0126 | -0.0034 | 0.0080 | 0/4 |
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | spend_r2 | -0.0099 | -0.0143 | -0.0053 | 0.0060 | 0/4 |
| Dunnhumby | fuzzy_rfm_t - spline_log_rfm_t | invoice_r2 | -0.0047 | -0.0088 | -0.0009 | 0.1050 | 0/4 |
| Dunnhumby | log_rfm_t - log_rfm | auc | +0.0013 | -0.0007 | +0.0034 | 0.5970 | 3/4 |
| Dunnhumby | log_rfm_t - log_rfm | spend_r2 | +0.0029 | -0.0007 | +0.0066 | 0.4280 | 4/4 |
| Dunnhumby | log_rfm_t - log_rfm | invoice_r2 | +0.0067 | +0.0037 | +0.0101 | 0.0060 | 4/4 |
| Dunnhumby | spline_log_rfm_t - spline_log_rfm | auc | -0.0001 | -0.0008 | +0.0007 | 1.0000 | 3/4 |
| Dunnhumby | spline_log_rfm_t - spline_log_rfm | spend_r2 | +0.0004 | -0.0014 | +0.0023 | 1.0000 | 2/4 |
| Dunnhumby | spline_log_rfm_t - spline_log_rfm | invoice_r2 | +0.0017 | +0.0001 | +0.0034 | 0.2160 | 3/4 |
| Online Retail II | fuzzy_rfm_t - log_rfm_t | auc | +0.0067 | +0.0048 | +0.0086 | 0.0060 | 5/5 |
| Online Retail II | fuzzy_rfm_t - log_rfm_t | spend_r2 | +0.0204 | +0.0161 | +0.0247 | 0.0060 | 4/5 |
| Online Retail II | fuzzy_rfm_t - log_rfm_t | invoice_r2 | +0.0170 | +0.0102 | +0.0239 | 0.0060 | 4/5 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | auc | +0.0040 | +0.0022 | +0.0059 | 0.0060 | 4/5 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | spend_r2 | +0.0090 | +0.0057 | +0.0123 | 0.0060 | 4/5 |
| Online Retail II | fuzzy_rfm_t - spline_log_rfm_t | invoice_r2 | -0.0010 | -0.0084 | +0.0070 | 1.0000 | 3/5 |
| Online Retail II | log_rfm_t - log_rfm | auc | -0.0001 | -0.0008 | +0.0005 | 1.0000 | 3/5 |
| Online Retail II | log_rfm_t - log_rfm | spend_r2 | +0.0052 | +0.0037 | +0.0067 | 0.0060 | 5/5 |
| Online Retail II | log_rfm_t - log_rfm | invoice_r2 | +0.0213 | +0.0182 | +0.0240 | 0.0060 | 5/5 |
| Online Retail II | spline_log_rfm_t - spline_log_rfm | auc | +0.0029 | +0.0013 | +0.0047 | 0.0060 | 4/5 |
| Online Retail II | spline_log_rfm_t - spline_log_rfm | spend_r2 | +0.0037 | +0.0021 | +0.0054 | 0.0060 | 5/5 |
| Online Retail II | spline_log_rfm_t - spline_log_rfm | invoice_r2 | +0.0040 | +0.0023 | +0.0057 | 0.0060 | 5/5 |

Per-origin metrics and comparisons: `per_origin_metrics.csv`, `paired_comparisons.csv`.

Command: `python scripts/temporal_nonfca_baseline.py`
Runtime: 14.9 minutes.

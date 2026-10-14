# Exploratory LRFM extension results

This analysis is exploratory and cannot confirm prior findings. Non-significance is not evidence of equivalence.

## Locked protocol

- CV fold seeds: [0, 1, 2, 3, 4]; customer-clustered paired bootstrap: 10,000 draws per seed.
- The reported mixture interval and two-sided raw p-value use all 50,000 draws; Holm correction is within each dataset over 15 tests.
- Estimates are dataset-specific and are not pooled across datasets. The table reports the mean over seeds, seed range, 95% percentile interval, raw p-value, and Holm-adjusted p-value.

## Inference results

| dataset | id | comparison | metric | delta_mean_over_seeds | delta_seed0 | delta_min_seed | delta_max_seed | ci_low | ci_high | p_boot | seed0_reproduction_max_abs_diff | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDNOW | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | auc | 0.0038 | 0.0054 | 0.0000 | 0.0060 | -0.0011 | 0.0071 | 0.2055 | 0.0000 | 1.0000 |
| CDNOW | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | spend_r2 | 0.0185 | 0.0185 | 0.0181 | 0.0189 | 0.0158 | 0.0211 | 0.0000 | 0.0000 | 0.0003 |
| CDNOW | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | invoice_r2 | 0.0257 | 0.0257 | 0.0254 | 0.0264 | 0.0223 | 0.0292 | 0.0000 | 0.0000 | 0.0003 |
| CDNOW | E2 | fuzzy_lrfm_fca - spline_log_lrfm | auc | 0.0005 | 0.0024 | -0.0017 | 0.0024 | -0.0029 | 0.0034 | 0.7846 | 0.0000 | 1.0000 |
| CDNOW | E2 | fuzzy_lrfm_fca - spline_log_lrfm | spend_r2 | -0.0091 | -0.0091 | -0.0094 | -0.0087 | -0.0129 | -0.0052 | 0.0000 | 0.0000 | 0.0003 |
| CDNOW | E2 | fuzzy_lrfm_fca - spline_log_lrfm | invoice_r2 | -0.0318 | -0.0323 | -0.0323 | -0.0311 | -0.0418 | -0.0225 | 0.0000 | 0.0000 | 0.0003 |
| CDNOW | E3 | hybrid_lrfm - spline_log_lrfm | auc | 0.0030 | 0.0027 | 0.0021 | 0.0042 | 0.0012 | 0.0051 | 0.0003 | 0.0000 | 0.0028 |
| CDNOW | E3 | hybrid_lrfm - spline_log_lrfm | spend_r2 | 0.0045 | 0.0046 | 0.0043 | 0.0046 | 0.0029 | 0.0062 | 0.0000 | 0.0000 | 0.0003 |
| CDNOW | E3 | hybrid_lrfm - spline_log_lrfm | invoice_r2 | 0.0027 | 0.0027 | 0.0024 | 0.0028 | 0.0007 | 0.0048 | 0.0098 | 0.0000 | 0.0886 |
| CDNOW | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | auc | 0.0003 | -0.0004 | -0.0012 | 0.0042 | -0.0017 | 0.0048 | 0.5382 | 0.0000 | 1.0000 |
| CDNOW | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | spend_r2 | 0.0002 | 0.0000 | 0.0000 | 0.0005 | -0.0004 | 0.0008 | 0.5618 | 0.0000 | 1.0000 |
| CDNOW | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | invoice_r2 | 0.0007 | 0.0004 | 0.0004 | 0.0011 | -0.0001 | 0.0016 | 0.0848 | 0.0000 | 0.6538 |
| CDNOW | E5 | spline_log_lrfm - spline_log_rfm | auc | 0.0014 | 0.0017 | 0.0004 | 0.0017 | -0.0002 | 0.0025 | 0.0817 | 0.0000 | 0.6538 |
| CDNOW | E5 | spline_log_lrfm - spline_log_rfm | spend_r2 | 0.0000 | 0.0001 | -0.0000 | 0.0001 | -0.0003 | 0.0004 | 0.7944 | 0.0000 | 1.0000 |
| CDNOW | E5 | spline_log_lrfm - spline_log_rfm | invoice_r2 | 0.0001 | 0.0002 | -0.0000 | 0.0003 | -0.0003 | 0.0006 | 0.6292 | 0.0000 | 1.0000 |
| Dunnhumby | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | auc | 0.0155 | 0.0176 | 0.0104 | 0.0234 | 0.0049 | 0.0277 | 0.0025 | 0.0000 | 0.0248 |
| Dunnhumby | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | spend_r2 | 0.0237 | 0.0233 | 0.0232 | 0.0246 | 0.0187 | 0.0289 | 0.0000 | 0.0000 | 0.0003 |
| Dunnhumby | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | invoice_r2 | 0.0202 | 0.0197 | 0.0197 | 0.0208 | 0.0164 | 0.0242 | 0.0000 | 0.0000 | 0.0003 |
| Dunnhumby | E2 | fuzzy_lrfm_fca - spline_log_lrfm | auc | -0.0137 | -0.0111 | -0.0184 | -0.0082 | -0.0224 | -0.0058 | 0.0000 | 0.0000 | 0.0003 |
| Dunnhumby | E2 | fuzzy_lrfm_fca - spline_log_lrfm | spend_r2 | -0.0103 | -0.0114 | -0.0114 | -0.0089 | -0.0150 | -0.0055 | 0.0000 | 0.0000 | 0.0003 |
| Dunnhumby | E2 | fuzzy_lrfm_fca - spline_log_lrfm | invoice_r2 | -0.0074 | -0.0083 | -0.0083 | -0.0065 | -0.0122 | -0.0024 | 0.0044 | 0.0000 | 0.0396 |
| Dunnhumby | E3 | hybrid_lrfm - spline_log_lrfm | auc | -0.0043 | -0.0037 | -0.0062 | -0.0024 | -0.0091 | -0.0001 | 0.0438 | 0.0000 | 0.2626 |
| Dunnhumby | E3 | hybrid_lrfm - spline_log_lrfm | spend_r2 | -0.0045 | -0.0044 | -0.0058 | -0.0035 | -0.0084 | -0.0005 | 0.0274 | 0.0000 | 0.1915 |
| Dunnhumby | E3 | hybrid_lrfm - spline_log_lrfm | invoice_r2 | 0.0015 | 0.0016 | 0.0009 | 0.0021 | -0.0017 | 0.0049 | 0.3612 | 0.0000 | 1.0000 |
| Dunnhumby | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | auc | -0.0075 | -0.0054 | -0.0142 | -0.0023 | -0.0171 | 0.0004 | 0.0683 | 0.0000 | 0.2731 |
| Dunnhumby | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | spend_r2 | -0.0044 | -0.0054 | -0.0054 | -0.0037 | -0.0071 | -0.0017 | 0.0014 | 0.0000 | 0.0158 |
| Dunnhumby | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | invoice_r2 | -0.0026 | -0.0033 | -0.0033 | -0.0017 | -0.0052 | -0.0000 | 0.0480 | 0.0000 | 0.2626 |
| Dunnhumby | E5 | spline_log_lrfm - spline_log_rfm | auc | 0.0001 | 0.0007 | -0.0005 | 0.0007 | -0.0015 | 0.0023 | 0.9810 | 0.0000 | 1.0000 |
| Dunnhumby | E5 | spline_log_lrfm - spline_log_rfm | spend_r2 | -0.0001 | 0.0001 | -0.0010 | 0.0002 | -0.0022 | 0.0017 | 0.9174 | 0.0000 | 1.0000 |
| Dunnhumby | E5 | spline_log_lrfm - spline_log_rfm | invoice_r2 | 0.0020 | 0.0019 | 0.0017 | 0.0022 | 0.0004 | 0.0036 | 0.0140 | 0.0000 | 0.1123 |
| Online Retail II | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | auc | 0.0089 | 0.0089 | 0.0077 | 0.0101 | 0.0061 | 0.0118 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | spend_r2 | 0.0171 | 0.0166 | 0.0162 | 0.0187 | 0.0135 | 0.0209 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E1 | fuzzy_lrfm_fca - crisp_lrfm_fca | invoice_r2 | 0.0321 | 0.0312 | 0.0312 | 0.0335 | 0.0267 | 0.0375 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E2 | fuzzy_lrfm_fca - spline_log_lrfm | auc | 0.0041 | 0.0029 | 0.0029 | 0.0047 | 0.0017 | 0.0063 | 0.0007 | 0.0000 | 0.0036 |
| Online Retail II | E2 | fuzzy_lrfm_fca - spline_log_lrfm | spend_r2 | 0.0068 | 0.0058 | 0.0058 | 0.0078 | 0.0018 | 0.0122 | 0.0075 | 0.0000 | 0.0150 |
| Online Retail II | E2 | fuzzy_lrfm_fca - spline_log_lrfm | invoice_r2 | -0.0254 | -0.0262 | -0.0267 | -0.0235 | -0.0404 | -0.0094 | 0.0027 | 0.0000 | 0.0082 |
| Online Retail II | E3 | hybrid_lrfm - spline_log_lrfm | auc | 0.0040 | 0.0029 | 0.0029 | 0.0048 | 0.0018 | 0.0061 | 0.0004 | 0.0000 | 0.0022 |
| Online Retail II | E3 | hybrid_lrfm - spline_log_lrfm | spend_r2 | 0.0102 | 0.0091 | 0.0091 | 0.0112 | 0.0065 | 0.0141 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E3 | hybrid_lrfm - spline_log_lrfm | invoice_r2 | 0.0080 | 0.0079 | 0.0071 | 0.0088 | 0.0027 | 0.0141 | 0.0017 | 0.0000 | 0.0069 |
| Online Retail II | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | auc | 0.0056 | 0.0071 | 0.0034 | 0.0071 | 0.0025 | 0.0083 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | spend_r2 | 0.0080 | 0.0078 | 0.0078 | 0.0083 | 0.0056 | 0.0104 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E4 | fuzzy_lrfm_fca - fuzzy_rfm_fca | invoice_r2 | 0.0111 | 0.0111 | 0.0108 | 0.0115 | 0.0081 | 0.0141 | 0.0000 | 0.0000 | 0.0003 |
| Online Retail II | E5 | spline_log_lrfm - spline_log_rfm | auc | 0.0023 | 0.0035 | 0.0012 | 0.0035 | 0.0003 | 0.0044 | 0.0220 | 0.0000 | 0.0220 |
| Online Retail II | E5 | spline_log_lrfm - spline_log_rfm | spend_r2 | 0.0033 | 0.0037 | 0.0031 | 0.0037 | 0.0016 | 0.0050 | 0.0002 | 0.0000 | 0.0014 |
| Online Retail II | E5 | spline_log_lrfm - spline_log_rfm | invoice_r2 | 0.0053 | 0.0055 | 0.0049 | 0.0056 | 0.0030 | 0.0075 | 0.0000 | 0.0000 | 0.0003 |

## Limitations

- L is censored by the beginning of each observation window for Online Retail II and Dunnhumby; this was not corrected.
- Origins overlap in customers and history, and the bootstrap reflects test-sample variability rather than refitting variability.
- The same datasets were already analysed, so these results cannot confirm the prior evidence. A nonsignificant result must not be interpreted as equivalence.
- No parameters were tuned; concept-mining failures stop the run rather than changing the precommitted settings.

# Base Paper vs Fuzzy RFM-FCA

Rungruang et al. (2024) report no predictive metrics, so their method (crisp quintile RFM-FCA, support >= 0.04) was re-implemented and evaluated under the same protocol as ours: rolling origins with a 91-day holdout, stratified 5-fold CV, train-only fitting, paired customer bootstrap, and Holm correction. Source: `results/baseline_ladder_rolling_origin/`.

## Headline (mean over rolling origins)

| dataset | metric | base paper method | fuzzy RFM-FCA (ours) | gain (points) | relative gain % | 95% CI (points) | p (Holm) | origins won | strong non-FCA baseline |
|---|---|---|---|---|---|---|---|---|---|
| Online Retail II (base-paper dataset) | ROC AUC (repurchase) | 0.7743 | 0.7799 | 0.5592 | 0.7222 | [0.4, 0.8] | 0.0135 | 6/6 | 0.7805 |
| Online Retail II (base-paper dataset) | Spend R² | 0.2975 | 0.3114 | 1.3872 | 4.6626 | [1.1, 1.6] | 0.0135 | 6/6 | 0.3097 |
| Online Retail II (base-paper dataset) | Invoice R² | 0.3464 | 0.3737 | 2.7280 | 7.8757 | [2.3, 3.2] | 0.0135 | 6/6 | 0.4054 |
| Dunnhumby (reproduction) | ROC AUC (repurchase) | 0.8689 | 0.8899 | 2.1043 | 2.4217 | [1.5, 2.7] | 0.0135 | 4/4 | 0.8950 |
| Dunnhumby (reproduction) | Spend R² | 0.5112 | 0.5382 | 2.6951 | 5.2718 | [2.2, 3.2] | 0.0135 | 4/4 | 0.5442 |
| Dunnhumby (reproduction) | Invoice R² | 0.6013 | 0.6227 | 2.1343 | 3.5494 | [1.7, 2.6] | 0.0135 | 4/4 | 0.6257 |

'Origins won' counts every origin, including the Online Retail II primary protocol. 'Strong non-FCA baseline' is a spline on log-RFM, shown for transparency: fuzzy RFM-FCA does not beat it, and is significantly worse on Online Retail II Invoice R² (see `docs/RESULTS_SUMMARY.md` §B).

## Online Retail II, original protocol (2010-12-09 cutoff, 365-day holdout)

| origin | auc_base | auc_ours | auc_delta | spend_r2_base | spend_r2_ours | spend_r2_delta | invoice_r2_base | invoice_r2_ours | invoice_r2_delta |
|---|---|---|---|---|---|---|---|---|---|
| 2010-12-09 (primary, 365d) | 0.7615 | 0.7848 | 0.0233 | 0.3404 | 0.3519 | 0.0115 | 0.4418 | 0.4621 | 0.0203 |

## Every origin

| dataset | origin | auc_base | auc_ours | auc_delta | spend_r2_base | spend_r2_ours | spend_r2_delta | invoice_r2_base | invoice_r2_ours | invoice_r2_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| Dunnhumby | day 347 | 0.8556 | 0.8965 | 0.0409 | 0.5163 | 0.5499 | 0.0337 | 0.6002 | 0.6244 | 0.0242 |
| Dunnhumby | day 438 | 0.8700 | 0.8892 | 0.0192 | 0.5188 | 0.5481 | 0.0293 | 0.6059 | 0.6299 | 0.0240 |
| Dunnhumby | day 529 | 0.8932 | 0.9005 | 0.0074 | 0.5163 | 0.5378 | 0.0215 | 0.5999 | 0.6166 | 0.0167 |
| Dunnhumby | day 620 | 0.8568 | 0.8736 | 0.0168 | 0.4936 | 0.5169 | 0.0233 | 0.5992 | 0.6197 | 0.0204 |
| Online Retail II | 2010-09-10 | 0.7047 | 0.7064 | 0.0017 | 0.2251 | 0.2324 | 0.0073 | 0.2966 | 0.3157 | 0.0192 |
| Online Retail II | 2010-12-10 | 0.7866 | 0.7958 | 0.0093 | 0.3018 | 0.3197 | 0.0179 | 0.3392 | 0.3697 | 0.0305 |
| Online Retail II | 2011-03-11 | 0.7791 | 0.7857 | 0.0066 | 0.2959 | 0.3164 | 0.0205 | 0.3443 | 0.3806 | 0.0363 |
| Online Retail II | 2011-06-10 | 0.8119 | 0.8181 | 0.0063 | 0.3497 | 0.3619 | 0.0122 | 0.3874 | 0.4119 | 0.0246 |
| Online Retail II | 2011-09-09 | 0.7891 | 0.7933 | 0.0042 | 0.3150 | 0.3266 | 0.0116 | 0.3645 | 0.3904 | 0.0258 |
| Online Retail II | 2010-12-09 (primary, 365d) | 0.7615 | 0.7848 | 0.0233 | 0.3404 | 0.3519 | 0.0115 | 0.4418 | 0.4621 | 0.0203 |

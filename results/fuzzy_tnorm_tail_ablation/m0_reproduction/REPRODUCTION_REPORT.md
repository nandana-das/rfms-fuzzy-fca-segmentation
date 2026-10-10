# M0 (fuzzy_rfm_fca) & Crisp RFM-FCA Baseline Reproduction Report

- **Status:** ALL CHECKS PASSED
- **Timestamp:** 2026-10-10 17:32:22.243094
- **Evaluated Cohorts:** Dunnhumby day 347, Online Retail II 2010-09-10
- **Evaluated Arms:** fuzzy_rfm_fca, crisp_rfm_fca
- **Max Absolute Difference:** 8.33e-17 (Tolerance: 1e-12)

## Detailed Metric Verification Table

| dataset          | origin     | arm           | metric     | computed_value      | frozen_value       | abs_difference        | tolerance | passed |
| ---------------- | ---------- | ------------- | ---------- | ------------------- | ------------------ | --------------------- | --------- | ------ |
| Dunnhumby        | day 347    | fuzzy_rfm_fca | auc        | 0.8964623849940966  | 0.8964623849940966 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | fuzzy_rfm_fca | spend_r2   | 0.5499331432839238  | 0.5499331432839238 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | fuzzy_rfm_fca | invoice_r2 | 0.6244314853034746  | 0.6244314853034746 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | crisp_rfm_fca | auc        | 0.8555971396110406  | 0.8555971396110406 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | crisp_rfm_fca | spend_r2   | 0.5162790656716345  | 0.5162790656716345 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | crisp_rfm_fca | invoice_r2 | 0.6002485752787994  | 0.6002485752787994 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | fuzzy_rfm_fca | auc        | 0.7063957958760724  | 0.7063957958760724 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | fuzzy_rfm_fca | spend_r2   | 0.23239893990983218 | 0.2323989399098321 | 8.326672684688674e-17 | 1e-12     | True   |
| Online Retail II | 2010-09-10 | fuzzy_rfm_fca | invoice_r2 | 0.31574263699112215 | 0.3157426369911221 | 5.551115123125783e-17 | 1e-12     | True   |
| Online Retail II | 2010-09-10 | crisp_rfm_fca | auc        | 0.7046948589594894  | 0.7046948589594894 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | crisp_rfm_fca | spend_r2   | 0.2251396234467392  | 0.2251396234467392 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | crisp_rfm_fca | invoice_r2 | 0.2965855583193776  | 0.2965855583193776 | 0.0                   | 1e-12     | True   |

Total runtime: 49.8s
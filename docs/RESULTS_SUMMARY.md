# Results Summary — Final RFM-FCA Study

## Scope
Final study: RFM only, using Dunnhumby The Complete Journey as the primary validation domain and Online Retail II as the independent cross-domain domain. Olist, RFMS/Satisfaction, and F* are excluded.

## Dunnhumby audit
- Observation cohort: 2,499 households.
- Observation window: days 1–620; holdout: days 621–711.
- F = distinct BASKET_ID count.
- M = sum(SALES_VALUE); no quantity multiplication.
- Raw fuzzy concepts: 502.
- Retained after J_max = 0.80 suppression: 123.
- Fixed-split predictive features: 114.
- max_len=None.

## Fixed-split predictive results
| Representation | Features | Repurchase AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|---:|
| Raw RFM | 3 | 0.8694 | 0.3604 | 0.4745 |
| Crisp RFM-FCA | 40 | 0.8611 | 0.4629 | 0.5536 |
| Fuzzy RFM-FCA (Suppressed) | 114 | 0.8629 | 0.5152 | 0.6188 |
| FCM Soft (matched k) | 40 | 0.8284 | 0.4445 | 0.5537 |

## Ten-split validation
| Representation | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| Raw RFM | 0.8624 ± 0.0275 | 0.3664 ± 0.0459 | 0.4515 ± 0.0334 |
| Crisp RFM-FCA | 0.8541 ± 0.0242 | 0.4800 ± 0.0351 | 0.5548 ± 0.0344 |
| Fuzzy RFM-FCA (Suppressed) | 0.8605 ± 0.0292 | 0.5028 ± 0.0376 | 0.6045 ± 0.0330 |
| FCM Soft (matched k) | 0.8379 ± 0.0312 | 0.4482 ± 0.0338 | 0.5520 ± 0.0254 |

## Fixed-split paired bootstrap
Fuzzy FCA versus raw RFM: AUC Δ = −0.0065, 95% CI [−0.0349, 0.0196], p = 0.330; Spend R² Δ = +0.1548, 95% CI [0.1019, 0.2136], p < 0.001; Invoice R² Δ = +0.1443, 95% CI [0.1028, 0.1866], p < 0.001.

## Geometric benchmark
All methods are evaluated in standardized raw RFM space.

| Method | k | Silhouette | Davies-Bouldin |
|---|---:|---:|---:|
| K-Means | 4 | 0.4663 | 0.7944 |
| Ward | 4 | 0.3462 | 0.8421 |
| FCM | 4 | 0.4204 | 0.7998 |
| Fuzzy RFM-FCA, top-k hardening | 4 | −0.0851 | 2.2278 |
| K-Means | 5 | 0.4747 | 0.7300 |
| Ward | 5 | 0.3571 | 0.8441 |
| FCM | 5 | 0.3227 | 0.9233 |
| Fuzzy RFM-FCA, top-k hardening | 5 | −0.2099 | 2.5513 |
| K-Means | 6 | 0.4603 | 0.7768 |
| Ward | 6 | 0.3816 | 0.7642 |
| FCM | 6 | 0.3880 | 0.8268 |
| Fuzzy RFM-FCA, top-k hardening | 6 | −0.2357 | 3.3258 |

These geometric values are diagnostics of hardened partitions, not a universal ranking of FCA against clustering methods.

## Audit terminology
Kneedle is Kneedle-inspired; stability is a proxy; FPC/Xie-Beni are restricted to canonical FCM; the k-matched FCA benchmark is top-k membership hardening, not alpha-cut; the separate alpha-cut analysis uses α = 0.5; predictive outputs were preserved during the final correction pass.
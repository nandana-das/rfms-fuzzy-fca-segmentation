# Fuzzy RFM-FCA Customer Segmentation

A leakage-controlled research implementation extending an established RFM-FCA customer-segmentation framework from a crisp binary formal context to a centroid-based fuzzy representation.

## Final research scope

The final study is **RFM only** and uses two transaction domains: Dunnhumby — The Complete Journey, as the primary validation domain, and Online Retail II, as the independent cross-domain domain.

**Excluded from the final study:** Olist, RFMS/Satisfaction, the F* purchase-intensity heuristic, and marketplace-frequency-sparsity experiments.

## Methodology

RFM is constructed from transaction history. For Dunnhumby, R is the observation-cutoff day minus last purchase day, F is the number of distinct BASKET_ID values per household, and M is the sum of SALES_VALUE. SALES_VALUE is not multiplied by QUANTITY.

Each RFM dimension is converted to five dense-rank bands. Centroid-based piecewise-linear membership with outer saturation produces graded memberships. L-fuzzy thresholds are 0.3, 0.5, and 0.7. Closed concept mining is uncapped.

Concept reduction uses a **Kneedle-inspired** normalized maximum-distance-from-chord rule; it is not claimed to be canonical Kneedle. Stability is an object-profile proxy, not canonical Kuznetsov stability.

Greedy extent-Jaccard suppression uses J_max = 0.80. The audited Dunnhumby run contains **502 raw fuzzy concepts, 123 retained concepts, and 114 fixed-split predictive features**.

### Clustering terminology

Geometric evaluation uses standardized raw RFM space for K-Means, Ward, canonical FCM, and the FCA benchmark.

**Top-k membership hardening** is used for the k-matched FCA geometric benchmark: the top-k non-trivial concepts are selected and each customer is assigned by maximum membership. This is **not an alpha-cut**.

The separate alpha-cut analysis thresholds top-level fuzzy memberships at α = 0.5.

Silhouette and Davies-Bouldin are geometric diagnostics. Canonical FPC and Xie-Beni are reserved for canonical FCM.

## Audited Dunnhumby results

| Representation | Features | Repurchase AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|---:|
| Raw RFM | 3 | 0.8694 | 0.3604 | 0.4745 |
| Crisp RFM-FCA | 40 | 0.8611 | 0.4629 | 0.5536 |
| Fuzzy RFM-FCA (Suppressed) | 114 | 0.8629 | 0.5152 | 0.6188 |
| FCM Soft (matched k) | 40 | 0.8284 | 0.4445 | 0.5537 |

Ten-split validation (seeds 1000–1009): raw RFM AUC 0.8624 ± 0.0275; crisp FCA 0.8541 ± 0.0242; fuzzy FCA 0.8605 ± 0.0292; FCM 0.8379 ± 0.0312.

These results describe representation utility; they are not a universal ranking of segmentation methods.

## Key files

- scripts/dunnhumby_rfm_fca_validation.py — primary Dunnhumby pipeline.
- scripts/fair_comparison_retail2.py — shared RFM/FCA implementation and Retail II pipeline.
- scripts/concept_redundancy.py — extent-Jaccard redundancy suppression.
- scripts/fcm.py — canonical FCM.
- results/dunnhumby_rfm_fca/ — audited Dunnhumby outputs.
- docs/METHODOLOGY_V4.md — locked final methodology.
- docs/PROJECT_DOCUMENT.md — current research framing.
- docs/RESULTS_SUMMARY.md — current results summary.

Historical Olist/RFMS/F* experiments are not part of the final paper and must not be cited as current methodology or results.

## Base paper

Rungruang et al., “RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA,” Expert Systems with Applications (2024).

The present work extends that framework; it does not claim to invent RFM-FCA.
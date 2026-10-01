# Methodology v4 — Final RFM-FCA Study

## 1. Study design
The final study evaluates a fuzzy extension of an established RFM-FCA customer-segmentation framework across Dunnhumby The Complete Journey and Online Retail II. The study uses RFM only. Satisfaction, RFMS, Olist, and F* are excluded.

## 2. RFM construction
Dunnhumby customer = household_key; order = BASKET_ID. R = 620 − max(DAY) within the observation window; F = distinct BASKET_ID count; M = sum(SALES_VALUE). The observation window is days 1–620 and the holdout is days 621–711. SALES_VALUE is not multiplied by QUANTITY.

Online Retail II uses CustomerID, InvoiceNo, reference-date recency, distinct invoice frequency, and summed line value.

## 3. Fuzzy formal context
RFM values are converted to five dense-rank score bands. Membership uses centroid-based piecewise-linear interpolation with outer saturation. L-fuzzy thresholds are 0.3, 0.5, and 0.7. Closed concept mining uses uncapped FP-growth with max_len=None.

## 4. Pruning and suppression
Pruning uses a Kneedle-inspired normalized maximum-distance-from-chord rule. Stability is an object-profile diversity proxy and is not canonical Kuznetsov stability. Greedy extent-Jaccard suppression uses J_max = 0.80.

Audited Dunnhumby counts: 502 raw fuzzy concepts → 123 after suppression → 114 fixed-split predictive features.

## 5. Clustering benchmark
All geometric methods are evaluated in standardized raw RFM space. K-Means and Ward use k = 4, 5, 6. Canonical FCM uses k = 4, 5, 6 and m = 2.0. The k-matched FCA benchmark uses top-k concept membership hardening by argmax. It is not an alpha-cut.

The separate alpha-cut analysis thresholds top-level memberships at α = 0.5. Silhouette and Davies-Bouldin are geometric diagnostics; canonical FPC and Xie-Beni are restricted to canonical FCM.

## 6. Predictive validation
The fixed split uses seed 42. Multi-split validation uses seeds 1000–1009. Tasks are future repurchase classification, future log-spend regression, and future invoice-count regression. Preprocessing, scoring cutoffs, fuzzy centroids, concept mining, suppression, FCM prototypes, and predictive models are fitted on training data only.

## 7. Cross-domain validation
The same RFM-FCA architecture is independently evaluated on Dunnhumby and Online Retail II. Cross-domain analysis covers concept persistence, overlap structure, geometric diagnostics, predictive utility, and sensitivity to frequency distributions.

## 8. Limitations
The final study does not validate satisfaction-aware RFMS segmentation or sparse marketplace-frequency behavior. Both are future-work directions. Geometric metrics should not be treated as a universal measure of FCA quality because FCA concepts encode partial-order and overlap structure.
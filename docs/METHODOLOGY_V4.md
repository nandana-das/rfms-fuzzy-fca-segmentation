# Methodology v4 — Final RFM-FCA Study

## 1. Study design
The final study evaluates a fuzzy extension of an established RFM-FCA customer-segmentation framework across Dunnhumby The Complete Journey and Online Retail II. The study uses RFM only. Satisfaction, RFMS, Olist, and F* are excluded.

## 2. RFM construction
Dunnhumby customer = household_key; order = BASKET_ID. R = 620 − max(DAY) within the observation window; F = distinct BASKET_ID count; M = sum(SALES_VALUE). The observation window is days 1–620 and the holdout is days 621–711. SALES_VALUE is not multiplied by QUANTITY.

Online Retail II uses CustomerID, InvoiceNo, reference-date recency, distinct invoice frequency, and summed line value.

## 3. Fuzzy formal context
RFM values are converted to five dense-rank score bands. Membership uses centroid-based piecewise-linear interpolation with outer saturation, reducing hard boundary discontinuities while retaining structural design choices (five behavioral bands, centroid construction, and $\mathcal{L}$-fuzzy threshold cuts). Baseline L-fuzzy thresholds are 0.3, 0.5, and 0.7; their sensitivity is evaluated against alternative configurations (0.2, 0.5, 0.8) and (0.4, 0.5, 0.6). Closed concept mining uses uncapped FP-growth with max_len=None.

## 4. Pruning and suppression
Closed fuzzy concepts are first generated subject to the minimum support criterion (0.04), noting that the base paper does not report a data-driven procedure for selecting this threshold. The Kneedle-inspired normalized maximum-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on the observed support and stability-proxy distributions (where stability is an object-profile diversity proxy, distinct from canonical Kuznetsov stability). Greedy extent-Jaccard suppression ($J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$) mitigates concept proliferation by suppressing near-duplicate concepts, with downstream predictive performance remaining comparable to the unsuppressed representation under the evaluated protocol.

Audited Dunnhumby counts: 502 raw fuzzy concepts → 123 after suppression → 114 fixed-split predictive features.

## 5. Clustering benchmark
All geometric methods are evaluated in standardized raw RFM space. K-Means and Ward use k = 4, 5, 6. Canonical FCM uses k = 4, 5, 6 and m = 2.0. The k-matched FCA benchmark uses top-k concept membership hardening by argmax. It is not an alpha-cut.

The separate alpha-cut analysis thresholds top-level memberships at α = 0.5. The negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties. Silhouette and Davies-Bouldin are geometric diagnostics; canonical FPC and Xie-Beni are restricted to canonical FCM.

## 6. Predictive validation
The fixed split uses seed 42. Multi-split validation uses seeds 1000–1009. Tasks are future repurchase classification, future log-spend regression, and future invoice-count regression. Preprocessing, scoring cutoffs, fuzzy centroids, concept mining, suppression, FCM prototypes, and predictive models are fitted on training data only.

Because 98.08% of households repurchased during the 91-day holdout, Dunnhumby repurchase classification exhibits a strong ceiling effect; the fuzzy-versus-crisp AUC difference is small and spans zero under fixed-split bootstrap comparison. Predictive gains are primarily observed in future monetary expenditure and transaction activity. Formal statistical significance ($p < 0.001$) is supported by paired bootstrap testing ($B=1,000$) on the fixed split, while the 10-split CV reports mean ± SD and split win-rates demonstrating consistency across splits.

## 7. Cross-domain validation
The same RFM-FCA architecture is independently evaluated on Dunnhumby and Online Retail II. The results provide evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.

Our Online Retail II reconstruction recovered all 31 published frequent concept intents reported by Rungruang et al. However, exact customer counts matched for only 3 of the 31 concepts. The remaining discrepancies are consistent with the paper's unspecified tie-breaking procedure for customers sharing identical frequency values, particularly the 1,623 customers with F = 1.

## 8. Fuzzy membership threshold sensitivity analysis
A dedicated one-factor sensitivity analysis on Dunnhumby evaluates the effect of the $\mathcal{L}$-fuzzy threshold tuple across three configurations under the locked evaluation protocol (observation days 1–620, holdout 621–711, 10 temporal splits with seeds 1000–1009, support 0.04, $J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$, train-only fitting):
- **Conservative / permissive:** $(0.2, 0.5, 0.8)$
- **Baseline:** $(0.3, 0.5, 0.7)$
- **Tight:** $(0.4, 0.5, 0.6)$

### Concept-space sensitivity (Full cohort, $n=2,499$):
- $(0.2, 0.5, 0.8)$: 652 raw concepts $\to$ 265 suppressed ($2.46\times$ compression, $59.4\%$ reduction).
- $(0.3, 0.5, 0.7)$: 502 raw concepts $\to$ 123 suppressed ($4.08\times$ compression, $75.5\%$ reduction).
- $(0.4, 0.5, 0.6)$: 420 raw concepts $\to$ 38 suppressed ($11.05\times$ compression, $91.0\%$ reduction).

### Multi-split predictive performance (10-split mean ± std):
- **Crisp RFM-FCA (Fixed):** AUC $0.8541 \pm 0.0242$, Spend $R^2 = 0.4800 \pm 0.0351$, Invoice $R^2 = 0.5548 \pm 0.0344$.
- **$(0.2, 0.5, 0.8)$:** AUC $0.8611 \pm 0.0299$ ($\Delta = +0.0070$, 6/10 wins); Spend $R^2 = 0.5037 \pm 0.0387$ ($\Delta = +0.0237$, 9/10 wins); Invoice $R^2 = 0.6071 \pm 0.0313$ ($\Delta = +0.0523$, 10/10 wins).
- **$(0.3, 0.5, 0.7)$ [Baseline]:** AUC $0.8605 \pm 0.0292$ ($\Delta = +0.0064$, 7/10 wins); Spend $R^2 = 0.5028 \pm 0.0376$ ($\Delta = +0.0228$, 9/10 wins); Invoice $R^2 = 0.6045 \pm 0.0330$ ($\Delta = +0.0497$, 10/10 wins).
- **$(0.4, 0.5, 0.6)$:** AUC $0.8673 \pm 0.0243$ ($\Delta = +0.0132$, 8/10 wins); Spend $R^2 = 0.5090 \pm 0.0339$ ($\Delta = +0.0290$, 10/10 wins); Invoice $R^2 = 0.6079 \pm 0.0306$ ($\Delta = +0.0531$, 10/10 wins).

### Interpretation & robustness:
The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space, downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations. Redundancy suppression substantially reduces concept-space size in all configurations, confirming that the fuzzy extension is not dependent on one narrowly tuned threshold configuration. Thresholds were not optimized post-hoc.

## 9. Limitations
1. Five-band fuzzy membership is a structural design choice.
2. $\mathcal{L}$-fuzzy thresholds are design parameters rather than mathematically unique values.
3. The pruning heuristic is a simplified normalized max-distance-from-chord procedure, not canonical Kneedle.
4. Concept stability is evaluated via an object-profile diversity proxy, not canonical Kuznetsov stability.
5. Base-paper replication customer counts cannot be uniquely reconstructed because of unspecified frequency tie-breaking on $F=1$.
6. Dunnhumby repurchase classification exhibits a strong ceiling effect due to the 98.08% repeat rate, limiting classification discrimination.
7. Geometric metrics (Silhouette, Davies-Bouldin) do not reflect Galois intent closure and serve as diagnostics rather than quality rankings.
8. Cross-domain validation covers two retail transaction datasets; broader validation across diverse commercial regimes is required before general claims can be made.
9. Satisfaction-aware RFMS segmentation and sparse marketplace frequency behavior remain out of scope for the final paper.
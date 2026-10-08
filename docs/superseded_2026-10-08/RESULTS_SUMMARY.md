> **SUPERSEDED (2026-10-08).** Historical copy of `docs/RESULTS_SUMMARY.md` as last committed (git HEAD 8a420e2), before the audit. It reports dense-rank scoring, an untransformed Raw RFM baseline, an incorrect 98.08% repurchase rate and "temporal CV" labels. Do not cite. Current documents: `README.md`, `docs/RESULTS_SUMMARY.md`; errata: `docs/AUDIT_ERRATA.md`.

# Fuzzy Formal Concept Analysis (FCA) Customer Segmentation — Consolidated Results Summary

**Document Status:** Definitive Results Summary (Locked Methodology v4)  
**Primary Domain:** Dunnhumby "The Complete Journey" ($N = 2,499$ households)  
**Second Domain:** Online Retail II ($N = 5,878$ clean customers)  
**Methodology:** Strictly RFM Only ($R, F, M$). Olist / RFMS / $F^*$ are preserved as historical artifacts (Section 4) and are out-of-scope for the final paper.

---

## Executive Overview & Research Lineage

This document consolidates all audited quantitative findings, structural concept counts, clustering benchmarks, and out-of-sample predictive evaluations for the Fuzzy RFM-FCA customer segmentation framework.

The framework builds upon the crisp binary RFM-FCA model of Rungruang et al. (2024), directly answering their call for representing RFM values in a **non-binary / fuzzy formal context**. It introduces:
1. Centroid-based piecewise-linear fuzzy memberships with outer shoulder saturation, reducing hard boundary discontinuities while retaining structural design choices (five behavioral bands, centroid construction, and $\mathcal{L}$-fuzzy threshold cuts).
2. Kneedle-inspired normalized max-distance-from-chord heuristic providing a data-driven secondary pruning criterion on support and stability-proxy distributions, complementing the 0.04 minimum support threshold used during concept generation.
3. Greedy extent-Jaccard redundancy suppression ($J_{\max} = 0.80$) mitigating concept proliferation.
4. Principled clustering evaluation via canonical Fuzzy C-Means (FCM, $m=2.0$) and Top-$k$ Membership Hardening evaluated in standardized raw RFM space.
5. Strictly leakage-free temporal predictive validation on repeat-rich consumer purchasing.
6. Sensitivity analysis of the fuzzy threshold configuration confirming that the predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations.

---

## 1. Primary Validation: Dunnhumby "The Complete Journey"

### 1.1 Data Audit Summary
- **Total Transactions:** 2,595,732 product-line records.
- **Total Households:** 2,500 unique households.
- **Total Baskets:** 276,484 distinct baskets across 711 continuous days.
- **Repeat Rate:** **99.68%** (2,497 households with $\ge 2$ baskets; 2,492 with $\ge 3$ baskets).
- **Mean Baskets per Household (Full):** $110.59$ (median $79.0$, max $1,300$).
- **Observation Cutoff:** Days 1–620 ($N=2,499$ households, 238,873 baskets).
- **Future Holdout:** Days 621–711 (91 days; repurchase rate = $98.08\%$, 2,451 repurchasers).

*Classification Ceiling Effect:* Because 98.08% of households repurchased during the 91-day holdout, repurchase classification exhibits a strong class-imbalance/ceiling effect. The fuzzy-versus-crisp AUC difference is therefore small and spans zero under the fixed-split bootstrap comparison. The clearer predictive gains occur in future monetary expenditure and transaction activity.

### 1.2 RFM Distributional Statistics

| Cohort | Metric | Count | Mean | Std | Min | P25 | P50 (Median) | P75 | P90 | P95 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Full Population (711 Days)** | R (days) | 2,500 | 25.57 | 62.78 | 0.0 | 1.0 | 6.0 | 20.0 | 62.0 | 117.05 | 657.0 |
| **Full Population (711 Days)** | F (baskets) | 2,500 | 110.59 | 115.65 | 1.0 | 39.0 | 79.0 | 142.25 | 229.10 | 321.00 | 1,300.0 |
| **Full Population (711 Days)** | M (\$) | 2,500 | 3,222.99 | 3,348.36 | 8.17 | 970.74 | 2,157.75 | 4,413.32 | 7,448.22 | 9,754.14 | 38,319.79 |
| **Observation (620 Days)** | R (days) | 2,499 | 26.73 | 67.39 | 0.0 | 1.0 | 6.0 | 19.0 | 63.0 | 129.20 | 590.0 |
| **Observation (620 Days)** | F (baskets) | 2,499 | 95.59 | 102.09 | 1.0 | 33.0 | 67.0 | 123.0 | 202.0 | 277.10 | 1,169.0 |
| **Observation (620 Days)** | M (\$) | 2,499 | 2,742.18 | 2,874.59 | 4.49 | 790.90 | 1,823.70 | 3,739.70 | 6,385.79 | 8,339.92 | 31,548.37 |
| **Future Holdout (91 Days)** | Future Spend (\$) | 2,499 | 482.05 | 558.46 | 0.0 | 99.42 | 296.79 | 669.37 | 1,175.15 | 1,554.78 | 6,771.42 |
| **Future Holdout (91 Days)** | Future Baskets | 2,499 | 15.05 | 17.08 | 0.0 | 4.0 | 10.0 | 20.0 | 33.0 | 47.10 | 169.0 |

### 1.3 Concept Lattice & Redundancy Suppression
- **Raw Crisp Concepts:** 38 concepts (min support = 0.04).
- **Raw Fuzzy Concepts:** 502 concepts (`max_len = None`, min support = 0.04).
- **Kneedle-Inspired Secondary Pruning:** Closed fuzzy concepts are first generated subject to the 0.04 minimum support criterion. The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on observed distributions, identifying support cutoff $= 0.2129$ and stability-proxy cutoff $= 0.4285$ (evaluated via an object-profile diversity proxy, distinct from canonical Kuznetsov stability).
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Retains **123 concepts** (379 concepts removed; 4.1× compression).
- **Concepts per Customer ($\mu \ge 0.5$):** Pre-suppression Mean $= 57.0$, Median $= 61.6$ (falls to ~4.3 post-suppression).

### 1.4 Out-of-Sample Predictive Metrics (Fixed Split, Seed = 42)
*Train $N=1,749$, Test $N=750$. All preprocessing, fuzzy centroids, concept lattices, and scalers fit on train only.*

| Representation Arm | Features ($D$) | Repurchase AUC | Brier Score | Spend $R^2$ (log1p) | Spend MAE | Spend $\rho$ | Invoice $R^2$ (log1p) | Invoice MAE | Invoice $\rho$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | 0.8694 | 0.0615 | 0.3604 | 1.0707 | 0.7680 | 0.4745 | 0.6212 | 0.7660 |
| **Crisp RFM-FCA** | 40 | 0.8611 | 0.0548 | 0.4629 | 0.9703 | 0.7477 | 0.5536 | 0.5612 | 0.7623 |
| **Fuzzy RFM-FCA (Supp.)** | **114** | **0.8629** | **0.0529** | **0.5152** | **0.8973** | **0.7924** | **0.6188** | **0.5133** | **0.8065** |
| **FCM Soft ($k=40$)** | 40 | 0.8284 | 0.0557 | 0.4445 | 0.9618 | 0.7505 | 0.5537 | 0.5564 | 0.7678 |

#### Paired Bootstrap 95% Confidence Intervals ($B=1,000$ Resamples)
- **Fuzzy RFM-FCA vs Raw RFM Baseline:**
  - Repurchase AUC: $\Delta = -0.0065$ [$-0.0349, +0.0196$], $p = 0.330$ (spans zero; both methods achieve ceiling-level AUC $\approx 0.86$).
  - Spend $R^2$: $\mathbf{\Delta = +0.1548}$ [$+0.1019, +0.2136$], $p < 0.001$ (excludes zero; +15.5 percentage points).
  - Invoice $R^2$: $\mathbf{\Delta = +0.1443}$ [$+0.1028, +0.1866$], $p < 0.001$ (excludes zero; +14.4 percentage points).
- **Fuzzy RFM-FCA vs Crisp RFM-FCA:**
  - Spend $R^2$: $\mathbf{\Delta = +0.0523}$ [$+0.0302, +0.0769$], $p < 0.001$ (excludes zero).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$ [$+0.0391, +0.0905$], $p < 0.001$ (excludes zero).
- **Fuzzy RFM-FCA vs FCM Soft ($k=40$):**
  - Repurchase AUC: $\mathbf{\Delta = +0.0345}$ [$+0.0053, +0.0663$], $p = 0.011$ (excludes zero).
  - Spend $R^2$: $\mathbf{\Delta = +0.0707}$ [$+0.0328, +0.1075$], $p < 0.001$ (excludes zero).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$ [$+0.0358, +0.0948$], $p < 0.001$ (excludes zero).

### 1.5 10-Split Temporal Cross-Validation Performance (Seeds 1000–1009)

| Representation Arm | Repurchase AUC (Mean ± Std) | Spend $R^2$ (Mean ± Std) | Invoices $R^2$ (Mean ± Std) |
|---|:---:|:---:|:---:|
| **Raw RFM Baseline** | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | **$0.8605 \pm 0.0292$** | **$0.5028 \pm 0.0376$** | **$0.6045 \pm 0.0330$** |
| **FCM Soft (matched $k$)** | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Summary:* The 10-split temporal cross-validation confirms that Fuzzy RFM-FCA achieves consistent gains across splits in predicting future customer monetary volume ($+0.1364$ vs raw RFM, $+0.0546$ vs FCM) and shopping visit frequency ($+0.1530$ vs raw RFM, $+0.0525$ vs FCM). Formal statistical significance ($p < 0.001$) is supported by paired bootstrap testing on the fixed split, while 10-split CV demonstrates consistency across splits (9/10 and 10/10 split wins).

### 1.6 Geometric Hard Clustering Benchmark (Standardized Raw RFM Space)

| Method | Hardening Procedure | $k$ | Evaluation Space | Silhouette | Davies–Bouldin |
|---|---|:---:|---|:---:|:---:|
| **K-Means** | Partition (k-means) | 4 | Raw RFM (Standardized) | 0.4663 | 0.7944 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 4 | Raw RFM (Standardized) | 0.3462 | 0.8421 |
| **FCM (Canonical)** | Argmax Soft Membership | 4 | Raw RFM (Standardized) | 0.4204 | 0.7998 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 4 | Raw RFM (Standardized) | -0.0851 | 2.2278 |
| **K-Means** | Partition (k-means) | 5 | Raw RFM (Standardized) | 0.4747 | 0.7300 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 5 | Raw RFM (Standardized) | 0.3571 | 0.8441 |
| **FCM (Canonical)** | Argmax Soft Membership | 5 | Raw RFM (Standardized) | 0.3227 | 0.9233 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 5 | Raw RFM (Standardized) | -0.2099 | 2.5513 |
| **K-Means** | Partition (k-means) | 6 | Raw RFM (Standardized) | 0.4603 | 0.7768 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 6 | Raw RFM (Standardized) | 0.3816 | 0.7642 |
| **FCM (Canonical)** | Argmax Soft Membership | 6 | Raw RFM (Standardized) | 0.3880 | 0.8268 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 6 | Raw RFM (Standardized) | -0.2357 | 3.3258 |
| **Crisp RFM-FCA** | Natural Hardening Rule | 6 | Raw RFM (Standardized) | 0.0008 | 1.3503 |
| **Fuzzy RFM-FCA** | Natural Hardening Rule | 12 | Raw RFM (Standardized) | -0.3999 | 3.5972 |

*Diagnostic Note:* The negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties. Distance-based metrics evaluate hyperspherical compactness and serve as geometric diagnostics rather than universal quality rankings.

### 1.7 Fuzzy Membership Threshold Sensitivity Analysis

To verify that the predictive advantages of Fuzzy RFM-FCA are not an artifact of the baseline threshold configuration ($\mathcal{L} = \{0.3, 0.5, 0.7\}$), a dedicated one-factor sensitivity analysis was executed across three threshold configurations on Dunnhumby under the locked evaluation protocol (observation days 1–620, holdout days 621–711, 10 temporal splits with seeds 1000–1009, 70/30 stratified train/test split, support cutoff 0.04, $J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$, train-only fitting):
- **Conservative / Permissive:** $(0.2, 0.5, 0.8)$
- **Baseline:** $(0.3, 0.5, 0.7)$
- **Tight:** $(0.4, 0.5, 0.6)$

#### Empirical Sensitivity Comparison:

| Configuration | Full-Cohort Raw Concepts | Full-Cohort Suppressed ($k$) | Compression Ratio | Repurchase AUC (Mean ± SD) | $\Delta$ AUC vs Crisp | Spend $R^2$ (Mean ± SD) | $\Delta$ Spend $R^2$ vs Crisp | Invoice $R^2$ (Mean ± SD) | $\Delta$ Invoice $R^2$ vs Crisp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Crisp Baseline** | 38 | 38 | 1.00× | $0.8541 \pm 0.0242$ | *0.0000* | $0.4800 \pm 0.0351$ | *0.0000* | $0.5548 \pm 0.0344$ | *0.0000* |
| **(0.2, 0.5, 0.8)** *(Permissive)* | 652 | 265 | 2.46× (59.4%) | $0.8611 \pm 0.0299$ | **+0.0070** (6/10) | $0.5037 \pm 0.0387$ | **+0.0237** (9/10) | $0.6071 \pm 0.0313$ | **+0.0523** (10/10) |
| **(0.3, 0.5, 0.7)** *(Baseline)* | 502 | 123 | 4.08× (75.5%) | $0.8605 \pm 0.0292$ | **+0.0064** (7/10) | $0.5028 \pm 0.0376$ | **+0.0228** (9/10) | $0.6045 \pm 0.0330$ | **+0.0497** (10/10) |
| **(0.4, 0.5, 0.6)** *(Tight)* | 420 | 38 | 11.05× (91.0%) | $0.8673 \pm 0.0243$ | **+0.0132** (8/10) | $0.5090 \pm 0.0339$ | **+0.0290** (10/10) | $0.6079 \pm 0.0306$ | **+0.0531** (10/10) |

*Paired Split Wins vs Crisp Baseline Across 10 Splits:*
- Repurchase AUC: 6/10 for (0.2, 0.5, 0.8); 7/10 for (0.3, 0.5, 0.7); 8/10 for (0.4, 0.5, 0.6).
- Future Spend $R^2$: 9/10 for (0.2, 0.5, 0.8); 9/10 for (0.3, 0.5, 0.7); 10/10 for (0.4, 0.5, 0.6).
- Future Invoice $R^2$: 10/10 for all three configurations.

#### Methodological Interpretation:
The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space, downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations.

- **Concept-space structure is sensitive to threshold choice:** Relaxing the outer threshold to 0.2 expands raw concepts to 652 and retains 265 concepts post-suppression, whereas tightening to 0.4 reduces raw concepts to 420 and compresses them to 38 retained concepts.
- **Predictive utility is comparatively stable:** Downstream spend $R^2$ remains within $0.5028$–$0.5090$ (all exceeding crisp by $+0.023$ to $+0.029$) and invoice $R^2$ remains within $0.6045$–$0.6079$ (all exceeding crisp by $+0.050$ to $+0.053$, with unanimous 10/10 split wins).
- **Redundancy suppression is consistently effective:** The greedy extent-Jaccard procedure mitigates concept proliferation by suppressing near-duplicate concepts under the specified $J_{\max} = 0.80$ criterion, substantially compressing the concept space across all settings ($59.4\%$ to $91.0\%$ reduction).
- **Non-tuning assurance:** These findings confirm that the fuzzy extension is not dependent on one narrowly tuned threshold configuration. Thresholds were not optimized post-hoc based on test set results.

Artifacts:
- Visual Diagnostic: [`results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png`](../results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png)
- Audit Report: [`results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md`](../results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md)
- Metric CSVs: [`fuzzy_membership_sensitivity_summary.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_summary.csv), [`fuzzy_membership_sensitivity_splits.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_splits.csv), [`fuzzy_membership_sensitivity_full_cohort.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_full_cohort.csv)

---

## 2. Independent Second Domain: Online Retail II

### 2.1 Base Paper Replication
- **Customer Cohort:** Replicated the 5,878 clean unique customer cohort from Rungruang et al. (2024).
- **Intent Recovery:** Our reconstruction recovered all 31 published frequent concept intents ($\text{support} > 0.04$) reported by Rungruang et al. However, exact customer counts matched for only 3 of the 31 concepts. The remaining discrepancies are consistent with the paper's unspecified tie-breaking procedure for customers sharing identical frequency values, particularly the 1,623 customers with $F = 1$.
- **Clustering Geometry:** Replicated reported K-Means and Ward behavior across $k=2..10$ on 5,633 outlier-filtered customers (Silhouette ~0.33–0.38, DB ~0.99–1.07).

### 2.2 Redundancy Suppression & Predictive Evaluation
- **Uncapped Closed Concepts:** 1,064 concepts discovered under `max_len = None`.
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Mitigates concept proliferation by compressing 445 candidate concepts into **95 concepts** (4.7× compression), reducing near-duplicate pairs from $3.3\%$ to $0.0\%$. After suppression, downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
- **10-Split Temporal Performance (Seeds 1000–1009):**
  - Raw RFM Baseline: AUC $0.7770 \pm 0.0117$, Spend $R^2 = 0.3476$, Invoice $R^2 = 0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** ($p = 2.3 \times 10^{-4}$ vs crisp via paired t-test), Spend $R^2 = \mathbf{0.3664}$, Invoice $R^2 = \mathbf{0.4952}$.

---

## 3. Cross-Domain Comparative Synthesis

| Metric / Dimension | Primary Domain: Dunnhumby | Second Domain: Online Retail II |
|---|---|---|
| **Setting** | Supermarket Grocery | Non-Store Giftware Retail |
| **Customer Population ($N$)** | 2,499 households | 5,878 customers |
| **Observation Window** | 620 days continuous | 730 days continuous |
| **Repeat Purchase Rate** | **99.68%** | **72.39%** |
| **Median Purchases ($F$)** | 79.0 baskets | 3.0 orders |
| **Raw Discovered Concepts** | 502 | 1,064 |
| **Suppressed Feature Dimensions** | 123 | 95 |
| **Compression Ratio** | **4.1×** (502 $\to$ 123) | **4.7×** (445 $\to$ 95) |
| **Predictive Lift: Spend $R^2$** | **+0.1364** vs raw ($0.5028$ vs $0.3664$) | **+0.0188** vs raw ($0.3664$ vs $0.3476$) |
| **Predictive Lift: Invoice $R^2$** | **+0.1530** vs raw ($0.6045$ vs $0.4515$) | **+0.0310** vs raw ($0.4952$ vs $0.4642$) |
| **Threshold Sensitivity** | Predictive relationship consistent across (0.2, 0.5, 0.8), (0.3, 0.5, 0.7), (0.4, 0.5, 0.6) | Locked baseline (0.3, 0.5, 0.7) |

*Synthesis Conclusion:* The results provide evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.

---

## 4. Historical Exploratory Studies (Out of Scope for Final Paper)

Earlier preliminary experiments investigated the **Olist Brazilian E-Commerce marketplace** ($N = 93,357$), testing an expanded **RFMS context** (with customer review Satisfaction $S$) and an entropy-optimized purchase-intensity index ($F^*$).

### Summary of Historical Findings:
1. **Severe Marketplace Frequency Sparsity:**
   - 97.00% of Olist customers ($90,556 / 93,357$) made only a single purchase (repeat rate = 3.00%).
   - Under controlled ablation (`scripts/ablation_frequency_variants.py`), neither literal $F$, composite $F^*$, nor engagement indices could overcome the 97% single-order barrier.
2. **Temporal Review Leakage Audit:**
   - An audit (`scripts/audit_olist_temporal_leakage.py`) showed that earlier inflated holdout results (AUC ~0.667) were artifacts of post-cutoff review aggregation.
   - Once strictly purged, Olist predictive performance realistically collapsed to baseline (Fuzzy AUC $0.5612 \pm 0.0125$ vs Raw RFMS $0.5587 \pm 0.0107$; raw won in 4/10 splits).
3. **Methodological Disposition:**
   - These findings demonstrated that representation engineering cannot overcome intrinsic physical frequency sparsity.
   - Consequently, the final study was locked to repeat-rich consumer transaction domains (Dunnhumby and Online Retail II) under Methodology v4. All historical Olist code and data remain archived in `scripts/` and `results/` for scientific provenance.

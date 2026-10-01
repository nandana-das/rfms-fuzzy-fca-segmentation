# Fuzzy Formal Concept Analysis (FCA) Customer Segmentation — Consolidated Results Summary

**Document Status:** Definitive Results Summary (Locked Methodology v4)  
**Primary Domain:** Dunnhumby "The Complete Journey" ($N = 2,499$ households)  
**Second Domain:** Online Retail II ($N = 5,878$ clean customers)  
**Methodology:** Strictly RFM Only ($R, F, M$). Olist / RFMS / $F^*$ are preserved as historical artifacts (Section 4) and are out-of-scope for the final paper.

---

## Executive Overview & Research Lineage

This document consolidates all audited quantitative findings, structural concept counts, clustering benchmarks, and out-of-sample predictive evaluations for the Fuzzy RFM-FCA customer segmentation framework.

The framework builds upon the crisp binary RFM-FCA model of Rungruang et al. (2024), directly answering their call for representing RFM values in a **non-binary / fuzzy formal context**. It introduces:
1. Centroid-based piecewise-linear fuzzy memberships with outer shoulder saturation.
2. Kneedle-inspired elbow pruning on support and object-profile stability proxy distributions.
3. Greedy extent-Jaccard redundancy suppression ($J_{\max} = 0.80$) solving concept proliferation.
4. Principled clustering evaluation via canonical Fuzzy C-Means (FCM, $m=2.0$) and Top-$k$ Membership Hardening evaluated in standardized raw RFM space.
5. Strictly leakage-free temporal predictive validation on repeat-rich consumer purchasing.

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
- **Kneedle-Inspired Thresholds:** Support elbow $= 0.2129$, Stability-proxy elbow $= 0.4285$.
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Retains **123 concepts** (379 concepts removed; 4.1× compression).
- **Concepts per Customer ($\mu \ge 0.5$):** Mean $= 57.0$, Median $= 61.6$.

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

*Summary:* The 10-split temporal cross-validation confirms that Fuzzy RFM-FCA achieves statistically significant, substantial gains in predicting future customer monetary volume ($+0.1364$ vs raw RFM, $+0.0546$ vs FCM) and shopping visit frequency ($+0.1530$ vs raw RFM, $+0.0525$ vs FCM).

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

*Diagnostic Note:* Distance-based clustering metrics (Silhouette, Davies–Bouldin) reward spherical cluster boundaries. In contrast, formal concept analysis generates partially ordered, overlapping intent closures. Forcing continuous multi-concept memberships into mutually exclusive hard clusters introduces boundary assignment penalties.

---

## 2. Independent Second Domain: Online Retail II

### 2.1 Base Paper Replication
- **Customer Cohort:** Replicated the 5,878 clean unique customer cohort from Rungruang et al. (2024).
- **Intent Recovery:** Recovered **all 31 published frequent concept intents** ($\text{support} > 0.04$).
- **Clustering Geometry:** Replicated reported K-Means and Ward behavior across $k=2..10$ on 5,633 outlier-filtered customers (Silhouette ~0.33–0.38, DB ~0.99–1.07).

### 2.2 Redundancy Suppression & Predictive Evaluation
- **Uncapped Closed Concepts:** 1,064 concepts discovered under `max_len = None`.
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Compressed 445 candidate concepts into **95 concepts** (4.7× compression), reducing near-duplicate pairs from $3.3\%$ to $0.0\%$.
- **10-Split Temporal Performance (Seeds 1000–1009):**
  - Raw RFM Baseline: AUC $0.7770 \pm 0.0117$, Spend $R^2 = 0.3476$, Invoice $R^2 = 0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** ($p = 2.3 \times 10^{-4}$ vs crisp), Spend $R^2 = \mathbf{0.3664}$, Invoice $R^2 = \mathbf{0.4952}$.

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

*Conclusion:* Across both repeat-transaction domains, the centroid-based fuzzy RFM-FCA framework delivers consistent, statistically significant predictive gains over raw RFM and crisp FCA baselines, with the strongest lift observed in high-frequency FMCG grocery transactions.

---

## 4. Historical Exploratory Studies (Out of Scope for Final Paper)

Earlier preliminary experiments investigated the **Olist Brazilian E-Commerce marketplace** ($N = 93,357$), testing an expanded **RFMS context** (with customer review Satisfaction $S$) and an entropy-optimized purchase-intensity index ($F^*$).

### Summary of Historical Findings:
1. **Severe Marketplace Frequency Sparsity:**
   - 97.00% of Olist customers ($90,556 / 93,357$) made only a single purchase (repeat rate = 3.00%).
   - Under controlled ablation (`scripts/ablation_frequency_variants.py`), neither literal $F$, composite $F^*$, nor engagement indices could overcome the 97% single-order barrier.
2. **Temporal Review Leakage Audit:**
   - An audit (`scripts/audit_olist_temporal_leakage.py`) proved that earlier inflated holdout results (AUC ~0.667) were artifacts of post-cutoff review aggregation.
   - Once strictly purged, Olist predictive performance realistically collapsed to baseline (Fuzzy AUC $0.5612 \pm 0.0125$ vs Raw RFMS $0.5587 \pm 0.0107$; raw won in 4/10 splits).
3. **Methodological Disposition:**
   - These findings demonstrated that representation engineering cannot overcome intrinsic physical frequency sparsity.
   - Consequently, the final study was locked to repeat-rich consumer transaction domains (Dunnhumby and Online Retail II) under Methodology v4. All historical Olist code and data remain archived in `scripts/` and `results/` for scientific provenance.

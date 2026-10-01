# Empirical Results (Methodology v4)

> **Document Status:** Authoritative Empirical Results Section  
> **Primary Domain:** Dunnhumby "The Complete Journey" ($N = 2,499$ households)  
> **Second Domain:** Online Retail II ($N = 5,878$ clean customers)  
> **Methodological Scope:** Strictly RFM Only ($R, F, M$).

---

## 1. Primary Domain: Dunnhumby "The Complete Journey"

### 1.1 Dataset & RFM Structure
The Dunnhumby "The Complete Journey" dataset comprises 2,595,732 transaction records across 2,500 unique households and 276,484 shopping baskets over 711 continuous days.
- **Repeat Rate:** **99.68%** of households made $\ge 2$ baskets (mean $F = 110.59$ baskets, median $79.0$, max $1,300$).
- **Observation Window:** Days 1–620 ($N = 2,499$ households).
- **Holdout Window:** Days 621–711 (91 days; repurchase rate = $98.08\%$, mean spend = \$482.05, mean baskets = $15.05$).

*Classification Ceiling Effect:* Because 98.08% of households repurchased during the 91-day holdout, repurchase classification exhibits a strong class-imbalance/ceiling effect. The fuzzy-versus-crisp AUC difference is therefore small and spans zero under the fixed-split bootstrap comparison. The clearer predictive gains occur in future monetary expenditure and transaction activity.

### 1.2 FCA Concept Lattice & Redundancy Suppression
- **Raw Concepts:** Uncapped FP-growth (`max_len = None`, min support = 0.04) identified **502 closed fuzzy concepts** (vs 38 crisp concepts).
- **Kneedle-Inspired Secondary Pruning:** Closed fuzzy concepts are first generated subject to the 0.04 minimum support criterion. The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on observed distributions, identifying support cutoff $= 0.2129$ and stability-proxy cutoff $= 0.4285$ (evaluated via an object-profile diversity proxy, distinct from canonical Kuznetsov stability).
- **Redundancy Suppression ($J_{\max} = 0.80$):** The greedy extent-Jaccard procedure mitigates concept proliferation by suppressing near-duplicate concepts under the specified $J_{\max} = 0.80$ criterion, compressing the representation to **123 concepts** (4.1× reduction; 379 redundant concepts removed).

### 1.3 Out-of-Sample Predictive Performance (10-Split Temporal Cross-Validation)

| Representation Arm | Features ($D$) | Repurchase AUC | Spend $R^2$ (log1p) | Invoices $R^2$ (log1p) |
|---|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | 40 | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | **114** | **$0.8605 \pm 0.0292$** | **$0.5028 \pm 0.0376$** | **$0.6045 \pm 0.0330$** |
| **FCM Soft (matched $k$)** | 40 | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Empirical Finding:* On repeat-rich consumer purchasing, Fuzzy RFM-FCA delivers substantial regression gains over raw RFM ($\Delta R^2_{\text{spend}} = +0.1364$, $\Delta R^2_{\text{inv}} = +0.1530$) and canonical FCM soft memberships ($\Delta R^2_{\text{spend}} = +0.0546$, $\Delta R^2_{\text{inv}} = +0.0525$). The 10-split CV demonstrates consistency across splits (9/10 and 10/10 split wins), while formal paired bootstrap testing on the fixed split confirms statistical significance ($p < 0.001$).

### 1.4 Fixed-Split Predictive Metrics (Seed = 42)

| Arm | $D$ | Repurchase AUC | Brier Score | Spend $R^2$ | Spend MAE | Spend $\rho$ | Invoice $R^2$ | Invoice MAE | Invoice $\rho$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | 0.8694 | 0.0615 | 0.3604 | 1.0707 | 0.7680 | 0.4745 | 0.6212 | 0.7660 |
| **Crisp RFM-FCA** | 40 | 0.8611 | 0.0548 | 0.4629 | 0.9703 | 0.7477 | 0.5536 | 0.5612 | 0.7623 |
| **Fuzzy RFM-FCA (Supp.)** | **114** | **0.8629** | **0.0529** | **0.5152** | **0.8973** | **0.7924** | **0.6188** | **0.5133** | **0.8065** |
| **FCM Soft ($k=40$)** | 40 | 0.8284 | 0.0557 | 0.4445 | 0.9618 | 0.7505 | 0.5537 | 0.5564 | 0.7678 |

#### Paired Bootstrap CIs ($B=1,000$ Resamples)
- **Fuzzy RFM-FCA vs Raw RFM Baseline:**
  - Spend $R^2$: $\mathbf{\Delta = +0.1548}$, $95\%$ CI $[+0.1019, +0.2136]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.1443}$, $95\%$ CI $[+0.1028, +0.1866]$ ($p < 0.001$).
  - Repurchase AUC: $\Delta = -0.0065$, $95\%$ CI $[-0.0349, +0.0196]$ ($p = 0.330$, spans zero).
- **Fuzzy RFM-FCA vs Crisp RFM-FCA:**
  - Spend $R^2$: $\mathbf{\Delta = +0.0523}$, $95\%$ CI $[+0.0302, +0.0769]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0391, +0.0905]$ ($p < 0.001$).
- **Fuzzy RFM-FCA vs FCM Soft ($k=40$):**
  - Repurchase AUC: $\mathbf{\Delta = +0.0345}$, $95\%$ CI $[+0.0053, +0.0663]$ ($p = 0.011$).
  - Spend $R^2$: $\mathbf{\Delta = +0.0707}$, $95\%$ CI $[+0.0328, +0.1075]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0358, +0.0948]$ ($p < 0.001$).

### 1.5 Geometric Clustering Benchmark (Standardized Raw RFM Space)

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

*Geometric Diagnostic Interpretation:* The negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties. Distance-based metrics (Silhouette, Davies-Bouldin) evaluate hyperspherical compactness and serve as geometric diagnostics rather than universal quality rankings.

### 1.6 Fuzzy Membership Threshold Sensitivity Analysis

To verify that the predictive utility of the fuzzy representation is not an artifact of the baseline threshold selection ($\mathcal{L} = \{0.3, 0.5, 0.7\}$), a one-factor sensitivity analysis was conducted across three threshold configurations on Dunnhumby under the locked evaluation protocol (observation days 1–620, holdout days 621–711, 10 temporal splits with seeds 1000–1009, 70/30 train/test split, support cutoff 0.04, $J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$, train-only fitting).

#### Concept-Space & Predictive Sensitivity Across Configurations:

| Configuration | Full-Cohort Raw Concepts | Full-Cohort Suppressed ($k$) | Compression Ratio | Mean AUC | $\Delta$ AUC vs Crisp | Mean Spend $R^2$ | $\Delta$ Spend $R^2$ vs Crisp | Mean Invoice $R^2$ | $\Delta$ Invoice $R^2$ vs Crisp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Crisp Baseline** | 38 | 38 | 1.00× | $0.8541 \pm 0.0242$ | *0.0000* | $0.4800 \pm 0.0351$ | *0.0000* | $0.5548 \pm 0.0344$ | *0.0000* |
| **(0.2, 0.5, 0.8)** *(Permissive)* | 652 | 265 | 2.46× (59.4%) | $0.8611 \pm 0.0299$ | **+0.0070** (6/10) | $0.5037 \pm 0.0387$ | **+0.0237** (9/10) | $0.6071 \pm 0.0313$ | **+0.0523** (10/10) |
| **(0.3, 0.5, 0.7)** *(Baseline)* | 502 | 123 | 4.08× (75.5%) | $0.8605 \pm 0.0292$ | **+0.0064** (7/10) | $0.5028 \pm 0.0376$ | **+0.0228** (9/10) | $0.6045 \pm 0.0330$ | **+0.0497** (10/10) |
| **(0.4, 0.5, 0.6)** *(Tight)* | 420 | 38 | 11.05× (91.0%) | $0.8673 \pm 0.0243$ | **+0.0132** (8/10) | $0.5090 \pm 0.0339$ | **+0.0290** (10/10) | $0.6079 \pm 0.0306$ | **+0.0531** (10/10) |

*Paired Split Wins vs Crisp Baseline:*
- Repurchase AUC: 6/10 for (0.2, 0.5, 0.8); 7/10 for (0.3, 0.5, 0.7); 8/10 for (0.4, 0.5, 0.6).
- Future Spend $R^2$: 9/10 for (0.2, 0.5, 0.8); 9/10 for (0.3, 0.5, 0.7); 10/10 for (0.4, 0.5, 0.6).
- Future Invoice $R^2$: 10/10 for all three configurations.

#### Methodological Interpretation:
The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space, downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations.

Concept-space structure is sensitive to threshold choice (widening to 0.2 expands raw candidate concepts to 652, while tightening to 0.4 reduces raw concepts to 420 and compresses to 38). However, predictive utility is comparatively stable (spend $R^2$ spans $0.5028$–$0.5090$, invoice $R^2$ spans $0.6045$–$0.6079$). Redundancy suppression substantially reduces concept-space size across all configurations. Therefore, the fuzzy extension is not dependent on one narrowly tuned threshold configuration. Thresholds were not optimized post-hoc based on test set results.

Detailed artifacts and comparison plots:
- Figure: [`fig_predictive_sensitivity.png`](../results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png)
- Report: [`sensitivity_analysis_report.md`](../results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md)
- Summary CSV: [`fuzzy_membership_sensitivity_summary.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_summary.csv)
- Split CSV: [`fuzzy_membership_sensitivity_splits.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_splits.csv)
- Full Cohort CSV: [`fuzzy_membership_sensitivity_full_cohort.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_full_cohort.csv)

---

## 2. Independent Second Domain: Online Retail II

### 2.1 Base-Paper Replication
- Replicated the 5,878 clean unique customer cohort from Rungruang et al. (2024).
- Our reconstruction recovered all 31 published frequent concept intents ($\text{support} > 0.04$) reported by Rungruang et al. However, exact customer counts matched for only 3 of the 31 concepts. The remaining discrepancies are consistent with the paper's unspecified tie-breaking procedure for customers sharing identical frequency values, particularly the 1,623 customers with $F = 1$.
- Replicated K-Means and Ward clustering geometries across $k=2..10$ (Silhouette ~0.33–0.38, DB ~0.99–1.07).

### 2.2 Crisp RFM-FCA versus Fuzzy RFM-FCA
- Discovered 1,064 closed concepts under uncapped FP-growth (`max_len = None`).
- Greedy redundancy suppression ($J_{\max} = 0.80$) mitigates concept proliferation, compressing 445 candidate concepts to **95 concepts** (4.7× reduction), with pairwise extent Jaccard $\ge 0.80$ dropped from $3.3\%$ to $0.0\%$. After suppression, downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
- **10-Split Temporal Performance:**
  - Raw RFM: AUC $0.7770 \pm 0.0117$, Spend $R^2 = 0.3476$, Invoice $R^2 = 0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** ($p = 2.3 \times 10^{-4}$ vs crisp via paired t-test), Spend $R^2 = \mathbf{0.3664}$, Invoice $R^2 = \mathbf{0.4952}$.

---

## 3. Cross-Domain Comparative Synthesis

| Dimension | Primary Domain: Dunnhumby | Second Domain: Online Retail II |
|---|---|---|
| **Setting** | Supermarket Grocery | Non-Store Giftware Retail |
| **Population ($N$)** | 2,499 households | 5,878 customers |
| **Repeat Purchase Rate** | **99.68%** | **72.39%** |
| **Median Purchases ($F$)** | 79.0 baskets | 3.0 orders |
| **Suppressed Concepts** | 123 (from 502) | 95 (from 445) |
| **Compression Ratio** | **4.1×** | **4.7×** |
| **Predictive Lift: Spend $R^2$** | **+0.1364** vs raw ($0.5028$ vs $0.3664$) | **+0.0188** vs raw ($0.3664$ vs $0.3476$) |
| **Predictive Lift: Invoice $R^2$** | **+0.1530** vs raw ($0.6045$ vs $0.4515$) | **+0.0310** vs raw ($0.4952$ vs $0.4642$) |
| **Threshold Sensitivity** | Predictive relationship consistent on (0.2, 0.5, 0.8), (0.3, 0.5, 0.7), (0.4, 0.5, 0.6) | Locked baseline (0.3, 0.5, 0.7) |

*Synthesis Interpretation:* The results provide evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.


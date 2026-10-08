> **SUPERSEDED (2026-10-08).** Historical copy of `docs/PROJECT_DOCUMENT.md` as last committed (git HEAD 8a420e2), before the audit. It reports dense-rank scoring, an untransformed Raw RFM baseline, an incorrect 98.08% repurchase rate and "temporal CV" labels. Do not cite. Current documents: `README.md`, `docs/RESULTS_SUMMARY.md`; errata: `docs/AUDIT_ERRATA.md`.

# Fuzzy Formal Concept Analysis (FCA) for Customer Segmentation: Research Gap, Methodology, Audited Resolutions, and Final Cross-Domain Findings

**Document Status:** Definitive Research Project Document (Locked Methodology v4)  
**Primary Validation Domain:** Dunnhumby "The Complete Journey" ($N = 2,499$ households)  
**Second Validation Domain:** Online Retail II ($N = 5,878$ clean customers)  
**Base Paper:** Rungruang et al., *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*, *Expert Systems with Applications* (2024)  
**Repository:** [rfms-fuzzy-fca-segmentation](https://github.com/nandana-das/rfms-fuzzy-fca-segmentation)

---

## 1. Research Gap & Theoretical Lineage

Customer relationship management and marketing segmentation rely heavily on the classical Recency, Frequency, and Monetary (RFM) model. In a foundational study, Rungruang et al. (2024) integrated RFM quintile segmentation with Formal Concept Analysis (FCA) on the UK Online Retail II transaction dataset, demonstrating that concept lattices generate interpretable, multi-dimensional customer profiles with hierarchical partial-order structures.

Crucially, Rungruang et al. concluded their study by explicitly highlighting structural limitations and mapping future research avenues:
1. **Crisp Binary Formal Context:** The base model applied hard quintile binning (5 crisp bins per dimension, $|M| = 15$), causing artificial boundary discontinuities where borderline customers with negligible RFM differences were placed in disjoint bins. The authors explicitly proposed:  
   > *"we will modify and improve this model by representing RFM values in a non-binary formal context in future studies."*
2. **Concept Proliferation & Redundancy:** Continuous or multi-level fuzzy scalings generate near-duplicate concepts covering almost identical customer subsets, leading to high-dimensional feature explosion.
3. **Fixed Support-Based Filtering:** The base paper uses a fixed support threshold of 0.04 and does not report a data-driven procedure for selecting this threshold.
4. **Single-Domain Validation:** The base paper evaluated only a single giftware retail transaction dataset; cross-domain validation on repeat-rich consumer purchasing (e.g., household supermarket transactions) was unexamined.
5. **Asymmetric Clustering Evaluation:** The base paper evaluated FCA qualitatively while evaluating K-Means and Ward clustering using geometric compactness metrics (Silhouette, Davies–Bouldin).
6. **Absence of Predictive Validation:** The utility of discovered concept structures for out-of-sample customer forecasting (future repurchase, spending, and basket counts) remained untested under leakage-free temporal protocols.
7. **Threshold Sensitivity & Robustness:** Prior fuzzy FCA formulations lacked sensitivity evaluations of multi-level cutoffs, leaving unexamined whether predictive gains rely on narrow parameter tuning.

This project directly operationalizes the non-binary formal context research agenda proposed by Rungruang et al. while addressing these structural gaps.

---

## 2. Research Objectives (Locked Methodology v4)

1. **Develop a Centroid-Based Fuzzy Formal Context:** Construct continuous piecewise-linear fuzzy membership functions with outer shoulder saturation, converting continuous RFM attributes into continuous membership degrees $\mu \in [0, 1]$ and reducing hard boundary discontinuities introduced by crisp quintile discretization, while acknowledging structural choices (five behavioral bands, centroid construction, and $\mathcal{L}$-fuzzy threshold cuts).
2. **Implement Data-Driven Concept Reduction:** The Kneedle-inspired heuristic provides a data-driven secondary pruning criterion over the mined candidate concepts, complementing the minimum support threshold used during concept generation.
3. **Develop Greedy Extent-Jaccard Redundancy Suppression:** Introduce a concept-deduplication pass ($J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$) that mitigates concept proliferation by suppressing near-duplicate concepts, with downstream predictive performance remaining comparable to the unsuppressed representation under the evaluated protocol.
4. **Establish a Principled Fuzzy Clustering Benchmark:** Implement canonical **Fuzzy C-Means (FCM, $m=2.0$)** as a soft-clustering benchmark and evaluate hard partitions via **Top-$k$ Membership Hardening** in the standardized raw RFM space, separating spatial compactness from conceptual lattice closure.
5. **Ensure Leakage-Free Temporal Predictive Validation:** Formulate a strictly controlled out-of-sample prediction protocol where all transformations, cutoffs, and concept extractions are fit exclusively on training folds.
6. **Conduct Rigorous Cross-Domain Validation:** Validate the framework across two repeat-transaction domains: **Dunnhumby "The Complete Journey"** (Primary Domain) and **Online Retail II** (Second Domain).
7. **Evaluate Fuzzy Threshold Sensitivity:** Conduct one-factor sensitivity analysis across alternative $\mathcal{L}$-fuzzy threshold tuples to verify that the predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested configurations and is not an artifact of threshold tuning.

> **Out-of-Scope Notice:** The final study is strictly **RFM only** ($R, F, M$). Early exploratory investigations into marketplace frequency sparsity ($F^*$), customer review satisfaction ($S$), and the single-buyer Olist marketplace are documented separately as historical experiments and are formally excluded from the final paper.

---

## 3. Formal Methodology

### 3.1 Data Sanitation & Persistent Entity Resolution
- **Customer Entity Key ($c_j$):** Customer transactions are aggregated using persistent household/customer identifiers (`household_key` for Dunnhumby; `CustomerID` for Retail II), strictly avoiding transaction- or basket-scoped keys.
- **Filtering Rules:** Excludes cancelled transactions, missing customer identifiers, and non-positive prices/quantities.
- **RFM Metric Definitions:**
  - **Recency ($R_j$):** Elapsed days from the customer's last observed transaction to the observation cutoff timestamp:
    $$R_j = T_{\text{cutoff}} - \max_{o_i \in D(c_j)} \text{timestamp}(o_i)$$
  - **Frequency ($F_j$):** Total count of distinct shopping baskets / invoices placed by the customer:
    $$F_j = |\{ \text{BASKET\_ID} \mid o_i \in D(c_j) \}|$$
    *Note:* For Dunnhumby, product-line rows are aggregated to distinct `BASKET_ID` values. No $F^*$ or item-weighting is applied.
  - **Monetary ($M_j$):** Total monetary expenditure across all observed transactions:
    $$M_j = \sum_{o_i \in D(c_j)} \text{SALES\_VALUE}(o_i)$$
    *Note:* `SALES_VALUE` is aggregated directly without multiplying by quantity, as recorded in the transaction register.

### 3.2 Dense-Rank Fractional Scoring
To prevent empty quantile bins caused by identical raw transaction values, quintile scores are assigned via fractional dense ranking on the training fold:
$$\text{score}(X_j) = \left\lceil 5 \cdot \frac{\text{dense\_rank}(X_j)}{K_X} \right\rceil$$
where $K_X$ is the total count of distinct raw values observed for feature $X$. For Recency, the rank is inverted such that lower raw elapsed days receive higher scores.

### 3.3 Centroid-Based Piecewise-Linear Fuzzy Context
1. **Band Centroids:** Centroids $c_1 < c_2 < c_3 < c_4 < c_5$ are computed as the medians of raw feature values within each score band.
2. **Piecewise-Linear Interpolation:** Membership degrees $\mu_k(X_j) \in [0, 1]$ interpolate linearly between adjacent centroids, with outer shoulder saturation for bands 1 and 5:
   - For $x \le c_1$: $\mu_1(x) = 1.0$, all other $\mu_k(x) = 0.0$.
   - For $c_k \le x \le c_{k+1}$:
     $$\mu_k(x) = \frac{c_{k+1} - x}{c_{k+1} - c_k}, \quad \mu_{k+1}(x) = \frac{x - c_k}{c_{k+1} - c_k}$$
   - For $x \ge c_5$: $\mu_5(x) = 1.0$, all other $\mu_k(x) = 0.0$.
   Row-sums per dimension strictly equal 1.0: $\sum_{k=1}^5 \mu_k(X_j) = 1.0$, reducing hard boundary discontinuities while retaining structural design choices.
3. **$\mathcal{L}$-Fuzzy Scaling:** Continuous memberships are scaled into a multi-level binary context $\mathbb{K}_L$ using Belohlavek threshold scaling at baseline cuts $\mathcal{L} = \{0.3, 0.5, 0.7\}$, yielding $3 \times 15 = 45$ binary attributes. Sensitivity of this threshold tuple is systematically evaluated against $(0.2, 0.5, 0.8)$ and $(0.4, 0.5, 0.6)$ (Section 4.7).
4. **Closed Concept Mining:** Closed frequent itemsets are discovered using uncapped FP-growth (`max_len = None`, $\text{min\_support} = 0.04$).

### 3.4 Concept Reduction & Redundancy Suppression
1. **Kneedle-Inspired Secondary Pruning:** Closed fuzzy concepts are first generated subject to the minimum support criterion (0.04). The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on the observed support ($\text{supp}_{\min}^*$) and stability-proxy ($\theta^*$) distributions, complementing the candidate generation threshold. Stability is an object-profile diversity proxy, distinct from canonical Kuznetsov stability.
2. **Greedy Extent-Jaccard Redundancy Suppression ($J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$):**
   - Candidate concepts ordered by Support ($\downarrow$), Intent Size ($\uparrow$), and Stability ($\downarrow$).
   - A candidate concept $C_{\text{cand}}$ is retained if and only if its core extent (customers with $\mu \ge 0.5$) satisfies:
     $$J(A(C_{\text{cand}}), A(C_{\text{kept}})) = \frac{|A(C_{\text{cand}}) \cap A(C_{\text{kept}})|}{|A(C_{\text{cand}}) \cup A(C_{\text{kept}})|} < 0.80 \quad \forall C_{\text{kept}}$$
   - Downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
3. **Continuous Concept Membership Matrix:** Customer membership in concept $C = (A, B)$ is computed using the Gödel minimum t-norm:
   $$\mu_C(c_j) = \min_{b \in B} \mu_b(c_j)$$

### 3.5 Benchmarking & Evaluation Protocol
- **Hard Clustering Evaluation:** Evaluates K-Means, Ward, canonical FCM (argmax partition), and Fuzzy RFM-FCA (**Top-$k$ Membership Hardening**) in the common standardized Raw RFM space ($X_{\text{raw}}$, shape $N \times 3$) using Silhouette and Davies–Bouldin metrics.
  - *Geometric Diagnostic Note:* The negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties. Distance-based metrics serve as geometric diagnostics rather than universal quality rankings.
  - *Terminology Distinction:* **Alpha-cut** thresholds continuous membership at $\alpha \ge 0.5$ (used for concept extent definition and redundancy suppression). **Top-$k$ Membership Hardening** selects the $k$ highest-support non-trivial concepts and assigns each customer via $\operatorname{argmax}_{j \in \{1..k\}} \mu_j(x)$.
- **Fuzzy Clustering Baseline:** Canonical Fuzzy C-Means ($m=2.0$, K-means++ seeded) evaluated with Fuzzy Partition Coefficient (FPC) and Xie-Beni index (scoped strictly to FCM).
- **Leakage-Free Temporal Prediction:** Models are trained to predict future customer holdout outcomes:
  - Repurchase classification: LogisticRegressionCV ($C_s=10$, 5-fold CV) evaluated via ROC-AUC and Brier Score.
  - Future spend & invoice regression: RidgeCV ($\alpha \in [10^{-3}, 10^3]$, 5-fold CV) on $\ln(1 + \text{spend})$ and $\ln(1 + \text{invoices})$ evaluated via $R^2$, MAE, and Spearman rank correlation ($\rho$).
  - Evaluated on a fixed single temporal split (with 1,000 paired customer bootstrap resamples for formal significance) and across 10 independent temporal cross-validation splits (seeds 1000–1009) to assess consistency across splits.

---

## 4. Empirical Results: Dunnhumby Primary Validation

### 4.1 Cohort Statistics
- **Full Population (Days 1–711):** 2,500 households, 2,595,732 transaction records, 276,484 baskets. Mean $F = 110.6$ baskets (median 79.0, max 1,300).
- **Observation Cohort (Days 1–620):** 2,499 households, 238,873 baskets. Mean $F = 95.6$ baskets (median 67.0).
- **Holdout Cohort (Days 621–711, 91 Days):** 2,499 households. Future repurchase rate = $98.08\%$ (2,451 repurchasers). Mean future spend = \$482.05 (std \$558.46); mean future invoices = $15.05$ (std $17.08$).

*Classification Ceiling Effect:* Because 98.08% of households repurchased during the 91-day holdout, repurchase classification exhibits a strong class-imbalance/ceiling effect. The fuzzy-versus-crisp AUC difference is therefore small and spans zero under the fixed-split bootstrap comparison. The clearer predictive gains occur in future monetary expenditure and transaction activity.

### 4.2 Concept Lattice & Compression
- **Raw Closed Concepts:** 502 fuzzy concepts (min support = 0.04, `max_len = None`).
- **Kneedle-Inspired Secondary Pruning:** Closed fuzzy concepts are first generated subject to the 0.04 minimum support criterion. The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on observed distributions, yielding support cutoff $= 0.2129$ and stability-proxy cutoff $= 0.4285$ (evaluated via an object-profile diversity proxy, distinct from canonical Kuznetsov stability).
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Retains **123 concepts** (379 concepts removed; 4.1× compression). Pre-suppression Mean concepts per customer $= 57.0$ (median 61.6); falls to ~4.3 post-suppression.

### 4.3 10-Split Temporal Cross-Validation Performance (Seeds 1000–1009)

| Representation Arm | Features ($D$) | Repurchase AUC | Spend $R^2$ (log1p) | Invoices $R^2$ (log1p) |
|---|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | 40 | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | **114** | **$0.8605 \pm 0.0292$** | **$0.5028 \pm 0.0376$** | **$0.6045 \pm 0.0330$** |
| **FCM Soft (matched $k$)** | 40 | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Summary:* The 10-split temporal cross-validation demonstrates consistency across splits (9/10 wins on future spend, 10/10 wins on future invoices). Formal statistical significance ($p < 0.001$) is supported by paired bootstrap testing on the fixed split (Section 4.5).

### 4.4 Fixed-Split Predictive Metrics (Seed 42, Train $N=1,749$, Test $N=750$)

| Arm | $D$ | Repurchase AUC | Brier Score | Spend $R^2$ | Spend MAE | Spend $\rho$ | Invoice $R^2$ | Invoice MAE | Invoice $\rho$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | 0.8694 | 0.0615 | 0.3604 | 1.0707 | 0.7680 | 0.4745 | 0.6212 | 0.7660 |
| **Crisp RFM-FCA** | 40 | 0.8611 | 0.0548 | 0.4629 | 0.9703 | 0.7477 | 0.5536 | 0.5612 | 0.7623 |
| **Fuzzy RFM-FCA (Supp.)** | **114** | **0.8629** | **0.0529** | **0.5152** | **0.8973** | **0.7924** | **0.6188** | **0.5133** | **0.8065** |
| **FCM Soft ($k=40$)** | 40 | 0.8284 | 0.0557 | 0.4445 | 0.9618 | 0.7505 | 0.5537 | 0.5564 | 0.7678 |

### 4.5 Paired Bootstrap Confidence Intervals ($B=1,000$ Resamples)
- **Fuzzy RFM-FCA vs Raw RFM Baseline:**
  - Repurchase AUC: $\Delta = -0.0065$, $95\%$ CI $[-0.0349, +0.0196]$ (spans zero, $p=0.330$).
  - Spend $R^2$: $\mathbf{\Delta = +0.1548}$, $95\%$ CI $[+0.1019, +0.2136]$ (excludes zero, $p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.1443}$, $95\%$ CI $[+0.1028, +0.1866]$ (excludes zero, $p < 0.001$).
- **Fuzzy RFM-FCA vs Crisp RFM-FCA:**
  - Spend $R^2$: $\mathbf{\Delta = +0.0523}$, $95\%$ CI $[+0.0302, +0.0769]$ (excludes zero, $p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0391, +0.0905]$ (excludes zero, $p < 0.001$).
- **Fuzzy RFM-FCA vs FCM Soft ($k=40$):**
  - Repurchase AUC: $\mathbf{\Delta = +0.0345}$, $95\%$ CI $[+0.0053, +0.0663]$ (excludes zero, $p = 0.011$).
  - Spend $R^2$: $\mathbf{\Delta = +0.0707}$, $95\%$ CI $[+0.0328, +0.1075]$ (excludes zero, $p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0358, +0.0948]$ (excludes zero, $p < 0.001$).

### 4.6 Geometric Hard Clustering Benchmark (Standardized Raw RFM Space)

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

### 4.7 Fuzzy Membership Threshold Sensitivity Analysis

To rigorously assess whether the empirical advantages of the fuzzy representation depend on the baseline threshold tuple ($\mathcal{L} = \{0.3, 0.5, 0.7\}$), a dedicated one-factor sensitivity analysis was executed across three threshold configurations on Dunnhumby under the locked evaluation protocol (observation days 1–620, holdout days 621–711, 10 temporal splits with seeds 1000–1009, 70/30 stratified train/test split, support cutoff 0.04, $J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$, train-only fitting).

#### Experimental Configurations:
1. **Conservative / Permissive:** $(0.2, 0.5, 0.8)$ — expands candidate generation with a lower entry cut (0.2).
2. **Baseline:** $(0.3, 0.5, 0.7)$ — locked primary methodology.
3. **Tight:** $(0.4, 0.5, 0.6)$ — restricts candidate attribute generation with a narrower cut interval.

#### Concept-Space & Predictive Sensitivity (Full-Cohort Concept Counts & 10-Split Predictive Means):

| Configuration | Full-Cohort Raw Concepts | Full-Cohort Suppressed ($k$) | Compression Ratio | Repurchase AUC (Mean ± SD) | $\Delta$ AUC vs Crisp | Future Spend $R^2$ (Mean ± SD) | $\Delta$ Spend $R^2$ vs Crisp | Future Invoice $R^2$ (Mean ± SD) | $\Delta$ Invoice $R^2$ vs Crisp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Crisp Baseline** | 38 | 38 | 1.00× | $0.8541 \pm 0.0242$ | *0.0000* | $0.4800 \pm 0.0351$ | *0.0000* | $0.5548 \pm 0.0344$ | *0.0000* |
| **(0.2, 0.5, 0.8)** *(Permissive)* | 652 | 265 | 2.46× (59.4%) | $0.8611 \pm 0.0299$ | **+0.0070** (6/10) | $0.5037 \pm 0.0387$ | **+0.0237** (9/10) | $0.6071 \pm 0.0313$ | **+0.0523** (10/10) |
| **(0.3, 0.5, 0.7)** *(Baseline)* | 502 | 123 | 4.08× (75.5%) | $0.8605 \pm 0.0292$ | **+0.0064** (7/10) | $0.5028 \pm 0.0376$ | **+0.0228** (9/10) | $0.6045 \pm 0.0330$ | **+0.0497** (10/10) |
| **(0.4, 0.5, 0.6)** *(Tight)* | 420 | 38 | 11.05× (91.0%) | $0.8673 \pm 0.0243$ | **+0.0132** (8/10) | $0.5090 \pm 0.0339$ | **+0.0290** (10/10) | $0.6079 \pm 0.0306$ | **+0.0531** (10/10) |

*Paired Split Wins across 10 Splits:*
- Repurchase AUC: 6/10 for (0.2, 0.5, 0.8); 7/10 for (0.3, 0.5, 0.7); 8/10 for (0.4, 0.5, 0.6).
- Future Spend $R^2$: 9/10 for (0.2, 0.5, 0.8); 9/10 for (0.3, 0.5, 0.7); 10/10 for (0.4, 0.5, 0.6).
- Future Invoice $R^2$: 10/10 for all three configurations.

#### Methodological Findings & Robustness Interpretation:
The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space, downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations.

Specifically:
- **Concept-space structure is sensitive to threshold choice:** Relaxing the outer cut to 0.2 expands raw candidate concepts to 652 (265 retained post-suppression), whereas tightening to 0.4 reduces raw concepts to 420 and compresses them aggressively to 38 retained concepts.
- **Predictive utility is comparatively stable:** Across all three configurations, downstream spend $R^2$ remains within $0.5028$–$0.5090$ (all exceeding crisp by $+0.023$ to $+0.029$) and invoice $R^2$ remains within $0.6045$–$0.6079$ (all exceeding crisp by $+0.050$ to $+0.053$, with unanimous 10/10 split wins).
- **Redundancy suppression is consistently effective:** The greedy extent-Jaccard procedure mitigates concept proliferation by suppressing near-duplicate concepts under the specified $J_{\max} = 0.80$ criterion, substantially reducing concept-space size under all configurations ($59.4\%$ to $91.0\%$ reduction).
- **Non-tuning assurance:** These findings confirm that the predictive advantage of the fuzzy representation over crisp RFM-FCA does not depend on narrow post-hoc threshold optimization. Thresholds were not tuned post-hoc.

Generated Artifacts:
- Visual Diagnostic: [`results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png`](../results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png)
- Full Audit Report: [`results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md`](../results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md)
- Metric CSVs: [`fuzzy_membership_sensitivity_summary.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_summary.csv), [`fuzzy_membership_sensitivity_splits.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_splits.csv), [`fuzzy_membership_sensitivity_full_cohort.csv`](../results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_full_cohort.csv)

---

## 5. Independent Validation: Online Retail II

### 5.1 Base Paper Replication
- **Clean Customer Cohort:** 5,878 clean customers from Online Retail II (Table 3 match).
- **Intent Recovery:** Our reconstruction recovered all 31 published frequent concept intents ($\text{support} > 0.04$) reported by Rungruang et al. However, exact customer counts matched for only 3 of the 31 concepts. The remaining discrepancies are consistent with the paper's unspecified tie-breaking procedure for customers sharing identical frequency values, particularly the 1,623 customers with $F = 1$.
- **Clustering Geometry Replication:** Replicated K-Means and Ward clustering across $k=2..10$ on 5,633 outlier-filtered customers (Silhouette ~0.33–0.38, DB ~0.99–1.07).

### 5.2 Concept Reduction & Predictive Holdout (10 Splits)
- **Uncapped Mining:** Discovered 1,064 closed concepts (`max_len = None`).
- **Greedy Redundancy Suppression ($J_{\max} = 0.80$):** Mitigates concept proliferation by compressing 445 candidate concepts into **95 concepts** (4.7× reduction), suppressing all near-duplicate concept pairs (pairs with Jaccard $\ge 0.80$ dropped from $3.3\%$ to $0.0\%$). After suppression, downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
- **10-Split Temporal Performance:**
  - Raw RFM: AUC $0.7770 \pm 0.0117$, Spend $R^2$ $0.3476$, Invoice $R^2$ $0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** ($p = 2.3 \times 10^{-4}$ vs crisp via paired t-test), Spend $R^2$ **$0.3664$**, Invoice $R^2$ **$0.4952$**.

---

## 6. Cross-Domain Comparative Synthesis

| Dimension | Primary Domain: Dunnhumby | Second Domain: Online Retail II |
|---|---|---|
| **Setting / Transaction Type** | Fast-Moving Consumer Goods (Grocery) | Non-Store Giftware Retail |
| **Customer Population ($N$)** | 2,499 households | 5,878 customers |
| **Observation Window** | 620 days continuous | 730 days continuous |
| **Repeat Purchase Rate** | **99.68%** | **72.39%** |
| **Median Purchases ($F$)** | 79.0 baskets | 3.0 orders |
| **Discovered Closed Concepts** | 502 | 1,064 |
| **Suppressed Concepts** | 123 (from 502) | 95 (from 445) |
| **Predictive Lift vs Raw RFM (Spend $R^2$)** | **+0.1364** ($0.5028$ vs $0.3664$) | **+0.0188** ($0.3664$ vs $0.3476$) |
| **Predictive Lift vs Raw RFM (Invoice $R^2$)** | **+0.1530** ($0.6045$ vs $0.4515$) | **+0.0310** ($0.4952$ vs $0.4642$) |
| **Threshold Sensitivity** | Predictive relationship consistent on (0.2, 0.5, 0.8), (0.3, 0.5, 0.7), (0.4, 0.5, 0.6) | Locked baseline (0.3, 0.5, 0.7) |

*Synthesis Interpretation:* The results provide evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.

---

## 7. Audited Methodological Resolutions

1. **Permanent Redundancy Suppression Integer Fix:** Cast extent indicator arrays to `np.int32` before the matrix multiplication `@` in `scripts/concept_redundancy.py`, preventing boolean matrix multiplication overflow and ensuring genuine Jaccard overlap computation.
2. **Unified Matched-$k$ FCM Labeling:** Standardized the dynamic matched-$k$ FCM baseline across temporal splits as `"FCM Soft (matched k)"`.
3. **Hard-Clustering Protocol Alignment:** Standardized evaluation of K-Means, Ward, FCM, and FCA hard partitions directly in the agreed customer space ($X_{\text{raw}}$, shape $N \times 3$) and designated the benchmark **Top-$k$ Membership Hardening** (distinct from an $\alpha$-cut).
4. **Metric Scope Isolation:** Preserved canonical FPC and Xie-Beni exclusively for canonical FCM.
5. **No Data Leakage:** All scalers, dense-rank cutoffs, fuzzy centroids, concept lattices, and predictive models are fit strictly on training splits.

---

## 8. Defensible Contributions & Paper Positioning

### What This Work Does NOT Claim:
- Does **not** claim to invent RFM + FCA (established by Rungruang et al., 2024).
- Does **not** claim that hierarchical or overlapping concept lattices are novel to this work.
- Does **not** claim universal superiority over K-Means, Ward, or FCM in geometric compactness.
- Does **not** claim that negative silhouette values prove clustering failure or superiority; the negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties.
- Does **not** claim canonical Kneedle (the threshold is Kneedle-inspired).
- Does **not** claim canonical Kuznetsov stability (the metric is an object-profile stability proxy).
- Does **not** claim that redundancy suppression proves removed concepts were useless in isolation.
- Does **not** claim that exact replication of base-paper customer counts is uniquely determinable without the omitted tie-breaking rule.
- Does **not** claim that threshold configurations were tuned or optimized based on test results (the study is strictly a sensitivity analysis, not model selection).
- Does **not** claim that the baseline threshold tuple is mathematically optimal.
- Does **not** claim that representation engineering solves physical frequency sparsity or generalizes universally to all e-commerce settings.

### Validated Methodological Contributions:
1. **Centroid-Based Fuzzy RFM-FCA:** Operationalizes the proposed future work of Rungruang et al. (2024) by replacing crisp binary quintiles with centroid-based piecewise-linear fuzzy memberships with outer shoulder saturation, reducing hard boundary discontinuities while acknowledging structural parameter choices.
2. **Data-Driven Concept Reduction:** The Kneedle-inspired heuristic provides a data-driven secondary pruning criterion over the mined candidate concepts, complementing the minimum support threshold used during concept generation.
3. **Greedy Extent-Jaccard Redundancy Suppression:** Establishes a concept deduplication mechanism ($J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$) that achieves 4×–5× concept compression without degrading downstream predictive performance under the evaluated protocol.

### Evaluative & Empirical Contributions:
4. **Controlled Replication:** Reconstructs the 5,878 clean customer cohort on Online Retail II and recovers all 31 published frequent concept intents, explicitly documenting the tie-breaking limitations for $F=1$.
5. **Principled Clustering Evaluation:** Benchmarks against canonical Fuzzy C-Means ($m=2.0$) and evaluates Top-$k$ Membership Hardening in standardized feature space, separating spatial compactness from Galois closure.
6. **Leakage-Free Predictive Validation:** Demonstrates consistent out-of-sample predictive gains on future spending and transaction volumes across repeat-rich consumer transaction domains, with formal statistical significance confirmed by paired bootstrap testing on fixed holdouts ($p < 0.001$).
7. **Cross-Domain Repeat-Purchasing Validation:** Validates the framework across grocery supermarket purchasing (Dunnhumby) and giftware retail (Online Retail II), demonstrating domain-dependent regression lift.
8. **Threshold Sensitivity Analysis:** Demonstrates that the predictive relationship between fuzzy and crisp RFM-FCA remains consistent across reasonable perturbations of the L-fuzzy threshold tuple ($(0.2, 0.5, 0.8)$, $(0.3, 0.5, 0.7)$, and $(0.4, 0.5, 0.6)$), confirming that performance does not rely on narrow parameter tuning.

---

## 9. Historical Exploratory Studies (Out of Scope)

Preliminary exploratory research in this project evaluated the **Olist Brazilian E-Commerce marketplace** ($N = 93,357$), testing:
- An expanded **RFMS context** incorporating customer review satisfaction ($S$).
- An entropy-optimized composite purchase-intensity index ($F^*$).
- Marketplace frequency sparsity where 97% of customers made only a single purchase.

**Why Olist Was Excluded from the Final Paper:**
1. **Severe Frequency Sparsity:** Under controlled ablation (`scripts/ablation_frequency_variants.py`), neither literal $F$, composite $F^*$, nor engagement indices could overcome the 97% single-buyer boundary.
2. **Temporal Review Leakage:** An audit (`scripts/audit_olist_temporal_leakage.py`) showed that earlier inflated holdout results (AUC ~0.667) were artifacts of post-cutoff review aggregation. Once purged, Olist predictive performance collapsed to baseline (AUC 0.5612 vs 0.5587).
3. **Scientific Conclusion:** These investigations demonstrated that representation engineering cannot override intrinsic physical domain sparsity. The final study was therefore locked to repeat-rich consumer transaction domains (Dunnhumby and Online Retail II) under Methodology v4. All historical Olist scripts and results are preserved in `scripts/` and `results/` for transparency.

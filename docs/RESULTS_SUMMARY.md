# RFMS-Fuzzy FCA Customer Segmentation — Comprehensive Results Summary

## Executive Overview & Project Evolution

This document consolidates all quantitative findings, structural diagnostics, ablation experiments, and cross-domain evaluations for the RFMS-Fuzzy Formal Concept Analysis (FCA) customer segmentation framework across the **Olist Brazilian E-Commerce marketplace** ($N = 93,357$) and the **UK Online Retail II dataset** ($N = 5,878$).

### Chronological Development & Audit Phases:
1. **Core Pipeline (v3):** Removed the binding `max_len=6` restriction in FP-growth (revealing the true closed concept counts: 1,369 Olist, 1,064 Retail II), formalized centroid-based piecewise-linear fuzzy memberships, executed Kneedle sensitivity analyses (Step 8), tested $F1$-group structure recovery (Step 9), extracted interpretable overlap profiles (Step 10), and evaluated fuzzy validity diagnostics (Step 11).
2. **Improvements Line (v4):** Adopted greedy extent-Jaccard redundancy suppression ($J_{\max}=0.80$), cutting Retail II concepts from 445 to 95 with zero predictive loss; integrated canonical Fuzzy C-Means (FCM) to isolate fuzzy membership from lattice structure; and implemented customer-level multi-split temporal validation.
3. **The Four Audited Problems (Frozen Resolution):**
   - **Problem 1 (Frequency Sparsity Ablation):** Controlled ablation across Literal ($F_1$), Current composite ($F_2 / F^*$), and Revised Engagement ($F_3$) definitions. Confirmed that single-order dominance (97%) is an intrinsic domain property; retained $F^*$ as an entropy-maximizing purchase-intensity heuristic with documented caveats.
   - **Problem 2 (Clustering Metric Mismatch):** Diagnosed why hard-converting fuzzy overlapping concepts degrades Silhouette/DB scores (99.86% overlap; 8 of 12 clusters negative). Integrated canonical FCM ($m=2.0$) as the methodologically sound fuzzy-to-fuzzy benchmark.
   - **Problem 3 (Temporal Review Leakage Audit & Elimination):** Uncovered that full-period review aggregation contaminated 45.67% of repurchasers with post-cutoff satisfaction information, inflating earlier holdout AUC to ~0.667. Permanently retracted the leaked result; established a strictly leakage-free protocol under which Olist predictive performance realistically aligns with raw baseline features (AUC 0.5612 vs 0.5587).
   - **Problem 4 (Base-Paper Replication & Lineage Audit):** Faithfully replicated Rungruang et al. (2024) on Online Retail II ($N=5,878$), recovering all 31 published frequent concept intents ($>0.04$ support) and replicating K-means/Ward cluster geometries ($k=2..10$), while establishing that non-binary/fuzzy FCA was explicitly proposed by the base authors as future work.

---

## 1. Core Pipeline Results (Steps 1–13)

### Step 1 — RFMS Feature Engineering
- **Customer Population:** 96,096 unique customer identifiers; 93,357 retained after delivered-status and valid-payment filtering.
- **Repeat-Buyer Rate:** 3.00% (single-order rate 97.00%).
- **Entropy-Optimized $F^*$:** Grid search selects $\alpha=0.20, \beta=0.05, \gamma=0.75$, producing 5-band Shannon entropy of $0.1457$ nats ($9.05\%$ of $\ln(5)$). Score band 1 share is $97.00\%$, exactly matching the single-order rate.
- **Continuous Residual Spread:** Within band 1 ($N=90,553$), raw $F^*$ retains 15 distinct values with standard deviation $0.0082$.
- **Satisfaction Imputation:** 603 customers ($0.65\%$) without observed reviews receive the observed median/mode ($S=5.0$), increasing the $S=5.0$ share from $58.70\%$ to $58.97\%$ (+0.27 percentage points).

### Step 2 — Fuzzy Formal Context Construction
- **Attributes:** 20 fuzzy attributes ($R1..R5, F1..F5, M1..M5, S1..S5$). Membership matrix dimensions: $93,357 \times 20$. Row-sums per dimension strictly equal 1.0.
- **Centroid-Based Piecewise-Linear Memberships:** Continuous interpolation between medians of adjacent score bands, with outer shoulder saturation.
- **$\mathcal{L}$-Fuzzy Scaling:** Discretized at cut thresholds $\{0.3, 0.5, 0.7\}$ into a 60-attribute crisp multi-level context.
- **Closed Concept Mining:** Uncapped FP-growth (`max_len=None`, $\text{min\_support}=0.02$) discovers **40,906 frequent itemsets** and **1,369 closed concepts** (true maximum itemset length: 12).

### Step 3 — Stability & Kneedle Iceberg Pruning
- **Kneedle Thresholds:** $\text{supp}_{\min}^* = 0.1025$, $\theta^* = 0.9876$.
- **Pruned Concepts:** **153 concepts** survive with **304 Hasse covering edges**.
- **Stability Proxy:** Object-profile diversity proxy ($1 - \text{unique\_profiles}/|A|$) used for tractable pruning.
- **Sensitivity Analysis (Step 8):** Support threshold is robust (monotonic $\pm 20\%$ concept variation under $\pm 10\%$ shift). Stability threshold $\theta^*$ is robust downward, but saturated near the 1.0 ceiling upward on Olist.

### Step 4 — Alpha-Cut Hard Cluster Assignment
- **Top-Level Concepts:** 12 single-dimension concepts ($F1, S5, M1, M2, R4, R5, R3, M3, R2, S4, M4, R1$).
- **Dominant Hard Assignment:** $F1$ captures $90.19\%$ ($84,199 / 93,357$) under mode assignment.
- **Multi-Membership Overlap:** **99.86%** of customers satisfy the alpha-cut ($\mu \ge 0.5$) in more than one top-level concept.

### Steps 5–6 — Clustering Benchmark Comparison (Olist)

| Method | $k$ | Silhouette | Davies–Bouldin | FPC |
|---|---:|---:|---:|---:|
| **Fuzzy-FCA (Natural $k$)** | 12 | 0.1656 | 5.9467 | 0.2370 |
| **Fuzzy-FCA (Matched)** | 4 | 0.2591 | 2.2196 | — |
| **Fuzzy-FCA (Matched)** | 5 | 0.2467 | 2.3919 | — |
| **Fuzzy-FCA (Matched)** | 6 | 0.2307 | 2.5921 | — |
| **K-Means Baseline** | 4 | 0.3494 | 0.9952 | — |
| **K-Means Baseline** | 5 | 0.3867 | 0.8391 | — |
| **K-Means Baseline** | 6 | 0.3917 | 0.8666 | — |
| **Ward's Hierarchical** | 5 | 0.3264 | 0.9300 | — |
| **Canonical FCM (Fuzzy)** | 5 | 0.3847 | 0.8445 | 0.5005 |
| **Canonical FCM (Fuzzy)** | 12 | 0.3222 | 1.1134 | 0.3758 |

*Diagnostic Interpretation:* The lower Silhouette and higher DB scores of hard-assigned FCA reflect boundary assignment artifacts of overlapping concepts, not conceptual invalidity. Canonical FCM matches K-means geometry while providing soft membership.

### Steps 9–11 — Diagnostics, Recovery & Overlap Profiles
- **$F1$-Group Recovery (Step 9):** Within the 90,553 frequency-tied customers, $7.02\%$ fan out into non-$F1$ hard clusters, and concept-membership standard deviation (14.49) matches the population standard deviation (14.63). However, correlation between raw $F^*$ residual and concept membership is $-0.151$, indicating differentiation is driven by $R, M, S$, not by $F^*$ itself.
- **Overlap Profiles (Step 10):** The dominant hard cluster $F1$ spans Recency 0 to 694 days, Monetary \$9.59 to \$13,664, and contains **111 distinct multi-band overlap signatures**. FCA preserves the multi-dimensional nuance that single crisp labels collapse.
- **Fuzzy Validity Diagnostics (Step 11):**
  - *Membership Entropy:* $R$ ($0.410$) and $M$ ($0.383$) are genuinely fuzzy; $F$ ($0.014$) and $S$ ($0.005$) are practically crisp due to marketplace tie mass and discrete integer reviews.
  - *Resampling Persistence:* **100% of the 153 pruned concepts persist across 10 independent 80% subsamples**.

---

## 2. Cross-Domain Comparative Benchmark (Retail II vs. Olist)

| Metric | Online Retail II (Repeat Retail) | Olist (Sparse Marketplace) |
|---|---|---|
| **Raw Transactions** | 1,067,371 rows | 100,000+ orders |
| **Clean Customers** | 5,878 | 93,357 |
| **Repeat-Buyer Rate** | **72.39%** | **3.00%** |
| **Analyzed Dimensions** | RFM (15 fuzzy attributes) | RFMS (20 fuzzy attributes) |
| **Entropy-Optimal Weights** | $\alpha=0.05, \beta=0.15, \gamma=0.80$ | $\alpha=0.20, \beta=0.05, \gamma=0.75$ |
| **Raw Closed Concepts** | 1,064 | 1,369 |
| **Kneedle Pruned Concepts** | 115 | 153 |
| **Redundancy Suppression** | **445 → 95 concepts** (4.7×) | **633 → 627 concepts** (negligible) |
| **Holdout Predictive Lift** | Significant across all metrics | Near baseline (AUC 0.5612 vs 0.5587) |

---

## 3. Problem 1 Resolution: Frequency Dimension Ablation Study

A controlled ablation was executed on the full Olist customer population ($N=93,357$) comparing three frequency formulations under identical downstream fuzzy-context construction, uncapped FP-growth, Kneedle pruning, and clustering benchmarks:
- **Variant A (Literal Frequency, $F_1$):** $F_1 = n_{\text{orders}}$
- **Variant B (Current Project $F^*$, $F_2$):** $F_2 = 0.20\,n_{\text{orders}} + 0.05\,\ln(1+\text{qty}) + 0.75\,\mathbb{1}[n>1]$
- **Variant C (Revised Engagement, $F_3$):** $F_3 = 0.5\ln(1+n_{\text{orders}}) + 0.5\ln(1+\text{item\_lines})$

### Quantitative Ablation Results:

| Metric | Variant A: Literal ($F_1$) | Variant B: Current $F^*$ ($F_2$) | Variant C: Engagement ($F_3$) |
|---|:---:|:---:|:---:|
| **Unique Raw Values** | 9 | **76** | 46 |
| **Band 1 Share ($F1$)** | 97.00% (90,556) | **97.00% (90,553)** | **99.76% (93,129)** |
| **Band 2 Share ($F2$)** | 2.95% (2,754) | 2.74% (2,557) | 0.16% (149) |
| **Bands 3–5 Share** | 0.05% (47) | **0.26% (247)** | 0.08% (79) |
| **Shannon Entropy** | 0.1376 nats | **0.1457 nats** | **0.0196 nats** ($-86.6\%$) |
| **Entropy % of $\ln(5)$** | 8.55% | **9.05%** | **1.22%** |
| **Within-F1 Raw Std** | 0.0000 | 0.0082 | **0.1072** |
| **Closed Concepts** | 1,368 | 1,369 | **2,610** ($+90.7\%$) |
| **Kneedle Pruned Concepts** | 153 | 153 | **203** |
| **Natural $k$ Concepts** | 12 | 12 | 11 (lost $R1$) |
| **FCA Natural-$k$ Silhouette** | **0.5987** | 0.1656 | **-0.0694** |
| **FCA $k=5$ Silhouette** | **0.6089** | 0.2467 | **-0.0720** |
| **K-Means $k=5$ Silhouette** | 0.3888 | 0.3867 | 0.3777 |

### Ablation Findings & Verdict:
1. **$F_3$ Collapses Discrete Entropy:** Removing the discrete $+0.75$ repeat jump in $F_3$ pulls small-basket repeat buyers into band 1, causing **99.76% of customers to collapse into $F1$**, collapsing Shannon entropy to $0.0196$ nats.
2. **$F_3$ Causes Negative Silhouette Scores:** Downstream clustering under $F_3$ produces negative Silhouette values ($-0.0694$ at natural $k$; $-0.0720$ at $k=5$), signifying severe assignment distortion.
3. **Verdict:** $F_3$ is rejected. The current $F^*$ ($F_2$) is retained as the project's entropy-maximizing heuristic with explicit disclosure that it does not resolve physical frequency sparsity.

---

## 4. Problem 2 Resolution: Clustering Metric Mismatch & FCM Benchmark

### Investigation Summary:
To explain why hard-assigned FCA produces lower Silhouette (0.1656) and higher Davies–Bouldin (5.9467) than K-means (Silhouette 0.3867, DB 0.8391), an in-depth geometric cluster decomposition was conducted (`scripts/investigate_problem2_clustering.py`):
1. **Overlapping Boundary Misclassification:** At $\alpha=0.5$, 99.86% of customers qualify for multiple concepts. Forcing these customers into a single hard cluster by mode assignment assigns boundary customers with near-identical memberships to arbitrary clusters.
2. **Per-Cluster Silhouette Decomposition:** In the 12-cluster hard FCA assignment, **8 of the 12 clusters exhibit negative mean silhouette scores** (Cluster 0: -0.174, Cluster 2: -0.146, Cluster 4: -0.125, Cluster 5: -0.217, Cluster 7: -0.155, Cluster 8: -0.188, Cluster 9: -0.222, Cluster 10: -0.168). Only the dominant $F1$ cluster achieves positive silhouette (0.285) due to its high density.
3. **Canonical FCM as the Appropriate Benchmark:** Canonical Fuzzy C-Means ($m=2.0$) natively optimizes a fuzzy partition objective. At $k=5$, FCM achieves Silhouette $0.3847$, DB $0.8445$, and FPC $0.5005$. At $k=12$, FCM achieves Silhouette $0.3222$, DB $1.1134$, and FPC $0.3758$.
4. **Resolution:** Geometric partition metrics assess hyperspherical cluster compactness, which is mathematically misaligned with partial-order concept containment. Geometric metrics are retained as conversion diagnostics, while canonical FCM serves as the principled fuzzy benchmark.

---

## 5. Problem 3 Resolution: Satisfaction Temporal Leakage Audit & Elimination

### Leakage Audit Findings (`scripts/audit_olist_temporal_leakage.py`):
Evaluating temporal holdout models using features aggregated across all customer orders inadvertently introduced future review information into the observation period:
- **Affected Customers:** 2,341 observation customers ($10.81\%$) had reviews answered after the cutoff date ($T_{\text{cutoff}} = 2017\text{-}08\text{-}31$).
- **Numerical Score Shifts:** 1,003 customers ($4.63\%$) had their numeric Satisfaction score shifted by post-cutoff reviews (mean absolute shift: 1.645; max shift: 4.0).
- **Repurchaser Contamination:** **253 of the 554 future repurchasers (45.67%) had their observation features altered by post-cutoff reviews**, leaking future engagement into features and producing an artificially inflated test AUC of ~0.667.
- **Action:** The leaked AUC ~0.667 result was **permanently retracted**.

### Corrected Leakage-Free Predictive Results:

#### Fixed-Split Holdout (70/30, Seed 42, B=1,000 Paired Bootstrap)

| Model Representation | Features | Test AUC | Holdout Spend $R^2$ | Holdout Invoices $R^2$ | Delta AUC vs. Fuzzy [95% CI] |
|---|---:|---:|---:|---:|:---:|
| **Raw RFMS Baseline** | 4 | **0.5572** | **0.0009** | **0.0010** | +0.0024 [-0.0341, +0.0379] |
| **Crisp RFMS-FCA** | 44 | 0.5000 | -0.0003 | -0.0003 | -0.0548 [-0.0947, -0.0170] |
| **Fuzzy RFMS-FCA (Suppressed)** | 331 | 0.5548 | 0.0004 | 0.0005 | Reference |
| **FCM Soft (Matched $k=44$)** | 44 | 0.5426 | 0.0002 | 0.0003 | -0.0122 [-0.0493, +0.0236] |

#### Multi-Split Validation Across 10 Random Splits (Seeds 1000–1009)

| Dataset | Model Representation | Features | Mean AUC ± SD | Mean Spend $R^2$ ± SD | Mean Invoice $R^2$ ± SD | Wins vs Ref |
|---|---|---:|---:|---:|---:|:---:|
| **Retail II** | Raw RFM Baseline | 3.0 | 0.7770 ± 0.0117 | 0.2179 ± 0.0103 | 0.3420 ± 0.0214 | 0 / 10 |
| **Retail II** | Crisp RFM-FCA | 29.6 | 0.7768 ± 0.0121 | 0.3398 ± 0.0194 | 0.4418 ± 0.0145 | 0 / 10 |
| **Retail II** | **Fuzzy RFM-FCA (Suppressed)** | 96.7 | **0.7857 ± 0.0121** | **0.3559 ± 0.0195** | **0.4750 ± 0.0138** | **Ref (10/10)** |
| **Olist** | Raw RFMS Baseline | 4.0 | 0.5587 ± 0.0107 | 0.0007 ± 0.0006 | 0.0009 ± 0.0006 | 4 / 10 |
| **Olist** | Crisp RFMS-FCA | 44.7 | 0.5353 ± 0.0153 | 0.0005 ± 0.0005 | 0.0005 ± 0.0005 | 0 / 10 |
| **Olist** | **Fuzzy RFMS-FCA (Suppressed)** | 339.7 | **0.5612 ± 0.0125** | **0.0010 ± 0.0005** | **0.0011 ± 0.0006** | **Ref** |
| **Olist** | FCM Soft (Matched $k$) | 44.7 | 0.5517 ± 0.0115 | 0.0005 ± 0.0010 | 0.0004 ± 0.0012 | 1 / 10 |

#### Cross-Domain Takeaway:
- On repeat-buyer data (Retail II), fuzzy concept features provide statistically significant and parsimonious predictive gains (+0.009 AUC, +0.016 Spend $R^2$, +0.033 Invoices $R^2$; $p < 0.001$).
- On sparse marketplace data (Olist), removing review leakage reveals that predictive performance remains essentially at baseline (Fuzzy AUC 0.5612 vs Raw 0.5587), demonstrating that representation engineering cannot override physical frequency sparsity.

---

## 6. Problem 4 Resolution: Base-Paper Replication & Contribution Audit

### Replication Findings (`scripts/replicate_base_paper.py`):
1. **Customer Cohort Verification:** Successfully reproduced the exact customer cohort ($N = 5,878$) from Online Retail II (Table 3 of Rungruang et al., 2024).
2. **Concept Intent Recovery:** Under the base paper's support criterion ($>0.04$), our replication recovered **all 31 published frequent concept intents** (Table 6/7).
3. **Clustering Reproduction:** Replicated K-means and Ward hierarchical clustering over $k \in [2, 10]$ on 5,633 outlier-removed customers, confirming Silhouette values in the 0.33–0.38 range and DB values in the 0.99–1.07 range (matching Figures 9 & 10).
4. **Discrepancy Documented:** Exact concept customer counts differed modestly due to an **unstated tie-breaking procedure** in the base paper for customers with identical frequency ($27.61\%$ of customers have $F=1$).
5. **Lineage Alignment:** Confirmed that Rungruang et al. explicitly proposed non-binary / fuzzy FCA and variable expansion in their Section 6 future work. Our work serves as the direct execution and rigorous evaluation of their proposed agenda.

---

## 7. Complete Repository Structure & Results Inventory

```
rfms_fca_project/
├── data/
│   ├── olist_raw/                               # 9 raw Olist CSVs
│   ├── raw_retail2/                             # 2 Online Retail II CSVs
│   └── processed/
│       ├── olist_rfms_features.csv              # Extracted RFMS features (93,357 rows)
│       ├── olist_rfms_with_hard_clusters.csv    # Hard-assigned cluster labels
│       ├── soft_membership_top_level.csv        # Top-level alpha-cut soft memberships
│       └── retail2_rfm_features.csv             # Cleaned Retail II RFM features
├── docs/
│   ├── PROJECT_DOCUMENT.md                      # Comprehensive project documentation
│   ├── RESULTS_SUMMARY.md                       # Complete numerical results and audits (this file)
│   ├── results_section.md                       # Consolidated results section
│   ├── methodology_rfms_fca_olist.md            # Formal mathematical methodology
│   ├── fuzzy_improvements_retail2.md            # Redundancy suppression decision record
│   ├── base_paper_methodology_audit.md          # 25-point audit of Rungruang et al. (2024)
│   ├── contribution_audit.md                    # 12-item granular contribution audit
│   ├── claim_audit.md                           # Claim-by-claim verification and audit
│   ├── final_methodology_contribution_framing.md# Standardized contribution and paper framing
│   └── base_paper.pdf                           # Reference PDF (Rungruang et al., 2024)
├── results/
│   ├── ablation_frequency_variants/             # Problem 1: F1 vs F2 vs F3 ablation report & data
│   ├── base_paper_replication/                  # Problem 4: Base paper replication CSV
│   ├── problem2_clustering_investigation/       # Problem 2: Geometry mismatch & per-cluster sil.
│   ├── representation_comparison_olist/         # Representation comparison artifacts & figures
│   ├── temporal_holdout/                        # Problem 3: Satisfaction leakage audit & metrics
│   ├── multisplit_validation/                   # 10-split validation for Retail II and Olist
│   ├── olist_rfms_comparison/                   # Olist RFMS suppression and holdout outputs
│   ├── fair_comparison_retail2/                 # Retail II holdout bootstrap comparisons
│   ├── fcm_baseline_retail2/                    # FCM baseline comparisons on Retail II
│   ├── fuzzy_improvements_retail2/              # Retail II suppression experiments
│   ├── retail2_base_vs_fuzzy/                   # Base paper vs fuzzy matched comparison
│   ├── figures/                                 # Publication figures (fig1–fig4)
│   ├── fuzzy_concepts_raw.pkl                   # 1,369 raw closed concepts (Olist)
│   ├── pruned_fuzzy_concepts.pkl                # 153 Kneedle pruned concepts (Olist)
│   ├── hasse_edges.pkl                          # 304 Hasse diagram edges (Olist)
│   ├── retail2_fuzzy_concepts_raw.pkl           # 1,064 raw closed concepts (Retail II)
│   ├── retail2_pruned_fuzzy_concepts.pkl        # 115 Kneedle pruned concepts (Retail II)
│   ├── benchmark_comparison.csv                 # Core clustering benchmark table
│   └── cross_domain_comparison.csv              # Summary cross-domain comparison table
└── scripts/
    ├── step1_rfms_prep.py                       # Core Step 1: RFMS preparation
    ├── step2_fuzzy_fca.py                       # Core Step 2: Uncapped fuzzy FCA
    ├── step3_stability_pruning.py               # Core Step 3: Stability & Kneedle pruning
    ├── step4_alpha_cut_clusters.py              # Core Step 4: Alpha-cut hard assignments
    ├── step5_benchmark.py                       # Core Step 5: Clustering benchmarks
    ├── step7_cross_domain_retail2.py            # Core Step 7: Online Retail II replication
    ├── step8_kneedle_sensitivity.py             # Core Step 8: Kneedle sensitivity analysis
    ├── step9_f1_group_recovery.py               # Core Step 9: F1-group recovery test
    ├── step10_overlap_profiles.py               # Core Step 10: Multi-band overlap profiling
    ├── step11_fuzzy_validity_diagnostics.py     # Core Step 11: Entropy & persistence diagnostics
    ├── step12_retail2_parity.py                 # Core Step 12: Retail II parity evaluation
    ├── step13_figures.py                        # Core Step 13: Figure generation
    ├── step14_base_vs_fuzzy_retail2.py          # Step 14: Base vs fuzzy comparison
    ├── concept_redundancy.py                    # Reusable greedy extent-Jaccard module
    ├── fcm.py                                   # Reusable canonical FCM implementation
    ├── fcm_baseline_retail2.py                  # FCM benchmarking on Retail II
    ├── fuzzy_improvements_retail2.py            # Redundancy suppression experiments
    ├── olist_rfms_comparison.py                 # Olist RFMS holdout & suppression
    ├── fair_comparison_retail2.py               # Fair predictive comparison on Retail II
    ├── multisplit_validation.py                 # 10-split cross-dataset validation
    ├── ablation_frequency_variants.py           # Problem 1: Controlled frequency ablation
    ├── investigate_problem2_clustering.py       # Problem 2: Clustering mismatch diagnosis
    ├── audit_olist_temporal_leakage.py          # Problem 3: Satisfaction leakage audit
    ├── representation_comparison_olist.py       # Olist representation comparison
    └── replicate_base_paper.py                  # Problem 4: Base paper replication
```

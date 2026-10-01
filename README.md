# Fuzzy Formal Concept Analysis (FCA) for Customer Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Paper](https://img.shields.io/badge/Base_Paper-Rungruang_et_al._(2024)-orange.svg)](docs/base_paper_methodology_audit.md)
[![Status: Audited & Frozen (Methodology v4)](https://img.shields.io/badge/Status-Audited_&_Frozen_(Methodology_v4)-success.svg)](docs/RESULTS_SUMMARY.md)

An end-to-end framework applying **Recency, Frequency, and Monetary (RFM) feature engineering**, **centroid-based piecewise-linear fuzzy Formal Concept Analysis (FCA)**, **data-driven concept reduction**, **greedy extent-Jaccard redundancy suppression**, and **strictly leakage-free temporal predictive validation** to customer transaction data.

Under the locked **Methodology v4**, the final research study is evaluated across two repeat-transaction domains:
1. **Primary Validation Domain:** **Dunnhumby "The Complete Journey"** ($N = 2,499$ households; 276,484 shopping baskets; 2,595,732 transaction records; 711 days of continuous household grocery purchasing).
2. **Independent Second Domain:** **UK Online Retail II** ($N = 5,878$ clean customers; 1,067,371 transaction records; giftware retail).

> **Scope Note:** The final research study is strictly **RFM only** ($R, F, M$). Exploratory investigations on marketplace frequency sparsity ($F^*$), customer satisfaction ($S$), and the Olist marketplace are preserved in the repository as historical artifacts under `scripts/` and `results/`, but are formally out-of-scope for the final paper.

---

## Table of Contents
- [1. Research Lineage & Executive Summary](#1-research-lineage--executive-summary)
- [2. Methodological Architecture (Methodology v4)](#2-methodological-architecture-methodology-v4)
- [3. Primary Validation: Dunnhumby "The Complete Journey"](#3-primary-validation-dunnhumby-the-complete-journey)
- [4. Independent Validation: Online Retail II](#4-independent-validation-online-retail-ii)
- [5. Cross-Domain Comparative Synthesis](#5-cross-domain-comparative-synthesis)
- [6. Audited Methodological Resolutions](#6-audited-methodological-resolutions)
- [7. Historical Exploratory Studies (Out of Scope)](#7-historical-exploratory-studies-out-of-scope)
- [8. Repository Structure](#8-repository-structure)
- [9. Environment Setup & Execution](#9-environment-setup--execution)
- [10. Documentation Index](#10-documentation-index)
- [11. Citation & Attribution](#11-citation--attribution)

---

## 1. Research Lineage & Executive Summary

This research builds directly upon the foundational customer segmentation framework of:
> **Chongkolnee Rungruang, Pakwan Riyapan, Arthit Intarasit, Khanchit Chuarkham, and Jirapond Muangprathub (2024)**  
> *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*,  
> *Expert Systems with Applications*, 237, 121449.

In their concluding section, Rungruang et al. explicitly highlighted key limitations of crisp binary contexts:
> *"we will modify and improve this model by representing RFM values in a non-binary formal context in future studies."*

This repository directly executes that agenda while addressing foundational methodological gaps:
1. **Continuous Fuzzy Memberships:** Fuzzy membership reduces the hard boundary discontinuities introduced by crisp quintile discretization, while explicitly acknowledging structural choices such as five behavioral bands, centroid construction, and $\mathcal{L}$-fuzzy threshold configuration.
2. **Data-Driven Concept Reduction:** Closed fuzzy concepts are first generated subject to the minimum support criterion (0.04). The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on the observed support and stability-proxy distributions, complementing the candidate generation threshold.
3. **Redundancy Suppression:** The greedy extent-Jaccard procedure mitigates concept proliferation by suppressing near-duplicate concepts under the specified $J_{\max} = 0.80$ criterion (compressing concept counts 4×–5×), while downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
4. **Principled Clustering Evaluation:** Benchmarking against canonical Fuzzy C-Means (FCM, $m=2.0$) and evaluating hard partitions via Top-$k$ Membership Hardening in the standardized raw RFM space, separating geometric spatial diagnostics from lattice concept closure.
5. **Leakage-Free Temporal Prediction:** Establishing strict temporal cutoffs ($T_{\text{obs}} \to T_{\text{holdout}}$) where all transformations, cutoffs, and concepts are fit strictly on training folds.
6. **Threshold Sensitivity & Robustness Analysis:** Evaluating the sensitivity of the fuzzy representation across conservative/permissive $(0.2, 0.5, 0.8)$, baseline $(0.3, 0.5, 0.7)$, and tight $(0.4, 0.5, 0.6)$ $\mathcal{L}$-fuzzy threshold tuples on Dunnhumby, confirming that the predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Concept-space size changes substantially, while downstream predictive metrics vary comparatively modestly.

---

## 2. Methodological Architecture (Methodology v4)

```
[Raw Customer Transaction Data]
       │
       ▼
Phase 1: Feature Engineering & Preprocessing
 ├── Persistent customer key aggregation (household_key / CustomerID)
 ├── Recency (R: days since last purchase), Monetary (M: sum of transaction sales value)
 ├── Frequency (F: distinct basket / invoice count; strictly unweighted, no F*)
 └── Training-only dense-rank fractional scoring into quintile bands (1 to 5)
       │
       ▼
Phase 2: Centroid-Based Fuzzy Formal Context
 ├── Medians of score bands serve as band centroids c_1 < c_2 < c_3 < c_4 < c_5
 ├── Continuous piecewise-linear membership functions with outer shoulder saturation
 ├── L-fuzzy scaling at baseline cuts {0.3, 0.5, 0.7} (tested vs {0.2, 0.5, 0.8} & {0.4, 0.5, 0.6}) -> K_L
 └── Uncapped FP-growth closed frequent itemset mining (max_len=None, min_support=0.04)
       │
       ▼
Phase 3: Concept Reduction & Redundancy Suppression
 ├── Kneedle-inspired normalized max-chord-distance secondary pruning on support & stability proxy
 ├── Greedy extent-Jaccard redundancy suppression (J_max = 0.80, alpha-cut core mu >= 0.5)
 └── Continuous Godel minimum t-norm concept membership matrix computation
       │
       ▼
Phase 4: Multidimensional Benchmarking & Predictive Holdouts
 ├── Hard Clustering: Top-k Membership Hardening vs K-Means & Ward on standardized RFM
 ├── Fuzzy Clustering: Canonical Fuzzy C-Means (FCM, m=2.0) with FPC and Xie-Beni
 └── Downstream Predictive Holdout: Repurchase classification (AUC, Brier) &
     Spend / Invoice volume regression (R^2, MAE, Spearman rho) across 10 temporal splits
```

---

## 3. Primary Validation: Dunnhumby "The Complete Journey"

### 3.1 Dataset & RFM Characteristics
- **Population:** 2,499 clean households with 276,484 baskets across 711 continuous days.
- **Repeat Rate:** **99.68%** (2,497 of 2,499 households made $\ge 2$ baskets; mean $F = 110.6$ baskets).
- **Temporal Split:** Observation window = Days 1–620 ($N=2,499$); Future holdout = Days 621–711 (91 days).
- **Outcomes:** Future repurchase rate = $98.08\%$; future spend mean = \$482.05; future baskets mean = $15.05$.
- **Classification Ceiling Effect:** Because 98.08% of households repurchased during the 91-day holdout, repurchase classification exhibits a strong class-imbalance/ceiling effect. The fuzzy-versus-crisp AUC difference is therefore small and spans zero under the fixed-split bootstrap comparison. The clearer predictive gains occur in future monetary expenditure and transaction activity.

### 3.2 Concept Lattice & Pruning
- **Raw Closed Concepts:** 502 fuzzy concepts (min support = 0.04, `max_len = None`).
- **Kneedle-Inspired Secondary Pruning:** Closed fuzzy concepts are first generated subject to the 0.04 minimum support criterion. The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on observed distributions, yielding support threshold $= 0.2129$ and stability-proxy threshold $= 0.4285$.
- **Redundancy Suppression ($J_{\max} = 0.80$):** Retains **123 concepts** (379 redundant concepts removed; 4.1× compression). After suppression, downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.

### 3.3 10-Split Temporal Cross-Validation Performance (Seeds 1000–1009)

| Representation Arm | Features ($D$) | Repurchase AUC | Spend $R^2$ (log1p) | Invoices $R^2$ (log1p) |
|---|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | 40 | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | **114** | **$0.8605 \pm 0.0292$** | **$0.5028 \pm 0.0376$** | **$0.6045 \pm 0.0330$** |
| **FCM Soft (matched $k$)** | 40 | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Evaluation & Statistical Interpretation:*
- **10-Split Temporal CV:** Demonstrates consistent performance across temporal windows (mean ± SD, 9/10 split wins for Spend $R^2$, 10/10 split wins for Invoice $R^2$). Paired split win-rates illustrate split consistency and are not a formal significance test on their own.
- **Fixed-Split Paired Bootstrap ($B=1,000$):** Yields empirical $p < 0.001$ for Spend $R^2$ ($\Delta = +0.0267$, 95% CI $[0.0125, 0.0416]$) and Invoice $R^2$ ($\Delta = +0.0531$, 95% CI $[0.0377, 0.0684]$), whereas Repurchase AUC difference spans zero ($\Delta = -0.0065$, 95% CI $[-0.0202, 0.0063]$, $p = 0.330$) due to the repurchase ceiling effect.

### 3.4 Geometric Hard Clustering Benchmark (Standardized Raw RFM Space)

| Method | Hardening Procedure | $k$ | Silhouette | Davies-Bouldin |
|---|---|:---:|:---:|:---:|
| **K-Means** | Partition (k-means) | 4 | 0.4663 | 0.7944 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 4 | 0.3462 | 0.8421 |
| **FCM (Canonical)** | Argmax Soft Membership | 4 | 0.4204 | 0.7998 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 4 | -0.0851 | 2.2278 |
| **K-Means** | Partition (k-means) | 5 | 0.4747 | 0.7300 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 5 | 0.3571 | 0.8441 |
| **FCM (Canonical)** | Argmax Soft Membership | 5 | 0.3227 | 0.9233 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 5 | -0.2099 | 2.5513 |

*Clustering Interpretation:* The negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties. Silhouette and Davies–Bouldin act as geometric diagnostics rather than proofs of clustering superiority; fuzzy FCA optimizes attribute-closure intent coverage rather than Euclidean spherical variance.

### 3.5 Fuzzy Membership Threshold Sensitivity Analysis

To verify that downstream predictive performance is not dependent on the baseline threshold configuration ($\mathcal{L} = \{0.3, 0.5, 0.7\}$), a dedicated one-factor sensitivity analysis was evaluated across three threshold configurations on Dunnhumby under the locked evaluation protocol (observation days 1–620, holdout days 621–711, 10 temporal splits with seeds 1000–1009, 70/30 stratified train/test split, support cutoff 0.04, $J_{\max} = 0.80$, $\mu_{\text{cut}} = 0.5$, train-only fitting).

| Configuration | Full-Cohort Raw Concepts | Full-Cohort Suppressed ($k$) | Compression Ratio | Repurchase AUC (Mean ± SD) | $\Delta$ AUC vs Crisp | Spend $R^2$ (Mean ± SD) | $\Delta$ Spend $R^2$ vs Crisp | Invoice $R^2$ (Mean ± SD) | $\Delta$ Invoice $R^2$ vs Crisp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Crisp Baseline** | 38 | 38 | 1.00× | $0.8541 \pm 0.0242$ | *0.0000* | $0.4800 \pm 0.0351$ | *0.0000* | $0.5548 \pm 0.0344$ | *0.0000* |
| **(0.2, 0.5, 0.8)** *(Permissive)* | 652 | 265 | 2.46× (59.4%) | $0.8611 \pm 0.0299$ | **+0.0070** (6/10) | $0.5037 \pm 0.0387$ | **+0.0237** (9/10) | $0.6071 \pm 0.0313$ | **+0.0523** (10/10) |
| **(0.3, 0.5, 0.7)** *(Baseline)* | 502 | 123 | 4.08× (75.5%) | $0.8605 \pm 0.0292$ | **+0.0064** (7/10) | $0.5028 \pm 0.0376$ | **+0.0228** (9/10) | $0.6045 \pm 0.0330$ | **+0.0497** (10/10) |
| **(0.4, 0.5, 0.6)** *(Tight)* | 420 | 38 | 11.05× (91.0%) | $0.8673 \pm 0.0243$ | **+0.0132** (8/10) | $0.5090 \pm 0.0339$ | **+0.0290** (10/10) | $0.6079 \pm 0.0306$ | **+0.0531** (10/10) |

*Paired Split Wins across 10 Splits:*
- Repurchase AUC: 6/10 for (0.2, 0.5, 0.8); 7/10 for (0.3, 0.5, 0.7); 8/10 for (0.4, 0.5, 0.6).
- Future Spend $R^2$: 9/10 for (0.2, 0.5, 0.8); 9/10 for (0.3, 0.5, 0.7); 10/10 for (0.4, 0.5, 0.6).
- Future Invoice $R^2$: 10/10 for all three configurations.

*Core Interpretation:*
> The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space, downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations.

- **Concept-space structure is sensitive to threshold choice:** Relaxing cuts expands candidate concepts (652 raw, 265 suppressed), while tightening contracts them (420 raw, 38 suppressed).
- **Predictive utility is comparatively consistent:** Across all configurations, downstream spend $R^2$ ($0.5028$–$0.5090$) and invoice $R^2$ ($0.6045$–$0.6079$) remain consistently above the crisp baseline.
- **Redundancy suppression is consistently effective:** Substantially compresses concept-space size in all configurations ($59.4\%$ to $91.0\%$ reduction).
- **Non-tuning assurance:** Demonstrates that the fuzzy extension is not dependent on a single threshold configuration. Thresholds were not optimized post-hoc based on test set results.
- **Visual Diagnostic & Full Report:** See [`results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png`](results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png) and [`results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md`](results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md).

---

## 4. Independent Validation: Online Retail II

### 4.1 Base Paper Replication
- **Cohort:** 5,878 clean customers from the UK Online Retail II transaction dataset.
- **Intent Recovery:** Our reconstruction recovered all 31 published frequent concept intents reported by Rungruang et al. However, exact customer counts matched for only 3 of the 31 concepts. The remaining discrepancies are consistent with the paper's unspecified tie-breaking procedure for customers sharing identical frequency values, particularly the 1,623 customers with $F = 1$. The omitted tie-breaking rule prevents uniquely determining the original allocation.

### 4.2 Concept Reduction & Predictive Holdout
- **Uncapped Mining:** 1,064 closed concepts discovered (`max_len = None`).
- **Redundancy Suppression ($J_{\max} = 0.80$):** Reduced 445 candidate concepts to **95 concepts** (4.7× compression; pairwise extent Jaccard $\ge 0.80$ dropped from $3.3\%$ to $0.0\%$). After suppression, downstream predictive performance remains comparable to the unsuppressed representation under the evaluated protocol.
- **Predictive Performance (10 Splits):**
  - Raw RFM: AUC $0.7770 \pm 0.0117$, Spend $R^2$ $0.3476$, Invoice $R^2$ $0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** (paired t-test $p = 2.3 \times 10^{-4}$ vs crisp), Spend $R^2$ **$0.3664$**, Invoice $R^2$ **$0.4952$**.

---

## 5. Cross-Domain Comparative Synthesis

| Dimension | Primary Domain: Dunnhumby | Second Domain: Online Retail II |
|---|---|---|
| **Industry / Setting** | Household Grocery Supermarket | Online Giftware Retail |
| **Customer Population** | 2,499 households | 5,878 customers |
| **Repeat-Buyer Rate** | **99.68%** | **72.39%** |
| **Median Purchases ($F$)** | 79.0 baskets | 3.0 orders |
| **Feature Representation** | Strict RFM (15 fuzzy attributes) | Strict RFM (15 fuzzy attributes) |
| **Raw Closed Concepts** | 502 | 1,064 |
| **Suppressed Concepts** | 123 (from 502) | 95 (from 445) |
| **Predictive Lift (Spend $R^2$)** | **+0.1364** vs raw baseline | **+0.0188** vs raw baseline |
| **Predictive Lift (Invoice $R^2$)** | **+0.1530** vs raw baseline | **+0.0310** vs raw baseline |
| **Threshold Sensitivity** | Confirmed consistent across (0.2, 0.5, 0.8), (0.3, 0.5, 0.7), (0.4, 0.5, 0.6) | Locked baseline (0.3, 0.5, 0.7) |

*Cross-Domain Interpretation:* The results provide evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.

---

## 6. Audited Methodological Resolutions

All pipeline components have been audited and verified:
1. **Redundancy-Suppression Matrix Math Fix:** Corrected boolean matrix multiplication bug in `scripts/concept_redundancy.py` by converting extent vectors to `np.int32`, ensuring genuine intersection counts.
2. **Harmonized Multi-Split Labeling:** Unified dynamic matched-$k$ FCM labeling across splits to `"FCM Soft (matched k)"`.
3. **Hard-Clustering Protocol Alignment:** Fixed evaluation space to standardized Raw RFM ($X_{\text{raw}}$, shape $N \times 3$) and standardized terminology to **Top-$k$ Membership Hardening** (distinct from an $\alpha$-cut).
4. **Exclusion of Inappropriate Metrics:** Scoped canonical FPC and Xie-Beni strictly to canonical FCM.
5. **No Synthetic Variables:** Strictly preserved $F = \text{distinct basket count}$ and $M = \sum \text{SALES\_VALUE}$ without post-hoc multipliers or synthetic metrics.

---

## 7. Historical Exploratory Studies (Out of Scope)

Early exploratory experiments in this repository examined the **Olist Brazilian E-Commerce marketplace** ($N = 93,357$), testing whether an expanded **RFMS context** (incorporating customer review Satisfaction $S$) and an entropy-optimized purchase-intensity index ($F^*$) could overcome extreme marketplace frequency sparsity (97% single-order buyers).

**Key Findings from Historical Studies:**
- **Physical Frequency Sparsity:** Under controlled ablation (`scripts/ablation_frequency_variants.py`), neither literal $F$, composite $F^*$, nor engagement formulations could overcome the 97% single-buyer ceiling.
- **Temporal Review Leakage:** An audit (`scripts/audit_olist_temporal_leakage.py`) revealed that aggregating full-period reviews leaked future satisfaction into observation features. Purging this leakage collapsed Olist predictive performance to baseline (AUC 0.5612 vs 0.5587).
- **Disposition:** These exploratory findings confirmed that representation engineering cannot override physical domain sparsity. As a result, the final research study shifted focus to repeat-transaction domains (Dunnhumby and Online Retail II) under locked RFM-only Methodology v4. All Olist scripts and artifacts are preserved in `scripts/` and `results/` for historical provenance.

---

## 8. Repository Structure

```
├── data/
│   ├── processed/
│   │   ├── dunnhumby_rfm_features.csv      # Processed Dunnhumby RFM features
│   │   └── retail2_rfm_features.csv        # Processed Online Retail II features
│   └── raw_retail2/                        # Online Retail II raw spreadsheets
├── docs/
│   ├── PROJECT_DOCUMENT.md                 # Definitive comprehensive research document
│   ├── RESULTS_SUMMARY.md                  # Consolidated empirical results summary
│   ├── final_methodology_contribution_framing.md  # Official paper contribution positioning
│   ├── claim_audit.md                      # Audit of empirical claims & replacement phrasing
│   ├── contribution_audit.md               # 25-dimension audit against base paper
│   └── base_paper_methodology_audit.md     # Rungruang et al. (2024) replication audit
├── results/
│   ├── dunnhumby_rfm_fca/                  # Primary validation results (Methodology v4)
│   │   ├── summary.md                      # Complete Dunnhumby summary report
│   │   ├── audit_report.md                 # Complete audit report (Status: PASS)
│   │   ├── predictive_metrics.csv          # Fixed-split predictive metrics
│   │   ├── split_metrics.csv               # 10-split temporal cross-validation metrics
│   │   ├── clustering_metrics.csv          # Standardized RFM clustering benchmarks
│   │   ├── rfm_statistics.csv              # Full, observation, and holdout RFM statistics
│   │   └── bootstrap_ci.csv                # Paired bootstrap confidence intervals
│   ├── fuzzy_membership_sensitivity/       # Threshold sensitivity results (Methodology v4)
│   │   ├── sensitivity_analysis_report.md  # Detailed sensitivity report
│   │   ├── fig_predictive_sensitivity.png  # 3-panel predictive sensitivity visualization
│   │   ├── fuzzy_membership_sensitivity_summary.csv # Mean metrics and paired deltas
│   │   ├── fuzzy_membership_sensitivity_splits.csv  # 10-split granular evaluations
│   │   └── fuzzy_membership_sensitivity_full_cohort.csv # Concept counts & compression
│   ├── fair_comparison_retail2/            # Online Retail II validation results
│   ├── ablation_rfm_crossdomain/           # Cross-domain RFM comparison
│   └── ablation_fcm_hybrid_olist/          # Historical FCM-FCA hybrid study
└── scripts/
    ├── dunnhumby_rfm_fca_validation.py     # Primary validation pipeline (Methodology v4)
    ├── fair_comparison_retail2.py          # Online Retail II validation script
    ├── fuzzy_membership_sensitivity.py     # L-fuzzy threshold sensitivity evaluation
    ├── concept_redundancy.py               # Extent-Jaccard redundancy suppression module
    ├── fcm.py                              # Canonical Fuzzy C-Means implementation
    ├── replicate_base_paper.py             # Faithful replication of Rungruang et al. (2024)
    └── project_paths.py                    # Centralized project directory paths
```

---

## 9. Environment Setup & Execution

### Prerequisites
- Python 3.11+
- Virtual environment: `.venv`

```bash
# Clone the repository
git clone https://github.com/nandana-das/rfms-fuzzy-fca-segmentation.git
cd rfms-fuzzy-fca-segmentation

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing Dunnhumby Primary Validation (Methodology v4)
```bash
python scripts/dunnhumby_rfm_fca_validation.py
```

### Reproducing Online Retail II Validation
```bash
python scripts/fair_comparison_retail2.py
```

### Reproducing Fuzzy Membership Threshold Sensitivity Analysis
```bash
python scripts/fuzzy_membership_sensitivity.py
```

---

## 10. Documentation Index

- [PROJECT_DOCUMENT.md](docs/PROJECT_DOCUMENT.md): Definitive technical project overview.
- [RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md): Consolidated empirical tables across Dunnhumby and Retail II.
- [METHODOLOGY_V4.md](docs/METHODOLOGY_V4.md): Locked study design, mathematical definitions, and sensitivity protocol.
- [final_methodology_contribution_framing.md](docs/final_methodology_contribution_framing.md): Defensible contributions and paper framing.
- [results_section.md](docs/results_section.md): Empirical results section formatted for research manuscripts.
- [sensitivity_analysis_report.md](results/fuzzy_membership_sensitivity/sensitivity_analysis_report.md): Fuzzy membership threshold sensitivity analysis report.
- [audit_report.md](results/dunnhumby_rfm_fca/audit_report.md): Methodology v4 audit report (Status: PASS).
- [base_paper_methodology_audit.md](docs/base_paper_methodology_audit.md): Replication of Rungruang et al. (2024).

---

## 11. Citation & Attribution

If you use this framework or experimental setup, please cite:

```bibtex
@article{rungruang2024rfm,
  title={RFM model customer segmentation based on hierarchical approach using FCA},
  author={Rungruang, Chongkolnee and Riyapan, Pakwan and Intarasit, Arthit and Chuarkham, Khanchit and Muangprathub, Jirapond},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121449},
  year={2024},
  publisher={Elsevier}
}
```

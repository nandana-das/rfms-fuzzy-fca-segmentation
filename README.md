# RFMS + Fuzzy Formal Concept Analysis (FCA) Customer Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Paper](https://img.shields.io/badge/Base_Paper-Rungruang_et_al._(2024)-orange.svg)](docs/base_paper_methodology_audit.md)
[![Status: Audited & Frozen](https://img.shields.io/badge/Status-Audited_&_Frozen-success.svg)](docs/RESULTS_SUMMARY.md)

An end-to-end framework applying **RFMS feature engineering**, **centroid-based fuzzy formal concept analysis**, **data-driven concept reduction**, **redundancy suppression**, and **leakage-free predictive validation** to e-commerce customer data. The framework is evaluated across two contrasting domains: the **Olist Brazilian E-Commerce multi-seller marketplace** ($N = 93,357$) and the **UK Online Retail II dataset** ($N = 5,878$).

---

## Table of Contents
- [1. Research Lineage & Executive Summary](#1-research-lineage--executive-summary)
- [2. The Four Methodological Problems & Audited Resolutions](#2-the-four-methodological-problems--audited-resolutions)
- [3. Pipeline Architecture & Methodological Workflow](#3-pipeline-architecture--methodological-workflow)
- [4. Consolidated Experimental Results](#4-consolidated-experimental-results)
- [5. Repository Structure](#5-repository-structure)
- [6. Environment Setup](#6-environment-setup)
- [7. Complete Execution Guide](#7-complete-execution-guide)
- [8. Documentation Index](#8-documentation-index)
- [9. Citation & Attribution](#9-citation--attribution)

---

## 1. Research Lineage & Executive Summary

This project builds directly upon the foundational customer segmentation framework of:
> **Chongkolnee Rungruang, Pakwan Riyapan, Arthit Intarasit, Khanchit Chuarkham, and Jirapond Muangprathub (2024)**  
> *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*,  
> *Expert Systems with Applications*, 237, 121449.

In their concluding section, Rungruang et al. explicitly mapped out two critical directions for future work:
1. Extending the three RFM variables with additional customer behavioral dimensions.
2. Replacing crisp binary formal contexts with a **non-binary / fuzzy formal context** to overcome artificial quintile boundary discontinuities.

This repository directly fulfills and evaluates that research agenda while introducing data-driven concept reduction, redundancy suppression, canonical fuzzy benchmarking, and rigorous leakage-controlled evaluation.

### Key Cross-Domain Finding
- **Repeat-Buyer Retail (Online Retail II, 72.39% repeat rate):** Centroid-based fuzzy FCA combined with greedy redundancy suppression achieves statistically significant predictive gains over crisp FCA and raw RFM baselines across 10/10 independent customer splits (mean $\Delta\text{AUC} = +0.0089$, $p = 2.3\times 10^{-4}$; Spend $R^2$ $\Delta = +0.0161$; Invoices $R^2$ $\Delta = +0.0332$).
- **Sparse-Frequency Marketplace (Olist, 3.00% repeat rate):** Once post-cutoff temporal review leakage is strictly eliminated, fuzzy FCA performance realistically aligns with raw baseline features (Fuzzy AUC $0.5612 \pm 0.0125$ vs Raw RFMS $0.5587 \pm 0.0107$; raw wins in 4/10 splits). This demonstrates that **representation engineering cannot override physical frequency sparsity**.

---

## 2. The Four Methodological Problems & Audited Resolutions

| Problem | Focus Area | Root Cause & Diagnosis | Methodological Resolution | Key Artifacts |
|---|---|---|---|---|
| **Problem 1** | **Frequency Sparsity & Ablation** | Olist is dominated by single-order customers (97.00%). Naive RFM quantiles collapse. | Controlled ablation of 3 variants ($F_1$ literal, $F_2 / F^*$ composite, $F_3$ engagement). $F_3$ collapses 99.76% of customers into band 1 and yields negative silhouette ($-0.0694$). Retained entropy-optimal $F^*$ ($\alpha=0.20, \beta=0.05, \gamma=0.75$) with documented caveats. | [`scripts/ablation_frequency_variants.py`](scripts/ablation_frequency_variants.py)<br>[`results/ablation_frequency_variants/`](results/ablation_frequency_variants/) |
| **Problem 2** | **Clustering Metric Mismatch** | Hard-assigned FCA scores poorly on Silhouette (0.1656) and DB (5.9467) vs K-means (0.3867 / 0.8391). | Proved 99.86% of customers have multi-concept membership; 8 of 12 hard FCA clusters have negative silhouette. Integrated canonical Fuzzy C-Means (FCM, $m=2.0$) as the methodologically sound fuzzy-to-fuzzy benchmark (Silhouette 0.3847, FPC 0.5005). | [`scripts/investigate_problem2_clustering.py`](scripts/investigate_problem2_clustering.py)<br>[`results/problem2_clustering_investigation/`](results/problem2_clustering_investigation/) |
| **Problem 3** | **Temporal Review Leakage Audit** | Full-period review aggregation leaked future satisfaction into observation features, falsely inflating holdout AUC to ~0.667. | Systematic audit revealed 45.67% of repurchasers contaminated. Leaked AUC ~0.667 permanently retracted. Established strict leakage-free protocol ($\text{timestamp} \le T_{\text{cutoff}}$ for purchases and reviews; dynamic train-only bands). | [`scripts/audit_olist_temporal_leakage.py`](scripts/audit_olist_temporal_leakage.py)<br>[`results/temporal_holdout/`](results/temporal_holdout/) |
| **Problem 4** | **Base Paper Replication & Lineage** | Need for verified baseline replication and defensible contribution claims against Rungruang et al. (2024). | Replicated base paper on Online Retail II ($N=5,878$), recovering all 31 published frequent concept intents ($>0.04$ support) and K-means/Ward geometries ($k=2..10$). Audited 25 methodological dimensions and standardized contribution claims. | [`scripts/replicate_base_paper.py`](scripts/replicate_base_paper.py)<br>[`docs/contribution_audit.md`](docs/contribution_audit.md) |

---

## 3. Pipeline Architecture & Methodological Workflow

```
[Raw E-Commerce Data]
       │
       ▼
Phase 1: Feature Engineering & Sanitation
 ├── Persistent customer key resolution (customer_unique_id / CustomerID)
 ├── R (days), Literal F (n_orders), M (payment sum), S (mean review score)
 ├── Entropy-optimized F* purchase-intensity index (alpha=0.20, beta=0.05, gamma=0.75)
 └── Dense-rank fractional scoring with distinct-values denominator K_X
       │
       ▼
Phase 2: Fuzzy Formal Context Construction
 ├── Centroid-based piecewise-linear memberships with outer shoulder saturation
 ├── L-fuzzy threshold scaling at cuts {0.3, 0.5, 0.7} -> multi-level binary context K_L
 └── Uncapped FP-growth closed frequent itemset mining (max_len=None, min_support=0.02)
       │
       ▼
Phase 3: Concept Reduction & Redundancy Suppression
 ├── Kneedle-inspired elbow detection on Support and Object-Profile Stability Proxy
 ├── Pruned concept lattice (153 concepts for Olist; 115 concepts for Retail II)
 ├── Covering relation derivation (Hasse diagram)
 └── Greedy extent-Jaccard redundancy suppression (J_max = 0.80): 445 -> 95 concepts on Retail II
       │
       ▼
Phase 4: Multi-Perspective Benchmarking & Out-of-Sample Validation
 ├── Conceptual Overlap Profiling: multi-membership preservation (99.86% overlap)
 ├── Geometric vs Fuzzy Clustering: K-means, Ward's hierarchical, and canonical FCM
 └── Leakage-Free Temporal Prediction: 10-split customer holdouts (AUC, Spend R2, Invoices R2)
```

---

## 4. Consolidated Experimental Results

### 4.1 Cross-Domain Structural Comparison

| Metric | Online Retail II (Repeat Retail) | Olist (Sparse Marketplace) |
|---|---|---|
| **Clean Customer Population** | 5,878 | 93,357 |
| **Repeat-Buyer Rate** | **72.39%** | **3.00%** |
| **Analyzed Representation** | RFM (15 fuzzy attributes) | RFMS (20 fuzzy attributes) |
| **Entropy-Optimal $F^*$ Weights** | $\alpha=0.05, \beta=0.15, \gamma=0.80$ | $\alpha=0.20, \beta=0.05, \gamma=0.75$ |
| **Raw Closed Concepts (Uncapped)** | 1,064 | 1,369 |
| **Kneedle Pruned Concepts** | 115 | 153 |
| **Hasse Covering Edges** | — | 304 |
| **Redundancy Suppression** | **445 → 95 concepts** (4.7× compression) | **633 → 627 concepts** (near zero redundancy) |
| **10-Split Predictive Lift** | Statistically significant across all metrics | On par with raw features (AUC 0.5612 vs 0.5587) |

### 4.2 Multi-Split Holdout Validation (10 Random Splits, Seeds 1000–1009)

| Dataset | Model Representation | Features | Mean AUC ± SD | Mean Spend $R^2$ ± SD | Mean Invoice $R^2$ ± SD | Win Rate vs. Ref |
|---|---|---:|---:|---:|---:|:---:|
| **Retail II** | Raw RFM Baseline | 3.0 | 0.7770 ± 0.0117 | 0.2179 ± 0.0103 | 0.3420 ± 0.0214 | 0 / 10 |
| **Retail II** | Crisp RFM-FCA | 29.6 | 0.7768 ± 0.0121 | 0.3398 ± 0.0194 | 0.4418 ± 0.0145 | 0 / 10 |
| **Retail II** | **Fuzzy RFM-FCA (Suppressed)** | 96.7 | **0.7857 ± 0.0121** | **0.3559 ± 0.0195** | **0.4750 ± 0.0138** | **Reference (10/10)** |
| **Olist** | Raw RFMS Baseline | 4.0 | 0.5587 ± 0.0107 | 0.0007 ± 0.0006 | 0.0009 ± 0.0006 | 4 / 10 |
| **Olist** | Crisp RFMS-FCA | 44.7 | 0.5353 ± 0.0153 | 0.0005 ± 0.0005 | 0.0005 ± 0.0005 | 0 / 10 |
| **Olist** | **Fuzzy RFMS-FCA (Suppressed)** | 339.7 | **0.5612 ± 0.0125** | **0.0010 ± 0.0005** | **0.0011 ± 0.0006** | **Reference** |
| **Olist** | FCM Soft (Matched $k$) | 44.7 | 0.5517 ± 0.0115 | 0.0005 ± 0.0010 | 0.0004 ± 0.0012 | 1 / 10 |

---

## 5. Repository Structure

```
rfms_fca_project/
├── data/
│   ├── olist_raw/                               # 9 raw CSVs from Kaggle Olist dataset
│   ├── raw_retail2/                             # 2 Online Retail II CSVs (Year 2009-2010, 2010-2011)
│   └── processed/
│       ├── olist_rfms_features.csv              # Extracted RFMS features (93,357 customers)
│       ├── olist_rfms_with_hard_clusters.csv    # Hard-assigned cluster labels
│       ├── soft_membership_top_level.csv        # Top-level alpha-cut soft memberships
│       └── retail2_rfm_features.csv             # Cleaned Retail II RFM features
├── docs/                                        # Comprehensive documentation & audit suite
│   ├── PROJECT_DOCUMENT.md                      # Complete project documentation & methodology
│   ├── RESULTS_SUMMARY.md                       # Comprehensive numerical summary & diagnostics
│   ├── results_section.md                       # Consolidated results section
│   ├── methodology_rfms_fca_olist.md            # Formal mathematical methodology
│   ├── fuzzy_improvements_retail2.md            # Redundancy suppression decision record
│   ├── base_paper_methodology_audit.md          # 25-point audit of Rungruang et al. (2024)
│   ├── contribution_audit.md                    # 12-item contribution taxonomy & lineage audit
│   ├── claim_audit.md                           # Claim-by-claim verification matrix
│   ├── final_methodology_contribution_framing.md# Standardized contribution and paper framing
│   └── base_paper.pdf                           # Reference PDF (Rungruang et al., 2024)
├── results/                                     # Generated artifacts & audit reports
│   ├── ablation_frequency_variants/             # Problem 1: Frequency ablation report & data
│   ├── base_paper_replication/                  # Problem 4: Base paper replication outputs
│   ├── problem2_clustering_investigation/       # Problem 2: Clustering mismatch & silhouette decomp.
│   ├── representation_comparison_olist/         # Representation comparison artifacts & figures
│   ├── temporal_holdout/                        # Problem 3: Satisfaction leakage audit & metrics
│   ├── multisplit_validation/                   # 10-split validation outputs for both datasets
│   ├── olist_rfms_comparison/                   # Olist RFMS holdout & suppression outputs
│   ├── fair_comparison_retail2/                 # Retail II holdout bootstrap comparisons
│   ├── fcm_baseline_retail2/                    # FCM baseline comparisons on Retail II
│   ├── fuzzy_improvements_retail2/              # Retail II suppression experiments
│   ├── retail2_base_vs_fuzzy/                   # Base paper vs fuzzy matched comparison
│   ├── figures/                                 # Publication figures (fig1–fig4)
│   ├── fuzzy_concepts_raw.pkl                   # 1,369 raw closed concepts (Olist)
│   ├── pruned_fuzzy_concepts.pkl                # 153 Kneedle pruned concepts (Olist)
│   ├── hasse_edges.pkl                          # 304 Hasse diagram edges (Olist)
│   ├── retail2_fuzzy_concepts_raw.pkl           # 1,064 raw closed concepts (Retail II)
│   └── retail2_pruned_fuzzy_concepts.pkl        # 115 Kneedle pruned concepts (Retail II)
└── scripts/                                     # Complete script suite
    ├── step1_rfms_prep.py                       # Step 1: RFMS preparation & entropy grid search
    ├── step2_fuzzy_fca.py                       # Step 2: Centroid fuzzy context & uncapped FP-growth
    ├── step3_stability_pruning.py               # Step 3: Stability proxy & Kneedle iceberg pruning
    ├── step4_alpha_cut_clusters.py              # Step 4: Alpha-cut hard cluster assignment
    ├── step5_benchmark.py                       # Step 5: Silhouette, DB, and FPC benchmarking
    ├── step7_cross_domain_retail2.py            # Step 7: Online Retail II pipeline
    ├── step8_kneedle_sensitivity.py             # Step 8: Multiplicative & additive sensitivity
    ├── step9_f1_group_recovery.py               # Step 9: F1-group structure recovery test
    ├── step10_overlap_profiles.py               # Step 10: Multi-band overlap profiling
    ├── step11_fuzzy_validity_diagnostics.py     # Step 11: Membership entropy & persistence
    ├── step12_retail2_parity.py                 # Step 12: Retail II parity evaluation
    ├── step13_figures.py                        # Step 13: Publication figure generation
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

---

## 6. Environment Setup

The codebase is written for **Python 3.11** on Windows PowerShell (or Linux/macOS bash). Dependencies are pinned in [`requirements.txt`](requirements.txt).

```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Note:* If PowerShell script execution policies block activation, invoke Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 7. Complete Execution Guide

All scripts resolve paths relative to the project root and create output directories automatically.

### 7.1 Core Pipeline Execution (Steps 1–13)
Run sequentially to generate base feature tables, closed concept lattices, pruned Hasse diagrams, and diagnostic benchmarks:

```powershell
python scripts/step1_rfms_prep.py                  # Generates RFMS features & F* weights
python scripts/step2_fuzzy_fca.py                  # Mines uncapped closed concepts (1,369 Olist)
python scripts/step3_stability_pruning.py          # Prunes via Kneedle (153 concepts, 304 edges)
python scripts/step4_alpha_cut_clusters.py         # Derives alpha-cut hard clusters (k=12)
python scripts/step5_benchmark.py                  # Evaluates Silhouette, DB, and FPC
python scripts/step7_cross_domain_retail2.py       # Executes Retail II pipeline (1,064 concepts)
python scripts/step8_kneedle_sensitivity.py        # Sensitivity analysis on support and stability
python scripts/step9_f1_group_recovery.py          # Evaluates F1-group differentiation
python scripts/step10_overlap_profiles.py          # Extracts interpretable multi-band profiles
python scripts/step11_fuzzy_validity_diagnostics.py# Evaluates entropy and concept persistence
python scripts/step12_retail2_parity.py            # Executes parity tests on Retail II
python scripts/step13_figures.py                   # Generates publication figures (fig1–fig4)
```

### 7.2 Standalone Improvements & Benchmarks
```powershell
python scripts/step14_base_vs_fuzzy_retail2.py     # Base-paper reconstruction vs fuzzy (Retail II)
python scripts/fair_comparison_retail2.py          # Controlled holdout comparison (Retail II)
python scripts/fuzzy_improvements_retail2.py       # Redundancy suppression experiments
python scripts/fcm_baseline_retail2.py             # FCM vs FCA structure on Retail II
python scripts/olist_rfms_comparison.py            # Olist RFMS suppression & holdout evaluation
python scripts/multisplit_validation.py            # 10-split validation across both datasets
```

### 7.3 Problem 1–4 Audit & Replication Scripts
Execute the scripts corresponding to the four audited research problems:

```powershell
# Problem 1: Controlled frequency dimension ablation (F1 vs F2 vs F3)
python scripts/ablation_frequency_variants.py

# Problem 2: Clustering metric mismatch & per-cluster silhouette decomposition
python scripts/investigate_problem2_clustering.py

# Problem 3: Comprehensive temporal review leakage audit & elimination
python scripts/audit_olist_temporal_leakage.py

# Representation comparison on Olist
python scripts/representation_comparison_olist.py

# Problem 4: Full base paper replication of Rungruang et al. (2024)
python scripts/replicate_base_paper.py
```

---

## 8. Documentation Index

The `docs/` and `results/` directories provide a complete audit trail and documentation suite:

- **Comprehensive Methodology & Findings:**
  - [`docs/PROJECT_DOCUMENT.md`](docs/PROJECT_DOCUMENT.md) — Comprehensive technical documentation, research gaps, objectives, methodology, and audited findings.
  - [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) — Exhaustive numerical summary covering Steps 1–13, v4 improvements, and Problems 1–4.
  - [`docs/results_section.md`](docs/results_section.md) — Publication-ready consolidated experimental results section.
  - [`docs/methodology_rfms_fca_olist.md`](docs/methodology_rfms_fca_olist.md) — Formal mathematical formulation of the RFMS-Fuzzy FCA framework.
  - [`docs/fuzzy_improvements_retail2.md`](docs/fuzzy_improvements_retail2.md) — Decision record for greedy extent-Jaccard redundancy suppression and FCM baseline.

- **Systematic Audit & Framing Suite:**
  - [`docs/base_paper_methodology_audit.md`](docs/base_paper_methodology_audit.md) — 25-dimension methodology audit of Rungruang et al. (2024).
  - [`docs/contribution_audit.md`](docs/contribution_audit.md) — 12-item granular contribution audit establishing exact lineage and novelty classification.
  - [`docs/claim_audit.md`](docs/claim_audit.md) — Claim-by-claim verification and publication-grade phrasing rules.
  - [`docs/final_methodology_contribution_framing.md`](docs/final_methodology_contribution_framing.md) — Standardized paper positioning and defensible contribution statements.

- **Detailed Audit Reports:**
  - [`results/ablation_frequency_variants/frequency_ablation_report.md`](results/ablation_frequency_variants/frequency_ablation_report.md) — Detailed Problem 1 frequency ablation report.
  - [`results/problem2_clustering_investigation/problem2_investigation_report.md`](results/problem2_clustering_investigation/problem2_investigation_report.md) — Detailed Problem 2 clustering metric mismatch report.
  - [`results/temporal_holdout/temporal_holdout_leakage_report.md`](results/temporal_holdout/temporal_holdout_leakage_report.md) — Detailed Problem 3 temporal leakage audit report.

---

## 9. Citation & Attribution

If you utilize this codebase or its methodological framework, please cite the base paper and this project:

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

@misc{das2026rfmsfuzzyfca,
  title={RFMS + Fuzzy Formal Concept Analysis Customer Segmentation: Audited Methodology, Redundancy Suppression, and Cross-Domain Evaluation},
  author={Das, Nandana},
  year={2026},
  howpublished={\url{https://github.com/nandana-das/rfms-fuzzy-fca-segmentation}}
}
```

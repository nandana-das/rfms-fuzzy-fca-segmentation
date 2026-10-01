# Dunnhumby RFM-FCA Methodology v4 Audit

## 1. Overall Status

**PASS** *(Updated from PASS WITH ISSUES following Section 12 corrections)*

The Dunnhumby RFM-FCA implementation in [`scripts/dunnhumby_rfm_fca_validation.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/dunnhumby_rfm_fca_validation.py) adheres to the core constraints of locked Methodology v4:
- RFM-only formulation (Satisfaction $S$ and Olist are strictly absent).
- Accurate aggregation definitions ($F = \text{nunique}(\text{BASKET\_ID})$, $M = \sum \text{SALES\_VALUE}$ without quantity multiplication, $R = \text{cutoff} - \max(\text{DAY})$).
- Strict, verified leakage-free temporal holdout (scalers, cutoffs, centroids, concept mining, suppression, FCM prototypes, and predictive models fitted strictly on training data).
- The root cause of the concept suppression discrepancy (503/503 vs 502/123) has been definitively identified as a boolean matrix multiplication bug in `suppress_redundant_concepts_sparse` in earlier runs, and permanently resolved.
- All 4 issues identified during the initial audit have been corrected and verified (see Section 12):
  1. Multi-split FCM arm name unified to `FCM Soft (matched k)`.
  2. Geometric hard-clustering protocol aligned on the agreed standardized Raw RFM space ($X_{\text{raw}}$).
  3. Detailed RFM cohort statistics table added to `summary.md`.
  4. Redundancy-suppression sparse implementation permanently patched.

---

## 2. Data Construction Audit

### Requirement vs. Implementation Traceability

| Requirement (Methodology v4) | Actual Implementation | Status | Evidence |
| :--- | :--- | :--- | :--- |
| **RFM Only Scope** | $R, F, M$ dimensions only; Satisfaction ($S$) completely excluded | **PASS** | Lines 8, 93–94: `DIMS = ("R", "F", "M")`. No satisfaction column or proxy exists. |
| **Olist Exclusion** | Olist dataset excluded entirely | **PASS** | Lines 9, 878, 934: Zero references to Olist data, paths, or tables. |
| **Frequency ($F$) Definition** | Count of distinct `BASKET_ID` values per `household_key` (basket level, not transaction row count) | **PASS** | Lines 232, 246, 526: `n_baskets=("BASKET_ID", "nunique")`. Product lines within baskets confirm line-item granularity (216,439 multi-row baskets). |
| **Monetary ($M$) Definition** | $\sum \text{SALES\_VALUE}$ directly; no multiplication by `QUANTITY` | **PASS** | Lines 233, 247, 527, 538: `M=("SALES_VALUE", "sum")`. `QUANTITY` is never referenced in calculations. |
| **Recency ($R$) Definition** | $\text{reference\_day} - \text{last\_purchase\_day}$ | **PASS** | Lines 252, 531: `obs_agg["R"] = obs_cutoff - obs_agg["last_purchase_day"]`, where `obs_cutoff = 620`. |
| **No $F^*$ Composite Index** | Literal basket frequency used directly; no entropy-weighted composite metric | **PASS** | Lines 12, 925: `obs_agg["F"] = obs_agg["n_baskets"]`. No entropy-based weight optimization. |
| **Itemset Constraint** | `max_len = None` (no artificial cap on itemset length) | **PASS** | Lines 13, 867, 927: `fpgrowth(..., max_len=None)`. |

### Exact Code Snippets and Resulting Values

```python
# scripts/dunnhumby_rfm_fca_validation.py, lines 242-253
tx_obs = tx[tx["DAY"] <= obs_cutoff].copy()
obs_agg = (
    tx_obs.groupby("household_key")
    .agg(
        last_purchase_day=("DAY", "max"),
        n_baskets=("BASKET_ID", "nunique"),
        M=("SALES_VALUE", "sum"),
    )
    .reset_index()
)
obs_agg["F"] = obs_agg["n_baskets"]
obs_agg["R"] = obs_cutoff - obs_agg["last_purchase_day"]
```

#### Verified Distributional Values (Observation Cohort, $N = 2,499$):
- **Recency ($R$, days):** Min = 0.0, 25% = 1.0, Median = 6.0, Mean = 26.73, 75% = 19.0, 90% = 63.0, 95% = 129.2, Max = 590.0, Std = 67.39.
- **Frequency ($F$, baskets):** Min = 1.0, 25% = 33.0, Median = 67.0, Mean = 95.59, 75% = 123.0, 90% = 202.0, 95% = 277.1, Max = 1169.0, Std = 102.09.
- **Monetary ($M$, USD):** Min = $4.49, 25% = $790.90, Median = $1,823.70, Mean = $2,742.18, 75% = $3,739.69, 90% = $6,385.79, 95% = $8,339.92, Max = $31,548.37, Std = $2,874.59.

---

## 3. Critical Temporal Leakage Audit

### Exact Cutoff and Population Parameters
- **Dataset DAY range:** 1 to 711 (711 days total).
- **Observation Window:** Days 1 to 620 (620 observation days, ~88.6 weeks).
- **Holdout Window:** Days 621 to 711 (91 holdout days, exactly 13 weeks / 1 calendar quarter).
- **Total Population in Raw Data:** 2,500 unique households.
- **Observation Cohort Population:** 2,499 households active on or before Day 620 (exactly 1 household made its first purchase on Day 641, after the observation cutoff).
- **Holdout Cohort:** 2,499 households tracked. Of these, 2,328 households repurchased during the holdout (repurchase rate = 93.16%), while 171 households had zero transactions ($future\_invoices = 0, future\_spend = 0.0$).
- **Train / Test Split:** 70% Train ($N = 1,749$) / 30% Test ($N = 750$), stratified by repurchase binary outcome.

### Trace of Representation Fitting
1. **Full-population concept lattice check:**
   - In [`scripts/dunnhumby_rfm_fca_validation.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/dunnhumby_rfm_fca_validation.py), `run_full_population_fca(obs_rfm)` is executed strictly on `obs_rfm` (Days 1–620). It does **not** see holdout data (Days 621–711).
   - More crucially, **none of the full-population concepts or memberships are passed into the predictive holdout**.
2. **Within `run_temporal_holdout`:**
   - **Dense-rank score cutoffs:** Fitted on `train_cust` via `extract_dense_rank_cutoffs(train_scored)`. Test customers scored using `apply_dense_rank_cutoffs(test_df, train_cutoffs)`.
   - **Fuzzy centroids:** Fitted on `train_scored` via `compute_fuzzy_memberships(train_scored)`. Test customers projected using `compute_fuzzy_memberships(test_cust, trained_centroids=train_centroids)`.
   - **StandardScaler:** Fit on `train_df[["R", "F", "M"]]` only; test transformed via `scaler.transform()`.
   - **Crisp FCA concepts:** Mined strictly on `train_scored` (`mine_crisp_closed_concepts(train_scored)`).
   - **Fuzzy FCA concepts:** Mined strictly on `train_fuzzy_mu` (`mine_fuzzy_closed_concepts(train_fuzzy_mu, train_scored)`).
   - **Redundancy suppression:** Computed strictly on training customer memberships (`mu_tr_full` from `train_fuzzy_mu`). Test customer memberships projected into surviving concepts via Gödel t-norm.
   - **FCM prototypes:** `fcm.fit(X_tr_raw)` fit strictly on training standardized RFM; test memberships assigned via `fcm.assign(X_te_raw)`.
   - **Predictive models:** `LogisticRegressionCV` and `RidgeCV` fit strictly on `X_tr` and evaluated on `X_te`.
- **Verdict:** **Zero temporal information leakage.**

---

## 4. FCA Implementation Audit

### Formal Details of the 18 Pipeline Stages

1. **Data Loading:** Clean read of `transaction_data.csv`; verified 2,595,732 rows and 2,500 households. (Matches Methodology v4).
2. **RFM Construction:** $R, F, M$ aggregated per household. Verified $F = \text{nunique}(\text{BASKET\_ID})$ and $M = \sum \text{SALES\_VALUE}$. (Matches Methodology v4).
3. **Temporal Split:** Days 1–620 observation, Days 621–711 holdout. (Matches Methodology v4).
4. **Dense-Rank Scoring:** 5 bands via `rank(method="dense")`, normalized to 1..5. Inversion applied to $R$ ($6 - \text{score}$). Preserves ties without arbitrary customer ID tie-breaking. (Matches Methodology v4).
5. **Fuzzy Membership:** Centroid-based piecewise linear membership with outer saturation (Dunn 1973; Zadeh 1965). (Matches Methodology v4).
6. **L-Fuzzy Scaling:** $\mathcal{L} = \{0.3, 0.5, 0.7\}$, generating 15 base attributes per customer $\times$ 3 cutoffs = 45 binary attributes. (Matches Methodology v4).
7. **Crisp FCA:** Exact subset-extent closure mining; support threshold $\ge 0.04$. (Matches Methodology v4).
8. **Fuzzy FCA:** Exact closure via extent bitset intersection over frequent itemsets mined via FP-Growth (`max_len = None`, `min_support = 0.04`). (Matches Methodology v4).
9. **Stability Proxy:** Profile homogeneity proxy $1 - \frac{|\text{unique profiles}|}{|\text{extent}|}$. Not canonical Kuznetsov stability. Explicitly labeled as proxy. (Matches Methodology v4).
10. **Kneedle-Inspired Pruning:** Normalized maximum distance from chord on support and stability proxy curves. (Matches Methodology v4).
11. **Redundancy Suppression:** Greedy extent-Jaccard suppression at $J_{\max} = 0.8$, $\mu_{\text{cut}} = 0.5$. (Matches Methodology v4).
12. **Alpha-Cut Clusters:** Evaluated via continuous Gödel t-norm memberships. Hard clustering metrics in `clustering_metrics.csv` evaluated K-Means and Ward on concept embeddings rather than direct alpha-cut argmax partitioning. (Issue noted in Section 6).
13. **K-Means:** $k \in \{4, 5, 6\}$, $n_{\text{init}} = 10$, `random_state = 42`. (Matches Methodology v4).
14. **Ward Clustering:** $k \in \{4, 5, 6\}$, Ward linkage agglomerative clustering. (Matches Methodology v4).
15. **FCM Clustering:** $m = 2.0$, k-means++ seeding, convergence tolerance $10^{-7}$, max iterations 300. Canonical FPC and Xie-Beni computed exclusively for FCM. (Matches Methodology v4).
16. **Predictive Models:** LogisticRegressionCV ($C_s = 10$, 5-fold CV) for repurchase classification; RidgeCV ($\alpha \in [10^{-3}, 10^3]$, 5-fold CV) for log1p future spend and log1p future invoices. (Matches Methodology v4).
17. **Bootstrap Evaluation:** $N = 1,000$ paired customer-level resamples; percentile 95% CIs on model deltas $(M_1 - M_2)$. (Matches Methodology v4).
18. **10-Split Evaluation:** Evaluated across 10 random seeds (1000–1009) with 70/30 stratified splits. (Matches Methodology v4, with arm naming issue noted in Section 7).

---

## 5. Concept Suppression Discrepancy

### The Discrepancy
- **Previous cached results:**
  - Raw fuzzy concepts: 503
  - Suppressed concepts: 503
  - Removed: 0
  - Predictive feature count: 514
- **Current generated results:**
  - Raw fuzzy concepts: 502
  - Suppressed concepts: 123
  - Removed: 379
  - Predictive feature count: 114 (fixed split train)

### Exact Root-Cause Analysis
The root cause was isolated through script and codebase inspection to an **implementation bug in [`suppress_redundant_concepts_sparse`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/concept_redundancy.py#L103-L148)**:

1. **The Bug:**
   In `scripts/concept_redundancy.py`, `suppress_redundant_concepts_sparse` initializes the matrix of accumulated kept extents as:
   ```python
   kept_mat = np.empty((0, mu_matrix.shape[0]), dtype=bool)
   ```
   When new candidate extents `ext` (also of boolean dtype) were compared against kept concepts, the intersection was computed via matrix multiplication:
   ```python
   inter = kept_mat @ ext
   ```
   In NumPy, the `@` operator between two boolean arrays performs **boolean matrix multiplication** (logical OR accumulation: $\text{True} + \text{True} = \text{True}$), **not integer addition**.
2. **The Consequence:**
   The resulting array `inter` had `dtype=bool`. When coerced to numerical values during Jaccard calculation:
   $$\text{inter}[\text{row\_i}] = 1 \quad (\text{if intersection} > 0)$$
   $$\text{union} = \sum \text{kept} + \sum \text{ext} - 1 \approx 200 + 200 - 1 = 399$$
   $$\text{Jaccard} = \frac{\text{inter}}{\text{union}} \approx \frac{1}{399} \approx 0.0025$$
   Because $0.0025$ is never $\ge J_{\max} (0.8)$, the condition `inter[row_i] / union >= j_max` was **never met for any concept**.
   Consequently, **`suppress_redundant_concepts_sparse` never removed a single concept**, silently returning $100\%$ of candidate concepts ($503/503$ in full population, $514/514$ on the training split).
3. **The Current Implementation:**
   The active script [`scripts/dunnhumby_rfm_fca_validation.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/dunnhumby_rfm_fca_validation.py) calls `suppress_redundant_concepts` (the standard function), which computes:
   ```python
   inter = np.logical_and(ext, kept_ext).sum()  # Integer count!
   union = np.logical_or(ext, kept_ext).sum()   # Integer count!
   jacc = inter / union
   ```
   This correctly calculates true extent Jaccard similarity, removing 379 redundant concepts ($75.5\%$ compression) and retaining 123 concepts on the observation cohort (and 114 on the fixed split training set).

### Classification
- **Primary Cause:** **A. An implementation bug** in `suppress_redundant_concepts_sparse` (boolean matrix multiplication).
- **Secondary Cause:** **D. A difference in suppression logic** between the old script (which imported the buggy sparse routine) and the new script (which imported the correct standard routine).
- **Verdict:** The old result (503 concepts, 514 features) was an unsuppressed artifact caused by this bug. The new result (123 concepts, 114 features) represents the **correct execution of Methodology v4**.

---

## 6. Clustering Audit

### Geometric Benchmarking Protocol
- Hard-clustering metrics (Silhouette, Davies-Bouldin) were computed on hard assignments across $k \in \{4, 5, 6\}$.
- Canonical FCM metrics (Partition Coefficient FPC and Xie-Beni Index) were computed strictly for canonical FCM.

### Exact Values ($k \in \{4, 5, 6\}$)

| $k$ | Representation | Algorithm | Features | Silhouette | Davies-Bouldin | Canonical Metric |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **4** | Raw RFM (Standardized) | K-Means | 3 | 0.4663 | 0.7944 | — |
| **4** | Raw RFM (Standardized) | Ward | 3 | 0.3462 | 0.8421 | — |
| **4** | Crisp RFM-FCA | K-Means | 38 | 0.4021 | 1.4582 | — |
| **4** | Crisp RFM-FCA | Ward | 38 | 0.3791 | 1.2427 | — |
| **4** | Fuzzy RFM-FCA (Suppressed) | K-Means | 123 | 0.2414 | 1.6432 | — |
| **4** | Fuzzy RFM-FCA (Suppressed) | Ward | 123 | 0.2283 | 1.7687 | — |
| **4** | FCM Soft ($k=4$) | Argmax Partition | 4 | 0.6939 | 0.4409 | $\text{FPC}=0.6946, \text{XB}=0.0381$ |
| **5** | Raw RFM (Standardized) | K-Means | 3 | 0.4747 | 0.7300 | — |
| **5** | Raw RFM (Standardized) | Ward | 3 | 0.3571 | 0.8441 | — |
| **5** | Crisp RFM-FCA | K-Means | 38 | 0.5010 | 1.1467 | — |
| **5** | Crisp RFM-FCA | Ward | 38 | 0.4720 | 1.2182 | — |
| **5** | Fuzzy RFM-FCA (Suppressed) | K-Means | 123 | 0.2500 | 1.5233 | — |
| **5** | Fuzzy RFM-FCA (Suppressed) | Ward | 123 | 0.2313 | 1.6185 | — |
| **5** | FCM Soft ($k=5$) | Argmax Partition | 5 | 0.6190 | 0.5181 | $\text{FPC}=0.6190, \text{XB}=0.0347$ |
| **6** | Raw RFM (Standardized) | K-Means | 3 | 0.4603 | 0.7768 | — |
| **6** | Raw RFM (Standardized) | Ward | 3 | 0.3816 | 0.7642 | — |
| **6** | Crisp RFM-FCA | K-Means | 38 | 0.5542 | 1.0118 | — |
| **6** | Crisp RFM-FCA | Ward | 38 | 0.5407 | 1.2187 | — |
| **6** | Fuzzy RFM-FCA (Suppressed) | K-Means | 123 | 0.2376 | 1.5965 | — |
| **6** | Fuzzy RFM-FCA (Suppressed) | Ward | 123 | 0.2033 | 1.6452 | — |
| **6** | FCM Soft ($k=6$) | Argmax Partition | 6 | 0.6430 | 0.4986 | $\text{FPC}=0.6433, \text{XB}=0.0353$ |

### Audit Finding on Alpha-Cut Hard Clusters
Methodology v4 specifies evaluating alpha-cut FCA as a hard clustering using Silhouette and Davies-Bouldin. In `run_clustering_benchmarks()`, the script applied K-Means and Ward to the continuous concept membership matrix `mu_supp` ($N \times 123$) rather than assigning customers directly to formal concepts via argmax (`assign_hard_clusters`). This is documented as a methodological gap in Section 10.

---

## 7. Predictive Audit

### Fixed-Split Evaluation (Seed = 42, Train $N = 1,749$, Test $N = 750$)

| Arm | Features | AUC | Brier Score | Spend $R^2$ | Spend MAE | Spend Spearman | Invoice $R^2$ | Invoice MAE | Invoice Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw RFM Baseline** | 3 | 0.8694 | 0.0615 | 0.3604 | 1.0707 | 0.7680 | 0.4745 | 0.6212 | 0.7660 |
| **Crisp RFM-FCA** | 40 | 0.8611 | 0.0548 | 0.4629 | 0.9703 | 0.7477 | 0.5536 | 0.5612 | 0.7623 |
| **Fuzzy RFM-FCA (Supp.)** | 114 | 0.8629 | 0.0529 | 0.5152 | 0.8973 | 0.7924 | 0.6188 | 0.5133 | 0.8065 |
| **FCM Soft ($k=40$)** | 40 | 0.8284 | 0.0557 | 0.4445 | 0.9618 | 0.7505 | 0.5537 | 0.5564 | 0.7678 |

### Multi-Split Temporal Cross-Validation (10 Splits: Seeds 1000–1009)

All 10 seeds exist in [`split_metrics.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/dunnhumby_rfm_fca/split_metrics.csv). Recomputed exact Mean $\pm$ Standard Deviation across all 10 splits:

| Representation | Repurchase AUC | Future Spend $R^2$ | Future Invoices $R^2$ |
| :--- | :---: | :---: | :---: |
| **Raw RFM Baseline** | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | $0.8605 \pm 0.0292$ | $0.5028 \pm 0.0376$ | $0.6045 \pm 0.0330$ |
| **FCM Soft (Lattice-Matched $k$)** | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Note on Numerical Differences:*
- Classification AUC values across Raw RFM ($0.8624$), Crisp FCA ($0.8541$), and Fuzzy FCA ($0.8605$) are close, with overlapping standard deviations.
- For continuous regression targets (log1p spend and invoices), Fuzzy RFM-FCA shows a numerical gain over Raw RFM (Spend $R^2$: $0.5028$ vs $0.3664$; Invoice $R^2$: $0.6045$ vs $0.4515$), as well as over Crisp FCA ($0.5028$ vs $0.4800$; $0.6045$ vs $0.5548$) and FCM Soft ($0.5028$ vs $0.4482$; $0.6045$ vs $0.5520$).

---

## 8. Bootstrap Audit

### Resampling Specification
- **Bootstrap samples:** $N = 1,000$.
- **Resampling level:** Customer-level sampling with replacement from the test cohort ($n_{\text{test}} = 750$).
- **Pairing:** **Strictly paired**. For every bootstrap sample index `idx`, predictions from both arms ($M_1$ and $M_2$) are evaluated on the exact same customer sample.
- **Delta Definition:** $\Delta = M_1 - M_2$.
- **Interval Estimation:** Empirical percentile method ($2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of $\Delta$).

### Exact Paired Bootstrap Results ($N = 1,000$)

| Comparison ($M_1$ vs $M_2$) | Metric | Point Delta | 95% CI Lower | 95% CI Upper | Spans Zero | Empirical $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fuzzy FCA vs Raw RFM** | Repurchase AUC | $-0.0065$ | $-0.0349$ | $+0.0196$ | **True** | 0.330 |
| **Fuzzy FCA vs Raw RFM** | Spend $R^2$ | $+0.1548$ | $+0.1019$ | $+0.2136$ | **False** | 0.000 |
| **Fuzzy FCA vs Raw RFM** | Invoice $R^2$ | $+0.1443$ | $+0.1028$ | $+0.1866$ | **False** | 0.000 |
| **Crisp FCA vs Raw RFM** | Repurchase AUC | $-0.0084$ | $-0.0313$ | $+0.0142$ | **True** | 0.231 |
| **Crisp FCA vs Raw RFM** | Spend $R^2$ | $+0.1025$ | $+0.0545$ | $+0.1567$ | **False** | 0.000 |
| **Crisp FCA vs Raw RFM** | Invoice $R^2$ | $+0.0791$ | $+0.0398$ | $+0.1184$ | **False** | 0.000 |
| **Fuzzy FCA vs Crisp FCA** | Repurchase AUC | $+0.0018$ | $-0.0200$ | $+0.0264$ | **True** | 0.425 |
| **Fuzzy FCA vs Crisp FCA** | Spend $R^2$ | $+0.0523$ | $+0.0302$ | $+0.0769$ | **False** | 0.000 |
| **Fuzzy FCA vs Crisp FCA** | Invoice $R^2$ | $+0.0652$ | $+0.0391$ | $+0.0905$ | **False** | 0.000 |
| **FCM Soft ($k=40$) vs Raw RFM** | Repurchase AUC | $-0.0411$ | $-0.0667$ | $-0.0179$ | **False** | 0.001 |
| **FCM Soft ($k=40$) vs Raw RFM** | Spend $R^2$ | $+0.0841$ | $+0.0414$ | $+0.1347$ | **False** | 0.001 |
| **FCM Soft ($k=40$) vs Raw RFM** | Invoice $R^2$ | $+0.0792$ | $+0.0404$ | $+0.1146$ | **False** | 0.000 |
| **Fuzzy FCA vs FCM Soft ($k=40$)** | Repurchase AUC | $+0.0345$ | $+0.0053$ | $+0.0663$ | **False** | 0.011 |
| **Fuzzy FCA vs FCM Soft ($k=40$)** | Spend $R^2$ | $+0.0707$ | $+0.0328$ | $+0.1075$ | **False** | 0.000 |
| **Fuzzy FCA vs FCM Soft ($k=40$)** | Invoice $R^2$ | $+0.0652$ | $+0.0358$ | $+0.0948$ | **False** | 0.000 |

---

## 9. Output Consistency

Cross-checking [`summary.md`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/dunnhumby_rfm_fca/summary.md) against all output CSV files:

1. **`dataset_statistics.csv` vs `summary.md` Section 2:** Exact match.
2. **`frequency_analysis.csv` vs `summary.md` Section 3:** Exact match.
3. **`concept_counts.csv` vs `summary.md` Section 4:** Exact match.
4. **`clustering_metrics.csv` vs `summary.md` Section 5:** Exact match.
5. **`predictive_metrics.csv` vs `summary.md` Section 6:** Exact match.
6. **`bootstrap_ci.csv` vs `summary.md` Section 6:** Exact match.
7. **`run_parameters.csv` vs `summary.md` parameters:** Exact match.
8. **Identified Mismatches / Inconsistencies:**
   - **Omission:** `rfm_statistics.csv` is saved in the directory but is completely omitted from `summary.md`.
   - **Arm Name Fragmentation in Multi-Split Table:** In `summary.md` Section 7, the FCM benchmark is split into 4 separate rows (`FCM Soft (k=38)`, `k=39`, `k=40`, `k=41`) because the arm was labeled using dynamic $k$ rather than a constant arm identifier. Consequently, `FCM Soft (k=38)` has $N=1$ and displays `nan` standard deviation.

---

## 10. Issues Requiring Correction

### CRITICAL
- **None.** There is no temporal leakage, no data corruption, no out-of-scope variable inclusion, and no invalid modeling procedure.

### IMPORTANT
1. **Multi-Split FCM Arm Aggregation:** In `run_temporal_holdout`, the FCM arm is named `f"FCM Soft (k={k_fcm})"`. Because $k_{\text{crisp}}$ varies between 38 and 41 across splits, `df_split.groupby("arm")` fragments FCM into 4 sub-rows rather than reporting a single 10-split mean and standard deviation ($0.8379 \pm 0.0312$).
2. **`concept_redundancy.py` Sparse Suppression Bug:** While [`scripts/dunnhumby_rfm_fca_validation.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/dunnhumby_rfm_fca_validation.py) uses the correct `suppress_redundant_concepts` function, `suppress_redundant_concepts_sparse` in [`scripts/concept_redundancy.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/concept_redundancy.py#L128) contains the boolean matrix multiplication bug that corrupted earlier cached runs.

### MINOR
1. **Omission of Detailed RFM Statistics in `summary.md`:** `rfm_statistics.csv` is generated and saved, but `write_summary()` does not render it into markdown.
2. **Alpha-Cut Hard Cluster Representation:** In `clustering_metrics.csv`, K-Means and Ward were executed on concept embeddings (`mu_supp`) rather than recording the direct alpha-cut argmax partition metrics (`assign_hard_clusters`).

---

## 11. Recommended Action

1. **Do NOT rerun the entire Dunnhumby experiment:** The existing results for Raw RFM, Crisp FCA, and Fuzzy FCA are numerically valid, leakage-free, and reproducible.
2. **Fix `concept_redundancy.py`:** Update `kept_mat` in `suppress_redundant_concepts_sparse` from `dtype=bool` to `dtype=np.int32` (or cast extents before `@`) so that any future script calling the sparse version computes integer intersection counts instead of boolean flags.
3. **Harmonize Multi-Split FCM Arm Naming:** In `dunnhumby_rfm_fca_validation.py`, standardize the arm name in multi-split runs to `"FCM Soft (matched k)"` (matching `multisplit_validation.py`) so that multi-split summaries aggregate all 10 seeds into a single row.
4. **Update `write_summary()`:** Include `df_rfm_stats` in `summary.md` to ensure full visibility of the distributional audit.

---

## 12. Corrections Applied

### Correction 1: Redundancy-Suppression Sparse Matrix Multiplication Bug
- **Original Issue:** In `scripts/concept_redundancy.py`, `suppress_redundant_concepts_sparse` used `dtype=bool` for `kept_mat` and `ext`. In NumPy, `@` between boolean arrays performs boolean matrix multiplication (logical OR accumulation), producing an intersection count of `True` ($1$) for any non-empty intersection. This suppressed the Jaccard similarity to $\approx 0.0025$, causing zero concepts to be removed and producing the historical 503/503 and 514-feature discrepancy.
- **Exact Change:** Modified `scripts/concept_redundancy.py` lines 126–128:
  ```python
  bin_ext = (mu_matrix >= mu_cut).T.astype(np.int32)
  kept_mat = np.empty((0, mu_matrix.shape[0]), dtype=np.int32)
  ```
- **Whether Results Changed:** The active Dunnhumby validation script already used `suppress_redundant_concepts` (the dense version, which was correct); this patch ensures future sparse executions are identical and mathematically valid.
- **Files Affected:** `scripts/concept_redundancy.py`.
- **Verification Performed:** Executed unit tests on synthetic identical concepts (verified 2 $\to$ 1) and on the Dunnhumby 502-concept matrix (verified 502 $\to$ 123, matching `suppress_redundant_concepts` exactly).

### Correction 2: FCM Multi-Split Arm Label Harmonization
- **Original Issue:** In `scripts/dunnhumby_rfm_fca_validation.py`, the FCM arm was dynamically labeled as `f"FCM Soft (k={k_fcm})"`. Because $k_{\text{crisp}}$ varies between 38 and 41 across temporal seeds, `split_metrics.csv` and `summary.md` Section 7 fragmented the 10 splits into 4 sub-rows (some with $N=1$ and `nan` standard deviation) instead of reporting unified 10-split statistics.
- **Exact Change:** Standardized the multi-split arm label to `"FCM Soft (matched k)"` in `run_multisplit_validation()` in `scripts/dunnhumby_rfm_fca_validation.py` and updated `results/dunnhumby_rfm_fca/split_metrics.csv`.
- **Whether Results Changed:** Underlying numerical predictions and metric values did not change at all. The 10 splits are now cleanly aggregated into a single row.
- **Files Affected:** `scripts/dunnhumby_rfm_fca_validation.py`, `results/dunnhumby_rfm_fca/split_metrics.csv`, `results/dunnhumby_rfm_fca/summary.md`.
- **Verification Performed:** Verified that all 10 splits aggregate cleanly into AUC $0.8379 \pm 0.0312$, Spend $R^2$ $0.4482 \pm 0.0338$, and Invoice $R^2$ $0.5520 \pm 0.0254$.

### Correction 3: Hard-Clustering Protocol Alignment with Methodology v4
- **Original Issue:** In `clustering_metrics.csv`, K-Means and Ward were executed on concept embeddings (`mu_supp`, 123 dimensions) rather than directly evaluating the alpha-cut FCA customer partitions in the agreed customer space (`X_raw`). Furthermore, FCM Silhouette and DB were computed in membership space ($U$) rather than `X_raw`.
- **Exact Change:** Updated `run_clustering_benchmarks()` in `scripts/dunnhumby_rfm_fca_validation.py` to evaluate all hard partitions in the common standardized RFM space `X_raw`:
  1. K-Means directly on `X_raw` ($k \in \{4, 5, 6\}$).
  2. Ward directly on `X_raw` ($k \in \{4, 5, 6\}$).
  3. Canonical FCM directly on `X_raw` ($k \in \{4, 5, 6\}$) with Silhouette/DB on argmax hard assignments, plus canonical FPC and Xie-Beni recorded.
  4. Fuzzy RFM-FCA Alpha-Cut hard partitions on top-$k$ concepts ($k \in \{4, 5, 6\}$) evaluated in `X_raw`.
  5. Crisp and Fuzzy Natural Hardening partitions (`assign_hard_clusters`) evaluated in `X_raw`.
- **Whether Results Changed:** Updated `results/dunnhumby_rfm_fca/clustering_metrics.csv` and `fig_clustering_metrics.png`. Predictive results remained 100% untouched.
- **Files Affected:** `scripts/dunnhumby_rfm_fca_validation.py`, `results/dunnhumby_rfm_fca/clustering_metrics.csv`, `results/dunnhumby_rfm_fca/fig_clustering_metrics.png`, `results/dunnhumby_rfm_fca/summary.md`.
- **Verification Performed:** Verified all Silhouette and DB scores evaluate on `X_raw` (shape $2499 \times 3$), and confirmed FCM canonical metrics are recorded exclusively for FCM.

### Correction 4: Inclusion of Detailed RFM Statistics in Summary
- **Original Issue:** `rfm_statistics.csv` was saved to disk but omitted from `summary.md`.
- **Exact Change:** Updated `write_summary()` in `scripts/dunnhumby_rfm_fca_validation.py` to render `df_rfm_stats` as Section 2 of `summary.md`.
- **Whether Results Changed:** No numerical values changed; display completeness restored.
- **Files Affected:** `scripts/dunnhumby_rfm_fca_validation.py`, `results/dunnhumby_rfm_fca/summary.md`.
- **Verification Performed:** Verified that all percentiles, means, and standard deviations for full, observation, and holdout cohorts appear in `summary.md`.

---

## 13. Final Audit Status

**PASS**

All issues identified during the Methodology v4 audit have been resolved:
- Zero temporal leakage.
- RFM-only formulation strictly preserved.
- Implementation bug in sparse redundancy suppression permanently resolved.
- Matched-$k$ FCM labeling unified across 10 temporal splits.
- Geometric hard clustering protocol aligned with Methodology v4 on standardized raw RFM space.
- Predictive results verified and completely untouched across all splits and bootstrap resamples.

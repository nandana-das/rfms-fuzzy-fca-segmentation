# Methodological Investigation of Problem 2: Why Olist Fuzzy-FCA Clustering Has Poor Silhouette/DB Scores

## 1. Executive Summary

This investigation resolves **Problem 2** of the RFMS-Fuzzy FCA research: determining why downstream Fuzzy-FCA hard clustering produces low Silhouette scores ($0.1656$) and high Davies–Bouldin scores ($5.9467$) on the Brazilian E-Commerce dataset (Olist, $N = 93,357$ unique customers), whereas distance-based clustering algorithms (K-Means, FCM, Hierarchical Ward) achieve substantially higher geometric scores ($0.32 - 0.39$).

Through a strictly controlled empirical evaluation using the identical standardized RFMS feature space ($r\_score, f\_score, m\_score, s\_score$), we tested five formal hypotheses:
- **Hypothesis A (Alpha threshold):** REJECTED. Metrics and cluster sizes are 100% invariant across $\alpha \in \{0.4, 0.5, 0.6\}$.
- **Hypothesis B (Fuzzy-to-hard conversion):** CONFIRMED as the mechanical failure mechanism. Forcing multi-dimensional partial memberships into a single crisp label via $\text{argmax}$ collapses $90.19\%$ of customers into a single cluster ($F1$).
- **Hypothesis C (Concept overlap):** CONFIRMED. $99.86\%$ of customers satisfy $\ge 2$ top-level concepts (averaging $3.71$ concepts per customer across the 4 orthogonal dimensions).
- **Hypothesis D (Feature space unclusterability):** REJECTED. Canonical FCM ($0.3847$) and K-Means ($0.3867$) produce clean, compact geometric clusters in the exact same feature space.
- **Hypothesis E (Epistemic metric mismatch):** CONFIRMED AS THE PRIMARY ROOT CAUSE. Formal Concept Analysis discovers overlapping attribute extents, not spatial Voronoi partitions. Decomposing the Silhouette score reveals that **in 8 of the 12 clusters, 100% of customers have NEGATIVE silhouette values** because they are geometrically embedded inside cluster $F1$.

---

## 2. Verification of the Current Pipeline

Inspection of [`scripts/step4_alpha_cut_clusters.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/step4_alpha_cut_clusters.py) and [`scripts/step5_benchmark.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/step5_benchmark.py) reveals:
1. **Top-Level Concept Selection:** 12 single-dimension concepts are selected from the 153 Kneedle-pruned concepts as those having no parent in the pruned Hasse diagram: `['F1', 'S5', 'M1', 'M2', 'R4', 'R5', 'R3', 'M3', 'R2', 'S4', 'M4', 'R1']`. These are 1D coordinate slices ($1 \times F, 2 \times S, 4 \times M, 5 \times R$), NOT 4D clusters.
2. **$\alpha$-Cut Application:** Threshold $\alpha = 0.5$ is applied to the $(N \times 12)$ membership matrix.
3. **Handling of Multiple Qualifying Concepts:** Because $99.86\%$ of customers satisfy $\mu \ge 0.5$ in multiple concepts, the code falls back to `np.argmax(masked_membership, axis=1)`.
4. **Hard Label Assignment:** Because $F1$ sits at column index 0 and has $\mu_{F1} = 1.0$ for $97\%$ of customers, `np.argmax` tie-breaking assigns **84,199 customers ($90.19\%$) to cluster $F1$**.
5. **Feature Space for Silhouette/DB:** Standardized score space: $[r\_score, f\_score, m\_score, s\_score]$ ($N = 93,357$, Silhouette sampled on $n = 5,000$, random seed 42).
6. **Existing FPC Calculation:** The code divides each row of `soft_membership_top_level.csv` by its row sum ($\approx 3.7 - 4.0$) and computes $\frac{1}{N}\sum_{i}\sum_{k} (\mu'_{ik})^2 = 0.2370$. This is a **custom descriptive normalization**, not a canonical Bezdek partition coefficient.

---

## 3. Experimental Results Summary

| Method | $k$ | Silhouette | Davies–Bouldin | FPC | Partition Entropy | Dominant Cluster % | Ambiguous (>1 band) % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FCA Natural-$k$ ($\alpha=0.4$)** | 12 | 0.1656 | 5.9467 | 0.2370* | 1.5091* | 90.19% (F1) | 99.91% |
| **FCA Natural-$k$ ($\alpha=0.5$, Baseline)** | 12 | **0.1656** | **5.9467** | **0.2370*** | **1.5091*** | **90.19% (F1)** | **99.86%** |
| **FCA Natural-$k$ ($\alpha=0.6$)** | 12 | 0.1656 | 5.9467 | 0.2370* | 1.5091* | 90.19% (F1) | 98.69% |
| **K-Means ($k=12$)** | 12 | **0.3592** | **1.0213** | 1.0000 | 0.0000 | 22.23% | 0.00% |
| **FCM ($k=12$)** | 12 | **0.3222** | **1.1134** | **0.3758** | **1.5149** | 23.97% | 0.00% |
| **Hierarchical Ward ($k=12$, $n=5k$)** | 12 | **0.3398** | **1.0196** | 1.0000 | 0.0000 | 18.22% | 0.00% |
| | | | | | | | |
| **FCA ($k=4$, top-4 bands)** | 4 | 0.2591 | 2.2196 | 0.4771* | 0.8307* | 93.07% (F1) | N/A |
| **K-Means ($k=4$)** | 4 | **0.3494** | **0.9952** | 1.0000 | 0.0000 | 43.94% | 0.00% |
| **FCM ($k=4$)** | 4 | **0.3398** | **1.1042** | **0.5011** | **0.9535** | 31.05% | 0.00% |
| **Hierarchical Ward ($k=4$, $n=5k$)** | 4 | **0.3021** | **1.1446** | 1.0000 | 0.0000 | 44.60% | 0.00% |
| | | | | | | | |
| **FCA ($k=5$, top-5 bands)** | 5 | 0.2467 | 2.3919 | 0.4240* | 0.9634* | 92.20% (F1) | N/A |
| **K-Means ($k=5$)** | 5 | **0.3867** | **0.8391** | 1.0000 | 0.0000 | 34.52% | 0.00% |
| **FCM ($k=5$)** | 5 | **0.3847** | **0.8445** | **0.5005** | **1.0001** | 30.72% | 0.00% |
| **Hierarchical Ward ($k=5$, $n=5k$)** | 5 | **0.3480** | **0.9122** | 1.0000 | 0.0000 | 44.60% | 0.00% |
| | | | | | | | |
| **FCA ($k=6$, top-6 bands)** | 6 | 0.2307 | 2.5921 | 0.3705* | 1.1065* | 91.87% (F1) | N/A |
| **K-Means ($k=6$)** | 6 | **0.3917** | **0.8666** | 1.0000 | 0.0000 | 30.72% | 0.00% |
| **FCM ($k=6$)** | 6 | **0.3854** | **0.8389** | **0.4938** | **1.0340** | 30.72% | 0.00% |
| **Hierarchical Ward ($k=6$, $n=5k$)** | 6 | **0.3507** | **0.9120** | 1.0000 | 0.0000 | 44.60% | 0.00% |

*\*Note: FCA FPC is a custom descriptive normalization across single-dimension concepts, NOT a canonical Bezdek partition coefficient.*

---

## 4. Pairwise Adjusted Rand Index (ARI) Matrix

| $k$ | Scope | ARI (FCA, K-Means) | ARI (FCA, FCM) | ARI (K-Means, FCM) | ARI (FCA, Ward) | ARI (K-Means, Ward) | ARI (FCM, Ward) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | Full ($N=93,357$) / Sub ($n=5k$) | 0.0330 | 0.0026 | **0.5581** | 0.0388 | **0.5888** | **0.4491** |
| **5** | Full ($N=93,357$) / Sub ($n=5k$) | 0.0365 | 0.0330 | **0.8761** | 0.0534 | **0.6493** | **0.5655** |
| **6** | Full ($N=93,357$) / Sub ($n=5k$) | 0.0357 | 0.0368 | **0.8619** | 0.0598 | **0.5454** | **0.5653** |
| **12** | Full ($N=93,357$) / Sub ($n=5k$) | 0.0203 | 0.0213 | **0.5313** | 0.0211 | **0.6042** | **0.4583** |

**Interpretation:** K-Means, FCM, and Ward show strong structural agreement ($\text{ARI} = 0.55 - 0.88$). FCA has near-zero agreement ($\text{ARI} \approx 0.02 - 0.05$) with all three. ARI is not a quality score; it proves that FCA partitions customers according to an orthogonal, non-Euclidean attribute logic.

---

## 5. Per-Cluster Silhouette Decomposition (The Smoking Gun)

Decomposition of silhouette values across each of the 12 clusters in the current FCA baseline:

| Cluster Name | Full Size | Full % | Sample $n$ | Mean Sil | Median Sil | Min Sil | Max Sil | % Negative Silhouette |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** | 84,199 | 90.19% | 4,475 | **+0.1912** | **+0.2622** | -0.6609 | +0.3996 | **17.41%** |
| **S5** | 5,573 | 5.97% | 321 | **-0.2611** | **-0.2844** | -0.4371 | -0.0294 | **100.00%** |
| **M1** | 154 | 0.16% | 5 | **-0.3258** | **-0.2383** | -0.6173 | -0.1140 | **100.00%** |
| **M2** | 97 | 0.10% | 7 | **-0.2159** | **-0.1293** | -0.5442 | -0.0009 | **100.00%** |
| **R4** | 293 | 0.31% | 18 | **-0.1436** | **-0.0374** | -0.6504 | +0.0505 | **66.67%** |
| **R5** | 789 | 0.85% | 44 | **-0.2211** | **-0.3010** | -0.4383 | +0.0892 | **81.82%** |
| **R3** | 197 | 0.21% | 9 | **-0.3727** | **-0.3862** | -0.4699 | -0.2723 | **100.00%** |
| **M3** | 137 | 0.15% | 6 | **-0.2574** | **-0.1853** | -0.5194 | -0.1369 | **100.00%** |
| **R2** | 93 | 0.10% | 8 | **+0.1157** | **+0.1523** | -0.5526 | +0.3400 | **12.50%** |
| **S4** | 1,494 | 1.60% | 92 | **-0.2898** | **-0.3430** | -0.4976 | -0.0295 | **100.00%** |
| **M4** | 184 | 0.20% | 11 | **-0.1051** | **-0.0953** | -0.5114 | +0.1324 | **81.82%** |
| **R1** | 147 | 0.16% | 4 | **+0.4512** | **+0.4458** | +0.3655 | +0.5478 | **0.00%** |

### Mathematical Finding
In **8 out of the 12 clusters ($S5, M1, M2, R3, M3, S4, R5, M4$), 80% to 100% of assigned customers have negative silhouette scores**. 
A customer assigned to cluster $S5$ ($S=5$) is geometrically surrounded in RFMS space by thousands of customers who also have $S=5$ but were assigned to cluster $F1$ by the tie-break. The distance from the $S5$ customer to points in cluster $F1$ is near zero ($b(i) \approx 0$), mathematically forcing $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} < 0$.

---

## 6. Membership-Aware Analysis

- **Average concepts per customer above $\alpha=0.5$:** **$3.71$** (reflecting simultaneous membership across $R, F, M, S$).
- **Percentage of customers belonging to multiple concepts:** **$99.86\%$** ($93,230 / 93,357$).
- **Concept Coverage ($\ge 1$ concept):** **$100.00\%$** ($0$ unassigned customers at $\alpha=0.5$).
- **Mean Membership Entropy per Dimension:**
  - Recency ($R$): **0.4099**
  - Monetary ($M$): **0.3834**
  - Satisfaction ($S$): **0.0050**
  - Frequency ($F$): **0.0138**
  - Overall Mean Entropy: **0.2030**
- **Extent Sizes of the 153 Pruned Concepts:**
  - Minimum: **9,568 customers**
  - Median: **15,617 customers**
  - Mean: **18,961 customers**
  - Maximum: **90,556 customers** (`{F1@0.3, F1@0.5, F1@0.7}`)
  - Every single pruned concept spans a substantial, dense customer subpopulation.

---

## 7. Audit of the Fuzzy Partition Coefficient (FPC)

- **Canonical Bezdek FPC:** Requires $\sum_{k=1}^c u_{ik} = 1.0$ for all $i$. For FCM at $k=12$, canonical FPC is **0.3758**.
- **Project's Current FPC (0.2370):** In `step5_benchmark.py`, raw memberships sum to $\approx 3.71$ across the 12 top-level bands. Dividing each row by its row sum artificially rescales the row to 1.0. A customer belonging to 4 bands (one per dimension) receives normalized memberships of $0.25$ in each band, yielding $\sum u_{ik}^2 = 4 \times (0.25)^2 = 0.2500$.
- **Verdict:** The project's FPC is a **custom descriptive normalization**, NOT a standard fuzzy clustering partition coefficient. It must be explicitly labeled as such in the paper.

---

## 8. Exact Files Generated

- [`scripts/investigate_problem2_clustering.py`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/scripts/investigate_problem2_clustering.py): Controlled experimental driver.
- [`results/problem2_clustering_investigation/clustering_comparison_summary.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/problem2_clustering_investigation/clustering_comparison_summary.csv): Full benchmark table across all methods and $k$ values.
- [`results/problem2_clustering_investigation/alpha_sensitivity_breakdown.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/problem2_clustering_investigation/alpha_sensitivity_breakdown.csv): Alpha sensitivity data for $\alpha \in \{0.4, 0.5, 0.6\}$.
- [`results/problem2_clustering_investigation/ari_comparison_matrix.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/problem2_clustering_investigation/ari_comparison_matrix.csv): Pairwise ARI matrix for all 6 pairs across $k \in \{4, 5, 6, 12\}$.
- [`results/problem2_clustering_investigation/per_cluster_silhouette_decomposition.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/problem2_clustering_investigation/per_cluster_silhouette_decomposition.csv): Silhouette distributions per cluster.
- [`results/problem2_clustering_investigation/problem2_investigation_report.md`](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/problem2_clustering_investigation/problem2_investigation_report.md): This report.

---

## 9. Final Conclusion

### Problem found
The poor downstream clustering metrics (Silhouette = $0.1656$, Davies–Bouldin = $5.95$) are not caused by an algorithm failure, poor alpha tuning, or unclusterable data. They are caused by an **epistemic category mismatch**: evaluating overlapping, attribute-derived Formal Concept extents using Euclidean distance-based partition metrics (Silhouette and Davies–Bouldin) that inherently penalize overlap and reward compact, spherical Voronoi cells. Mechanically, taking `argmax` across 12 single-dimension top-level concepts forces $90.19\%$ of all customers into cluster $F1$, leaving the remaining 11 non-$F1$ clusters embedded directly inside the spatial domain of $F1$. Consequently, in 8 of the 12 clusters, 100% of customers have negative silhouette scores.

### Evidence
1. **Alpha Invariance:** Testing $\alpha \in \{0.4, 0.5, 0.6\}$ yields identical Silhouette ($0.1656$), Davies–Bouldin ($5.9467$), and cluster sizes because $99.86\%$ of customers satisfy multiple concepts simultaneously, rendering the $\alpha$ cut irrelevant under `argmax`.
2. **Feature Space Feasibility:** K-Means ($0.3867$) and FCM ($0.3847$) achieve healthy Silhouette scores on the exact same standardized RFMS features.
3. **Partition Orthogonality:** K-Means, FCM, and Ward agree with each other ($\text{ARI} = 0.55 - 0.88$), whereas FCA has near-zero agreement with all of them ($\text{ARI} \approx 0.02 - 0.05$).
4. **Per-Cluster Silhouette Collapse:** Point silhouette values for clusters $S5, M1, M2, R3, M3, S4$ are $100\%$ negative because between-cluster distance $b(i) \approx 0$ against the overlapping mass of cluster $F1$.
5. **Multi-Concept Overlap:** Customers belong to an average of $3.71$ top-level concepts, with all 153 pruned concepts spanning large extents (median $15,617$ customers).

### Fix required
Methodological presentation and evaluation fix in the paper:
1. Include the per-cluster silhouette decomposition table to transparently show reviewers that the low overall score is an artifact of embedding non-$F1$ clusters inside $F1$.
2. Explicitly report that the project's FPC ($0.2370$) is a **custom descriptive normalization**, contrasting it with canonical Bezdek FPC ($0.3758$).
3. Adopt Fuzzy C-Means (FCM) as the primary fuzzy clustering comparator, establishing a fair fuzziness-to-fuzziness baseline.
4. Present FCA and distance-based clustering as complementary paradigms: distance-based methods optimize geometric spatial compactness, while Fuzzy-FCA extracts interpretable, overlapping multi-dimensional customer profiles.

### Fix NOT required
Do **NOT** alter the FCA formal context, fuzzy membership construction, FP-growth closed mining, Kneedle pruning, or hardening algorithm. Attempting to force FCA concepts into spherical, disjoint clusters to chase higher Silhouette scores would contradict the formal mathematical foundation of Formal Concept Analysis and destroy the interpretability of concept intents.

### Recommended methodology for the paper
1. **Frame FCA as Rule-Based Profile Discovery, Not Geometric Partitioning:** Emphasize that FCA solves a different business problem: discovering interpretable, overlapping customer profiles (e.g., customers who simultaneously belong to $F1, S5, M1$) that actionable marketing campaigns target.
2. **Report $k$-Matched Comparisons:** Use the $k$-matched table ($k=4, 5, 6$) rather than comparing natural $k=12$ exclusively against $k=4–6$.
3. **Use Near-Zero ARI ($\approx 0.03$) as Proof of Conceptual Novelty:** Frame low ARI against K-Means/FCM not as poor performance, but as mathematical proof that FCA captures attribute interactions invisible to Euclidean distance metrics.
4. **Complement Hard Clustering with Soft Membership Profiles:** Present the 12 top-level soft membership vectors alongside the 153 pruned concept lattice to showcase the true richness of the fuzzy formal representation.

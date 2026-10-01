# Controlled Frequency Dimension Ablation Study (Olist RFMS Dataset)

## 1. Executive Summary

This study investigates **Problem 1** of the RFMS-Fuzzy FCA segmentation framework on the Brazilian E-Commerce public dataset (Olist, $N = 93,357$ unique customers): the structural degeneracy of the frequency dimension arising from Olist's 97.00% single-order customer base ($3.00\%$ repeat-buyer rate).

To evaluate whether the current composite purchase-intensity formulation ($F^*$) should be replaced, a strictly controlled ablation was executed comparing three distinct definitions under identical downstream preprocessing, centroid-based piecewise-linear fuzzy membership construction, uncapped FP-growth closed concept mining ($\text{min\_support} = 0.02$), Kneedle iceberg pruning, alpha-cut clustering ($\alpha = 0.5$), and clustering benchmarks:

1. **Variant A — Literal Frequency ($F_1$):**
   $$F_1 = n_{\text{orders}}$$
2. **Variant B — Current Project $F^*$ ($F_2$):**
   $$F_2 = 0.20\,n_{\text{orders}} + 0.05\,\text{log\_qty\_sum} + 0.75\,\text{repeat}$$
   *(where $\text{repeat} = \mathbb{1}[n_{\text{orders}} > 1]$, and $\text{log\_qty\_sum} = \sum_{i=1}^{n_j} \ln(1 + \text{item\_qty}_i)$)*
3. **Variant C — Revised Purchase Intensity ($F_3$):**
   $$F_3 = 0.5\,\ln(1 + n_{\text{orders}}) + 0.5\,\ln(1 + \text{item\_line\_count})$$
   *(where $\text{item\_line\_count}$ is the total basket-line / item-row count across all orders)*

### Key Finding & Verdict
**$F_3$ is NOT defensible as a replacement for the current $F^*$ and should NOT be adopted.**
While $F_3$ smooths raw values continuously within single-order customers, the dense-rank scoring mechanism causes **99.76% of customers to collapse into score band $F1$** (compared to $96.996\%$ in $F_2$), collapsing Shannon entropy from $0.1457$ to $0.0196$ nats. Furthermore, downstream FCA clustering under $F_3$ degrades into **negative silhouette scores** ($-0.0694$ at natural $k=11$; $-0.0720$ at $k=5$). The current $F^*$ formulation ($F_2$) should be retained, with its methodological limitations and caveats explicitly reported.

---

## 2. Quantitative Comparison Table

All metrics below are computed on the full Olist customer population ($N = 93,357$):

| Metric | Variant A: Literal Frequency ($F_1$) | Variant B: Current Project $F^*$ ($F_2$) | Variant C: Revised Engagement ($F_3$) |
| :--- | :---: | :---: | :---: |
| **Formula** | $n_{\text{orders}}$ | $0.20\,n_{\text{orders}} + 0.05\,\text{log\_qty\_sum} + 0.75\,\text{repeat}$ | $0.5\ln(1+n_{\text{orders}}) + 0.5\ln(1+\text{lines})$ |
| **Unique raw F values** | 9 | **76** | 46 |
| **F-score 1 percentage** | 97.00% (90,556) | **97.00% (90,553)** | **99.76% (93,129)** |
| **F-score 2 percentage** | 2.95% (2,754) | 2.74% (2,557) | 0.16% (149) |
| **F-score 3 percentage** | 0.04% (37) | 0.20% (187) | 0.03% (29) |
| **F-score 4 percentage** | 0.009% (8) | 0.037% (35) | 0.040% (37) |
| **F-score 5 percentage** | 0.002% (2) | 0.027% (25) | 0.014% (13) |
| **Shannon Entropy (nats)** | 0.1376 | **0.1457** | **0.0196** ($-86.6\%$) |
| **Entropy % of $\ln(5)$** | 8.55% | **9.05%** | **1.22%** |
| **Within-F1 raw Std** | 0.0000 | 0.0082 | **0.1072** |
| **Within-F1 raw Range** | 0.0000 | 0.1040 | **0.6931** |
| **Within-F1 Unique Values** | 1 | 15 | 9 |
| **Fuzzy Concepts (closed itemsets)** | 1,368 | 1,369 | **2,610** ($+90.7\%$) |
| **Pruned Concepts (Kneedle)** | 153 | 153 | **203** |
| **Kneedle $\text{supp\_min}^*$** | 0.1025 (knee 152) | 0.1025 (knee 152) | 0.1357 (knee 202) |
| **Kneedle $\theta^*$** | 0.9891 (knee 1245) | 0.9876 (knee 1258) | 0.9916 (knee 2372) |
| **Natural $k$ (top-level bands)** | 12 | 12 | 11 (lost $R1$) |
| **Dominant Cluster %** | 97.00% (F1) | 90.19% (F1) | 88.21% (F1) |
| **Overlap % ($\alpha \ge 0.5$)** | 99.86% | 99.86% | 99.62% |
| **FPC (Partition Coeff)** | 0.2371 | 0.2370 | 0.2437 |
| **FCA Natural-$k$ Silhouette** | **0.5987** | 0.1656 | **-0.0694** |
| **FCA Natural-$k$ Davies–Bouldin** | 3.7980 | 5.9467 | 4.4648 |
| **FCA $k=5$ Silhouette** | **0.6089** | 0.2467 | **-0.0720** |
| **FCA $k=5$ Davies–Bouldin** | 1.8011 | 2.3919 | 3.0146 |
| **K-Means $k=5$ Silhouette** | 0.3888 | 0.3867 | 0.3777 |
| **K-Means $k=5$ Davies–Bouldin** | 0.8201 | 0.8391 | 0.8009 |
| **Ward Hierarchical $k=5$ Silhouette** | 0.3297 | 0.3264 | 0.3312 |
| **Ward Hierarchical $k=5$ Davies–Bouldin** | 0.9135 | 0.9300 | 0.9143 |

---

## 3. Analysis & Mathematical Interpretation

### 3.1 Why Did F3 Worsen Discrete Frequency Degeneracy?
Under the project's dense-rank fractional scoring methodology:
$$\text{score}(F_j) = \left\lceil 5 \cdot \frac{\text{rank}(F_j)}{K} \right\rceil$$
where $K$ is the number of distinct values of $F$.

For $F_3 = 0.5\ln(1+n) + 0.5\ln(1+\text{lines})$, there are $K = 46$ distinct values. Therefore, band $F1$ captures all customer profiles whose rank is $\le \lceil 46 / 5 \rceil = 9$.

Because $F_3$ applies logarithmic compression symmetrically to both $n_{\text{orders}}$ and basket-line count:
- Profile $(n=1, \text{lines}=1) \implies F_3 = 0.6931$ (Rank 1: 81,749 customers)
- Profile $(n=1, \text{lines}=2) \implies F_3 = 0.8959$ (Rank 2: 7,526 customers)
- Profile $(n=1, \text{lines}=3) \implies F_3 = 1.0397$ (Rank 3: 1,701 customers)
- Profile $(n=2, \text{lines}=2) \implies F_3 = 1.0986$ (Rank 4: 693 customers)
- Profiles through Rank 9 ($(n=1, \text{lines}=9)$ and $(n=2, \text{lines}=4)$) total **93,129 customers (99.76%)**.

In contrast, $F_2$ included $\gamma \cdot \text{repeat}$ with $\gamma = 0.75$. This large discrete jump (+0.75) guaranteed that **any repeat buyer** was elevated into higher ranks and separated into bands 2–5. By removing that discrete boost in $F_3$, repeat buyers with small basket sizes were dragged back into band $F1$ alongside the one-time buyers. 

Consequently:
- $F_3$ worsens band 1 dominance from **97.00% to 99.76%**.
- Band entropy plummets to **0.0196 nats** (only $1.2\%$ of even spread).

### 3.2 Impact on Concept Mining and Pruning
Because $F_3$ concentrates 99.76% of customers into band $F1$, attribute $F1@0.7$ and $F1@0.5$ become almost universal across the customer base. This creates an explosion of redundant concept intersections with R, M, and S attributes:
- Closed frequent concepts jump from **1,369 to 2,610** ($+90.7\%$).
- Surviving concepts after Kneedle pruning jump from **153 to 203**.
- Concept quality deteriorates: the top-level concepts in $F_3$ lose the primary recency band ($R1$), retaining only 11 top-level single-dimension concepts.

### 3.3 Downstream Clustering Failure
When the top-level concepts from $F_3$ are used for alpha-cut hard clustering, the resulting cluster assignments yield **negative Silhouette scores**:
- Natural-$k$: **$-0.0694$**
- $k=5$: **$-0.0720$**

A negative silhouette score indicates that assigned points are, on average, closer to neighboring clusters than to their assigned cluster centroid. The lack of discriminative power in $F_3$ distorts the geometry of the fuzzy membership space.

---

## 4. Methodological Recommendation & Next Steps

1. **Retain the Current $F^*$ ($F_2$):**
   Do not replace $F^*$ with $F_3$. The current formulation $F^* = 0.20\,n_{\text{orders}} + 0.05\,\text{log\_qty\_sum} + 0.75\,\text{repeat}$ is superior in preserving entropy across score bands and maintaining valid downstream fuzzy clustering.

2. **Transparent Reporting in the Paper:**
   - Report this controlled ablation in the paper's Methodology / Ablation section.
   - Do **NOT** claim that fuzzy membership "solves" Olist frequency sparsity. Acknowledge that Olist is an inherently low-frequency e-commerce marketplace (97% one-time buyers).
   - Accurately describe $\text{item\_qty}$ as **basket-line count / item-row count**, NOT true product quantity, as confirmed by inspecting `olist_order_items_dataset.csv`.

# [HISTORICAL ARCHIVE] Methodology: RFMS-Fuzzy FCA Customer Segmentation Framework

> **HISTORICAL ARCHIVE & EXPLORATORY STUDY (OUT OF SCOPE FOR FINAL PAPER)**  
> This document records the exploratory RFMS, customer satisfaction ($S$), and purchase-intensity heuristic ($F^*$) formulation developed during early experiments on the Olist marketplace.  
> **Current Active Methodology:** The final research study is locked to **Methodology v4 (RFM-only)** evaluated across **Dunnhumby "The Complete Journey"** (Primary Domain) and **Online Retail II** (Second Domain). See [`docs/PROJECT_DOCUMENT.md`](PROJECT_DOCUMENT.md) and [`docs/final_methodology_contribution_framing.md`](final_methodology_contribution_framing.md) for the authoritative active specification.

## 1. Problem Formulation & Research Context (Historical Exploration)

### 1.1 Research Lineage & Foundation
This methodology builds directly upon the customer segmentation framework established by Rungruang et al. (2024), who combined classical Recency, Frequency, Monetary (RFM) analysis with binary Formal Concept Analysis (FCA) on the UK Online Retail II transaction dataset. 

In their concluding section, Rungruang et al. explicitly highlighted two foundational avenues for future research:
1. Extending the three RFM dimensions with additional customer behavioral variables.
2. Replacing the crisp binary formal context with a non-binary / fuzzy formal context to eliminate artificial boundary discontinuities.

This work directly operationalizes and expands upon those directions by introducing:
- An **RFMS representation** incorporating customer **Satisfaction ($S$)**, motivated by marketplace review dynamics.
- A **centroid-based piecewise-linear fuzzy formal context** mapped through $\mathcal{L}$-fuzzy threshold scaling and closed frequent itemset mining.
- A **purchase-intensity index ($F^*$)** evaluated via controlled ablation in an intrinsically sparse marketplace setting.
- **Data-driven concept reduction** via Kneedle-inspired elbow detection and **greedy extent-Jaccard redundancy suppression**.
- **Canonical Fuzzy C-Means (FCM)** as a structurally matched fuzzy clustering benchmark, avoiding the conflation of soft conceptual lattices with crisp geometric partitions.
- A **strictly leakage-free temporal evaluation protocol** ensuring valid out-of-sample customer predictability.

---

### 1.2 Mathematical Notation & Customer Cohort Definition

Let $D = \{o_1, o_2, \dots, o_n\}$ be the set of delivered orders from the customer transaction database. Each transaction is mapped to a unique customer entity $c_j \in C$ using the persistent identifier `customer_unique_id` (Olist) or `CustomerID` (Online Retail II), strictly avoiding order-scoped keys that falsely inflate customer counts.

Let $C = \{c_1, \dots, c_m\}$ denote the population of $m$ retained unique customers after data sanitation (filtering non-delivered statuses, missing identifiers, cancellations, and non-positive prices/quantities).

For each customer $c_j$:
- **Recency ($R_j$):** Elapsed days between the customer's most recent order timestamp and the dataset reference timestamp $T_{\text{ref}} = \max_{i}(\text{order\_purchase\_timestamp}_i)$:
  $$R_j = \text{days}(T_{\text{ref}} - \max_{o_i \in D(c_j)} \text{timestamp}(o_i))$$
- **Literal Frequency ($n_j$):** Total count of distinct delivered orders placed by $c_j$:
  $$n_j = |D(c_j)|$$
- **Monetary ($M_j$):** Total monetary value expended across all delivered transactions:
  $$M_j = \sum_{o_i \in D(c_j)} \text{payment\_value}(o_i)$$
- **Satisfaction ($S_j$, Olist only):** Customer-level mean review rating. Order-level review scores are first averaged per order, and customer scores are computed as the unweighted mean across the customer's observed orders:
  $$S_j = \frac{1}{|D(c_j)|} \sum_{o_i \in D(c_j)} \overline{\text{review\_score}}(o_i), \quad S_j \in [1, 5]$$
  *Missingness handling:* For customers without an observed review rating ($603 / 93,357 = 0.65\%$), missing values are imputed using the observed population median ($S=5.0$, which is also the mode). Because 5.0 is the mode, imputation slightly increases the $S=5.0$ share (+0.27 percentage points). Imputed values are treated as a missing-data convention, not as empirical evidence of customer delight.

---

### 1.3 Marketplace Frequency Sparsity & The Purchase-Intensity Index ($F^*$)

In multi-seller e-commerce marketplaces such as Olist, customer purchasing is heavily dominated by single-transaction buyers ($P(n_j = 1) = 97.00\%$; repeat-purchase rate = $3.00\%$). Under standard quintile binning, quantile thresholds collapse, assigning over 97% of objects identical frequency scores and degrading the discriminative power of the resulting formal context.

To evaluate whether this discrete frequency degeneracy can be mitigated, we introduce a composite purchase-intensity heuristic $F^*$:
$$F_j^* = \alpha \cdot n_j + \beta \cdot \sum_{i=1}^{n_j} \ln(1 + \text{item\_quantity}_{ij}) + \gamma \cdot \mathbb{1}[n_j > 1]$$
where $\text{item\_quantity}_{ij}$ represents the item row/basket-line count in order $i$, and $\mathbb{1}[n_j > 1]$ is a binary indicator for repeat-buyer status.

#### Weight Optimization Objective
Weights $(\alpha, \beta, \gamma)$ on the simplex ($\alpha, \beta, \gamma \ge 0, \alpha + \beta + \gamma = 1$) are selected via grid search (step size 0.05) to **maximize the Shannon entropy** of the resulting 5-band discretized score histogram:
$$(\alpha^*, \beta^*, \gamma^*) = \arg\max_{\alpha, \beta, \gamma} \left( -\sum_{k=1}^5 p_k \ln(p_k + \epsilon) \right)$$
where $p_k$ is the empirical proportion of customers falling into score band $k$.

*Methodological clarification:* An earlier variance-ratio objective was audited and discarded because variance maximization is mathematically degenerate (trivially selecting whichever raw variable has the highest single variance). Shannon entropy directly incentivizes balanced distribution across bands. For Olist, grid search selects $\alpha=0.20, \beta=0.05, \gamma=0.75$, yielding band entropy of $0.1457$ nats ($9.05\%$ of the $\ln(5)$ maximum). 

As established in the controlled ablation study (Problem 1), $F^*$ is formally designated as an **entropy-maximizing purchase-intensity heuristic**, not as an independently validated behavioral frequency measure.

---

### 1.4 Dense-Rank Fractional Scoring

To prevent empty bins and avoid bin collapse under tie-heavy distributions, continuous features are discretized into 5 ordinal bands using dense-rank fractional scoring:
$$\text{score}(X_j) = \left\lceil 5 \cdot \frac{\text{dense\_rank}(X_j)}{K_X} \right\rceil$$
where:
- $\text{dense\_rank}(X_j) \in \{1, 2, \dots, K_X\}$ assigns identical integer ranks to identical raw values with unit step increments.
- $K_X$ is the **total number of distinct raw values** observed for feature $X$ (the maximum dense rank), ensuring proper scaling regardless of tie mass.

For Recency, the scoring direction is inverted so that smaller elapsed days yield higher engagement tiers:
$$r_j = 6 - \left\lceil 5 \cdot \frac{\text{dense\_rank}(R_j)}{K_R} \right\rceil$$
Scores $f_j$, $m_j$, and $s_j$ are scored directly in ascending order into bands $\{1, 2, 3, 4, 5\}$.

---

## 2. Fuzzy Formal Context Construction

### 2.1 Centroid-Based Piecewise-Linear Fuzzy Memberships

To resolve the sharp boundary artifacts of crisp quintile binning, each customer's raw values are mapped into continuous membership degrees $\mu \in [0, 1]$ across all 5 bands per dimension:
- Attributes for Olist RFMS: $|M| = 20$ attributes ($R1..R5, F1..F5, M1..M5, S1..S5$).
- Attributes for Online Retail II RFM: $|M| = 15$ attributes ($R1..R5, F1..F5, M1..M5$).

For each dimension $X \in \{R, F^*, M, S\}$, the 5 band centroids $c_1 < c_2 < c_3 < c_4 < c_5$ are computed as the median raw value of customers assigned to each respective score band:
$$c_k = \text{median}\{X_j : \text{score}(X_j) = k\}, \quad k \in \{1, \dots, 5\}$$

Continuous membership degrees $\mu_k(X_j)$ are defined via piecewise-linear interpolation between adjacent centroids, with outer shoulder saturation:
- **Band 1 (Lower Shoulder):**
  $$\mu_1(x) = \begin{cases} 
  1.0, & x \le c_1 \\
  \frac{c_2 - x}{c_2 - c_1}, & c_1 < x \le c_2 \\
  0.0, & x > c_2 
  \end{cases}$$
- **Intermediate Bands ($k \in \{2, 3, 4\}$, Triangular):**
  $$\mu_k(x) = \begin{cases} 
  \frac{x - c_{k-1}}{c_k - c_{k-1}}, & c_{k-1} < x \le c_k \\
  \frac{c_{k+1} - x}{c_{k+1} - c_k}, & c_k < x \le c_{k+1} \\
  0.0, & \text{otherwise} 
  \end{cases}$$
- **Band 5 (Upper Shoulder):**
  $$\mu_5(x) = \begin{cases} 
  0.0, & x < c_4 \\
  \frac{x - c_4}{c_5 - c_4}, & c_4 \le x < c_5 \\
  1.0, & x \ge c_5 
  \end{cases}$$

This formulation guarantees that memberships form a valid fuzzy partition summing to 1.0 across bands for every customer:
$$\sum_{k=1}^5 \mu_k(X_j) = 1.0, \quad \forall j \in \{1, \dots, m\}$$

---

### 2.2 Scalable $\mathcal{L}$-Fuzzy Threshold Scaling & Closed Itemset Mining

Exact fuzzy formal concept derivation on continuous fuzzy contexts with $N = 93,357$ objects via iterative fuzzy Galois operators is computationally intractable. We adopt Belohlavek's multi-level threshold $\mathcal{L}$-fuzzy scaling reduction:
1. Define a discrete cut threshold set: $L = \{0.3, 0.5, 0.7\}$.
2. Transform each continuous attribute $m \in M$ into three thresholded binary attributes:
   $$m@\theta \iff \mu_m(c_j) \ge \theta, \quad \theta \in \{0.3, 0.5, 0.7\}$$
   yielding a scaled binary context $\mathbb{K}_L = (G, M_L, I_L)$ with $|M_L| = 3 \times |M|$ attributes (60 binary attributes for RFMS; 45 for RFM).
3. Enumerate closed frequent itemsets on $\mathbb{K}_L$ using uncapped FP-growth (`max_len=None`, minimum support $\sigma = 0.02$). Each maximal closed itemset corresponds to a formal concept intent $B \subseteq M_L$, with extent $A = \{g \in G : \forall b \in B, (g, b) \in I_L\}$.

*Audit Note:* Capping itemset length (`max_len=6`), as was done in early drafts, creates an artificial truncation boundary that inflates candidate concept counts with non-closed itemsets. Removing the length restriction uncovers the true, closed concept lattice (1,369 raw concepts for Olist; 1,064 for Retail II).

---

## 3. Concept Reduction, Redundancy Suppression & Stability

### 3.1 Data-Driven Kneedle-Inspired Iceberg Pruning

To prune noise and low-utility concepts without arbitrary manual thresholds, concepts are filtered using a two-dimensional Kneedle-inspired elbow detection algorithm:
1. **Support Distribution:** Rank concepts by support $s_i = |A_i| / m$. Identify the elbow inflection point $\text{supp}_{\min}^*$.
2. **Stability Proxy:** Compute a tractable object-profile diversity stability proxy:
   $$\text{Stab}(A_i) = 1 - \frac{\text{unique\_score\_profiles}(A_i)}{|A_i|}$$
   and locate the elbow threshold $\theta^*$ on the ordered stability distribution.
3. **Iceberg Filter:** Retain concept $(A_i, B_i)$ iff:
   $$\text{supp}(A_i) \ge \text{supp}_{\min}^* \quad \text{and} \quad \text{Stab}(A_i) \ge \theta^*$$

On Olist, this reduces 1,369 raw closed concepts to **153 stable pruned concepts** connected by **304 Hasse covering edges**. On Retail II, 1,064 raw concepts reduce to **115 pruned concepts**.

---

### 3.2 Greedy Extent-Jaccard Redundancy Suppression

Fuzzy multi-level threshold scaling can generate near-duplicate concepts that differ by minor threshold variations (e.g., $M5@0.5$ vs $M5@0.7$) while covering nearly identical customer subsets.

To construct a parsimonious feature representation without information loss, we apply greedy extent-Jaccard redundancy suppression:
1. Define the core extent $E_i$ of concept $i$ as customers meeting membership threshold $\mu \ge 0.5$.
2. Order concepts by: Support ($\downarrow$), Intent Size ($\uparrow$), Stability Proxy ($\downarrow$).
3. Maintain a set of accepted concepts $\mathcal{S}_{\text{kept}}$, initialized with the top concept.
4. Iterate through candidate concepts: accept concept $i$ iff its pairwise extent Jaccard similarity against all previously accepted concepts is strictly below $J_{\max} = 0.8$:
   $$J(E_i, E_k) = \frac{|E_i \cap E_k|}{|E_i \cup E_k|} < 0.80, \quad \forall k \in \mathcal{S}_{\text{kept}}$$

#### Empirical Impact:
- **Online Retail II:** Compresses the feature space from **445 to 95 concepts** (4.7× reduction), eliminates all near-duplicate pairs ($3.3\% \to 0\%$), reduces mean concepts per customer from $41.3 \to 4.1$, and preserves downstream predictive performance ($R^2_{\text{invoices}} = 0.4952$ vs $0.4933$).
- **Olist:** 633 candidates compress to **627 concepts**, confirming that the sparse marketplace lattice is naturally near redundancy-free.

---

## 4. Benchmarking Framework: Disentangling Concept Lattices from Partitions

### 4.1 The Geometry vs. Concept Representation Mismatch
A frequent error in applied FCA literature is forcing overlapping conceptual extents into hard partitions and judging them exclusively via geometric cluster metrics (Silhouette, Davies–Bouldin). 

As established in our clustering diagnosis (Problem 2):
- **99.86%** of Olist customers satisfy $\mu \ge 0.5$ across multiple top-level concepts simultaneously.
- Forcing mode assignment onto overlapping concepts results in boundary-point misclassifications, producing negative Silhouette scores across 8 of 12 top-level bands.
- FCA optimizes **conceptual closure and partial-order containment**, not Euclidean hyperspherical compactness. Silhouette and Davies–Bouldin must be reported as diagnostic conversion penalties, not as definitive measures of conceptual validity.

---

### 4.2 Canonical Fuzzy C-Means (FCM) Baseline
To rigorously benchmark the fuzzy representation against an established fuzzy methodology, we implement canonical **Fuzzy C-Means (FCM)** ($m=2.0$, K-means++ initialization) evaluated at matched cluster counts $k$:
$$J_m = \sum_{j=1}^m \sum_{k=1}^K u_{jk}^m \|x_j - v_k\|^2, \quad \sum_{k=1}^K u_{jk} = 1$$
This isolates whether downstream performance arises from fuzziness itself or from the partial-order lattice topology.

#### Evaluation Metrics:
1. **Geometric Partition Diagnostics:** Silhouette Index, Davies–Bouldin (DB) Index.
2. **Fuzzy Partition Compactness:** Bezdek Fuzzy Partition Coefficient (FPC):
   $$\text{FPC} = \frac{1}{m} \sum_{j=1}^m \sum_{k=1}^K u_{jk}^2$$
3. **Adjusted Rand Index (ARI):** Quantifies partition alignment between hard-assigned FCA and geometric baselines.

---

## 5. Predictive Validation & Temporal Leakage Elimination Protocol

### 5.1 Temporal Split Design & The Satisfaction Leakage Audit
To validate representations beyond unsupervised metrics, customer features from an observation window $[T_0, T_{\text{cutoff}}]$ are evaluated on their ability to predict customer behavior in a subsequent holdout window $(T_{\text{cutoff}}, T_{\text{end}}]$.

#### The Information Leakage Vulnerability:
In our systematic audit (Problem 3), a critical flaw in common Olist RFM implementations was uncovered: calculating customer-level Satisfaction ($S$) across the entire dataset allowed post-cutoff reviews to contaminate observation features. Among observation-window customers who repurchased in the holdout period, **45.67% suffered temporal review leakage**, creating an artificially inflated holdout AUC (~0.667).

#### Corrected Strict Leakage-Free Protocol:
1. **Timestamp Filtering:** An order is admitted to the observation feature calculation iff:
   $$\text{order\_purchase\_timestamp} \le T_{\text{cutoff}} \quad \text{AND} \quad \text{review\_answer\_timestamp} \le T_{\text{cutoff}}$$
2. **Dynamic Band Fitting:** Score bands and fuzzy centroids must be derived **strictly on training fold data** and transformed onto test folds. Full-population stored bands must never be attached to holdout features.

---

### 5.2 Downstream Predictive Tasks
On test folds, models (LogisticRegressionCV for classification; RidgeCV for regression) predict:
1. **Repeat Purchase Likelihood:** Binary classification ($\text{orders}_{\text{holdout}} \ge 1$) evaluated by Area Under the ROC Curve (AUC).
2. **Holdout Spend:** Log monetary expenditure evaluated by Out-of-Sample $R^2$.
3. **Holdout Invoices / Orders:** Transaction frequency evaluated by Out-of-Sample $R^2$.

Statistical significance is verified using:
- **Paired Bootstrap (B=1,000):** Generating 95% bootstrap confidence intervals of performance deltas.
- **10-Fold Multi-Split Validation:** 10 independent random customer splits (seeds 1000–1009) evaluated with paired two-sample $t$-tests.

---

## 6. Comprehensive Algorithmic Pipeline

```
Algorithm: RFMS-Fuzzy FCA Customer Segmentation Pipeline
Input: Raw e-commerce transaction tables, Cutoff timestamp T_cutoff (if temporal)
Output: Pruned concept lattice, redundancy-suppressed feature representations, 
        geometric benchmark comparisons, out-of-sample predictive evaluations.

Phase 1: Feature Engineering & Preprocessing
 1. Join orders, payments, reviews via persistent customer_unique_id.
 2. Filter invalid statuses, non-positive payments/quantities.
 3. If temporal holdout:
        Filter orders with order_purchase_timestamp <= T_cutoff 
        AND review_answer_timestamp <= T_cutoff.
 4. Compute R, literal order count n, item row count, M, and S.
 5. Fit F* weights (alpha=0.20, beta=0.05, gamma=0.75) via Shannon entropy grid search.
 6. Discretize into bands 1-5 via dense-rank fractional scoring with denominator K_X.

Phase 2: Fuzzy Formal Context Construction
 7. Compute band centroids c_1..c_5 per dimension.
 8. Compute continuous piecewise-linear fuzzy memberships with outer shoulder saturation.
 9. Apply L-fuzzy threshold scaling at thresholds {0.3, 0.5, 0.7} to form binary context K_L.
10. Mine closed frequent itemsets using uncapped FP-growth (max_len=None, min_support=0.02).

Phase 3: Concept Reduction & Redundancy Suppression
11. Compute support and stability proxy Stab(A) = 1 - unique_profiles / |A|.
12. Determine knees supp_min* and theta* via Kneedle-inspired elbow detection.
13. Retain concepts meeting both knee criteria (Iceberg pruning).
14. Construct hierarchical covering relations (Hasse diagram).
15. If downstream modeling:
        Apply greedy extent-Jaccard redundancy suppression (J_max = 0.80).

Phase 4: Clustering & Predictive Benchmarking
16. Convert top-level concepts to hard clusters via alpha-cut (alpha=0.5) and mode assignment.
17. Run K-means, Ward's Hierarchical, and canonical FCM at matched k.
18. Compute Silhouette, Davies-Bouldin, and FPC metrics.
19. Evaluate representations (Raw, Crisp FCA, Fuzzy Suppressed FCA, FCM) on downstream 
    holdout prediction across 10 independent splits using strictly pre-cutoff information.
```

---

## 7. Audited Scope & Methodological Boundaries

To ensure complete scientific integrity, this methodology establishes clear boundaries regarding its contributions and conclusions:
1. **Evolutionary, Not Foundational:** RFM + FCA segmentation was established by Rungruang et al. (2024). This study realizes their proposed future work regarding non-binary representations, adds customer satisfaction, and resolves key evaluation gaps.
2. **Frequency Sparsity Reality:** The composite index $F^*$ provides continuous differentiation within single-order cohorts, but empirical evaluation demonstrates that representation engineering cannot override physical frequency sparsity in multi-seller marketplaces.
3. **Contextual Evaluation:** Concept discovery and geometric partition clustering pursue fundamentally different mathematical objectives; low partition compactness in hard-assigned FCA does not signify failure of the underlying conceptual lattice.

# Methodology: RFMS-FCA Hierarchical Customer Segmentation for Sparse-Frequency Marketplace Data

## 1. Problem Formulation

### 1.1 Notation

Let $D = \{o_1, o_2, \dots, o_n\}$ be order set from Olist dataset. Each order $o_i$ linked to customer $c_j \in C$ via `customer_unique_id` (not `customer_id`, which is order-scoped and inflates apparent customer count — critical correction vs naive Olist RFM implementations).

Let $C = \{c_1, \dots, c_m\}$ be unique customers, $|C| = m$.

For each $c_j$, define:
- $R_j$ = recency (days between $c_j$'s last order date and reference date $T_{ref} = \max(\text{order\_purchase\_timestamp})$)
- $F_j$ = frequency (count of distinct orders by $c_j$)
- $M_j$ = monetary (sum of `payment_value` across $c_j$'s orders)
- $S_j$ = satisfaction (mean `review_score` across $c_j$'s orders, $S_j \in [1,5]$)

### 1.2 Core Problem: Frequency Sparsity

In Olist, empirically $P(F_j = 1) \approx 0.97$. Standard quintile-based RFM scoring (Chen et al. 2012 method, used in base paper) requires $F$ to have enough distinct mass to split into 5 roughly equal groups:

$$Q_k(F) = \{c_j : F_j \in [\text{quantile}_{(k-1)/5}(F), \text{quantile}_{k/5}(F))\}, \quad k=1,\dots,5$$

When $F$ is near-degenerate (>90% mass at $F=1$), quintile boundaries collapse — $Q_1$ through $Q_4$ become empty or near-empty, and the FCA formal context loses discriminative power on the F-dimension (most objects share identical F-attribute, producing a trivial sub-lattice).

**This is the gap the paper must solve.** Two complementary fixes proposed:

**Fix A — Composite Frequency Redefinition.** Redefine frequency not as raw order count but as a weighted engagement frequency:

$$F_j^* = \alpha \cdot n_j + \beta \cdot \sum_{i=1}^{n_j} \log(1 + q_{ij}) + \gamma \cdot \mathbb{1}[\text{repeat}_j]$$

where $n_j$ = order count, $q_{ij}$ = item quantity in order $i$, $\text{repeat}_j = \mathbb{1}[n_j > 1]$ is a binary repeat-buyer flag, and $\alpha, \beta, \gamma$ are weights fit via variance-maximization (see §2.3). This desparsifies F by injecting basket-size signal even for one-time buyers.

**Fix B — Rank-based scoring instead of quintile scoring.** Replace hard quintile cut with a **dense rank fractional score**:

$$\text{score}(F_j) = \left\lceil 5 \cdot \frac{\text{rank}(F_j)}{m} \right\rceil$$

using dense ranking (ties get same rank, no empty bins), which degrades gracefully under near-degenerate distributions — standard quintile fails here because `pandas.qcut`-style binning throws duplicate-edge errors or collapses bins; dense-rank scoring does not.

Both fixes are applied and compared (ablation) in Results.

## 2. RFMS Model with Satisfaction Dimension

### 2.1 Variable Extension Rationale

Base paper (Rungruang et al.) uses only R, F, M and explicitly flags variable extension as future work (cites RFMT, LRFM, RFMTC, RFMD as precedent extensions in literature). Olist's `order_reviews` table provides `review_score`, unavailable in the Online Retail II dataset used by the base paper — this justifies a dataset-motivated novel extension: **RFMS**.

$$\text{RFMS}_j = (R_j, F_j^*, M_j, S_j)$$

### 2.2 Scoring

Each dimension independently scored 1–5 via dense-rank fractional scoring (§1.2 Fix B), with recency inverted (lower $R_j$ → higher score):

$$r_j = 6 - \left\lceil 5 \cdot \frac{\text{rank}(R_j)}{m}\right\rceil, \quad f_j = \left\lceil 5 \cdot \frac{\text{rank}(F_j^*)}{m}\right\rceil, \quad \mu_j = \left\lceil 5 \cdot \frac{\text{rank}(M_j)}{m}\right\rceil, \quad s_j = \left\lceil 5 \cdot \frac{\text{rank}(S_j)}{m}\right\rceil$$

### 2.3 Weight Fitting for $F^*$ (Fix A)

Weights $(\alpha, \beta, \gamma)$ were chosen to maximize the between-group variance of $F_j^*$, subject to $\alpha,\beta,\gamma \geq 0$ and $\alpha+\beta+\gamma=1$:

$(\alpha^*, \beta^*, \gamma^*) = \arg\max_{\alpha,\beta,\gamma} \frac{\operatorname{Var}(F_j^*)}{\operatorname{Var}(n_j)}$

The optimization was solved using a grid search with a step size of $0.05$ over the simplex, followed by cross-validation using the resulting quintile-bin population balance. The target was that no bin contain more than $40\%$ or less than $5\%$ of $m$.

## 3. Fuzzy Formal Concept Analysis (Core Methodological Contribution)

### 3.1 Motivation

Base paper uses **binary** FCA: each RFM(S) score band becomes a crisp binary attribute (e.g., $M5$ = "monetary score exactly 5"). This creates a hard boundary artifact — a customer at the 79.9th percentile (score 4) and one at 80.1st (score 5) are treated as categorically different, losing information the base paper's own future-work section flags as a limitation ("non-binary formal context... in future studies").

**This paper's primary contribution: implement and validate that future work — Fuzzy FCA (L-fuzzy context) applied to RFMS.**

### 3.2 Fuzzy Formal Context

Define fuzzy formal context $\mathbb{K}_f := (G, M, L, \tilde{I})$ where://
- $G = C$ (objects = customers)
- $M = \{R_1..R_5, F_1..F_5, M_1..M_5, S_1..S_5\}$ (20 fuzzy attributes — 5 bands × 4 dimensions)
- $L = [0,1]$ (membership lattice)
- $\tilde{I}: G \times M \to [0,1]$ fuzzy incidence relation

Membership computed via a triangular fuzzy membership function per band $k$ (soft quintile boundary, width parameter $w$):

$$\mu_k(x) = \max\left(0,\ 1 - \frac{|x - c_k|}{w}\right)$$

where $c_k$ is the band-$k$ centroid (e.g., median of that quintile in raw-score space) and $w$ is the inter-band spacing. This produces smooth membership decay near quintile boundaries instead of the base paper's step function, directly resolving the hard-cutoff artifact.

### 3.3 Fuzzy Derivation Operators

For fuzzy context, Galois connection generalizes via a fuzzy implication operator (Łukasiewicz t-norm $\otimes$, residuum $\to$):

$$A^{\uparrow}(m) = \bigwedge_{g \in G} \big(A(g) \to \tilde{I}(g,m)\big), \qquad B^{\downarrow}(g) = \bigwedge_{m \in M} \big(B(m) \to \tilde{I}(g,m)\big)$$

A fuzzy formal concept is a pair $(A,B)$, $A: G\to[0,1]$, $B: M\to[0,1]$, s.t. $A^{\uparrow}=B$ and $B^{\downarrow}=A$. Ordering:

$$(A_1,B_1) \leq (A_2,B_2) \iff A_1 \subseteq A_2 \iff B_2 \subseteq B_1$$

Implemented via the **fuzzy Next-Closure algorithm** (Belohlavek's extension of Ganter's algorithm) — see Algorithm 1.

### 3.4 Algorithm 1 — Fuzzy Concept Lattice Generation

```
Input: Fuzzy context K_f = (G, M, L, I~), threshold ε (min membership to consider)
Output: Set of fuzzy formal concepts L_f

1. Initialize B_0 = empty fuzzy set over M
2. repeat
3.     compute A = B_0↓  (fuzzy extent from current intent)
4.     compute B = A↑    (fuzzy closure)
5.     if B is lectically smallest fuzzy-closed set > B_0:
6.         add concept (A, B) to L_f
7.     B_0 = next lectic fuzzy closure via fuzzy-NextClosure(B_0, ε)
8. until B_0 = M (top reached)
9. return L_f
```//

Complexity: $O(|M|^2 \cdot |G| \cdot |L_f|)$ — same asymptotic order as crisp Next-Closure but with fuzzy membership evaluation cost per step; empirically bounded by discretizing $L$ into a finite fuzzy scale (e.g., 11 levels: 0, 0.1, ..., 1.0) to keep tractable, per Belohlavek & Vychodil (2005).

### 3.5 Iceberg Pruning (Secondary Contribution — replaces base paper's ad hoc 0.04 threshold)

Base paper prunes 208 concepts down to a working set using a manually chosen "support > 0.04" cutoff with no justification. This paper replaces it with a **stability-index-based iceberg lattice**:

$$\text{Stab}(A,B) = \frac{|\{A' \subseteq A : A'^{\uparrow} = B\}|}{2^{|A|}}$$

Stability measures how robust a concept is to removal of objects from its extent — concepts with low stability are noise-sensitive and pruned. Threshold $\theta$ selected via elbow method on the stability distribution (not arbitrary):

$$\theta^* = \arg\max_\theta \left| \frac{d}{d\theta}\left(\text{rank}(\text{Stab} > \theta)\right) \right|$$

Retain only concepts with $\text{Stab}(A,B) \geq \theta^*$ **and** $\text{supp}(A,B) \geq \text{supp}_{min}$ (support still applied as secondary filter, but now $\text{supp}_{min}$ is derived, not asserted — set via the same elbow method on the support distribution).

## 4. Cross-Domain Generalizability Protocol

To validate the base paper's untested generalizability claim, the identical RFMS-FCA pipeline (§1–3) is run independently on:
1. **Olist** (Brazilian marketplace, multi-seller, sparse-frequency, has satisfaction data)
2. **Online Retail II** (UK single-retailer, repeat-buyer-heavy, no satisfaction data — S dimension dropped, reduces to RFM only for this run)

Comparison metrics (domain-invariant):
- Concept lattice density: $\rho = |L_f| / 2^{|M|}$
- Degree of hierarchy overlap: mean number of concepts a customer belongs to, $\bar{k} = \frac{1}{m}\sum_j |\{(A,B) \in L_f : A(c_j) > 0\}|$
- Segment stability across domains via Adjusted Rand Index (ARI) between FCA-derived hard clusters (via $\alpha$-cut at $\alpha=0.5$) and each dataset's respective K-means baseline

## 5. Quantitative Benchmarking (Non-Negotiable Rigor Fix)

Base paper's core weakness: FCA output evaluated only narratively; K-means/hierarchical evaluated with Silhouette + Davies-Bouldin. This paper fixes the asymmetry.

### 5.1 Converting fuzzy concepts to comparable hard clusters

Apply $\alpha$-cut at $\alpha = 0.5$ to top-level concepts (those directly below $\top$) to derive crisp cluster assignment: $c_j \in \text{Cluster}_k \iff A_k(c_j) \geq 0.5$. Customers satisfying multiple cuts assigned to the concept of maximum membership (mode assignment) for the purpose of computing standard internal indices only (overlapping structure preserved separately for business interpretation).

### 5.2 Metrics computed identically across FCA, K-means, hierarchical

$$\text{Silhouette}(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \qquad \text{DB} = \frac{1}{k}\sum_{i=1}^k \max_{j\neq i}\left(\frac{\sigma_i + \sigma_j}{d(c_i,c_j)}\right)$$

computed on the same standardized RFMS feature space $(r_j, f_j, \mu_j, s_j)$ for all three methods, at $k = 4,5,6$ matching base paper's comparison points.

### 5.3 Additional fuzzy-specific validity metric

Since FCA method is uniquely capable of soft assignment, also report **Fuzzy Partition Coefficient**:

$$\text{FPC} = \frac{1}{m}\sum_{j=1}^m \sum_{k=1}^K \mu_{jk}^2$$

where $\mu_{jk}$ is customer $j$'s membership in concept $k$ (from $A_k(c_j)$, normalized to sum to 1 across top-level concepts) — this metric has no crisp-clustering equivalent, giving FCA a dimension of evaluation the baselines structurally cannot be scored on, strengthening the "richer than K-means" claim with a number, not just narrative.

## 6. Algorithm 2 — Full Pipeline

```
Input: Olist raw tables (orders, order_items, payments, reviews, customers)
Output: Fuzzy concept lattice, hard-cluster benchmark results, cross-domain ARI

1.  Join orders ↔ customers via customer_unique_id (NOT customer_id)
2.  Filter: remove orders with status ≠ 'delivered'; remove nulls in payment_value
3.  Compute per-customer R, n_j, {q_ij}, M_j, S_j
4.  Compute F*_j via Eq. (weights from §2.3 grid search)
5.  Compute dense-rank scores r_j, f_j, μ_j, s_j  (§2.2)
6.  Build fuzzy context K_f with triangular membership (§3.2)
7.  Run Algorithm 1 → L_f (fuzzy concept lattice)
8.  Compute stability + support for all concepts in L_f
9.  Derive θ*, supp_min via elbow method (§3.5); prune → L_f'
10. Build hierarchical Hasse diagram from L_f'
11. α-cut (α=0.5) top-level concepts → hard cluster assignment
12. Run K-means (k=4,5,6) and agglomerative hierarchical (k=4,5,6) on same
    standardized (r,f,μ,s) space
13. Compute Silhouette, DB index for all three methods at each k
14. Compute FPC for fuzzy method (no crisp equivalent)
15. Repeat steps 1–14 on Online Retail II (drop S dimension) for cross-domain run
16. Compute ARI between Olist FCA hard clusters and Olist K-means clusters;
    repeat for Online Retail II
17. Report comparative table (extends base paper's Table 10)
```

## 7. Expected Contribution Summary (for framing in Introduction/Contribution list)

1. First fuzzy (non-binary) FCA implementation for RFM(S)-based segmentation — resolves limitation explicitly flagged as future work in prior literature.
2. First RFMS variant incorporating post-purchase satisfaction as a formal concept dimension, motivated by marketplace review data unavailable in prior RFM-FCA studies.
3. Principled, data-driven concept lattice pruning (stability-index elbow method) replacing arbitrary support thresholds used in prior work.
4. First cross-domain validation of RFM-FCA segmentation (single-retailer vs multi-seller marketplace), with domain-invariant lattice metrics.
5. First symmetric quantitative benchmark (Silhouette/DB/FPC) placing FCA-based segmentation on equal evaluative footing with K-means/hierarchical baselines.
6. Methodological fix for frequency sparsity in marketplace data (composite $F^*$ + dense-rank scoring) — directly addresses a structural limitation of applying classical RFM to one-time-buyer-dominated platforms.

## 8. Suggested Target Venues

Expert Systems with Applications, Journal of Retailing and Consumer Services, Information Sciences, Knowledge-Based Systems — all precedented in base paper's own reference list, all publish FCA/clustering + marketing analytics work.

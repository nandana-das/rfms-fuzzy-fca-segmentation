# Final Methodology and Contribution Framing (Methodology v4)

## 1. Research Positioning & Lineage

This study builds directly upon the foundational RFM-FCA customer segmentation framework established by:
> **Chongkolnee Rungruang, Pakwan Riyapan, Arthit Intarasit, Khanchit Chuarkham, and Jirapond Muangprathub (2024)**  
> *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*,  
> *Expert Systems with Applications*, 237, 121449.

The base paper established the integration of Recency, Frequency, and Monetary (RFM) quintile analysis with Formal Concept Analysis (FCA) on transaction data, proving that concept lattices generate hierarchical, overlapping customer profiles.

In their concluding section, Rungruang et al. explicitly called for representing RFM values in a **non-binary / fuzzy formal context** to overcome the boundary sensitivity of crisp quintile discretization. This study directly executes that research agenda while addressing key structural limitations in concept proliferation, pruning, and predictive evaluation.

The study is strictly positioned as an **extension, refinement, and rigorous out-of-sample evaluation** of the RFM-FCA knowledge-discovery framework. It does not claim to introduce RFM-FCA itself.

---

## 2. Dataset Scope (Locked Methodology v4)

The final research study is evaluated across two repeat-rich transaction domains:
1. **Primary Validation Domain:** **Dunnhumby "The Complete Journey"** ($N = 2,499$ households, 276,484 baskets, 2,595,732 transaction records, 99.68% repeat rate).
2. **Independent Second Domain:** **Online Retail II** ($N = 5,878$ clean customers, 1,067,371 rows, 72.39% repeat rate).

### Formal Scope Boundaries:
- **Strictly RFM Only:** Feature representations, concept lattices, clustering benchmarks, and predictive models use exclusively Recency ($R$), Frequency ($F$), and Monetary ($M$).
- **Excluded / Out of Scope:** Customer Satisfaction ($S$), $F^*$, marketplace frequency sparsity experiments, and the single-order Olist marketplace are historical exploratory investigations that are excluded from the final paper.

---

## 3. Central Research Claim

Building upon the RFM-FCA framework of Rungruang et al. (2024), this work replaces crisp binary quintiles with a centroid-based piecewise-linear fuzzy RFM representation, introduces data-driven concept reduction and greedy extent-Jaccard redundancy suppression to control concept proliferation, and validates the resulting customer representation through canonical fuzzy clustering benchmarks and leakage-free temporal predictive evaluations across two independent repeat-transaction domains.

---

## 4. Validated Methodological Contributions

### Contribution 1: Centroid-Based Fuzzy RFM Formal Context
Extends crisp quintile discretization to a continuous fuzzy representation:
- Band centroids $c_1 < c_2 < c_3 < c_4 < c_5$ are computed as the medians of raw feature values within score quintiles.
- Piecewise-linear membership functions with outer shoulder saturation eliminate artificial boundary discontinuities.
- Row-sums per dimension strictly equal 1.0 ($\sum_{k=1}^5 \mu_k(X_j) = 1.0$).
- $\mathcal{L}$-fuzzy threshold scaling at $\{0.3, 0.5, 0.7\}$ enables uncapped closed frequent itemset mining (`max_len = None`).

### Contribution 2: Data-Driven Concept Reduction
Replaces arbitrary manual support cutoffs with an automated, data-driven pruning procedure:
- Uses a **Kneedle-inspired elbow detection algorithm** (normalised max-chord-distance) on support and object-profile stability-proxy distributions.
- Filters low-support and unstable noise concepts while preserving core lattice geometry.

### Contribution 3: Greedy Extent-Jaccard Redundancy Suppression
Resolves the severe concept proliferation and near-duplicate concept problem inherent to multi-level fuzzy scalings:
- Applies a greedy extent-Jaccard filter ($J_{\max} = 0.80$) on core concept extents ($\mu \ge 0.5$).
- Achieves 4×–5× concept compression (Dunnhumby: 502 $\to$ 123 concepts; Retail II: 445 $\to$ 95 concepts) with zero loss of downstream predictive utility.

### Contribution 4: Principled Fuzzy Benchmarking & Hardening Separation
Clarifies the methodological distinction between geometric clustering and conceptual lattice closure:
- Implements canonical **Fuzzy C-Means (FCM, $m=2.0$)** as a structurally matched fuzzy benchmark, recording Fuzzy Partition Coefficient (FPC) and Xie-Beni index (scoped strictly to FCM).
- Distinguishes **Alpha-cut** (thresholding continuous membership at $\alpha \ge 0.5$ to define extents) from **Top-$k$ Membership Hardening** (assigning customers via $\operatorname{argmax}$ over the $k$ highest-support fuzzy concepts).
- Evaluates hard partitions (K-Means, Ward, FCM argmax, and FCA hardening) in the common standardized Raw RFM space ($X_{\text{raw}}$).

### Contribution 5: Strictly Leakage-Free Temporal Predictive Validation
Establishes a rigorous predictive evaluation protocol:
- Pre-cutoff transactions strictly define observation features; post-cutoff transactions strictly define future holdout targets (repurchase, future spend, future invoice volume).
- Scalers, dense-rank cutoffs, fuzzy centroids, concept mining, suppression, and regression models are fitted strictly on training folds.
- Evaluates performance across single fixed holdouts and 10 independent temporal cross-validation splits with 1,000 paired bootstrap resamples.

### Contribution 6: Cross-Domain Repeat-Purchasing Validation
Validates the framework across two distinct repeat-purchasing consumer settings:
- Primary Domain: High-frequency household grocery supermarket purchasing (Dunnhumby, 99.68% repeat rate, mean 110.6 baskets).
- Second Domain: Non-store giftware retail (Online Retail II, 72.39% repeat rate).
- Demonstrates that fuzzy concept representations yield substantial, statistically significant predictive gains on future spend and invoice volume in repeat-transaction domains.

---

## 5. Explicit Claims to Avoid

To maintain scientific integrity and prevent reviewer objections, the paper must **NOT** claim:
1. That RFM-FCA itself is a new contribution (established by Rungruang et al., 2024).
2. That hierarchical or overlapping concept lattices were invented by this study.
3. That Fuzzy FCA universally outperforms K-Means, Ward, or FCM in geometric compactness. (FCA optimizes conceptual intent closure, whereas K-Means optimizes minimum Euclidean distance; they serve different analytical purposes).
4. That the elbow heuristic is canonical Kneedle (it is a Kneedle-inspired normalized max-chord-distance heuristic).
5. That the stability metric is canonical Kuznetsov stability (it is an object-profile diversity stability proxy).
6. That representation engineering solves physical frequency sparsity.
7. Any claims based on Olist, RFMS, Satisfaction ($S$), $F^*$, or legacy leaked holdout metrics.

---

## 6. One-Sentence Paper Summary

Building upon the RFM-FCA framework of Rungruang et al. (2024), this study introduces a centroid-based fuzzy RFM representation with data-driven concept reduction and redundancy suppression, demonstrating statistically significant predictive utility and structural interpretability across two independent repeat-transaction domains.

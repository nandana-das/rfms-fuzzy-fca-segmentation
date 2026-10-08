> **SUPERSEDED (2026-10-08).** Historical copy of `docs/final_methodology_contribution_framing.md` as last committed (git HEAD 8a420e2), before the audit. It reports dense-rank scoring, an untransformed Raw RFM baseline, an incorrect 98.08% repurchase rate and "temporal CV" labels. Do not cite. Current documents: `README.md`, `docs/RESULTS_SUMMARY.md`; errata: `docs/AUDIT_ERRATA.md`.

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

## 3. Central Research Claim (Locked Contribution Framing)

Building on Rungruang et al.’s RFM-FCA hierarchical segmentation framework, this study extends the binary RFM formal context to a fuzzy RFM representation, introduces data-driven concept pruning/redundancy suppression, evaluates canonical FCM alongside K-means/Ward, and validates across two domains using shared geometric and predictive protocols.

---

## 4. Primary Research Gaps Completed & Supporting Extensions

### Primary Research Gap 1: Binary/Crisp to Centroid-Based Fuzzy RFM Formal Context
Extends crisp quintile discretization to a continuous fuzzy representation:
- Band centroids $c_1 < c_2 < c_3 < c_4 < c_5$ are computed as medians of raw feature values within score quintiles.
- Piecewise-linear membership functions with outer shoulder saturation reduce the hard boundary discontinuities introduced by crisp quintile discretization, while retaining structural design choices (five behavioral bands, centroid construction, and $\mathcal{L}$-fuzzy threshold cuts).
- Row-sums per dimension strictly equal 1.0 ($\sum_{k=1}^5 \mu_k(X_j) = 1.0$).
- Baseline $\mathcal{L}$-fuzzy threshold scaling at $\{0.3, 0.5, 0.7\}$ enables uncapped closed frequent itemset mining (`max_len = None`).

### Primary Research Gap 2: Fixed Support-Based Concept Filtering to Data-Driven Knee-Based Pruning Heuristic
The Kneedle-inspired heuristic provides a data-driven secondary pruning criterion over the mined candidate concepts, complementing the minimum support threshold used during concept generation:
- The base paper uses a fixed support threshold of 0.04 and does not report a data-driven procedure for selecting this threshold.
- Closed fuzzy concepts are first generated subject to the minimum support criterion (0.04). The Kneedle-inspired normalized max-distance-from-chord heuristic then provides a data-driven secondary pruning criterion based on the observed support and stability-proxy distributions.
- Filters low-support and unstable noise concepts while preserving core lattice geometry.

### Primary Research Gap 3: Fuzzy Concept Proliferation to Extent-Level Jaccard Redundancy Suppression
Mitigates concept proliferation by suppressing near-duplicate concepts under the specified $J_{\max} = 0.80$ criterion:
- Applies a greedy extent-Jaccard filter ($J_{\max} = 0.80$) on core concept extents ($\mu \ge 0.5$).
- Achieves 4×–5× concept compression (Dunnhumby: 502 $\to$ 123 concepts; Retail II: 445 $\to$ 95 concepts), with downstream predictive performance remaining comparable to the unsuppressed representation under the evaluated protocol.

### Primary Research Gap 4: Single-Domain Evaluation to Independent Cross-Domain Predictive Validation
Provides evidence of cross-domain utility across two retail transaction datasets with different purchasing regimes:
- Primary Domain: High-frequency household grocery supermarket purchasing (Dunnhumby, 99.68% repeat rate, mean 110.6 baskets).
- Second Domain: Non-store giftware retail (Online Retail II, 72.39% repeat rate).
- Demonstrates that fuzzy concept representations yield substantial predictive gains on future spend and invoice volume in repeat-transaction domains, confirmed via paired bootstrap testing ($p < 0.001$) on the fixed temporal split and paired split win-rates across 10 temporal splits. The larger regression gains observed on Dunnhumby suggest that transaction-dense purchasing histories may provide more information for fuzzy concept representations, although broader validation is required to establish this relationship.

---

### Supporting Methodological Extensions:

#### Supporting Extension A: Leakage-Free Temporal Predictive Validation
- Rigorous predictive evaluation protocol forecasting future customer holdout outcomes (repurchase, future spend, future invoice volume).
- Scalers, dense-rank cutoffs, fuzzy centroids, concept mining, suppression, and regression models are fitted strictly on training folds.
- Evaluated across single fixed holdouts and 10 independent temporal cross-validation splits (seeds 1000–1009) with 1,000 paired bootstrap resamples.

#### Supporting Extension B: Principled Fuzzy Benchmarking & Hardening Separation
- Implements canonical **Fuzzy C-Means (FCM, $m=2.0$)** as a structurally matched fuzzy benchmark, recording Fuzzy Partition Coefficient (FPC) and Xie-Beni index (scoped strictly to FCM).
- Distinguishes **Alpha-cut** (thresholding continuous membership at $\alpha \ge 0.5$ to define extents) from **Top-$k$ Membership Hardening** (assigning customers via $\operatorname{argmax}$ over the $k$ highest-support fuzzy concepts).
- Evaluates hard partitions (K-Means, Ward, FCM argmax, and FCA hardening) in the common standardized Raw RFM space ($X_{\text{raw}}$).

#### Supporting Extension C: Sensitivity Analysis of Fuzzy Threshold Configuration
- Evaluates one-factor sensitivity of the $\mathcal{L}$-fuzzy threshold tuple on Dunnhumby under the locked evaluation protocol across three configurations: $(0.2, 0.5, 0.8)$ [permissive], $(0.3, 0.5, 0.7)$ [baseline], and $(0.4, 0.5, 0.6)$ [tight].
- **Key Finding:** The predictive relationship between fuzzy and crisp RFM-FCA remains consistent across the tested threshold configurations. Although the threshold tuple substantially changes the size and compression of the fuzzy concept space ($38$ to $265$ suppressed concepts; $2.46\times$ to $11.05\times$ compression), downstream predictive performance varies comparatively modestly and remains consistently above the crisp RFM-FCA baseline across the tested configurations ($10/10$ split wins on future invoices, $9/10$ to $10/10$ on future spend, higher mean AUC across all configurations).
- Establishes that the fuzzy representation's predictive utility is not dependent on narrow post-hoc parameter tuning.

---

## 5. Explicit Claims to Avoid

To maintain scientific integrity and prevent reviewer objections, the paper must **NOT** claim:
1. That RFM-FCA itself is a new contribution (established by Rungruang et al., 2024).
2. That hierarchical or overlapping concept lattices were invented by this study.
3. That Fuzzy FCA universally outperforms K-Means, Ward, or FCM in geometric compactness. (FCA optimizes conceptual intent closure, whereas K-Means optimizes minimum Euclidean distance; they serve different analytical purposes).
4. That negative silhouette values prove clustering failure or superiority; the negative silhouette values are consistent with the interpretation that forcing overlapping lattice concept memberships into mutually exclusive Euclidean partitions can impose substantial boundary penalties.
5. That the elbow heuristic is canonical Kneedle (it is a Kneedle-inspired normalized max-chord-distance heuristic).
6. That the stability metric is canonical Kuznetsov stability (it is an object-profile diversity stability proxy).
7. That redundancy suppression proves removed concepts were useless in isolation; it demonstrates that near-duplicate concepts can be suppressed without degrading downstream predictive performance under the evaluated protocol.
8. That exact customer counts from the base paper can be uniquely reconstructed; the omitted tie-breaking rule for identical frequency values (notably the 1,623 customers with $F=1$) prevents uniquely determining the original allocation.
9. That the threshold configuration was optimized or tuned post-hoc based on test results (the study is strictly a sensitivity analysis, not model selection).
10. That the baseline threshold tuple is mathematically optimal.
11. That representation engineering solves physical frequency sparsity or generalizes universally to all e-commerce settings.
12. Any claims based on Olist, RFMS, Satisfaction ($S$), $F^*$, or legacy leaked holdout metrics.

---

## 6. One-Sentence Paper Summary

Building upon the RFM-FCA framework of Rungruang et al. (2024), this study introduces a centroid-based fuzzy RFM representation with data-driven concept reduction and redundancy suppression, demonstrating consistent predictive utility and structural interpretability across two independent repeat-transaction domains, with regression gains confirmed by paired bootstrap testing on fixed holdouts and consistent wins across 10 temporal cross-validation splits.

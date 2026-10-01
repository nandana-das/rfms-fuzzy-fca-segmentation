# Systematic Contribution Audit: RFMS-Fuzzy FCA Project vs. Base Paper

**Base Paper:** Rungruang et al., *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*, *Expert Systems with Applications* (2024) / SSRN: 4416218.  
**Audited Repository:** [rfms-fuzzy-fca-segmentation](https://github.com/nandana-das/rfms-fuzzy-fca-segmentation)  
**Date:** September 2026 / October 2026 (Methodology v4 Update)  
**Status:** Problem 4 Deliverable D — Definitive Lineage and Contribution Assessment  
**Final Scope Note:** Under locked **Methodology v4**, the final paper is strictly **RFM only**, evaluating **Dunnhumby "The Complete Journey"** (Primary Domain) and **Online Retail II** (Second Domain). The 25-dimension audit below documents the direct execution of Rungruang et al.'s non-binary future work, while exploratory RFMS/Olist extensions represent historical investigations preserved for transparency.

---

## 1. Executive Summary & Lineage Foundation

A rigorous comparison between our repository and Rungruang et al. (2024) reveals a clear and direct evolutionary relationship. 

### Critical Context from Base Paper:
1. **RFM + FCA is NOT our contribution:** The combination of classical RFM quintile segmentation with Formal Concept Analysis (FCA) to generate a hierarchical concept lattice was established by Rungruang et al. (using `fcaR` in R).
2. **Overlapping Customer Segmentation is NOT our contribution:** The base paper explicitly defines concepts as overlapping groups where customers belong to multiple concepts simultaneously (Section 4.3, Table 10).
3. **Fuzzy / Non-Binary FCA is DIRECTLY PROPOSED by the Base Paper:** In Section 6 (Conclusions and Future Work, Page 17), the authors explicitly state:
   > *"In addition, we will modify and improve this model by representing RFM values in a non-binary formal context in future studies."*
   Furthermore, the authors note:
   > *"In future work, we will also consider other customer segmentation models that are extended from RFM models, such as LRFM, RFMT, and RFMTC models..."*

Therefore, our work does **not** invent fuzzy customer concept analysis in a vacuum; rather, it directly executes the foundational future work mapped out by Rungruang et al., addressing both the non-binary context representation and the variable extension.

---

## 2. Granular Audit of Claimed Contributions

Each of the 12 technical and empirical components of our repository is audited below against the base paper, establishing its exact taxonomy classification, technical relationship, and defensible scope.

```
Taxonomy Classes:
1. Already present in base paper
2. Direct extension of base paper
3. Methodological modification
4. New evaluation
5. New analysis
6. Cannot establish novelty from this paper alone
```

---

### Item 1: RFMS Instead of RFM (Satisfaction Dimension)
- **Classification:** **Direct extension of base paper** (dataset-motivated variable extension)
- **Base Paper Reality:** Employs solely classical 3-variable $R, F, M$ on Online Retail II. In Section 6 (Page 17), the authors explicitly suggest extending RFM to include additional customer dimensions (citing LRFM, RFMT, RFMTC).
- **Our Implementation:** Adds a 4th dimension, Satisfaction ($S$), calculated from customer review scores ($S \in [1, 5]$) on the Olist marketplace dataset. Imputes missing review scores (0.65%) with observed median (5.0).
- **Lineage / Defensibility:** Defensible as a domain-motivated realization of the base paper's suggested variable expansion. However, RFMS itself exists in broader customer analytics literature; our novelty is strictly integrating $S$ into a multi-attribute Formal Concept Analysis lattice.

---

### Item 2: Fuzzy / Non-Binary FCA Representation
- **Classification:** **Direct extension of base paper** (realizing base paper Section 6 future work)
- **Base Paper Reality:** Uses crisp binary attributes: 5 crisp quintile bins per dimension ($R1..R5, F1..F5, M1..M5$) producing a binary formal context $K = (G, M, I)$ with $|M|=15$. Section 6 explicitly states: *"we will modify and improve this model by representing RFM values in a non-binary formal context in future studies."*
- **Our Implementation:** 
  1. Computes cluster centroids per crisp band.
  2. Applies centroid-based piecewise-linear membership functions to assign each customer continuous degrees $\mu \in [0, 1]$.
  3. Applies multi-level threshold $\mathcal{L}$-fuzzy scaling ($\theta \in \{0.3, 0.5, 0.7\}$) followed by closed frequent itemset mining via FP-growth.
- **Lineage / Defensibility:** Fully defensible as the direct execution of Rungruang et al.'s proposed future work. It must be credited as resolving their stated limitation, not claimed as an unanticipated discovery.

---

### Item 3: $F^*$ Purchase-Intensity Index
- **Classification:** **Methodological modification**
- **Base Paper Reality:** Uses raw integer transaction frequency $F \in [1, 411]$ on Online Retail II (where 72.4% of customers are repeat buyers). Rank-first quintile scoring yields 5 non-degenerate bins.
- **Our Implementation:** On Olist, 97.0% of customers purchase exactly once ($F=1$). Classical quintile discretization collapses into an extreme degenerate spike. We formulated:
  $$F^* = \alpha \, n_{\text{orders}} + \beta \, \ln(1 + \text{item\_rows}) + \gamma \, \mathbf{1}_{[\text{repeat}]}$$
  with weights ($\alpha=0.20, \beta=0.05, \gamma=0.75$) selected via grid search to maximize the Shannon entropy of the 5 discretized score bands.
- **Lineage / Defensibility:** Defensible as an engineering heuristic for sparse e-commerce platforms. However, per Problem 1 findings, $F^*$ is a composite purchase-intensity proxy, not literal purchase frequency, and it does not eliminate the physical 97% single-order boundary.

---

### Item 4: Data-Driven / Kneedle-Inspired Pruning
- **Classification:** **Methodological modification**
- **Base Paper Reality:** Applies a hardcoded, arbitrary minimum support cutoff of $\text{supp} > 0.04$ (235 customers) to reduce 208 mined concepts down to 31 frequent concepts (Table 6, Table 7). No derivation or mathematical rule is provided.
- **Our Implementation:** Employs the automated Kneedle elbow/knee detection algorithm on the sorted support curve and stability proxy curve to derive data-driven thresholds $\text{supp}_{\min}^*$ and $\theta^*$ systematically without manual hand-tuning.
- **Lineage / Defensibility:** Defensible methodological contribution that eliminates arbitrary hyperparameter selection in FCA concept selection.

---

### Item 5: Stability Proxy
- **Classification:** **Methodological modification**
- **Base Paper Reality:** Concepts are selected purely by support threshold ($\text{supp} > 0.04$). Concept stability (intensional or extensional resilience to data perturbation) is completely absent.
- **Our Implementation:** Implements a tractable score-profile diversity proxy:
  $$\text{Stability Proxy} = 1 - \frac{\text{unique score profiles in extent}}{\text{extent size}}$$
- **Lineage / Defensibility:** Defensible heuristic diagnostic. Crucially, it must **not** be confused with or claimed as Kuznetsov's canonical formal concept stability (which has exponential complexity); it depends on the chosen discretization.

---

### Item 6: Redundancy Suppression (Fuzzy Multi-Level Jaccard)
- **Classification:** **Methodological modification**
- **Base Paper Reality:** Generates concepts using single-level binary attributes ($R1..R5$). Redundancy across multi-cut thresholds does not exist in crisp binary FCA.
- **Our Implementation:** Multi-level fuzzy scaling ($\theta \in \{0.3, 0.5, 0.7\}$) generates nested, redundant concept variants (e.g., identical extents differing only by attribute cut level). We implemented greedy Jaccard extent-overlap suppression ($\text{IoU} \ge 0.85$ or $0.90$) to eliminate duplicate profile concepts.
- **Lineage / Defensibility:** Essential technical necessity arising specifically from multi-level fuzzy scaling. Defensible as a required algorithmic component of scalable fuzzy FCA.

---

### Item 7: FCM (Fuzzy C-Means) Benchmark
- **Classification:** **New evaluation**
- **Base Paper Reality:** Compares FCA only against crisp partition baselines: K-means and Ward agglomerative hierarchical clustering. Fuzzy clustering algorithms are completely unbenchmarked.
- **Our Implementation:** Introduces canonical Fuzzy C-Means (FCM) using the standardized continuous feature space with fuzzifier $m=2.0$, evaluating both hard partition metrics (Silhouette, DB) and fuzzy membership quality (FPC).
- **Lineage / Defensibility:** Substantive methodological improvement. As demonstrated in Problem 2, comparing fuzzy FCA against K-means creates a structural representation mismatch; benchmarking against FCM provides a mathematically valid fuzzy-to-fuzzy baseline.

---

### Item 8: Predictive Validation (Out-of-Sample Holdout)
- **Classification:** **New evaluation**
- **Base Paper Reality:** Entirely descriptive and unsupervised. Employs zero out-of-sample or temporal predictive validation. Segments are evaluated solely by visual inspection, revenue concentration (Table 7), and unsupervised clustering indices (Figures 9 & 10).
- **Our Implementation:** Formulated a rigorous predictive validation protocol: segment representations (cluster assignments or fuzzy membership features) fit on an observation window are used to predict downstream customer retention (AUC), future spend ($R^2$), and future order count ($R^2$).
- **Lineage / Defensibility:** Outstanding scientific contribution. Unsupervised clustering papers rarely prove that discovered clusters carry actionable forward predictive validity. However, per Problem 3, on Olist (sparse repeat rate), the predictive advantage collapses once temporal review leakage is removed; predictive superiority is observed on Retail II but not on Olist.

---

### Item 9: Cross-Domain Validation (Marketplace vs. Single Retailer)
- **Classification:** **New analysis**
- **Base Paper Reality:** Validated solely on a single dataset: UK Online Retail II (high-frequency gift/homeware wholesaler, 72.4% repeat buyers).
- **Our Implementation:** Executes the pipeline across two contrasting e-commerce domains:
  1. Online Retail II (B2B/B2C UK giftware retailer, high repeat frequency).
  2. Brazilian E-Commerce Olist (B2C marketplace, 97% single-order transactions, multi-seller logistics).
- **Lineage / Defensibility:** High-value empirical analysis demonstrating where FCA segmentation translates well and where sparsity causes structural boundary issues.

---

### Item 10: Frequency-Sparsity Ablation
- **Classification:** **New analysis**
- **Base Paper Reality:** Did not encounter frequency sparsity due to the high-repeat nature of Online Retail II.
- **Our Implementation:** Problem 1 controlled ablation: compared the baseline $F^*$ against alternative formulations (raw log-orders, log-quantity, transaction engagement, and inverse recency-frequency blends). Quantified Shannon band entropy, variance spread, and downstream clustering impact.
- **Lineage / Defensibility:** Defensible empirical contribution explaining why mathematical feature engineering cannot fabricate behavioral frequency when the underlying phenomenon is physical one-time buying.

---

### Item 11: Temporal Leakage Audit
- **Classification:** **New evaluation** (methodological hygiene)
- **Base Paper Reality:** No temporal validation was performed, hence no temporal leakage could occur.
- **Our Implementation:** Problem 3 audit uncovered that static aggregation of customer review scores in holdout validation leaked post-cutoff satisfaction. Replaced with strict pre-cutoff review aggregation, documenting the resulting AUC contraction ($0.6659 \to 0.5639$) honestly.
- **Lineage / Defensibility:** A critical publication-grade audit demonstrating rigorous scientific integrity and preventing the dissemination of contaminated results.

---

### Item 12: Overlap-Profile Analysis
- **Classification:** **New analysis**
- **Base Paper Reality:** Mentions overlapping customer properties qualitatively in Table 10 and provides concept revenue sums in Table 7, but does not analyze customer-level multi-concept membership signatures or explain the practical information discarded by hard clustering.
- **Our Implementation:** Systematically extracted the 218 distinct $\alpha$-cut top-level concept signatures; proved that 99.86% of customers possess overlapping memberships; conducted the "Hard-Cluster Collapse Test" showing that customers collapsed into a single crisp "F1" hard cluster span 111 distinct behavioral profiles across $R, M,$ and $S$.
- **Lineage / Defensibility:** Defensible and compelling demonstration of the operational utility of overlapping formal concepts over crisp partitions.

---

## 3. Summary Contribution Taxonomy Matrix

| # | Project Component | Taxonomy Classification | Relation to Base Paper |
|---|-------------------|-------------------------|------------------------|
| 1 | **RFMS Formulation** | Direct extension of base paper | Realizes base paper Section 6 future work on RFM variable expansion |
| 2 | **Fuzzy / Non-Binary FCA** | Direct extension of base paper | Realizes base paper Section 6 explicit future work on non-binary context |
| 3 | **$F^*$ Purchase-Intensity** | Methodological modification | Novel heuristic designed to handle marketplace frequency sparsity |
| 4 | **Kneedle Pruning** | Methodological modification | Replaces base paper's arbitrary 0.04 support threshold with data-driven elbow |
| 5 | **Stability Proxy** | Methodological modification | Adds concept profile cohesion diagnostic absent in base paper |
| 6 | **Redundancy Suppression** | Methodological modification | Algorithmic requirement arising from multi-level fuzzy scaling |
| 7 | **FCM Benchmark** | New evaluation | Introduces appropriate fuzzy-to-fuzzy baseline missing in base paper |
| 8 | **Predictive Validation** | New evaluation | Introduces forward holdout prediction missing in base paper's descriptive study |
| 9 | **Cross-Domain Validation** | New analysis | Tests base paper methodology across contrasting marketplace vs retail domains |
| 10 | **Frequency-Sparsity Ablation** | New analysis | Investigates boundary conditions of RFM segmentation on one-time buyers |
| 11 | **Temporal Leakage Audit** | New evaluation | Rigorous protocol ensuring clean out-of-sample validation |
| 12 | **Overlap-Profile Analysis** | New analysis | Micro-level proof of information preserved by FCA vs discarded by hard clustering |

---

## 4. Methodological Boundaries & Negative Findings

To maintain absolute scientific integrity, the following boundaries must be explicitly recognized in all reporting:
1. **FCA is Not Superior on Geometric Distance Metrics:** Crisp or fuzzy FCA hard-cut clusters do not beat K-means or Ward on Silhouette or Davies-Bouldin indices. This is expected because FCA optimizes Galois concept closure, not Euclidean variance minimization.
2. **Fuzzy FCA Predictive Advantage is Domain-Dependent:** On Online Retail II (repeat-buyer cohort), Fuzzy FCA features improve forward spend and invoice prediction ($R^2$) over raw RFM. On Olist (one-time buyer marketplace), leakage-free predictive AUC ($\approx 0.564$) and $R^2$ ($\approx 0.001$) show no statistically significant improvement over baseline models.
3. **F* Does Not Create Real Purchase Frequency:** On a 97% single-order platform, $F^*$ spreads customers across bands primarily via item-line counts and the repeat indicator; it does not turn one-time shoppers into repeat shoppers.

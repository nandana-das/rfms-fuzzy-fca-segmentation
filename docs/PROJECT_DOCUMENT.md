# RFMS-Fuzzy FCA Customer Segmentation: Research Gap, Methodology, Audited Problem Resolutions, and Experimental Findings

**Document Status:** Comprehensive Consolidated Document (Post-Audit & Problem 1–4 Resolution)  
**Base Paper:** Rungruang et al., *"RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA"*, *Expert Systems with Applications* (2024).  
**Repository:** [rfms-fuzzy-fca-segmentation](https://github.com/nandana-das/rfms-fuzzy-fca-segmentation)

---

## 1. Research Gap & Theoretical Lineage

Customer relationship management relies heavily on Recency, Frequency, and Monetary (RFM) segmentation. Rungruang et al. (2024) pioneered the combination of RFM quintile segmentation with Formal Concept Analysis (FCA) on the UK Online Retail II transaction dataset, establishing that concept lattices generate interpretable, overlapping customer profiles superior to mutually exclusive partitions.

Crucially, Rungruang et al. concluded their paper by explicitly highlighting key limitations and future research avenues:
1. **Crisp Binary Formal Context:** The base study uses hard quintile cuts (5 crisp bins per dimension, $|M|=15$). The authors noted: *"we will modify and improve this model by representing RFM values in a non-binary formal context in future studies."*
2. **Variable Restriction:** The base model utilizes only 3 traditional variables ($R, F, M$), noting that extended dimensions (e.g., satisfaction, customer tenure) should be explored.
3. **Ad Hoc Pruning:** Concepts were pruned using an arbitrary manual threshold ($\text{support} > 0.04$) with no derivation methodology.
4. **Single-Domain Validation:** Tested only on a repeat-purchase-heavy UK giftware retailer ($72.39\%$ repeat rate); generalizability to sparse-frequency multi-seller marketplaces was unexamined.
5. **Asymmetric Evaluation:** FCA was evaluated narratively, whereas K-means and Ward clustering were evaluated using geometric compactness metrics (Silhouette, Davies–Bouldin).
6. **Concept Proliferation & Redundancy:** Continuous or multi-level fuzzy scalings generate near-duplicate concepts covering almost identical customer subsets, causing severe feature explosion.

This research directly executes the future-work agenda proposed by Rungruang et al. while addressing these structural gaps.

---

## 2. Research Objectives

1. **Develop a Centroid-Based Fuzzy Formal Context:** Construct piecewise-linear fuzzy membership functions with outer shoulder saturation, converting continuous attributes into continuous membership degrees $\mu \in [0, 1]$ and eliminating hard boundary discontinuities.
2. **Extend the Representation to RFMS:** Incorporate customer **Satisfaction ($S$)** derived from transaction review ratings on the Olist Brazilian marketplace, creating an expanded 20-attribute formal context.
3. **Evaluate Marketplace Frequency Sparsity via Controlled Ablation:** Introduce an entropy-maximizing purchase-intensity index ($F^*$) and investigate through controlled ablation whether representation engineering can overcome intrinsic marketplace frequency sparsity (97% single-order customers).
4. **Implement Data-Driven Concept Reduction:** Replace manual thresholds with **Kneedle-inspired elbow detection** across support and stability proxy distributions.
5. **Develop Greedy Extent-Jaccard Redundancy Suppression:** Introduce a principled concept-deduplication pass ($J_{\max} = 0.80$) that eliminates near-duplicate concepts while preserving downstream predictive utility.
6. **Establish a Principled Fuzzy Clustering Benchmark:** Implement canonical **Fuzzy C-Means (FCM)** at matched cluster counts $k$, isolating whether predictive gains stem from fuzziness itself or from the hierarchical partial-order lattice structure.
7. **Ensure Leakage-Free Temporal Predictive Validation:** Formulate a strictly controlled out-of-sample prediction protocol, auditing and purging post-cutoff review leakage.
8. **Conduct Rigorous Cross-Domain Evaluation:** Evaluate the common-component pipeline across both Online Retail II ($N = 5,878$) and Olist ($N = 93,357$).

---

## 3. Methodology

### 3.1 Data Preparation & Feature Construction
- **Entity Resolution:** Customer transactions are grouped using persistent entity keys (`customer_unique_id` for Olist; `CustomerID` for Retail II), strictly avoiding order-scoped keys.
- **Sanitation:** Filters non-delivered orders, cancellations, missing IDs, and non-positive prices/quantities.
- **Feature Computation:**
  - Recency ($R$): Days from latest order to reference date.
  - Literal Frequency ($n$): Count of distinct delivered orders.
  - Monetary ($M$): Total payment value across orders.
  - Satisfaction ($S$, Olist only): Mean order-level review rating across orders. Missing review scores ($0.65\%$) are imputed using the observed median/mode ($S=5.0$, $+0.27\%$ mode impact).
- **Purchase-Intensity Heuristic ($F^*$):**
  $$F_j^* = \alpha \cdot n_j + \beta \cdot \sum_{i=1}^{n_j} \ln(1 + \text{item\_qty}_{ij}) + \gamma \cdot \mathbb{1}[n_j > 1]$$
  Weights are selected via simplex grid search maximizing the **Shannon entropy** of the 5-band discretized histogram. For Olist, optimal weights are $\alpha=0.20, \beta=0.05, \gamma=0.75$ ($H = 0.1457$ nats, $9.05\%$ of $\ln(5)$).
- **Dense-Rank Fractional Scoring:**
  $$\text{score}(X_j) = \left\lceil 5 \cdot \frac{\text{dense\_rank}(X_j)}{K_X} \right\rceil$$
  where $K_X$ is the total count of distinct raw values observed for feature $X$.

### 3.2 Centroid-Based Piecewise-Linear Fuzzy Context
- Compute band centroids $c_1 < c_2 < c_3 < c_4 < c_5$ as medians of raw feature values within each score band.
- Assign continuous memberships $\mu_k(X_j) \in [0, 1]$ via piecewise-linear interpolation between adjacent centroids, with outer shoulder saturation for bands 1 and 5. Row-sums per dimension strictly equal 1.0.
- Apply Belohlavek threshold $\mathcal{L}$-fuzzy scaling at cuts $\{0.3, 0.5, 0.7\}$ to construct scaled binary context $\mathbb{K}_L$.
- Enumerate closed frequent itemsets using uncapped FP-growth (`max_len=None`, $\text{min\_support}=0.02$).

### 3.3 Concept Reduction & Redundancy Suppression
- **Kneedle-Inspired Iceberg Pruning:** Automatically identifies elbows on support ($\text{supp}_{\min}^*$) and stability proxy ($\theta^*$), filtering noise concepts.
- **Greedy Extent-Jaccard Suppression:** Candidate concepts ordered by Support ($\downarrow$), Intent Size ($\uparrow$), and Stability ($\downarrow$) are accepted iff their $\mu \ge 0.5$ core extent has Jaccard similarity $< 0.80$ against all previously retained concepts.

### 3.4 Benchmarking & Predictive Validation Protocols
- **Geometric Partitions:** Top-level concepts converted to hard clusters via $\alpha$-cut ($\alpha=0.5$) and mode assignment; evaluated with Silhouette and Davies–Bouldin.
- **Fuzzy Clustering Baseline:** Canonical Fuzzy C-Means ($m=2.0$, K-means++ seeded) evaluated at matched $k$.
- **Downstream Prediction:** Feature representations evaluated on predicting holdout repeat purchase (AUC), spend ($R^2$), and invoices ($R^2$).
- **Strict Leakage-Free Protocol:** Pre-cutoff filtering requires both $\text{order\_purchase\_timestamp} \le T_{\text{cutoff}}$ and $\text{review\_answer\_timestamp} \le T_{\text{cutoff}}$. Score bands and centroids are fitted dynamically on training folds only.

---

## 4. Empirical Results & Diagnostics (Core Steps 1–13)

1. **Uncapped Lattice Discovery:** Removing the binding `max_len=6` restriction revealed 1,369 closed concepts for Olist (down from 10,283 spurious concepts) and 1,064 closed concepts for Online Retail II.
2. **Kneedle Iceberg Pruning:** Retained 153 stable concepts on Olist (304 Hasse covering edges) and 115 stable concepts on Retail II.
3. **Clustering Benchmarks (Olist):** At natural $k=12$, hard-assigned FCA achieves Silhouette $0.1656$ and DB $5.9467$. At matched $k=5$, FCA achieves Silhouette $0.2467$ and DB $2.3919$, compared to K-means (Silhouette $0.3867$, DB $0.8391$) and canonical FCM (Silhouette $0.3847$, DB $0.8445$, FPC $0.5005$).
4. **Multi-Membership Overlap:** Under $\alpha=0.5$, $99.86\%$ of Olist customers belong to multiple top-level concepts. Dominant hard cluster $F1$ ($N=84,199$) spans Recency 0 to 694 days, Monetary \$9.59 to \$13,664, and encompasses 111 distinct overlap signatures.
5. **Concept Persistence:** $100\%$ of the 153 pruned Olist concepts persist across 10 independent 80% subsamples.

---

## 5. Resolution of the Four Methodological Problems

### 5.1 Problem 1: Controlled Frequency Dimension Ablation Study
To investigate whether the composite $F^*$ formulation should be replaced, a strictly controlled ablation evaluated three variants on Olist ($N = 93,357$):
- **Variant A (Literal, $F_1$):** $n_{\text{orders}}$
- **Variant B (Current $F^*$, $F_2$):** $0.20\,n + 0.05\,\ln(1+\text{qty}) + 0.75\,\mathbb{1}[n>1]$
- **Variant C (Revised Engagement, $F_3$):** $0.5\ln(1+n) + 0.5\ln(1+\text{lines})$

**Key Findings:**
- In $F_3$, removing the discrete $+0.75$ repeat-purchase jump caused small-basket repeat buyers to collapse into band 1 alongside one-time buyers.
- Band 1 share worsened from $97.00\%$ to **$99.76\%$ ($93,129 / 93,357$)**, and Shannon entropy plummeted from $0.1457$ to **$0.0196$ nats** (an $86.6\%$ collapse).
- Downstream FCA clustering under $F_3$ degenerated into **negative Silhouette scores** ($-0.0694$ at natural $k$; $-0.0720$ at $k=5$).
- **Verdict:** $F_3$ is rejected. The current $F^*$ ($F_2$) is retained as an entropy-maximizing heuristic with explicit disclosure of its limitations.

---

### 5.2 Problem 2: Clustering Metric Mismatch Diagnosis & FCM Baseline
To explain the lower Silhouette/DB scores of hard-assigned FCA, a detailed geometric decomposition was conducted:
- **Root Cause:** $99.86\%$ of customers satisfy $\alpha \ge 0.5$ in multiple concepts. Converting overlapping fuzzy memberships to hard clusters forces boundary customers into artificial bins.
- **Decomposition:** In the 12-cluster hard FCA solution, **8 of the 12 clusters exhibit negative mean silhouette values** (ranging from $-0.125$ to $-0.222$). Only the dense $F1$ cluster achieves positive silhouette ($0.285$).
- **Resolution:** Geometric partition metrics evaluate Euclidean hyperspherical compactness, whereas FCA optimizes partial-order conceptual closure. Canonical FCM provides the proper fuzzy clustering benchmark, achieving Silhouette $0.3847$ and FPC $0.5005$ at $k=5$.

---

### 5.3 Problem 3: Temporal Review Leakage Audit & Elimination
An audit of the Olist temporal holdout revealed that aggregating reviews across all customer orders leaked post-cutoff satisfaction information into observation features:
- **Audit Impact:** 2,341 observation customers ($10.81\%$) had post-cutoff reviews; 1,003 customers ($4.63\%$) experienced numerical Satisfaction shifts; **$45.67\%$ of future repurchasers were contaminated**.
- **Retraction:** The leaked result showing inflated holdout AUC (~0.667) was permanently retracted.
- **Leakage-Free Multi-Split Results (10 Random Splits):**
  - **Online Retail II:** Fuzzy Suppressed FCA achieves mean AUC **$0.7857 \pm 0.0121$**, outperforming crisp FCA ($0.7768 \pm 0.0121$, $p = 2.3\times 10^{-4}$) and raw RFM ($0.7770 \pm 0.0117$, $p = 1.0\times 10^{-4}$) across 10/10 splits.
  - **Olist Marketplace:** Fuzzy Suppressed FCA achieves mean AUC **$0.5612 \pm 0.0125$**, outperforming re-derived crisp FCA ($0.5353 \pm 0.0153$, $p = 3.0\times 10^{-5}$) and FCM ($0.5517 \pm 0.0115$), but remaining on par with raw RFMS ($0.5587 \pm 0.0107$; raw wins in 4/10 splits).
- **Cross-Domain Insight:** Fuzzy FCA provides genuine predictive lift on repeat-purchase retail data, but collapses to near baseline on one-time-buyer marketplace data, demonstrating that representation engineering cannot override physical frequency sparsity.

---

### 5.4 Problem 4: Base-Paper Replication & Contribution Audit
Replication of Rungruang et al. (2024) on Online Retail II (`scripts/replicate_base_paper.py`):
- **Cohort Match:** Successfully matched the 5,878 clean customer cohort (Table 3).
- **Intent Recovery:** Recovered **all 31 published frequent concept intents** ($>0.04$ support).
- **Clustering Replication:** Replicated K-means and Ward clustering geometries across $k=2..10$ on 5,633 outlier-removed customers (Silhouette ~0.33–0.38, DB ~0.99–1.07).
- **Unstated Tie-Breaking:** Documented that minor customer count discrepancies arise from an unstated tie-breaking rule on frequency ($27.61\%$ of customers have $F=1$).
- **Lineage Verification:** Confirmed that non-binary/fuzzy FCA was explicitly proposed as future work by the base authors. Our work directly executes that proposed agenda.

---

## 6. Defensible Contributions & Paper Positioning

### What This Work Does NOT Claim:
- It does **not** claim to invent RFM + FCA (established by Rungruang et al., 2024).
- It does **not** claim that hierarchical or overlapping FCA segmentation is novel to this study.
- It does **not** claim that FCA universally outperforms geometric clustering algorithms.
- It does **not** claim that $F^*$ solves marketplace frequency sparsity.
- It does **not** claim the invalid leaked temporal AUC ~0.667 result.

### Validated Contributions:
1. **Fuzzy RFMS Extension:** Realizes the base paper's proposed future work by constructing a centroid-based fuzzy RFMS formal context, integrating customer satisfaction.
2. **Controlled Frequency Sparsity Analysis:** Introduces an entropy-maximizing purchase-intensity index ($F^*$) and provides empirical proof through controlled ablation of the limits of representation engineering under marketplace sparsity.
3. **Data-Driven Concept Reduction & Suppression:** Combines Kneedle-inspired elbow pruning with greedy extent-Jaccard redundancy suppression ($J_{\max}=0.80$), reducing Retail II concepts 4.7× (445 → 95) with zero predictive loss.
4. **Principled Fuzzy Benchmarking:** Introduces canonical FCM as a structurally matched fuzzy benchmark, separating geometric partition metrics from conceptual lattice evaluation.
5. **Leakage-Controlled Predictive Validation:** Establishes a rigorous, leakage-free temporal evaluation protocol, demonstrating the cross-domain contrast between repeat-purchase retail and sparse-frequency marketplaces.

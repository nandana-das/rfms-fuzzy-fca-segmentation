# Results

## 1. Base-Paper Replication on Online Retail II

The controlled replication reproduced the Online Retail II customer cohort used by the base RFM-FCA study, with 5,878 clean unique customers. The crisp arm uses the 15 binary quintile attributes R1–R5, F1–F5, and M1–M5.

At the base-paper support criterion (>0.04), the replication recovered all 31 published frequent concept intents. Exact customer counts for some concepts differ because the base paper does not fully specify the tie-breaking procedure for customers sharing identical frequency values.

The isolated K-means/Ward replication also reproduced the reported geometric clustering behavior over k=2–10, with Silhouette values in approximately the 0.33–0.38 range and Davies–Bouldin values approximately 0.99–1.07. These values are used as a replication of the geometric clustering branch rather than as FCA performance measures.

## 2. Crisp RFM-FCA versus Fuzzy RFM-FCA

On Online Retail II, the predictive experiment provides a direct comparison between the crisp and fuzzy representations.

| Representation | Features | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|---:|
| Crisp RFM-FCA | 30 | 0.7753 | 0.3476 | 0.4642 |
| Fuzzy RFM-FCA | 445 | 0.7915 | 0.3647 | 0.4933 |
| Fuzzy + redundancy suppression | 95 | 0.7875 | 0.3664 | 0.4952 |

Relative to the crisp representation, the unsuppressed fuzzy representation changes the predictive metrics by +0.0162 AUC, +0.0171 Spend R², and +0.0291 Invoice R².

After redundancy suppression, the fuzzy representation retains a predictive advantage relative to the crisp control while reducing the feature representation from 445 to 95 concepts.

The appropriate interpretation is that fuzzy concept features provide modest additional predictive information in the evaluated Online Retail II experiment. This should not be generalized as universal superiority.

## 3. Redundancy Suppression

Greedy extent-Jaccard redundancy suppression reduces the fuzzy Online Retail II representation from 445 to 95 concepts, corresponding to approximately 4.7× compression.

The project reports that near-duplicate concept pairs decrease from 3.3% to 0%, while mean pairwise extent Jaccard decreases from 0.069 to 0.006. The mean number of concepts per customer at membership ≥0.5 decreases from 41.3 to 4.1, compared with 5.2 for the crisp control.

The predictive results remain similar after suppression:

- AUC: 0.7915 → 0.7875
- Spend R²: 0.3647 → 0.3664
- Invoice R²: 0.4933 → 0.4952

Thus, the reduction primarily removes redundancy rather than eliminating the predictive signal represented by the fuzzy concepts.

## 4. Olist RFMS Results

The Olist experiment extends the representation from RFM to RFMS by adding Satisfaction.

The Olist population contains 93,357 retained customers after the delivered/valid-payment filtering used in the main RFMS analysis. Approximately 97% are single-order customers, producing severe frequency sparsity.

For the current F* formulation:

- 97.00% of customers fall in F1.
- 2.74% fall in F2.
- 0.20% fall in F3.
- 0.037% fall in F4.
- 0.027% fall in F5.
- Raw F* has 76 distinct values.
- Within-F1 standard deviation is 0.0082.
- F1 contains 90,553 customers.

The final fuzzy concept mining produced 1,369 concepts before pruning and 153 after pruning.

These results show that F* provides some differentiation within the dominant one-order population, but the underlying marketplace remains intrinsically sparse.

## 5. Frequency Ablation

Three frequency constructions were evaluated:

| Variant | F1 share | Shannon entropy | Concepts | Pruned | Natural k | FCA Silhouette at natural k |
|---|---:|---:|---:|---:|---:|---:|
| Literal frequency | 97.00% | 0.1376 | 1,368 | 153 | 12 | 0.5987 |
| Current F* | 97.00% | 0.1457 | 1,369 | 153 | 12 | 0.1656 |
| Revised engagement | 99.76% | 0.0196 | 2,610 | 203 | 11 | -0.0694 |

The revised engagement formulation produces greater within-F1 raw spread but substantially worse band entropy, more concepts, and poorer clustering behavior. It is therefore not adopted.

The current F* is retained as the project formulation because it provides the highest entropy among the evaluated variants while avoiding the degradation observed with the revised engagement alternative.

F* should be described as a purchase-intensity heuristic rather than as a solution to frequency sparsity.

## 6. FCA, K-means, Ward, and FCM

The Olist geometric clustering investigation shows that FCA-derived hard assignments should not be interpreted as conventional geometric clusters.

At matched k values:

| k | FCA Sil. | FCA DB | K-means Sil. | K-means DB | FCM Sil. | FCM DB |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.2591 | 2.2196 | 0.3494 | 0.9952 | 0.3398 | 1.1042 |
| 5 | 0.2467 | 2.3919 | 0.3867 | 0.8391 | 0.3847 | 0.8445 |
| 6 | 0.2307 | 2.5921 | 0.3917 | 0.8666 | 0.3854 | 0.8389 |

The FCA natural-k solution uses k=12 and obtains Silhouette = 0.1656 and Davies–Bouldin = 5.9467.

The investigation found that 99.86% of customers satisfy membership ≥0.5 in multiple top-level concepts, meaning that converting fuzzy concept membership to a single hard cluster is dominated by overlap and tie-breaking. Eight of the 12 FCA hard clusters have predominantly negative silhouette values.

Canonical FCM provides a more appropriate fuzzy-clustering benchmark:

- k=4: Silhouette 0.3398, DB 1.1042, FPC 0.5011
- k=5: Silhouette 0.3847, DB 0.8445, FPC 0.5005
- k=6: Silhouette 0.3854, DB 0.8389, FPC 0.4938
- k=12: Silhouette 0.3222, DB 1.1134, FPC 0.3758

These results indicate that the FCA representation and geometric clustering methods partition the customer space differently. They should not be reduced to a single performance ranking.

## 7. Overlap Structure

The overlap analysis demonstrates that the fuzzy representation is genuinely multi-membership rather than merely producing a conventional partition.

At α=0.5:

- 99.86% of customers belong to multiple top-level concepts.
- Average concepts per customer above α=0.5 = 3.71.
- Concept coverage = 100%.
- Mean membership entropy by dimension:
  - Recency = 0.4099
  - Frequency = 0.0138
  - Monetary = 0.3834
  - Satisfaction = 0.0050
  - Overall = 0.2030

The dominant overlap signatures include combinations such as F1+S5+M1+R5 and F1+S5+M1+R4. These profiles illustrate that customers sharing a frequency/satisfaction band can still differ in recency and monetary behavior.

This supports interpreting the FCA output as overlapping behavioral profiles rather than forcing it into a single-label partition.

## 8. Cross-Domain Validation

Online Retail II and Olist exhibit substantially different purchasing structures.

Online Retail II contains a high proportion of repeat customers, whereas Olist is dominated by one-time customers. Consequently, raw concept density and clustering behavior should not be interpreted as direct measures of dataset-independent FCA richness.

For Online Retail II, the fuzzy representation shows modest predictive gains over the crisp RFM-FCA control.

For Olist, the temporal predictive experiment required correction after the leakage audit.

## 9. Temporal Leakage Audit

The original Olist temporal protocol used customer-level Satisfaction values calculated over the full observation period. This allowed post-cutoff review information to enter the feature representation.

The audit identified:

- 2,341 observation customers (10.81%) with post-cutoff review information.
- 1,003 customers (4.63%) with a numerical change in Satisfaction.
- 253 of 554 future repurchasers (45.67%) with a Satisfaction change.
- Mean absolute Satisfaction change among affected customers = 1.6450.
- Maximum change = 4.0.

The corrected protocol restricts Satisfaction information to reviews available by the cutoff, using review_answer_timestamp as the governing information-availability timestamp. A review-creation-date sensitivity analysis was also evaluated.

The leakage-free Olist predictive results under strict pre-cutoff review filtering (review_answer_timestamp <= cutoff) and train-derived score bands are:

### Fixed-Split Holdout (70/30, Seed 42)

| Model | Features | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|---:|
| Raw RFMS Baseline | 4 | 0.5572 | 0.0009 | 0.0010 |
| Crisp RFMS-FCA | 44 | 0.5000 | -0.0003 | -0.0003 |
| Fuzzy RFMS-FCA (Suppressed) | 331 | 0.5548 | 0.0004 | 0.0005 |
| FCM soft (matched k=44) | 44 | 0.5426 | 0.0002 | 0.0003 |

The paired bootstrap comparison (B=1,000) against Fuzzy RFMS-FCA (Suppressed) yields:
- Raw RFMS Baseline vs Fuzzy: Delta AUC = +0.0024, 95% CI [-0.0341, +0.0379] (spans zero)
- FCM soft vs Fuzzy: Delta AUC = -0.0122, 95% CI [-0.0493, +0.0236] (spans zero)
- Crisp RFMS-FCA vs Fuzzy: Delta AUC = -0.0548, 95% CI [-0.0947, -0.0170]

### Multi-Split Validation (10 Random Splits, Seeds 1000–1009)

| Dataset | Model | Mean Features | Mean AUC ± SD | Mean Spend R² ± SD | Mean Invoice R² ± SD | Wins vs Ref |
|---|---|---:|---:|---:|---:|---:|
| Retail II | Raw RFM Baseline | 3.0 | 0.7770 ± 0.0117 | 0.2179 ± 0.0103 | 0.3420 ± 0.0214 | 0 / 10 |
| Retail II | Crisp RFM-FCA | 29.6 | 0.7768 ± 0.0121 | 0.3398 ± 0.0194 | 0.4418 ± 0.0145 | 0 / 10 |
| Retail II | Fuzzy RFM-FCA (Suppressed) | 96.7 | 0.7857 ± 0.0121 | 0.3559 ± 0.0195 | 0.4750 ± 0.0138 | Ref |
| Olist | Raw RFMS Baseline | 4.0 | 0.5587 ± 0.0107 | 0.0007 ± 0.0006 | 0.0009 ± 0.0006 | 4 / 10 |
| Olist | Crisp RFMS-FCA | 44.7 | 0.5353 ± 0.0153 | 0.0005 ± 0.0005 | 0.0005 ± 0.0005 | 0 / 10 |
| Olist | Fuzzy RFMS-FCA (Suppressed) | 339.7 | 0.5612 ± 0.0125 | 0.0010 ± 0.0005 | 0.0011 ± 0.0006 | Ref |
| Olist | FCM soft (matched k) | 44.7 | 0.5517 ± 0.0115 | 0.0005 ± 0.0010 | 0.0004 ± 0.0012 | 1 / 10 |

On Retail II, Fuzzy RFM-FCA significantly outperforms crisp FCA (mean delta +0.0089, paired t = 5.89, p = 2.3e-04) and raw RFM in 10/10 splits.
On Olist, Fuzzy RFMS-FCA outperforms re-derived crisp FCA (mean delta +0.0259, paired t = 7.70, p = 3.0e-05) and FCM, but does not outperform the raw RFMS baseline (mean delta +0.0025; raw beats fuzzy in 4/10 splits).

Therefore, the earlier Olist temporal result showing strong predictive superiority (AUC ~0.667 fixed split, ~0.635 multisplit) must be permanently retracted. Under the leakage-free protocol, predictive gains on Olist are near zero, consistent with the structural reality of a 97% single-order customer base.

## 10. Overall Findings

The results support three main conclusions.

First, the fuzzy extension provides additional structure and modest predictive utility on the Online Retail II repeat-purchase cohort. The fuzzy representation remains useful after substantial redundancy reduction.

Second, the Olist marketplace presents a fundamentally different regime dominated by one-time purchases. The F* ablation confirms that alternative frequency formulations cannot eliminate this structural sparsity. The corrected temporal experiment does not establish a predictive advantage on this domain.

Third, FCA should be evaluated primarily as an overlapping concept/profile discovery framework rather than as a conventional geometric clustering algorithm. FCM is therefore an important fuzzy benchmark, while Silhouette and Davies–Bouldin should be interpreted only as diagnostic measurements after any FCA-to-hard assignment conversion.

## 11. Limitations

The results should be interpreted with the following limitations:

1. Olist frequency is intrinsically sparse, with approximately 97% of customers making a single order.
2. The Olist F* formulation uses an item-row count proxy rather than a verified physical quantity field.
3. The Kneedle implementation is Kneedle-inspired rather than claimed to be a canonical implementation.
4. The stability measure is a project-specific proxy and should not be called canonical Kuznetsov stability.
5. The project FPC reported for FCA is descriptive/custom rather than standard Bezdek FPC.
6. Online Retail II and Olist represent structurally different customer populations, so their raw clustering metrics are not directly comparable.
7. The temporal leakage audit demonstrates a substantial performance collapse after correcting information availability, but it does not by itself establish a causal decomposition of the original performance gain.

# Online Retail II: Crisp RFM-FCA vs. Fuzzy RFM-FCA

## Scope and comparison design

This is a separate comparison run. It leaves all existing Step 1-13 outputs untouched.
Both new arms use the same cleaned customer population and the same raw R, F, M definitions:
recency in days relative to the maximum cleaned InvoiceDate, frequency as distinct InvoiceNo per
CustomerID, and monetary value as the sum of Quantity × UnitPrice.

The published crisp arm uses five equal-population quintiles and a 15-attribute binary context.
The paper states quintile scoring and a concept-support cutoff >0.04, but does not specify tie handling.
This reproduction deterministically sorts ties by CustomerID to create five balanced groups; that is
an explicit implementation convention, not a claim to have recovered the authors' exact code.
The fuzzy matched arm uses the project's dense-rank scores, centroid piecewise memberships, L-fuzzy
thresholds {0.3, 0.5, 0.7}, uncapped closed-itemset mining at min support 0.02, and the project's
Kneedle support/stability pruning. It keeps literal invoice-count F so the underlying RFM variables
match the paper. It is an end-to-end comparison of the scoring/context pipelines, not a one-factor
test of fuzziness alone, because scoring and pruning also differ.
Raw F=1 applies to 1,623/5,878 customers (27.61%);
4,255 customers are repeat buyers. The dense-rank rule maps the lowest 18 of the 90
distinct F values to F-score 1, which contains 5,524/5,878 customers
(93.98%). The raw F values remain tied together under dense ranking,
while balanced crisp quintiles must split tied values where group boundaries fall. The crisp
implementation uses CustomerID as a deterministic tie-break, reproducible but not specified by the paper.

The existing Step 7 F* run is reported as a third, separate reference. Its F* definition differs from
the paper's literal invoice count, so differences involving that arm cannot be attributed only to fuzzy FCA.

## Population check

- Combined raw rows: **1,067,371**.
- Cleaned transaction rows: **805,549**.
- Customers: **5,878**.
- Reference date: **2011-12-09 12:50:00**.
- The cleaned rows/customer totals match the values reported for the base paper.

## Results

| Arm | Scoring/context | Attributes | Closed concepts | Support > 0.04 | Nontrivial support > 0.04 | Kneedle retained |
|---|---|---:|---:|---:|---:|---:|
| Crisp RFM-FCA reconstruction | quintile / binary | 15 | 205 | 55 incl. universal root | 54 | not used |
| Matched-feature fuzzy RFM-FCA | dense rank / fuzzy L-scaling | 45 | 609 | 359 | 359 | 59 |
| Existing full project F* fuzzy run | existing Step 7 | 45 | 1,064 | not recalculated | 115 |

### What the figures mean

- Crisp concepts retained by the paper's support rule: **55**.
- Of the crisp support-filtered concepts, **54** have nonempty intents; the remaining concept is the universal root (all customers, no shared score attribute).
- Matched fuzzy concepts above the same 0.04 support level: **359**.
- Matched fuzzy concepts retained by the project Kneedle rule: **59**.
- Matched fuzzy Kneedle thresholds: support **0.160939**, stability proxy **0.972477**.
- Mean pruned-concept memberships per customer in the matched fuzzy arm: **17.44**.
- Mean share of customers with >1 fuzzy band at membership ≥0.5, averaged over R/F/M: **0.13%**.
- Crisp concepts and L-fuzzy scaled concepts are formed in different attribute contexts (15 vs. 45
  possible scaled attributes) and use different pruning rules. Their raw concept counts are descriptive,
  not a standalone quality ranking.

## Check against published crisp concepts

The 31 concepts listed in the paper's Table 7 were checked. All 31 were
recovered as exact closed intents, but only 3/31 reconstructed
customer counts match the published counts exactly. See `published_table7_concept_check.csv` for all
reported and reconstructed customer counts. Several
R-M combinations are close, while some frequency combinations differ substantially (F2&M2: 423 here
versus 620 published; F2&M1: 411 versus 287; F5&M5: 899 versus 845). The cleaned population is
reproduced, but the published concept table is not exactly reproduced under the declared CustomerID
tie-break. The paper does not specify tie handling, and without its exact scoring rule/code, the cause
of each discrepancy cannot be determined. This limits claims of exact replication, especially for F.

## Reproducibility and artifacts

- `comparison_summary.csv`: top-line arm comparison.
- `score_distributions.csv`: raw unique-value counts and score-band counts for each new arm.
- `base_crisp_all_concepts.csv`: exhaustive classical FCA concepts.
- `base_crisp_concepts_support_gt_004.csv`: paper-style support-filtered concepts.
- `base_crisp_nontrivial_concepts_support_gt_004.csv`: support-filtered crisp concepts excluding the universal root.
- `published_table7_concept_check.csv`: published Table 7 counts versus this reconstruction.
- `matched_fuzzy_all_concepts.csv`: matched-feature L-fuzzy closed concepts at min support 0.02.
- `matched_fuzzy_concepts_support_gt_004.csv`: matched fuzzy concepts filtered at the paper's support cutoff.
- `matched_fuzzy_concepts_kneedle_pruned.csv`: matched fuzzy concepts after Kneedle support/stability pruning.
- `shared_rfm_customer_scores.csv`: customer-level raw RFM and both score encodings.

Paper: [Rungruang et al. (2024), Expert Systems with Applications, 237, 121449](https://doi.org/10.1016/j.eswa.2023.121449).
Dataset: [UCI Online Retail II](https://doi.org/10.24432/C5CG6D).

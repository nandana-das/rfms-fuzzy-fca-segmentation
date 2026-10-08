# Results (manuscript draft, frozen evidence)

Numbers are taken from `results/final_evidence/EVIDENCE_TABLES.md` (see `docs/RESULTS_SUMMARY.md` for the full tables). The pre-audit draft is kept in `docs/superseded_2026-10-08/results_section.md` and must not be used.

## 1. Experimental setting

We evaluate on two retail transaction datasets:
- **Online Retail II**, the dataset of Rungruang et al. (2024);
- **Dunnhumby "The Complete Journey"**, a reproduction in a high-frequency grocery setting.

RFM is computed from all transactions up to each observation cutoff, and outcomes are measured over the following 91 days:
- Dunnhumby: cutoffs at days 347, 438, 529 and 620; about 2,500 households; repurchase rate 92.2–93.2%.
- Online Retail II: five quarterly cutoffs from 2010-09-10 to 2011-09-09; 3,377–5,281 customers; repurchase rate 30.6–57.7%.
- Online Retail II original protocol (2010-12-09 cutoff, 365-day holdout): reported separately.

**Procedure.** Within each cutoff we use stratified 5-fold cross-validation over customers. Every transformation is fitted on the training folds: quintile cutpoints, fuzzy centroids, concept mining, redundancy suppression and splines.

**Metrics and models.** Repurchase is evaluated by ROC AUC, and log1p future spend and log1p future order count by R². All representations share the same downstream models (L2-regularized logistic and ridge regression).

**Inference.** Differences are assessed with a paired customer-level bootstrap and Holm-corrected within each dataset.

## 2. Fuzzy vs crisp RFM-FCA (H1)

Under the rolling-origin evaluation, fuzzy RFM-FCA improves on the crisp RFM-FCA of the base paper (Table A). The pooled improvement is statistically significant on all three metrics in both datasets (paired bootstrap, Holm p = 0.014). Descriptively, the difference is also positive at every individual cutoff. These cutoff counts summarise consistency and are not a significance test.
- **Online Retail II:** +0.006 ROC AUC, +0.014 Spend R², +0.027 Invoice R², all Holm p = 0.014; positive at 5/5 cutoffs (descriptive).
- **Dunnhumby:** +0.021, +0.027 and +0.021, all Holm p = 0.014; positive at 4/4 cutoffs (descriptive).
- **Online Retail II original protocol** (a single holdout; descriptive only): +0.023, +0.012 and +0.020.

The improvements are consistent but modest, and smallest for repurchase discrimination on Online Retail II. **H1 is supported.**

## 3. Comparison with non-FCA RFM baselines (H2)

Table B compares fuzzy RFM-FCA with representations that use no formal concepts.
- **Untransformed baseline.** Relative to standardized raw R/F/M, fuzzy RFM-FCA gains about 0.10 in Spend R² on both datasets. Most of this gain reflects the skewness of raw RFM in a linear model rather than concept structure: a three-feature log transform already captures most of it.
- **Spline baseline.** Against a spline on log-RFM, there is no significant difference on Dunnhumby for any metric, nor on Online Retail II for AUC and Spend R². On Online Retail II Invoice R², fuzzy RFM-FCA is significantly worse (−0.032, 0/5 cutoffs).
- **Concept structure.** Fuzzy band memberships without any concepts perform close to fuzzy RFM-FCA on Dunnhumby, while concepts add 0.008–0.019 on Online Retail II.

Five of six comparisons with the spline baseline were not significantly different, but equivalence was not formally established, because no equivalence margin was pre-specified. Online Retail II Invoice R² was significantly worse by 0.032. **H2 is inconclusive.** The evidence does not show that fuzzy RFM-FCA is superior to a strong nonlinear RFM model.

## 4. Structural stability (H3)

We measure how much a customer's set of core concepts (membership ≥ 0.5) overlaps between two fits (core-profile Jaccard).
- **Refit stability.** On refitting with 80% subsamples, fuzzy and crisp RFM-FCA show no significant difference in stability (about 0.95–0.97 for both; differences −0.005 and +0.005, intervals spanning zero).
- **Quarterly re-segmentation.** When the segmentation is refitted each quarter, fuzzy profiles change more than crisp ones (−0.035 on Dunnhumby, −0.020 on Online Retail II; unfavourable at every cutoff pair).
- **Unchanged-behaviour subset.** For customers whose RFM percentile ranks barely changed, there is no difference on Dunnhumby, and fuzzy is more stable on Online Retail II (+0.020).
- **Churn under every method.** About half of a typical customer's concept profile changes from one quarter to the next.

**Matched-count diagnostic.** A single pre-specified diagnostic truncates fuzzy RFM-FCA, per fit, to the crisp concept count. At matched count the quarterly gap disappears (+0.003 on both datasets, intervals including zero). Because truncation keeps the highest-support concepts, this supports a "concept count and/or concept size" explanation of the lower temporal stability: fuzzy retains about 92 vs 55 and 77 vs 49 concepts. It does not prove that concept count alone is the cause. Even so, no configuration shows fuzzification making segments *more* stable overall. **H3 is not supported.**

## 5. Stability-based pruning (ablation)

Adding a canonical Kuznetsov stability filter did not improve prediction consistently. Its only significant pooled difference (+0.003 AUC on Online Retail II) is positive at just one of five cutoffs.

It also made segments markedly less stable on every measure:
- refit core Jaccard drops by 0.14 on Dunnhumby and 0.07 on Online Retail II relative to crisp;
- only 64% and 81% of its concepts recur across refits.

The filter's threshold reduces to an absolute customer-count criterion that shifts with training-set size, and we therefore exclude it from the proposed method.

## 6. Summary of evidence

| Hypothesis | Status |
|---|---|
| H1: fuzzy RFM-FCA improves predictive performance over crisp RFM-FCA | **Supported** (pooled improvement significant on all metrics in both datasets, Holm p = 0.014; positive at every cutoff descriptively) |
| H2: fuzzy RFM-FCA achieves predictive performance comparable to strong nonlinear non-FCA RFM baselines | **Inconclusive** (5/6 spline comparisons not significantly different; equivalence not formally established; Retail II Invoice R² significantly worse by 0.032) |
| H3: fuzzy RFM-FCA improves segmentation stability over crisp RFM-FCA | **Not supported** (no difference on refits; less stable over quarters; the gap is consistent with a concept count and/or concept size explanation) |

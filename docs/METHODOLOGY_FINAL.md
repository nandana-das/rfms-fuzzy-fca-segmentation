# Final Frozen Methodology

**Status:** frozen 2026-10-08. This supersedes Methodology v4, whose pre-audit text is preserved in `docs/superseded_2026-10-08/METHODOLOGY_V4.md`. Defects, changes and their evidence are recorded in `docs/AUDIT_ERRATA.md`.

This document describes the method **exactly as implemented** in the scripts that produce the final evidence (`scripts/baseline_ladder_rolling_origin.py`, `scripts/segment_stability.py`, and the shared functions they import). No parameter below was chosen by looking at outcome data.

## 1. Research question

To what extent does fuzzifying the RFM formal context change (a) the out-of-sample predictive adequacy and (b) the structural stability of FCA-based customer segmentation, compared with conventional crisp RFM-FCA (Rungruang et al., 2024) and with strong non-FCA RFM baselines?

The hypotheses and their status are in `docs/PROJECT_DOCUMENT.md` §2.

## 2. Data and RFM construction (RFM only)

| | Dunnhumby "The Complete Journey" | Online Retail II (base-paper dataset) |
|---|---|---|
| Customer | `household_key` | `CustomerID` |
| Order | `BASKET_ID` | `InvoiceNo` |
| Date | `DAY` | `InvoiceDate` |
| Monetary | `SALES_VALUE` (not multiplied by quantity) | `Quantity × UnitPrice` |
| Cleaning | none needed | drop missing `CustomerID`, `Quantity ≤ 0`, `UnitPrice ≤ 0`, invoices starting with "C" |

*Cleaning discrepancy (unresolved).* Our cleaning follows the base paper's stated rules and yields 805,549 lines and 5,878 customers; the paper reports 805,620 lines and 5,878 customers. The 71-line gap equals exactly the number of zero-price lines (UnitPrice = 0) that pass the other filters: keeping them reproduces 805,620 lines, but then the customer count becomes 5,881. No single cleaning rule reproduces both published numbers, so this is recorded as an unresolved inconsistency in the base paper's reported counts; our pipeline keeps the stated rule (price > 0).

For an observation cutoff T (all transactions up to T are used):
- R = days from the customer's last purchase to T;
- F = number of distinct orders;
- M = total monetary value.

Outcomes are measured on the holdout window (T, T + h]:
- repurchase: at least one order;
- future spend: log1p(spend);
- future orders: log1p(order count).

RFMS, Satisfaction, F\* and Olist are out of scope.

## 3. Tie-preserving quintile scoring (fitted on training customers only)

- Cutpoints q₀.₂, q₀.₄, q₀.₆, q₀.₈ of each of R, F and M are computed on the training customers.
- A value x is in band k if q₍ₖ₋₁₎/₅ < x ≤ qₖ/₅, with q₀ = −∞ and q₁ = +∞.
- Identical raw values always share a band, and the same cutpoints place unseen customers.
- R is inverted, so band 5 = most recent.
- If two cutpoints coincide, fitting stops with an error rather than producing an empty band. This never occurred in any training fold.

Bands are equal-sized except at mass points, and occupancy is reported per origin in `results/baseline_ladder_rolling_origin/band_occupancy.csv`. Range over all origins:
- Dunnhumby: R 15.6–26.8% (about 15% of households have R = 0); F 19.2–20.9%; M 20.0%.
- Online Retail II: R 18.5–21.7%; F 11.7–37.9% (about 33% of customers have F = 1); M 20.0%.

*Why not the base paper's rank-first quintiles:* rank-first tie-breaking splits identical customers across bands and cannot be applied to new customers. *Why not dense rank (used before the audit):* it divides the range of distinct values, not customers, and put 84–95% of customers in a single band.

## 4. Fuzzy membership (frozen)

For each dimension, the centroids c₁ < … < c₅ are the medians of the raw values in each training band. Strict monotonicity was verified in all 60 training fits.

Piecewise-linear memberships with outer shoulders:
- μ₁(x) = 1 for x ≤ c₁;
- μ₅(x) = 1 for x ≥ c₅;
- for cₖ ≤ x ≤ cₖ₊₁: μₖ(x) = (cₖ₊₁ − x)/(cₖ₊₁ − cₖ) and μₖ₊₁(x) = (x − cₖ)/(cₖ₊₁ − cₖ).

Each customer has nonzero membership in at most two adjacent levels per dimension, and the memberships sum to 1 (verified). Test customers are projected with the training centroids.

**Saturation is deliberate.** Every value at or above c₅ (roughly the 90th percentile) has μ₅ = 1, exactly as crisp quintiles treat the whole top band alike. This is not changed in response to any result (see §9, limitation 1).

## 5. Formal context and concept mining

*Documentation correction:* the earlier "L-fuzzy" wording overstated what is implemented. The context is a **threshold-scaled binary context derived from the fuzzy memberships** (conceptual scaling); no residuated L-fuzzy concept lattice is computed.

1. Each of the 15 memberships μ_b is scaled at thresholds 0.3, 0.5 and 0.7, giving 45 binary attributes "b@t" (customer has μ_b ≥ t).
2. Frequent itemsets with support ≥ 0.04 are mined from the training customers by FP-growth (no length cap). Itemsets are grouped by their exact extent, which yields closed intents; concepts with support < 0.04 are discarded.

## 6. Customer–concept membership

The membership of customer i in a concept is the Gödel minimum of the band memberships named in its intent, μ_C(i) = min₍b ∈ bands(C)₎ μ_b(i). As implemented, the threshold levels in the intent are discarded (so "R5@0.3" and "R5@0.7" both map to μ_R5). Concepts that map to the same set of bands therefore produce identical features. The suppression step removes them.

## 7. Redundancy suppression

- Concepts are ranked by support (descending), intent size (ascending), and the object-profile diversity proxy (descending), then original order.
- Greedily, a concept is dropped if its core extent {i : μ_C(i) ≥ 0.5} has Jaccard ≥ 0.80 with an already-kept concept's core extent. As an implementation safeguard (not a contribution), concepts with an empty core extent are also dropped.
- The empty-core rule was added in the audit (defect 7). It removes concepts with no core customers, such as two adjacent bands of one dimension, which earlier passed through untested.

Resulting concept counts (mean over training folds and origins): Dunnhumby 91.6, Online Retail II 76.8. The crisp method retains 54.1 and 48.9.

## 8. Comparators

- **Crisp RFM-FCA (base-paper method):** the same quintile bands as binary attributes; crisp closed concepts with support ≥ 0.04; membership 1 if the customer is in every band of the intent. No suppression (the base paper has none).
- **Non-FCA RFM baselines** (identical downstream models):
  - standardized raw R/F/M;
  - standardized log1p R/F/M;
  - cubic B-spline (5 quantile knots) on log1p R/F/M;
  - the 15 crisp or fuzzy band indicators without concepts;
  - untuned gradient boosting on raw R/F/M, as a flexible-model reference.

## 9. Evaluation protocol

**Predictive** (`scripts/baseline_ladder_rolling_origin.py`):
- Rolling origins with a 91-day holdout. Dunnhumby: days 347, 438, 529, 620. Online Retail II: 2010-09-10, 2010-12-10, 2011-03-11, 2011-06-10, 2011-09-09.
- The Online Retail II original protocol (2010-12-09 cutoff, 365-day holdout) is reported separately and not pooled.
- Within each origin: stratified 5-fold CV over customers. Every transformation (cutpoints, centroids, splines, concept mining, suppression) is fit on the training folds; metrics come from the pooled out-of-fold predictions.
- Models: LogisticRegressionCV (10 Cs, 5-fold, ROC AUC, lbfgs) for repurchase, and RidgeCV (α ∈ logspace(−3, 3, 20), 5-fold) for log1p spend and log1p orders.
- Metrics: ROC AUC, Spend R² and Invoice R². PR AUC is not used; with 93% repurchase on Dunnhumby it carries little information.
- Inference: paired customer-level bootstrap (B = 2000) of the metric difference within each origin; pooled estimate = mean over origins; Holm correction within each dataset.

**Structural** (`scripts/segment_stability.py`):
- Primary measure: core-profile Jaccard (the concepts with μ ≥ 0.5 per customer, matched across fits by band set).
- Study A, refit stability: 30 subsamples of 80% vs the full-cohort fit at every origin.
- Study B, quarterly re-segmentation: refit at consecutive origins; all customers, plus an unchanged-behaviour subset (|Δ percentile rank| ≤ 0.05 in each of R, F and M).
- One diagnostic: the matched-concept-count diagnostic (`scripts/matched_count_stability_diagnostic.py`; rule in `AUDIT_ERRATA.md` §6.1).

**Base-paper reconstruction** (descriptive): recovery of the base paper's 31 published concept intents on Online Retail II, using the base paper's own rank-first quintiles on all 5,878 customers (`scripts/fair_comparison_retail2.py` → `results/fair_comparison_retail2/published_table7_reconstruction.csv`).

## 10. Components excluded from the final method

These are kept in the repository as ablation or negative evidence:

| Component | Reason for exclusion | Evidence |
|---|---|---|
| Kuznetsov stability filter | No demonstrated predictive benefit; the least stable arm on every stability measure; its threshold (loss ≤ 5.4e-20) is below float64 resolution and acts as an absolute customer-count rule that shifts with training-set size; the threshold was originally chosen on evaluation seeds | `results/final_evidence/`, `results/segment_stability/`, `AUDIT_ERRATA.md` §3, §6 |
| Kneedle-inspired pruning | Not applied in the evaluated pipeline; heuristic, not canonical Kneedle | `results/canonical_kneedle_*` (historical) |
| Stage 1 / Stage 2 optimization (granularity, selection thresholds) | Hyperparameter tuning, invalidated by implementation defects; not re-run by decision | `AUDIT_ERRATA.md` §1, §3 |
| Dense-rank scoring | Degenerate band occupancy | `AUDIT_ERRATA.md` §4 |
| L-threshold sensitivity study, KMeans/Ward/FCM silhouette benchmark, four-way comparison | Run under dense-rank scoring and the misspecified raw baseline; not re-run; not part of the hypotheses | historical result folders (bannered) |

## 11. Limitations

1. **Top-band saturation.** Ordinal RFM levels, crisp or fuzzy, discard magnitude within the top band. On Online Retail II, where frequency is heavy-tailed, fuzzy RFM-FCA is significantly worse than the spline baseline on Invoice R² (−0.032). The superseded dense-rank variant, which happened to keep tail resolution, scored 0.028 higher there. On Dunnhumby there is no significant shortfall.
2. **No demonstrated advantage over strong non-FCA models.** Against a spline on log-RFM, five of six comparisons were not significantly different, equivalence was not formally established, and Online Retail II Invoice R² was significantly worse by 0.032. The improvement over crisp RFM-FCA cannot be attributed to the concept structure alone. On Dunnhumby, fuzzy band memberships without concepts are not significantly different from fuzzy RFM-FCA on AUC or Spend R² (concepts add +0.006 Invoice R²). On Online Retail II, concepts add a significant 0.008–0.019.
3. **No stability advantage.** Fuzzy RFM-FCA is not more stable than crisp RFM-FCA (see `RESULTS_SUMMARY.md` §C). The matched-count diagnostic supports a concept count and/or concept size explanation of its lower temporal stability, not count alone.
4. **Interpretability is not measured.** No claim of greater interpretability is made.
5. **Design choices are conventions.** Five levels, thresholds 0.3/0.5/0.7, minimum support 0.04, J_max 0.80 and μ-cut 0.5 are conventional and were not tuned. Other values could give different concept counts.
6. **Narrow domain.** Two retail datasets; the Dunnhumby repurchase outcome is highly imbalanced (93.16% repurchase at the day-620 origin; about 171 non-repurchasers), so AUC differences there are noisy.
7. **Approximate inference.** The bootstrap captures test-sample variability, not refitting variability; origins share customers and history, so pooled intervals are approximate. With B = 2000, the smallest attainable bootstrap p-value is 0.0005, so the reported Holm p = 0.014 is the floor of the procedure, not a measure of effect strength.
8. **Inexact reconstruction.** The base paper's exact customer counts cannot be reconstructed, because its tie-breaking rule is unspecified (31/31 intents recovered; 3 of 31 counts match). Its reported cleaned-line and customer counts are also mutually inconsistent under any single cleaning rule (§2).

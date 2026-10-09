# v2 structural stability — analysis plan (exploratory, post hoc)

**Written:** 2026-10-09 on `experiment/v2-hybrid-fuzzy-fca`, before any v2 stability code was written or run. This plan is committed on its own, before any v2 stability code or results, so Git history records that it preceded the run. It was still written after the v1 and v2 predictive results were known, so it is an analysis plan, not a preregistration. v2 was developed after inspecting v1 results, so everything here is exploratory and post hoc. The frozen v1 evidence (commit `7607e09`) is not modified.

## 1. Implementation audit: where the log-RFM features enter

Source: `scripts/v2_hybrid_ladder.py` → `build_features`.

1. The hybrid calls the **unchanged v1 concept builder** (`baseline_ladder_rolling_origin.build_features` with arm `fuzzy_rfm_fca`, or `crisp_rfm_fca` for the crisp hybrid). This step produces the quintile bands, fuzzy memberships, concept mining, Jaccard suppression with the empty-core safeguard, and customer–concept memberships.
2. Afterwards, standardized log1p(R, F, M) is **appended as three extra columns of the model input matrix**.

| Stage | Uses log-RFM? |
|---|---|
| Quintile scoring, fuzzy membership | No |
| Concept mining (threshold-scaled context, support 0.04) | No |
| Redundancy suppression (Jaccard 0.80, μ-cut 0.5, empty core) | No |
| Customer–concept membership (segments, core profiles) | No |
| Downstream logistic/ridge prediction | **Yes (only here)** |

**Consequence.** For identical training customers:
- the v2 hybrid's segments (concept set and core profiles) are *by construction identical* to v1 fuzzy RFM-FCA's;
- the crisp hybrid's segments are identical to v1 crisp RFM-FCA's.

The log features cannot change segmentation stability, and no claim that they do will be made.

## 2. What "v2 segment stability" means here

v2 segment stability is the stability of the v2 concept set and of each customer's core profile (concepts with μ ≥ 0.5). This is the v1 core-profile Jaccard measure, unchanged. Because the segments are identical by construction, the v1 results (refit, quarterly re-segmentation, unchanged-behaviour subset, matched-count diagnostic; `results/segment_stability/`) **apply to v2 without re-estimation**:
- fuzzy hybrid = v1 fuzzy;
- crisp hybrid = v1 crisp.

Re-running those studies would reproduce identical numbers. Presenting them as new v2 evidence would double-count.

### Part A (planned): implementation-equivalence check, not a new experiment

The v1 stability study and the v2 hybrid use two different code paths (`segment_stability.fit_model`/`project` vs the ladder's `build_features`). For every fit in the v1 stability protocol, the following is checked:
- the 9 pooled origins × (full cohort + the same 30 subsamples, seed 2026, 80%);
- the 7 consecutive-origin temporal fits;
- for the fuzzy and crisp arms.

The check passes if:
- the v2-path concept set (band-set keys) equals the v1-stability-path concept set; and
- the core profiles of all cohort customers (μ ≥ 0.5) are identical.

- **If every fit matches:** the v1 stability results are stated to apply to v2, with no new numbers.
- **If any fit differs:** the mismatch is reported as an implementation discrepancy, and **no** stability conclusion is drawn for v2 until it is understood.

The matched-concept-count diagnostic is likewise inherited unchanged (v1 rule, `docs/AUDIT_ERRATA.md` §6.1), including its count-versus-size caveat.

### Part B (optional; run only if approved): predictive-representation refit stability

This is a separate property from segmentation stability and must never be reported as segment stability. It asks whether the **fitted predictions** change when the whole pipeline (representation + downstream models) is refit on different customers.

- **Methods:** v1 fuzzy RFM-FCA, v2 hybrid (fuzzy concepts + log RFM), crisp hybrid (crisp concepts + log RFM), and v1 crisp RFM-FCA as reference. No new methods; nothing tuned.
- **Protocol:**
  - At each of the 9 pooled origins, fit on the full cohort and on each of the same 30 subsamples (seed 2026, 80%).
  - Each fit uses the frozen v1 models (LogisticRegressionCV / RidgeCV, same settings) trained on that sample's own holdout outcomes, then predicts every customer in the cohort.
- **Measure:** Spearman rank correlation between the full-fit and subsample-fit predictions, per target (repurchase probability, log spend, log invoices).
  - No single target is designated primary. All three are reported.
  - Paired method differences use the same subsample index; Holm correction is applied within each dataset over the reported comparisons.
- **Comparisons:**
  - v2 hybrid − v1 fuzzy;
  - v2 hybrid − crisp hybrid;
  - (reference) v1 fuzzy − v1 crisp.
- **Inference:** a percentile interval over the 30 subsamples for each origin and pooled over origins; the number of origins with a positive difference is reported descriptively. A sign-flip test over the 4/5 origins has too few units, so no p-value is claimed from origins.
  - Subsamples overlap (80% of the same cohort), so the intervals describe refit variability, not sampling uncertainty for new populations.
  - Models are trained on outcomes of the customers they then score, so these are in-sample predictions; that is acceptable for a stability measure but not for accuracy.
- **Not done in Part B:**
  - **Temporal predictive stability:** predictions at successive origins target different future windows, so changes conflate behaviour change with model instability.
  - **A matched-count variant:** count matching concerns segment structure, not prediction.

**Decision (2026-10-09, before running): Part A only. Part B (predictive-representation stability) will not be run.**

## 3. Questions and how they will be answered

| Question | Answer route |
|---|---|
| Does v2 improve refit stability over v1? | Segmentation: no, identical by construction (confirmed by Part A). Predictive representation: not evaluated (Part B not run). |
| Does v2 improve temporal stability over v1? | Segmentation: no, identical by construction. Predictive: not evaluated (see above). |
| Does the crisp hybrid behave differently from the fuzzy hybrid? | Segmentation: exactly the v1 crisp-vs-fuzzy result. Predictive representation: not evaluated (Part B not run). |
| Does matching concept count alter the conclusion? | Inherited from v1 (gap no longer significant at matched count; count and/or size caveat). |
| Does any result justify a stability claim? | No. With Part B not run, no new stability claim is possible; v2 inherits the v1 segment-stability conclusions (H3 not supported). |

## 4. Constraints

- No changes to membership functions, thresholds, pruning, log transformation or models.
- No predictive optimization.
- Stability is not inferred from predictive performance.
- New work goes in a separate commit on the v2 branch; the v2 commit `f0342ef` is preserved.

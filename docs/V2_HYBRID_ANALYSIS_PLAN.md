# Version 2 (hybrid): analysis plan (exploratory, post hoc)

**Written:** 2026-10-08, on branch `experiment/v2-hybrid-fuzzy-fca`, after the v1 results were known.

**Ordering of plan and run (not established by Git).** The analysis plan (`docs/V2_HYBRID_ANALYSIS_PLAN.md`), the script and the results were first committed together in a single commit, so Git history does not show that the plan preceded the run. The only evidence of ordering is filesystem modification times (plan 2026-10-08 23:52:51 IST; script 23:53:09; first output 23:53:44), which are not tamper-evident. The plan was written after the v1 results were known. It should therefore be treated as an analysis plan, not a preregistration.
**Baseline:** the frozen v1 evidence (commit `7607e09`), which stays unchanged and remains the primary record.

## Honest status of this study

This is a **post hoc** study. The change below was motivated by a v1 result, the Online Retail II Invoice R² shortfall against the spline baseline (−0.032), which is attributed to top-band saturation. There is no unused data: v2 is evaluated on the same origins that v1 already reported. Gains measured here are therefore optimistic. Every v2 result must be labelled **"v2, chosen after seeing v1"** and must never replace or be merged into the v1 evidence tables.

## The single change

**Hybrid representation.** The features are the fuzzy RFM-FCA concept memberships (v1 method, unchanged) **plus** standardized log1p(R, F, M), fitted on the training folds. The concepts remain the interpretable segments; the log terms restore within-band magnitude that saturating top levels discard.

Nothing else changes:
- quintile scoring, fuzzy levels, thresholds, support 0.04, J_max 0.80, μ-cut 0.5 and the empty-core safeguard;
- downstream models;
- origins, folds and seeds;
- bootstrap and Holm procedure.

No parameter is tuned. There will be exactly one run.

## Arms

| Arm | Role |
|---|---|
| `hybrid_fuzzy_fca` | v2 candidate: fuzzy concepts + log RFM |
| `hybrid_crisp_fca` | Fair control: crisp concepts + log RFM, so that fuzzy vs crisp is compared on equal terms |
| `fuzzy_rfm_fca` | v1 method (expected to reproduce v1 exactly) |
| `crisp_rfm_fca` | Base-paper method |
| `log_rfm` | Log RFM alone, used to isolate what concepts add |
| `spline_log_rfm` | Strongest v1 baseline |
| `raw_std` | Reference only |

## Comparisons (Holm-corrected within each dataset; 5 comparisons × 3 metrics)

| ID | Comparison | Question |
|---|---|---|
| P1 | hybrid_fuzzy − spline_log_rfm | Does v2 beat the strongest baseline? |
| P2 | hybrid_fuzzy − fuzzy_rfm_fca | Does adding log magnitude fix the saturation shortfall? |
| P3 | hybrid_fuzzy − hybrid_crisp | Does fuzzification still help when both have log magnitude? |
| P4 | hybrid_fuzzy − log_rfm | Do the concepts add anything beyond log RFM? |
| P5 | hybrid_fuzzy − crisp_rfm_fca | v2 vs the base-paper method |

## Claim rules (fixed now)

- **"v2 improves on v1":** only if P2 is significantly positive on Retail II Invoice R² and no P2 metric is significantly negative on either dataset.
- **"v2 beats the spline baseline":** only for the metrics where P1 is significantly positive after Holm. Non-significant P1 is reported as "not significantly different", never as "equivalent".
- **"Concepts add value beyond log RFM":** only where P4 is significantly positive. If P4 is not significant, any gain over the spline must be attributed to the log terms, not to FCA.
- **"Fuzzification helps in the hybrid":** only where P3 is significantly positive.
- All results are reported, including unfavourable ones.
- The v1 hypotheses (H1–H3) and their statuses are not revisited by this study.

## Outputs

- Script: `scripts/v2_hybrid_ladder.py`
- Results: `results/v2_hybrid/`

The structural-stability properties of the hybrid are not assessed in this study.

# v2 structural stability — results (exploratory, post hoc)

> **Exploratory, post hoc.** v2 was developed after inspecting v1 results. The plan (`docs/V2_STABILITY_ANALYSIS_PLAN.md`) was committed on its own (commit `2582120`) before this check's code or results existed; it is an analysis plan, not a preregistration. Frozen v1 (commit `7607e09`) and the v2 predictive results (commit `f0342ef`) are unchanged.

## 1. What was evaluated, and why no new stability numbers are reported

The implementation audit (plan §1) found that the log-RFM features enter v2 **only as downstream model inputs**. Quintile scoring, fuzzy membership, concept mining, redundancy suppression (including the empty-core safeguard) and customer–concept membership are the unchanged v1 code. Therefore:
- **v2 hybrid segments = v1 fuzzy RFM-FCA segments;**
- **crisp hybrid segments = v1 crisp RFM-FCA segments.**

Segment stability is a property of the segmentation, not of the downstream model, so v2's segment stability **is** v1's.

Part B (the stability of fitted predictions, a different property) was **not run**, by decision recorded in the plan before running. No claim about prediction stability is made.

## 2. Part A — implementation-equivalence check (verification, not an experiment)

Script: `scripts/v2_segment_equivalence_check.py`. Output: `results/v2_stability/` (per-fit CSV, summary and report).

The check replays every fit of the v1 stability protocol, with the same cohorts, the same 30 subsamples (seed 2026, 80%) in the same draw order, and the same 7 consecutive-origin temporal projections. For each fit it compares the v2 code path with the v1 stability code path on:
- concept count;
- the multiset of concept-membership columns over all evaluated customers;
- core profiles (μ ≥ 0.5).

| Study | Dataset | Arm | Fits | Concept count equal | Membership columns identical | Core profiles identical | Max \|difference\| | Concepts retained (min–max) |
|---|---|---|---|---|---|---|---|---|
| A refit | Dunnhumby | fuzzy | 124 | 124 | 124 | 124 | 0 | 81–99 |
| A refit | Dunnhumby | crisp | 124 | 124 | 124 | 124 | 0 | 51–56 |
| A refit | Online Retail II | fuzzy | 155 | 155 | 155 | 155 | 0 | 64–84 |
| A refit | Online Retail II | crisp | 155 | 155 | 155 | 155 | 0 | 45–53 |
| B temporal | Dunnhumby | fuzzy | 6 | 6 | 6 | 6 | 0 | 90–94 |
| B temporal | Dunnhumby | crisp | 6 | 6 | 6 | 6 | 0 | 52–56 |
| B temporal | Online Retail II | fuzzy | 8 | 8 | 8 | 8 | 0 | 67–80 |
| B temporal | Online Retail II | crisp | 8 | 8 | 8 | 8 | 0 | 47–50 |

**Result: all 586 fit comparisons are identical (maximum absolute difference 0).**
- Negative controls confirmed the check can fail. Different training customers produced different concept counts (90 vs 87), and a 10⁻⁶ change to a single membership value was detected.
- Empty-core handling is part of the shared v1 code, so it is identical in both paths.

## 3. Inherited v1 stability results (apply to v2 unchanged)

Primary measure: core-profile Jaccard; difference = fuzzy − crisp, which here is also v2 fuzzy hybrid − crisp hybrid. Values are from `results/final_evidence/C_structural_stability.csv` (frozen v1). Refit intervals are 2.5–97.5 percentiles over subsamples, with no p-value. Temporal intervals are customer-bootstrap 95% CIs with Holm-adjusted p-values. "Favourable" counts are descriptive.

| Study | Dataset | Δ (exact) | 95% interval | p (Holm) | Origins / pairs favourable |
|---|---|---|---|---|---|
| A. Refit | Dunnhumby | −0.0054 | [−0.0203, +0.0042] | n/a | 1/4 |
| A. Refit | Online Retail II | +0.0045 | [−0.0023, +0.0161] | n/a | 2/5 |
| B. Quarterly, all customers | Dunnhumby | −0.0346 | [−0.0396, −0.0292] | 0.004 | 0/3 |
| B. Quarterly, all customers | Online Retail II | −0.0201 | [−0.0236, −0.0166] | 0.004 | 0/4 |
| B. Quarterly, unchanged-behaviour subset | Dunnhumby | −0.0016 | [−0.0180, +0.0148] | 0.837 | 1/3 |
| B. Quarterly, unchanged-behaviour subset | Online Retail II | +0.0204 | [+0.0125, +0.0287] | 0.004 | 4/4 |
| Matched count, quarterly all (fuzzy matched − crisp) | Dunnhumby | +0.0027 | [−0.0020, +0.0079] | 1.000 | 3/3 |
| Matched count, quarterly all (fuzzy matched − crisp) | Online Retail II | +0.0028 | [−0.0006, +0.0060] | 0.222 | 3/4 |
| Matched count, refit (fuzzy matched − crisp) | Dunnhumby | −0.0093 | [−0.0224, −0.0015] | n/a | 0/4 |
| Matched count, refit (fuzzy matched − crisp) | Online Retail II | +0.0011 | [−0.0067, +0.0106] | n/a | 3/5 |

- Per-subsample and per-window values are in `results/segment_stability/` (`refit_per_subsample.csv`, `temporal_summary.csv`) and in its `matched_count_diagnostic/` subfolder.
- The matched-count rule is the v1 rule (`docs/AUDIT_ERRATA.md` §6.1): truncate fuzzy to the crisp concept count by the suppressor's own order. It is applied unchanged, with no search over counts.

## 4. Answers

| Question | Answer |
|---|---|
| Does v2 improve refit stability over v1? | **No.** v2 segments are identical to v1 fuzzy segments (verified on all 586 fits), so refit stability is identical. |
| Does v2 improve temporal stability over v1? | **No.** The same reason applies; temporal stability is identical to v1 fuzzy. |
| Does the crisp hybrid behave differently from the fuzzy hybrid? | Exactly as v1 crisp vs v1 fuzzy: no significant refit difference; the fuzzy hybrid is less stable under quarterly re-segmentation for all customers (−0.0346 Dunnhumby, −0.0201 Retail II); on the unchanged-behaviour subset there is no significant difference on Dunnhumby, and +0.0204 on Retail II. |
| Does matching concept count alter the conclusion? | It alters the temporal all-customer gap, which is no longer significant at matched count (+0.0027, +0.0028). This supports a "concept count and/or concept size" explanation, not count alone, and it does not show fuzzy segments are *more* stable. |
| Does any result justify a stability claim for v2? | **No.** v2 inherits v1's H3 conclusion (not supported). Nothing here comes from v2's predictive performance, and prediction stability was not evaluated. |

## 5. Limitations

- **Overlapping temporal windows.** Expanding windows share customers and history across consecutive origins, so the temporal bootstrap understates uncertainty, and pooled intervals are approximate.
- **Resampling inference.** Refit intervals come from 80% subsamples of the same cohort. They describe refit variability, not uncertainty about new populations, and no p-values are claimed for refit stability.
- **Post hoc.** v2 was designed after seeing v1. This check adds no new evidence about stability; it only establishes that v1's stability evidence applies to v2's segments.

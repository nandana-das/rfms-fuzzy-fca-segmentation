# Limitation follow-ups ("quick wins") — analysis plan

**Branch:** `experiment/limitations-quick-wins` (from `8f3493a`).
**Status:** analysis plan, committed on its own before any code for these analyses is written or run. These are additions addressing limitations listed in the project report. They do not change the frozen v1 evidence (`7607e09`), the exploratory v2 evidence (`f0342ef`) or the CDNOW confirmation (`cb1c352`), and they are reported separately. Frozen settings (quintile scoring, five fuzzy levels, thresholds 0.3/0.5/0.7, support 0.04, J_max 0.80, μ-cut 0.5, empty-core safeguard, models) are unchanged. On CDNOW the planned tie-merging extension (`docs/CDNOW_CONFIRMATION_PLAN.md` §4 and A1) applies. Nothing is tuned.

## Part A — Segment stability on CDNOW (limitation: "CDNOW stability not evaluated")

- **Protocol:** exactly the v1 stability protocol (`scripts/segment_stability.py`), applied to the three CDNOW origins (1997-09-30, 1997-12-31, 1998-03-31).
  - Study A: refit on 30 subsamples of 80% (seed 2026) vs the full-cohort fit at each origin.
  - Study B: quarterly re-segmentation at the 2 consecutive origin pairs, for all customers and for the unchanged-behaviour subset (|Δ percentile rank| ≤ 0.05 in each of R, F, M).
- **Arms:** crisp RFM-FCA, fuzzy RFM-FCA, and the matched-count diagnostic arm (fuzzy truncated to the crisp concept count by the suppressor's own order; the rule of `docs/AUDIT_ERRATA.md` §6.1). The v2 hybrid has the same segments as fuzzy RFM-FCA, so it needs no separate arm.
- **Primary measure:** core-profile Jaccard. Secondary: graded Jaccard and concept recurrence.
- **Comparisons:**
  - fuzzy − crisp;
  - matched − crisp;
  - matched − fuzzy.

  Temporal comparisons use the customer bootstrap (B = 2000) with Holm correction within CDNOW. Refit comparisons use percentile intervals over subsamples.
- **Decision rule:**
  - H3 "replicates as not supported" unless fuzzy − crisp core Jaccard is significantly positive for refit or for temporal all-customers.
  - Any significant positive result is reported as a dataset-specific stability advantage.

## Part B — Inference robustness (limitation: test-sample-only bootstrap, overlapping origins, p-value floor)

- **Scope:** the key comparisons, each on all three datasets with each dataset's own origins and settings. Fold seeds, the number of resamples and the bootstrap scheme change; nothing else does.
  - K1: fuzzy RFM-FCA − crisp RFM-FCA (v1 H1).
  - K2: fuzzy RFM-FCA − spline on log RFM (v1 H2).
  - K3: v2 hybrid − spline on log RFM (v2 P1 / CDNOW C3).
  - K4: v2 hybrid − fuzzy RFM-FCA (v2 P2 / CDNOW C4).
- **Arms run:** crisp RFM-FCA, fuzzy RFM-FCA, spline on log RFM, v2 hybrid.
- **Changes to inference:**
  1. **Repeated cross-validation.** 5 fold seeds (0, 1, 2, 3, 4) per origin, adding refit variability. Seed 0 reproduces the original folds.
  2. **Customer-clustered bootstrap across origins.** Each resample draws customers with replacement from the union of customers over a dataset's origins and applies the same draw to every origin. Customers present at several origins are therefore resampled jointly, instead of treating origins as independent. Metrics use resample weights: weighted rank AUC and weighted R².
  3. **B = 10,000 resamples**, using the same weights for every arm and every fold seed (paired).
- **Estimates:**
  - Δ per seed = mean over origins of (arm − reference).
  - Reported Δ = mean over the 5 seeds; the range over seeds is also reported.
- **Interval and p-value:** the mixture of the 5 × 10,000 draws, which includes both test-sample and refit variability.
  - 95% percentile interval.
  - Two-sided bootstrap p = 2 × min(tail), floor 1/50,000.
  - Holm correction within each dataset over 4 comparisons × 3 metrics = 12 tests.
- **Decision rule** (per comparison and metric, against the committed result):
  - "robust" if the original significance and sign are reproduced;
  - "weakened" if an originally significant result is not significant;
  - "strengthened" if an originally non-significant result becomes significant.

  All changes in either direction are reported. The committed results are not replaced.

## Part C — Interpretability proxies (limitation: interpretability not measured)

These are outcome-free structural proxies, not measures of human interpretability; no claim of greater interpretability follows from them alone.

- **Fits:** crisp and fuzzy RFM-FCA (v2 segments = fuzzy segments) fitted on the full cohort at every pooled origin of all three datasets.
- **Measures:**
  1. **K:** number of retained concepts.
  2. **Coverage:** share of customers with at least one core concept (membership ≥ 0.5).
  3. **C80:** concepts needed to cover 80% of customers in their core extents, by greedy maximum coverage (repeatedly add the concept covering the most uncovered customers). Reported as not reachable if 80% cannot be covered.
  4. **Intent length:** mean number of distinct bands per concept.
  5. **Core load:** mean number of core concepts per customer.
  6. **Overlap:** mean pairwise Jaccard between the core extents of retained concepts.
- **Reporting:** mean and range over origins per dataset and arm. Descriptive only, with no significance tests.
  - Smaller C80, shorter intents, lower core load and lower overlap are described as "more compact", without claiming that this makes segments more interpretable to people.

## Outputs

- Scripts: `scripts/qw_cdnow_stability.py`, `scripts/qw_inference_robustness.py`, `scripts/qw_interpretability_proxies.py`.
- Results: `results/qw_cdnow_stability/`, `results/qw_inference_robustness/`, `results/qw_interpretability/`.
- Write-up: `docs/QUICK_WINS_RESULTS.md`.

Any implementation failure is documented before re-running; no setting is changed in response to results.

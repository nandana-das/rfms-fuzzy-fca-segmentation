# Audit Errata and Fixes (2026-10-08)

This file records the defects found in the October 2026 audit, what was fixed, which artifacts are invalidated, and the corrected evidence. Sections 4–5 were updated after the switch to quintile scoring; §6–7 record the stability evidence and the final freeze. Treat it as the source of truth over older documents when they disagree.

## 1. Code defects fixed

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `dense_rank_scores` always produced 5 bands (`fair_comparison_retail2.py`) | Stage 1 configs with 3–4 levels built centroids from only the lowest bands and saturated the rest: truncation, not coarser granularity | Added `n_levels` (int or per-dimension dict, default 5 = unchanged behaviour); Stage 1 and both smoke tests now pass it |
| 2 | Shared `_piecewise_membership_generic` / `_band_centroids_generic` (`fuzzy_membership_sensitivity.py`) used means and `max(d_left, d_right)/span` | Stage 2 "Optimized FCA" used a different, non-triangular membership function than every baseline | Replaced with the canonical median-centroid piecewise-linear functions, generalized to n levels; verified identical to `piecewise_membership`/`band_centroids` for 5 levels |
| 3 | Stage 2 baseline B PR AUC computed on crisp one-hot test features (`optimized_fuzzy_fca_stage2_run.py`, `optimized_fuzzy_fca_stage2.py`) | The only "significant" gain of Optimized FCA over the fuzzy baseline (PR AUC +0.0045) is an artifact; Stage 1 reports 0.9863 for the same pipeline vs 0.9822 here | Test customers projected with the train fuzzy centroids |
| 4 | Stage 2 baseline C trained on canonical memberships but tested on the faulty generic ones | "Existing Kuznetsov FCA (loss <= 1.0)", which should equal baseline B, scored 0.832 / 0.421 / 0.519 | Same membership function for train and test |
| 5 | Stage 2 "Raw RFM" was crisp RFM-FCA | Stage 2 deltas "vs Raw RFM" are really vs crisp RFM-FCA | Relabelled "Crisp RFM-FCA" (JSON keys kept for compatibility) |
| 6 | `four_way_method_comparison.py` referenced the undefined `train_fuzzy_mu` in `_kuznetsov_stability_filter_records` | The current script could not regenerate its own artifacts (NameError) | Uses the locally built `train_fuzzy_mu_for_stab` |
| 7 | `suppress_redundant_concepts` skipped every Jaccard test for concepts with an empty mu >= 0.5 extent | Such concepts (e.g. two adjacent bands of one dimension) and their exact duplicates were always kept: on one Dunnhumby split, 127 kept concepts gave only 67 distinct feature columns, and 86 had empty cores | `drop_empty_core=True` by default (`False` reproduces legacy behaviour); same fix in the sparse variant |
| 8 | Paired sign-flip test was Monte Carlo without a +1 correction | Reported p = 0.0015 is below the exact floor 2/2^10 = 0.00195 for n = 10 | Exact enumeration for n <= 20; (b + 1)/(B + 1) otherwise |
| 9 | Three Stage 2 scripts set `ROOT_DIR` to `scripts/` | Results were written to a stray `scripts/results/`, and the two copies differ slightly | Output now goes to `results/`. The stray folder was removed after its unique files were moved (§7.1) |

## 2. Documentation corrections

- **Dunnhumby holdout repurchase rate is 93.16% (2,328 of 2,499 households), not 98.08% / 2,451.** This was verified from the raw transactions; mean future spend ($482.05) and baskets (15.05) were already correct. There are 171 non-repurchasers in total, about 51 per 30% test split, so AUC differences of about 0.01 are within noise. PR AUC at this prevalence carries little information.
- The "10-split temporal cross-validation" was 10 repeated random 70/30 customer splits at **one** temporal cutoff. It has been relabelled in README and docs.
- Erratum banners were first added to README, METHODOLOGY_V4 (now `METHODOLOGY_FINAL.md`), PROJECT_DOCUMENT, RESULTS_SUMMARY, results_section and final_methodology_contribution_framing. At the final freeze (§7) these documents were rewritten with the current evidence, and their pre-audit versions moved to `docs/superseded_2026-10-08/`.

## 3. Invalidated or caveated artifacts (not re-run)

- `results/optimized_fuzzy_fca/stage1_multiseed_*`: **invalid** (defect 1). Granularity was never tested.
- `results/optimized_fuzzy_fca/stage2/*`, including `from_stray_scripts_results/` (the unique files rescued from the stray `scripts/results/` folder, which was then removed; §7.1): **invalid** (defects 2–5).
- `results/four_way_method_comparison/`: caveated. The Raw RFM baseline is misspecified, the Kuznetsov threshold was selected on the same seeds, and the p-values are below the exact floor (defects 7–8).
- `results/kuznetsov_pruning_*leakage_free/`: caveated. The threshold 5.4e-20 was selected as the best of 5 arms on the evaluation seeds. It is below float64 resolution of 1 − stability, so it equals `float(stability) == 1.0`, which is roughly a rule of 53 or more customers distinguishing the concept from each lower neighbour.

Stage 1/2 were deliberately not re-run. The audit found that this line of work is hyperparameter tuning and does not answer the research question.

## 4. Scoring changed from dense rank to tie-preserving quintiles

Dense-rank scoring, $\lceil 5 \cdot \text{dense\_rank}/K \rceil$, divides the range of *distinct values* into five groups, not the customers. On the full cohorts it placed 84–87% of Dunnhumby households in one Recency band, and 94–95% of Online Retail II customers in one Frequency band. That makes the RFM levels meaningless for segmentation, so it was replaced (author decision, 2026-10-08).

The new rule (`fit_quantile_cutoffs` / `quantile_scores` in `fair_comparison_retail2.py`):
- Cutpoints at the 20/40/60/80th percentiles, fitted on training customers. Band $k$ holds values in $(q_{(k-1)/5}, q_{k/5}]$, and R is inverted.
- Identical raw values always share a band, and the same cutpoints project unseen customers.
- The base paper's rank-first tie-breaking is not used. It splits identical customers across bands and cannot be applied out of sample.
- The scorer raises an error if two cutpoints coincide, so an empty band can never pass silently.

Band occupancy is reported per origin in `results/baseline_ladder_rolling_origin/band_occupancy.csv` and in that folder's REPORT.md. Range over all origins:

| Dataset | R | F | M |
|---|---|---|---|
| Dunnhumby | 15.6–26.8% (mass at R = 0) | 19.2–20.9% | 20.0% |
| Online Retail II | 18.5–21.7% | 11.7–37.9% (about 33% of customers have F = 1) | 20.0% |

`dense_rank_scores` is kept, for the historical scripts only. All historical result folders used dense rank.

## 5. Corrected evidence: baseline ladder with rolling-origin evaluation (quintile scoring)

`scripts/baseline_ladder_rolling_origin.py` → `results/baseline_ladder_rolling_origin/`; the dense-rank run is archived in `superseded_dense_rank_scoring/`. Protocol:
- 4 Dunnhumby origins (days 347/438/529/620) and 5 Online Retail II origins (2010-09-10 … 2011-09-09), each with a 91-day holdout.
- Stratified 5-fold CV over customers within each origin; every representation, including the quintile cutpoints, is fit on the training folds.
- Paired customer-level bootstrap with Holm correction.

Pooled deltas for fuzzy RFM-FCA with quintile scoring (Holm-adjusted p; origins with a positive delta):

| Reference | Dunnhumby Spend R² | Dunnhumby Invoice R² | Retail II Spend R² | Retail II Invoice R² |
|---|---|---|---|---|
| Raw standardized RFM (old baseline) | +0.102 (.014, 4/4) | +0.114 (.014, 4/4) | +0.101 (.014, 5/5) | +0.042 (.014, 5/5) |
| log1p RFM, 3 features | +0.019 (.014, 4/4) | +0.016 (.014, 4/4) | +0.015 (.014, 5/5) | +0.004 (ns, 4/5) |
| **Spline on log RFM** | **−0.006 (ns, 1/4)** | **−0.003 (ns, 1/4)** | **+0.002 (ns, 4/5)** | **−0.032 (.014, 0/5)** |
| **Crisp RFM-FCA, quintile (base-paper analogue)** | **+0.027 (.014, 4/4)** | **+0.021 (.014, 4/4)** | **+0.014 (.014, 5/5)** | **+0.027 (.014, 5/5)** |
| Fuzzy bands, quintile (no concepts) | −0.001 (ns) | +0.006 (.034, 4/4) | +0.008 (.014, 5/5) | +0.019 (.014, 5/5) |
| Same pipeline with dense-rank scoring | −0.002 (ns) | +0.003 (ns) | −0.002 (ns) | **−0.028 (.014, 0/5)** |

In the table, ns means not significant after Holm correction.

ROC AUC differences vs crisp RFM-FCA are +0.021 (Dunnhumby) and +0.006 (Retail II), both p = .014 on every origin. Against the spline baseline, AUC differences are within ±0.005 and not significant after Holm.

**The scoring trade-off on Online Retail II invoices.** The membership functions saturate above the top-band centroid, which is the median of band 5. With quintiles that centroid sits much lower in the heavy F tail than with dense rank: everyone above roughly the 90th percentile of F gets the same F membership. Dense rank happened to keep tail resolution only because its top bands were nearly empty.
- The cost is about 0.03 Invoice R² on Retail II, and none on Dunnhumby.
- It is a property of the outer-shoulder membership design, not of quintiles as such.
- **Decision (2026-10-08): keep the saturating outer shoulders.** RFM levels are ordinal segments: "F5" means "top 20% by frequency", and crisp quintiles, including the base paper's, discard within-band magnitude in the same way. The alternatives were rejected:
  - Anchoring band 5 at the band maximum would place it at an extreme outlier and leave the top level almost empty.
  - A log-scale centroid leaves the saturation point unchanged, because medians do not move under monotone transforms.
  - A sixth level would break the five-level correspondence with the base paper and would be a post hoc change.
- **Stated limitation:** ordinal RFM levels, crisp or fuzzy, give up magnitude information inside the top band. With heavy-tailed frequency (Online Retail II) this costs about 0.03 Invoice R² against a continuous spline model; on Dunnhumby it costs nothing. Magnitude-level forecasting should use continuous RFM; the FCA levels serve interpretable segmentation.

Concept counts (mean over folds and origins):

| | Dunnhumby | Retail II |
|---|---|---|
| Fuzzy RFM-FCA, quintile | 91.6 | 76.8 |
| Fuzzy RFM-FCA, dense rank | 33.5 | 40.3 |
| Crisp RFM-FCA, quintile | 54.1 | 48.9 |
| Kuznetsov-FCA, quintile | 30.4 | 51.9 |

Balanced bands let more conjunctions reach the 4% support level. All counts are after the empty-core fix (§1, defect 7).

**What the evidence supports:**
1. Fuzzy RFM-FCA beats crisp RFM-FCA on all three metrics, at every origin, on both datasets. This is the defensible extension of the base paper, and it is robust to the scoring change.
2. It does **not** beat a properly specified non-FCA model (spline on log RFM). It reaches parity on Dunnhumby and on Retail II spend and AUC, and falls short on Retail II invoices.
3. Concept conjunctions add little on Dunnhumby, but help on Retail II relative to single fuzzy bands. Under quintile scoring, the conjunctions partly compensate for saturation of the band memberships.
4. Kuznetsov stability yields fewer concepts at broadly equal accuracy (largest pooled difference about 0.003). It is a compactness result, not a predictive one. §6 later showed it is also the least stable arm, and it is excluded from the final method (§7).

## 6. Segment-stability experiment (structural-robustness hypothesis)

`scripts/segment_stability.py` → `results/segment_stability/`. Protocol fixed before running:
- Arms: crisp RFM-FCA, fuzzy RFM-FCA and Kuznetsov-FCA, all with quintile scoring.
- Primary measure: core-profile Jaccard, i.e. the set of concepts with membership ≥ 0.5 per customer, so fuzzy gets no partial credit for graded memberships.
- Study A: refit on 30 random 80% subsamples and compare with the full-data fit.
- Study B: refit at consecutive origins 91 days apart, for all customers and for customers whose R/F/M percentile ranks moved by ≤ 0.05.

| Primary measure (core Jaccard) | Dunnhumby fuzzy − crisp | Retail II fuzzy − crisp | Dunnhumby Kuznetsov − crisp | Retail II Kuznetsov − crisp |
|---|---|---|---|---|
| A. Refit stability | −0.005 (interval spans 0) | +0.005 (spans 0) | −0.141 | −0.065 |
| B. Re-segmentation, all customers | −0.035 (0/3 pairs) | −0.020 (0/4) | −0.009 | −0.033 |
| B. Re-segmentation, behaviourally stable | −0.002 (ns) | **+0.020 (4/4)** | −0.105 | −0.068 |

Unless marked ns or "spans 0", all study B differences are significant after Holm correction. Absolute levels: about 0.95–0.97 for refits, about 0.45–0.53 for quarterly re-segmentation of all customers, and about 0.88 for behaviourally stable customers.

**Conclusions:**
1. **The hypothesis that fuzzy RFM-FCA gives more stable segments than crisp RFM-FCA is not supported.**
   - Refit stability is equal.
   - Over time, fuzzy profiles change *more* than crisp ones.
   - The single favourable result is Retail II stable customers (+0.020). It does not replicate on Dunnhumby.
   - Fuzzy fits have more concepts (about 92 vs 55 on Dunnhumby) and more core concepts per customer, which gives more room for churn. That is a property of the method at the locked support threshold, not a confound to be adjusted away after the fact.
2. **Kuznetsov-FCA is the least stable arm on every measure.**
   - On Dunnhumby, only 64% of its concepts recur across refits.
   - Its stability rule is an absolute customer-count criterion, so it shifts with sample size.
   - This contradicts using "stability" as a selling point and supports removing it from the method.
3. **Quarterly re-segmentation changes about half of each customer's concept profile under every arm.** This is a substantive finding about RFM-FCA segments in general.

### 6.1 Matched-concept-count diagnostic: rule specified before computing any matched result

This rule was specified after the main stability results (above) were known, since the diagnostic exists to explain them, but before any matched-count result was computed. It is a diagnostic, not a method variant.

- **Matching rule (single, no search).** For every fit, let K be the number of concepts that crisp RFM-FCA retains on the *same* customers. Fuzzy RFM-FCA is fitted unchanged. Its retained concepts are ranked by the order the redundancy suppressor already uses (support descending, intent size ascending, stability proxy descending, then original row order), and the top K are kept. If fuzzy retains ≤ K concepts, all are kept. Crisp is untouched.
- **Everything else is identical** to §6: the same origins, the same 30 subsamples (seed 2026), the same consecutive-origin pairs and behaviourally-stable subset, and core-profile Jaccard as the primary measure.
- **Known asymmetry.** Truncating by support keeps fuzzy's largest concepts, which may be inherently more stable, while crisp keeps all of its concepts. This can only *favour* the matched fuzzy arm.
- **Interpretation rule (fixed in advance).** The target is the gap that was clearly negative: temporal re-segmentation, all customers, core Jaccard.
  - **"Explained by count":** the matched gap (fuzzy-matched − crisp) is ≥ 0 or its 95% interval includes 0, on both datasets.
  - **"Partly explained":** the gap shrinks by at least half but stays significantly negative.
  - **"Not explained":** otherwise.
  - Refit stability and the behaviourally-stable subset are reported alongside but do not change the verdict. Because of the asymmetry above, a favourable matched result supports "count *and/or* concept size", not count alone.

### 6.2 Matched-concept-count diagnostic: result

`scripts/matched_count_stability_diagnostic.py` → `results/segment_stability/matched_count_diagnostic/`. The run reused §6's study code unchanged, and its full-fuzzy and crisp arms reproduce §6 exactly.

| Target (temporal, all customers, core Jaccard) | Full fuzzy − crisp | Matched fuzzy − crisp | Verdict (rule in §6.1) |
|---|---|---|---|
| Dunnhumby | −0.035 | +0.003 [−0.002, +0.008] | **explained by count** |
| Online Retail II | −0.020 | +0.003 [−0.001, +0.006] | **explained by count** |

Secondary results (they do not change the verdict):
- Refit at matched count: Dunnhumby −0.009 [−0.022, −0.002], Retail II +0.001 (ns).
- Unchanged-behaviour subset: Dunnhumby −0.008 (ns), Retail II +0.015 (significant).

Following the pre-stated asymmetry, the attribution is to "concept count and/or concept size". The diagnostic is valid as specified; it explains the temporal deficit but provides no evidence that fuzzification *improves* stability. H3 remains **not supported**.

## 7. Final freeze (2026-10-08)

- **Methodology frozen** as described in `docs/METHODOLOGY_FINAL.md` (renamed from `METHODOLOGY_V4.md` at the pre-commit review): quintile scoring, five fuzzy levels, saturating shoulders, thresholds 0.3/0.5/0.7, support 0.04, J_max 0.80, μ-cut 0.5, empty-core suppression.
  - The membership mathematics was verified as consistent in all 60 training fits (strictly increasing centroids; memberships sum to 1), so it was not changed.
  - Two documentation inconsistencies were corrected in the docs only. (a) Kneedle pruning had been described as part of the method but is not applied in the evaluated pipeline. (b) "L-fuzzy context" overstated a threshold-scaled binary context whose concept features are Gödel minima over band memberships.
- **Kuznetsov-FCA excluded from the final method.** Its code, reports and results are preserved as ablation evidence. Reasons:
  - no demonstrated predictive benefit (one significant pooled AUC gain on Retail II, +0.003, positive at 1/5 origins);
  - lower refit and temporal stability than crisp RFM-FCA on every measure;
  - a threshold (loss ≤ 5.4e-20) below float64 resolution that acts as an absolute customer-count rule depending on training-set size;
  - the threshold was originally selected on evaluation seeds.
- **Final research question and hypotheses:** see `docs/PROJECT_DOCUMENT.md` §2. H1 supported, H2 inconclusive, H3 not supported.
- **Documents rewritten** with current quintile-based evidence: README, PROJECT_DOCUMENT, METHODOLOGY_V4 (now `METHODOLOGY_FINAL.md`), RESULTS_SUMMARY, results_section and final_methodology_contribution_framing. Pre-audit versions (from git HEAD 8a420e2) are in `docs/superseded_2026-10-08/`, bannered as superseded.
- **Evidence tables** are generated by `scripts/final_evidence_tables.py` → `results/final_evidence/`.
- No further experiments are to be run unless an implementation error is discovered.

## 7.1 Pre-commit review changes

- **Hypothesis wording.** H1 is now phrased as improvement over crisp RFM-FCA under the rolling-origin evaluation, distinguishing pooled Holm-corrected significance from descriptive per-origin wins. H2 states that 5/6 spline comparisons were not significantly different, that equivalence was not formally established, and that Retail II Invoice R² was significantly worse by 0.032. H3 keeps the matched-count result with its count and/or concept-size caveat. Equality phrasing about refit stability was replaced by "no significant difference".
- **Kuznetsov** was removed from the contribution list and kept only as an ablation / negative result. The empty-core rule is described as an implementation safeguard, not a contribution.
- **`scripts/replicate_base_paper.py`** now computes the Table 7 recovery (31/31 intents, 3/31 exact counts), the customer count and the cleaning reconciliation from the actual reconstruction code instead of hard-coded text. The verified values are unchanged. Its old explanation of the 71-line gap ("cancellation regex parsing") was wrong and has been replaced.
- **Retail II cleaning discrepancy (unresolved).** Our cleaning follows the base paper's stated rules and yields 805,549 lines and 5,878 customers; the paper reports 805,620 lines and 5,878 customers. The 71-line gap equals exactly the number of zero-price lines (UnitPrice = 0) that pass the other filters: keeping them reproduces 805,620 lines, but then the customer count becomes 5,881. No single cleaning rule reproduces both published numbers, so this is recorded as an unresolved inconsistency in the base paper's reported counts; our pipeline keeps the stated rule (price > 0).
- **Renamed** `docs/METHODOLOGY_V4.md` → `docs/METHODOLOGY_FINAL.md`.
- **`replicate_base_paper.py` comparison table cleaned.** Thirteen pre-audit rows with stale hard-coded values were removed from the active table: concept counts ("33", "359"), K-means/Ward/FCA silhouettes and Davies-Bouldin values, FCM k = 76, Olist and old predictive holdouts. The remaining rows are either computed in the run (customers, cleaned lines, Table 7 recovery) or configuration facts, labelled in a new "Value source" column. The script was not re-run, so the committed `results/base_paper_numerical_comparison.csv` and `results/base_paper_replication/` are historical artifacts that still contain the old values. Neither is part of the final evidence.
- **Stray copies removed.** The stray `scripts/results/` folder (created by a Stage 2 path bug) and the scratch file `stage2_func.txt` were removed. Unique historical files from `scripts/results/` were first moved to `results/optimized_fuzzy_fca/stage2/from_stray_scripts_results/`.

## 8. Remaining author actions

- Review the final package, then commit the untracked scripts and results so the paper is reproducible from version control.

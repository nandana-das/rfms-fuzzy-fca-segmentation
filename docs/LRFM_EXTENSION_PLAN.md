# LRFM extension (adding Length, L) — analysis plan

**Branch:** `experiment/lrfm-extension` (to be created from `0617bcd`, tip of `experiment/limitations-quick-wins`).
**Status:** analysis plan, committed on its own before any code for this analysis is written or run.
**Evidential status of the results: exploratory.** The three datasets have already been analysed, so these results cannot confirm anything. Confirmation requires a dataset not yet analysed, under its own plan, committed before that analysis.
This extension does not change the frozen v1 evidence (`7607e09`), the exploratory v2 evidence (`f0342ef`), the CDNOW confirmation (`cb1c352`) or the limitation follow-ups (`b5a464d`). Its results are reported separately.

## 1. Motivation

All RFM-only representations reach similar predictive performance, which suggests the ceiling is set by the information in R, F and M rather than by the representation. The base paper lists adding variables to RFM as future work, naming LRFM among others. This plan adds one variable, Length (L), to test whether more behavioural information raises performance and whether the fuzzy-versus-crisp and fuzzy-versus-baseline conclusions persist.

## 2. Variable definition

- **L = number of days between a customer's first and last purchase within the observation window** (all transactions up to and including the origin cutoff). It is computed only from pre-cutoff transactions, so the holdout cannot leak into it.
  - Online Retail II: `(max(InvoiceDate) − min(InvoiceDate)).days`, on the same cleaned transactions as v1.
  - Dunnhumby: `max(DAY) − min(DAY)`.
  - CDNOW: `(max(date) − min(date)).days`.
- Customers with a single purchase day have L = 0.
- Higher L receives a higher band (L is not inverted; only R is).
- **Known censoring:** L cannot exceed the time since the start of each dataset. This affects Online Retail II and Dunnhumby customers whose history began before the data window. CDNOW's cohort first purchased in 1997 Q1, so L is nearly uncensored there. This is reported as a limitation, not corrected.
- **Definition choice:** LRFM definitions vary across the literature; this plan fixes the definition above and no alternative will be tried.

## 3. Representation

The frozen v1 pipeline is extended from three to four dimensions (R, F, M, L), with every setting unchanged:
- train-fitted, tie-preserving quintile cutpoints;
- centroid piecewise-linear fuzzy memberships with saturating shoulders;
- threshold-scaled context at 0.3 / 0.5 / 0.7 (60 binary attributes instead of 45);
- support 0.04, Jaccard suppression with J_max 0.80 and μ-cut 0.5, and the empty-core safeguard;
- the same downstream models (LogisticRegressionCV, RidgeCV) and targets.

**Ties at L = 0.** Single-purchase customers make L = 0 a large mass point, so quintile cutpoints can coincide. The committed CDNOW tie/empty-band merging rule (`merged_cutpoints`, plan `04ce2cd`, addendum A1 `48b354a`) is applied to all four dimensions on all three datasets. That rule changes nothing when every band is populated, so R, F and M are unaffected on Online Retail II and Dunnhumby; §6 verifies this. L may have fewer than five levels at some origins; the number of levels is reported.

## 4. Arms

| Arm | Features | Role |
|---|---|---|
| `spline_log_rfm` | committed v1 arm | RFM reference (reproduction guard) |
| `fuzzy_rfm_fca` | committed v1 arm | RFM reference (reproduction guard) |
| `hybrid_fuzzy_fca` | committed v2 arm | RFM reference (reproduction guard) |
| `log_lrfm` | standardized log1p(R, F, M, L) | descriptive baseline (not tested) |
| `spline_log_lrfm` | cubic spline, 5 quantile knots, on log1p(R, F, M, L) | strong non-FCA baseline |
| `crisp_lrfm_fca` | crisp closed concepts on the 4-dimension context | base-paper method with L |
| `fuzzy_lrfm_fca` | fuzzy closed concepts on the 4-dimension context | proposed method with L |
| `hybrid_lrfm` | `fuzzy_lrfm_fca` memberships + standardized log1p(R, F, M, L) | hybrid with L |

Every baseline receives L, so any gain from L is available to all arms.

## 5. Comparisons (fixed before analysis)

Per dataset, each on ROC AUC, Spend R² and Invoice R²:

| ID | Comparison | Question |
|---|---|---|
| E1 | `fuzzy_lrfm_fca` − `crisp_lrfm_fca` | Does the fuzzy gain over crisp persist with L? |
| E2 | `fuzzy_lrfm_fca` − `spline_log_lrfm` | Fuzzy FCA vs a strong baseline with the same information |
| E3 | `hybrid_lrfm` − `spline_log_lrfm` | Hybrid vs a strong baseline with the same information |
| E4 | `fuzzy_lrfm_fca` − `fuzzy_rfm_fca` | Does L improve fuzzy RFM-FCA? |
| E5 | `spline_log_lrfm` − `spline_log_rfm` | Does L improve the strong baseline? |

5 comparisons × 3 metrics = **15 tests per dataset**.

## 6. Inference and reproduction guard

- **Inference:** the procedure of the limitation follow-ups (plan `67976b4`, Part B), used as the sole procedure:
  - 5 CV fold seeds (0–4) per origin;
  - customer-clustered bootstrap across origins, with B = 10,000 resamples per seed, paired across arms and seeds;
  - Δ = mean over seeds of the mean over origins; the range over seeds is reported;
  - 95% percentile interval and two-sided p from the mixture of the 5 × 10,000 draws;
  - Holm correction within each dataset over the 15 tests, giving a floor of 15 / 50,000 = 0.0003.
- **Origins:** each dataset's committed pooled origins (Online Retail II 5, Dunnhumby 4, CDNOW 3), each with a 91-day holdout.
- **Reproduction guard**, checked before any LRFM result is written; the run aborts if either check fails:
  1. at seed 0, `spline_log_rfm`, `fuzzy_rfm_fca` and `hybrid_fuzzy_fca` reproduce their committed per-origin metrics within 1e-9;
  2. on Online Retail II and Dunnhumby, the merged cutpoints for R, F and M equal the unmerged ones at every training fold.

## 7. Claim rules

- Each comparison is reported per dataset and metric with Δ, interval, Holm p and range over seeds. Results are never pooled across datasets and never summarised only as a count.
- "Significant" means Holm p < 0.05. Not significant is not evidence of equivalence.
- E4 significantly positive means L improves fuzzy RFM-FCA on that dataset and metric.
- E4 and E5 are not tested against each other. No claim that L helps FCA more (or less) than the spline is made.
- E1–E3 are compared with their RFM-only counterparts (K1–K3 of the limitation follow-ups) descriptively, not by a new test.
- Every result is labelled exploratory. A non-retail confirmation will need its own plan.

## 8. Descriptive outputs (outcome-free)

Per dataset and origin:
- the distribution of L: share with L = 0, quartiles and maximum;
- L band occupancy and its number of levels after merging;
- Spearman correlation of L with R, F and M;
- retained concept counts for `crisp_lrfm_fca` and `fuzzy_lrfm_fca`.

## 9. Out of scope

- Any other added variable, including purchase regularity.
- Tuning any setting: levels, thresholds, support, J_max, μ-cut or models.
- Segment stability and interpretability proxies with L.
- The non-retail dataset.

## 10. Outputs

- Script: `scripts/lrfm_extension.py`. It reuses the frozen ladder, the v2 builder, the CDNOW merging rule and the follow-up inference functions unchanged; only the cohort builders (adding L), the dimension tuple and the arm and comparison lists are added.
- Results: `results/lrfm_extension/`.
- Write-up: `docs/LRFM_EXTENSION_RESULTS.md`. Evidence-map rows are added later in a separate documentation commit.
- Raw CDNOW data stays git-ignored (SHA-256 as in plan `04ce2cd`).

## 11. Failure handling

Any implementation failure (including the guard failing, or concept mining becoming infeasible with 60 attributes) is documented in an addendum committed before re-running. No setting is changed in response to results.

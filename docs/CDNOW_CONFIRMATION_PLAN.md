# CDNOW confirmation — analysis plan (committed before any CDNOW analysis code or outcome data is examined)

**Branch:** `experiment/cdnow-confirmation` (from v2 commit `ccdaa9a`).
**Purpose:** an out-of-sample confirmation, on a third dataset never used before in this project, of:
- the frozen v1 findings (commit `7607e09`): H1 fuzzy vs crisp RFM-FCA, and H2 fuzzy vs strong non-FCA baselines;
- the exploratory v2 hybrid findings (commit `f0342ef`).

**Status:** analysis plan, not a formal preregistration. It is committed before any CDNOW modelling code exists or any holdout outcome is computed.

## 1. Data

- **Source:** Fader & Hardie, CDNOW master data set. Downloaded 2026-10-09 from `http://brucehardie.com/datasets/CDNOW_master.zip`, SHA-256 `94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49`.
- **Licence:** none stated on the source page. Used for academic research with citation:
  - Fader, P. S. & Hardie, B. G. S. (2001). Forecasting Repeat Sales at CDNOW: A Case Study. *Interfaces* 31(3), S94–S107.

  The raw file is **not committed** (it is git-ignored); this plan records how to obtain it.
- **Contents:** 23,570 customers who made their first-ever CDNOW purchase in Q1 1997, with all purchases to 1998-06-30. There are 69,659 records, each with customer ID, date (day resolution), number of CDs and dollar value.

**What was examined before writing this plan (covariates only).** Record counts, invalid values, and the R/F/M distributions at the candidate origins, computed from observation-window data only. No holdout data, repurchase, spend or model output was computed. Findings:
- 80 records have dollar value ≤ 0.
- 2,068 records repeat a customer–day pair.
- At all three origins, 53–60% of customers have exactly one purchase day. The F quintile cutpoints are therefore (1, 1, 2, 3): the 20th and 40th percentiles coincide. R and M cutpoints are strictly increasing.

## 2. RFM definitions (fixed)

- **Cleaning:** drop records with dollar value ≤ 0. This mirrors the frozen Online Retail II rule (non-positive line values removed). No other cleaning.
- **Order:** a distinct (customer, purchase day). There is no order ID, and same-day records are treated as one order, the CDNOW convention.
- At observation cutoff T:
  - **R** = days from the last purchase day to T;
  - **F** = number of distinct purchase days up to T;
  - **M** = total dollar value up to T.
- **Holdout** (T, T + 91 days]:
  - repurchase = at least one purchase day;
  - future spend = log1p(dollar value);
  - future orders = log1p(distinct purchase days).
- **Cohort:** every customer with at least one valid record up to T (all customers started in Q1 1997).

## 3. Origins (fixed)

Three rolling origins with a 91-day holdout:
- 1997-09-30 (holdout to 1997-12-30);
- 1997-12-31 (to 1998-04-01);
- 1998-03-31 (to 1998-06-30).

All three are pooled; there is no separate primary protocol.

## 4. Method: frozen v1 / v2, with one pre-specified extension

Everything is as frozen in `docs/METHODOLOGY_FINAL.md`, with no tuning:
- quintile scoring fitted on training folds, five fuzzy levels, band-median centroids, saturating shoulders;
- threshold scaling 0.3/0.5/0.7, support 0.04, J_max 0.80, μ-cut 0.5, empty-core safeguard;
- Gödel-min membership;
- the same LogisticRegressionCV / RidgeCV models, stratified 5-fold CV within each origin, CV seed 0, bootstrap B = 2000, seed 12345.

The v2 hybrid appends standardized log1p(R, F, M), exactly as in `scripts/v2_hybrid_ladder.py`.

**Extension (the only change): tie-merging for coinciding quintile cutpoints.**
- When two or more quintile cutpoints of a dimension coincide (fitted on the training folds), the duplicates are merged and that dimension gets fewer levels: unique cutpoints + 1. Bands keep the tie-preserving rule (q₍ₖ₋₁₎ < x ≤ qₖ).
- Fuzzy memberships use the same centroid and piecewise-linear construction with that number of levels. This is the generalized implementation verified identical to the 5-level one, and it still sums to 1 per dimension.
- On CDNOW this is expected to give F four levels: ≤ 1, 2, 3, ≥ 4 purchase days. R and M keep five.
- Crisp RFM-FCA uses the same merged bands.
- The extension changes nothing when cutpoints are strictly increasing, as in every Dunnhumby and Online Retail II fit, so frozen v1/v2 results are unaffected.

## 5. Arms and comparisons (fixed)

**Arms:**
- `raw_std`, `log_rfm`, `spline_log_rfm`;
- `crisp_rfm_fca`, `fuzzy_rfm_fca`;
- `hybrid_crisp_fca`, `hybrid_fuzzy_fca`.

**Comparisons:** Δ = arm − reference, pooled over the 3 origins; paired customer bootstrap; Holm correction over all 7 × 3 = 21 tests.

| ID | Comparison | Tests |
|---|---|---|
| C1 | fuzzy_rfm_fca − crisp_rfm_fca | v1 H1 |
| C2 | fuzzy_rfm_fca − spline_log_rfm | v1 H2 |
| C3 | hybrid_fuzzy_fca − spline_log_rfm | v2 P1 |
| C4 | hybrid_fuzzy_fca − fuzzy_rfm_fca | v2 P2 |
| C5 | hybrid_fuzzy_fca − hybrid_crisp_fca | v2 P3 |
| C6 | hybrid_fuzzy_fca − log_rfm | v2 P4 |
| C7 | hybrid_fuzzy_fca − crisp_rfm_fca | v2 P5 |

Metrics: ROC AUC, Spend R² and Invoice R² (here, future purchase days).

## 6. Decision rules (fixed now)

These apply per finding, with "significant" meaning Holm p < 0.05 and "Δ" the pooled estimate.

- **v1 H1 replicates** if C1 is significantly positive on all 3 metrics.
  - Partially replicates if significantly positive on 1–2 metrics and not significantly negative on any.
  - Fails to replicate otherwise.
- **v1 H2.** Not significantly different on a metric is reported as such, never as equivalence. A significant C2 in either direction is reported as superiority or shortfall.
- **v2 "beats spline" confirmed** only on metrics where C3 is significantly positive.
- **v2 "improves on v1" confirmed** only on metrics where C4 is significantly positive, with none significantly negative.
- **v2 "concepts add beyond linear log RFM"** only where C6 is significantly positive. This still does not show value beyond nonlinear RFM; that question is C3.
- **v2 "fuzzification helps within the hybrid"** only where C5 is significantly positive.
- **Every comparison is reported**, including unfavourable ones. Counts of origins with a positive Δ are descriptive only.
- **Limits of inference.** With 3 origins and B = 2000, the smallest attainable Holm p-value is 0.0005 × 21 = 0.0105. The bootstrap captures test-sample variability only, and origins share customers.

## 7. Out of scope

- Segment stability (v2 segments equal v1 segments by construction; a CDNOW stability study is not part of this confirmation).
- Any tuning or re-running with other settings.

If the run fails for an implementation reason, the fix and the reason will be documented before re-running; no setting will be changed in response to results.

## 8. Outputs

- Script: `scripts/cdnow_confirmation.py`
- Results: `results/cdnow_confirmation/`
- Write-up: `docs/CDNOW_CONFIRMATION_RESULTS.md`

# Second CV seed plan: FCA+T vs spline+T

Status: written 2026-10-14 before any second-seed run. Exploratory, post hoc replication of
`results/temporal_nonfca_baseline/` (seed 0).

## Scope

Only two comparisons are run (priorities 1–2). Everything else stays frozen at seed 0 (M0 vs crisp,
M0 vs log-RFM, matched-complexity arms, Step 0, Łukasiewicz).

| # | Dataset | Comparison | Seed-0 delta (AUC / Spend R² / Invoice R²) |
|---|---|---|---|
| 1 | Online Retail II | `fuzzy_rfm_t - spline_log_rfm_t` | +0.0040 / +0.0090 / −0.0010 |
| 2 | Dunnhumby | `fuzzy_rfm_t - spline_log_rfm_t` | −0.0079 / −0.0099 / −0.0047 |

## Fixed configuration

- Script: `scripts/temporal_nonfca_baseline.py`, arms `fuzzy_rfm_t` and `spline_log_rfm_t` only.
- Unchanged: cohorts (4 Dunnhumby, 5 pooled + 1 primary Retail II origins), 91-day holdouts, outcome
  definitions, training-fold T median imputation, feature construction, concept mining and suppression
  parameters, LogisticRegressionCV / RidgeCV settings, bootstrap B = 2000 with seed 12345.
- Changed: **CV seed 0 → 1** (StratifiedKFold `random_state=1`). Learner `random_state` follows the
  existing rule `cv_seed + fold`, so it changes with the seed; no other setting changes.
- All three outcomes are reported for both datasets.
- Output: `results/temporal_second_seed/` (seed-0 outputs are not modified).

## Decision criteria (per dataset × outcome cell)

Let Δ0 = seed-0 pooled delta and Δ1 = seed-1 pooled delta (FCA+T minus spline+T). The practical
threshold is 0.002, the order of the smallest differences treated as negligible elsewhere in this
project (for example the matched-complexity arms vs M0).

- **Replicates**: sign(Δ1) = sign(Δ0), |Δ1| ≥ 0.5·|Δ0|, and Δ1 has the same sign in a majority of
  pooled origins (≥ 3/5 Retail II, ≥ 3/4 Dunnhumby).
- **Attenuated**: same sign, but |Δ1| < 0.5·|Δ0| or the origin majority fails.
- **Reversed**: opposite sign and |Δ1| ≥ 0.002.
- **Null cell** (|Δ0| < 0.002; Retail II Invoice R²): consistent if |Δ1| < 0.002, otherwise flagged.

Claim-level rules:

- Retail II FCA-specific advantage is retained only if **both** AUC and Spend R² replicate.
- Dunnhumby spline advantage is retained if at least two of the three cells replicate.

Holm-adjusted bootstrap p-values (3 tests per dataset) are reported descriptively and are not the
decision criterion. Two seeds give a direction/magnitude check only; they do not estimate
fold-assignment variance.

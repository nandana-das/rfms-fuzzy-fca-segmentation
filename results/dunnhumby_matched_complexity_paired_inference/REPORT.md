# Dunnhumby matched-complexity: post hoc paired inference

Customer-level predictions for M0, unpruned RFM+T, first-K matched, and stability-ranked matched RFM+T were regenerated with the source scripts' code paths, folds, seeds, and learners. The stability-matched arm reused the saved corrected per-fold selections (`results/dunnhumby_stability_matched_complexity/concept_stability_per_fold.csv`) instead of repeating the 50 selection resamples.

## Validation

Every regenerated per-origin metric was compared with the cached CSVs at tolerance 1e-15: 48/48 passed; max abs difference 0. Inference proceeded.

## Paired comparisons (post hoc)

Household-cluster bootstrap, B=2000, seed 12345: each draw resamples the 2499 household IDs once and applies the same draw to all four origins. Delta = mean over origins of (first arm - second arm). Percentile 95% intervals; Holm over the 12 tests.

| comparison | metric | delta | CI low | CI high | p Holm | positive origins |
|---|---|---:|---:|---:|---:|---|
| fuzzy_rfm_t_matched - fuzzy_rfm | auc | +0.001990 | -0.000755 | +0.005029 | 1.0000 | 3/4 |
| fuzzy_rfm_t_matched - fuzzy_rfm | spend_r2 | -0.001747 | -0.003640 | +0.000254 | 0.6560 | 1/4 |
| fuzzy_rfm_t_matched - fuzzy_rfm | invoice_r2 | -0.000339 | -0.002372 | +0.001696 | 1.0000 | 2/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm | auc | +0.001525 | -0.001075 | +0.004195 | 1.0000 | 3/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm | spend_r2 | -0.001389 | -0.003167 | +0.000535 | 1.0000 | 2/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm | invoice_r2 | -0.000557 | -0.002549 | +0.001409 | 1.0000 | 2/4 |
| fuzzy_rfm_t_matched - fuzzy_rfm_t | auc | +0.004939 | +0.001838 | +0.008179 | 0.0220 | 4/4 |
| fuzzy_rfm_t_matched - fuzzy_rfm_t | spend_r2 | +0.001752 | +0.000249 | +0.003241 | 0.1530 | 3/4 |
| fuzzy_rfm_t_matched - fuzzy_rfm_t | invoice_r2 | -0.000390 | -0.001862 | +0.001040 | 1.0000 | 2/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm_t | auc | +0.004474 | +0.001654 | +0.007424 | 0.0120 | 4/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm_t | spend_r2 | +0.002110 | +0.000691 | +0.003600 | 0.0500 | 3/4 |
| fuzzy_rfm_t_stability_matched - fuzzy_rfm_t | invoice_r2 | -0.000608 | -0.001945 | +0.000791 | 1.0000 | 3/4 |

## Limitations

- Post hoc: neither matched-complexity diagnostic prespecified tests; the family above was fixed before inference was computed.
- The household-cluster bootstrap handles dependence from the same households appearing in all four origins. It reflects test-sample variability only.
- It does not capture model-refitting variability, alternative fold assignments (single CV seed), or uncertainty from choosing the matched-complexity rules after seeing earlier results.
- Matching feature count also changes concept composition, so these comparisons cannot isolate a pure representation-size effect.

Command: `python scripts/dunnhumby_matched_complexity_paired_inference.py`
Runtime: 9.7 minutes.

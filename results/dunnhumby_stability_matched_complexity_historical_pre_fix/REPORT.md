# Dunnhumby stability-ranked matched-complexity diagnostic

The standard RFM+T miner and suppressor were run once per training fold. Conditional suppression stability was then estimated from 50 training-only bootstrap resamples with the fold-fitted memberships and mined concept set fixed; the existing suppressor was reapplied to each resampled extent matrix. Exact canonical intent strings identified concepts across resamples. Concepts were ranked by stability descending, support descending, intent size ascending, and intent lexicographically; K equaled the cached same-fold M0 feature count.

This is exploratory: matching feature count also changes concept composition and does not isolate a pure representation-size effect. Stability is conditional on the base fold's mined concept set; it is not a full resampling stability analysis of concept discovery.

All retained concepts had stability `1.0` in all 20 folds. Consequently, stability did not discriminate among the retained concepts in this diagnostic; selection effectively fell back to base-fold support and deterministic tie-breakers.

## Pooled metrics

| Arm | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| Cached M0 | 0.889949 | 0.538188 | 0.622664 |
| Cached unpruned RFM+T | 0.886999 | 0.534689 | 0.622716 |
| Previous first-K matched | 0.891938 | 0.536441 | 0.622325 |
| Stability-ranked matched | 0.891473 | 0.536799 | 0.622108 |

Stability-ranked matched minus M0 is AUC `+0.001525`, Spend R² `−0.001389`, and Invoice R² `−0.000557`. Relative to unpruned RFM+T, the changes are `+0.004474`, `+0.002110`, and `−0.000608`, respectively. Relative to the previous first-K diagnostic, the changes are `−0.000465`, `+0.000358`, and `−0.000218`.

## Per-origin metrics

| Origin | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| day 347 | 0.898759 | 0.551148 | 0.627147 |
| day 438 | 0.893453 | 0.543981 | 0.623483 |
| day 529 | 0.901099 | 0.538390 | 0.618276 |
| day 620 | 0.872581 | 0.513675 | 0.619524 |

## Feature counts and stability

The selected feature count matched M0 exactly in all 20 folds. Mean counts by origin were:

| Origin | M0 | Unpruned RFM+T | Stability matched |
|---|---:|---:|---:|
| day 347 | 90.0 | 180.6 | 90.0 |
| day 438 | 93.8 | 171.6 | 93.8 |
| day 529 | 89.4 | 163.2 | 89.4 |
| day 620 | 93.2 | 170.6 | 93.2 |

All selected concepts had stability `1.0`; full concept-level values are in `concept_stability_per_fold.csv`.

Paired uncertainty intervals and Holm tests against cached arms are unavailable because customer-level cached predictions were not retained.

Outputs: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `concept_stability_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.

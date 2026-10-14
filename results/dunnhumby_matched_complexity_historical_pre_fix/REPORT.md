# Dunnhumby RFM+T matched-complexity diagnostic

## Selection rule

The existing RFM+T temporal feature construction, training-fold membership fitting, concept mining, and Jaccard suppression were unchanged. After suppression, the matched arm retained the first `K` concepts in the existing deterministic post-suppression order, where `K` was the number of retained M0 concepts in the same training fold. The rule used no future outcomes, test-fold information, or model performance. Because the first-K rule is order-dependent, it changes which concepts are retained as well as the feature count.

This is an exploratory, outcome-independent, order-dependent truncation diagnostic. It matches feature counts but changes concept composition, so it cannot isolate a pure representation-size effect or establish that feature count alone caused the temporal effect. The first-K rule has no separate methodological justification beyond deterministic, training-only truncation of the existing representation.

## Pooled metrics

| Arm | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| Cached M0 | 0.889949 | 0.538188 | 0.622664 |
| Cached RFM+T | 0.886999 | 0.534689 | 0.622716 |
| Matched RFM+T | 0.891938 | 0.536441 | 0.622325 |

Relative to cached M0, matched RFM+T changes are AUC `+0.001989`, Spend R² `−0.001747`, and Invoice R² `−0.000339`. Relative to unpruned RFM+T, changes are AUC `+0.004939`, Spend R² `+0.001752`, and Invoice R² `−0.000391`.

## Per-origin differences

Values are matched RFM+T minus the cached reference; each cell is `M0 difference / unpruned RFM+T difference`.

| Origin | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| day 347 | +0.003250 / +0.007768 | +0.001159 / +0.004187 | +0.003869 / +0.001993 |
| day 438 | +0.005357 / +0.004197 | −0.004858 / −0.002763 | −0.005771 / −0.005288 |
| day 529 | +0.000301 / +0.003957 | −0.000111 / +0.002749 | +0.001129 / −0.000162 |
| day 620 | −0.000950 / +0.003833 | −0.003177 / +0.002836 | −0.000582 / +0.001896 |

The matched arm removes the pooled Spend R² deficit relative to unpruned RFM+T, but it remains slightly below M0 on pooled Spend R². It improves AUC over both references in aggregate and has mixed invoice-count differences by origin.

## Feature counts

| Origin | M0 mean | RFM+T mean | Matched mean |
|---|---:|---:|---:|
| day 347 | 90.0 | 180.6 | 90.0 |
| day 438 | 93.8 | 171.6 | 93.8 |
| day 529 | 89.4 | 163.2 | 89.4 |
| day 620 | 93.2 | 170.6 | 93.2 |

All 20 folds matched the M0 count exactly. Full fold-level counts are in `feature_counts_per_fold.csv`.

## Inference and provenance limits

The cached temporal artifacts contain metrics but not customer-level M0/RFM+T predictions. Therefore valid paired bootstrap intervals and Holm-adjusted tests against those cached arms cannot be reconstructed. `comparisons_to_cached.csv` reports point differences and marks paired inference as unavailable; no unpaired or fabricated significance test was substituted.

The new run reproduced the cached Dunnhumby M0 and RFM+T metrics with maximum absolute difference `0.0` for all three outcomes and all four origins. A provenance discrepancy remains in the diagnostic fields: the prior temporal-ablation CSV reports zero `t_missing_train` and `t_missing_test` values in every retained fold, whereas this targeted run using the unchanged feature function reports nonzero fold-local missing T counts. The historical cause could not be established from the retained artifacts. The original values are therefore discrepant and unverified; they are preserved and are not silently corrected. This does not change the cached metric parity.

## Conclusion

Matching the RFM+T representation to M0 complexity improves Spend R² by `+0.001752` relative to unpruned RFM+T, but remains `−0.001747` below M0 on pooled Dunnhumby Spend R². This is consistent with representation size contributing to the original reduction, but it does not establish that feature count alone caused it because concept composition also changed. The experiment is exploratory and diagnostic, not a replacement for the original temporal ablation.

## Outputs and command

- Command: `python scripts/dunnhumby_matched_complexity.py`
- Outputs: `results/dunnhumby_matched_complexity/`
- Files: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`, `REPORT.md`.
- Existing temporal-ablation, Step 0, manuscript, and other evidence files were not modified.

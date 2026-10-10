# Exploratory RFM tail-information ablation plan

**Branch:** `experiment/rfm-tail-information`  
**Status:** exploratory plan, committed before implementation or execution

## Objective

Test whether continuous RFM magnitude information improves the predictive
performance of the existing fuzzy RFM-FCA representation, especially for
customers whose fifth-centroid membership saturates.

## Predefined arms

1. `fuzzy_rfm_fca`: the frozen fuzzy RFM-FCA arm from
   `scripts/baseline_ladder_rolling_origin.py`, reused without modification.
2. `tail_augmented_fuzzy_rfm_fca`: the same fold-local fuzzy RFM-FCA concept
   features, concatenated with standardized `log1p(R, F, M)` features. The
   scaler is fit on the training customers in each fold and applied unchanged
   to that fold's test customers. This is a hybrid feature augmentation, not
   pure FCA.
3. `spline_log_rfm`: the frozen cubic spline baseline on `log1p(R, F, M)`,
   reused without modification.

No membership thresholds, centroids, cutpoints, concept-mining parameters,
pruning rules, or downstream model settings are changed.

## Evaluation

The driver reuses the frozen cohorts, 91-day holdouts, customer populations,
outcomes, stratified five-fold splits, and metric transformations. It reports
repurchase ROC AUC, future-spend R², and future-order-count R². All feature
engineering and scaling are fit within training folds only, and all arms use
the same fold indices.

The planned comparisons are:

- tail-augmented fuzzy RFM-FCA minus frozen fuzzy RFM-FCA;
- tail-augmented fuzzy RFM-FCA minus spline log-RFM;
- frozen fuzzy RFM-FCA minus spline log-RFM.

Per-origin metrics and paired differences are reported. If the established
bootstrap/Holm inference is run, it uses the existing baseline conventions
without changing the frozen experiment or its outputs.

## Scope and safeguards

- Datasets and rolling origins match the frozen baseline comparison.
- The Online Retail II 365-day primary cohort remains separate and is not
  pooled with the five rolling origins.
- The frozen arm must reproduce the reference feature construction and
  predictions within the established numerical tolerance.
- Tests verify train-only scaling, identical customer splits, no test influence
  on feature construction, and output isolation under `results/rfm_tail_information/`.
- A guard or smoke-test failure stops execution before result files are written.
- Existing frozen methodology, scripts, historical results, and prior
  experiment outputs are not modified or reused when invalidated by
  `docs/AUDIT_ERRATA.md`.

## Interpretation

Any gain of the augmented arm over fuzzy RFM-FCA is attributable to adding
continuous magnitude features to the FCA representation, not to FCA alone.
Results are exploratory, dataset-specific, and cannot establish equivalence
for nonsignificant comparisons. The tail-saturation motivation is descriptive;
no parameters are tuned to produce a favorable result.

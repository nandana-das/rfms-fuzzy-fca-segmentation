# Dunnhumby RFM+T matched-complexity diagnostic

The existing M0 and unpruned RFM+T results are retained as cached references. The matched arm uses the existing RFM+T concepts after standard training-only mining and Jaccard suppression, then keeps exactly the same number of concepts as M0 in the same fold, preserving post-suppression order.

This is an outcome-independent complexity diagnostic, not evidence that feature count alone causes the temporal effect. Paired confidence intervals and Holm tests against cached M0/RFM+T are unavailable because the source artifacts retain metrics but not customer-level predictions.

Outputs: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.

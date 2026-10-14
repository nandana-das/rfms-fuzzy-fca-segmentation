# Dunnhumby stability-ranked matched-complexity diagnostic

The standard RFM+T miner and suppressor were run once per training fold. Conditional suppression stability was then estimated from training-only bootstrap resamples with the fold-fitted memberships and mined concept set fixed; the existing suppressor was reapplied to each resampled extent matrix. Exact canonical intent strings identified concepts across resamples. Concepts were ranked by stability descending, support descending, intent size ascending, and intent lexicographically; K equaled the cached same-fold M0 feature count.

This is exploratory: matching feature count also changes concept composition and does not isolate a pure representation-size effect. Stability is conditional on the base fold's mined concept set; it is not a full resampling stability analysis of concept discovery.

Paired uncertainty intervals and Holm tests against cached arms are unavailable because customer-level cached predictions were not retained.

Outputs: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `concept_stability_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.

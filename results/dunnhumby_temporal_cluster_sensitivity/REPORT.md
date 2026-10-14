# Dunnhumby RFM+T − M0: household-cluster sensitivity check

Post hoc sensitivity check of an existing comparison (not a new test family). One household resample (B=2000, seed 12345, 2499 households) is applied jointly to all four origins; statistic = mean over origins of the per-origin difference. Predictions are the validated regenerated predictions; no model was refit.

| metric | delta | original CI | original Holm p | cluster CI | cluster p (raw) | ×9 bound |
|---|---:|---|---:|---|---:|---:|
| auc | -0.0029 | [-0.0062, +0.0003] | 0.3120 | [-0.0062, +0.0002] | 0.0710 | 0.6390 |
| spend_r2 | -0.0035 | [-0.0053, -0.0017] | 0.0045 | [-0.0052, -0.0017] | 0.0010 | 0.0090 |
| invoice_r2 | +0.0001 | [-0.0017, +0.0018] | 1.0000 | [-0.0018, +0.0019] | 0.9450 | 1.0000 |

The ×9 bound multiplies the raw cluster p by the size of the original Holm family (a conservative Bonferroni bound). Intervals reflect test-sample variability only; refitting and fold-assignment variability are not captured.

Command: `python scripts/dunnhumby_temporal_cluster_sensitivity.py`

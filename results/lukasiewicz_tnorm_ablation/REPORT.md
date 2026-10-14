# Łukasiewicz t-norm ablation

This narrow run reuses the existing M0 threshold-mined concepts and tests only aggregation. It is not continuous concept discovery.

## Mathematical definitions and provenance

- M0/Gödel-min: `min(mu_b)` over the base memberships named by each existing concept.
- M2/product: `prod(mu_b)` over the same concepts.
- M4/Łukasiewicz: `max(0, sum(mu_b) - (n_attributes - 1))` over the same concepts.
- Memberships, threshold context, support, concept mining, suppression, folds, learners, outcomes, and holdouts were held fixed.
- M0 and product were recomputed only in this narrow run to retain paired customer-level prediction vectors; the prior product results were unchanged.

Pooled means and the complete per-origin metrics are in `per_origin_metrics.csv`; fold-level concept diagnostics are in `concept_counts_per_fold.csv`.

## Pooled paired comparisons

| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |
|---|---|---|---:|---:|---:|---:|---|
| Dunnhumby | M4 - M0 | auc | -0.0035 | -0.0061 | -0.0009 | 0.0420 | 1/4 |
| Dunnhumby | M4 - M0 | spend_r2 | +0.0010 | -0.0006 | +0.0026 | 0.5640 | 4/4 |
| Dunnhumby | M4 - M0 | invoice_r2 | -0.0001 | -0.0018 | +0.0017 | 0.9060 | 2/4 |
| Dunnhumby | M2 - M0 | auc | -0.0029 | -0.0050 | -0.0008 | 0.0600 | 0/4 |
| Dunnhumby | M2 - M0 | spend_r2 | +0.0021 | +0.0009 | +0.0033 | 0.0090 | 4/4 |
| Dunnhumby | M2 - M0 | invoice_r2 | +0.0017 | +0.0005 | +0.0027 | 0.0160 | 4/4 |
| Dunnhumby | M4 - M2 | auc | -0.0006 | -0.0018 | +0.0004 | 0.5640 | 1/4 |
| Dunnhumby | M4 - M2 | spend_r2 | -0.0011 | -0.0024 | +0.0002 | 0.4120 | 1/4 |
| Dunnhumby | M4 - M2 | invoice_r2 | -0.0018 | -0.0031 | -0.0005 | 0.0650 | 0/4 |
| Online Retail II | M4 - M0 | auc | +0.0014 | +0.0006 | +0.0023 | 0.0060 | 3/5 |
| Online Retail II | M4 - M0 | spend_r2 | -0.0005 | -0.0019 | +0.0008 | 0.5380 | 2/5 |
| Online Retail II | M4 - M0 | invoice_r2 | +0.0014 | +0.0000 | +0.0028 | 0.1840 | 4/5 |
| Online Retail II | M2 - M0 | auc | +0.0036 | +0.0023 | +0.0050 | 0.0045 | 4/5 |
| Online Retail II | M2 - M0 | spend_r2 | +0.0007 | -0.0001 | +0.0015 | 0.2580 | 4/5 |
| Online Retail II | M2 - M0 | invoice_r2 | +0.0020 | +0.0012 | +0.0028 | 0.0045 | 5/5 |
| Online Retail II | M4 - M2 | auc | -0.0022 | -0.0032 | -0.0013 | 0.0045 | 1/5 |
| Online Retail II | M4 - M2 | spend_r2 | -0.0012 | -0.0020 | -0.0004 | 0.0200 | 0/5 |
| Online Retail II | M4 - M2 | invoice_r2 | -0.0006 | -0.0016 | +0.0004 | 0.5380 | 1/5 |

Per-origin metrics and fold diagnostics are in the CSV outputs.

All representations were built inside each training fold. No validation or holdout rows were used for membership fitting, concept mining, suppression, or downstream preprocessing. This experiment tests aggregation over existing threshold-mined concepts, not continuous concept discovery.

Command: `python scripts\\lukasiewicz_tnorm_ablation.py`.

Runtime: 6.5 minutes.

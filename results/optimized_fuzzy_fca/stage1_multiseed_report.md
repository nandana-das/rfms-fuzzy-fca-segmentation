# Stage-1 Multi-Seed Validation Report

> **INVALIDATED (2026-10-08 audit).** `dense_rank_scores` always produced 5 bands, so configurations with 3 or 4 levels used centroids from only the lowest bands and saturated the rest. This experiment did not test fuzzy granularity. The scoring bug is fixed; the experiment has not been re-run. See `docs/AUDIT_ERRATA.md`.

## Protocol

- Dataset: Dunnhumby Complete Journey.
- Outer seeds: 1000-1009 (n=10).
- Outer split: 70/30 stratified.
- Inner split: 70/30 within outer train; configuration selection uses inner validation ONLY.
- Configs evaluated: 12 (fixed 5/5/5 baseline included).
- FCA selection fixed for Stage 1: min_support=0.04, L=(0.3, 0.5, 0.7), J_max=0.8, mu_cut=0.5.
- Selection: inner-validation Pareto + combined normalized score.
- Outer test evaluated once per outer seed after config frozen.

## Configuration Selection Frequency

| Configuration | Selections |
|---------------|------------|
| 5/5/3 | 9/10 |
| 4/4/5 | 1/10 |

## Per-Metric Aggregation (Selected vs Baseline 5/5/5)

| Metric | Selected Mean | Selected SD | Baseline Mean | Baseline SD | Paired Δ Mean | Paired Δ SD | 95% CI | Paired t | p-value |
|--------|---------------|-------------|---------------|-------------|---------------|-------------|--------|----------|--------|
| ROC AUC | 0.8575 | 0.0285 | 0.8605 | 0.0292 | -0.0030 | 0.0173 | [-0.0153, +0.0094] | -0.5474 | 0.5974 |
| PR AUC | 0.9851 | 0.0045 | 0.9863 | 0.0039 | -0.0012 | 0.0029 | [-0.0032, +0.0009] | -1.2539 | 0.2415 |
| Spend R² | 0.4768 | 0.0322 | 0.5028 | 0.0376 | -0.0260 | 0.0178 | [-0.0388, -0.0133] | -4.6139 | 0.0013 |
| Invoice R² | 0.5994 | 0.0318 | 0.6045 | 0.0330 | -0.0051 | 0.0242 | [-0.0224, +0.0122] | -0.6662 | 0.5220 |

## Per-Seed Results

| Outer Seed | Selected | Baseline | Sel. Inner AUC | Sel. Inner Spend R2 | Sel. Inner Invoice R2 | Sel. Test AUC | Sel. Test PR AUC | Sel. Test Spend R2 | Sel. Test Invoice R2 | Baseline Test AUC | Baseline Test PR AUC | Baseline Test Spend R2 | Baseline Test Invoice R2 | Δ AUC | Δ PR AUC | Δ Spend R2 | Δ Invoice R2 | Sel. Concepts | Baseline Concepts |
|------------|----------|----------|----------------|---------------------|------------------------|---------------|------------------|--------------------|----------------------|-------------------|----------------------|-----------------------|-------------------------|-------|----------|------------|---------------|---------------|-------------------|
| 1000 | 5/5/3 | 5/5/5 | 0.8661 | 0.4572 | 0.6041 | 0.8985 | 0.9897 | 0.5485 | 0.6659 | 0.8958 | 0.9904 | 0.5747 | 0.6626 | +0.0027 | -0.0007 | -0.0262 | +0.0033 | 108 | 127 |
| 1001 | 5/5/3 | 5/5/5 | 0.8872 | 0.5363 | 0.6498 | 0.8246 | 0.9803 | 0.4613 | 0.5989 | 0.8269 | 0.9834 | 0.4851 | 0.5989 | -0.0022 | -0.0031 | -0.0238 | +0.0001 | 62 | 95 |
| 1002 | 5/5/3 | 5/5/5 | 0.8947 | 0.4918 | 0.5785 | 0.8780 | 0.9865 | 0.5014 | 0.6344 | 0.8717 | 0.9868 | 0.5358 | 0.6351 | +0.0063 | -0.0002 | -0.0344 | -0.0007 | 77 | 103 |
| 1003 | 5/5/3 | 5/5/5 | 0.8732 | 0.5057 | 0.6120 | 0.8946 | 0.9917 | 0.4761 | 0.5977 | 0.8958 | 0.9918 | 0.5029 | 0.5997 | -0.0012 | -0.0001 | -0.0268 | -0.0021 | 89 | 113 |
| 1004 | 5/5/3 | 5/5/5 | 0.8524 | 0.4624 | 0.6017 | 0.8620 | 0.9869 | 0.4470 | 0.5821 | 0.8572 | 0.9853 | 0.4767 | 0.5804 | +0.0048 | +0.0016 | -0.0297 | +0.0018 | 95 | 114 |
| 1005 | 4/4/5 | 5/5/5 | 0.8624 | 0.5141 | 0.6113 | 0.8326 | 0.9797 | 0.4861 | 0.5651 | 0.8821 | 0.9883 | 0.5395 | 0.6376 | -0.0495 | -0.0086 | -0.0534 | -0.0725 | 76 | 107 |
| 1006 | 5/5/3 | 5/5/5 | 0.8595 | 0.4553 | 0.6198 | 0.8216 | 0.9797 | 0.4290 | 0.5599 | 0.8186 | 0.9796 | 0.4567 | 0.5579 | +0.0030 | +0.0001 | -0.0276 | +0.0020 | 87 | 107 |
| 1007 | 5/5/3 | 5/5/5 | 0.8764 | 0.4487 | 0.5928 | 0.8703 | 0.9881 | 0.4752 | 0.6033 | 0.8658 | 0.9869 | 0.4985 | 0.5991 | +0.0045 | +0.0012 | -0.0233 | +0.0042 | 84 | 101 |
| 1008 | 5/5/3 | 5/5/5 | 0.8860 | 0.5104 | 0.6401 | 0.8322 | 0.9812 | 0.4667 | 0.6062 | 0.8211 | 0.9816 | 0.5001 | 0.6084 | +0.0112 | -0.0004 | -0.0334 | -0.0021 | 111 | 126 |
| 1009 | 5/5/3 | 5/5/5 | 0.8983 | 0.5359 | 0.6417 | 0.8604 | 0.9874 | 0.4764 | 0.5805 | 0.8698 | 0.9886 | 0.4581 | 0.5655 | -0.0094 | -0.0012 | +0.0183 | +0.0150 | 127 | 137 |

## Leakage Assertions

- Inner validation metrics computed on inner_val ONLY.
- Outer test metrics computed AFTER config selection per outer seed.
- Config selection uses inner validation Pareto ONLY.
- No outer test metrics passed to optimizer.
- Baseline 5/5/5 included and evaluated independently each outer seed.

Total runtime: 498.2s
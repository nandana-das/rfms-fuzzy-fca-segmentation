# Stage-2 Report: Optimized FCA Concept Selection (Dunnhumby, 5/5/5 Fuzzy Fixed)

> **INVALIDATED (2026-10-08 audit).** (1) The optimized arm used a faulty `_piecewise_membership_generic` (mean centroids, inverted interior memberships), unlike the canonical baselines. (2) Baseline B's PR AUC was computed on crisp one-hot test features, so the 'significant' PR AUC gain over the fuzzy baseline is an artifact. (3) Baseline C mixed membership functions between train and test. (4) 'Raw RFM' here is crisp RFM-FCA, not raw R/F/M. The bugs are fixed; the experiment has not been re-run. See `docs/AUDIT_ERRATA.md`.

## Protocol

- Dataset: Dunnhumby Complete Journey.
- Fuzzy representation fixed at 5/5/5.
- Grid: reduced 108-config (3 min_support × 3 j_max × 3 mu_cut × 4 kuz_loss).
- Outer seeds: 1000-1009 (n=10).
- Outer split: 70/30 stratified.
- Inner split: 70/30 within outer train; config selection uses inner validation ONLY.
- Inner seed: 42.
- FCA optimization variables: min_support, J_max, mu_cut, canonical Kuznetsov stability-loss threshold.
- Exact canonical stability computed from training context only (no proxy, no Kneedle).
- Selection: inner-validation Pareto + average of normalized predictive metrics; outer test evaluated once after config frozen.

## Baselines (evaluated every outer seed)

- A. Raw RFM (crisp 5-band + crisp closed concepts).
- B. Existing 5/5/5 Fuzzy RFM-FCA (Jaccard only; no Kneedle, no Kuznetsov).
- C. Existing 5/5/5 Kuznetsov-FCA (canonical stability filter at loss<=1.0, J_max=0.80, mu_cut=0.5).
- D. Optimized 5/5/5 FCA (selected from inner validation per outer seed).

## Configuration Selection Frequency

| Configuration | Selections |
|---------------|------------|
| ms=2.000e-02_jm=0.70_mc=0.45_kl=1.000e+00 | 2/10 |
| ms=2.000e-02_jm=0.90_mc=0.45_kl=1.000e-05 | 1/10 |
| ms=5.000e-02_jm=0.90_mc=0.45_kl=1.000e+00 | 1/10 |
| ms=5.000e-02_jm=0.80_mc=0.45_kl=1.000e+00 | 1/10 |
| ms=2.000e-02_jm=0.70_mc=0.45_kl=1.000e-300 | 1/10 |
| ms=4.000e-02_jm=0.80_mc=0.45_kl=1.000e+00 | 1/10 |
| ms=4.000e-02_jm=0.90_mc=0.45_kl=1.000e-05 | 1/10 |
| ms=2.000e-02_jm=0.70_mc=0.50_kl=1.000e-300 | 1/10 |
| ms=5.000e-02_jm=0.70_mc=0.45_kl=6.200e-02 | 1/10 |

## Outer-Test Means Across Seeds

| Method | ROC AUC | PR AUC | Spend R² | Invoice R² |
|--------|---------|--------|----------|------------|
| Raw RFM | 0.8541 | 0.9848 | 0.4800 | 0.5548 |
| Existing Fuzzy FCA | 0.8605 | 0.9822 | 0.5028 | 0.6045 |
| Existing Kuznetsov FCA | 0.8318 | 0.9833 | 0.4212 | 0.5190 |
| Optimized FCA | 0.8641 | 0.9867 | 0.4972 | 0.6008 |

## Paired Deltas vs Raw RFM (outer test)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |
|--------|--------|----|----------|--------|---|---|---|
| ROC AUC | +0.0100 | 0.0117 | +0.0096 | [+0.0017,+0.0184] | +2.724 | 0.0235 | 10 |
| PR AUC | +0.0019 | 0.0021 | +0.0026 | [+0.0004,+0.0034] | +2.856 | 0.0189 | 10 |
| Spend R² | +0.0172 | 0.0111 | +0.0171 | [+0.0093,+0.0251] | +4.906 | 0.0008 | 10 |
| Invoice R² | +0.0460 | 0.0082 | +0.0441 | [+0.0401,+0.0519] | +17.716 | 0.0000 | 10 |

## Paired Deltas vs Existing Fuzzy FCA (outer test)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |
|--------|--------|----|----------|--------|---|---|---|
| ROC AUC | +0.0037 | 0.0130 | +0.0007 | [-0.0057,+0.0130] | +0.885 | 0.3992 | 10 |
| PR AUC | +0.0045 | 0.0017 | +0.0044 | [+0.0033,+0.0057] | +8.415 | 0.0000 | 10 |
| Spend R² | -0.0056 | 0.0190 | -0.0101 | [-0.0192,+0.0080] | -0.932 | 0.3755 | 10 |
| Invoice R² | -0.0037 | 0.0106 | -0.0062 | [-0.0113,+0.0039] | -1.109 | 0.2961 | 10 |

## Paired Deltas vs Existing Kuznetsov FCA (outer test)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |
|--------|--------|----|----------|--------|---|---|---|
| ROC AUC | +0.0323 | 0.0152 | +0.0351 | [+0.0215,+0.0431] | +6.738 | 0.0001 | 10 |
| PR AUC | +0.0033 | 0.0022 | +0.0040 | [+0.0018,+0.0049] | +4.884 | 0.0009 | 10 |
| Spend R² | +0.0760 | 0.0230 | +0.0790 | [+0.0595,+0.0924] | +10.453 | 0.0000 | 10 |
| Invoice R² | +0.0818 | 0.0179 | +0.0814 | [+0.0690,+0.0946] | +14.453 | 0.0000 | 10 |

## Complexity Across Seeds

| Method | Candidates | After Stability | Dropped by Stability | Final | Dropped by Jaccard |
|--------|------------|----------------|----------------------|-------|--------------------|
| Optimized FCA | 668.00 | 433.60 | 234.40 | 94.40 | 339.20 |
| Existing Fuzzy FCA | 517.40 | - | - | 113.00 | - |
| Existing Kuznetsov FCA | 517.40 | 517.40 | 0.00 | 113.00 | 404.40 |

## Per-Seed Results

| Seed | Sel ms | Sel jm | Sel mc | Sel kl | Sel Outer AUC | Sel Outer PR | Sel Outer Spend R2 | Sel Outer Inv R2 | Raw AUC | Fuzzy AUC | Kuz AUC | Sel Concepts | Fuzzy Concepts | Kuz Concepts |
|------|--------|--------|--------|--------|---------:|---------:|----------:|----------:|--------:|----------:|--------:|-----------:|-------------:|-----------:|
| 1000 | 2.00e-02 | 0.70 | 0.45 | 1.00e+00 | 0.8972 | 0.9916 | 0.5677 | 0.6556 | 0.8894 | 0.8958 | 0.8614 | 131 | 127 | 127 |
| 1001 | 2.00e-02 | 0.90 | 0.45 | 1.00e-05 | 0.8224 | 0.9810 | 0.4759 | 0.5836 | 0.8119 | 0.8269 | 0.7734 | 98 | 95 | 95 |
| 1002 | 5.00e-02 | 0.90 | 0.45 | 1.00e+00 | 0.8716 | 0.9886 | 0.5248 | 0.6361 | 0.8778 | 0.8717 | 0.8219 | 134 | 103 | 103 |
| 1003 | 5.00e-02 | 0.80 | 0.45 | 1.00e+00 | 0.8957 | 0.9918 | 0.4880 | 0.5800 | 0.8745 | 0.8958 | 0.8960 | 114 | 113 | 113 |
| 1004 | 2.00e-02 | 0.70 | 0.45 | 1.00e+00 | 0.8509 | 0.9848 | 0.4564 | 0.5719 | 0.8357 | 0.8572 | 0.8274 | 144 | 114 | 114 |
| 1005 | 2.00e-02 | 0.70 | 0.45 | 1.00e-300 | 0.8912 | 0.9891 | 0.5220 | 0.6293 | 0.8637 | 0.8821 | 0.8558 | 21 | 107 | 107 |
| 1006 | 4.00e-02 | 0.80 | 0.45 | 1.00e+00 | 0.8258 | 0.9795 | 0.4487 | 0.5718 | 0.8351 | 0.8186 | 0.7909 | 114 | 107 | 107 |
| 1007 | 4.00e-02 | 0.90 | 0.45 | 1.00e-05 | 0.8744 | 0.9887 | 0.4846 | 0.6003 | 0.8656 | 0.8658 | 0.8410 | 90 | 101 | 101 |
| 1008 | 2.00e-02 | 0.70 | 0.50 | 1.00e-300 | 0.8559 | 0.9857 | 0.5001 | 0.6030 | 0.8357 | 0.8211 | 0.8127 | 26 | 126 | 126 |
| 1009 | 5.00e-02 | 0.70 | 0.45 | 6.20e-02 | 0.8561 | 0.9858 | 0.5041 | 0.5762 | 0.8512 | 0.8698 | 0.8377 | 92 | 137 | 137 |

## Leakage Assertions

- Inner validation metrics computed on inner_val ONLY.
- Outer test metrics computed AFTER config selection per outer seed.
- Config selection uses inner validation Pareto ONLY.
- No outer test metrics passed to optimizer or stability/complexity selection.
- All four baselines evaluated independently each outer seed.
- Exact canonical Kuznetsov stability computed from training context only.

Total runtime: 5290.7s

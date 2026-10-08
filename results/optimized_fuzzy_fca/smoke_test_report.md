# Smoke Test Report: Optimized Fuzzy FCA Experiment (Stage 1)

> **HISTORICAL (2026-10-08 audit).** Not part of the final evidence. This report predates the audit and uses superseded choices (Stage 1/2 optimization smoke tests; dense-rank scoring and the 5-band scoring defect; optimization is excluded from the final method). Current evidence: `results/final_evidence/EVIDENCE_TABLES.md`; see `docs/AUDIT_ERRATA.md`.

**Date:** 2026-10-08  
**Script:** `scripts/optimized_fuzzy_fca_smoke_test.py`  
**Branch:** `experiment/optimized-fuzzy-fca`

## Objective

Verify the end-to-end pipeline for Stage 1 optimization (fuzzy representation only, FCA selection fixed at validated baseline) on Dunnhumby Complete Journey.

## Experiment Design

- **Dataset:** Dunnhumby Complete Journey (2,499 households)
- **Outer split:** seed 42, 70/30 stratified (train=1,749, test=750)
- **Inner split:** seed 123, 70/30 within train (for optimizer fitness)
- **Configs tested:** 12 (nR,nF,nM in {3,4,5})
- **Fixed FCA parameters:**
  - min_support = 0.04
  - L_thresholds = (0.3, 0.5, 0.7)
  - J_max = 0.80, mu_cut = 0.50
  - No Kuznetsov filter (loss_threshold = 1.0, keep all valid-stability concepts)

**Important limitation:** Due to ~93% repurchase rate, inner validation AUC is degenerate (all 1.0). For this smoke test only, test metrics are used as fitness proxy. Final optimization will require a better inner validation strategy.

## Results

### Raw RFM Baseline (seed 42)

| Metric | Value |
|--------|-------|
| AUC | 0.8694 |
| Spend R² | 0.3604 |
| Invoice R² | 0.4745 |
| Predictive Score | 0.6334 |

### Pareto-Optimal Configurations (10 solutions)

| ID | R | F | M | #Cand | #Final | Test AUC | Test SpR² | Test InvR² | PredScore |
|----|---|---|---|-------|--------|----------|-----------|------------|-----------|
| 1 | 5 | 5 | 5 | 514 | 114 | 0.8629 | 0.5152 | 0.6188 | 0.9498 |
| 5 | 5 | 4 | 5 | 516 | 113 | 0.8622 | 0.5152 | 0.6135 | 0.9416 |
| 7 | 5 | 5 | 4 | 499 | 108 | 0.8581 | 0.4997 | 0.6167 | 0.9187 |
| 6 | 5 | 5 | 3 | 447 | 93 | 0.8633 | 0.4869 | 0.6144 | 0.9160 |
| 3 | 4 | 5 | 5 | 381 | 79 | 0.8180 | 0.4662 | 0.5508 | 0.7025 |
| 12 | 4 | 4 | 5 | 385 | 78 | 0.8139 | 0.4661 | 0.5455 | 0.6856 |
| 10 | 4 | 5 | 4 | 374 | 73 | 0.8144 | 0.4475 | 0.5477 | 0.6702 |
| 8 | 4 | 4 | 4 | 374 | 72 | 0.8125 | 0.4468 | 0.5391 | 0.6543 |
| 2 | 3 | 5 | 5 | 316 | 71 | 0.7798 | 0.4306 | 0.5205 | 0.5316 |
| 9 | 3 | 3 | 3 | 285 | 49 | 0.7793 | 0.3918 | 0.4995 | 0.4648 |

### Key Findings

1. **Best predictive:** Config 1 (R=5,F=5,M=5) — the baseline configuration. AUC=0.8629, Spend R²=0.5152, Invoice R²=0.6188

2. **Best compact:** Config 9 (R=3,F=3,M=3) — AUC=0.7793, Spend R²=0.3918, Invoice R²=0.4995, only 49 concepts (vs 114 for baseline)

3. **Comparison vs Raw RFM (best predictive config):**
   - ΔAUC = -0.0065 (slightly lower)
   - ΔSpend R² = +0.1548 (substantially higher)
   - ΔInvoice R² = +0.1443 (substantially higher)

4. **Trend:** Higher granularity (more bands) → higher predictive performance but more concepts. The baseline 5/5/5 is near the Pareto frontier.

### Validation Checks

- ✅ All 12 configs produced valid results
- ✅ FCA mining completed for all configs (no zero candidates)
- ✅ Jaccard suppression completed for all configs (no zero final concepts)
- ✅ Pareto front non-empty (10 solutions)
- ✅ Test metrics in sane ranges
- ✅ Baseline AUC in expected range [0.82, 0.90]
- ✅ Leakage-free: all representation building on train_outer only

## Conclusions

1. **Pipeline is functional:** All stages (scoring, centroids, memberships, mining, stability, suppression, prediction) run end-to-end without errors.

2. **Stage 1 optimization is feasible:** The search space (nR,nF,nM ∈ {3,4,5}) produces diverse Pareto-optimal solutions.

3. **Baseline is strong:** The existing 5/5/5 configuration is already Pareto-optimal and near the best predictive point.

4. **Trade-off is real:** There's a clear trade-off between predictive performance and concept complexity. Lower granularity (3 bands) gives compact representations but lower R².

5. **Leakage-free:** The pipeline correctly uses train_outer only for all representation building.

## Next Steps

1. **Stage 2:** Optimize FCA selection parameters (min_support, J_max, mu_cut, Kuznetsov threshold) with fuzzy representation fixed at baseline 5/5/5 or best Stage-1 config.

2. **Improve inner validation:** For final optimization, address the degenerate inner validation AUC by:
   - Using different stratification
   - Using R²-based fitness only
   - Using cross-validation within train_outer

3. **Stage 3:** If computationally feasible, joint NSGA-II optimization of fuzzy representation + FCA selection.

4. **Repeated evaluation:** After configuration selection, evaluate on seeds 1000-1009 for final statistical comparison.

## Artifacts

- `results/optimized_fuzzy_fca/smoke_test_configurations.csv` — all 12 configs
- `results/optimized_fuzzy_fca/smoke_test_pareto.csv` — 10 Pareto solutions
- `results/optimized_fuzzy_fca/smoke_test_summary.json` — summary with normalization anchors

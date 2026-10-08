# Corrected Nested Optimization Smoke Test Report

> **HISTORICAL (2026-10-08 audit).** Not part of the final evidence. This report predates the audit and uses superseded choices (Stage 1/2 optimization smoke tests; dense-rank scoring and the 5-band scoring defect; optimization is excluded from the final method). Current evidence: `results/final_evidence/EVIDENCE_TABLES.md`; see `docs/AUDIT_ERRATA.md`.

**Date:** 2026-10-08  
**Script:** `scripts/optimized_fuzzy_fca_nested_smoke_test.py`  
**Branch:** `experiment/optimized-fuzzy-fca`

## Purpose

Verify that we can perform **legitimate leakage-free optimization** that can distinguish candidate fuzzy representations.

This corrects the methodological issue from the previous smoke test where outer test metrics were incorrectly used as optimizer fitness.

## Protocol

**Nested optimization with strict leakage prevention:**

```
OUTER TRAIN (seed 42, 70% of cohort)
├── INNER TRAIN (70% of outer train, seed 123)
│   ├── Fit fuzzy representation (nR, nF, nM bands)
│   ├── Mine FCA concepts
│   ├── Jaccard suppress
│   └── Fit predictive models
│
├── INNER VALIDATION (30% of outer train)
│   ├── Evaluate ALL 12 configs
│   ├── Compute Pareto front
│   └── SELECT best config (INNER VALIDATION ONLY)
│
├── REFIT selected config on FULL outer train
│
└── OUTER TEST (30% of cohort) — held-out evaluation
    └── Evaluate EXACTLY ONCE after config frozen
```

**Critical:** Outer test metrics are NEVER passed to optimizer fitness.

## Results

### Inner Validation Can Distinguish Configurations

| Metric | Range | Can Distinguish? |
|--------|-------|------------------|
| ROC AUC | 0.7938 - 0.8557 | ✅ YES |
| PR AUC | 0.9749 - 0.9855 | ✅ YES |
| Spend R² | 0.3991 - 0.4980 | ✅ YES |
| Invoice R² | 0.5270 - 0.6211 | ✅ YES |

**Answer to key question:** ✅ YES — inner validation CAN distinguish configurations.

### Pareto Front (12 solutions on inner validation)

All 12 configs are Pareto-optimal (no config dominates another on all objectives):

| ID | R | F | M | #Final | ROC AUC | Spend R² | Invoice R² | Baseline |
|----|---|---|---|--------|---------|----------|------------|----------|
| 1 | 5 | 5 | 5 | 117 | 0.8557 | 0.4973 | 0.6186 | YES |
| 4 | 5 | 3 | 5 | 116 | 0.8555 | 0.4980 | 0.6047 | |
| 6 | 5 | 5 | 3 | 100 | 0.8546 | 0.4705 | 0.6211 | |
| 5 | 5 | 4 | 5 | 115 | 0.8540 | 0.4971 | 0.6125 | |
| 11 | 5 | 4 | 4 | 101 | 0.8523 | 0.4876 | 0.6104 | |
| 7 | 5 | 5 | 4 | 102 | 0.8521 | 0.4876 | 0.6161 | |
| 3 | 4 | 5 | 5 | 95 | 0.8396 | 0.4779 | 0.5855 | |
| 12 | 4 | 4 | 5 | 93 | 0.8390 | 0.4779 | 0.5790 | |
| 8 | 4 | 4 | 4 | 81 | 0.8380 | 0.4704 | 0.5769 | |
| 10 | 4 | 5 | 4 | 82 | 0.8356 | 0.4706 | 0.5832 | |
| 2 | 3 | 5 | 5 | 78 | 0.8056 | 0.4426 | 0.5516 | |
| 9 | 3 | 3 | 3 | 64 | 0.7938 | 0.3991 | 0.5270 | |

### Configuration Selection (Inner Validation Only)

Selected config: **6 (R=5, F=5, M=3)**

Selection criterion: best combined normalized score on inner validation Pareto front.

| Metric | Inner Validation | Outer Test (held-out) |
|--------|-----------------|----------------------|
| ROC AUC | 0.8546 | 0.8633 |
| PR AUC | 0.9846 | 0.9844 |
| Spend R² | 0.4705 | 0.4869 |
| Invoice R² | 0.6211 | 0.6144 |
| Concepts | 100 | 93 |

### Comparison: Selected vs Baseline 5/5/5 (Outer Test)

| Metric | Selected (R=5,F=5,M=3) | Baseline (5/5/5) | Δ |
|--------|------------------------|-------------------|---|
| ROC AUC | 0.8633 | 0.8629 | +0.0004 |
| PR AUC | 0.9844 | 0.9849 | -0.0005 |
| Spend R² | 0.4869 | 0.5152 | **-0.0283** |
| Invoice R² | 0.6144 | 0.6188 | -0.0045 |
| Concepts | 93 | 114 | -21 |

**Key finding:** The selected config (R=5,F=5,M=3) performs COMPARABLY to baseline on outer test:
- ROC AUC essentially tied (+0.0004)
- Spend R² lower by 0.0283 (baseline better)
- Invoice R² lower by 0.0045 (baseline better)
- 21 fewer concepts (more compact)

**This suggests the baseline 5/5/5 is already near-optimal for predictive performance.**

### Leakage Prevention Verification

- ✅ Inner validation metrics computed on inner_val ONLY
- ✅ Outer test metrics computed AFTER config selection
- ✅ Config selection uses inner validation Pareto ONLY
- ✅ No outer test metrics passed to optimizer
- ✅ Baseline 5/5/5 included in optimization
- ✅ Outer test evaluation is exactly once, after config frozen
- ✅ All 12 configs evaluated, including baseline

### Validation Checks

All validation checks passed:
- ✅ All 12 configs produced valid inner metrics
- ✅ Selected config has outer test metrics
- ✅ Baseline has outer test metrics
- ✅ Pareto front non-empty (12 solutions)
- ✅ Inner validation can distinguish configs (range > 0)
- ✅ Outer test metrics vary across configs

## Conclusions

### Can we perform legitimate leakage-free optimization?

**YES.** The corrected protocol successfully:
1. Uses inner validation ONLY for optimizer fitness
2. Keeps outer test completely isolated
3. Can distinguish between configurations
4. Produces a meaningful Pareto front
5. Selects a config that performs comparably to baseline on held-out test

### Is the baseline 5/5/5 competitive?

**YES.** The baseline 5/5/5 remains highly competitive:
- Tied for best ROC AUC on inner validation
- Best Spend R² on inner validation
- Second-best Invoice R² on inner validation
- On outer test, baseline outperforms the selected config on Spend R² and Invoice R²

### Can fuzzy representation optimization improve on raw RFM?

The optimization can distinguish configurations, but the baseline 5/5/5 appears near-optimal. Alternative granularities offer trade-offs (e.g., R=5,F=5,M=3 gives fewer concepts but lower R²), but don't clearly dominate the baseline.

### Next Steps

The corrected smoke test answers the key question: **Yes, we can perform legitimate leakage-free optimization.**

However, the results suggest:
1. **Stage 1 optimization may not find configurations that clearly dominate 5/5/5** on predictive metrics
2. **The main benefit may be concept compression** (fewer concepts for comparable performance)
3. **Stage 2 (FCA selection optimization) may be more impactful** than Stage 1

Before proceeding to Stage 2, we should:
1. Run the corrected protocol on multiple outer seeds (1000-1009) to verify stability
2. Consider whether the optimization objective should weight concept compression more heavily
3. Investigate if different R/F/M combinations have systematic effects

## Artifacts

- `results/optimized_fuzzy_fca/nested_smoke_test_configurations.csv` — all 12 configs with inner + outer metrics
- `results/optimized_fuzzy_fca/nested_smoke_test_pareto.csv` — 12 Pareto solutions
- `results/optimized_fuzzy_fca/nested_smoke_test_summary.json` — summary with protocol details

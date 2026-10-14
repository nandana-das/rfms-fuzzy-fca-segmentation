# Corrected temporal-ablation run provenance

This file records the single corrected temporal-ablation rerun executed after the R/F/M parity and statistical-generation audit.

- Command: `python scripts/temporal_rfm_fca_ablation.py`
- Source script SHA-256: `C404A840C9B3817694D6C587FE1063198C716A3C506A652898424706C82197E1`
- Corrected output directory: `results/temporal_rfm_fca_ablation/`
- Preserved historical directory: `results/temporal_rfm_fca_ablation_historical_pre_fix/`
- Evaluation settings: the existing three arms, rolling origins, 91-day holdouts, five customer-level folds, existing learners, bootstrap settings, and outcomes.
- Statistical family: nine pooled tests per dataset (three arm comparisons by three metrics), Holm-adjusted.

The corrected code removes the duplicate temporal recency inversion and uses explicit pooled-origin alignment when constructing paired deltas. The preceding regression checks passed: `pytest -q tests/test_temporal_rfm_parity.py tests/test_temporal_statistics.py` reported `5 passed in 6.33s`.

Verification after the rerun found:

- Per-origin AUC, Spend R², and Invoice R² matched the archived historical CSV to floating-point precision (maximum absolute difference `2.22e-16`).
- All pooled deltas recomputed from `per_origin_metrics.csv` matched `paired_comparisons.csv` exactly.
- Holm-adjusted p-values recomputed from the pooled bootstrap p-values matched the reported values for both dataset families with no mismatches.

The historical CSVs and report were preserved unchanged in the archive. No exploratory arm, new dataset, manuscript edit, or unrelated experiment was run in this verification.

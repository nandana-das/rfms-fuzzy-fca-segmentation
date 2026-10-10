# Final-paper figure cleanup manifest

**Status:** proposed and checked before deletion  
**Scope:** generated figure files only; source CSVs, result tables, scripts, logs, methodology documents, audit reports, and evidence maps are preserved.

## Retained final-paper visuals

| Path | Purpose | Authority |
|---|---|---|
| `outputs/figures/final_paper/methodology_pipeline.svg` | Frozen implemented methodology diagram | `docs/METHODOLOGY_FINAL.md`, `scripts/baseline_ladder_rolling_origin.py`, `scripts/segment_stability.py` |
| `outputs/figures/final_paper/predictive_performance_v1.png` | Consolidated v1 Crisp-versus-Baseline-Fuzzy comparison for ROC AUC, Spend R², and Invoice R² across Dunnhumby and Online Retail II | `results/baseline_ladder_rolling_origin/per_origin_metrics.csv`; generated from existing values without metric recomputation |
| `results/final_evidence/EVIDENCE_TABLES.md` | Stability and non-FCA baseline results retained as tables rather than figures | Frozen evidence tables |

## Explicitly deleted generated exports

The following are derived figure exports superseded by the consolidated final-paper figure or outside the requested main-paper scope:

- Every PNG under `outputs/figures/final_fca_comparison/`.
- `outputs/figures/unified_method_comparison/auc_four_method_comparison.pdf`
- `outputs/figures/unified_method_comparison/auc_four_method_comparison.png`
- `outputs/figures/unified_method_comparison/invoice_r2_four_method_comparison.pdf`
- `outputs/figures/unified_method_comparison/invoice_r2_four_method_comparison.png`
- `outputs/figures/unified_method_comparison/spend_r2_four_method_comparison.pdf`
- `outputs/figures/unified_method_comparison/spend_r2_four_method_comparison.png`

These exports include v2, CDNOW, hybrid, Raw RFM, or incompatible legacy/tail-information comparisons and are not used by the final-paper figure set.

## Left untouched for review or historical provenance

- All figure files under `results/dunnhumby_rfm_fca/`.
- All figure files under `results/fair_comparison_retail2/`.
- `results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png`.

These are not required for the main-paper visual set, but their historical/provenance status is not sufficiently unambiguous to delete as part of this minimal cleanup. Their source results and documentation remain preserved.

## Validation basis

- The v1 source contains `crisp_rfm_fca` and `fuzzy_rfm_fca` on the same Dunnhumby and Online Retail II rolling origins.
- Only `pooled=True` rolling-origin rows are plotted.
- The Online Retail II 365-day primary holdout is excluded from the rolling-origin figure.
- Hybrid v2, CDNOW, LRFM, membership ablation, Kneedle/Kuznetsov, and obsolete optimization outputs are excluded from the main-paper figure set.
- No experiment, training job, metric recomputation, result CSV, raw dataset, tag, branch, or historical evidence is deleted.

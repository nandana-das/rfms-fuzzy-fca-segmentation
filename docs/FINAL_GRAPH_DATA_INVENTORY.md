# Final Graph Data Inventory

**Audit date:** 2026-10-10  
**Repository:** `D:\Nandana\MTECH\Semester 3\Projects\MP\rfms_fca_project`  
**Scope:** read-only provenance inventory. No graphs, experiments, training, evaluation, or existing-file edits were performed.

## Executive recommendation

**Go for source selection, not yet for graph generation without an explicit labeling decision.** The canonical three-method source for the v2 comparison is:

`results/v2_hybrid/per_origin_metrics.csv`

Filter to `arm` values `crisp_rfm_fca`, `fuzzy_rfm_fca`, and `hybrid_fuzzy_fca`, and use only rows with `pooled=True` for the pooled 91-day rolling-origin graphs. This file contains all three required methods on the same datasets and origins, with full-precision per-origin metrics. It keeps the v2 comparison internally fair, but the entire v2 study is explicitly exploratory/post hoc.

For the frozen primary Crisp-versus-Baseline-Fuzzy comparison, use:

`results/baseline_ladder_rolling_origin/per_origin_metrics.csv`

Filter to `arm` values `crisp_rfm_fca` and `fuzzy_rfm_fca`. Keep the Online Retail II `2010-12-09 (primary, 365d)` cohort separate from pooled rolling origins.

For a CDNOW three-method confirmation figure, use:

`results/cdnow_confirmation/per_origin_metrics.csv`

Filter to the same three FCA arms. CDNOW has three chronological 91-day origins and no separate primary holdout. The repository does not contain a pre-pooled three-method CDNOW metric CSV; any future graph must aggregate the full-precision per-origin rows explicitly and label the result as CDNOW confirmation.

## 1. Candidate CSV inventory

| CSV | Git status | Version/role | Rows; columns | Coverage and granularity | Required methods present? | Suitability |
|---|---|---|---|---|---|---|
| `results/baseline_ladder_rolling_origin/per_origin_metrics.csv` | tracked | Frozen v1 primary | 100; `dataset,origin,pooled,n,repurchase_rate,arm,auc,spend_r2,invoice_r2` | Dunnhumby 4 origins; Retail II 5 rolling + separate 365-day primary; per-origin, 10 arms | Crisp and baseline fuzzy present; hybrid absent | **Canonical v1 Crisp/Fuzzy source.** Full precision; contains extra non-FCA and historical arms that must be filtered, not deleted. |
| `results/baseline_ladder_rolling_origin/paired_comparisons.csv` | tracked | Frozen v1 inference | 324; `dataset,scope,comparison,metric,delta,ci_low,ci_high,p_boot,origins_positive,p_holm` | Per-origin and `POOLED`; pairwise deltas, not method-level metric values | Crisp/Fuzzy comparisons exist; no hybrid | Suitable for statistical annotations, not the three-method graph data source. |
| `results/v2_hybrid/per_origin_metrics.csv` | tracked | v2 exploratory/post hoc | 70; `dataset,origin,pooled,n,repurchase_rate,arm,auc,spend_r2,invoice_r2` | Same 4 Dunnhumby + 5 rolling Retail II origins + separate Retail II primary; per-origin | **All three required methods present** under exact identifiers | **Canonical v2 three-method source.** Full precision; same v2 folds/origins by design and metadata. |
| `results/v2_hybrid/paired_comparisons.csv` | tracked | v2 exploratory/post hoc inference | 180; `dataset,scope,comparison,metric,delta,ci_low,ci_high,p_boot,origins_positive,p_holm` | Per-origin and `POOLED`; comparisons against spline, v1, hybrid crisp, log, and crisp | Required methods appear through pairwise comparisons, not as graph values | Suitable for optional significance/delta annotations; not a replacement for per-origin metric values. |
| `results/cdnow_confirmation/per_origin_metrics.csv` | tracked | CDNOW confirmatory dataset for v1/v2 | 21; `dataset,origin,n,repurchase_rate,arm,auc,spend_r2,invoice_r2` | Three CDNOW chronological 91-day origins; per-origin | **All three required methods present** | Suitable for a separate CDNOW confirmation figure after explicit aggregation or per-origin plotting. |
| `results/cdnow_confirmation/comparisons.csv` | tracked | CDNOW confirmation inference | 21; `id,comparison,metric,delta,ci_low,ci_high,p_boot,origins_positive,p_holm` | Pooled over 3 origins; pairwise comparisons | Required pairwise comparisons present | Suitable for inference annotations, not direct three-method metric values. |
| `results/cdnow_confirmation_verification/recomputed_comparisons.csv` | tracked | Independent CDNOW verification | 21; `id,metric,delta,ci_low,ci_high,p_boot,p_holm` | Recomputed pairwise comparisons | No method-level values | Verification evidence only; not graph data. |
| `results/rfm_tail_information/per_origin_metrics.csv` | untracked | Exploratory tail-information hybrid | 30; `dataset,origin,pooled,n,arm,auc,spend_r2,invoice_r2` | Dunnhumby/Retail II, same origin labels including primary; per-origin | Baseline fuzzy and tail hybrid present; crisp absent | **Not suitable for required three-method graph** because it has no Crisp RFM-FCA arm and its hybrid identifier is `tail_augmented_fuzzy_rfm_fca`. |
| `results/rfm_tail_information/paired_comparisons.csv` | untracked | Exploratory tail-information inference | 18; `dataset,comparison,metric,delta,ci_low,ci_high,p_boot,origins_positive,p_holm` | Pooled comparison deltas | No crisp method-level graph values | Not suitable for required graph data. Do not substitute it for v2 hybrid. |
| `results/membership_function_ablation/full_run/per_origin_metrics.csv` | untracked | Exploratory membership-function ablation | 40; `dataset,origin,pooled,n,repurchase_rate,method,auc,spend_r2,invoice_r2` | 10 cohorts × 4 methods; per-origin | M0 and alternatives only; no Crisp/Fuzzy/Hybrid identifiers | **Explicitly excluded.** It is a separate ablation and not a substitute for the main study. |
| `outputs/figures/unified_method_comparison/verification_table.csv` | untracked | Derived figure audit table | 36; `dataset,method,metric,value,source_file,evaluation_split,verification_status` | Pooled summary values for four display methods; method labels transformed to graph labels | Contains all three labels, but not per-origin rows | Audit/verification artifact, not authoritative raw result source. It is derived from multiple CSVs and includes Raw RFM. |
| `outputs/figures/unified_method_comparison/hybrid_delta_audit.csv` | untracked | Derived figure delta audit | 9; `dataset,metric,computed_hybrid_minus_fuzzy,saved_delta,saved_95ci,saved_p_holm,source_file,verification_status` | Pooled hybrid-minus-fuzzy deltas | Pairwise only | Verification artifact; not graph metric values. |

All values in the authoritative per-origin CSVs are stored at full floating-point precision. Markdown reports are rounded summaries and were not used as numeric sources. The verification table is also full-precision as stored, but is a transformed derived summary rather than a raw per-origin result.

## 2. Stored identifiers and graph labels

| Stored identifier | Graph label | Meaning | Use |
|---|---|---|---|
| `crisp_rfm_fca` | Crisp RFM-FCA | Base-paper crisp FCA comparator | Include |
| `fuzzy_rfm_fca` | Baseline Fuzzy RFM-FCA (M0) | Frozen v1 fuzzy FCA method | Include |
| `hybrid_fuzzy_fca` | Hybrid Fuzzy RFM-FCA | v2 fuzzy FCA concepts plus standardized fold-local `log1p(R,F,M)` | Include for v2/CDNOW only |
| `hybrid_crisp_fca` | Crisp-hybrid control | Crisp concepts plus log-RFM | Exclude from final three-method graph; retain as evidence/control |
| `tail_augmented_fuzzy_rfm_fca` | Tail-augmented hybrid | Exploratory tail-information experiment; not the v2 identifier | Exclude; do not rename into `hybrid_fuzzy_fca` |
| `raw_std`, `log_rfm`, `spline_log_rfm`, `gbm_raw_rfm`, `crisp_bands`, `fuzzy_bands`, `fuzzy_rfm_fca_denserank`, `kuznetsov_fca` | Various comparator labels | Non-FCA or historical/ablation arms | Exclude from final three-method figures, retain in repository |

Required metric columns are exactly `auc`, `spend_r2`, and `invoice_r2`; graph labels should be ROC AUC, Spend R², and Invoice R² respectively. `dataset`, `origin`, `pooled`, `n`, and `repurchase_rate` are grouping/context fields.

## 3. Compatibility checks

### v1 primary Crisp-versus-Fuzzy

`results/baseline_ladder_rolling_origin/per_origin_metrics.csv` is generated by `scripts/baseline_ladder_rolling_origin.py` and uses the frozen v1 rolling-origin protocol. It contains both required v1 arms on the same cohorts and origins. Use `pooled=True` for pooled 91-day results. The Retail II row labeled `2010-12-09 (primary, 365d)` has `pooled=False` and must not be averaged into the five rolling origins.

### v2 three-method comparison

`results/v2_hybrid/per_origin_metrics.csv` is generated by `scripts/v2_hybrid_ladder.py`. Its plan states that the v2 arms use the same origins, folds, seeds, bootstrap procedure, and downstream models as v1. Its five v1-shared arms reproduce v1 values exactly according to `docs/V2_HYBRID_RESULTS.md`. Therefore the three required arms can be selected from this one CSV without mixing files or protocols. Use `pooled=True` for the rolling-origin graph; keep the separate Retail II primary holdout distinct.

The v2 comparison is nevertheless **post hoc and exploratory**: the plan was written after v1 results were known, and the repository explicitly says it does not replace v1 evidence.

### CDNOW confirmation

`results/cdnow_confirmation/per_origin_metrics.csv` is generated by `scripts/cdnow_confirmation.py` under `docs/CDNOW_CONFIRMATION_PLAN.md`. It contains all three required FCA methods, three chronological 91-day origins, five folds within each origin, and the pre-specified CDNOW tie/empty-band handling. `results/cdnow_confirmation_verification/` independently verifies the committed comparisons, but its CSVs are pairwise verification outputs rather than graph metric values.

### Incompatible or incomplete candidates

- The v1 per-origin file cannot supply Hybrid Fuzzy RFM-FCA.
- The tail-information file cannot supply Crisp RFM-FCA and its hybrid is a distinct exploratory arm.
- The membership-ablation `full_run/` does not contain the main-study identifiers and is explicitly excluded.
- The derived output verification table should not be treated as an independent source because it combines baseline, tail-information, and CDNOW files and maps stored identifiers to display labels.

## 4. Generation and provenance trace

| Purpose | Generator | Authoritative input |
|---|---|---|
| Frozen v1 rolling-origin metrics | `scripts/baseline_ladder_rolling_origin.py` | `results/baseline_ladder_rolling_origin/per_origin_metrics.csv` |
| Frozen v1 inference | same runner | `results/baseline_ladder_rolling_origin/paired_comparisons.csv` |
| v2 hybrid metrics | `scripts/v2_hybrid_ladder.py` | `results/v2_hybrid/per_origin_metrics.csv` |
| v2 hybrid inference | same runner | `results/v2_hybrid/paired_comparisons.csv` |
| CDNOW confirmation metrics | `scripts/cdnow_confirmation.py` | `results/cdnow_confirmation/per_origin_metrics.csv` |
| CDNOW independent comparison verification | `scripts/verify_cdnow_confirmation.py` | `results/cdnow_confirmation_verification/recomputed_comparisons.csv` and `checks.csv` |
| Paper evidence map | `scripts/build_paper_evidence_map.py` | References the v1/v2/CDNOW result CSVs; validated by `scripts/validate_paper_evidence_map.py` |
| Existing derived graph audit | `analysis/plot_experimental_results.py` | Combines baseline, tail-information, and CDNOW sources; its outputs are not the canonical raw sources |

The evidence map and README identify v1 as frozen primary evidence and v2 as exploratory/post hoc. No source CSV under `evidence/` was present in the current working tree; the repository uses tagged/history-backed evidence and the `results/` paths above.

## 5. Provenance gaps and unresolved questions

1. The v2 plan/results state that plan-before-run ordering is not established by Git; filesystem times are not tamper-evident.
2. The v2 hybrid result CSV is tracked, but the v2 hybrid script and plan status should still be read as exploratory/post hoc, not preregistered confirmation.
3. The existing derived figure outputs are untracked and include Raw RFM; they are not recommended as the source for the requested three-method graphs.
4. The current working tree contains untracked exploratory files, including tail-information and membership-ablation outputs. They must not be silently folded into the canonical graph source.
5. CDNOW has no dedicated pooled metric CSV containing the three methods; future graph code must aggregate the per-origin values or plot origins separately, preserving full precision.

## 6. Go/no-go

**Go for a controlled graph-generation step using the exact source selections above**, provided the graphs are labeled by study status:

- v1 Crisp/Fuzzy: frozen primary;
- v2 three-method comparison: exploratory/post hoc;
- CDNOW three-method comparison: confirmation dataset, with its separate protocol and origin handling.

**Do not** generate a single undifferentiated graph that combines v1, v2, and CDNOW pooled values as though they were one protocol. Do not use Markdown-rounded values, the membership-ablation `full_run/`, the tail-information CSV as a v2 substitute, or the derived verification table as the authoritative raw source.

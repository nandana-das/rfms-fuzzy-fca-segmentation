# Łukasiewicz t-norm ablation — provenance

Status: **single seed, exploratory.** No analysis plan was committed before this run.

## How the files were produced

- CSVs and `run_parameters.json` were written 2026-10-13 22:27:22 by `scripts/lukasiewicz_tnorm_ablation.py`
  (runtime 6.5 minutes).
- The script was edited afterwards (22:27:43); its committed version is that later edit. The M4
  computation lives in `scripts/ablation_ladder_rolling_origin.py`, last functionally changed at 22:20,
  before the CSVs were written. Later edits to that file (2026-10-14) touch the docstring only.
- `REPORT.md` was edited by hand after generation (22:28:16). It omits the pooled-means table the script
  writes and rewords some sections. It is preserved as written and was not regenerated; rerunning the
  script would replace it with the script's own report text.

## Verification (2026-10-14, no rerun)

- `REPORT.md` comparison table vs `paired_comparisons.csv`: 18 rows, 0 mismatches.
- Pooled deltas recomputed from `per_origin_metrics.csv`: 0 mismatches.
- Holm adjustment recomputed from `p_boot` (9 pooled tests per dataset): 0 mismatches.
- `run_parameters.json` matches the committed script's parameter block.
- M0 vs `results/baseline_ladder_rolling_origin/` `fuzzy_rfm_fca`: 10 origins, max abs difference 0.0.
- M0 and product (M2) vs `results/fuzzy_tnorm_tail_ablation/multi_origin_metrics.csv`: 18 rows, max abs
  difference 0.0.
- M4 has no independent reference; its formula is covered by `tests/test_lukasiewicz_tnorm.py`.

Conclusion: no numerical discrepancy found; the remaining issue is presentation (hand-edited report).

## Scope limits

Concepts are mined and suppressed once with Gödel-min on the threshold context; only the aggregation of
each concept's memberships changes. This is not continuous concept discovery. Pooled intervals average
independent per-origin bootstraps although origins share customers. Consolidated in
`results/consolidated_evidence/CONSOLIDATED_RESULTS.md` §6.

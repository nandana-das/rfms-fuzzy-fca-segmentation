# Membership-ablation `full_run/` audit

**Audit date:** 2026-10-10  
**Repository:** `D:\Nandana\MTECH\Semester 3\Projects\MP\rfms_fca_project`  
**Mode:** read-only inspection. No experiment, training job, regeneration, or existing-file modification was performed.

## Conclusion

`results/membership_function_ablation/full_run/` is the output of a complete execution of the implemented ablation protocol, not a smoke test. The strongest evidence is the directory's `run_parameters.json`, which records `smoke: false`, `bootstrap_n: 2000`, and all ten intended cohorts, together with output cardinalities covering 10 cohorts × 4 membership-function arms × 5 folds.

The outputs are scientifically usable only as exploratory ablation evidence. They are not part of the frozen v1 evidence, are not confirmatory evidence, and should not silently replace or update the existing smoke-only documentation. The artifacts support descriptive full-protocol comparisons, but exact rerun provenance is incomplete because no execution log, saved bootstrap indices, or committed Git revision for this run was found.

**Recommendation:** preserve the artifacts as-is and investigate provenance/documentation before citing any full-run result. A rerun is not justified solely by the smoke/full-run labeling conflict.

## 1. Artifact inventory

| File | Size | Rows/content | Role | Git status |
|---|---:|---:|---|---|
| `concept_counts_per_fold.csv` | 21,557 B | 200 rows | Per-cohort, per-fold concept counts and descriptive membership diagnostics | Untracked |
| `concept_stability.csv` | 1,978 B | 40 rows | One concept-set Jaccard row per cohort and method | Untracked |
| `customer_splits.csv` | 833,318 B | 200 rows | Customer IDs for the five shared test folds, repeated for each method | Untracked |
| `paired_comparisons.csv` | 12,860 B | 108 rows | Per-origin and pooled method-minus-M0 comparisons, intervals, bootstrap and Holm fields | Untracked |
| `per_origin_metrics.csv` | 4,738 B | 40 rows | AUC, spend R², invoice R² for each method and cohort | Untracked |
| `REPORT.md` | 14,462 B | Generated report | Metrics, paired comparisons, stability table, runtime | Untracked |
| `run_parameters.json` | 1,004 B | Run metadata | Methods, thresholds, folds, seeds, bootstrap count, smoke flag, selected cohorts | Untracked |

No separate log, stdout capture, fold-specific parameter archive, OOF prediction archive, or bootstrap-index archive was found in `full_run/`. The repository's `logs/` directory contains only three unrelated `stage2_run*.log` files; no membership-ablation execution log was found.

## 2. What actually executed

### Membership-function arms

The runner defines and the metadata records all four planned arms:

- **M0:** frozen centroid-based piecewise-linear memberships with saturating outer shoulders;
- **M1:** normalized Gaussian memberships;
- **M2:** normalized generalized-bell memberships with fixed `b = 2`;
- **M3:** normalized Gaussian memberships on `log1p(R)`, `log1p(F)`, and `log1p(M)` coordinates.

The implementation uses the documented thresholds `(0.3, 0.5, 0.7)`, minimum support `0.04`, redundancy threshold `J_max = 0.80`, membership core cutoff `0.50`, five folds, CV seed `0`, bootstrap seed `12345`, and 2,000 bootstrap draws for non-smoke execution. These values are independently visible in `scripts/membership_function_ablation.py` and `full_run/run_parameters.json`.

### Datasets and cohorts

The metadata lists ten selected cohorts:

- Dunnhumby: days 347, 438, 529, and 620;
- Online Retail II: five pooled rolling origins dated 2010-09-10, 2010-12-10, 2011-03-11, 2011-06-10, and 2011-09-09;
- Online Retail II primary 365-day cohort dated 2010-12-09, explicitly marked non-pooled.

`per_origin_metrics.csv` contains exactly 40 rows: 10 cohorts × 4 methods. The four Dunnhumby cohorts and the six Retail II cohorts each have four method rows. The primary Retail II cohort has four metric rows but is excluded from pooled inference, consistent with the script's `pooled` flags and the plan.

### Folds and outputs

- `concept_counts_per_fold.csv`: 200 rows = 10 cohorts × 5 folds × 4 methods.
- `customer_splits.csv`: 200 rows with the same structure, indicating shared customer splits across methods.
- `concept_stability.csv`: 40 rows = 10 cohorts × 4 methods.
- `paired_comparisons.csv`: 108 rows, including per-origin comparisons and pooled comparisons. Pooled output has 9 rows per dataset: 3 alternative arms × 3 metrics for Dunnhumby and Online Retail II.
- `REPORT.md` records a runtime of 10.98 minutes and is labeled `Membership-function ablation results`, not smoke results.

The metrics are ROC AUC for repurchase, R² for log-transformed future spend, and R² for log-transformed future invoices. The script uses fold-local feature construction and downstream LogisticRegressionCV/RidgeCV models, then computes paired customer-bootstrap deltas against M0 and Holm-adjusts pooled comparisons.

## 3. Completed versus missing

### Completed according to artifacts

- All four membership functions were evaluated.
- Both intended datasets were evaluated.
- All four Dunnhumby and five rolling Retail II origins were evaluated.
- The separate Retail II 365-day primary cohort was evaluated and retained separately.
- Five stratified customer folds were run for every cohort and method.
- Per-origin predictive metrics were written.
- Per-fold concept counts and shared customer split identifiers were written.
- Concept-set Jaccard diagnostics were written.
- Paired bootstrap comparison tables were written with `bootstrap_n = 2000` metadata.
- Pooled inference was separated from the non-pooled primary Retail II cohort.

### Not available or not independently established

- No execution log proves the process exited successfully beyond the internally generated complete output set.
- No saved bootstrap index matrix or OOF predictions are available for independent numerical recomputation of every interval and p-value.
- No independent verification script/report exists for this ablation analogous to the CDNOW verification artifacts.
- No full-run documentation update exists: the plan and both smoke reports still say full evaluation was not launched or remains blocked.
- The results are not committed in Git. `git status` shows the full-run files as untracked, and `git log --follow` returns no history for them. The named local branch `experiment/membership-function-ablation` exists, but its committed tree does not contain these artifacts; it points to the shared pre-ablation history.

## 4. Protocol comparison and deviations

### Agreement with the planned protocol

The observed metadata and output structure agree with the plan on the four arms, RFM-only scope, training-only feature construction, five folds, thresholds, support and suppression settings, rolling origins, separate primary cohort, paired comparisons, and exploratory status.

### Deviations or status conflicts

1. **Documentation conflict:** `docs/MEMBERSHIP_FUNCTION_ABLATION_PLAN.md` says the full evaluation was intentionally not launched, and `docs/MEMBERSHIP_FUNCTION_ABLATION_RESULTS.md` plus `docs/MEMBERSHIP_FUNCTION_ABLATION_CORRECTED_RESULTS.md` describe only a one-origin, 200-draw smoke run. The `full_run/` metadata and output cardinalities contradict that current documentation.
2. **Missing provenance artifacts:** the full run has no captured command line, host/environment fingerprint, completion log, or persisted bootstrap indices. This limits exact independent auditability but does not make the output set internally inconsistent.
3. **No new scientific protocol is apparent:** the full-run metadata does not show a changed method, threshold, split, or dataset selection relative to the plan. The actual runner's `smoke: false` path selects all cohorts and uses 2,000 draws.

## 5. Scientific usability

The artifacts are suitable for **exploratory, descriptive full-protocol ablation analysis**, subject to clearly labeling them as uncommitted and post hoc relative to the frozen v1 study. They are not suitable as:

- frozen primary evidence;
- confirmatory evidence;
- a replacement for the v1 results;
- evidence that a membership function is generally superior without a pre-specified confirmatory analysis and independent verification;
- a basis for claiming structural stability, since the recorded stability measure is fold-level concept-set Jaccard and not the repository's customer-level temporal stability study.

The smoke reports' numerical claims should remain interpreted as smoke claims. The full-run outputs may support a new, explicitly labeled exploratory report after provenance is reconciled, but this audit does not promote them or alter existing claims.

## 6. Uncertainties

- It cannot be determined from the repository whether the full run was launched manually, from an unrecorded command, or by a separate process.
- It cannot be determined whether the output files were copied from another checkout before appearing untracked on `main`.
- The exact bootstrap random draws cannot be reconstructed from the saved artifacts alone, despite the recorded seed, because the generated index matrix was not saved.
- The complete run's relationship to the earlier smoke/corrected-smoke artifacts is not recorded in a manifest; timestamps show the full run was written later, but timestamps alone do not establish authorship or command history.

## 7. Final disposition

Preserve `results/membership_function_ablation/full_run/` unchanged. Investigate and reconcile the documentation/provenance conflict before using its values in a manuscript, evidence map, or confirmatory argument. No rerun, cleanup, or claim update is recommended as part of this read-only audit.

# Data note: customer_predictions.csv.gz

## Contents

39,968 rows: Dunnhumby "The Complete Journey", 4 rolling origins × 4 arms × 2,497–2,499 households.
Columns: `dataset, origin, fold, CustomerID, arm, repurchased, y_log_spend, y_log_invoices, pred_prob,
pred_log_spend, pred_log_invoices`. `CustomerID` is dunnhumby's anonymised integer `household_key`.
No names, addresses, or other direct identifiers.

## Status (2026-10-14): not in Git

Redistribution permission has not been established, so the file is git-ignored and was removed from the
unpublished branch history (backup tag `backup/temporal-second-seed-pre-untrack`, local only). The
aggregate outputs in this folder (`paired_comparisons.csv`, `validation.csv`, `REPORT.md`) stay in Git.

To regenerate it (needs the raw Dunnhumby data, ~10 minutes, deterministic):

    python scripts/dunnhumby_matched_complexity_paired_inference.py

The run validates every per-origin metric against the cached CSVs (48/48 at tolerance 1e-15) and stops
before inference if any check fails. Expected SHA-256 of the file produced on 2026-10-14:
`9b988b2abcbe5ad4...` (first 16 hex digits; compression output may differ across zlib versions, so
compare the validation result rather than the hash if they disagree).
`scripts/dunnhumby_temporal_cluster_sensitivity.py` needs this file and exits with these instructions
when it is missing.

## Open question and plan

dunnhumby's Source Files page (https://www.dunnhumby.com/source-files/) presents The Complete Journey for
classroom use and academic research; the Terms & Conditions (https://www.dunnhumby.com/terms-and-conditions/)
limit website materials to research, personal, or non-commercial use and reserve other rights. Neither
clearly covers public redistribution of derived, household-keyed features, holdout outcomes, or model
predictions. Written clarification from dunnhumby is being sought.

- Permission granted in writing: record it here; the file can then be committed if wanted.
- Until then (or if denied): keep it out of Git as above.

Separate issue: `data/processed/dunnhumby_rfm_features.csv` (household-level, public on `main` since
`07fadfe`) needs its own assessment under the same terms; it is not evidence that this file may be published.

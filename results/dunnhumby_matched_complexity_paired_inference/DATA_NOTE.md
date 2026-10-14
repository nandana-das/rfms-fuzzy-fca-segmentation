# Data note: customer_predictions.csv.gz

## Contents

39,968 rows: Dunnhumby "The Complete Journey", 4 rolling origins × 4 arms × 2,497–2,499 households.
Columns: `dataset, origin, fold, CustomerID, arm, repurchased, y_log_spend, y_log_invoices, pred_prob,
pred_log_spend, pred_log_invoices`. `CustomerID` is dunnhumby's anonymised integer `household_key`.
No names, addresses, or other direct identifiers.

## Status (2026-10-14): kept locally, pending redistribution clarification

The branch carrying this file is not pushed. Redistribution permission has not been established;
anonymised keys and the existence of other public files do not settle it. Reasons it is currently
kept on the local branch:

- Needed for clean-checkout reproducibility: `scripts/dunnhumby_temporal_cluster_sensitivity.py` reads it,
  and paired inference in this folder was computed from it.
- Regeneration needs the git-ignored raw Dunnhumby data and ~10 minutes:
  `python scripts/dunnhumby_matched_complexity_paired_inference.py` (validated 48/48 against cached
  metrics at tolerance 1e-15).

## Open question and plan

dunnhumby's Source Files page (https://www.dunnhumby.com/source-files/) presents The Complete Journey for
classroom use and academic research; the Terms & Conditions (https://www.dunnhumby.com/terms-and-conditions/)
limit website materials to research, personal, or non-commercial use and reserve other rights. Neither
clearly covers public redistribution of derived, household-keyed features, holdout outcomes, or model
predictions. Written clarification from dunnhumby is being sought.

- Permission granted in writing: record it here and push.
- Permission unclear or denied: remove this file from the unpublished branch history (a local rewrite; it
  does not touch `main`), have the sensitivity check regenerate it locally from the raw data, and keep the
  aggregate results and regeneration instructions in Git.

Separate issue: `data/processed/dunnhumby_rfm_features.csv` (household-level, public on `main` since
`07fadfe`) needs its own assessment under the same terms; it is not evidence that this file may be published.

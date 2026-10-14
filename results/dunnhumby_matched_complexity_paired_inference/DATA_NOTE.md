# Data note: customer_predictions.csv.gz

## Contents

39,968 rows: Dunnhumby "The Complete Journey", 4 rolling origins × 4 arms × 2,497–2,499 households.
Columns: `dataset, origin, fold, CustomerID, arm, repurchased, y_log_spend, y_log_invoices, pred_prob,
pred_log_spend, pred_log_invoices`. `CustomerID` is dunnhumby's anonymised integer `household_key`.
No names, addresses, or other direct identifiers.

## Why it is kept in the repository (decision 2026-10-14)

- Needed for clean-checkout reproducibility: `scripts/dunnhumby_temporal_cluster_sensitivity.py` reads it,
  and paired inference in this folder was computed from it.
- Regeneration needs the git-ignored raw Dunnhumby data and ~10 minutes:
  `python scripts/dunnhumby_matched_complexity_paired_inference.py` (validated 48/48 against cached
  metrics at tolerance 1e-15).
- Household-level Dunnhumby derived data already exists in the public repository:
  `data/processed/dunnhumby_rfm_features.csv` (all 2,500 households; R, F, M, purchase days, tenure),
  on `main` since commit `07fadfe` (2026-10-01). This file adds holdout outcomes and model predictions
  for the same keys.

## Open question

dunnhumby's terms for redistributing household-level derived data were not verified: the Source Files
page (https://www.dunnhumby.com/source-files/) lists the dataset for classroom and academic research use
but shows no licence text. The question applies equally to `data/processed/dunnhumby_rfm_features.csv`.
If the terms prohibit redistribution, both files should be removed from the repository and its history.

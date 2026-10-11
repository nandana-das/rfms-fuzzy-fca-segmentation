# M1-LAR-PW decision report

## Decision

**B — Promising but exploratory; do not promote M1-LAR-PW to the primary methodology yet.**

The retained nine-origin results are encouraging: M1 exceeds M0 on Invoice R² and Spend R² at all 9 origins, and on AUC at 7/9 origins. Mean paired differences are +0.00338 and +0.00178 on Dunnhumby, and +0.02765 and +0.00277 on Online Retail II, for Invoice and Spend R² respectively. These are descriptive origin summaries, not confirmatory inference.

Promotion is not justified because the retained artifacts lack customer-level paired predictions, M1-specific refit/temporal stability results, seed variation, an independent M1 validation, and an execution commit/branch. The code also uses mean deviation from `a4` for the zero-IQR fallback although the requested specification calls for MAD; this must be resolved before a definitive method claim.

## Supported claim

“Across four Dunnhumby and five Online Retail II rolling origins, the exploratory M1-LAR-PW ablation produced favorable paired point estimates relative to M0, most consistently for the two R² outcomes.”

## Unsupported claims

- M1-LAR-PW is superior or confirmatory.
- M1 improves segment stability or interpretability.
- M1 is better than Gaussian/generalized-bell alternatives.
- M1 is better than the log-RFM hybrid.
- The nine origins are independent replications.
- The current fallback specification is the requested MAD estimator.

## Manuscript recommendation

Keep M1 in the supplementary exploratory ablation. The main paper should retain the foundational Crisp RFM-FCA versus M0 question. If M1 is later promoted, add a separate membership-function research question and report it after the foundational comparison; keep the Gaussian/generalized-bell and log-RFM hybrid experiments explicitly separate. The methods, results, abstract, limitations, and conclusion must state that M1 evidence is prospective/exploratory until paired uncertainty, M1 stability, and validation evidence are completed.

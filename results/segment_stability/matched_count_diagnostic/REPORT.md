# Matched-Concept-Count Stability Diagnostic

Rule and interpretation fixed in `docs/AUDIT_ERRATA.md` §6.1 before this was run. Fuzzy RFM-FCA is truncated, per fit, to the crisp concept count using the suppressor's own quality order.

## Verdict (temporal re-segmentation, all customers, core Jaccard)

| dataset | full_gap | matched_gap | matched_ci_low | matched_ci_high | verdict |
|---|---|---|---|---|---|
| Dunnhumby | -0.0346 | 0.0027 | -0.0020 | 0.0079 | explained by count |
| Online Retail II | -0.0201 | 0.0028 | -0.0006 | 0.0060 | explained by count |

## Refit stability

| dataset | arm | core_jaccard | fuzzy_jaccard | concept_recurrence | n_concepts_ref | core_concepts_per_customer |
|---|---|---|---|---|---|---|
| Dunnhumby | crisp_rfm_fca | 0.9560 | 0.9560 | 0.9711 | 54.5000 | 5.5440 |
| Dunnhumby | fuzzy_rfm_fca | 0.9505 | 0.9182 | 0.9452 | 92.0000 | 6.4966 |
| Dunnhumby | fuzzy_rfm_fca_matched | 0.9466 | 0.9290 | 0.9601 | 54.5000 | 5.5728 |
| Online Retail II | crisp_rfm_fca | 0.9701 | 0.9701 | 0.9824 | 48.6000 | 5.3147 |
| Online Retail II | fuzzy_rfm_fca | 0.9746 | 0.9547 | 0.9697 | 75.8000 | 6.0572 |
| Online Retail II | fuzzy_rfm_fca_matched | 0.9712 | 0.9545 | 0.9691 | 48.6000 | 5.2178 |

| dataset | comparison | measure | delta | interval_low | interval_high | origins_positive |
|---|---|---|---|---|---|---|
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | core_jaccard | -0.0093 | -0.0224 | -0.0015 | 0/4 |
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | fuzzy_jaccard | -0.0269 | -0.0443 | -0.0115 | 0/4 |
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | concept_recurrence | -0.0110 | -0.0274 | 0.0019 | 1/4 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | core_jaccard | -0.0054 | -0.0203 | 0.0042 | 1/4 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | fuzzy_jaccard | -0.0378 | -0.0574 | -0.0211 | 0/4 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | concept_recurrence | -0.0259 | -0.0434 | -0.0048 | 0/4 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | core_jaccard | -0.0039 | -0.0119 | 0.0040 | 1/4 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | fuzzy_jaccard | 0.0109 | 0.0018 | 0.0202 | 4/4 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | concept_recurrence | 0.0149 | -0.0025 | 0.0277 | 4/4 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | core_jaccard | 0.0011 | -0.0067 | 0.0106 | 3/5 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | fuzzy_jaccard | -0.0156 | -0.0261 | -0.0027 | 0/5 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | concept_recurrence | -0.0133 | -0.0231 | -0.0020 | 0/5 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | core_jaccard | 0.0045 | -0.0023 | 0.0161 | 2/5 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | fuzzy_jaccard | -0.0154 | -0.0245 | -0.0027 | 0/5 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | concept_recurrence | -0.0127 | -0.0257 | 0.0034 | 1/5 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | core_jaccard | -0.0033 | -0.0114 | 0.0035 | 1/5 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | fuzzy_jaccard | -0.0002 | -0.0064 | 0.0088 | 3/5 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | concept_recurrence | -0.0006 | -0.0130 | 0.0145 | 2/5 |

## Temporal re-segmentation

| dataset | arm | subset | n_customers | core_jaccard | fuzzy_jaccard |
|---|---|---|---|---|---|
| Dunnhumby | crisp_rfm_fca | all | 2497.6667 | 0.4836 | 0.4836 |
| Dunnhumby | crisp_rfm_fca | stable | 379.6667 | 0.8785 | 0.8785 |
| Dunnhumby | fuzzy_rfm_fca | all | 2497.6667 | 0.4489 | 0.4403 |
| Dunnhumby | fuzzy_rfm_fca | stable | 379.6667 | 0.8769 | 0.8380 |
| Dunnhumby | fuzzy_rfm_fca_matched | all | 2497.6667 | 0.4863 | 0.4835 |
| Dunnhumby | fuzzy_rfm_fca_matched | stable | 379.6667 | 0.8704 | 0.8434 |
| Online Retail II | crisp_rfm_fca | all | 4318.2500 | 0.5266 | 0.5266 |
| Online Retail II | crisp_rfm_fca | stable | 1109.2500 | 0.8741 | 0.8741 |
| Online Retail II | fuzzy_rfm_fca | all | 4318.2500 | 0.5065 | 0.4953 |
| Online Retail II | fuzzy_rfm_fca | stable | 1109.2500 | 0.8945 | 0.8623 |
| Online Retail II | fuzzy_rfm_fca_matched | all | 4318.2500 | 0.5294 | 0.5196 |
| Online Retail II | fuzzy_rfm_fca_matched | stable | 1109.2500 | 0.8894 | 0.8606 |

| dataset | comparison | subset | measure | delta | ci_low | ci_high | p_boot | pairs_positive | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | core_jaccard | 0.0027 | -0.0020 | 0.0079 | 0.2860 | 3/3 | 1.0000 |
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | fuzzy_jaccard | -0.0001 | -0.0051 | 0.0050 | 0.9570 | 1/3 | 1.0000 |
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | core_jaccard | -0.0081 | -0.0240 | 0.0075 | 0.3180 | 0/3 | 1.0000 |
| Dunnhumby | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0351 | -0.0479 | -0.0221 | 0.0005 | 0/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | all | core_jaccard | -0.0346 | -0.0396 | -0.0292 | 0.0005 | 0/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | all | fuzzy_jaccard | -0.0433 | -0.0484 | -0.0383 | 0.0005 | 0/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | stable | core_jaccard | -0.0016 | -0.0180 | 0.0148 | 0.8370 | 1/3 | 1.0000 |
| Dunnhumby | fuzzy_rfm_fca - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0404 | -0.0536 | -0.0270 | 0.0005 | 0/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | core_jaccard | 0.0374 | 0.0359 | 0.0389 | 0.0005 | 3/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | fuzzy_jaccard | 0.0432 | 0.0423 | 0.0443 | 0.0005 | 3/3 | 0.0060 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | core_jaccard | -0.0066 | -0.0106 | -0.0024 | 0.0020 | 0/3 | 0.0100 |
| Dunnhumby | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | fuzzy_jaccard | 0.0054 | 0.0027 | 0.0079 | 0.0005 | 2/3 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | core_jaccard | 0.0028 | -0.0006 | 0.0060 | 0.1110 | 3/4 | 0.2220 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | fuzzy_jaccard | -0.0071 | -0.0100 | -0.0040 | 0.0005 | 1/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | core_jaccard | 0.0153 | 0.0070 | 0.0234 | 0.0005 | 3/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0135 | -0.0221 | -0.0051 | 0.0010 | 1/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | all | core_jaccard | -0.0201 | -0.0236 | -0.0166 | 0.0005 | 0/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | all | fuzzy_jaccard | -0.0314 | -0.0345 | -0.0282 | 0.0005 | 0/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | stable | core_jaccard | 0.0204 | 0.0125 | 0.0287 | 0.0005 | 4/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0118 | -0.0201 | -0.0034 | 0.0030 | 1/4 | 0.0120 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | core_jaccard | 0.0229 | 0.0216 | 0.0240 | 0.0005 | 4/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | fuzzy_jaccard | 0.0243 | 0.0234 | 0.0252 | 0.0005 | 4/4 | 0.0060 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | core_jaccard | -0.0050 | -0.0083 | -0.0018 | 0.0030 | 2/4 | 0.0120 |
| Online Retail II | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | fuzzy_jaccard | -0.0016 | -0.0038 | 0.0005 | 0.1240 | 2/4 | 0.2220 |

Caveat: truncation keeps fuzzy's highest-support concepts while crisp keeps all of its concepts, so this design can only favour the matched fuzzy arm.

Runtime: 13.4 min.

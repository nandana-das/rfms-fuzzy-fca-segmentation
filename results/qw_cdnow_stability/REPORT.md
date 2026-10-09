# CDNOW segment stability (limitation follow-up, Part A)

Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). v1 stability protocol on the three CDNOW origins.

**Decision-rule outcome: H3 replicates as not supported on CDNOW.**

## Refit stability (80% subsamples vs full fit)

| dataset | arm | core_jaccard | fuzzy_jaccard | concept_recurrence | n_concepts_ref | core_concepts_per_customer |
|---|---|---|---|---|---|---|
| CDNOW | crisp_rfm_fca | 0.9751 | 0.9751 | 0.9821 | 50.3333 | 5.9492 |
| CDNOW | fuzzy_rfm_fca | 0.9818 | 0.9606 | 0.9694 | 58.6667 | 5.4496 |
| CDNOW | fuzzy_rfm_fca_matched | 0.9750 | 0.9565 | 0.9569 | 50.3333 | 5.2176 |

| dataset | comparison | measure | delta | interval_low | interval_high | origins_positive |
|---|---|---|---|---|---|---|
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | core_jaccard | 0.0066 | -0.0026 | 0.0178 | 3/3 |
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | fuzzy_jaccard | -0.0145 | -0.0313 | 0.0039 | 0/3 |
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | concept_recurrence | -0.0126 | -0.0277 | 0.0020 | 0/3 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | core_jaccard | -0.0001 | -0.0088 | 0.0089 | 1/3 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | fuzzy_jaccard | -0.0187 | -0.0373 | -0.0013 | 0/3 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | concept_recurrence | -0.0252 | -0.0398 | -0.0066 | 0/3 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | core_jaccard | -0.0067 | -0.0135 | -0.0010 | 0/3 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | fuzzy_jaccard | -0.0041 | -0.0132 | 0.0036 | 0/3 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | concept_recurrence | -0.0125 | -0.0261 | 0.0078 | 0/3 |

## Quarterly re-segmentation

| dataset | pair | arm | subset | n_customers | core_jaccard | fuzzy_jaccard |
|---|---|---|---|---|---|---|
| CDNOW | 1997-09-30 -> 1997-12-31 | crisp_rfm_fca | all | 23500 | 0.6968 | 0.6968 |
| CDNOW | 1997-09-30 -> 1997-12-31 | crisp_rfm_fca | stable | 14275 | 0.8399 | 0.8399 |
| CDNOW | 1997-09-30 -> 1997-12-31 | fuzzy_rfm_fca | all | 23500 | 0.6167 | 0.5866 |
| CDNOW | 1997-09-30 -> 1997-12-31 | fuzzy_rfm_fca | stable | 14275 | 0.7593 | 0.7031 |
| CDNOW | 1997-09-30 -> 1997-12-31 | fuzzy_rfm_fca_matched | all | 23500 | 0.6162 | 0.5876 |
| CDNOW | 1997-09-30 -> 1997-12-31 | fuzzy_rfm_fca_matched | stable | 14275 | 0.7609 | 0.7079 |
| CDNOW | 1997-12-31 -> 1998-03-31 | crisp_rfm_fca | all | 23502 | 0.7844 | 0.7844 |
| CDNOW | 1997-12-31 -> 1998-03-31 | crisp_rfm_fca | stable | 16373 | 0.9012 | 0.9012 |
| CDNOW | 1997-12-31 -> 1998-03-31 | fuzzy_rfm_fca | all | 23502 | 0.7905 | 0.7344 |
| CDNOW | 1997-12-31 -> 1998-03-31 | fuzzy_rfm_fca | stable | 16373 | 0.9041 | 0.8447 |
| CDNOW | 1997-12-31 -> 1998-03-31 | fuzzy_rfm_fca_matched | all | 23502 | 0.7911 | 0.7363 |
| CDNOW | 1997-12-31 -> 1998-03-31 | fuzzy_rfm_fca_matched | stable | 16373 | 0.9024 | 0.8465 |

| dataset | comparison | subset | measure | delta | ci_low | ci_high | p_boot | pairs_positive | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | all | core_jaccard | -0.0370 | -0.0399 | -0.0342 | 0.0005 | 1/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | all | fuzzy_jaccard | -0.0801 | -0.0825 | -0.0778 | 0.0005 | 0/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | stable | core_jaccard | -0.0388 | -0.0423 | -0.0351 | 0.0005 | 1/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0966 | -0.0996 | -0.0935 | 0.0005 | 0/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | core_jaccard | -0.0370 | -0.0398 | -0.0343 | 0.0005 | 1/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | all | fuzzy_jaccard | -0.0787 | -0.0810 | -0.0764 | 0.0005 | 0/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | core_jaccard | -0.0389 | -0.0424 | -0.0353 | 0.0005 | 1/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - crisp_rfm_fca | stable | fuzzy_jaccard | -0.0933 | -0.0962 | -0.0902 | 0.0005 | 0/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | core_jaccard | 0.0000 | -0.0005 | 0.0005 | 0.9530 | 1/2 | 1.0000 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | all | fuzzy_jaccard | 0.0014 | 0.0011 | 0.0017 | 0.0005 | 2/2 | 0.0060 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | core_jaccard | -0.0001 | -0.0007 | 0.0005 | 0.7560 | 1/2 | 1.0000 |
| CDNOW | fuzzy_rfm_fca_matched - fuzzy_rfm_fca | stable | fuzzy_jaccard | 0.0033 | 0.0029 | 0.0037 | 0.0005 | 2/2 | 0.0060 |

Runtime: 22.1 min.

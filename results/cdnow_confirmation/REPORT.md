# CDNOW confirmation — results

Plan: `docs/CDNOW_CONFIRMATION_PLAN.md` (committed before this script). One run; no tuning.

## Decision-rule outcomes

| finding | rule | outcome |
|---|---|---|
| v1 H1 (fuzzy > crisp RFM-FCA) | C1 significantly positive on all 3 metrics | replicates |
| v1 H2 (fuzzy vs spline) | C2 per metric (non-significant is not equivalence) | auc: significantly higher; spend_r2: significantly lower; invoice_r2: significantly lower |
| v2 beats spline | C3 per metric (non-significant is not equivalence) | auc: significantly higher; spend_r2: significantly higher; invoice_r2: significantly higher |
| v2 improves on v1 | C4 significantly positive, none significantly negative | confirmed on: spend_r2, invoice_r2 |
| v2 concepts add beyond linear log RFM | C6 significantly positive (per metric) | supported on: spend_r2, invoice_r2 |
| fuzzification helps within the hybrid | C5 significantly positive (per metric) | supported on: auc, spend_r2, invoice_r2 |

## Mean metrics over the 3 origins

| arm | auc | spend_r2 | invoice_r2 |
|---|---|---|---|
| raw_std | 0.8021 | 0.2501 | 0.2934 |
| log_rfm | 0.8057 | 0.2557 | 0.2774 |
| spline_log_rfm | 0.8020 | 0.2704 | 0.3073 |
| crisp_rfm_fca | 0.8005 | 0.2417 | 0.2475 |
| fuzzy_rfm_fca | 0.8064 | 0.2614 | 0.2747 |
| hybrid_crisp_fca | 0.8050 | 0.2724 | 0.3066 |
| hybrid_fuzzy_fca | 0.8070 | 0.2753 | 0.3104 |

## Comparisons (Holm over 21 tests; origins_positive is descriptive)

| id | comparison | metric | delta | ci_low | ci_high | p_boot | origins_positive | p_holm |
|---|---|---|---|---|---|---|---|---|
| C1 | fuzzy_rfm_fca - crisp_rfm_fca | auc | 0.0060 | 0.0044 | 0.0075 | 0.0005 | 3/3 | 0.0105 |
| C1 | fuzzy_rfm_fca - crisp_rfm_fca | spend_r2 | 0.0197 | 0.0170 | 0.0223 | 0.0005 | 3/3 | 0.0105 |
| C1 | fuzzy_rfm_fca - crisp_rfm_fca | invoice_r2 | 0.0272 | 0.0237 | 0.0302 | 0.0005 | 3/3 | 0.0105 |
| C2 | fuzzy_rfm_fca - spline_log_rfm | auc | 0.0044 | 0.0030 | 0.0058 | 0.0005 | 1/3 | 0.0105 |
| C2 | fuzzy_rfm_fca - spline_log_rfm | spend_r2 | -0.0090 | -0.0123 | -0.0057 | 0.0005 | 0/3 | 0.0105 |
| C2 | fuzzy_rfm_fca - spline_log_rfm | invoice_r2 | -0.0326 | -0.0391 | -0.0261 | 0.0005 | 0/3 | 0.0105 |
| C3 | hybrid_fuzzy_fca - spline_log_rfm | auc | 0.0051 | 0.0035 | 0.0065 | 0.0005 | 2/3 | 0.0105 |
| C3 | hybrid_fuzzy_fca - spline_log_rfm | spend_r2 | 0.0049 | 0.0033 | 0.0064 | 0.0005 | 3/3 | 0.0105 |
| C3 | hybrid_fuzzy_fca - spline_log_rfm | invoice_r2 | 0.0031 | 0.0015 | 0.0047 | 0.0005 | 3/3 | 0.0105 |
| C4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | auc | 0.0006 | -0.0003 | 0.0015 | 0.1930 | 2/3 | 0.1930 |
| C4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | spend_r2 | 0.0140 | 0.0116 | 0.0164 | 0.0005 | 3/3 | 0.0105 |
| C4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | invoice_r2 | 0.0357 | 0.0301 | 0.0411 | 0.0005 | 3/3 | 0.0105 |
| C5 | hybrid_fuzzy_fca - hybrid_crisp_fca | auc | 0.0020 | 0.0005 | 0.0036 | 0.0100 | 3/3 | 0.0300 |
| C5 | hybrid_fuzzy_fca - hybrid_crisp_fca | spend_r2 | 0.0029 | 0.0018 | 0.0040 | 0.0005 | 3/3 | 0.0105 |
| C5 | hybrid_fuzzy_fca - hybrid_crisp_fca | invoice_r2 | 0.0038 | 0.0025 | 0.0050 | 0.0005 | 3/3 | 0.0105 |
| C6 | hybrid_fuzzy_fca - log_rfm | auc | 0.0013 | 0.0001 | 0.0025 | 0.0360 | 3/3 | 0.0720 |
| C6 | hybrid_fuzzy_fca - log_rfm | spend_r2 | 0.0197 | 0.0172 | 0.0221 | 0.0005 | 3/3 | 0.0105 |
| C6 | hybrid_fuzzy_fca - log_rfm | invoice_r2 | 0.0329 | 0.0290 | 0.0371 | 0.0005 | 3/3 | 0.0105 |
| C7 | hybrid_fuzzy_fca - crisp_rfm_fca | auc | 0.0066 | 0.0048 | 0.0082 | 0.0005 | 3/3 | 0.0105 |
| C7 | hybrid_fuzzy_fca - crisp_rfm_fca | spend_r2 | 0.0336 | 0.0298 | 0.0374 | 0.0005 | 3/3 | 0.0105 |
| C7 | hybrid_fuzzy_fca - crisp_rfm_fca | invoice_r2 | 0.0628 | 0.0558 | 0.0698 | 0.0005 | 3/3 | 0.0105 |

## Cohorts

| origin | n | repurchase_rate |
|---|---|---|
| 1997-09-30 | 23500 | 0.1789 |
| 1997-12-31 | 23502 | 0.1630 |
| 1998-03-31 | 23502 | 0.1411 |

Runtime: 13.1 min.

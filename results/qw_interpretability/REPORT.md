# Interpretability proxies (limitation follow-up, Part C)

Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). Outcome-free structural proxies; not measures of human interpretability. Means over origins (ranges in summary.csv). v2 hybrid segments equal fuzzy segments.

| dataset | arm | K | coverage | C80 | intent_length | core_load | overlap |
|---|---|---|---|---|---|---|---|
| Dunnhumby | crisp_rfm_fca | 54.5000 | 1.0000 | 4.2500 | 1.7610 | 5.5440 | 0.0470 |
| Dunnhumby | fuzzy_rfm_fca | 92.0000 | 1.0000 | 4.0000 | 2.0870 | 6.4970 | 0.0280 |
| Online Retail II | crisp_rfm_fca | 48.6000 | 1.0000 | 4.0000 | 1.7610 | 5.3150 | 0.0520 |
| Online Retail II | fuzzy_rfm_fca | 75.8000 | 1.0000 | 4.0000 | 1.9440 | 6.0570 | 0.0350 |
| CDNOW | crisp_rfm_fca | 50.3330 | 1.0000 | 3.0000 | 1.8680 | 5.9490 | 0.0560 |
| CDNOW | fuzzy_rfm_fca | 58.6670 | 1.0000 | 2.6670 | 1.8920 | 5.4500 | 0.0420 |

| dataset | origin | arm | K | coverage | C80 | intent_length | core_load | overlap |
|---|---|---|---|---|---|---|---|---|
| Dunnhumby | day 347 | crisp_rfm_fca | 56 | 1.0000 | 4.0000 | 1.7680 | 5.6320 | 0.0460 |
| Dunnhumby | day 347 | fuzzy_rfm_fca | 94 | 1.0000 | 4.0000 | 2.1060 | 6.5260 | 0.0280 |
| Dunnhumby | day 438 | crisp_rfm_fca | 52 | 1.0000 | 5.0000 | 1.7500 | 5.4580 | 0.0490 |
| Dunnhumby | day 438 | fuzzy_rfm_fca | 93 | 1.0000 | 4.0000 | 2.0970 | 6.4760 | 0.0270 |
| Dunnhumby | day 529 | crisp_rfm_fca | 55 | 1.0000 | 4.0000 | 1.7640 | 5.5550 | 0.0460 |
| Dunnhumby | day 529 | fuzzy_rfm_fca | 90 | 1.0000 | 4.0000 | 2.0670 | 6.4110 | 0.0280 |
| Dunnhumby | day 620 | crisp_rfm_fca | 55 | 1.0000 | 4.0000 | 1.7640 | 5.5310 | 0.0460 |
| Dunnhumby | day 620 | fuzzy_rfm_fca | 91 | 1.0000 | 4.0000 | 2.0770 | 6.5730 | 0.0300 |
| Online Retail II | 2010-09-10 | crisp_rfm_fca | 49 | 1.0000 | 4.0000 | 1.7760 | 5.3300 | 0.0510 |
| Online Retail II | 2010-09-10 | fuzzy_rfm_fca | 67 | 1.0000 | 4.0000 | 1.8510 | 5.7730 | 0.0360 |
| Online Retail II | 2010-12-10 | crisp_rfm_fca | 49 | 1.0000 | 4.0000 | 1.7550 | 5.2930 | 0.0510 |
| Online Retail II | 2010-12-10 | fuzzy_rfm_fca | 77 | 1.0000 | 4.0000 | 1.9090 | 6.2310 | 0.0360 |
| Online Retail II | 2011-03-11 | crisp_rfm_fca | 48 | 1.0000 | 4.0000 | 1.7710 | 5.3200 | 0.0540 |
| Online Retail II | 2011-03-11 | fuzzy_rfm_fca | 77 | 1.0000 | 4.0000 | 1.9610 | 6.2960 | 0.0370 |
| Online Retail II | 2011-06-10 | crisp_rfm_fca | 47 | 1.0000 | 4.0000 | 1.7450 | 5.2470 | 0.0550 |
| Online Retail II | 2011-06-10 | fuzzy_rfm_fca | 80 | 1.0000 | 4.0000 | 2.0000 | 6.0130 | 0.0310 |
| Online Retail II | 2011-09-09 | crisp_rfm_fca | 50 | 1.0000 | 4.0000 | 1.7600 | 5.3830 | 0.0510 |
| Online Retail II | 2011-09-09 | fuzzy_rfm_fca | 78 | 1.0000 | 4.0000 | 2.0000 | 5.9730 | 0.0330 |
| CDNOW | 1997-09-30 | crisp_rfm_fca | 50 | 1.0000 | 3.0000 | 1.8800 | 5.8750 | 0.0550 |
| CDNOW | 1997-09-30 | fuzzy_rfm_fca | 58 | 1.0000 | 2.0000 | 1.8970 | 5.2150 | 0.0400 |
| CDNOW | 1997-12-31 | crisp_rfm_fca | 51 | 1.0000 | 3.0000 | 1.8630 | 5.9770 | 0.0560 |
| CDNOW | 1997-12-31 | fuzzy_rfm_fca | 59 | 1.0000 | 3.0000 | 1.8810 | 5.6530 | 0.0450 |
| CDNOW | 1998-03-31 | crisp_rfm_fca | 50 | 1.0000 | 3.0000 | 1.8600 | 5.9960 | 0.0580 |
| CDNOW | 1998-03-31 | fuzzy_rfm_fca | 59 | 1.0000 | 3.0000 | 1.8980 | 5.4810 | 0.0400 |

Runtime: 0.9 min.

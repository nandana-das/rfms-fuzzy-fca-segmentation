# Inference robustness (limitation follow-up, Part B)

Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). 5 CV fold seeds; customer-clustered bootstrap across origins; B = 10,000 per seed; mixture interval and p over 50,000 draws (floor 2e-5 before Holm); Holm over 12 tests per dataset. Committed results are not replaced.

Verdict counts: robust (significant, same sign): 20; robust (not significant): 11; weakened (no longer significant): 5

| dataset | id | comparison | metric | committed_delta | committed_p_holm | delta_mean_over_seeds | delta_min_seed | delta_max_seed | ci_low | ci_high | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retail2 | K1 | fuzzy_rfm_fca - crisp_rfm_fca | auc | 0.0056 | 0.0135 | 0.0091 | 0.0056 | 0.0120 | 0.0044 | 0.0137 | 0.0002 | robust (significant, same sign) |
| retail2 | K1 | fuzzy_rfm_fca - crisp_rfm_fca | spend_r2 | 0.0139 | 0.0135 | 0.0138 | 0.0133 | 0.0146 | 0.0109 | 0.0169 | 0.0002 | robust (significant, same sign) |
| retail2 | K1 | fuzzy_rfm_fca - crisp_rfm_fca | invoice_r2 | 0.0273 | 0.0135 | 0.0279 | 0.0273 | 0.0292 | 0.0227 | 0.0330 | 0.0002 | robust (significant, same sign) |
| retail2 | K2 | fuzzy_rfm_fca - spline_log_rfm | auc | -0.0006 | 1.0000 | 0.0009 | -0.0006 | 0.0029 | -0.0020 | 0.0038 | 0.7205 | robust (not significant) |
| retail2 | K2 | fuzzy_rfm_fca - spline_log_rfm | spend_r2 | 0.0017 | 1.0000 | 0.0021 | 0.0012 | 0.0030 | -0.0023 | 0.0068 | 0.7205 | robust (not significant) |
| retail2 | K2 | fuzzy_rfm_fca - spline_log_rfm | invoice_r2 | -0.0318 | 0.0135 | -0.0312 | -0.0323 | -0.0298 | -0.0461 | -0.0162 | 0.0022 | robust (significant, same sign) |
| retail2 | K3 | hybrid_fuzzy_fca - spline_log_rfm | auc | 0.0032 | 0.0075 | 0.0030 | 0.0014 | 0.0037 | 0.0005 | 0.0049 | 0.0744 | weakened (no longer significant) |
| retail2 | K3 | hybrid_fuzzy_fca - spline_log_rfm | spend_r2 | 0.0055 | 0.0075 | 0.0060 | 0.0052 | 0.0068 | 0.0030 | 0.0090 | 0.0013 | robust (significant, same sign) |
| retail2 | K3 | hybrid_fuzzy_fca - spline_log_rfm | invoice_r2 | 0.0045 | 0.0075 | 0.0049 | 0.0044 | 0.0056 | 0.0007 | 0.0095 | 0.0789 | weakened (no longer significant) |
| retail2 | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | auc | 0.0038 | 0.0075 | 0.0021 | 0.0004 | 0.0038 | -0.0001 | 0.0047 | 0.2426 | weakened (no longer significant) |
| retail2 | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | spend_r2 | 0.0039 | 0.0075 | 0.0039 | 0.0037 | 0.0041 | 0.0013 | 0.0064 | 0.0247 | robust (significant, same sign) |
| retail2 | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | invoice_r2 | 0.0362 | 0.0075 | 0.0361 | 0.0352 | 0.0367 | 0.0236 | 0.0492 | 0.0002 | robust (significant, same sign) |
| dunnhumby | K1 | fuzzy_rfm_fca - crisp_rfm_fca | auc | 0.0210 | 0.0135 | 0.0207 | 0.0169 | 0.0262 | 0.0124 | 0.0303 | 0.0002 | robust (significant, same sign) |
| dunnhumby | K1 | fuzzy_rfm_fca - crisp_rfm_fca | spend_r2 | 0.0270 | 0.0135 | 0.0269 | 0.0259 | 0.0284 | 0.0212 | 0.0327 | 0.0002 | robust (significant, same sign) |
| dunnhumby | K1 | fuzzy_rfm_fca - crisp_rfm_fca | invoice_r2 | 0.0213 | 0.0135 | 0.0219 | 0.0213 | 0.0226 | 0.0175 | 0.0263 | 0.0002 | robust (significant, same sign) |
| dunnhumby | K2 | fuzzy_rfm_fca - spline_log_rfm | auc | -0.0050 | 0.0960 | -0.0061 | -0.0076 | -0.0047 | -0.0111 | -0.0015 | 0.0531 | robust (not significant) |
| dunnhumby | K2 | fuzzy_rfm_fca - spline_log_rfm | spend_r2 | -0.0060 | 0.1260 | -0.0060 | -0.0073 | -0.0043 | -0.0107 | -0.0013 | 0.0946 | robust (not significant) |
| dunnhumby | K2 | fuzzy_rfm_fca - spline_log_rfm | invoice_r2 | -0.0030 | 1.0000 | -0.0028 | -0.0037 | -0.0016 | -0.0079 | 0.0026 | 0.8857 | robust (not significant) |
| dunnhumby | K3 | hybrid_fuzzy_fca - spline_log_rfm | auc | -0.0011 | 1.0000 | -0.0018 | -0.0036 | -0.0006 | -0.0066 | 0.0024 | 0.8857 | robust (not significant) |
| dunnhumby | K3 | hybrid_fuzzy_fca - spline_log_rfm | spend_r2 | -0.0022 | 1.0000 | -0.0032 | -0.0040 | -0.0022 | -0.0073 | 0.0009 | 0.6044 | robust (not significant) |
| dunnhumby | K3 | hybrid_fuzzy_fca - spline_log_rfm | invoice_r2 | 0.0024 | 0.8800 | 0.0022 | 0.0014 | 0.0029 | -0.0019 | 0.0064 | 0.8857 | robust (not significant) |
| dunnhumby | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | auc | 0.0040 | 0.1540 | 0.0043 | 0.0027 | 0.0060 | 0.0005 | 0.0082 | 0.1438 | robust (not significant) |
| dunnhumby | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | spend_r2 | 0.0038 | 0.1860 | 0.0028 | 0.0018 | 0.0038 | -0.0009 | 0.0064 | 0.6044 | robust (not significant) |
| dunnhumby | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | invoice_r2 | 0.0055 | 0.0075 | 0.0050 | 0.0045 | 0.0055 | 0.0022 | 0.0078 | 0.0058 | robust (significant, same sign) |
| cdnow | K1 | fuzzy_rfm_fca - crisp_rfm_fca | auc | 0.0060 | 0.0105 | 0.0030 | 0.0009 | 0.0060 | -0.0004 | 0.0069 | 0.3412 | weakened (no longer significant) |
| cdnow | K1 | fuzzy_rfm_fca - crisp_rfm_fca | spend_r2 | 0.0197 | 0.0105 | 0.0196 | 0.0191 | 0.0203 | 0.0168 | 0.0225 | 0.0002 | robust (significant, same sign) |
| cdnow | K1 | fuzzy_rfm_fca - crisp_rfm_fca | invoice_r2 | 0.0272 | 0.0105 | 0.0270 | 0.0262 | 0.0277 | 0.0234 | 0.0305 | 0.0002 | robust (significant, same sign) |
| cdnow | K2 | fuzzy_rfm_fca - spline_log_rfm | auc | 0.0044 | 0.0105 | 0.0016 | -0.0016 | 0.0044 | -0.0027 | 0.0054 | 0.6005 | weakened (no longer significant) |
| cdnow | K2 | fuzzy_rfm_fca - spline_log_rfm | spend_r2 | -0.0090 | 0.0105 | -0.0092 | -0.0096 | -0.0088 | -0.0131 | -0.0053 | 0.0002 | robust (significant, same sign) |
| cdnow | K2 | fuzzy_rfm_fca - spline_log_rfm | invoice_r2 | -0.0326 | 0.0105 | -0.0324 | -0.0332 | -0.0316 | -0.0425 | -0.0230 | 0.0002 | robust (significant, same sign) |
| cdnow | K3 | hybrid_fuzzy_fca - spline_log_rfm | auc | 0.0051 | 0.0105 | 0.0045 | 0.0039 | 0.0051 | 0.0027 | 0.0061 | 0.0002 | robust (significant, same sign) |
| cdnow | K3 | hybrid_fuzzy_fca - spline_log_rfm | spend_r2 | 0.0049 | 0.0105 | 0.0047 | 0.0046 | 0.0049 | 0.0032 | 0.0064 | 0.0002 | robust (significant, same sign) |
| cdnow | K3 | hybrid_fuzzy_fca - spline_log_rfm | invoice_r2 | 0.0031 | 0.0105 | 0.0030 | 0.0026 | 0.0034 | 0.0010 | 0.0050 | 0.0118 | robust (significant, same sign) |
| cdnow | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | auc | 0.0006 | 0.1930 | 0.0029 | 0.0002 | 0.0056 | -0.0005 | 0.0064 | 0.3911 | robust (not significant) |
| cdnow | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | spend_r2 | 0.0140 | 0.0105 | 0.0139 | 0.0136 | 0.0142 | 0.0109 | 0.0170 | 0.0002 | robust (significant, same sign) |
| cdnow | K4 | hybrid_fuzzy_fca - fuzzy_rfm_fca | invoice_r2 | 0.0357 | 0.0105 | 0.0354 | 0.0349 | 0.0361 | 0.0272 | 0.0443 | 0.0002 | robust (significant, same sign) |

# Limitation follow-ups — results

Plan: `docs/QUICK_WINS_PLAN.md`, committed (`67976b4`) before any of this code was written or run. These results are additions; they do not replace the frozen v1 evidence, the exploratory v2 results or the CDNOW confirmation. All numbers below are read from `results/qw_cdnow_stability/`, `results/qw_inference_robustness/` and `results/qw_interpretability/`.

## Part A — Segment stability on CDNOW

v1 stability protocol on the three CDNOW origins, with the planned tie-merging extension. Primary measure: core-profile Jaccard. **Decision-rule outcome: H3 replicates as not supported on CDNOW.**

Refit (80% subsamples; 2.5–97.5 percentile interval over 30 subsamples; no p-value). Refit levels: crisp 0.9751, fuzzy 0.9818, matched 0.9750; retained concepts crisp 50.3, fuzzy 58.7.

| Comparison | Δ | Interval | Origins positive (descriptive) |
|---|---|---|---|
| fuzzy − crisp | +0.0066 | [−0.0026, +0.0178] | 3/3 |
| matched − crisp | −0.0001 | [−0.0088, +0.0089] | 1/3 |
| matched − fuzzy | −0.0067 | [−0.0135, −0.0010] | 0/3 |

Quarterly re-segmentation (customer bootstrap, Holm within CDNOW):

| Comparison | Subset | Δ | 95% CI | Holm p | Pairs positive (descriptive) |
|---|---|---|---|---|---|
| fuzzy − crisp | all customers | −0.0370 | [−0.0399, −0.0342] | 0.0060 | 1/2 |
| fuzzy − crisp | unchanged behaviour | −0.0388 | [−0.0423, −0.0351] | 0.0060 | 1/2 |
| matched − crisp | all customers | −0.0370 | [−0.0398, −0.0343] | 0.0060 | 1/2 |
| matched − crisp | unchanged behaviour | −0.0389 | [−0.0424, −0.0353] | 0.0060 | 1/2 |
| matched − fuzzy | all customers | +0.0000 | [−0.0005, +0.0005] | 1.0000 | 1/2 |
| matched − fuzzy | unchanged behaviour | −0.0001 | [−0.0007, +0.0005] | 1.0000 | 1/2 |

Per origin pair (all customers, core Jaccard):

| Pair | Crisp | Fuzzy | Matched |
|---|---|---|---|
| 1997-09-30 -> 1997-12-31 | 0.6968 | 0.6167 | 0.6162 |
| 1997-12-31 -> 1998-03-31 | 0.7844 | 0.7905 | 0.7911 |

Interpretation:
- **No refit difference.** As on Online Retail II and Dunnhumby, refit stability shows no clear difference between fuzzy and crisp.
- **Fuzzy is less stable quarter to quarter**, for all customers and for the unchanged-behaviour subset. On the subset this differs from Online Retail II, where fuzzy was more stable. The pooled gap is driven by the first pair; the second pair is slightly in favour of fuzzy (descriptive).
- **Unlike Online Retail II and Dunnhumby, matching the concept count does not remove the gap on CDNOW.** The fuzzy and crisp concept counts are already close there. The "concept count and/or concept size" explanation from the other two datasets therefore does not carry over to CDNOW, and the source of the CDNOW gap is not identified by this design.
- **H3 remains not supported**, now on all three datasets.

## Part B — Inference robustness

Key comparisons on all three datasets; 5 CV fold seeds (seed 0 reproduces the committed per-origin metrics to within 3×10⁻¹⁵); customer-clustered bootstrap across origins; B = 10,000 per seed; mixture interval and p over 50,000 draws; Holm over 12 tests per dataset.

**Verdicts: robust (significant, same sign): 20; robust (not significant): 11; weakened (no longer significant): 5.** No result changed sign and none became newly significant.

| Dataset | Comparison | Metric | Committed Δ (Holm p) | Δ (mean over seeds) | Range over seeds | 95% interval (mixture) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Online Retail II | fuzzy − crisp (H1) | ROC AUC | +0.0056 (p 0.0135) | +0.0091 | +0.0056 to +0.0120 | [+0.0044, +0.0137] | 0.0002 | robust (significant, same sign) |
| Online Retail II | fuzzy − crisp (H1) | Spend R² | +0.0139 (p 0.0135) | +0.0138 | +0.0133 to +0.0146 | [+0.0109, +0.0169] | 0.0002 | robust (significant, same sign) |
| Online Retail II | fuzzy − crisp (H1) | Invoice R² | +0.0273 (p 0.0135) | +0.0279 | +0.0273 to +0.0292 | [+0.0227, +0.0330] | 0.0002 | robust (significant, same sign) |
| Online Retail II | fuzzy − spline (H2) | ROC AUC | −0.0006 (p 1.0000) | +0.0009 | −0.0006 to +0.0029 | [−0.0020, +0.0038] | 0.7205 | robust (not significant) |
| Online Retail II | fuzzy − spline (H2) | Spend R² | +0.0017 (p 1.0000) | +0.0021 | +0.0012 to +0.0030 | [−0.0023, +0.0068] | 0.7205 | robust (not significant) |
| Online Retail II | fuzzy − spline (H2) | Invoice R² | −0.0318 (p 0.0135) | −0.0312 | −0.0323 to −0.0298 | [−0.0461, −0.0162] | 0.0022 | robust (significant, same sign) |
| Online Retail II | hybrid − spline | ROC AUC | +0.0032 (p 0.0075) | +0.0030 | +0.0014 to +0.0037 | [+0.0005, +0.0049] | 0.0744 | weakened (no longer significant) |
| Online Retail II | hybrid − spline | Spend R² | +0.0055 (p 0.0075) | +0.0060 | +0.0052 to +0.0068 | [+0.0030, +0.0090] | 0.0013 | robust (significant, same sign) |
| Online Retail II | hybrid − spline | Invoice R² | +0.0045 (p 0.0075) | +0.0049 | +0.0044 to +0.0056 | [+0.0007, +0.0095] | 0.0789 | weakened (no longer significant) |
| Online Retail II | hybrid − v1 fuzzy | ROC AUC | +0.0038 (p 0.0075) | +0.0021 | +0.0004 to +0.0038 | [−0.0001, +0.0047] | 0.2426 | weakened (no longer significant) |
| Online Retail II | hybrid − v1 fuzzy | Spend R² | +0.0039 (p 0.0075) | +0.0039 | +0.0037 to +0.0041 | [+0.0013, +0.0064] | 0.0247 | robust (significant, same sign) |
| Online Retail II | hybrid − v1 fuzzy | Invoice R² | +0.0362 (p 0.0075) | +0.0361 | +0.0352 to +0.0367 | [+0.0236, +0.0492] | 0.0002 | robust (significant, same sign) |
| Dunnhumby | fuzzy − crisp (H1) | ROC AUC | +0.0210 (p 0.0135) | +0.0207 | +0.0169 to +0.0262 | [+0.0124, +0.0303] | 0.0002 | robust (significant, same sign) |
| Dunnhumby | fuzzy − crisp (H1) | Spend R² | +0.0270 (p 0.0135) | +0.0269 | +0.0259 to +0.0284 | [+0.0212, +0.0327] | 0.0002 | robust (significant, same sign) |
| Dunnhumby | fuzzy − crisp (H1) | Invoice R² | +0.0213 (p 0.0135) | +0.0219 | +0.0213 to +0.0226 | [+0.0175, +0.0263] | 0.0002 | robust (significant, same sign) |
| Dunnhumby | fuzzy − spline (H2) | ROC AUC | −0.0050 (p 0.0960) | −0.0061 | −0.0076 to −0.0047 | [−0.0111, −0.0015] | 0.0531 | robust (not significant) |
| Dunnhumby | fuzzy − spline (H2) | Spend R² | −0.0060 (p 0.1260) | −0.0060 | −0.0073 to −0.0043 | [−0.0107, −0.0013] | 0.0946 | robust (not significant) |
| Dunnhumby | fuzzy − spline (H2) | Invoice R² | −0.0030 (p 1.0000) | −0.0028 | −0.0037 to −0.0016 | [−0.0079, +0.0026] | 0.8857 | robust (not significant) |
| Dunnhumby | hybrid − spline | ROC AUC | −0.0011 (p 1.0000) | −0.0018 | −0.0036 to −0.0006 | [−0.0066, +0.0024] | 0.8857 | robust (not significant) |
| Dunnhumby | hybrid − spline | Spend R² | −0.0022 (p 1.0000) | −0.0032 | −0.0040 to −0.0022 | [−0.0073, +0.0009] | 0.6044 | robust (not significant) |
| Dunnhumby | hybrid − spline | Invoice R² | +0.0024 (p 0.8800) | +0.0022 | +0.0014 to +0.0029 | [−0.0019, +0.0064] | 0.8857 | robust (not significant) |
| Dunnhumby | hybrid − v1 fuzzy | ROC AUC | +0.0040 (p 0.1540) | +0.0043 | +0.0027 to +0.0060 | [+0.0005, +0.0082] | 0.1438 | robust (not significant) |
| Dunnhumby | hybrid − v1 fuzzy | Spend R² | +0.0038 (p 0.1860) | +0.0028 | +0.0018 to +0.0038 | [−0.0009, +0.0064] | 0.6044 | robust (not significant) |
| Dunnhumby | hybrid − v1 fuzzy | Invoice R² | +0.0055 (p 0.0075) | +0.0050 | +0.0045 to +0.0055 | [+0.0022, +0.0078] | 0.0058 | robust (significant, same sign) |
| CDNOW | fuzzy − crisp (H1) | ROC AUC | +0.0060 (p 0.0105) | +0.0030 | +0.0009 to +0.0060 | [−0.0004, +0.0069] | 0.3412 | weakened (no longer significant) |
| CDNOW | fuzzy − crisp (H1) | Spend R² | +0.0197 (p 0.0105) | +0.0196 | +0.0191 to +0.0203 | [+0.0168, +0.0225] | 0.0002 | robust (significant, same sign) |
| CDNOW | fuzzy − crisp (H1) | Invoice R² | +0.0272 (p 0.0105) | +0.0270 | +0.0262 to +0.0277 | [+0.0234, +0.0305] | 0.0002 | robust (significant, same sign) |
| CDNOW | fuzzy − spline (H2) | ROC AUC | +0.0044 (p 0.0105) | +0.0016 | −0.0016 to +0.0044 | [−0.0027, +0.0054] | 0.6005 | weakened (no longer significant) |
| CDNOW | fuzzy − spline (H2) | Spend R² | −0.0090 (p 0.0105) | −0.0092 | −0.0096 to −0.0088 | [−0.0131, −0.0053] | 0.0002 | robust (significant, same sign) |
| CDNOW | fuzzy − spline (H2) | Invoice R² | −0.0326 (p 0.0105) | −0.0324 | −0.0332 to −0.0316 | [−0.0425, −0.0230] | 0.0002 | robust (significant, same sign) |
| CDNOW | hybrid − spline | ROC AUC | +0.0051 (p 0.0105) | +0.0045 | +0.0039 to +0.0051 | [+0.0027, +0.0061] | 0.0002 | robust (significant, same sign) |
| CDNOW | hybrid − spline | Spend R² | +0.0049 (p 0.0105) | +0.0047 | +0.0046 to +0.0049 | [+0.0032, +0.0064] | 0.0002 | robust (significant, same sign) |
| CDNOW | hybrid − spline | Invoice R² | +0.0031 (p 0.0105) | +0.0030 | +0.0026 to +0.0034 | [+0.0010, +0.0050] | 0.0118 | robust (significant, same sign) |
| CDNOW | hybrid − v1 fuzzy | ROC AUC | +0.0006 (p 0.1930) | +0.0029 | +0.0002 to +0.0056 | [−0.0005, +0.0064] | 0.3911 | robust (not significant) |
| CDNOW | hybrid − v1 fuzzy | Spend R² | +0.0140 (p 0.0105) | +0.0139 | +0.0136 to +0.0142 | [+0.0109, +0.0170] | 0.0002 | robust (significant, same sign) |
| CDNOW | hybrid − v1 fuzzy | Invoice R² | +0.0357 (p 0.0105) | +0.0354 | +0.0349 to +0.0361 | [+0.0272, +0.0443] | 0.0002 | robust (significant, same sign) |

What changes:
- **Weakened results** (5): Online Retail II hybrid − spline ROC AUC; Online Retail II hybrid − spline Invoice R²; Online Retail II hybrid − v1 fuzzy ROC AUC; CDNOW fuzzy − crisp (H1) ROC AUC; CDNOW fuzzy − spline (H2) ROC AUC. Each keeps its sign; it is no longer significant once refit variability and customer clustering across origins are included.
- **v2 hybrid vs spline.** On Online Retail II the advantage now holds only for Spend R². On CDNOW it remains significant on all three metrics. On Dunnhumby it remains not significant.
- **H1 (fuzzy vs crisp)** remains significant on both regression targets on all three datasets, and on AUC on Online Retail II and Dunnhumby. On CDNOW the AUC gain is no longer significant.
- **v1's regression shortfall against the spline** (Online Retail II Invoice R²; CDNOW Spend and Invoice R²) is robust. The CDNOW AUC advantage of v1 over the spline is not.

## Part C — Interpretability proxies

Outcome-free structural proxies; not measures of human interpretability. Means over origins (concept-count range in brackets).

| Dataset | Arm | K retained | Coverage | C80 | Intent length | Core load | Overlap |
|---|---|---|---|---|---|---|---|
| Dunnhumby | crisp | 54.5 (52–56) | 1.000 | 4.25 | 1.76 | 5.54 | 0.047 |
| Dunnhumby | fuzzy (= v2 segments) | 92.0 (90–94) | 1.000 | 4.00 | 2.09 | 6.50 | 0.028 |
| Online Retail II | crisp | 48.6 (47–50) | 1.000 | 4.00 | 1.76 | 5.31 | 0.052 |
| Online Retail II | fuzzy (= v2 segments) | 75.8 (67–80) | 1.000 | 4.00 | 1.94 | 6.06 | 0.035 |
| CDNOW | crisp | 50.3 (50–51) | 1.000 | 3.00 | 1.87 | 5.95 | 0.056 |
| CDNOW | fuzzy (= v2 segments) | 58.7 (58–59) | 1.000 | 2.67 | 1.89 | 5.45 | 0.042 |

Interpretation:
- **Fuzzy is less compact** on the number of retained concepts (all datasets), and on intent length and core load on Online Retail II and Dunnhumby.
- **Fuzzy is more compact** on overlap (lower on all datasets) and needs no more concepts to cover 80% of customers.
- **No claim of greater interpretability follows.** Interpretability remains a limitation, now partly quantified.

## Effect on the report's limitations

| Limitation | Status after these additions |
|---|---|
| CDNOW segment stability not evaluated | Evaluated: no stability advantage on CDNOW. The matched-count explanation does not transfer. |
| Inference ignores refit variability and customer overlap across origins; p-values at the floor | Addressed for the key comparisons. 31 of 36 results unchanged; 5 weakened (listed above). |
| Interpretability not measured | Partly addressed with structural proxies. The result is mixed, and human interpretability is still unmeasured. |

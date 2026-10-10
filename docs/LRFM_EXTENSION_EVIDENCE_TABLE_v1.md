# LRFM extension evidence table â€” version 1

**Status:** exploratory, manuscript-ready audit table
**Frozen source:** `results/lrfm_extension/paired_comparisons.csv`
**Freeze:** 2026-10-10

All estimates are mean paired differences over seeds. Positive values favor the
first arm in the comparison. Intervals are 95% percentile intervals from the
50,000-draw mixture. `p_Holm` is adjusted separately within each dataset over
15 tests. â€œSupportedâ€ means `p_Holm < 0.05`; â€œinconclusiveâ€ does not mean
equivalence.

## Cohort accounting

The Online Retail II data contain six checkpoint origins. The origin
`2010-12-09 (primary, 365d)` is the separately defined 365-day primary cohort
and is not one of the five pooled rolling origins used for the paired
cross-origin inference reported below. It is retained in the checkpoint
inventory and per-origin outputs for the planned primary-cohort analysis. The
Online Retail II rows in this evidence table use only the five pooled rolling
origins. Dunnhumby uses four pooled origins and CDNOW uses three pooled origins.

## Primary evidence

| Dataset | Comparison | Metric | Î” | 95% CI | p_Holm | Decision |
|---|---|---:|---:|---:|---:|---|
| CDNOW | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | AUC | +0.0038 | [âˆ’0.0011, +0.0071] | 1.0000 | Inconclusive |
| CDNOW | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Spend RÂ² | +0.0185 | [+0.0158, +0.0211] | 0.0003 | Supported |
| CDNOW | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Invoice RÂ² | +0.0257 | [+0.0223, +0.0292] | 0.0003 | Supported |
| CDNOW | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | AUC | +0.0005 | [âˆ’0.0029, +0.0034] | 1.0000 | Inconclusive |
| CDNOW | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Spend RÂ² | âˆ’0.0091 | [âˆ’0.0129, âˆ’0.0052] | 0.0003 | Supported, negative |
| CDNOW | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Invoice RÂ² | âˆ’0.0318 | [âˆ’0.0418, âˆ’0.0225] | 0.0003 | Supported, negative |
| CDNOW | E3 hybrid LRFM âˆ’ spline LRFM | AUC | +0.0030 | [+0.0012, +0.0051] | 0.0028 | Supported |
| CDNOW | E3 hybrid LRFM âˆ’ spline LRFM | Spend RÂ² | +0.0045 | [+0.0029, +0.0062] | 0.0003 | Supported |
| CDNOW | E3 hybrid LRFM âˆ’ spline LRFM | Invoice RÂ² | +0.0027 | [+0.0007, +0.0048] | 0.0886 | Inconclusive |
| CDNOW | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | AUC | +0.0003 | [âˆ’0.0017, +0.0048] | 1.0000 | Inconclusive |
| CDNOW | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Spend RÂ² | +0.0002 | [âˆ’0.0004, +0.0008] | 1.0000 | Inconclusive |
| CDNOW | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Invoice RÂ² | +0.0007 | [âˆ’0.0001, +0.0016] | 0.6538 | Inconclusive |
| CDNOW | E5 spline LRFM âˆ’ spline RFM | AUC | +0.0014 | [âˆ’0.0002, +0.0025] | 0.6538 | Inconclusive |
| CDNOW | E5 spline LRFM âˆ’ spline RFM | Spend RÂ² | +0.0000 | [âˆ’0.0003, +0.0004] | 1.0000 | Inconclusive |
| CDNOW | E5 spline LRFM âˆ’ spline RFM | Invoice RÂ² | +0.0001 | [âˆ’0.0003, +0.0006] | 1.0000 | Inconclusive |
| Dunnhumby | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | AUC | +0.0155 | [+0.0049, +0.0277] | 0.0248 | Supported |
| Dunnhumby | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Spend RÂ² | +0.0237 | [+0.0187, +0.0289] | 0.0003 | Supported |
| Dunnhumby | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Invoice RÂ² | +0.0202 | [+0.0164, +0.0242] | 0.0003 | Supported |
| Dunnhumby | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | AUC | âˆ’0.0137 | [âˆ’0.0224, âˆ’0.0058] | 0.0003 | Supported, negative |
| Dunnhumby | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Spend RÂ² | âˆ’0.0103 | [âˆ’0.0150, âˆ’0.0055] | 0.0003 | Supported, negative |
| Dunnhumby | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Invoice RÂ² | âˆ’0.0074 | [âˆ’0.0122, âˆ’0.0024] | 0.0396 | Supported, negative |
| Dunnhumby | E3 hybrid LRFM âˆ’ spline LRFM | AUC | âˆ’0.0043 | [âˆ’0.0091, âˆ’0.0001] | 0.2626 | Inconclusive |
| Dunnhumby | E3 hybrid LRFM âˆ’ spline LRFM | Spend RÂ² | âˆ’0.0045 | [âˆ’0.0084, âˆ’0.0005] | 0.1915 | Inconclusive |
| Dunnhumby | E3 hybrid LRFM âˆ’ spline LRFM | Invoice RÂ² | +0.0015 | [âˆ’0.0017, +0.0049] | 1.0000 | Inconclusive |
| Dunnhumby | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | AUC | âˆ’0.0075 | [âˆ’0.0171, +0.0004] | 0.2731 | Inconclusive |
| Dunnhumby | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Spend RÂ² | âˆ’0.0044 | [âˆ’0.0071, âˆ’0.0017] | 0.0158 | Supported, negative |
| Dunnhumby | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Invoice RÂ² | âˆ’0.0026 | [âˆ’0.0052, âˆ’0.0000] | 0.2626 | Inconclusive |
| Dunnhumby | E5 spline LRFM âˆ’ spline RFM | AUC | +0.0001 | [âˆ’0.0015, +0.0023] | 1.0000 | Inconclusive |
| Dunnhumby | E5 spline LRFM âˆ’ spline RFM | Spend RÂ² | âˆ’0.0001 | [âˆ’0.0022, +0.0017] | 1.0000 | Inconclusive |
| Dunnhumby | E5 spline LRFM âˆ’ spline RFM | Invoice RÂ² | +0.0020 | [+0.0004, +0.0036] | 0.1123 | Inconclusive |
| Online Retail II | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | AUC | +0.0089 | [+0.0061, +0.0118] | 0.0003 | Supported |
| Online Retail II | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Spend RÂ² | +0.0171 | [+0.0135, +0.0209] | 0.0003 | Supported |
| Online Retail II | E1 fuzzy LRFM-FCA âˆ’ crisp LRFM-FCA | Invoice RÂ² | +0.0321 | [+0.0267, +0.0375] | 0.0003 | Supported |
| Online Retail II | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | AUC | +0.0041 | [+0.0017, +0.0063] | 0.0036 | Supported |
| Online Retail II | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Spend RÂ² | +0.0068 | [+0.0018, +0.0122] | 0.0150 | Supported |
| Online Retail II | E2 fuzzy LRFM-FCA âˆ’ spline LRFM | Invoice RÂ² | âˆ’0.0254 | [âˆ’0.0404, âˆ’0.0094] | 0.0082 | Supported, negative |
| Online Retail II | E3 hybrid LRFM âˆ’ spline LRFM | AUC | +0.0040 | [+0.0018, +0.0061] | 0.0022 | Supported |
| Online Retail II | E3 hybrid LRFM âˆ’ spline LRFM | Spend RÂ² | +0.0102 | [+0.0065, +0.0141] | 0.0003 | Supported |
| Online Retail II | E3 hybrid LRFM âˆ’ spline LRFM | Invoice RÂ² | +0.0080 | [+0.0027, +0.0141] | 0.0069 | Supported |
| Online Retail II | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | AUC | +0.0056 | [+0.0025, +0.0083] | 0.0003 | Supported |
| Online Retail II | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Spend RÂ² | +0.0080 | [+0.0056, +0.0104] | 0.0003 | Supported |
| Online Retail II | E4 fuzzy LRFM-FCA âˆ’ fuzzy RFM-FCA | Invoice RÂ² | +0.0111 | [+0.0081, +0.0141] | 0.0003 | Supported |
| Online Retail II | E5 spline LRFM âˆ’ spline RFM | AUC | +0.0023 | [+0.0003, +0.0044] | 0.0220 | Supported |
| Online Retail II | E5 spline LRFM âˆ’ spline RFM | Spend RÂ² | +0.0033 | [+0.0016, +0.0050] | 0.0014 | Supported |
| Online Retail II | E5 spline LRFM âˆ’ spline RFM | Invoice RÂ² | +0.0053 | [+0.0030, +0.0075] | 0.0003 | Supported |

## Main findings

1. **Fuzzy versus crisp FCA (E1):** supported on all three metrics for Online
   Retail II and Dunnhumby, and on both RÂ² outcomes but not AUC for CDNOW.
   This is the most consistent representation-level result.
2. **Fuzzy FCA versus the same-information spline (E2):** mixed and
   dataset-dependent. It is positive for Online Retail II AUC and Spend RÂ² but
   negative for Invoice RÂ²; it is negative on all three Dunnhumby metrics and
   negative on CDNOW RÂ² outcomes.
3. **Adding duration to fuzzy FCA (E4):** supported and positive on all three
   Online Retail II metrics, but inconclusive on CDNOW and mixed on Dunnhumby:
   Spend RÂ² is significantly negative while AUC and Invoice RÂ² are
   inconclusive.
4. **Adding duration to the spline baseline (E5):** supported and positive on
   all three Online Retail II metrics, but inconclusive on Dunnhumby and CDNOW.
5. **Hybrid LRFM versus spline LRFM (E3):** consistently supported and positive
   for Online Retail II; supported for CDNOW AUC and Spend RÂ² only; and
   inconclusive for Dunnhumby after Holm correction.

## Contribution decision

The defensible contribution is **not** a universal claim that purchase duration
improves prediction, nor a universal claim that FCA dominates a conventional
spline model. The evidence supports a narrower contribution:

> Adding purchase duration can provide incremental predictive information in
> Online Retail II, while its value is dataset-dependent; within the same
> four-dimensional representation, fuzzy FCA is generally better than crisp FCA,
> but it is not uniformly better than a strong spline baseline.

This supports framing the paper around the value and limits of an LRFM
extension, with fuzzy-versus-crisp FCA as the most reproducible method result.
The spline comparisons should be presented as an important boundary condition:
FCA representation choice is not a substitute for testing against a strong
non-FCA baseline. All conclusions remain exploratory and do not establish
equivalence for inconclusive comparisons.

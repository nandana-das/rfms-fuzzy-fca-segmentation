# CDNOW confirmation — results

**Provenance.**
- **Plan:** `docs/CDNOW_CONFIRMATION_PLAN.md`, committed (`04ce2cd`) before any CDNOW analysis code existed or any holdout outcome was computed; only covariate distributions had been examined.
- **Addendum A1** (`48b354a`): an implementation fix after the first run stopped on a guard. No model output had been produced; only the first origin's repurchase rate had been printed. A1 was committed before the re-run.
- **This result:** the single run after A1, with no tuning. Source: `results/cdnow_confirmation/` (`scripts/cdnow_confirmation.py`).
- **Data:** CDNOW master (Fader & Hardie, 2001): music retail, a 1997 Q1 acquisition cohort, 23,500–23,502 customers per origin after cleaning, repurchase 14.1–17.9%. This dataset was never used before in the project.
- **Status:** an analysis plan, not a formal preregistration.

**Method notes.**
- Frozen v1/v2 settings, plus the planned tie-merging extension (plan §4 and A1).
- On CDNOW, F has 4 fuzzy levels (cutpoints 1, 2, 3 purchase days) because more than half the customers bought on only one day. R and M have 5 levels.
- The extension is verified to change nothing on Dunnhumby and Online Retail II.
- Retained concepts (mean over folds): crisp 50.1; fuzzy 58.9 (from 311.6 mined); no empty-core concepts.

## Mean metrics (3 origins, descriptive; rounded to 4 decimals)

| Representation | ROC AUC | Spend R² | Invoice R² (purchase days) |
|---|---|---|---|
| Raw RFM (standardized) | 0.8021 | 0.2501 | 0.2934 |
| Log RFM | 0.8057 | 0.2557 | 0.2774 |
| Spline on log RFM | 0.8020 | 0.2704 | 0.3073 |
| Crisp RFM-FCA (base-paper method) | 0.8005 | 0.2417 | 0.2475 |
| Fuzzy RFM-FCA (v1) | 0.8064 | 0.2614 | 0.2747 |
| Hybrid crisp FCA + log RFM | 0.8050 | 0.2724 | 0.3066 |
| Hybrid fuzzy FCA + log RFM (v2) | **0.8070** | **0.2753** | **0.3104** |

## Comparisons: exact estimates

Paired customer bootstrap (B = 2000) pooled over 3 origins; Holm correction over 21 tests. The smallest attainable Holm p-value is 0.0105, so p = 0.0105 means "at the procedure's floor". "Origins positive" is descriptive.

| ID | Comparison | Metric | Δ | 95% CI | p (Holm) | Origins positive |
|---|---|---|---|---|---|---|
| C1 | fuzzy − crisp RFM-FCA | AUC | +0.0060 | [+0.0044, +0.0075] | 0.0105 | 3/3 |
| C1 | fuzzy − crisp RFM-FCA | Spend R² | +0.0197 | [+0.0170, +0.0223] | 0.0105 | 3/3 |
| C1 | fuzzy − crisp RFM-FCA | Invoice R² | +0.0272 | [+0.0237, +0.0302] | 0.0105 | 3/3 |
| C2 | fuzzy − spline | AUC | +0.0044 | [+0.0030, +0.0058] | 0.0105 | 1/3 |
| C2 | fuzzy − spline | Spend R² | −0.0090 | [−0.0123, −0.0057] | 0.0105 | 0/3 |
| C2 | fuzzy − spline | Invoice R² | −0.0326 | [−0.0391, −0.0261] | 0.0105 | 0/3 |
| C3 | v2 hybrid − spline | AUC | +0.0051 | [+0.0035, +0.0065] | 0.0105 | 2/3 |
| C3 | v2 hybrid − spline | Spend R² | +0.0049 | [+0.0033, +0.0064] | 0.0105 | 3/3 |
| C3 | v2 hybrid − spline | Invoice R² | +0.0031 | [+0.0015, +0.0047] | 0.0105 | 3/3 |
| C4 | v2 hybrid − v1 fuzzy | AUC | +0.0006 | [−0.0003, +0.0015] | 0.1930 | 2/3 |
| C4 | v2 hybrid − v1 fuzzy | Spend R² | +0.0140 | [+0.0116, +0.0164] | 0.0105 | 3/3 |
| C4 | v2 hybrid − v1 fuzzy | Invoice R² | +0.0357 | [+0.0301, +0.0411] | 0.0105 | 3/3 |
| C5 | v2 hybrid − crisp hybrid | AUC | +0.0020 | [+0.0005, +0.0036] | 0.0300 | 3/3 |
| C5 | v2 hybrid − crisp hybrid | Spend R² | +0.0029 | [+0.0018, +0.0040] | 0.0105 | 3/3 |
| C5 | v2 hybrid − crisp hybrid | Invoice R² | +0.0038 | [+0.0025, +0.0050] | 0.0105 | 3/3 |
| C6 | v2 hybrid − log RFM | AUC | +0.0013 | [+0.0001, +0.0025] | 0.0720 | 3/3 |
| C6 | v2 hybrid − log RFM | Spend R² | +0.0197 | [+0.0172, +0.0221] | 0.0105 | 3/3 |
| C6 | v2 hybrid − log RFM | Invoice R² | +0.0329 | [+0.0290, +0.0371] | 0.0105 | 3/3 |
| C7 | v2 hybrid − crisp RFM-FCA | AUC | +0.0066 | [+0.0048, +0.0082] | 0.0105 | 3/3 |
| C7 | v2 hybrid − crisp RFM-FCA | Spend R² | +0.0336 | [+0.0298, +0.0374] | 0.0105 | 3/3 |
| C7 | v2 hybrid − crisp RFM-FCA | Invoice R² | +0.0628 | [+0.0558, +0.0698] | 0.0105 | 3/3 |

Per-origin estimates are in `results/cdnow_confirmation/per_origin_comparisons.csv`.

## Decision-rule outcomes (rules fixed in plan §6)

| Finding | Outcome on CDNOW |
|---|---|
| v1 H1: fuzzy > crisp RFM-FCA | **Replicates.** Significantly positive on all 3 metrics; positive at all 3 origins. |
| v1 H2: fuzzy vs spline | **Mixed.** AUC significantly higher (+0.0044), but Spend R² (−0.0090) and Invoice R² (−0.0326) significantly lower, the same top-band saturation shortfall seen on Online Retail II. |
| v2 beats spline | **Confirmed on all 3 metrics**, by small margins: +0.0051 AUC, +0.0049 Spend R², +0.0031 Invoice R². |
| v2 improves on v1 | **Confirmed on Spend R² and Invoice R²** (+0.0140, +0.0357). AUC not significantly different; no metric significantly negative. |
| v2 concepts add beyond linear log RFM | **Supported on Spend R² and Invoice R².** AUC not significant (p = 0.072). This concerns *linear* log RFM; value beyond nonlinear RFM is C3. |
| Fuzzification helps within the hybrid | **Supported on all 3 metrics** (+0.0020 AUC, +0.0029 Spend R², +0.0038 Invoice R²). |
| v2 vs base-paper method | Significantly better on all 3 metrics (Invoice R² +0.0628). |

## Interpretation

1. **The core v1 finding replicates on a third, independent dataset.** Fuzzifying the RFM formal context improves predictive adequacy over the crisp RFM-FCA of the base paper.
2. **The v1 limitation also replicates.** Without magnitude information, fuzzy RFM-FCA falls significantly short of a spline baseline on the regression targets, which is consistent with top-band saturation. v1 H2 remains not supported as a superiority claim.
3. **The exploratory v2 result is confirmed out of sample.** The hybrid (fuzzy concepts + log RFM) beats the spline baseline on all three metrics.
   - CDNOW was untouched when v2 was designed, and the plan was committed before any CDNOW analysis.
   - The margins are small (0.003–0.005), and the p-values sit at the procedure's floor with only 3 origins. "Confirmed" here means "the same direction and significance as on Online Retail II", not "large".
4. **Concepts and fuzzification.**
   - Within the hybrid, the fuzzy concepts add significantly over linear log RFM (regression targets) and over crisp concepts (all metrics).
   - The concepts still are not shown to add value beyond *any* nonlinear transform. Only C3 addresses that, and its margin is small.

## Limitations

- One additional dataset: music retail, 1997–98, a single acquisition cohort.
- Only three overlapping origins; the bootstrap reflects test-sample variability only.
- F has 4 levels via the planned extension.
- The first run failed and was re-run after the documented fix A1. The fix was determined by the failure mode, and no model results were seen before it.
- Segment stability on CDNOW was not studied (out of scope in the plan).
- This confirmation does not change the frozen v1 documents. It adds evidence that should be reported alongside them.

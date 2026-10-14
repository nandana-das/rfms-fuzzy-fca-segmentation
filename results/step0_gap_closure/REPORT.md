# Step 0 — Predictive ceiling audit and gap closure

This report reuses the verified outputs of `results/baseline_ladder_rolling_origin/`; it does not rerun the predictive pipeline.

## Audit result

All required Step 0 arms are present in the existing run. Raw RFM and log RFM use standardized three-feature inputs with LogisticRegressionCV/RidgeCV; crisp RFM-FCA and M0 use FCA-derived features with the same downstream learners; gradient boosting uses raw R/F/M with untuned histogram gradient boosting. The run uses train-fold-only preprocessing, customer-level stratified 5-fold CV within each rolling origin, 91-day holdouts, paired bootstrap comparisons, and Holm correction.

| Arm | Existing verified output | Features | Learner | Status |
|---|---|---|---|---|
| Raw RFM without FCA | `per_origin_metrics.csv` | standardized R, F, M | LogisticRegressionCV / RidgeCV | existing |
| Log RFM without FCA | `per_origin_metrics.csv` | standardized log1p(R), log1p(F), log1p(M) | LogisticRegressionCV / RidgeCV | existing |
| Crisp RFM-FCA | `per_origin_metrics.csv` | crisp closed-concept memberships | LogisticRegressionCV / RidgeCV | existing |
| M0 fuzzy RFM-FCA | `per_origin_metrics.csv` | fuzzy closed-concept memberships with Jaccard suppression | LogisticRegressionCV / RidgeCV | existing |
| Gradient boosting baseline | `per_origin_metrics.csv` | raw R, F, M | untuned histogram gradient boosting | existing |

No missing predictive comparison was identified; therefore no experiment was rerun. The newly computed artifacts are the pooled arm summary and gap-closure table below.

## Pooled performance and gap closure

The ratio is `(M0 − Crisp) / (Raw − Crisp)`, using pooled rolling-origin means and the metric's native direction (higher is better).

| Dataset | Metric | Raw RFM | Log RFM | Crisp RFM-FCA | M0 fuzzy RFM-FCA | GBM raw RFM | Numerator | Denominator | Gap closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dunnhumby | auc | 0.8822 | 0.8917 | 0.8689 | 0.8899 | 0.8809 | +0.0210 | +0.0133 | 1.5808 |
| Dunnhumby | spend_r2 | 0.4367 | 0.5195 | 0.5112 | 0.5382 | 0.5062 | +0.0270 | -0.0745 | -0.3617 |
| Dunnhumby | invoice_r2 | 0.5086 | 0.6066 | 0.6013 | 0.6227 | 0.6061 | +0.0213 | -0.0928 | -0.2301 |
| Online Retail II | auc | 0.7772 | 0.7809 | 0.7743 | 0.7799 | 0.7732 | +0.0056 | +0.0029 | undefined/unstable |
| Online Retail II | spend_r2 | 0.2105 | 0.2968 | 0.2975 | 0.3114 | 0.2941 | +0.0139 | -0.0870 | -0.1594 |
| Online Retail II | invoice_r2 | 0.3319 | 0.3702 | 0.3464 | 0.3737 | 0.3892 | +0.0273 | -0.0145 | -1.8787 |

## Interpretation

The ratio is not a percentage improvement. A value between 0 and 1 means M0 closes part of the raw-to-crisp gap; a value above 1 means M0 exceeds the raw baseline relative to crisp; a negative value means M0 moves opposite to the raw-to-crisp direction. This report flags absolute denominators below 0.005 as undefined/unstable. When raw and crisp are nearly tied, the ratio should not be interpreted as a ceiling estimate.

Across the evaluated datasets, learners, outcomes, and rolling-origin protocol, the results support a limited predictive-ceiling statement only. Log transformation recovers much of the performance that raw standardized RFM leaves on the table, while M0 is generally close to log-RFM and the nonlinear spline reference. Thus crisp RFM-FCA appears near the ceiling for some RFM-only comparisons, but the evidence does not establish a universal ceiling: the ratio is metric- and dataset-dependent, and the denominator is unstable for AUC because raw and crisp are close or reversed.

## Limitations and provenance

- The original paper reports clustering validity, not predictive metrics; these are comparisons against the repository's crisp reconstruction.
- Pooled intervals are approximate because origins overlap; bootstrap intervals capture test-sample variability, not refitting variability.
- The gap closure calculation is descriptive; it does not add new hypothesis tests or multiple-testing corrections.
- No claim is made beyond the two evaluated datasets, three outcomes, learners, and locked protocol.

## Commands and outputs

- Audit/read: `rg -n -i "raw|log[- ]?rfm|crisp|m0|gradient|boost|rolling|91[- ]day|holdout|gap closure" scripts results docs README.md`.
- Existing experiment command recorded by the project: `python scripts/baseline_ladder_rolling_origin.py` (already completed; not rerun for Step 0).
- New reporting command: `python scripts/step0_gap_closure.py`.
- Generated: `results/step0_gap_closure/pooled_arm_means.csv`, `results/step0_gap_closure/gap_closure.csv`, and this report.

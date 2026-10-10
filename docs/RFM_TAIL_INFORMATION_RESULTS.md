# Exploratory RFM tail-information ablation results

This analysis is exploratory. It does not alter or replace the frozen baseline
evidence, and the tail-augmented arm is a hybrid representation rather than a
pure FCA method.

## Protocol and validation

- Arms: frozen `fuzzy_rfm_fca`, `tail_augmented_fuzzy_rfm_fca`, and frozen
  `spline_log_rfm`.
- The augmented arm appends standardized `log1p(R, F, M)` features to the
  exact fuzzy RFM-FCA concept features.
- Scaling was fit separately within each training fold and applied unchanged
  to its test fold.
- All arms used the same stratified five-fold customer splits and downstream
  models.
- Datasets: Dunnhumby and Online Retail II.
- Holdout: 91 days, with the same outcomes and metric transformations as the
  frozen baseline.
- Bootstrap: 2,000 paired customer resamples per origin, with origin-level
  differences and Holm correction over nine pooled tests per dataset
  (three comparisons × three metrics).
- The Online Retail II `2010-12-09 (primary, 365d)` cohort is retained in the
  per-origin output but excluded from pooled rolling-origin inference. The
  pooled Online Retail II results use the five rolling origins.

Validation checks passed before and during the run:

- syntax checks;
- five focused tests;
- frozen fuzzy-arm feature/prediction reproduction guard;
- complete out-of-fold predictions for all arms;
- exact metric agreement between the frozen fuzzy and spline arms and
  `results/baseline_ladder_rolling_origin/per_origin_metrics.csv` on all ten
  cohorts;
- output isolation under `results/rfm_tail_information/`.

## Pooled paired comparisons

Positive differences favor the first arm. Intervals are 95% percentile
bootstrap intervals. “Supported” means Holm-adjusted `p < 0.05`; an
inconclusive result is not evidence of equivalence.

| Dataset | Comparison | Metric | Δ | 95% CI | Holm p | Decision |
|---|---|---:|---:|---:|---:|---|
| Dunnhumby | Tail-augmented fuzzy FCA − frozen fuzzy FCA | AUC | +0.0040 | [+0.0006, +0.0076] | 0.1320 | Inconclusive |
| Dunnhumby | Tail-augmented fuzzy FCA − frozen fuzzy FCA | Spend R² | +0.0038 | [+0.0003, +0.0072] | 0.1550 | Inconclusive |
| Dunnhumby | Tail-augmented fuzzy FCA − frozen fuzzy FCA | Invoice R² | +0.0055 | [+0.0032, +0.0078] | 0.0045 | Supported |
| Dunnhumby | Tail-augmented fuzzy FCA − spline log-RFM | AUC | −0.0011 | [−0.0043, +0.0021] | 0.5560 | Inconclusive |
| Dunnhumby | Tail-augmented fuzzy FCA − spline log-RFM | Spend R² | −0.0022 | [−0.0060, +0.0016] | 0.5560 | Inconclusive |
| Dunnhumby | Tail-augmented fuzzy FCA − spline log-RFM | Invoice R² | +0.0024 | [−0.0010, +0.0058] | 0.5560 | Inconclusive |
| Dunnhumby | Frozen fuzzy FCA − spline log-RFM | AUC | −0.0050 | [−0.0088, −0.0013] | 0.0480 | Supported, negative |
| Dunnhumby | Frozen fuzzy FCA − spline log-RFM | Spend R² | −0.0060 | [−0.0102, −0.0016] | 0.0630 | Inconclusive |
| Dunnhumby | Frozen fuzzy FCA − spline log-RFM | Invoice R² | −0.0030 | [−0.0071, +0.0009] | 0.5560 | Inconclusive |
| Online Retail II | Tail-augmented fuzzy FCA − frozen fuzzy FCA | AUC | +0.0038 | [+0.0024, +0.0053] | 0.0045 | Supported |
| Online Retail II | Tail-augmented fuzzy FCA − frozen fuzzy FCA | Spend R² | +0.0039 | [+0.0021, +0.0056] | 0.0045 | Supported |
| Online Retail II | Tail-augmented fuzzy FCA − frozen fuzzy FCA | Invoice R² | +0.0362 | [+0.0294, +0.0432] | 0.0045 | Supported |
| Online Retail II | Tail-augmented fuzzy FCA − spline log-RFM | AUC | +0.0032 | [+0.0016, +0.0048] | 0.0045 | Supported |
| Online Retail II | Tail-augmented fuzzy FCA − spline log-RFM | Spend R² | +0.0055 | [+0.0033, +0.0078] | 0.0045 | Supported |
| Online Retail II | Tail-augmented fuzzy FCA − spline log-RFM | Invoice R² | +0.0045 | [+0.0011, +0.0077] | 0.0060 | Supported |
| Online Retail II | Frozen fuzzy FCA − spline log-RFM | AUC | −0.0006 | [−0.0027, +0.0014] | 0.5780 | Inconclusive |
| Online Retail II | Frozen fuzzy FCA − spline log-RFM | Spend R² | +0.0017 | [−0.0013, +0.0048] | 0.5780 | Inconclusive |
| Online Retail II | Frozen fuzzy FCA − spline log-RFM | Invoice R² | −0.0318 | [−0.0404, −0.0231] | 0.0045 | Supported, negative |

## Interpretation

### Online Retail II

The tail-augmented arm improves on frozen fuzzy RFM-FCA on all three metrics
after Holm correction. It also improves on the spline baseline on all three
metrics. This is consistent with continuous RFM magnitude features adding
information that the saturated FCA memberships do not retain in this dataset.
The result is about the hybrid augmentation: it does not show that FCA itself
is superior, because the augmented arm has additional continuous inputs.

The frozen fuzzy arm is not distinguishable from the spline on AUC or Spend
R², and is significantly worse on Invoice R². The augmented arm therefore
closes and reverses that particular comparison, but the gain cannot be
attributed to the FCA representation alone.

### Dunnhumby

The augmented arm is significantly better than frozen fuzzy RFM-FCA only on
Invoice R². AUC and Spend R² gains are positive but inconclusive. The
augmented arm is not significantly different from the spline on any metric.
The frozen fuzzy arm is significantly below the spline on AUC after Holm
correction, with the other two comparisons inconclusive.

This is weaker and less consistent evidence for a tail-information benefit
than Online Retail II. It does not support a universal claim that continuous
RFM magnitudes improve fuzzy FCA prediction.

## Conclusion

The ablation supports a dataset-dependent conclusion: continuous
`log1p(R, F, M)` augmentation can materially improve the fuzzy RFM-FCA
representation, especially for Online Retail II, but the effect is much
smaller and only partly supported on Dunnhumby. The appropriate contribution
is therefore a limitation and diagnostic result about information lost by
saturating FCA memberships, not a claim that FCA alone outperforms a strong
continuous baseline.

The analysis remains exploratory, uses the same previously analyzed datasets,
and does not establish equivalence for nonsignificant comparisons. Bootstrap
intervals capture customer test-sample variability and not refitting
variability; rolling-origin populations overlap. The exact bootstrap draws are
not persisted as separate files.

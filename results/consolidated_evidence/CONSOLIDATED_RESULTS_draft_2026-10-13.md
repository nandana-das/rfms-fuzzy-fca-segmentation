# Consolidated RFMS-Fuzzy FCA Experimental Evidence

## Executive summary

This report consolidates the completed Step 0 predictive-ceiling analysis, the temporal-attribute ablation, and the aggregation-operator experiments. A single corrected temporal-ablation rerun was subsequently executed after the R/F/M parity and statistical-generation audit; historical outputs remain preserved and unchanged.

The evidence is conditional on the two evaluated retail datasets, the evaluated outcomes and learners, and the locked rolling-origin protocol. It does not establish a universal predictive ceiling or a universal benefit from temporal attributes or alternative t-norms.

Main findings:

- Step 0 shows that log-RFM and nonlinear RFM representations recover much of the performance left by standardized raw RFM. Gap-closure ratios are metric-dependent and unstable when Raw and Crisp are nearly tied.
- Temporal attributes are dataset-dependent: T significantly improves all three Online Retail II outcomes but significantly reduces Dunnhumby Spend R². Adding L after T yields only a small significant Online Retail II Invoice R² increment.
- Łukasiewicz aggregation does not improve uniformly over M0: it lowers Dunnhumby AUC and raises Online Retail II AUC, with no significant R² improvement versus M0.

## 1. Comparison of completed experiments

| Experiment | Arms / comparison | Main result | Evidence status |
|---|---|---|---|
| Step 0 predictive ceiling | Raw RFM, Log RFM, Crisp RFM-FCA, M0, raw-RFM GBM | M0 is close to log/nonlinear RFM on several outcomes; the raw-to-crisp denominator is often reversed or small | Existing verified results; descriptive gap closure |
| Temporal attributes | M0, Fuzzy RFM+T, Fuzzy RFM+T+L | T helps all Online Retail II metrics but harms Dunnhumby Spend R²; L adds only Online Retail II Invoice R² after T | Completed ablation with paired bootstrap and Holm correction |
| Aggregation | Gödel-min M0, product M2, Łukasiewicz M4 | Łukasiewicz AUC is lower than M0 on Dunnhumby and higher on Online Retail II; no significant R² gains versus M0 | Completed narrow paired run; existing concepts reused |

## 2. Step 0 — Predictive ceiling and gap closure

The evaluated protocol used rolling origins with 91-day holdouts, customer-level stratified five-fold CV, train-fold-only preprocessing, paired customer bootstrap, and Holm correction in the underlying baseline ladder. The original paper reported clustering-validity results rather than predictive metrics; these are therefore comparisons against the repository’s crisp RFM-FCA reconstruction.

### Pooled performance and gap closure

The source definition is:

\[
G = \frac{M0 - \mathrm{Crisp\ RFM\text{-}FCA}}
{\mathrm{Raw\ RFM} - \mathrm{Crisp\ RFM\text{-}FCA}}.
\]

Higher metric values are treated as better. The ratio is not a percentage improvement.

| Dataset | Metric | Raw RFM | Log RFM | Crisp RFM-FCA | M0 | Raw-RFM GBM | Numerator | Denominator | Gap closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dunnhumby | AUC | 0.8822 | 0.8917 | 0.8689 | 0.8899 | 0.8809 | +0.0210 | +0.0133 | 1.5808 |
| Dunnhumby | Spend R² | 0.4367 | 0.5195 | 0.5112 | 0.5382 | 0.5062 | +0.0270 | −0.0745 | −0.3617 |
| Dunnhumby | Invoice R² | 0.5086 | 0.6066 | 0.6013 | 0.6227 | 0.6061 | +0.0213 | −0.0928 | −0.2301 |
| Online Retail II | AUC | 0.7772 | 0.7809 | 0.7743 | 0.7799 | 0.7732 | +0.0056 | +0.0029 | undefined/unstable |
| Online Retail II | Spend R² | 0.2105 | 0.2968 | 0.2975 | 0.3114 | 0.2941 | +0.0139 | −0.0870 | −0.1594 |
| Online Retail II | Invoice R² | 0.3319 | 0.3702 | 0.3464 | 0.3737 | 0.3892 | +0.0273 | −0.0145 | −1.8787 |

The Online Retail II AUC ratio is explicitly unstable because `Raw − Crisp = 0.0029`, below the source report’s 0.005 stability threshold. Negative ratios occur when Raw RFM underperforms Crisp RFM-FCA, making the denominator negative; they do not mean that M0 has a negative percentage improvement. They indicate that the specified numerator and denominator point in opposite directions.

The defensible conclusion is limited: within these datasets, outcomes, learners, and temporal protocol, crisp RFM-FCA is already close to some RFM-only predictive alternatives, while log transformation and nonlinear RFM encodings account for substantial performance differences. This is not a universal predictive ceiling.

## 3. Temporal RFM-FCA ablation

Definitions were fixed before implementation:

- `R`: days since the last observed purchase at the origin.
- `F`: number of unique invoices/baskets observed through the origin.
- `M`: total observed monetary value through the origin.
- `T`: mean interval between consecutive distinct observed purchase dates through the origin.
- `L`: days from the first observed purchase date to the origin.
- For customers with fewer than two distinct purchase dates, T was missing, imputed with the relevant training-fold median only, and accompanied by `repeat_history` (1 for at least two observed purchase dates, otherwise 0).

All temporal features, imputation, memberships, concept mining, suppression, and learners were constructed inside training folds. Future outcomes used only the subsequent 91-day holdout.

### Reported paired differences

| Dataset | Comparison | Metric | Delta | Holm p | Interpretation |
|---|---|---:|---:|---:|---|
| Online Retail II | RFM+T − M0 | AUC | +0.0076 | 0.0045 | Significant improvement |
| Online Retail II | RFM+T − M0 | Spend R² | +0.0110 | 0.0045 | Significant improvement |
| Online Retail II | RFM+T − M0 | Invoice R² | +0.0348 | 0.0045 | Significant improvement |
| Dunnhumby | RFM+T − M0 | Spend R² | −0.0035 | 0.0045 | Significant reduction |
| Dunnhumby | RFM+T − M0 | AUC and Invoice R² | Not significant | — | No supported change |
| Online Retail II | RFM+T+L − RFM+T | Invoice R² | +0.0028 | 0.0390 | Small significant improvement |
| Online Retail II | RFM+T+L − RFM+T | AUC and Spend R² | Not significant | — | No supported change |
| Dunnhumby | RFM+T+L − RFM+T | All metrics | Not significant improvement | — | No supported improvement |

The effect of T is therefore dataset-dependent. L does not provide a general incremental improvement after T.

### Corrected rerun and historical comparison

The original temporal outputs were archived unchanged at `results/temporal_rfm_fca_ablation_historical_pre_fix/` before the corrected run. The corrected source was `scripts/temporal_rfm_fca_ablation.py` at SHA-256 `C404A840C9B3817694D6C587FE1063198C716A3C506A652898424706C82197E1`; it removes the duplicate temporal recency inversion and aligns pooled paired deltas explicitly by origin. The corrected run used the same three arms, rolling origins, 91-day holdouts, customer-level folds, learners, outcomes, and bootstrap/Holm protocol.

The corrected run reproduced the historical per-origin AUC, Spend R², and Invoice R² values to floating-point precision (maximum absolute difference `2.22e-16`). Its pooled comparisons were unchanged: Dunnhumby RFM+T minus M0 was AUC `−0.002949` (Holm `p=0.312`), Spend R² `−0.003499` (`p=0.0045`), and Invoice R² `+0.000051` (`p=1.000`); Online Retail II changes were AUC `+0.007615`, Spend R² `+0.011010`, and Invoice R² `+0.034754`, all Holm `p=0.0045`. The T+L-minus-T results were likewise unchanged: Dunnhumby AUC `−0.000348` (`p=1.000`), Spend R² `−0.005682` (`p=0.0045`), Invoice R² `−0.004001` (`p=0.0045`); Online Retail II AUC `+0.000672` (`p=0.400`), Spend R² `+0.000766` (`p=0.400`), and Invoice R² `+0.002816` (`p=0.039`). Independently recomputed pooled deltas and both nine-test Holm families had no mismatches.

The corrected rerun therefore confirms the numerical temporal findings under the verified implementation, while the historical CSVs remain provenance-limited records of the earlier run. The full verification record is `results/temporal_rfm_fca_ablation/CORRECTED_RUN_PROVENANCE.md`.

### Matched-complexity diagnostic

The Dunnhumby matched-complexity follow-up retained exactly the M0 feature count within each training fold by truncating the already mined and Jaccard-suppressed RFM+T concepts to the first `K` concepts in their deterministic post-suppression order. This was an exploratory, order-dependent truncation diagnostic: it matched feature counts but also changed concept composition. It therefore cannot isolate a pure representation-size effect or establish that feature count alone caused the Spend R² reduction.

The diagnostic reproduced the cached M0 and RFM+T metrics with maximum absolute difference `0.0`. Its matched RFM+T pooled Spend R² was `0.536441`, compared with `0.538188` for M0 and `0.534689` for unpruned RFM+T. The matched arm improved over unpruned RFM+T by `+0.001752` but remained `−0.001747` below M0. This supports a possible representation-size contribution but is not causal evidence.

The historical temporal-ablation CSV (`results/temporal_rfm_fca_ablation_historical_pre_fix/`) reports zero `t_missing_train` and `t_missing_test` values in every fold. Those CSVs were written by an earlier revision of the script (CSV timestamps precede the script's last edit in that session); the corrected rerun reports nonzero fold-local missing-T counts (train 6–1270, test 0–329 per fold), consistent with the matched-complexity run. The zero values are therefore an artifact of the earlier revision's bookkeeping, not of the data. Metrics are unaffected: the corrected rerun reproduced the historical per-origin metrics to floating-point precision.

### Stability-ranked matched-complexity follow-up

A further Dunnhumby-only diagnostic estimated conditional suppression stability from 50 training-only bootstrap resamples per fold. The standard RFM+T miner and suppressor were run once per fold; the fold-fitted memberships and mined concept set were then held fixed while the existing suppressor was reapplied to resampled training extents. Exact canonical intent strings identified concepts across resamples. Concepts were ranked by stability descending, support descending, intent size ascending, and lexicographic intent, with `K` equal to the same-fold M0 feature count.

This remains exploratory. It matches feature counts but changes concept composition, so it does not isolate a pure representation-size effect. Moreover, every retained concept had stability `1.0` in all 20 folds; stability therefore did not discriminate among concepts, and the effective ranking fell back to support and deterministic tie-breakers. The stability measure is conditional on the base fold's mined concept set, not a full resampling analysis of concept discovery.

| Dunnhumby arm | AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| M0 | 0.889949 | 0.538188 | 0.622664 |
| Unpruned RFM+T | 0.886999 | 0.534689 | 0.622716 |
| Previous first-K matched | 0.891938 | 0.536441 | 0.622325 |
| Stability-ranked matched | 0.891473 | 0.536799 | 0.622108 |

Stability-ranked matched minus M0 was AUC `+0.001525`, Spend R² `−0.001389`, and Invoice R² `−0.000557`. Relative to unpruned RFM+T, the changes were AUC `+0.004474`, Spend R² `+0.002110`, and Invoice R² `−0.000608`. No paired uncertainty intervals or Holm-adjusted tests were available because customer-level predictions for the cached reference arms were not retained.

## 4. Aggregation operators

The aggregation experiment holds the threshold-scaled M0 context, mined concepts, support threshold, Jaccard suppression, memberships, folds, learners, and outcomes fixed. It does not perform continuous concept discovery.

For a concept with base memberships \(\mu_b\), the tested aggregations are:

\[
\mu_C^{\mathrm{G\ddot{o}del}} = \min_b \mu_b,
\]

\[
\mu_C^{\mathrm{Product}} = \prod_b \mu_b,
\]

\[
\mu_C^{\mathrm{Lukasiewicz}} = \max\left(0, \sum_b \mu_b - (|I_C|-1)\right).
\]

### Reported paired differences

| Dataset | Comparison | Metric | Delta | 95% interval | Holm p | Positive origins |
|---|---|---:|---:|---:|---:|---:|
| Dunnhumby | Łukasiewicz − M0 | AUC | −0.0035 | [−0.0061, −0.0009] | 0.0420 | 1/4 |
| Dunnhumby | Łukasiewicz − M0 | Spend R² | +0.0010 | [−0.0006, +0.0026] | 0.5640 | 4/4 |
| Dunnhumby | Łukasiewicz − M0 | Invoice R² | −0.0001 | [−0.0018, +0.0017] | 0.9060 | 2/4 |
| Online Retail II | Łukasiewicz − M0 | AUC | +0.0014 | [+0.0006, +0.0023] | 0.0060 | 3/5 |
| Online Retail II | Łukasiewicz − M0 | Spend R² | −0.0005 | [−0.0019, +0.0008] | 0.5380 | 2/5 |
| Online Retail II | Łukasiewicz − M0 | Invoice R² | +0.0014 | [+0.0000, +0.0028] | 0.1840 | 4/5 |
| Online Retail II | Łukasiewicz − Product | AUC | −0.0022 | [−0.0032, −0.0013] | 0.0045 | 1/5 |
| Online Retail II | Łukasiewicz − Product | Spend R² | −0.0012 | [−0.0020, −0.0004] | 0.0200 | 0/5 |
| Online Retail II | Łukasiewicz − Product | Invoice R² | −0.0006 | [−0.0016, +0.0004] | 0.5380 | 1/5 |

No significant R² effects versus M0 were reported for Łukasiewicz. The Online Retail II AUC result favors Łukasiewicz over M0 but not over product; this is not a uniform aggregation improvement.

## 5. Validation and provenance

### Integrity checks

- Temporal ablation M0 reproduction: maximum absolute difference from the verified M0 results was `0.0`.
- Łukasiewicz narrow run M0 and product parity: maximum absolute difference from cached product/M0 metrics was `0.0`.
- Łukasiewicz unit tests: four tests passed, covering single-attribute identity, zero-membership clipping, boundary values, and invalid operators.
- The single corrected temporal-ablation rerun was the only new experiment run for this verification; no exploratory arm or unrelated experiment was run.

### Source artifacts

- Step 0: `results/step0_gap_closure/REPORT.md`, `gap_closure.csv`, `pooled_arm_means.csv`.
- Temporal ablation: `results/temporal_rfm_fca_ablation/REPORT.md`, `per_origin_metrics.csv`, `paired_comparisons.csv`, `run_parameters.json`.
- Corrected temporal provenance: `results/temporal_rfm_fca_ablation/CORRECTED_RUN_PROVENANCE.md`; preserved historical outputs: `results/temporal_rfm_fca_ablation_historical_pre_fix/`.
- Dunnhumby matched-complexity diagnostic: `results/dunnhumby_matched_complexity/REPORT.md`, `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.
- Dunnhumby stability-ranked matched-complexity diagnostic: `results/dunnhumby_stability_matched_complexity/REPORT.md`, `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `concept_stability_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.
- Existing product-t-norm evidence: `results/fuzzy_tnorm_tail_ablation/MULTI_ORIGIN_REPORT.md`, `multi_origin_metrics.csv`, `run_manifest.json`, `m0_parity_verification.csv`.
- Łukasiewicz ablation: `results/lukasiewicz_tnorm_ablation/REPORT.md`, `per_origin_metrics.csv`, `paired_comparisons.csv`, `concept_counts_per_fold.csv`, `run_parameters.json`.
- Relevant implementation: `scripts/baseline_ladder_rolling_origin.py`, `scripts/ablation_ladder_rolling_origin.py`, `scripts/temporal_rfm_fca_ablation.py`, and `scripts/lukasiewicz_tnorm_ablation.py`.

### Commands recorded in source reports

The experiments were previously run with:

```text
python scripts\baseline_ladder_rolling_origin.py
python scripts\step0_gap_closure.py
python scripts\temporal_rfm_fca_ablation.py
python scripts\lukasiewicz_tnorm_ablation.py
pytest -q tests\test_lukasiewicz_tnorm.py
pytest -q tests\test_temporal_rfm_parity.py tests\test_temporal_statistics.py
```

The corrected temporal command and regression-test command were executed for the verification rerun; the other listed commands are provenance from the existing evidence.

## 6. Unresolved issues and limitations

- The gap-closure ratio is descriptive, not a new inferential test; small or sign-reversed denominators make it unstable or difficult to interpret.
- Rolling origins overlap in customers and history. Existing bootstrap intervals characterize test-sample variability under the observed windows and do not quantify all refitting or temporal-dependence uncertainty.
- Temporal effects are dataset-dependent and cannot be generalized beyond the evaluated datasets, origins, learners, and outcomes.
- The t-norm experiment changes aggregation over existing threshold-mined concepts. It does not test a continuous concept lattice or continuous concept discovery.
- The original product-t-norm report explicitly cautions that its stored multi-origin results should not be treated as confirmatory Holm-controlled evidence; the paired Holm comparisons reported here come from the later narrow M0/product/Łukasiewicz run used to retain paired prediction vectors.
- No additional datasets, attributes, concept-selection methods, learned memberships, alternative closure algorithms, or manuscript changes were introduced.

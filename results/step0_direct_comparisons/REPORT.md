# Step 0 — direct M0 comparisons (standalone re-analysis)

Standalone re-analysis of the Step 0 comparisons. It does not replace the original Step 0 report: `results/step0_gap_closure/` is preserved unchanged as the historical record. This analysis does not use the gap-closure ratio `(M0 − Crisp) / (Raw − Crisp)` as a ceiling measure, because its denominator uses standardized raw RFM, a known misspecified baseline that scores below crisp RFM-FCA in 4 of 6 cells, so the denominator is small or of the wrong sign and the ratio is unstable or misleading. This report shows direct paired differences instead.

## Source and verification

All numbers come from the frozen baseline ladder (`results/baseline_ladder_rolling_origin/`, `scripts/baseline_ladder_rolling_origin.py`). Within each origin every arm used the same customer folds (StratifiedKFold, seed 0), learner seed, outcomes, 91-day holdout, and bootstrap indices, so differences are paired. No model was refit for this report.

- Pooled deltas recomputed from `per_origin_metrics.csv`: 0 mismatches.
- Holm adjustment recomputed over the ladder's family ({'Dunnhumby': 27, 'Online Retail II': 27} pooled tests): 0 mismatches.

## Direct comparisons (pooled 91-day origins)

Delta = M0 minus reference. 95% paired-bootstrap interval (B=2000); Holm p from the ladder's 27-test family per dataset; verdict at Holm p < 0.05.

| dataset | reference | metric | M0 | reference | delta | CI low | CI high | p Holm | origins M0 ahead | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Dunnhumby | Crisp RFM-FCA | auc | 0.8899 | 0.8689 | +0.0210 | +0.0152 | +0.0271 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | Crisp RFM-FCA | spend_r2 | 0.5382 | 0.5112 | +0.0270 | +0.0217 | +0.0324 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | Crisp RFM-FCA | invoice_r2 | 0.6227 | 0.6013 | +0.0213 | +0.0174 | +0.0255 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | Log-RFM | auc | 0.8899 | 0.8917 | -0.0018 | -0.0073 | +0.0042 | 1.0000 | 2/4 | not significant |
| Dunnhumby | Log-RFM | spend_r2 | 0.5382 | 0.5195 | +0.0187 | +0.0110 | +0.0265 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | Log-RFM | invoice_r2 | 0.6227 | 0.6066 | +0.0161 | +0.0100 | +0.0223 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | GBM (raw RFM, untuned) | auc | 0.8899 | 0.8809 | +0.0090 | +0.0026 | +0.0155 | 0.1050 | 4/4 | not significant |
| Dunnhumby | GBM (raw RFM, untuned) | spend_r2 | 0.5382 | 0.5062 | +0.0320 | +0.0233 | +0.0418 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | GBM (raw RFM, untuned) | invoice_r2 | 0.6227 | 0.6061 | +0.0166 | +0.0111 | +0.0222 | 0.0135 | 4/4 | M0 better |
| Dunnhumby | Spline log-RFM (context) | auc | 0.8899 | 0.8950 | -0.0050 | -0.0088 | -0.0013 | 0.0960 | 0/4 | not significant |
| Dunnhumby | Spline log-RFM (context) | spend_r2 | 0.5382 | 0.5442 | -0.0060 | -0.0102 | -0.0016 | 0.1260 | 1/4 | not significant |
| Dunnhumby | Spline log-RFM (context) | invoice_r2 | 0.6227 | 0.6257 | -0.0030 | -0.0071 | +0.0009 | 1.0000 | 1/4 | not significant |
| Online Retail II | Crisp RFM-FCA | auc | 0.7799 | 0.7743 | +0.0056 | +0.0036 | +0.0075 | 0.0135 | 5/5 | M0 better |
| Online Retail II | Crisp RFM-FCA | spend_r2 | 0.3114 | 0.2975 | +0.0139 | +0.0111 | +0.0165 | 0.0135 | 5/5 | M0 better |
| Online Retail II | Crisp RFM-FCA | invoice_r2 | 0.3737 | 0.3464 | +0.0273 | +0.0232 | +0.0316 | 0.0135 | 5/5 | M0 better |
| Online Retail II | Log-RFM | auc | 0.7799 | 0.7809 | -0.0010 | -0.0032 | +0.0011 | 1.0000 | 3/5 | not significant |
| Online Retail II | Log-RFM | spend_r2 | 0.3114 | 0.2968 | +0.0146 | +0.0103 | +0.0186 | 0.0135 | 5/5 | M0 better |
| Online Retail II | Log-RFM | invoice_r2 | 0.3737 | 0.3702 | +0.0035 | -0.0031 | +0.0102 | 1.0000 | 4/5 | not significant |
| Online Retail II | GBM (raw RFM, untuned) | auc | 0.7799 | 0.7732 | +0.0066 | +0.0037 | +0.0097 | 0.0135 | 4/5 | M0 better |
| Online Retail II | GBM (raw RFM, untuned) | spend_r2 | 0.3114 | 0.2941 | +0.0173 | +0.0128 | +0.0218 | 0.0135 | 5/5 | M0 better |
| Online Retail II | GBM (raw RFM, untuned) | invoice_r2 | 0.3737 | 0.3892 | -0.0155 | -0.0241 | -0.0069 | 0.0135 | 0/5 | M0 worse |
| Online Retail II | Spline log-RFM (context) | auc | 0.7799 | 0.7805 | -0.0006 | -0.0027 | +0.0014 | 1.0000 | 4/5 | not significant |
| Online Retail II | Spline log-RFM (context) | spend_r2 | 0.3114 | 0.3097 | +0.0017 | -0.0013 | +0.0048 | 1.0000 | 4/5 | not significant |
| Online Retail II | Spline log-RFM (context) | invoice_r2 | 0.3737 | 0.4054 | -0.0318 | -0.0404 | -0.0231 | 0.0135 | 0/5 | M0 worse |

## Summary by reference

- Dunnhumby, M0 vs Crisp RFM-FCA: auc M0 better; spend_r2 M0 better; invoice_r2 M0 better.
- Dunnhumby, M0 vs Log-RFM: auc not significant; spend_r2 M0 better; invoice_r2 M0 better.
- Dunnhumby, M0 vs GBM (raw RFM, untuned): auc not significant; spend_r2 M0 better; invoice_r2 M0 better.
- Dunnhumby, M0 vs Spline log-RFM (context): auc not significant; spend_r2 not significant; invoice_r2 not significant.
- Online Retail II, M0 vs Crisp RFM-FCA: auc M0 better; spend_r2 M0 better; invoice_r2 M0 better.
- Online Retail II, M0 vs Log-RFM: auc not significant; spend_r2 M0 better; invoice_r2 not significant.
- Online Retail II, M0 vs GBM (raw RFM, untuned): auc M0 better; spend_r2 M0 better; invoice_r2 M0 worse.
- Online Retail II, M0 vs Spline log-RFM (context): auc not significant; spend_r2 not significant; invoice_r2 M0 worse.

## Points to keep visible

- Pooled point estimates alone do not establish superiority; only the Holm verdicts above do. Where M0's point estimate exceeds a reference but the verdict is 'not significant', no advantage is claimed.
- Rows where M0 is significantly worse than a reference are reported, not omitted (for example GBM and spline on Online Retail II Invoice R²).
- The GBM is untuned (fixed parameters) on raw R/F/M; M0 > GBM therefore shows M0 beats this GBM, not tuned boosting.

## Limitations

- Pooled intervals average independent per-origin bootstraps. Origins overlap heavily in customers (Dunnhumby: 2,497 of 2,499 households appear in all four origins), so pooled intervals and p-values are approximate and likely too narrow. The ladder did not retain customer-level predictions, so a household-cluster bootstrap cannot be computed without refitting.
- The smallest attainable Holm p is 0.0135 (27 tests × 1/2000); many cells sit at this floor, so their p-values only bound significance from above.
- Single CV seed; bootstrap intervals reflect test-sample variability, not refitting or fold-assignment variability.
- Two retail datasets, three outcomes, linear learners for all non-GBM arms.

Command: `python scripts/step0_direct_comparisons.py`. Outputs: `direct_comparisons.csv`, this report.

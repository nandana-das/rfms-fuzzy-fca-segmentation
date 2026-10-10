# Four-method RFM-FCA figure audit

The three figures were generated from saved structured experiment outputs; no models or experiments were rerun.

## Sources

- Raw, Crisp, and Baseline Fuzzy RFM-FCA: `results/baseline_ladder_rolling_origin/per_origin_metrics.csv` for pooled 91-day rolling origins.
- Hybrid Fuzzy RFM-FCA for Dunnhumby and Online Retail II: `results/rfm_tail_information/per_origin_metrics.csv`, using the saved tail-augmented arm and pooled 91-day origins.
- All four CDNOW methods: `results/cdnow_confirmation/per_origin_metrics.csv`, averaged over the three chronological 91-day origins.

## Verification

- Plotted points verified: 36/36.
- Missing plotted values: 0.
- Hybrid delta checks confirmed: 9/9.
- Hybrid delta mismatches or missing comparisons: 0.
- CDNOW Raw RFM and Crisp RFM-FCA are present in the CDNOW confirmation per-origin result file and were not inferred from another experiment.
- No visual separation is treated as statistical significance.
- The hybrid is a fuzzy FCA plus continuous log-RFM representation, not a pure FCA-only method.

## Reported aggregate deltas

See `hybrid_delta_audit.csv` for computed values, saved paired-comparison values, confidence intervals, and Holm-adjusted p-values.

## Regeneration

```powershell
python analysis/plot_experimental_results.py
```

Primary figures are in this directory as PNG (300 DPI) and vector PDF files.

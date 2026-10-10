# Experimental result figures

`plot_experimental_results.py` generates publication-oriented PNG and vector PDF
figures from saved experiment CSV files. It does not run models or modify the
source result folders.

Run from the repository root on Windows:

```powershell
python analysis/plot_experimental_results.py
```

The default command generates the three primary unified figures in
`outputs/figures/unified_method_comparison/`. Each graph contains Raw RFM,
Crisp RFM-FCA, Baseline Fuzzy RFM-FCA, and Hybrid Fuzzy RFM-FCA when the saved
source supports that method-dataset combination. It never overwrites source
experiment results. Use `--no-overwrite` when an existing derived figure should
cause the run to stop instead.

The primary files are:

- `unified_method_comparison/auc_four_method_comparison.png` and `.pdf`
- `unified_method_comparison/spend_r2_four_method_comparison.png` and `.pdf`
- `unified_method_comparison/invoice_r2_four_method_comparison.png` and `.pdf`
- `unified_method_comparison/verification_table.csv`: one source-traceable row per plotted dataset/method/metric value.
- `unified_method_comparison/hybrid_delta_audit.csv`: computed and saved hybrid-minus-baseline deltas with saved uncertainty estimates.
- `unified_method_comparison/AUDIT_REPORT.md`: source, protocol, verification, and regeneration audit.

The loader validates required columns, numeric metrics, duplicate keys, supported
arms, matched arms before subtraction, and missing values. It fails instead of
silently dropping invalid rows, filling missing values, or mixing the Online
Retail II 365-day primary cohort into rolling-origin means.

Lines in Figure 1 connect categorical dataset summaries as visual guides; they
are not interpreted as a continuous dataset trend. Lines in Figure 3 connect
the actual chronological CDNOW origins.

Optional strict existence check:

```powershell
python analysis/plot_experimental_results.py --no-overwrite
```

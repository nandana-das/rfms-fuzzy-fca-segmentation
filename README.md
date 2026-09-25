# RFMS + Fuzzy FCA Customer Segmentation

This project applies RFMS feature construction and fuzzy formal concept analysis
to Olist Brazilian e-commerce data, with Online Retail II used for cross-domain
comparison. The scripts, datasets, regenerated analysis outputs, and figures are
organized in this repository.

## Validated v3 run

The supplied scripts have been run in sequence using the v3 design. Step 1 uses
the Shannon-entropy grid-search objective for the purchase-intensity index F*;
the selected Olist weights are `α=0.20`, `β=0.05`, and `γ=0.75`. This is the
entropy-maximizing combination within the searched grid and scoring procedure,
not a general claim that F* is a validated behavioral frequency measure. Literal
order count remains available separately. The resulting Olist F-score is sparse:
97.00% of customers receive score 1.

Uncapped FP-growth (`max_len=None`) found 1,369 Olist closed concepts; Kneedle
pruning retained 153, with 304 Hasse covering edges. The Retail II pipeline found
1,064 closed concepts and retained 115. The regenerated tables and figures are
under `results/` and `results/figures/`; processed feature tables are under
`data/processed/`.

Olist Satisfaction is the customer mean of available order-level review means.
For the 603 customers with no observed customer-level score, the pipeline uses
the observed median (5.0). Because 5.0 is also the observed mode, this imputation
increases the S=5.0 share by 0.27 percentage points. Interpret support for the
highest-S concepts with that limitation in mind; imputation does not establish
the missing customers' true satisfaction.

The benchmark reports conventional crisp clustering metrics for FCA's hard
assignments and for baseline algorithms. FCA's overlapping representation can be
penalized by these crisp metrics; interpret them as a comparison of hard
assignments, alongside the separate fuzzy diagnostics.

## Environment setup (Windows PowerShell)

Use Python 3.11. The dependency versions are pinned in `requirements.txt`. From
the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The virtual environment is intentionally excluded from Git; recreate it with
the commands above.

## Data and output locations

- Olist source CSVs: `data/olist_raw/`
- Online Retail II source CSVs: `data/raw_retail2/`
- Processed feature tables: `data/processed/`
- Analysis tables and serialized intermediates: `results/`
- Figures: `results/figures/`

All scripts resolve paths relative to the project directory and create output
directories when needed.

## Run order

Run these from the project root. Each step reads outputs from preceding steps;
Step 7 is a separate Retail II pipeline, and Step 8 combines both datasets.

```powershell
python scripts/step1_rfms_prep.py
python scripts/step2_fuzzy_fca.py
python scripts/step3_stability_pruning.py
python scripts/step4_alpha_cut_clusters.py
python scripts/step5_benchmark.py
python scripts/step7_cross_domain_retail2.py
python scripts/step8_kneedle_sensitivity.py
python scripts/step9_f1_group_recovery.py
python scripts/step10_overlap_profiles.py
python scripts/step11_fuzzy_validity_diagnostics.py
python scripts/step12_retail2_parity.py
python scripts/step13_figures.py
```

There is no Step 6 script in the supplied pipeline. Running these scripts
regenerates same-named outputs under `data/processed/` and `results/`.
`scripts/duplicates/step9_f1_group_recovery (1).py` is a duplicate; use the
canonical Step 9 command above.

## Main documents

- `docs/PROJECT_DOCUMENT.md` — project methodology and findings.
- `docs/RESULTS_SUMMARY.md` — results and diagnostics.
- `docs/methodology_rfms_fca_olist.md` — RFMS/FCA methodology notes.

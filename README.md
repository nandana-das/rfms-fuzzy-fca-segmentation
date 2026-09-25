# RFMS + Fuzzy FCA project

## Environment setup (Windows PowerShell)

Use Python 3.11 (the project environment created for this workspace uses Python
3.11.9). From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The scripts resolve paths relative to this project folder. Olist source CSVs
belong in `data/olist_raw/`; Online Retail II CSVs belong in
`data/raw_retail2/`. Processed intermediates are written to `data/processed/`,
and analysis outputs are written to `results/` (figures to `results/figures/`).
Output folders are created by the scripts when needed.

## Script dependencies and order

The Olist scripts use the outputs of preceding steps:

1. `python scripts/step1_rfms_prep.py`
2. `python scripts/step2_fuzzy_fca.py`
3. `python scripts/step3_stability_pruning.py`
4. `python scripts/step4_alpha_cut_clusters.py`
5. `python scripts/step5_benchmark.py`
6. `python scripts/step7_cross_domain_retail2.py` (separate Retail II pipeline)
7. `python scripts/step8_kneedle_sensitivity.py`
8. `python scripts/step9_f1_group_recovery.py`
9. `python scripts/step10_overlap_profiles.py`
10. `python scripts/step11_fuzzy_validity_diagnostics.py`
11. `python scripts/step12_retail2_parity.py`
12. `python scripts/step13_figures.py`

**Research gate:** this is an execution map, not authorization to run the
pipeline. The RFMS feature/scoring methodology is still awaiting resolution;
do not run Step 2 or later until that decision is approved. The scripts and
supplied reports also describe different F* selection methods, so resolve that
inconsistency before treating regenerated outputs as the audited results.

`scripts/duplicates/step9_f1_group_recovery (1).py` is a duplicate copy; use
the canonical Step 9 script listed above.

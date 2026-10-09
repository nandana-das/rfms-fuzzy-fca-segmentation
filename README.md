# Fuzzy RFM-FCA for Customer Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Base Paper](https://img.shields.io/badge/Base_Paper-Rungruang_et_al._(2024)-orange.svg)](docs/base_paper_methodology_audit.md)
[![Status: evidence frozen, under review](https://img.shields.io/badge/Status-Evidence_frozen_(under_review)-yellow.svg)](docs/RESULTS_SUMMARY.md)

This repository extends the crisp RFM-FCA customer segmentation of **Rungruang et al. (2024)** to a **fuzzy (non-binary) RFM formal context**, the extension the base paper proposed as future work. It evaluates the extension out of sample against the crisp original and strong non-FCA RFM baselines, and tests whether it gives more stable segments.

**Status (2026-10-08):** the methodology and evidence are frozen after an internal audit; defects, changes and superseded results are listed in [docs/AUDIT_ERRATA.md](docs/AUDIT_ERRATA.md). The pre-audit version of this README is kept in [docs/superseded_2026-10-08/README.md](docs/superseded_2026-10-08/README.md) and must not be cited.

Scope: **RFM only**, on two datasets:
- **Online Retail II**, the base paper's dataset;
- **Dunnhumby "The Complete Journey"**, used as a reproduction.

## Research question and findings

> To what extent does fuzzifying the RFM formal context change (a) the out-of-sample predictive adequacy and (b) the structural stability of FCA-based customer segmentation, compared with conventional crisp RFM-FCA and with strong non-FCA RFM baselines?

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: fuzzy RFM-FCA improves prediction over crisp RFM-FCA | **Supported** | Under the rolling-origin evaluation, the pooled improvement over crisp RFM-FCA is statistically significant on AUC, Spend R² and Invoice R² in both datasets (paired bootstrap, Holm p = 0.014). Descriptively, the difference is also positive at every individual origin (4/4 Dunnhumby, 5/5 Retail II); these origin counts are not a significance test |
| H2: fuzzy RFM-FCA achieves predictive performance comparable to strong nonlinear non-FCA RFM baselines | **Inconclusive** | Five of six comparisons with the spline-on-log-RFM baseline were not significantly different, but equivalence was not formally established (no equivalence margin was pre-specified), and Retail II Invoice R² was significantly worse by 0.032 |
| H3: fuzzy RFM-FCA improves segment stability over crisp RFM-FCA | **Not supported** | No significant difference under refitting; less stable under quarterly re-segmentation (−0.035 Dunnhumby, −0.020 Retail II). At matched concept count the temporal gap is no longer significant (+0.003 on both), which supports a "concept count and/or concept size" explanation, not concept count alone as the cause |

### Predictive headline: fuzzy vs the base-paper method

Means over rolling origins:

| | Base-paper method (crisp RFM-FCA) | Fuzzy RFM-FCA | Δ | Spline on log-RFM (no FCA) |
|---|---|---|---|---|
| Retail II ROC AUC | 0.774 | 0.780 | +0.006 | 0.781 |
| Retail II Spend R² | 0.298 | 0.311 | +0.014 | 0.310 |
| Retail II Invoice R² | 0.346 | 0.374 | +0.027 | 0.405 |
| Dunnhumby ROC AUC | 0.869 | 0.890 | +0.021 | 0.895 |
| Dunnhumby Spend R² | 0.511 | 0.538 | +0.027 | 0.544 |
| Dunnhumby Invoice R² | 0.601 | 0.623 | +0.021 | 0.626 |

What these results do and do not show:
- Fuzzy RFM-FCA consistently beats the base paper's crisp method.
- It does **not** beat a strong nonlinear RFM model.
- It is **not** more stable.
- Interpretability was not measured.

Full tables: [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md).

## Version 2 (post hoc, exploratory)

On branch `experiment/v2-hybrid-fuzzy-fca`, a hybrid representation (fuzzy concept memberships plus log R/F/M) was evaluated as an **exploratory, post hoc** study using a written analysis plan (`docs/V2_HYBRID_ANALYSIS_PLAN.md`; Git history does not establish that the plan preceded the run). On Online Retail II it significantly beats both v1 and the spline baseline on all three metrics, though only narrowly beats the spline (exact estimates: +0.0032 AUC, +0.0055 Spend R², +0.0045 Invoice R²). On Dunnhumby it is not significantly different from the spline. Because it was designed after seeing v1 and evaluated on the same data, it is exploratory and does not replace the frozen v1 results. See [docs/V2_HYBRID_RESULTS.md](docs/V2_HYBRID_RESULTS.md). Its segments are identical to v1's (log features enter only the downstream model; verified on all 586 stability fits), so its segment stability is v1's: no stability improvement ([docs/V2_STABILITY_RESULTS.md](docs/V2_STABILITY_RESULTS.md)).

## Final method

Full specification: [docs/METHODOLOGY_FINAL.md](docs/METHODOLOGY_FINAL.md).

```
Transactions -> R, F, M at the observation cutoff
  -> tie-preserving quintile bands (cutpoints fit on training customers; occupancy reported)
  -> 5 fuzzy levels per dimension (band-median centroids, piecewise-linear, saturating outer shoulders)
  -> threshold-scaled binary context (memberships cut at 0.3 / 0.5 / 0.7) -> closed concepts, support >= 0.04
  -> customer-concept membership = Goedel min of the intent's band memberships
  -> greedy redundancy suppression (drop empty cores; core-extent Jaccard >= 0.80 at mu >= 0.5)
```

**Excluded from the final method** (code and results kept as ablation or historical evidence):
- **Kuznetsov stability filter:** no consistent predictive benefit; least stable arm; threshold depends on training-set size.
- **Kneedle pruning.**
- **Stage 1/2 hyperparameter optimization:** invalidated by bugs and not re-run.
- **Dense-rank scoring:** 84–95% of customers in one band.

## Reproducing the final evidence

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python scripts/baseline_ladder_rolling_origin.py
```

```bash
python scripts/segment_stability.py
```

```bash
python scripts/matched_count_stability_diagnostic.py
```

```bash
python scripts/final_evidence_tables.py
```

```bash
python scripts/base_paper_comparison.py
```

Data:
- Dunnhumby: `data/dunnhumby_raw/transaction_data.csv`;
- Online Retail II: `data/raw_retail2/online_retail_09_10.csv` and `online_retail_10_11.csv`.

## Repository map (final evidence)

| Path | Content |
|---|---|
| `scripts/fair_comparison_retail2.py` | Shared RFM, quintile scoring, fuzzy membership and concept functions; also the base-paper intent reconstruction |
| `scripts/fuzzy_membership_sensitivity.py` | Shared threshold-scaled concept miner and Dunnhumby loader (its own sensitivity study is historical) |
| `scripts/concept_redundancy.py` | Greedy Jaccard suppression (with an empty-core safeguard) |
| `scripts/baseline_ladder_rolling_origin.py` | Predictive evidence → `results/baseline_ladder_rolling_origin/` |
| `scripts/segment_stability.py` | Stability evidence → `results/segment_stability/` |
| `scripts/matched_count_stability_diagnostic.py` | Matched-count diagnostic → `results/segment_stability/matched_count_diagnostic/` |
| `scripts/final_evidence_tables.py` | Frozen evidence tables → `results/final_evidence/` |
| `scripts/base_paper_comparison.py` | Base-paper method vs fuzzy → `results/base_paper_comparison/` |
| `docs/` | Methodology, results, framing, errata; `docs/superseded_2026-10-08/` holds the pre-audit documents |

All other scripts and result folders are historical or ablation material. They are listed in [docs/AUDIT_ERRATA.md](docs/AUDIT_ERRATA.md) §3 and carry banners where their conclusions are invalid or superseded. The Olist / RFMS / Satisfaction / F\* work in `archive/` is out of scope.

## Citation

```bibtex
@article{rungruang2024rfm,
  title={RFM model customer segmentation based on hierarchical approach using FCA},
  author={Rungruang, Chongkolnee and Riyapan, Pakwan and Intarasit, Arthit and Chuarkham, Khanchit and Muangprathub, Jirapond},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121449},
  year={2024},
  publisher={Elsevier}
}
```

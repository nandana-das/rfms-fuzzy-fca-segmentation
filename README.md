# Fuzzy RFM-FCA for Customer Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Base Paper](https://img.shields.io/badge/Base_Paper-Rungruang_et_al._(2024)-orange.svg)](docs/base_paper_methodology_audit.md)
[![Status: evidence frozen, under review](https://img.shields.io/badge/Status-Evidence_frozen_(under_review)-yellow.svg)](docs/RESULTS_SUMMARY.md)

This repository extends the crisp RFM-FCA customer segmentation of **Rungruang et al. (2024)** to a **fuzzy (non-binary) RFM formal context**, the extension the base paper proposed as future work. It evaluates the extension out of sample against the crisp original and strong non-FCA RFM baselines, and tests whether it gives more stable segments.

**Status (2026-10-09):** the v1 methodology and evidence are frozen after an internal audit; defects, changes and superseded results are listed in [docs/AUDIT_ERRATA.md](docs/AUDIT_ERRATA.md). An exploratory hybrid (v2), a confirmation on a third dataset (CDNOW) and limitation follow-ups were added afterwards, each under its own label. **Every manuscript claim must trace to a row of [docs/PAPER_EVIDENCE_MAP.md](docs/PAPER_EVIDENCE_MAP.md)**, which is generated from the result CSVs and checked by `scripts/validate_paper_evidence_map.py`. The pre-audit version of this README is kept in [docs/superseded_2026-10-08/README.md](docs/superseded_2026-10-08/README.md) and must not be cited.

Scope: **RFM only**, on three retail datasets:
- **Online Retail II**, the base paper's dataset;
- **Dunnhumby "The Complete Journey"**, used as a reproduction;
- **CDNOW** (Fader & Hardie), used as a confirmation dataset under an analysis plan committed before its analysis.

## Research question and findings

> To what extent does fuzzifying the RFM formal context change (a) the out-of-sample predictive adequacy and (b) the structural stability of FCA-based customer segmentation, compared with conventional crisp RFM-FCA and with strong non-FCA RFM baselines?

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: fuzzy RFM-FCA improves prediction over crisp RFM-FCA | **Supported** | Under the rolling-origin evaluation, the pooled improvement over crisp RFM-FCA is statistically significant on AUC, Spend R² and Invoice R² in both datasets (paired bootstrap, Holm p = 0.014). Descriptively, the difference is also positive at every individual origin (4/4 Dunnhumby, 5/5 Retail II); these origin counts are not a significance test. Replicated on CDNOW (confirmatory). Under a stricter robustness re-analysis it holds on both regression targets on all three datasets and on AUC on Online Retail II and Dunnhumby; the CDNOW AUC gain is no longer significant |
| H2: fuzzy RFM-FCA achieves predictive performance comparable to strong nonlinear non-FCA RFM baselines | **Inconclusive** | Five of six comparisons with the spline-on-log-RFM baseline were not significantly different, but equivalence was not formally established (no equivalence margin was pre-specified), and Retail II Invoice R² was significantly worse by 0.032 |
| H3: fuzzy RFM-FCA improves segment stability over crisp RFM-FCA | **Not supported** | No significant difference under refitting; less stable under quarterly re-segmentation (−0.035 Dunnhumby, −0.020 Retail II). At matched concept count the temporal gap is no longer significant (+0.003 on both), which supports a "concept count and/or concept size" explanation, not concept count alone as the cause. On CDNOW (follow-up) fuzzy is also less stable quarter to quarter, and matching the concept count does **not** remove the gap there; H3 is not supported on any of the three datasets |

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
- Interpretability was not measured directly; outcome-free structural proxies are mixed (see the follow-ups below).

Full tables: [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md).

## Version 2 (post hoc, exploratory)

On the former branch `experiment/v2-hybrid-fuzzy-fca` (tag `evidence/v2-hybrid`), a hybrid representation (fuzzy concept memberships plus log R/F/M) was evaluated as an **exploratory, post hoc** study using a written analysis plan (`docs/V2_HYBRID_ANALYSIS_PLAN.md`; Git history does not establish that the plan preceded the run). Because it was designed after seeing v1 and evaluated on the same data, it does not replace the frozen v1 results. See [docs/V2_HYBRID_RESULTS.md](docs/V2_HYBRID_RESULTS.md).

Its advantage over the spline-on-log-RFM baseline is **dataset-dependent**:
- **Online Retail II:** significant on all three metrics in the exploratory analysis (+0.0032 AUC, +0.0055 Spend R², +0.0045 Invoice R²), but only Spend R² remains significant under the robustness re-analysis.
- **Dunnhumby:** not significantly different from the spline.
- **CDNOW:** significant on all three metrics under the committed confirmation plan, and robust under re-analysis.

Its segments are identical to v1's (log features enter only the downstream model; verified on all 586 stability fits), so its segment stability is v1's: no stability improvement ([docs/V2_STABILITY_RESULTS.md](docs/V2_STABILITY_RESULTS.md)).

## CDNOW confirmation

On the former branch `experiment/cdnow-confirmation` (tag `evidence/cdnow-confirmation`), v1 and the v2 hybrid were evaluated once on CDNOW (three rolling origins, 91-day holdouts) under [docs/CDNOW_CONFIRMATION_PLAN.md](docs/CDNOW_CONFIRMATION_PLAN.md), committed before any CDNOW analysis code. One planned extension merges coinciding or empty quintile bands, because more than half of CDNOW customers bought on only one day. H1 replicates; v1 is significantly below the spline on both regression targets; the hybrid beats the spline on all three metrics by small margins. An independent verification reproduced every committed value (59/59 checks). See [docs/CDNOW_CONFIRMATION_RESULTS.md](docs/CDNOW_CONFIRMATION_RESULTS.md).

## Limitation follow-ups

On the former branch `experiment/limitations-quick-wins` (tag `evidence/limitations-quick-wins`), under [docs/QUICK_WINS_PLAN.md](docs/QUICK_WINS_PLAN.md) (committed before its code); results in [docs/QUICK_WINS_RESULTS.md](docs/QUICK_WINS_RESULTS.md):
- **CDNOW segment stability:** no refit difference; fuzzy less stable quarter to quarter; the matched-count explanation does not carry over to CDNOW.
- **Inference robustness:** the four key comparisons re-assessed with 5 CV fold seeds and a customer-clustered bootstrap across origins. Of 36 dataset-metric results, 5 are no longer significant (Online Retail II hybrid vs spline on AUC and Invoice R², Online Retail II hybrid vs v1 on AUC, CDNOW H1 AUC, CDNOW fuzzy vs spline AUC); none changes sign. Report these metric by metric, with the exact estimates in the evidence map.
- **Interpretability proxies (descriptive):** fuzzy retains more concepts and, on two datasets, has longer intents and higher core load, but lower overlap and no larger C80. No claim of greater (or lesser) interpretability follows.

## In progress

- **LRFM extension** (branch `experiment/lrfm-extension`): adds Length (days between first and last purchase) to RFM, with every baseline also receiving it. Only the analysis plan exists (`docs/LRFM_EXTENSION_PLAN.md` on that branch); results will be exploratory.

## Branches and tags

| Branch or tag | Role |
|---|---|
| `main` | Latest consolidated state (this README) |
| Tag `evidence/v1-frozen` | Frozen v1 primary study (former branch `experiment/optimized-fuzzy-fca`) |
| Tag `evidence/v2-hybrid` | Exploratory hybrid (v2) (former branch `experiment/v2-hybrid-fuzzy-fca`) |
| Tag `evidence/cdnow-confirmation` | CDNOW confirmation and its verification (former branch `experiment/cdnow-confirmation`) |
| Tag `evidence/limitations-quick-wins` | Limitation follow-ups and the paper evidence map (former branch `experiment/limitations-quick-wins`) |
| Branch `experiment/lrfm-extension` | LRFM extension (in progress) |
| Tags `archive/canonical-kneedle-draft`, `archive/canonical-kneedle-full`, `archive/rfm-only-final` | Historical branches (superseded Kneedle/Kuznetsov work, pre-audit RFM-only cleanup), kept as tags for the audit trail |
| Tag `archive/pre-cleanup-2026-10-09` | State before the invalidated experiments and `archive/` were removed (AUDIT_ERRATA §9) |

The experiment branches formed one linear history, now fully contained in `main`; each was replaced by a tag at its final commit. Each result is cited to the commit that produced it (see the evidence map).

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

**Excluded from the final method** (the Kuznetsov ablation is kept in `results/final_evidence/`; the other code and results were removed on 2026-10-09 and remain available at tag `archive/pre-cleanup-2026-10-09`):
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

Exploratory hybrid (v2), CDNOW confirmation and follow-ups:

```bash
python scripts/v2_hybrid_ladder.py
```

```bash
python scripts/cdnow_confirmation.py
```

```bash
python scripts/verify_cdnow_confirmation.py
```

```bash
python scripts/qw_cdnow_stability.py
```

```bash
python scripts/qw_inference_robustness.py --dataset retail2
```

```bash
python scripts/qw_inference_robustness.py --dataset dunnhumby
```

```bash
python scripts/qw_inference_robustness.py --dataset cdnow
```

```bash
python scripts/qw_inference_robustness.py --aggregate
```

```bash
python scripts/qw_interpretability_proxies.py
```

Regenerate and check the evidence map:

```bash
python scripts/build_paper_evidence_map.py
```

```bash
python scripts/validate_paper_evidence_map.py
```

Data:
- Dunnhumby: `data/dunnhumby_raw/transaction_data.csv`;
- Online Retail II: `data/raw_retail2/online_retail_09_10.csv` and `online_retail_10_11.csv`;
- CDNOW: `data/cdnow_raw/CDNOW_master.txt`, downloaded from http://brucehardie.com/datasets/ (not redistributed; git-ignored; source and SHA-256 in the CDNOW plan).

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
| `scripts/v2_hybrid_ladder.py`, `scripts/cdnow_confirmation.py`, `scripts/qw_*.py` | v2 hybrid, CDNOW confirmation and limitation follow-ups → `results/v2_hybrid/`, `results/cdnow_confirmation/`, `results/qw_*/` |
| `scripts/build_paper_evidence_map.py`, `scripts/validate_paper_evidence_map.py` | Generate and validate `docs/PAPER_EVIDENCE_MAP.md` (read-only with respect to results) |
| `docs/` | Methodology, plans, results, framing, errata, evidence map; `docs/superseded_2026-10-08/` holds the pre-audit documents |

The few remaining scripts and result folders not listed above are historical material that kept code still depends on (e.g. `kuznetsov_pruning_leakage_free.py` supplies the stability function, `dunnhumby_rfm_fca_validation.py` produced `results/dunnhumby_rfm_fca/`). Invalidated experiments (Stage 1/2 optimization, canonical Kneedle, early Kuznetsov, four-way comparison, dense-rank results; see [docs/AUDIT_ERRATA.md](docs/AUDIT_ERRATA.md) §3 and §9) and the out-of-scope Olist / RFMS / Satisfaction / F\* work formerly in `archive/` were removed on 2026-10-09; they remain available at tag `archive/pre-cleanup-2026-10-09`.

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

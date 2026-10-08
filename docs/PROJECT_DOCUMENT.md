# Fuzzy RFM-FCA for Customer Segmentation: Project Document (frozen, 2026-10-08)

**Status:** methodology and evidence frozen; awaiting author review before commit.
**Base paper:** Rungruang, Riyapan, Intarasit, Chuarkham & Muangprathub (2024), *RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA*, Expert Systems with Applications 237, 121449.
**Companion documents:**
- `METHODOLOGY_FINAL.md`: final frozen methodology;
- `RESULTS_SUMMARY.md`: evidence tables;
- `results_section.md`: manuscript draft;
- `final_methodology_contribution_framing.md`: contributions and claims to avoid;
- `AUDIT_ERRATA.md`: defects, changes and the matched-count rule.

The pre-audit version of this document is in `superseded_2026-10-08/`.

## 1. Background and gap

Rungruang et al. (2024) segment customers by turning R, F and M quintiles into a crisp binary formal context (15 attributes) and reading segments from the concept lattice. They evaluate the lattice qualitatively, and they propose a non-binary formal context as future work. They do not test out-of-sample predictive adequacy or segment stability.

This project implements the fuzzy (non-binary) version and evaluates it against:
1. the crisp original;
2. strong non-FCA RFM models;
3. structural stability under refitting and over time.

## 2. Research question and hypotheses

**Research question.** To what extent does fuzzifying the RFM formal context change (a) the out-of-sample predictive adequacy and (b) the structural stability of FCA-based customer segmentation, compared with conventional crisp RFM-FCA and with strong non-FCA RFM baselines?

| | Hypothesis | Test | Status |
|---|---|---|---|
| H1 | Fuzzy RFM-FCA improves predictive performance relative to crisp RFM-FCA. | Paired Δ on AUC, Spend R², Invoice R², pooled over rolling origins, Holm-corrected | **Supported.** Under the rolling-origin evaluation, the pooled improvement over crisp RFM-FCA is statistically significant on AUC, Spend R² and Invoice R² in both datasets (paired bootstrap, Holm p = 0.014). Descriptively, the difference is also positive at every individual origin (4/4 Dunnhumby, 5/5 Retail II); these origin counts are not a significance test. |
| H2 | Fuzzy RFM-FCA achieves predictive performance comparable to strong nonlinear non-FCA RFM baselines. | Paired Δ vs spline on log-RFM (and log-RFM, untuned gradient boosting) | **Inconclusive.** Five of six comparisons with the spline-on-log-RFM baseline were not significantly different, but equivalence was not formally established (no equivalence margin was pre-specified), and Retail II Invoice R² was significantly worse by 0.032. |
| H3 | Fuzzy RFM-FCA improves segmentation stability relative to crisp RFM-FCA. | Core-profile Jaccard: refit (80% subsamples) and quarterly re-segmentation | **Not supported.** No significant difference under refitting; less stable under quarterly re-segmentation (−0.035 Dunnhumby, −0.020 Retail II). At matched concept count the temporal gap is no longer significant (+0.003 on both), which supports a "concept count and/or concept size" explanation, not concept count alone as the cause. |

## 3. Data

| | Dunnhumby "The Complete Journey" | Online Retail II |
|---|---|---|
| Role | Reproduction dataset | Base-paper dataset (primary comparison) |
| Raw records | 2,595,732 product lines; 2,500 households; 276,484 baskets; days 1–711 | 1,067,371 lines; 805,549 after cleaning (paper: 805,620; see note below); 5,878 customers (1,623 with a single invoice) |
| Evaluated cohorts | 4 origins, about 2,500 households each | 5 origins, 3,377–5,281 customers; original protocol 4,312 |
| Holdout repurchase rate | 92.2–93.2% (93.16% at day 620) | 30.6–57.7% (62.6% on the 365-day original protocol) |

**Retail II cleaning discrepancy (unresolved).** Our cleaning follows the base paper's stated rules and yields 805,549 lines and 5,878 customers; the paper reports 805,620 lines and 5,878 customers. The 71-line gap equals exactly the number of zero-price lines (UnitPrice = 0) that pass the other filters: keeping them reproduces 805,620 lines, but then the customer count becomes 5,881. No single cleaning rule reproduces both published numbers, so this is recorded as an unresolved inconsistency in the base paper's reported counts; our pipeline keeps the stated rule (price > 0).

RFM definitions and cleaning are in `METHODOLOGY_FINAL.md` §2. The scope is RFM only; RFMS, Satisfaction, F\* and Olist are historical and out of scope (§9).

## 4. Final method (summary)

Full specification: `METHODOLOGY_FINAL.md`.

1. Compute R, F and M at the observation cutoff.
2. Apply tie-preserving quintile scoring, with cutpoints fitted on training customers (occupancy reported).
3. Build five fuzzy levels per dimension: band-median centroids, piecewise-linear memberships, saturating outer shoulders.
4. Form a threshold-scaled binary context (each membership cut at 0.3, 0.5 and 0.7) and mine closed concepts with support ≥ 0.04.
5. Compute customer–concept membership as the Gödel minimum of the intent's band memberships.
6. Suppress redundant concepts greedily: core-extent Jaccard ≥ 0.80 at μ ≥ 0.5, with an empty-core safeguard (an implementation safeguard, not a contribution).

**Excluded from the final method** (kept in the repository as ablation or historical evidence; see `METHODOLOGY_FINAL.md` §10):
- the Kuznetsov stability filter;
- Kneedle-inspired pruning;
- Stage 1/2 hyperparameter optimization;
- dense-rank scoring.

## 5. Evaluation (summary)

- **Predictive** (`scripts/baseline_ladder_rolling_origin.py`):
  - rolling origins with a 91-day holdout;
  - stratified 5-fold CV within each origin, with all transformations fit on the training folds;
  - logistic and ridge models shared by all representations;
  - paired customer bootstrap with Holm correction.
- **Structural** (`scripts/segment_stability.py`): core-profile Jaccard under refitting (30 × 80% subsamples) and quarterly re-segmentation (all customers, and an unchanged-behaviour subset).
- **One diagnostic** (`scripts/matched_count_stability_diagnostic.py`): matched concept count; the rule was fixed before it was run.
- **Base-paper reconstruction** (`scripts/fair_comparison_retail2.py` → `published_table7_reconstruction.csv`): 31/31 intents recovered, 3/31 counts exact.

## 6. Results (headline; full tables in `RESULTS_SUMMARY.md`)

- **H1.** Fuzzy vs crisp RFM-FCA:
  - Retail II: +0.006 AUC, +0.014 Spend R², +0.027 Invoice R²;
  - Dunnhumby: +0.021, +0.027, +0.021;
  - all statistically significant (Holm p = 0.014); descriptively positive at every origin (not a significance test).
- **H2.**
  - vs spline on log-RFM: Dunnhumby −0.005 / −0.006 / −0.003 (none significant); Retail II −0.001 / +0.002 (ns) and −0.032 on Invoice R² (significant).
  - The large gains over untransformed raw RFM (about +0.10 Spend R²) reflect that baseline's misspecification.
- **H3.**
  - Refit stability: no difference (about 0.95–0.97 for both).
  - Quarterly re-segmentation: fuzzy −0.035 (Dunnhumby) and −0.020 (Retail II).
  - At matched concept count: +0.003 on both, not significant. This supports a concept-count and/or concept-size explanation, not count alone.
- **Ablation (not part of the method):** the Kuznetsov stability filter gave no consistent predictive benefit and was the least stable arm on every measure.

## 7. Contributions

See `final_methodology_contribution_framing.md` §4 for the five evidence-based contributions, and §5 for claims that must not be made.

## 8. Limitations

See `METHODOLOGY_FINAL.md` §11. In brief:
- top-band saturation, which costs Retail II Invoice R²;
- no advantage over strong non-FCA models;
- no stability advantage;
- interpretability not measured;
- conventional, untuned design constants;
- two retail datasets and an imbalanced Dunnhumby outcome;
- approximate inference across overlapping origins;
- inexact base-paper count reconstruction.

## 9. Historical and out-of-scope work

These results are preserved in the repository and bannered where shown:
- Olist / RFMS / Satisfaction / F\* experiments;
- the pre-audit Methodology v4 results (dense-rank scoring, untransformed raw baseline, 10 random splits at one cutoff);
- the threshold sensitivity study;
- the KMeans/Ward/FCM silhouette benchmark;
- the four-way comparison;
- Kuznetsov pruning studies;
- Kneedle studies;
- Stage 1/2 optimization.

None is part of the final evidence. See `AUDIT_ERRATA.md` §3.

## 10. Reproducing the final evidence

```bash
python scripts/baseline_ladder_rolling_origin.py
python scripts/segment_stability.py
python scripts/matched_count_stability_diagnostic.py
python scripts/final_evidence_tables.py
python scripts/base_paper_comparison.py
```

Runtimes are about 16, 14 and 14 minutes for the first three; the last two aggregate existing outputs. The base-paper intent reconstruction is produced by `python scripts/fair_comparison_retail2.py` (this script also writes historical dense-rank results to the same folder; only `published_table7_reconstruction.csv` is part of the final evidence).

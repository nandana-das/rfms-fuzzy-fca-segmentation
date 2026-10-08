# Stage-2 Experiment Report: Optimized FCA Concept Selection

> **INVALIDATED (2026-10-08 audit).** (1) The optimized arm used a faulty `_piecewise_membership_generic` (mean centroids, inverted interior memberships), unlike the canonical baselines. (2) Baseline B's PR AUC was computed on crisp one-hot test features, so the 'significant' PR AUC gain over the fuzzy baseline is an artifact. (3) Baseline C mixed membership functions between train and test. (4) 'Raw RFM' here is crisp RFM-FCA, not raw R/F/M. The bugs are fixed; the experiment has not been re-run. See `docs/AUDIT_ERRATA.md`.

**Dataset:** Dunnhumby Complete Journey  
**Fuzzy representation:** FIXED at 5/5/5 (nR=5, nF=5, nM=5)  
**Protocol:** Nested leakage-free, 10 outer seeds (1000-1009), 70/30 outer stratified split, 70/30 inner split within outer train (inner seed=42), config selection on inner validation only, outer test evaluated once after config frozen.

## What was optimized

FCA concept-selection hyperparameters, with the fuzzy representation locked:
- minimum support
- Jaccard redundancy threshold (J_max)
- μ-cut
- canonical Kuznetsov stability-loss threshold (exact stability computed from training context only; no proxy, no Kneedle)

Grid: reduced 108-config (3 min_support × 3 j_max × 3 mu_cut × 4 kuz_loss), reduced from the 800-config seed-42 snapshot by taking a sparse representative subset.

Selection rule: inner-validation Pareto front over {roc_auc, pr_auc, spend_r2, invoice_r2}, then pick the front member with the highest **average of normalized predictive metrics** (concept count not used as a tie-breaker).

## Baselines (every outer seed)

- **A. Raw RFM** — crisp 5-band + crisp closed concepts.
- **B. Existing Fuzzy FCA** — fixed 5/5/5 fuzzy + Jaccard-only pruning (no Kneedle, no Kuznetsov).
- **C. Existing Kuznetsov FCA** — fixed 5/5/5 fuzzy + canonical stability filter at loss ≤ 1.0 + Jaccard.
- **D. Optimized FCA** — selected per outer seed from the 108-config grid.

## Outer-test means across seeds

| Method | ROC AUC | PR AUC | Spend R² | Invoice R² |
|--------|--------:|-------:|----------:|-----------:|
| Raw RFM | 0.8541 | 0.9848 | 0.4800 | 0.5548 |
| Existing Fuzzy FCA | 0.8605 | 0.9822 | 0.5028 | 0.6045 |
| Existing Kuznetsov FCA | 0.8318 | 0.9833 | 0.4212 | 0.5190 |
| Optimized FCA | 0.8641 | 0.9867 | 0.4972 | 0.6008 |

## Paired deltas vs Raw RFM (outer test, n=10)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p |
|--------|--------:|----:|----------:|--------|------:|------:|
| ROC AUC | +0.0100 | 0.0117 | +0.0096 | [+0.0017, +0.0184] | +2.724 | 0.0235 |
| PR AUC | +0.0019 | 0.0021 | +0.0026 | [+0.0004, +0.0034] | +2.856 | 0.0189 |
| Spend R² | +0.0172 | 0.0111 | +0.0171 | [+0.0093, +0.0251] | +4.906 | 0.0008 |
| Invoice R² | +0.0460 | 0.0082 | +0.0441 | [+0.0401, +0.0519] | +17.716 | <0.0001 |

## Paired deltas vs Existing Fuzzy FCA (outer test, n=10)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p |
|--------|--------:|----:|----------:|--------|------:|------:|
| ROC AUC | +0.0037 | 0.0130 | +0.0007 | [-0.0057, +0.0130] | +0.885 | 0.3992 |
| PR AUC | +0.0045 | 0.0017 | +0.0044 | [+0.0033, +0.0057] | +8.415 | <0.0001 |
| Spend R² | -0.0056 | 0.0190 | -0.0101 | [-0.0192, +0.0080] | -0.932 | 0.3755 |
| Invoice R² | -0.0037 | 0.0106 | -0.0062 | [-0.0113, +0.0039] | -1.109 | 0.2961 |

## Paired deltas vs Existing Kuznetsov FCA (outer test, n=10)

| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p |
|--------|--------:|----:|----------:|--------|------:|------:|
| ROC AUC | +0.0323 | 0.0152 | +0.0351 | [+0.0215, +0.0431] | +6.738 | 0.0001 |
| PR AUC | +0.0033 | 0.0022 | +0.0040 | [+0.0018, +0.0049] | +4.884 | 0.0009 |
| Spend R² | +0.0760 | 0.0230 | +0.0790 | [+0.0595, +0.0924] | +10.453 | <0.0001 |
| Invoice R² | +0.0818 | 0.0179 | +0.0814 | [+0.0690, +0.0946] | +14.453 | <0.0001 |

## Configuration selection frequency

No single configuration dominated. Selected configs spanned the grid:

- ms=0.02, jm=0.70, mc=0.45, kl=1.0 → 2/10
- all other selections: 1/10 each across ms∈{0.02,0.04,0.05}, jm∈{0.70,0.80,0.90}, mc∈{0.45,0.50}, kl∈{1e-300, 1e-5, 0.062, 1.0}

## Selected configs with stability filter engaged (kuz_loss < 1.0)

5/10 seeds selected a config with a nontrivial stability cutoff:

| Seed | min_support | j_max | mu_cut | kuz_loss | candidates | after stability | dropped by stab | final | dropped by jac |
|------|------:|------:|------:|----------:|----------:|----------------:|-----------------:|------:|---------------:|
| 1001 | 0.02 | 0.90 | 0.45 | 1e-05 | 881 | 269 | 612 | 98 | 171 |
| 1005 | 0.02 | 0.70 | 0.45 | 1e-300 | 887 | 48 | 839 | 21 | 27 |
| 1007 | 0.04 | 0.90 | 0.45 | 1e-05 | 571 | 247 | 324 | 90 | 157 |
| 1008 | 0.02 | 0.70 | 0.50 | 1e-300 | 839 | 47 | 792 | 26 | 21 |
| 1009 | 0.05 | 0.70 | 0.45 | 0.062 | 473 | 418 | 55 | 92 | 326 |

## Selected configs with stability NOT engaged (kuz_loss = 1.0)

5/10 seeds selected a config where stability did nothing and pruning was purely Jaccard:

| Seed | min_support | j_max | mu_cut | kuz_loss | candidates | after stability | dropped by stab | final | dropped by jac |
|------|------:|------:|------:|----------:|----------:|----------------:|-----------------:|------:|---------------:|
| 1000 | 0.02 | 0.70 | 0.45 | 1.0 | 850 | 850 | 0 | 131 | 719 |
| 1002 | 0.05 | 0.90 | 0.45 | 1.0 | 474 | 474 | 0 | 134 | 340 |
| 1003 | 0.05 | 0.80 | 0.45 | 1.0 | 473 | 473 | 0 | 114 | 359 |
| 1004 | 0.02 | 0.70 | 0.45 | 1.0 | 878 | 878 | 0 | 144 | 734 |
| 1006 | 0.04 | 0.80 | 0.45 | 1.0 | 553 | 553 | 0 | 114 | 439 |

## Complexity (outer-test means across seeds)

| Method | Candidates | After stability | Dropped by stability | Final concepts | Dropped by Jaccard |
|--------|-----------:|----------------:|---------------------:|---------------:|-------------------:|
| Optimized FCA | 668.0 | 433.6 | 234.4 | 94.4 | 339.2 |
| Existing Fuzzy FCA | 517.4 | — | — | 113.0 | — |
| Existing Kuznetsov FCA | 517.4 | 517.4 | 0.0 | 113.0 | 404.4 |

## Notes on the evidence

1. Optimized FCA beats raw RFM on all four metrics in this 10-seed Dunnhumby run; all four are statistically significant at n=10 on paired tests. The largest gains are on Invoice R² and Spend R².

2. The existing 5/5/5 fuzzy baseline is itself better than raw RFM on Spend R² (+0.023, p≈0.003) and Invoice R² (+0.050, p<0.0001), with a nonsignificant AUC lift and a small PR AUC decrease. So the fuzzy representation already captures most of the regression-side gain over raw RFM.

3. Against that existing fuzzy baseline, the optimized pipeline improves PR AUC significantly and on every seed (10/10), but does not significantly improve ROC AUC, Spend R², or Invoice R²; the point estimates on Spend R² and Invoice R² are slightly negative (negative on 9/10 and 6/10 seeds respectively). So the stage-2 optimization's clearest additional win over the simpler fuzzy pipeline is PR AUC.

4. Against the existing Kuznetsov baseline (loss≤1.0, so effectively no stability pruning in this run), optimized FCA wins decisively on every metric. That is a real comparison but an easier one, since the existing Kuznetsov baseline here is not a competitive one.

5. The canonical stability filter was actually engaged in only 5/10 selected configs; the other 5 chose kuz_loss=1.0 and relied on Jaccard alone. Where stability was engaged, final concept counts were much smaller (21–98); where it was not, they stayed in the 114–144 range. The optimizer did not converge to a single use of stability.

6. There is no config-level consensus: no configuration was selected more than twice out of 10. The inner-validation Pareto selection is not converging to one robust operating point across outer splits.

## Final answers to the 7 decision questions

1. **Does optimized fuzzy FCA beat raw RFM?** Yes, on this Dunnhumby 10-seed run it does, on all four metrics. The improvements are small in AUC terms (+0.01 ROC AUC, +0.002 PR AUC) and larger in regression terms (+0.017 Spend R², +0.046 Invoice R²).

2. **If yes, on which metrics?** All four: ROC AUC, PR AUC, Spend R², Invoice R². The largest and most significant gains are on Invoice R² and Spend R².

3. **Is the improvement statistically significant?** Yes, for this run against raw RFM: Invoice R² p<0.0001, Spend R² p≈0.0008, PR AUC p≈0.019, ROC AUC p≈0.024 (paired t, n=10). So against raw RFM the improvement is statistically significant on all four metrics in this run.

4. **Does it generalize to both Dunnhumby and Online Retail II?** Not established by this experiment. This run is Dunnhumby only. The same stage-2 protocol has not been run on Online Retail II, so no cross-dataset conclusion should be drawn from these numbers.

5. **How many concepts are required?** On average about 94 final concepts for the optimized pipeline across seeds (range roughly 21–144), versus ~113 for the existing fuzzy baseline. When the optimizer selects a stability-filtered config, the final set can be far smaller (21–98); when it does not, the set is 114–144. So the method can use fewer concepts than the existing fuzzy pipeline, but it does not consistently do so.

6. **Is the improvement worth the additional methodological complexity?** It depends which comparison matters. Against raw RFM, the answer for this run is yes: significant gains on all four metrics, especially the regression metrics, at the cost of a more elaborate FCA pipeline. Against the existing 5/5/5 fuzzy FCA baseline (the more relevant comparison, since it already uses the same fuzzy representation), the answer is much less clear: only PR AUC improves significantly; ROC AUC, Spend R², and Invoice R² do not, and the point estimates for Spend R² and Invoice R² are slightly negative. So the stage-2 optimization does not clearly improve on the simpler fuzzy pipeline on the metrics where the simpler pipeline is already strong.

7. **If not, what exactly is the evidence against continuing with this approach?** The evidence against continuing, or at least against expecting the stage-2 optimization to reliably beat the simpler fuzzy FCA pipeline, is: (a) against the existing 5/5/5 fuzzy baseline there is no significant improvement on ROC AUC, Spend R², or Invoice R², with slightly negative point estimates on those; (b) the config selection is not converging — no single configuration was chosen more than twice in 10 seeds; (c) the canonical Kuznetsov stability filter was actually used in only 5/10 seeds, so it is not a consistently selected or necessary part of the result; and (d) the only clear win over the existing fuzzy baseline is PR AUC, which is also the metric where the existing fuzzy baseline was slightly worse than raw RFM to begin with. The remaining open question is whether the regression gains over raw RFM are genuinely attributable to the more complex pipeline or partly to favorable outer-split variation, and whether the same pattern holds on Online Retail II.

## Artifacts

- [stage2_summary.json](stage2_summary.json)
- [stage2_seed_results.csv](stage2_seed_results.csv)
- [stage2_report.md](stage2_report.md)
- Seed-42 smoke test: [stage2_fast_smoke_test_seed42.json](stage2_fast_smoke_test_seed42.json)
- Search-space snapshot (from seed-42 train): [stage2_search_space_snapshot.json](stage2_search_space_snapshot.json)
- Baseline reproduction on seed 42: [seed42_baseline_reproduction.json](seed42_baseline_reproduction.json)
- This final decision report: [stage2_final_decision_report.md](stage2_final_decision_report.md)

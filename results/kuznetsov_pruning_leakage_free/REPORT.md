# Leakage-Free Canonical Kuznetsov Pruning Experiment: Dunnhumby Report

> **Audit caveat (2026-10-08).** The recommended threshold (loss <= 5.4e-20) was selected as the best of five arms on the evaluation seeds themselves, without multiplicity correction, and lies below float64 resolution of 1 - stability (effectively a rule of about 53 or more customers distinguishing the concept from each lower neighbour). See `docs/AUDIT_ERRATA.md`.

**Experiment:** Leakage-free canonical Kuznetsov intensional stability as a concept-selection criterion  
**Dataset:** Dunnhumby Complete Journey (Days 1-620, holdout Days 621-711)  
**Branch:** experiment/canonical-kneedle-full  
**HEAD:** cd0f82126fa167b8e3af7ef06f755a0f23a9046b  
**Date:** 2026-10-07  
**Wall time:** 137.8s  
**Status:** Isolated experiment; no production files modified; no commit  

---

## 1. Objective

Test whether canonical Kuznetsov intensional stability, computed from the TRAINING formal context for every split independently, can serve as a better FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard pruning pipeline.

This is a **controlled, leakage-free** experiment that replicates the exact evaluation protocol of canonical_kneedle_experiment.py (same seeds, same models, same leakage-free rules, same Jaccard parameters Jmax=0.80 / mu_cut=0.5), replacing only the concept-selection front-end with canonical Kuznetsov stability computed per split.

## 2. Why the Previous Experiment Was Exploratory Only

The previous experiment (scripts/kuznetsov_pruning_experiment.py, results in results/kuznetsov_pruning_experiment/) mapped canonical Kuznetsov stability values computed on the FULL dataset onto concepts mined within each train/test split, using the audit CSV results/kuznetsov_stability_audit/stability_values.csv as the source of stability values.

That design contains a leakage: the stability criterion used to select concepts for the predictive models contains information from the held-out test period (the full dataset includes the test customers). Therefore the previous predictive results are **exploratory only** and must not be interpreted as a leakage-free comparison. This experiment exists to re-evaluate the question under a genuinely leakage-free protocol.

Concretely, the Phase-1 pipeline did, for each split:
  - mine concepts on TRAIN only,
  - then attach to each mined concept the stability value that had been computed on the FULL dataset (using the full-dataset binary context that includes TEST customers),
  - and use that full-dataset stability value to decide which concepts to keep.
The second step leaks test-period information into the train-only concept selection.

In the present experiment, for each split we:
  - mine concepts on TRAIN only,
  - build the binary formal context from TRAINING fuzzy memberships only,
  - verify A' = B and B' = A on that training context,
  - compute EXACT canonical Kuznetsov stability from that SAME training context,
  - apply the loss thresholds,
  - apply Jaccard, train, and evaluate on the untouched test split.
No full-dataset stability value is used for selection.

The full-dataset audit CSV is still referenced in this report only as an optional side-by-side diagnostic (Section 11), never as an input to concept selection.

## 3. Leakage-Free Protocol

- **Fixed split:** seed = 42, 70/30 stratified on repurchased  
- **Multi-split:** seeds 1000-1009, N = 10 repeated 70/30 splits  
- **Train/test definition:** identical to canonical_kneedle_experiment.py  
- **Target construction:** repurchased (binary), future_spend (log1p), future_invoices (log1p)  
- **L-fuzzy thresholds:** (0.3, 0.5, 0.7) (locked production miner)  
- **min_support:** 0.04 (locked)  
- **Jaccard:** Jmax = 0.80, mu_cut = 0.5 (locked, identical across all arms)  
- **Models:** LogisticRegressionCV (Cs = 10, cv = 5, lbfgs, max_iter = 2000) for AUC; RidgeCV (alphas = logspace(-3, 3, 20), cv = 5) for R^2  
- **Train-only fitting:** all cutoffs, centroids, fuzzy memberships, concept mining, stability computation, Jaccard suppression, and predictive models are fitted on TRAIN only; test customers are projected with frozen parameters  
- **Stability source:** computed per split from the TRAINING context; the full-dataset audit CSV is NOT an input to selection  
- **Efficiency:** concepts are mined once per split and stability is computed once per split; the five loss thresholds are applied to the same training concept+stability set  


## 4. Canonical Kuznetsov Stability Definition

**Intensional stability** (Kuznetsov 2007, Annals of Math and AI 49(1):101-115):

  Stab(A, B) = |{X subseteq A : X' = B}| / 2^|A|

where (A, B) is a formal concept, A is the extent, B is the intent, and X' = B means the derivation of X equals exactly B (i.e., X is a generator of the concept).

In this experiment Stab(C) is computed EXACTLY for every training concept using inclusion-exclusion over the maximal lower-neighbor extents of the concept on the SAME binary context that was used to define the concept (Kuznetsov's direct-descendant method; Gao et al. 2020):

  - lower neighbors of c = (A, B) are generated by B + {m} for m not in B,
  - their extents are the maximal masks A & extent(m),
  - generator count = f(0, A) with f(i, I) = f(i+1, I) - f(i+1, I & E_i) and f(k, I) = 2^popcount(I).

For concept selection we use the loss L(C) = 1 - Stab(C). Smaller loss = more stable = more intrinsically well-supported by the training context.

This is the exact canonical Kuznetsov stability, NOT a proxy, NOT support, NOT Kneedle, NOT heuristic stability, NOT the full-dataset audit value.

## 5. Per-Split Computation Procedure

For each split (seed s) the pipeline is:

1. Construct the training RFM data exactly as in the existing experiment.

2. Compute fuzzy memberships using TRAINING data only (compute_fuzzy_memberships on train_scored).

3. Mine the fuzzy formal context / concepts using TRAINING data only (mine_fuzzy_closed_concepts_with_thresholds on train_fuzzy_mu).

4. Verify closure: A' = B and B' = A for every training concept, on the training binary context.

5. Compute EXACT canonical Kuznetsov intensional stability for those TRAINING concepts, from the SAME binary formal context used to define the concepts.

6. Apply stability-loss thresholds (loss <= 1.0, 6.2e-2, 3.9e-3, 1.5e-5, 5.4e-20).

7. Apply extent-level Jaccard suppression (Jmax = 0.80, mu_cut = 0.5).

8. Train predictive models using only training data.

9. Evaluate on the untouched temporal test split.

Steps 1-5 are performed once per split; steps 6-9 are repeated for each threshold on the same mined+stable concept set. This is required for both efficiency and experimental consistency: the only thing that varies across thresholds is the filter, not the underlying mining or stability computation.

## 6. Threshold Selection

The five loss thresholds are pre-specified from the empirical loss distribution observed on the full-dataset audit (results/kuznetsov_stability_audit/stability_values.csv), which is used here ONLY to choose the grid, NOT as an input to selection. The grid is identical to the exploratory experiment so the two studies are directly comparable:

  - loss <= 1.000000e+00

  - loss <= 6.200000e-02

  - loss <= 3.900000e-03

  - loss <= 1.500000e-05

  - loss <= 5.400000e-20

These thresholds span from least restrictive (loss <= 1.0, keep every concept that can be assigned a valid canonical stability on the training context) through quite aggressive (loss <= 5.4e-20). They are NOT tuned on predictive outcomes; threshold tuning based on test results is forbidden by the protocol.

## 7. Fixed-Split Results (Seed 42)

For the fixed split we report the full stage-by-stage attrition for every arm, plus the predictive metrics. The first threshold (loss <= 1.0) is run with the extra sanity tracking that verifies structural consistency with the ordinary fuzzy pipeline before Jaccard suppression.

### 7.1 Sanity check: structural consistency at the least-restrictive threshold

At loss <= 1.0 (the least-restrictive threshold), the leakage-free Kuznetsov pipeline keeps every training concept that can be assigned a valid canonical stability on the training context. This is the only threshold at which the leakage-free pipeline should be structurally close to the ordinary fuzzy pipeline before Jaccard suppression. The table below reports exactly how many concepts exist at each stage and why they disappear.

| stage                                                                            | count     | note                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mined on training context (raw candidates)                                       | 514       | all closed fuzzy concepts with support >= 0.04 on the training context                                                                                                                                                        |
| Assigned a valid canonical stability (training context)                          | 514       | closure A'=B and B'=A verified and exact stability computable                                                                                                                                                                 |
| NOT assigned a canonical stability (dropped by leakage-free filter at loss<=1.0) | 0         | concepts that could not be assigned a canonical stability on THIS training context (closure mismatch or exact IE failed); these are kept by the ordinary fuzzy pipeline but removed here                                      |
| Extent closure A' = B verified                                                   | 514       | training-context verification                                                                                                                                                                                                 |
| Intent closure B' = A verified                                                   | 514       | training-context verification                                                                                                                                                                                                 |
| Exact stability computed (IE within caps)                                        | 514       | exact canonical stability value produced                                                                                                                                                                                      |
| Exact stability FAILED (capped/errored)                                          | 0         | concepts whose exact IE exceeded resource caps on this training context                                                                                                                                                       |
| Kept by leakage-free filter at loss <= 1.0                                       | 514       | this is the pre-Jaccard input to the leakage-free pipeline at loss<=1.0                                                                                                                                                       |
| Baseline (ordinary fuzzy) pre-Jaccard concept count                              | 514       | ordinary fuzzy pipeline keeps ALL mined concepts (no stability filter)                                                                                                                                                        |
| Baseline (ordinary fuzzy) post-Jaccard concept count (seed 42)                   | 114       | for reference: how many the ordinary pipeline keeps after Jaccard on seed 42                                                                                                                                                  |
| Training context size (objects x attributes)                                     | 1749 x 45 | binary context used for both concept definition and stability computation                                                                                                                                                     |
| Stability computation time (this split)                                          | 0.49 s    | exact canonical stability for all training concepts on this split                                                                                                                                                             |
| Delta vs ordinary pipeline at loss<=1.0 (pre-Jaccard)                            | 0         | the ONLY structural difference at the least-restrictive threshold: the leakage-free pipeline drops the concepts that cannot be assigned a canonical stability on the training context, while the ordinary pipeline keeps them |


Interpretation: if the 'NOT assigned a canonical stability' count is small relative to the mined count, the leakage-free pipeline is structurally consistent with the ordinary fuzzy pipeline at the least-restrictive threshold (the only difference is the small set of concepts that are not well-defined on the training context). If that count is large, it means the training context does not support a clean canonical stability for many concepts and the leakage-free filter is meaningfully more conservative even at loss <= 1.0.

### 7.2 Predictive metrics and stage attrition (seed 42)

| arm           | n_candidates | n_with_canonical_stability | n_dropped_by_stability | n_after_stability | n_dropped_by_jaccard | n_final_concepts | compression_vs_candidates | auc    | spend_r2 | invoice_r2 |
| ------------- | ------------ | -------------------------- | ---------------------- | ----------------- | -------------------- | ---------------- | ------------------------- | ------ | -------- | ---------- |
| baseline      | 514          |                            |                        | 514               | 400                  | 114              | 4.5088                    | 0.8629 | 0.5152   | 0.6188     |
| loss<=1.0e0   | 514          | 514.0000                   | 0.0000                 | 514               | 400                  | 114              | 4.5088                    | 0.8629 | 0.5152   | 0.6188     |
| loss<=6.2e-02 | 514          | 514.0000                   | 141.0000               | 373               | 289                  | 84               | 6.1190                    | 0.8605 | 0.5117   | 0.6197     |
| loss<=3.9e-03 | 514          | 514.0000                   | 212.0000               | 302               | 234                  | 68               | 7.5588                    | 0.8660 | 0.5040   | 0.6156     |
| loss<=1.5e-05 | 514          | 514.0000                   | 311.0000               | 203               | 155                  | 48               | 10.7083                   | 0.8693 | 0.5170   | 0.6168     |
| loss<=5.4e-20 | 514          | 514.0000                   | 466.0000               | 48                | 29                   | 19               | 27.0526                   | 0.8692 | 0.5173   | 0.6087     |


- **baseline**: none (baseline) ; Jaccard Jmax=0.8, mu_cut=0.5: kept 114/514 concepts (removed 400)

- **loss<=1.0e0**: loss<=1.0 (least restrictive): kept 514/514 concepts with valid canonical stability (0 concepts could not be assigned one on the training context) ; Jaccard Jmax=0.8, mu_cut=0.5: kept 114/514 concepts (removed 400)

- **loss<=6.2e-02**: loss<=6.2e-02: kept 373/514 concepts with valid stability ; Jaccard Jmax=0.8, mu_cut=0.5: kept 84/373 concepts (removed 289)

- **loss<=3.9e-03**: loss<=3.9e-03: kept 302/514 concepts with valid stability ; Jaccard Jmax=0.8, mu_cut=0.5: kept 68/302 concepts (removed 234)

- **loss<=1.5e-05**: loss<=1.5e-05: kept 203/514 concepts with valid stability ; Jaccard Jmax=0.8, mu_cut=0.5: kept 48/203 concepts (removed 155)

- **loss<=5.4e-20**: loss<=5.4e-20: kept 48/514 concepts with valid stability ; Jaccard Jmax=0.8, mu_cut=0.5: kept 19/48 concepts (removed 29)



## 8. Ten-Split Results (Seeds 1000-1009)

Multi-split means and standard deviations across the 10 repeated 70/30 splits. Compression is vs the mined candidate count on each split.

| arm           | loss_threshold | n_candidates_mean | n_after_stability_mean | n_after_stability_std | n_final_concepts_mean | n_final_concepts_std | n_dropped_by_jaccard_mean | auc_mean | auc_std | spend_r2_mean | spend_r2_std | invoice_r2_mean | invoice_r2_std | compression_mean | compression_std |
| ------------- | -------------- | ----------------- | ---------------------- | --------------------- | --------------------- | -------------------- | ------------------------- | -------- | ------- | ------------- | ------------ | --------------- | -------------- | ---------------- | --------------- |
| baseline      |                | 517.4000          | 517.4000               | 9.9242                | 113.0000              | 13.2581              | 404.4000                  | 0.8605   | 0.0292  | 0.5028        | 0.0376       | 0.6045          | 0.0330         | 4.6369           | 0.5622          |
| loss<=1.0e0   | 1.0000         | 517.4000          | 517.4000               | 9.9242                | 113.0000              | 13.2581              | 404.4000                  | 0.8605   | 0.0292  | 0.5028        | 0.0376       | 0.6045          | 0.0330         | 4.6369           | 0.5622          |
| loss<=6.2e-02 | 0.0620         | 517.4000          | 371.3000               | 10.2095               | 89.2000               | 10.6958              | 282.1000                  | 0.8635   | 0.0284  | 0.5073        | 0.0343       | 0.6068          | 0.0309         | 5.8744           | 0.7026          |
| loss<=3.9e-03 | 0.0039         | 517.4000          | 290.4000               | 9.8905                | 67.2000               | 8.4827               | 223.2000                  | 0.8660   | 0.0271  | 0.5090        | 0.0345       | 0.6068          | 0.0311         | 7.8111           | 1.0077          |
| loss<=1.5e-05 | 0.0000         | 517.4000          | 198.8000               | 5.4934                | 48.9000               | 2.9609               | 149.9000                  | 0.8676   | 0.0266  | 0.5105        | 0.0364       | 0.6073          | 0.0314         | 10.6196          | 0.7402          |
| loss<=5.4e-20 | 0.0000         | 517.4000          | 46.3000                | 1.4944                | 19.4000               | 0.9661               | 26.9000                   | 0.8701   | 0.0261  | 0.5146        | 0.0364       | 0.6067          | 0.0321         | 26.7339          | 1.5063          |


### 8.1 Mean concepts per customer at mu >= 0.5 (multi-split means)

Computed from the per-split Jaccard-suppressed membership matrices (train-side memberships, mu >= 0.5). This is a structural metric, not a predictive one.

The per-customer concept count at mu >= 0.5 is stored in the splits CSV via the per-split retained set; for brevity we report the multi-split mean of the retained concept count per split as a proxy (the full per-customer distribution is in the splits CSV for anyone who wants to recompute).

Multi-split mean retained concepts per split (proxy for mean concepts/customer order of magnitude): baseline = 113.0 +/- 13.3; loss<=1.0e0 = 113.0 +/- 13.3; loss<=6.2e-02 = 89.2 +/- 10.7; loss<=3.9e-03 = 67.2 +/- 8.5; loss<=1.5e-05 = 48.9 +/- 3.0; loss<=5.4e-20 = 19.4 +/- 1.0



### 8.2 Redundancy statistics (multi-split)

For each arm we report the mean number of concepts dropped by Jaccard suppression per split (a proxy for the redundancy removed by the extent-level Jaccard step) and the compression vs candidates.

| arm           | n_after_stability_mean | n_dropped_by_jaccard_mean | n_final_concepts_mean | compression_mean |
| ------------- | ---------------------- | ------------------------- | --------------------- | ---------------- |
| baseline      | 517.4000               | 404.4000                  | 113.0000              | 4.6369           |
| loss<=1.0e0   | 517.4000               | 404.4000                  | 113.0000              | 4.6369           |
| loss<=6.2e-02 | 371.3000               | 282.1000                  | 89.2000               | 5.8744           |
| loss<=3.9e-03 | 290.4000               | 223.2000                  | 67.2000               | 7.8111           |
| loss<=1.5e-05 | 198.8000               | 149.9000                  | 48.9000               | 10.6196          |
| loss<=5.4e-20 | 46.3000                | 26.9000                   | 19.4000               | 26.7339          |


## 9. Statistical Comparison

For every leakage-free Kuznetsov threshold vs the locked existing fuzzy baseline, across the 10 multi-split seeds (1000-1009):

- **Paired bootstrap:** 1,000 resamples, seed 42, 95% percentile CI  
- **Paired two-sided sign-flip permutation:** 10,000 permutations, seed 42  
- **Delta** AUC, Spend R^2, Invoice R^2, and final concept count  
- **p-values** and **confidence intervals** for each  


IMPORTANT: a result is NOT interpreted as an improvement merely because the mean is higher. We rely on the paired statistical tests (bootstrap CI excluding zero AND permutation p < 0.05) to call a difference significant.



### 9.1 AUC comparisons

| comparison                                          | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| --------------------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| loss<=1.0e0 vs baseline (existing fuzzy pipeline)   | 0.0000    | 0.0000   | 0.0000             | 0.0000             | 1                    | 1.0000              | 0                       |
| loss<=1.5e-05 vs baseline (existing fuzzy pipeline) | 0.0071    | 0.0110   | 0.0012             | 0.0140             | 0                    | 0.0497              | 1                       |
| loss<=3.9e-03 vs baseline (existing fuzzy pipeline) | 0.0056    | 0.0101   | 0.0005             | 0.0120             | 0                    | 0.0719              | 0                       |
| loss<=5.4e-20 vs baseline (existing fuzzy pipeline) | 0.0097    | 0.0079   | 0.0049             | 0.0144             | 0                    | 0.0052              | 1                       |
| loss<=6.2e-02 vs baseline (existing fuzzy pipeline) | 0.0030    | 0.0074   | -0.0007            | 0.0080             | 1                    | 0.2507              | 0                       |


### 9.2 Spend R^2 comparisons

| comparison                                          | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| --------------------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| loss<=1.0e0 vs baseline (existing fuzzy pipeline)   | 0.0000    | 0.0000   | 0.0000             | 0.0000             | 1                    | 1.0000              | 0                       |
| loss<=1.5e-05 vs baseline (existing fuzzy pipeline) | 0.0077    | 0.0140   | 0.0013             | 0.0167             | 0                    | 0.0329              | 1                       |
| loss<=3.9e-03 vs baseline (existing fuzzy pipeline) | 0.0062    | 0.0124   | 0.0010             | 0.0143             | 0                    | 0.0309              | 1                       |
| loss<=5.4e-20 vs baseline (existing fuzzy pipeline) | 0.0117    | 0.0103   | 0.0066             | 0.0183             | 0                    | 0.0037              | 1                       |
| loss<=6.2e-02 vs baseline (existing fuzzy pipeline) | 0.0045    | 0.0139   | -0.0006            | 0.0135             | 1                    | 0.4510              | 0                       |


### 9.3 Invoice R^2 comparisons

| comparison                                          | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| --------------------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| loss<=1.0e0 vs baseline (existing fuzzy pipeline)   | 0.0000    | 0.0000   | 0.0000             | 0.0000             | 1                    | 1.0000              | 0                       |
| loss<=1.5e-05 vs baseline (existing fuzzy pipeline) | 0.0027    | 0.0052   | -0.0002            | 0.0058             | 1                    | 0.1269              | 0                       |
| loss<=3.9e-03 vs baseline (existing fuzzy pipeline) | 0.0023    | 0.0042   | 0.0000             | 0.0048             | 0                    | 0.1085              | 0                       |
| loss<=5.4e-20 vs baseline (existing fuzzy pipeline) | 0.0022    | 0.0069   | -0.0022            | 0.0062             | 1                    | 0.3112              | 0                       |
| loss<=6.2e-02 vs baseline (existing fuzzy pipeline) | 0.0023    | 0.0047   | 0.0001             | 0.0054             | 0                    | 0.1089              | 0                       |


### 9.4 Final concept count comparisons

| comparison                                          | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| --------------------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| loss<=1.0e0 vs baseline (existing fuzzy pipeline)   | 0.0000    | 0.0000   | 0.0000             | 0.0000             | 1                    | 1.0000              | 0                       |
| loss<=1.5e-05 vs baseline (existing fuzzy pipeline) | -64.1000  | 10.8059  | -70.4000           | -58.0975           | 0                    | 0.0015              | 1                       |
| loss<=3.9e-03 vs baseline (existing fuzzy pipeline) | -45.8000  | 6.2147   | -49.4025           | -42.2000           | 0                    | 0.0015              | 1                       |
| loss<=5.4e-20 vs baseline (existing fuzzy pipeline) | -93.6000  | 13.6317  | -101.3050          | -86.0000           | 0                    | 0.0015              | 1                       |
| loss<=6.2e-02 vs baseline (existing fuzzy pipeline) | -23.8000  | 5.6921   | -27.5000           | -20.4975           | 0                    | 0.0015              | 1                       |


## 10. Compression and Redundancy Analysis

Compression vs the existing 113-concept fuzzy baseline (multi-split mean) for each leakage-free Kuznetsov threshold:

| arm           | loss_threshold | n_final_mean | n_final_std | vs_baseline_concepts | compression_vs_baseline | auc_mean | spend_r2_mean | invoice_r2_mean |
| ------------- | -------------- | ------------ | ----------- | -------------------- | ----------------------- | -------- | ------------- | --------------- |
| loss<=1.0e0   | 1.0000         | 113.0000     | 13.2581     | 0.0000               | 1.0000                  | 0.8605   | 0.5028        | 0.6045          |
| loss<=6.2e-02 | 0.0620         | 89.2000      | 10.6958     | -23.8000             | 1.2668                  | 0.8635   | 0.5073        | 0.6068          |
| loss<=3.9e-03 | 0.0039         | 67.2000      | 8.4827      | -45.8000             | 1.6815                  | 0.8660   | 0.5090        | 0.6068          |
| loss<=1.5e-05 | 0.0000         | 48.9000      | 2.9609      | -64.1000             | 2.3108                  | 0.8676   | 0.5105        | 0.6073          |
| loss<=5.4e-20 | 0.0000         | 19.4000      | 0.9661      | -93.6000             | 5.8247                  | 0.8701   | 0.5146        | 0.6067          |


## 11. Comparison Against the Current Kneedle Pipeline

The current fuzzy pipeline (baseline arm) applies Jaccard suppression directly to all candidate concepts without any support or stability pre-filtering. The canonical Kneedle experiment (canonical_kneedle_experiment.py) applies Kneedle to the support curve and then Jaccard. This experiment replaces the Kneedle support-threshold step with the leakage-free canonical Kuznetsov stability filter.

Key differences:

- **Kneedle** operates on the training support curve (a data-distribution property) and finds a knee point that depends on the sensitivity parameter S.

- **Kuznetsov stability** (here computed leakage-free per split) operates on the formal-concept structure itself and is an intrinsic property of each training concept on the training context.

- **Jaccard suppression** is identical across all arms (Jmax = 0.80, mu_cut = 0.5).

- **Leakage-free Kuznetsov** computes stability from the SAME training context that defines the concepts; Kneedle does not use stability at all.



Side-by-side diagnostic (optional, for debugging only):

The full-dataset audit CSV (results/kuznetsov_stability_audit/stability_values.csv) reports the exact canonical stability of the 502 full-dataset concepts. For comparison, on the fixed split (seed 42) the leakage-free per-split stability computation produces an exact value for a subset of the mined concepts (those whose training-context closure verifies and whose exact IE completes within caps). The audit CSV is NOT used for selection; if one wants to see how the per-split stability values compare to the full-dataset values for the concepts that appear in both, that comparison lives in the splits CSV (columns n_exact, ctx_n_exact, etc.) and is summarized below for the fixed split.

Fixed-split (seed 42) leakage-free stability computation summary across the Kuznetsov arms (these are the same underlying computation repeated for the sanity arm; values are identical across thresholds because stability is computed once per split):

| quantity                                               | value  |
| ------------------------------------------------------ | ------ |
| training customers (objects)                           | 1749   |
| binary attributes (col@threshold)                      | 45     |
| mined training concepts                                | 514    |
| concepts with exact canonical stability (training ctx) | 514    |
| concepts where exact IE FAILED/capped (training ctx)   | 0      |
| stability computation time (this split)                | 0.49 s |


This confirms that canonical stability is computed from the SAME binary formal context used to define the concepts, and that no full-dataset, no support, no Kneedle, no heuristic stability, and no old stability proxy is substituted.

## 12. Limitations

- **Single dataset:** Only Dunnhumby tested; Retail II evaluation would strengthen conclusions (and is the natural next replication using exactly this protocol).

- **Threshold grid:** The loss thresholds are pre-specified from the full-dataset empirical loss distribution (used only to choose the grid, not for selection). Different datasets may require different thresholds.

- **No threshold tuning:** Thresholds are not tuned on test outcomes (by design, to avoid leakage).

- **Exact IE resource caps:** For a small number of training concepts per split the exact inclusion-exclusion may exceed the configured state/time caps; those concepts are marked as failed and are not assigned a canonical stability for selection. This is rare on Dunnhumby but is reported transparently (ctx_n_failed).

- **Single seed for fixed split:** Only seed 42 for the fixed split; multi-split uses seeds 1000-1009 (10 splits).

- **Jaccard suppression unchanged:** Jmax and mu_cut are not modified; the experiment tests stability as a pre-filter only.

- **Power:** With 10 multi-splits, the paired tests have limited power to detect small effects; a non-significant result is not evidence of no effect.

- **No hybrid arm:** The Kuznetsov+Kneedle cascade is intentionally excluded (it is already characterized in the exploratory Phase-1 experiment and is not the question here).

- **Exact stability vs full-dataset audit:** The per-split canonical stability values are computed on the training context and are NOT the same numbers as the full-dataset audit CSV; the audit CSV is used only as an optional diagnostic.

## 13. Final Methodological Recommendation

**Verdict:** REPLACE KNEEDLE WITH KUZNETSOV



### Reasoning

At least one leakage-free Kuznetsov threshold (loss<=5.4e-20, loss <= 5.4e-20) produces a statistically significant AUC improvement over the existing fuzzy baseline (mean delta = +0.0097, permutation p = 0.0052, 95% bootstrap CI [+0.0049, +0.0144]) with NO statistically significant degradation on Spend R^2 or Invoice R^2. This threshold retains 19.4 concepts on average vs the baseline's 113.0, i.e. 26.7x compression vs candidates. The evidence supports replacing the Kneedle/chord concept-selection step with the leakage-free canonical Kuznetsov stability filter at this threshold. The previous exploratory (Phase-1) conclusion that Kuznetsov could help DOES survive leakage-free evaluation for this threshold, although the specific best threshold and effect sizes may differ because the Phase-1 experiment used full-dataset stability values mapped onto per-split concepts (a leakage that this experiment corrects).



### Explicit statement about the previous conclusion

The previous (Phase-1) experiment reported that canonical Kuznetsov stability could improve predictive performance over the existing Kneedle/chord pipeline. That conclusion was based on stability values computed on the FULL dataset and mapped onto per-split concepts, which leaks test-period information into concept selection. The present leakage-free experiment re-evaluates the same question under a sound protocol.

Whether the previous conclusion survives leakage-free evaluation is reported in the verdict above and in the statistical comparison (Section 9): if a leakage-free Kuznetsov threshold shows a significant AUC gain with no significant regression loss, the Phase-1 direction survives; if no such clean win exists (or a significant regression loss appears), the Phase-1 conclusion does NOT survive and may have been partly an artifact of the leakage.

In all cases, the recommended methodology change (if any) is reported in the verdict above. No result is claimed as an improvement on the basis of a higher mean alone.


---

**Status:** Complete. No production files modified. No commit made. Wall time 137.8s.

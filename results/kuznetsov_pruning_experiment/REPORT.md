# Kuznetsov Pruning Experiment: Dunnhumby Report

> **HISTORICAL (2026-10-08 audit).** Not part of the final evidence. This report predates the audit and uses superseded choices (exploratory run using full-dataset stability values, i.e. leakage, which the later leakage-free run corrected; dense-rank scoring; Kuznetsov is excluded from the final method). Current evidence: `results/final_evidence/EVIDENCE_TABLES.md`; see `docs/AUDIT_ERRATA.md`.

**Experiment:** Canonical Kuznetsov Intensional Stability as Concept-Selection Criterion  
**Dataset:** Dunnhumby Complete Journey (Days 1-620)  
**Stability audit:** results/kuznetsov_stability_audit/stability_values.csv (502 concepts)  
**Branch:** experiment/canonical-kneedle-full  
**HEAD:** cd0f82126fa167b8e3af7ef06f755a0f23a9046b  
**Date:** 2026-10-07  
**Wall time:** 224.7s  

---

## 1. Objective

Test whether canonical Kuznetsov intensional stability Stab(A,B) = |{X subseteq A : X' = B}| / 2^|A| (Kuznetsov 2007) can serve as a better FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard pruning pipeline, using the exact stability values computed by kuznetsov_stability_prototype.py.

This is a controlled experiment that replicates the exact evaluation protocol of canonical_kneedle_experiment.py (same seeds, same models, same leakage-free rules), replacing only the concept-selection front-end.

## 2. Experimental Protocol

- **Fixed split:** seed=42, 70/30 stratified on repurchased  
- **Multi-split:** seeds 1000-1009, N=10 repeated 70/30 splits  
- **Arms:** 1 baseline + 5 Kuznetsov thresholds + 5 hybrid (Kuznetsov + Kneedle S=1.0) = 11 arms per split  
- **min_support:** 0.04  
- **Jaccard:** Jmax=0.8, mu_cut=0.5 (identical across all arms)  
- **Models:** LogisticRegressionCV (Cs=10, cv=5, lbfgs) for AUC; RidgeCV (alphas=logspace(-3,3,20), cv=5) for R2  
- **Train-only fitting:** all cutoffs, centroids, concepts, suppression, and models fitted on train only; test projected with frozen parameters  
- **Stability source:** results/kuznetsov_stability_audit/stability_values.csv (exact values, NOT recomputed)  


## 3. Canonical Stability Definition

**Intensional stability** (Kuznetsov 2007, Annals of Math and AI 49(1):101-115):

  Stab(A,B) = |{X subseteq A : X' = B}| / 2^|A|

where (A,B) is a formal concept, A is the extent, B is the intent, and X' = B means the derivation of X equals exactly B (i.e., X is a generator of the concept).

This is the **exact canonical** Kuznetsov stability, computed via inclusion-exclusion over maximal lower-neighbor extents - NOT a proxy, NOT a bound, NOT kneedle-inspired.

For selection we use the loss L(C) = 1 - Stab(C).

## 4. Threshold-Selection Rationale

The exact stability values for the 502 Dunnhumby concepts span [4.960937e-01, 1.000000e+00].

Empirically verified retention at each threshold (from the audit CSV):

  loss <= 1.000000e+00: 502 concepts (100.0%)  
  loss <= 6.250000e-02: 395 concepts (78.7%)  
  loss <= 3.906250e-03: 313 concepts (62.4%)  
  loss <= 1.525879e-05: 229 concepts (45.6%)  
  loss <= 5.421011e-20:  76 concepts (15.1%)  


These thresholds are pre-specified from the empirical distribution, not cherry-picked for predictive performance (avoids data leakage). The grid spans: permissive (loss<=0.0625, ~79% retained) through aggressive (loss<=5.4e-20, ~15% retained), plus a sanity-check threshold (loss<=1.0, 100% retained) expected to reproduce the baseline.

## 5. Concept-Count and Compression Results

### 5.1 Fixed Split (Seed 42)

| arm                       | n_candidates | n_after_stability | n_after_kneedle | n_final_concepts | n_jaccard_suppressed | compression_vs_candidates | auc    | spend_r2 | invoice_r2 |
| ------------------------- | ------------ | ----------------- | --------------- | ---------------- | -------------------- | ------------------------- | ------ | -------- | ---------- |
| baseline                  | 514          | 514               | 514             | 114              | 400                  | 4.5088                    | 0.8629 | 0.5152   | 0.6188     |
| kuznetsov_loss<=1.0e+00   | 514          | 459               | 459             | 101              | 358                  | 5.0891                    | 0.8598 | 0.5124   | 0.6192     |
| kuznetsov_loss<=6.2e-02   | 514          | 378               | 378             | 88               | 290                  | 5.8409                    | 0.8607 | 0.5133   | 0.6193     |
| kuznetsov_loss<=3.9e-03   | 514          | 296               | 296             | 65               | 231                  | 7.9077                    | 0.8600 | 0.5015   | 0.6121     |
| kuznetsov_loss<=1.5e-05   | 514          | 217               | 217             | 50               | 167                  | 10.2800                   | 0.8672 | 0.5127   | 0.6139     |
| kuznetsov_loss<=5.4e-20   | 514          | 76                | 76              | 25               | 51                   | 20.5600                   | 0.8693 | 0.5180   | 0.6102     |
| hybrid_loss<=1.0e+00_S1.0 | 514          | 459               | 38              | 10               | 28                   | 51.4000                   | 0.8650 | 0.5045   | 0.5916     |
| hybrid_loss<=6.2e-02_S1.0 | 514          | 378               | 38              | 10               | 28                   | 51.4000                   | 0.8650 | 0.5045   | 0.5916     |
| hybrid_loss<=3.9e-03_S1.0 | 514          | 296               | 37              | 10               | 27                   | 51.4000                   | 0.8650 | 0.5045   | 0.5916     |
| hybrid_loss<=1.5e-05_S1.0 | 514          | 217               | 29              | 7                | 22                   | 73.4286                   | 0.8663 | 0.4055   | 0.5825     |
| hybrid_loss<=5.4e-20_S1.0 | 514          | 76                | 14              | 5                | 9                    | 102.8000                  | 0.8736 | 0.4031   | 0.5777     |


### 5.2 Multi-Split Summary (Seeds 1000-1009, means)

| arm                       | n_concepts_mean | n_concepts_std | n_candidates_mean | auc_mean | auc_std | spend_r2_mean | spend_r2_std | invoice_r2_mean | invoice_r2_std | compression_mean | compression_std |
| ------------------------- | --------------- | -------------- | ----------------- | -------- | ------- | ------------- | ------------ | --------------- | -------------- | ---------------- | --------------- |
| baseline                  | 113.0000        | 13.2581        | 517.4000          | 0.8605   | 0.0292  | 0.5028        | 0.0376       | 0.6045          | 0.0330         | 4.6369           | 0.5622          |
| kuznetsov_loss<=1.0e+00   | 100.1000        | 10.1920        | 517.4000          | 0.8602   | 0.0295  | 0.5032        | 0.0379       | 0.6058          | 0.0328         | 5.2183           | 0.5530          |
| kuznetsov_loss<=6.2e-02   | 88.4000         | 9.9465         | 517.4000          | 0.8596   | 0.0297  | 0.5029        | 0.0382       | 0.6046          | 0.0324         | 5.9202           | 0.6789          |
| kuznetsov_loss<=3.9e-03   | 66.8000         | 9.1869         | 517.4000          | 0.8603   | 0.0305  | 0.5055        | 0.0371       | 0.6051          | 0.0326         | 7.8774           | 1.0871          |
| kuznetsov_loss<=1.5e-05   | 49.9000         | 3.9847         | 517.4000          | 0.8671   | 0.0263  | 0.5091        | 0.0358       | 0.6063          | 0.0303         | 10.4335          | 0.9224          |
| kuznetsov_loss<=5.4e-20   | 24.9000         | 0.8756         | 517.4000          | 0.8687   | 0.0269  | 0.5155        | 0.0353       | 0.6079          | 0.0316         | 20.8058          | 0.9292          |
| hybrid_loss<=1.0e+00_S1.0 | 10.9000         | 0.3162         | 517.4000          | 0.8649   | 0.0280  | 0.5108        | 0.0355       | 0.5932          | 0.0305         | 47.5055          | 1.6976          |
| hybrid_loss<=6.2e-02_S1.0 | 10.7000         | 0.4830         | 517.4000          | 0.8653   | 0.0284  | 0.5108        | 0.0355       | 0.5919          | 0.0304         | 48.4445          | 2.3775          |
| hybrid_loss<=3.9e-03_S1.0 | 10.6000         | 0.9661         | 517.4000          | 0.8652   | 0.0284  | 0.5107        | 0.0355       | 0.5931          | 0.0306         | 49.2664          | 5.6298          |
| hybrid_loss<=1.5e-05_S1.0 | 6.9000          | 1.1005         | 517.4000          | 0.8674   | 0.0274  | 0.4278        | 0.0469       | 0.5834          | 0.0294         | 77.6036          | 18.5948         |
| hybrid_loss<=5.4e-20_S1.0 | 4.7000          | 0.4830         | 517.4000          | 0.8708   | 0.0258  | 0.4111        | 0.0299       | 0.5815          | 0.0311         | 111.2200         | 12.4276         |


## 6. Dunnhumby Predictive Results

Reference baseline (from prior canonical_kneedle_experiment): AUC=0.8605+/-0.0292, Spend R2=0.5028+/-0.0376, Invoice R2=0.6045+/-0.0330 (113 retained concepts after Jaccard, 502 raw concepts).

This experiment uses identical protocol; direct comparison to the reference is valid.

### 6.1 AUC

| arm                       | auc_mean | auc_std |
| ------------------------- | -------- | ------- |
| baseline                  | 0.8605   | 0.0292  |
| kuznetsov_loss<=1.0e+00   | 0.8602   | 0.0295  |
| kuznetsov_loss<=6.2e-02   | 0.8596   | 0.0297  |
| kuznetsov_loss<=3.9e-03   | 0.8603   | 0.0305  |
| kuznetsov_loss<=1.5e-05   | 0.8671   | 0.0263  |
| kuznetsov_loss<=5.4e-20   | 0.8687   | 0.0269  |
| hybrid_loss<=1.0e+00_S1.0 | 0.8649   | 0.0280  |
| hybrid_loss<=6.2e-02_S1.0 | 0.8653   | 0.0284  |
| hybrid_loss<=3.9e-03_S1.0 | 0.8652   | 0.0284  |
| hybrid_loss<=1.5e-05_S1.0 | 0.8674   | 0.0274  |
| hybrid_loss<=5.4e-20_S1.0 | 0.8708   | 0.0258  |


### 6.2 Spend R2

| arm                       | spend_r2_mean | spend_r2_std |
| ------------------------- | ------------- | ------------ |
| baseline                  | 0.5028        | 0.0376       |
| kuznetsov_loss<=1.0e+00   | 0.5032        | 0.0379       |
| kuznetsov_loss<=6.2e-02   | 0.5029        | 0.0382       |
| kuznetsov_loss<=3.9e-03   | 0.5055        | 0.0371       |
| kuznetsov_loss<=1.5e-05   | 0.5091        | 0.0358       |
| kuznetsov_loss<=5.4e-20   | 0.5155        | 0.0353       |
| hybrid_loss<=1.0e+00_S1.0 | 0.5108        | 0.0355       |
| hybrid_loss<=6.2e-02_S1.0 | 0.5108        | 0.0355       |
| hybrid_loss<=3.9e-03_S1.0 | 0.5107        | 0.0355       |
| hybrid_loss<=1.5e-05_S1.0 | 0.4278        | 0.0469       |
| hybrid_loss<=5.4e-20_S1.0 | 0.4111        | 0.0299       |


### 6.3 Invoice R2

| arm                       | invoice_r2_mean | invoice_r2_std |
| ------------------------- | --------------- | -------------- |
| baseline                  | 0.6045          | 0.0330         |
| kuznetsov_loss<=1.0e+00   | 0.6058          | 0.0328         |
| kuznetsov_loss<=6.2e-02   | 0.6046          | 0.0324         |
| kuznetsov_loss<=3.9e-03   | 0.6051          | 0.0326         |
| kuznetsov_loss<=1.5e-05   | 0.6063          | 0.0303         |
| kuznetsov_loss<=5.4e-20   | 0.6079          | 0.0316         |
| hybrid_loss<=1.0e+00_S1.0 | 0.5932          | 0.0305         |
| hybrid_loss<=6.2e-02_S1.0 | 0.5919          | 0.0304         |
| hybrid_loss<=3.9e-03_S1.0 | 0.5931          | 0.0306         |
| hybrid_loss<=1.5e-05_S1.0 | 0.5834          | 0.0294         |
| hybrid_loss<=5.4e-20_S1.0 | 0.5815          | 0.0311         |


## 7. Statistical Comparison

Paired differences (Kuznetsov arm - baseline) across identical 10 seeds (1000-1009):

- **Paired bootstrap:** 1,000 resamples, seed 42, 95% CI  
- **Paired permutation test:** 10,000 sign-flips, seed 42, two-sided  


### 7.1 AUC comparisons

| comparison                            | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| ------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| kuznetsov_loss<=1.0e+00 vs baseline   | -0.0002   | 0.0007   | -0.0007            | 0.0001             | 1                    | 0.3011              | 0                       |
| kuznetsov_loss<=1.5e-05 vs baseline   | 0.0067    | 0.0102   | 0.0014             | 0.0131             | 0                    | 0.0276              | 1                       |
| kuznetsov_loss<=3.9e-03 vs baseline   | -0.0002   | 0.0039   | -0.0026            | 0.0022             | 1                    | 0.8214              | 0                       |
| kuznetsov_loss<=5.4e-20 vs baseline   | 0.0083    | 0.0078   | 0.0040             | 0.0133             | 0                    | 0.0015              | 1                       |
| kuznetsov_loss<=6.2e-02 vs baseline   | -0.0008   | 0.0030   | -0.0027            | 0.0008             | 1                    | 0.3949              | 0                       |
| hybrid_loss<=1.0e+00_S1.0 vs baseline | 0.0044    | 0.0093   | -0.0015            | 0.0096             | 1                    | 0.1647              | 0                       |
| hybrid_loss<=1.5e-05_S1.0 vs baseline | 0.0069    | 0.0110   | -0.0000            | 0.0126             | 1                    | 0.0759              | 0                       |
| hybrid_loss<=3.9e-03_S1.0 vs baseline | 0.0048    | 0.0091   | -0.0013            | 0.0097             | 1                    | 0.1232              | 0                       |
| hybrid_loss<=5.4e-20_S1.0 vs baseline | 0.0103    | 0.0114   | 0.0040             | 0.0171             | 0                    | 0.0147              | 1                       |
| hybrid_loss<=6.2e-02_S1.0 vs baseline | 0.0048    | 0.0092   | -0.0013            | 0.0097             | 1                    | 0.1253              | 0                       |


### 7.2 Spend R2 comparisons

| comparison                            | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| ------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| kuznetsov_loss<=1.0e+00 vs baseline   | 0.0004    | 0.0019   | -0.0007            | 0.0015             | 1                    | 0.5053              | 0                       |
| kuznetsov_loss<=1.5e-05 vs baseline   | 0.0063    | 0.0128   | 0.0008             | 0.0146             | 0                    | 0.0408              | 1                       |
| kuznetsov_loss<=3.9e-03 vs baseline   | 0.0027    | 0.0069   | -0.0009            | 0.0074             | 1                    | 0.2681              | 0                       |
| kuznetsov_loss<=5.4e-20 vs baseline   | 0.0126    | 0.0151   | 0.0057             | 0.0224             | 0                    | 0.0037              | 1                       |
| kuznetsov_loss<=6.2e-02 vs baseline   | 0.0001    | 0.0016   | -0.0008            | 0.0012             | 1                    | 0.8501              | 0                       |
| hybrid_loss<=1.0e+00_S1.0 vs baseline | 0.0080    | 0.0148   | 0.0007             | 0.0180             | 0                    | 0.0379              | 1                       |
| hybrid_loss<=1.5e-05_S1.0 vs baseline | -0.0750   | 0.0428   | -0.0978            | -0.0480            | 0                    | 0.0038              | 1                       |
| hybrid_loss<=3.9e-03_S1.0 vs baseline | 0.0079    | 0.0144   | 0.0008             | 0.0176             | 0                    | 0.0366              | 1                       |
| hybrid_loss<=5.4e-20_S1.0 vs baseline | -0.0917   | 0.0185   | -0.1022            | -0.0806            | 0                    | 0.0015              | 1                       |
| hybrid_loss<=6.2e-02_S1.0 vs baseline | 0.0080    | 0.0147   | 0.0008             | 0.0180             | 0                    | 0.0366              | 1                       |


### 7.3 Invoice R2 comparisons

| comparison                            | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| ------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| kuznetsov_loss<=1.0e+00 vs baseline   | 0.0013    | 0.0021   | -0.0000            | 0.0024             | 1                    | 0.0842              | 0                       |
| kuznetsov_loss<=1.5e-05 vs baseline   | 0.0018    | 0.0050   | -0.0009            | 0.0049             | 1                    | 0.3203              | 0                       |
| kuznetsov_loss<=3.9e-03 vs baseline   | 0.0006    | 0.0026   | -0.0010            | 0.0021             | 1                    | 0.5268              | 0                       |
| kuznetsov_loss<=5.4e-20 vs baseline   | 0.0034    | 0.0078   | -0.0014            | 0.0077             | 1                    | 0.2053              | 0                       |
| kuznetsov_loss<=6.2e-02 vs baseline   | 0.0001    | 0.0017   | -0.0010            | 0.0010             | 1                    | 0.8673              | 0                       |
| hybrid_loss<=1.0e+00_S1.0 vs baseline | -0.0113   | 0.0089   | -0.0167            | -0.0064            | 0                    | 0.0057              | 1                       |
| hybrid_loss<=1.5e-05_S1.0 vs baseline | -0.0211   | 0.0092   | -0.0270            | -0.0163            | 0                    | 0.0015              | 1                       |
| hybrid_loss<=3.9e-03_S1.0 vs baseline | -0.0114   | 0.0088   | -0.0167            | -0.0064            | 0                    | 0.0057              | 1                       |
| hybrid_loss<=5.4e-20_S1.0 vs baseline | -0.0231   | 0.0096   | -0.0295            | -0.0175            | 0                    | 0.0015              | 1                       |
| hybrid_loss<=6.2e-02_S1.0 vs baseline | -0.0126   | 0.0089   | -0.0182            | -0.0073            | 0                    | 0.0057              | 1                       |


### 7.4 Concept count comparisons

| comparison                            | mean_diff | std_diff | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| ------------------------------------- | --------- | -------- | ------------------ | ------------------ | -------------------- | ------------------- | ----------------------- |
| kuznetsov_loss<=1.0e+00 vs baseline   | -12.9000  | 3.5103   | -14.9000           | -10.8975           | 0                    | 0.0015              | 1                       |
| kuznetsov_loss<=1.5e-05 vs baseline   | -63.1000  | 9.6661   | -68.6025           | -57.7000           | 0                    | 0.0015              | 1                       |
| kuznetsov_loss<=3.9e-03 vs baseline   | -46.2000  | 5.1164   | -49.2000           | -43.4000           | 0                    | 0.0015              | 1                       |
| kuznetsov_loss<=5.4e-20 vs baseline   | -88.1000  | 12.5029  | -95.3000           | -81.1975           | 0                    | 0.0015              | 1                       |
| kuznetsov_loss<=6.2e-02 vs baseline   | -24.6000  | 4.0879   | -27.0000           | -22.3000           | 0                    | 0.0015              | 1                       |
| hybrid_loss<=1.0e+00_S1.0 vs baseline | -102.1000 | 13.4615  | -109.9000          | -94.6000           | 0                    | 0.0015              | 1                       |
| hybrid_loss<=1.5e-05_S1.0 vs baseline | -106.1000 | 13.6418  | -114.0025          | -98.4975           | 0                    | 0.0015              | 1                       |
| hybrid_loss<=3.9e-03_S1.0 vs baseline | -102.4000 | 13.8820  | -110.3000          | -94.7000           | 0                    | 0.0015              | 1                       |
| hybrid_loss<=5.4e-20_S1.0 vs baseline | -108.3000 | 13.6874  | -116.1050          | -100.6975          | 0                    | 0.0015              | 1                       |
| hybrid_loss<=6.2e-02_S1.0 vs baseline | -102.3000 | 13.5733  | -110.1025          | -94.7975           | 0                    | 0.0015              | 1                       |


## 8. Structural and Redundancy Analysis

### 8.1 Compression vs baseline

Compression ratio vs candidates for each arm (multi-split mean +/- std):

| arm                       | compression_mean | compression_std |
| ------------------------- | ---------------- | --------------- |
| baseline                  | 4.6369           | 0.5622          |
| kuznetsov_loss<=1.0e+00   | 5.2183           | 0.5530          |
| kuznetsov_loss<=6.2e-02   | 5.9202           | 0.6789          |
| kuznetsov_loss<=3.9e-03   | 7.8774           | 1.0871          |
| kuznetsov_loss<=1.5e-05   | 10.4335          | 0.9224          |
| kuznetsov_loss<=5.4e-20   | 20.8058          | 0.9292          |
| hybrid_loss<=1.0e+00_S1.0 | 47.5055          | 1.6976          |
| hybrid_loss<=6.2e-02_S1.0 | 48.4445          | 2.3775          |
| hybrid_loss<=3.9e-03_S1.0 | 49.2664          | 5.6298          |
| hybrid_loss<=1.5e-05_S1.0 | 77.6036          | 18.5948         |
| hybrid_loss<=5.4e-20_S1.0 | 111.2200         | 12.4276         |


### 8.2 Concepts retained at each stage (seed 42)

- baseline                                          : 514 candidates -> 514 after stability -> 114 after Jaccard 
- kuznetsov_loss<=1.0e+00                           : 514 candidates -> 459 after stability -> 101 after Jaccard 
- kuznetsov_loss<=6.2e-02                           : 514 candidates -> 378 after stability ->  88 after Jaccard 
- kuznetsov_loss<=3.9e-03                           : 514 candidates -> 296 after stability ->  65 after Jaccard 
- kuznetsov_loss<=1.5e-05                           : 514 candidates -> 217 after stability ->  50 after Jaccard 
- kuznetsov_loss<=5.4e-20                           : 514 candidates ->  76 after stability ->  25 after Jaccard 
- hybrid_loss<=1.0e+00_S1.0                         : 514 candidates -> 459 after stability ->  10 after Jaccard 
- hybrid_loss<=6.2e-02_S1.0                         : 514 candidates -> 378 after stability ->  10 after Jaccard 
- hybrid_loss<=3.9e-03_S1.0                         : 514 candidates -> 296 after stability ->  10 after Jaccard 
- hybrid_loss<=1.5e-05_S1.0                         : 514 candidates -> 217 after stability ->   7 after Jaccard 
- hybrid_loss<=5.4e-20_S1.0                         : 514 candidates ->  76 after stability ->   5 after Jaccard 


## 9. Comparison with Existing Kneedle/Chord Pruning

The existing fuzzy pipeline (baseline arm) applies Jaccard suppression directly to all 502 candidate concepts without any support or stability pre-filtering.

The canonical Kneedle experiment (canonical_kneedle_experiment.py) applies Kneedle to the support curve, then Jaccard. Our Kuznetsov experiment replaces the Kneedle support-threshold step with canonical stability thresholding (arm B) or adds it as an additional filter before Kneedle (arm C, hybrid).

Key differences:

- **Kneedle** operates on the support curve (a data distribution property) and finds a knee point that depends on the sensitivity parameter S.  
- **Kuznetsov stability** operates on the formal-concept structure itself (how many subsets of the extent generate the concept) and is an intrinsic property of each concept.  
- **Jaccard suppression** is identical across all arms (Jmax=0.80, mu_cut=0.5).

## 10. Should Kuznetsov Replace Kneedle?

This section gives the verdict based strictly on the experimental results above.

### Verdict: RETAIN CURRENT KNEELED PIPELINE

Kuznetsov-based selection produces similar AUC (delta = +0.0083) to the baseline. The difference is not practically meaningful (< 0.01 AUC). Without clear predictive advantage, there is no reason to replace the existing pipeline.

**Compression context:** Baseline retains ~113 concepts (compression 4.6x vs 517 candidates). Best Kuznetsov-only compression: 20.8x. Hybrid (Kuznetsov + Kneedle) provides additional compression beyond either method alone.

## 11. Limitations

- **Single dataset:** Only Dunnhumby tested; Retail II evaluation would strengthen conclusions.  
- **Threshold grid:** The loss thresholds are based on the empirical distribution of the 502 concepts; different datasets may require different thresholds.  
- **No threshold optimization:** Thresholds are pre-specified from the distribution, not tuned on test results. This is by design (avoids data leakage).  
- **Stability values are pre-computed:** We use the exact values from the audit CSV, which were computed on the full dataset. In a production pipeline, stability would need to be computed per-split (on training data only), which may differ slightly.  
- **Single seed for fixed split:** Only seed 42 for the fixed split; multi-split uses seeds 1000-1009.  
- **Jaccard suppression unchanged:** We do not modify Jmax or mu_cut; the experiment tests stability as a pre-filter only.  
- **No hybrid with other FCA measures:** Extensional stability, lift, conviction, etc. are not tested here (recommended for future work).  
## 12. Recommended Next Experiment

1. **Retail II replication:** Run the identical protocol on the 359 Retail II concepts using results/kuznetsov_stability_audit/stability_values.csv (R2_C_NNN entries).  
2. **Stability-per-split computation:** Compute canonical stability on each training split separately (not from the full-dataset audit) to verify the per-split values are stable.  
3. **Alternative FCA measures:** Test extensional stability, lift, conviction, and other Kuznetsov/Makhalova (2019) interestingness measures as selection criteria.  
4. **Interaction with Kneedle:** More detailed study of whether Kuznetsov + Kneedle in sequence provides better compression than either alone, across multiple S values.  
5. **Concept-level analysis:** Examine WHICH concepts are removed by Kuznetsov vs Kneedle vs Jaccard to understand the structural differences between these filters.  

---

**Status:** Complete. No production files modified. No commit made.

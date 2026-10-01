# Experiment 6: Controlled Representation Ablation Study on Olist

## 1. Executive Summary & Core Hypothesis

**Hypothesis Tested**: Do Fuzzy C-Means (FCM) soft-cluster memberships provide information complementary to the Fuzzy-FCA concept representation, such that a hybrid representation improves downstream predictive performance over its individual constituent representations?

### Key Findings:
1. **Fixed-Split Repurchase AUC**:
   - **A: Raw RFMS**: 0.5572
   - **B: Fuzzy-FCA**: 0.5548
   - **C: FCM (k=44)**: 0.5426
   - **D: Raw RFMS + FCM**: 0.5494
   - **E: Raw RFMS + Fuzzy-FCA**: 0.5506
   - **F: Hybrid (Raw + FCM + Fuzzy-FCA)**: 0.5511

2. **Complementarity Evaluation (Hybrid vs Constituents)**:
   - **F vs B (Hybrid vs Fuzzy-FCA)**: Δ AUC = -0.0038
   - **F vs E (Hybrid vs Raw+FCA)**: Δ AUC = +0.0005
   - **F vs A (Hybrid vs Raw RFMS)**: Δ AUC = -0.0061

## 2. Fixed-Split Six-Way Ablation Results

| arm                         | n_features | test_repurchase_auc | test_brier_score     | test_spend_r2          | test_spend_mae      | test_spend_spearman  | test_invoices_r2       | test_invoices_spearman |
| --------------------------- | ---------- | ------------------- | -------------------- | ---------------------- | ------------------- | -------------------- | ---------------------- | ---------------------- |
| A: Raw RFMS                 | 4          | 0.5571861173776255  | 0.024861523068996978 | 0.000942923586125799   | 0.237194200916639   | 0.033256581858002045 | 0.001008094253710401   | 0.03236007235130711    |
| B: Fuzzy-FCA                | 331        | 0.5548193722157337  | 0.024894531407762092 | 0.00040307684111628994 | 0.23736126890594414 | 0.03149576455077491  | 0.0004941585428520634  | 0.031890061802706794   |
| C: FCM                      | 44         | 0.5426137768630569  | 0.02488986996571552  | 0.00021323678946116864 | 0.23681053867719226 | 0.02352599514162136  | 0.00027682747743495995 | 0.02448901494649968    |
| D: Raw + FCM                | 48         | 0.5494301170580649  | 0.02487168725511653  | 0.0007539324817582571  | 0.23697987004971652 | 0.02906805455199119  | 0.0007589368718755596  | 0.028094061449676508   |
| E: Raw + Fuzzy-FCA          | 335        | 0.550554285344726   | 0.02489776073578963  | 0.0003558797973781669  | 0.23735412535223174 | 0.030552960967081325 | 0.0004437395621825768  | 0.030040482719968772   |
| F: Hybrid (Raw + FCM + FCA) | 379        | 0.5510631094000251  | 0.0248975122843782   | 0.0003632878298610587  | 0.23734177132459733 | 0.030725253259038715 | 0.0004482104843387402  | 0.03024032426139038    |

## 3. Feature Accounting

| arm                         | total_features | raw_features | fcm_features | fca_features | train_samples | test_samples |
| --------------------------- | -------------- | ------------ | ------------ | ------------ | ------------- | ------------ |
| A: Raw RFMS                 | 4              | 4            | 0            | 0            | 15164         | 6500         |
| B: Fuzzy-FCA                | 331            | 0            | 0            | 331          | 15164         | 6500         |
| C: FCM                      | 44             | 0            | 44           | 0            | 15164         | 6500         |
| D: Raw + FCM                | 48             | 4            | 44           | 0            | 15164         | 6500         |
| E: Raw + Fuzzy-FCA          | 335            | 4            | 0            | 331          | 15164         | 6500         |
| F: Hybrid (Raw + FCM + FCA) | 379            | 4            | 44           | 331          | 15164         | 6500         |

## 4. Paired Bootstrap Statistical Comparison (Fixed Split, N=1,000)

| comparison                                        | metric               | point_estimate          | ci_lower_95             | ci_upper_95            | spans_zero | exceeds_decision_threshold |
| ------------------------------------------------- | -------------------- | ----------------------- | ----------------------- | ---------------------- | ---------- | -------------------------- |
| C: FCM vs B: Fuzzy-FCA                            | Delta Repurchase AUC | -0.012205595352676846   | -0.049297267081205104   | 0.023620078547153523   | True       | False                      |
| C: FCM vs B: Fuzzy-FCA                            | Delta Spend R2       | -0.0001898400516551213  | -0.0017079091212202856  | 0.0014701796945867011  | True       | False                      |
| C: FCM vs B: Fuzzy-FCA                            | Delta Invoices R2    | -0.00021733106541710345 | -0.0016499483836517198  | 0.001366085179655005   | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs B: Fuzzy-FCA       | Delta Repurchase AUC | -0.003756262815708622   | -0.010672994833764039   | 0.0028802192628185324  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs B: Fuzzy-FCA       | Delta Spend R2       | -3.9789011255231266e-05 | -0.0003326644721137029  | 0.00023059532061094342 | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs B: Fuzzy-FCA       | Delta Invoices R2    | -4.5948058513323176e-05 | -0.0003774763398586761  | 0.0002383449082331178  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs C: FCM             | Delta Repurchase AUC | 0.008449332536968224    | -0.02642863532871815    | 0.045024938386189775   | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs C: FCM             | Delta Spend R2       | 0.00015005104039989003  | -0.0014852228577405674  | 0.00161731897664866    | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs C: FCM             | Delta Invoices R2    | 0.00017138300690378028  | -0.0014119487499858168  | 0.0015654068674773557  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs A: Raw RFMS        | Delta Repurchase AUC | -0.006123007977600348   | -0.039415938123471876   | 0.027927992639008338   | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs A: Raw RFMS        | Delta Spend R2       | -0.0005796357562647403  | -0.0017211798931959677  | 0.000644271337598265   | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs A: Raw RFMS        | Delta Invoices R2    | -0.0005598837693716607  | -0.0017806688355143513  | 0.0006007294862959699  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs E: Raw + Fuzzy-FCA | Delta Repurchase AUC | 0.000508824055299173    | -0.0010584106812614458  | 0.002322220110039752   | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs E: Raw + Fuzzy-FCA | Delta Spend R2       | 7.408032482891791e-06   | -2.2958249757931702e-05 | 3.923166461727855e-05  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs E: Raw + Fuzzy-FCA | Delta Invoices R2    | 4.470922156163404e-06   | -2.2971580801522952e-05 | 3.306652315933078e-05  | True       | False                      |
| B: Fuzzy-FCA vs A: Raw RFMS                       | Delta Repurchase AUC | -0.0023667451618917257  | -0.037856253893527134   | 0.03405654405121319    | True       | False                      |
| B: Fuzzy-FCA vs A: Raw RFMS                       | Delta Spend R2       | -0.000539846745009509   | -0.0017501267397682986  | 0.0007029351118680995  | True       | False                      |
| B: Fuzzy-FCA vs A: Raw RFMS                       | Delta Invoices R2    | -0.0005139357108583376  | -0.0018126848420837837  | 0.0006938427329549438  | True       | False                      |
| C: FCM vs A: Raw RFMS                             | Delta Repurchase AUC | -0.014572340514568571   | -0.05547610380538965    | 0.02926865607962619    | True       | False                      |
| C: FCM vs A: Raw RFMS                             | Delta Spend R2       | -0.0007296867966646303  | -0.002505256316238666   | 0.0010880603126245806  | True       | False                      |
| C: FCM vs A: Raw RFMS                             | Delta Invoices R2    | -0.000731266776275441   | -0.0023047392027659708  | 0.00082463686499069    | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs D: Raw + FCM       | Delta Repurchase AUC | 0.0016329923419602377   | -0.023660997881529405   | 0.02867314864068826    | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs D: Raw + FCM       | Delta Spend R2       | -0.00039064465189719844 | -0.0014347260173194736  | 0.0006627550446926531  | True       | False                      |
| F: Hybrid (Raw + FCM + FCA) vs D: Raw + FCM       | Delta Invoices R2    | -0.00031072638753681936 | -0.001403491169094287   | 0.0007579963417339545  | True       | False                      |

## 5. Complementarity & Representation Redundancy Analysis

- **Mean absolute cross-correlation between all FCM cluster memberships and Fuzzy-FCA concepts**: 0.0886
- **Maximum cross-correlation across any pair**: 0.8358
- **Mean peak correlation per FCM cluster**: 0.5006
- **FCM clusters with peak correlation > 0.5 with any Fuzzy-FCA concept**: 28/44 (63.6%)
- **FCM clusters with peak correlation > 0.8 with any Fuzzy-FCA concept**: 3/44 (6.8%)
- **Average variance of FCM membership explained linearly by all Fuzzy-FCA concepts ($R^2$)**: 0.4983
- **Variance of Raw RFMS explained by FCA vs FCM vs Both**: FCA=0.8997, FCM=0.9427, Both=0.9913

## 6. Multi-Split Validation Across 10 Stratified Splits

| arm                         | n_features_mean | test_auc_mean      | test_auc_sd          | test_spend_r2_mean    | test_spend_r2_sd      | test_invoices_r2_mean | test_invoices_r2_sd   | wins_vs_raw_auc | wins_vs_fuzzy_auc | wins_vs_fcm_auc |
| --------------------------- | --------------- | ------------------ | -------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------- | ----------------- | --------------- |
| A: Raw RFMS                 | 4.0             | 0.5587348446517362 | 0.010673718008979654 | 0.0007197970151939459 | 0.0005823986150234257 | 0.0008998509969074031 | 0.0005758823428984972 | 0               | 4                 | 6               |
| B: Fuzzy-FCA                | 339.7           | 0.5611929403753315 | 0.012453160581892778 | 0.0010276990976394672 | 0.0004982318358008172 | 0.0011164369902081027 | 0.0005715168019827881 | 6               | 0                 | 8               |
| C: FCM                      | 44.0            | 0.5527533563366189 | 0.012531482712627974 | 0.0005372633443241903 | 0.00109804348404891   | 0.0004215069309479014 | 0.0012774180189330296 | 4               | 2                 | 0               |
| D: Raw + FCM                | 48.0            | 0.5593581778963026 | 0.010251541060718088 | 0.0006231396379045128 | 0.0010856839929282095 | 0.000689306833826786  | 0.0010499787459069278 | 8               | 5                 | 7               |
| E: Raw + Fuzzy-FCA          | 343.7           | 0.5614773587561487 | 0.010419529621197802 | 0.0010010152145820018 | 0.0005299711606007258 | 0.0010799416638796245 | 0.0006583905551894575 | 6               | 6                 | 9               |
| F: Hybrid (Raw + FCM + FCA) | 387.7           | 0.5617035239156817 | 0.010501363188144971 | 0.001008045976564631  | 0.0005305025674164134 | 0.0010834427460304275 | 0.0006598361971263868 | 6               | 6                 | 9               |

## 7. Discussion & Methodological Assessment

Does adding FCM membership information to Fuzzy-FCA improve predictive performance?
- **Conclusion**: The hybrid representation does NOT meaningfully or statistically improve over the standalone representations. Adding FCM soft memberships to Fuzzy-FCA provides no substantial predictive advantage, as indicated by overlapping bootstrap confidence intervals that span zero.

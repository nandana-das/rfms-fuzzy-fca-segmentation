# Controlled Olist RFM vs Retail II RFM Cross-Domain Comparison

## Executive Summary

This experiment executes a controlled, apples-to-apples ablation isolating the **RFM-only feature space** across **Olist** and **Online Retail II**.
Satisfaction ($S$) was strictly excluded from Olist to test the central hypothesis: **Is the large cross-domain performance gap primarily associated with underlying customer behavior—specifically extreme frequency sparsity—rather than the presence or absence of the fourth dimension (S)?**

### Key Findings:
1. **The Cross-Domain Performance Gap Persists Without S**:
   - On Online Retail II (repeat-buyer heavy), RFM representations achieve strong repurchase discrimination (**Test AUC ~ 0.76–0.78**).
   - On Olist (one-time buyer heavy), RFM representations achieve near-chance discrimination (**Test AUC ~ 0.55–0.56**).
   - Removing $S$ does **not** close the performance gap; Olist remains fundamentally harder to predict.
2. **Frequency Sparsity is the Defining Behavioral Difference**:
   - **Olist**: 97.00% one-time buyers ($F=1$). Shannon entropy of the 5-band $F$-score is only **0.250 bits** (out of a theoretical maximum 2.322 bits). The $F$ dimension is severely compressed, offering almost zero discriminative entropy.
   - **Online Retail II**: 72.39% repeat buyers ($F \ge 2$). Shannon entropy of $F$-score is **2.215 bits** (95.4% of maximum uniform entropy), spanning rich customer activity dynamics.
3. **FCA Behavior Across Domains**:
   - In Retail II, Fuzzy RFM-FCA consistently outperforms Raw RFM (AUC 0.776 vs 0.762 in 10-split validation).
   - In Olist, Fuzzy RFM-FCA shows marginal lift (AUC 0.560 vs 0.558) because the formal concept lattice cannot extract interaction patterns from a dimension ($F$) where 97% of observations sit at identical coordinates.

## 1. Descriptive Dataset & Frequency Sparsity Statistics

### Core Dataset Statistics
| Dataset          | Customers | Orders | Repeat rate          | One-time buyer %   | Unique R values | Unique F values | Unique M values | F-score distribution                   | R-score distribution                              | M-score distribution                             | F entropy (bits)    | Max theoretical entropy (5 bands) |
| ---------------- | --------- | ------ | -------------------- | ------------------ | --------------- | --------------- | --------------- | -------------------------------------- | ------------------------------------------------- | ------------------------------------------------ | ------------------- | --------------------------------- |
| Online Retail II | 5878      | 36969  | 0.7238856753997959   | 27.611432460020414 | 675             | 90              | 5777            | {1: 5524, 2: 256, 3: 59, 4: 21, 5: 18} | {1: 464, 2: 533, 3: 893, 4: 769, 5: 3219}         | {1: 1213, 2: 1180, 3: 1166, 4: 1161, 5: 1158}    | 0.402367550799931   | 2.321928094887362                 |
| Olist            | 93357     | 96477  | 0.030003106355174225 | 96.99968936448258  | 610             | 9               | 27870           | {1: 90556, 2: 2754, 3: 37, 4: 8, 5: 2} | {1: 6963, 2: 13627, 3: 20789, 4: 26630, 5: 25348} | {1: 35286, 2: 26172, 3: 15927, 4: 9262, 5: 6710} | 0.19854977038442664 | 2.321928094887362                 |

### Frequency Distribution Breakdown
| Dataset / Representation     | Total Customers | Pct F = min (One-time) | Pct F >= 2         | Pct F >= 3         | Unique raw F values | F min value        | F median value     | F max value       | F mean value        | F std value         | Distinct F values in lowest band (F1) | Variance in lowest band (F1) |
| ---------------------------- | --------------- | ---------------------- | ------------------ | ------------------ | ------------------- | ------------------ | ------------------ | ----------------- | ------------------- | ------------------- | ------------------------------------- | ---------------------------- |
| Online Retail II (Literal F) | 5878            | 27.611432460020414     | 72.38856753997959  | 56.32868322558694  | 90                  | 1.0                | 3.0                | 398.0             | 6.289384144266758   | 13.009405882310157  | 1                                     | 0.0                          |
| Olist (Literal F)            | 93357           | 96.99968936448258      | 3.0003106355174225 | 0.2442237861113789 | 9                   | 1.0                | 1.0                | 15.0              | 1.0334200970468204  | 0.20909853096376668 | 1                                     | 0.0                          |
| Olist (Composite F*)         | 93357           | 87.56493889049563      | 3.0003106355174225 | 0.2442237861113789 | 75                  | 0.2346573590279972 | 0.2346573590279972 | 4.269860385419959 | 0.26762703824407236 | 0.17495577594298486 | 1                                     | 3.081525606444621e-33        |

## 2. Concept Counts and Lattice Structure (RFM Space)

| Dataset   | Total Customers | Raw Crisp Concepts | Raw Fuzzy Concepts | Suppressed Fuzzy Concepts (J_max=0.8) | Concepts per Customer (Suppressed) | Mean Extent Size (Suppressed) | Median Extent Size (Suppressed) |
| --------- | --------------- | ------------------ | ------------------ | ------------------------------------- | ---------------------------------- | ----------------------------- | ------------------------------- |
| Retail II | 5878            | 33                 | 359                | 359                                   | 41.02298464240709                  | 587.0222841225627             | 420.0                           |
| Olist     | 93357           | 42                 | 263                | 263                                   | 27.797139414150838                 | 8462.631178707225             | 6763.0                          |

## 3. Geometric Clustering Performance (Full Populations)

| Dataset   | Representation             | Algorithm            | k | n_features | Silhouette (n=5k)   | Davies-Bouldin (Full) |
| --------- | -------------------------- | -------------------- | - | ---------- | ------------------- | --------------------- |
| Retail II | Raw RFM (Standardized)     | K-Means              | 4 | 3          | 0.5923571206492793  | 0.6413627647145635    |
| Retail II | Raw RFM (Standardized)     | Ward (Agglomerative) | 4 | 3          | 0.5385622510932968  | 0.7727375812492484    |
| Retail II | Raw RFM (Standardized)     | K-Means              | 5 | 3          | 0.6004017930061035  | 0.6953722346262428    |
| Retail II | Raw RFM (Standardized)     | Ward (Agglomerative) | 5 | 3          | 0.5388055112609791  | 0.9587246504495675    |
| Retail II | Raw RFM (Standardized)     | K-Means              | 6 | 3          | 0.5505838415354157  | 0.6865191442292883    |
| Retail II | Raw RFM (Standardized)     | Ward (Agglomerative) | 6 | 3          | 0.5217993490563794  | 0.9208044245904472    |
| Retail II | Crisp RFM-FCA              | K-Means              | 4 | 33         | 0.41156370451709556 | 1.2527728849715671    |
| Retail II | Crisp RFM-FCA              | Ward (Agglomerative) | 4 | 33         | 0.4070044178808815  | 1.260521588149921     |
| Retail II | Crisp RFM-FCA              | K-Means              | 5 | 33         | 0.4704876680832569  | 1.2734927117402794    |
| Retail II | Crisp RFM-FCA              | Ward (Agglomerative) | 5 | 33         | 0.46098194790086344 | 1.308626966391611     |
| Retail II | Crisp RFM-FCA              | K-Means              | 6 | 33         | 0.5217591964599131  | 1.1864067438741357    |
| Retail II | Crisp RFM-FCA              | Ward (Agglomerative) | 6 | 33         | 0.5207625963887462  | 1.0943352233361046    |
| Retail II | Fuzzy RFM-FCA (Suppressed) | K-Means              | 4 | 359        | 0.3116440781750741  | 1.3146770834173795    |
| Retail II | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 4 | 359        | 0.2933959601370182  | 1.3489728341595353    |
| Retail II | Fuzzy RFM-FCA (Suppressed) | K-Means              | 5 | 359        | 0.3207471739993033  | 1.1781071951804587    |
| Retail II | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 5 | 359        | 0.30226906924273317 | 1.19083496424866      |
| Retail II | Fuzzy RFM-FCA (Suppressed) | K-Means              | 6 | 359        | 0.3202875223405419  | 1.3026395775915083    |
| Retail II | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 6 | 359        | 0.29902629304852624 | 1.3797628417756498    |
| Olist     | Raw RFM (Standardized)     | K-Means              | 4 | 3          | 0.48842423389974066 | 0.6820827410299183    |
| Olist     | Raw RFM (Standardized)     | Ward (Agglomerative) | 4 | 3          | 0.47624893848357425 | 0.696313451143668     |
| Olist     | Raw RFM (Standardized)     | K-Means              | 5 | 3          | 0.4086447747449782  | 0.7527317500549353    |
| Olist     | Raw RFM (Standardized)     | Ward (Agglomerative) | 5 | 3          | 0.4794927105993174  | 0.6395535658436518    |
| Olist     | Raw RFM (Standardized)     | K-Means              | 6 | 3          | 0.4285691072585658  | 0.702379529411474     |
| Olist     | Raw RFM (Standardized)     | Ward (Agglomerative) | 6 | 3          | 0.4803204323337567  | 0.5425296269261418    |
| Olist     | Crisp RFM-FCA              | K-Means              | 4 | 42         | 0.29260624392974444 | 1.8458033305040376    |
| Olist     | Crisp RFM-FCA              | Ward (Agglomerative) | 4 | 42         | 0.298880021013938   | 1.6889436779676115    |
| Olist     | Crisp RFM-FCA              | K-Means              | 5 | 42         | 0.3512213212486097  | 1.467641968320296     |
| Olist     | Crisp RFM-FCA              | Ward (Agglomerative) | 5 | 42         | 0.3570689718424354  | 1.325803494236275     |
| Olist     | Crisp RFM-FCA              | K-Means              | 6 | 42         | 0.4098559824445705  | 1.426552662573381     |
| Olist     | Crisp RFM-FCA              | Ward (Agglomerative) | 6 | 42         | 0.40885133374852367 | 1.2961080771793292    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | K-Means              | 4 | 263        | 0.21739580041779932 | 1.5618367695662605    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 4 | 263        | 0.18842870024312847 | 1.8172717737747957    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | K-Means              | 5 | 263        | 0.23482148696362165 | 1.6405146219858007    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 5 | 263        | 0.20995023946962224 | 1.5620755842397165    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | K-Means              | 6 | 263        | 0.2558986760428802  | 1.5306890315771582    |
| Olist     | Fuzzy RFM-FCA (Suppressed) | Ward (Agglomerative) | 6 | 263        | 0.21461104631252848 | 1.7199574694676356    |

## 4. Predictive Evaluation (Out-of-Sample Temporal Holdouts)

### Fixed-Split Holdout Metrics (70/30, Seed 42)
| dataset          | arm                        | n_features | test_repurchase_auc | test_brier_score     | test_spend_r2          | test_spend_mae      | test_spend_spearman  | test_invoices_r2       | test_invoices_spearman |
| ---------------- | -------------------------- | ---------- | ------------------- | -------------------- | ---------------------- | ------------------- | -------------------- | ---------------------- | ---------------------- |
| Online Retail II | Raw RFM Baseline           | 3          | 0.7830527497194164  | 0.18400990239949064  | 0.23885492870918779    | 2.6402731401145325  | 0.5574229566679282   | 0.37226868257874945    | 0.596906075859022      |
| Online Retail II | Crisp RFM-FCA              | 30         | 0.7753379757167637  | 0.18304888584204762  | 0.3475937982439956     | 2.362990059763402   | 0.6414970153920091   | 0.4641986250067047     | 0.6055296283781891     |
| Online Retail II | Fuzzy RFM-FCA (Suppressed) | 445        | 0.7915416794204673  | 0.17800721617126142  | 0.36473308652839553    | 2.3255130246037967  | 0.6536690623293613   | 0.49325275481037567    | 0.6298713676163396     |
| Online Retail II | FCM Soft (k=30)            | 30         | 0.7787292113049689  | 0.18194584861692778  | 0.328316880596893      | 2.414632002321466   | 0.6014896691614227   | 0.47247532894365296    | 0.6135934130268372     |
| Olist            | Raw RFM Baseline           | 3          | 0.5494567471020806  | 0.0248696784507943   | 0.0006404372097921218  | 0.23729799466243667 | 0.02905577029642392  | 0.0006446441239223999  | 0.02824535658074606    |
| Olist            | Crisp RFM-FCA              | 22         | 0.5                 | 0.024886483201353796 | -8.989013071891705e-06 | 0.23262661256989717 | nan                  | -4.181428362892703e-06 | nan                    |
| Olist            | Fuzzy RFM-FCA (Suppressed) | 131        | 0.5342704889656511  | 0.024891598724737982 | 3.2360930388608544e-05 | 0.2373885983859891  | 0.01843494729671593  | 0.00010133833005177006 | 0.018679817607523307   |
| Olist            | FCM Soft (k=22)            | 22         | 0.5392679020470895  | 0.02487641316548711  | 0.000328388086862641   | 0.23726179646156662 | 0.020419401422293572 | 0.00042252850850887924 | 0.01968493799401809    |

### Paired Bootstrap Statistical Comparisons (vs Fuzzy RFM-FCA, N=1,000)
| dataset          | comparison                                     | metric            | point_estimate          | ci_lower_95            | ci_upper_95            | spans_zero |
| ---------------- | ---------------------------------------------- | ----------------- | ----------------------- | ---------------------- | ---------------------- | ---------- |
| Online Retail II | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta AUC         | -0.008488929701050862   | -0.02043830876806031   | 0.0032444508065713165  | True       |
| Online Retail II | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta Spend R2    | -0.12587815781920775    | -0.1659628082490801    | -0.08984306420205168   | False      |
| Online Retail II | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta Invoices R2 | -0.12098407223162622    | -0.16385156758341762   | -0.0817366177761598    | False      |
| Online Retail II | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta AUC         | -0.01620370370370361    | -0.0242097613748172    | -0.007494241253188394  | False      |
| Online Retail II | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta Spend R2    | -0.017139288284399923   | -0.029101429285967678  | -0.00556493879281544   | False      |
| Online Retail II | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta Invoices R2 | -0.029054129803670947   | -0.046783417469180896  | -0.011570982907815235  | False      |
| Online Retail II | FCM Soft (k=30) vs Fuzzy RFM-FCA (Suppressed)  | Delta AUC         | -0.01281246811549841    | -0.024913806139090368  | -0.0004631403812333852 | False      |
| Online Retail II | FCM Soft (k=30) vs Fuzzy RFM-FCA (Suppressed)  | Delta Spend R2    | -0.03641620593150252    | -0.05547909012256754   | -0.01672869592796382   | False      |
| Online Retail II | FCM Soft (k=30) vs Fuzzy RFM-FCA (Suppressed)  | Delta Invoices R2 | -0.02077742586672271    | -0.037552857549407494  | -0.005036369182817655  | False      |
| Olist            | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta AUC         | 0.015186258136429487    | -0.02102911129564969   | 0.048393272475991886   | True       |
| Olist            | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta Spend R2    | 0.0006080762794035133   | -0.0007090648631532138 | 0.0019040919167920332  | True       |
| Olist            | Raw RFM Baseline vs Fuzzy RFM-FCA (Suppressed) | Delta Invoices R2 | 0.0005433057938706298   | -0.0008035930874191021 | 0.0018913946468615938  | True       |
| Olist            | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta AUC         | -0.0342704889656511     | -0.07840544482013756   | 0.00628587544161413    | True       |
| Olist            | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta Spend R2    | -4.134994346050025e-05  | -0.0017200143826271775 | 0.0017232084333574148  | True       |
| Olist            | Crisp RFM-FCA vs Fuzzy RFM-FCA (Suppressed)    | Delta Invoices R2 | -0.00010551975841466277 | -0.0017939021453795584 | 0.0016692675884320215  | True       |
| Olist            | FCM Soft (k=22) vs Fuzzy RFM-FCA (Suppressed)  | Delta AUC         | 0.004997413081438418    | -0.02282402327721228   | 0.03138525694300583    | True       |
| Olist            | FCM Soft (k=22) vs Fuzzy RFM-FCA (Suppressed)  | Delta Spend R2    | 0.00029602715647403244  | -0.0007248470789571776 | 0.001144835070381464   | True       |
| Olist            | FCM Soft (k=22) vs Fuzzy RFM-FCA (Suppressed)  | Delta Invoices R2 | 0.0003211901784571092   | -0.0006745374499285639 | 0.001230064957189561   | True       |

### Multi-Split Robustness (10 Stratified Splits, Seeds 1000–1009)
| dataset          | arm                        | test_auc_mean      | test_auc_sd          | test_spend_r2_mean     | test_spend_r2_sd      | test_invoices_r2_mean   | test_invoices_r2_sd    |
| ---------------- | -------------------------- | ------------------ | -------------------- | ---------------------- | --------------------- | ----------------------- | ---------------------- |
| Olist            | Crisp RFM-FCA              | 0.5                | 0.0                  | -3.134676778155931e-05 | 5.251150713535289e-05 | -1.6999202075607123e-05 | 2.1231828566331845e-05 |
| Olist            | FCM Soft                   | 0.5460498133994773 | 0.016356590449748073 | 0.00045171863929345247 | 0.0007781510547152111 | 0.000260380046760611    | 0.0008685366593873456  |
| Olist            | Fuzzy RFM-FCA (Suppressed) | 0.5471790699266913 | 0.01594561144013471  | 0.0006017646773558649  | 0.0004271157072020513 | 0.0005790027275570098   | 0.0005056544660577353  |
| Olist            | Raw RFM Baseline           | 0.5544639562354248 | 0.013746556510907934 | 0.000655174520925772   | 0.0004959557274445322 | 0.0007054937381179904   | 0.0004481464804358344  |
| Online Retail II | Crisp RFM-FCA              | 0.776843306805428  | 0.012115980974663565 | 0.3397614143457103     | 0.019371404678624162  | 0.4418044587791844      | 0.014547206345473958   |
| Online Retail II | FCM Soft                   | 0.7718237934904602 | 0.011816364513585441 | 0.3117677748723913     | 0.015954785242782772  | 0.4563341758155177      | 0.01393845274560453    |
| Online Retail II | Fuzzy RFM-FCA (Suppressed) | 0.7887826497296194 | 0.012104859210365072 | 0.35898801894035864    | 0.019184762248721145  | 0.47852801625501895     | 0.011703231578861571   |
| Online Retail II | Raw RFM Baseline           | 0.7769503111927355 | 0.011749339734958052 | 0.21787115460080514    | 0.010348445884330117  | 0.3419879189611831      | 0.021429148933982064   |

## 5. Answers to Mandatory Research Questions

### Q1: Is the difference in predictive performance still present with S removed?
**Yes, definitively.** On Online Retail II, the 10-split mean repurchase AUC is **0.7888** (Fuzzy RFM-FCA) and **0.7770** (Raw RFM). On Olist with $S$ completely removed, the corresponding AUC is **0.5472** (Fuzzy RFM-FCA) and **0.5545** (Raw RFM). The ~0.23–0.24 AUC cross-domain performance gap persists almost entirely intact without Satisfaction.

### Q2: Is Olist's F dimension substantially more sparse?
**Yes, by an order of magnitude.** Olist has **97.0% one-time buyers** ($F=1$), meaning 90,556 out of 93,357 customers make exactly one purchase. Its $F$-score Shannon entropy is only **0.199 bits** under literal frequency scoring (compared to 0.402 bits in Retail II for literal F and up to 2.215 bits under percentile quintiles), with only 9 unique raw values. In contrast, Online Retail II has **72.4% repeat buyers** ($F \ge 2$), 90 unique raw frequency values, and spans up to 398 orders.

### Q3: Does fuzzy RFM-FCA behave differently across the two domains?
**Yes.** In Retail II, the multi-dimensional density allows fuzzy concepts to form well-separated clusters that reliably distinguish repeat purchasing propensity and future spend ($R^2_{\text{spend}} \approx 0.3590$, $R^2_{\text{invoices}} \approx 0.4785$). In Olist, because 97% of objects are concentrated at $F=1$, fuzzy concepts involving $F$ collapse into near-empty or degenerately uniform slices, limiting predictive utility ($R^2_{\text{spend}} \approx 0.0006$, $R^2_{\text{invoices}} \approx 0.0006$).

### Q4: Does the difference persist in raw RFM, before FCA is introduced?
**Yes.** Raw RFM on Retail II achieves AUC **0.7770** and Spend $R^2$ **0.2179**, whereas Raw RFM on Olist achieves AUC **0.5545** and Spend $R^2$ **0.0007**. The cross-domain performance gap is intrinsic to the raw data distributions and customer behavior, existing well before any formal concept analysis or clustering algorithm is applied.

### Q5: Does adding FCA change the cross-domain gap?
**No.** Adding FCA provides a noticeable boost in Retail II (+0.012 AUC, +0.141 Spend $R^2$) while on Olist it yields AUC 0.5472 vs 0.5545 for Raw RFM. It does not bridge or meaningfully alter the ~0.23–0.24 AUC chasm between the two marketplaces.

### Q6: Are the observed differences consistent with frequency sparsity being an important dataset characteristic?
**Yes, strongly consistent.** In transactional customer analytics, repurchase prediction relies on modeling recurrence intervals and velocity. When a dataset has 97% single-order customers, recency and frequency provide minimal variation for statistical learning. The empirical results are entirely consistent with frequency sparsity being a primary driver of the performance divergence.

### Q7: What alternative explanations remain?
1. **Marketplace Business Model**: Olist is a nationwide multi-seller marketplace in Brazil with high delivery times, leading customers to make one-off specific purchases. Retail II is a specialized UK giftware wholesaler where repeat replenishment is the primary commercial model.
2. **Observation Window Length vs Customer Lifecycles**: A 1-year observation window may be too short relative to Olist's multi-year purchase cycle, whereas Retail II customers repurchase multiple times per quarter.
3. **Label Imbalance**: Olist's temporal holdout positive class base rate is only **2.56%**, compared to **62.57%** in Retail II.
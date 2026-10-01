# Dunnhumby Complete Journey: RFM-FCA Cross-Domain Validation

## Executive Summary

Rigorous evaluation of the **Fuzzy RFM-FCA framework** on **Dunnhumby 'The Complete Journey'** as the second primary validation domain alongside Online Retail II.
**RFM only** — Satisfaction (S) is entirely excluded. Olist is out of scope.

**Methodology Notes:**
- Stability metric is a **proxy** only — NOT canonical Kuznetsov stability.
- Kneedle threshold is **Kneedle-inspired** (normalised max-chord-distance) — NOT canonical Kneedle.
- **FPC/Xie-Beni** computed for canonical FCM **only** — not applied to FCA clusters.
- FCA hard clusters (Top-k Membership Hardening and Natural Hardening) evaluated with Silhouette and Davies-Bouldin only.
- Alpha-cut (alpha >= 0.5) is preserved for extent thresholding and redundancy suppression, distinct from Top-k Membership Hardening.
- `max_len = None` — no artificial itemset length cap.

## 1. Data Audit

- **n_rows**: 2595732
- **n_households**: 2500
- **n_baskets**: 276484
- **n_unique_days**: 711
- **day_min**: 1
- **day_max**: 711
- **n_products**: 92339
- **n_missing_values**: 0
- **n_duplicate_rows**: 0
- **hh_with_1_basket**: 3
- **hh_with_2plus_baskets**: 2497
- **hh_with_3plus_baskets**: 2492
- **n_distinct_basket_counts**: 395
- **mean_F_full**: 110.5936
- **median_F_full**: 79.0
- **std_F_full**: 115.66936759609275
- **max_F_full**: 1300.0
- **multi_row_baskets_confirms_product_line_granularity**: 216439

## 2. RFM Distributional Statistics

| Cohort                        | Metric          | Count | Mean               | Std                | Min  | P25    | P50_Median | P75               | P90                | P95                | Max      |
| ----------------------------- | --------------- | ----- | ------------------ | ------------------ | ---- | ------ | ---------- | ----------------- | ------------------ | ------------------ | -------- |
| Full Population (711 Days)    | R               | 2500  | 25.574             | 62.778095893392624 | 0.0  | 1.0    | 6.0        | 20.0              | 62.0               | 117.04999999999971 | 657.0    |
| Full Population (711 Days)    | F               | 2500  | 110.5936           | 115.64623140872338 | 1.0  | 39.0   | 79.0       | 142.25            | 229.0999999999999  | 321.0              | 1300.0   |
| Full Population (711 Days)    | M               | 2500  | 3222.985232        | 3348.356204210189  | 8.17 | 970.74 | 2157.75    | 4413.320000000001 | 7448.2159999999985 | 9754.144499999986  | 38319.79 |
| Observation Cohort (620 Days) | R               | 2499  | 26.726690676270508 | 67.3946596397069   | 0.0  | 1.0    | 6.0        | 19.0              | 63.0               | 129.19999999999982 | 590.0    |
| Observation Cohort (620 Days) | F               | 2499  | 95.5874349739896   | 102.0909872910322  | 1.0  | 33.0   | 67.0       | 123.0             | 202.0              | 277.0999999999999  | 1169.0   |
| Observation Cohort (620 Days) | M               | 2499  | 2742.177739095638  | 2874.5945380618523 | 4.49 | 790.9  | 1823.7     | 3739.695          | 6385.7860000000055 | 8339.915999999992  | 31548.37 |
| Future Holdout (91 Days)      | future_spend    | 2499  | 482.0546418567427  | 558.4589357068343  | 0.0  | 99.42  | 296.79     | 669.37            | 1175.146000000001  | 1554.7759999999998 | 6771.42  |
| Future Holdout (91 Days)      | future_invoices | 2499  | 15.04921968787515  | 17.076352629264957 | 0.0  | 4.0    | 10.0       | 20.0              | 33.0               | 47.09999999999991  | 169.0    |

## 3. Dataset Summary

| Dataset                    | Customers | Orders | Repeat rate        | One-time buyer %   | Unique R values | Unique F values | Unique M values | F-score distribution                     | R-score distribution                   | M-score distribution                     | F entropy (bits)   | Max theoretical entropy (5 bands) |
| -------------------------- | --------- | ------ | ------------------ | ------------------ | --------------- | --------------- | --------------- | ---------------------------------------- | -------------------------------------- | ---------------------------------------- | ------------------ | --------------------------------- |
| Dunnhumby Complete Journey | 2499      | 238873 | 0.9967987194877952 | 0.3201280512204882 | 217             | 365             | 2495            | {1: 1337, 2: 694, 3: 268, 4: 118, 5: 82} | {1: 48, 2: 60, 3: 74, 4: 141, 5: 2176} | {1: 501, 2: 500, 3: 499, 4: 500, 5: 499} | 1.7112406415448118 | 2.321928094887362                 |

## 4. Frequency Sparsity Analysis

| Dataset / Representation                           | Total Customers | Pct F = min (One-time) | Pct F >= 2       | Pct F >= 3        | Unique raw F values | F min value | F median value | F max value | F mean value     | F std value       | Distinct F values in lowest band (F1) | Variance in lowest band (F1) | F-score Shannon entropy (bits) |
| -------------------------------------------------- | --------------- | ---------------------- | ---------------- | ----------------- | ------------------- | ----------- | -------------- | ----------- | ---------------- | ----------------- | ------------------------------------- | ---------------------------- | ------------------------------ |
| Dunnhumby Complete Journey (Baskets per Household) | 2499            | 0.3201280512204882     | 99.6798719487795 | 99.35974389755904 | 365                 | 1.0         | 67.0           | 1169.0      | 95.5874349739896 | 102.0909872910322 | 73                                    | 372.35580500668783           | 1.7112406415448118             |

## 5. FCA Concept Lattice and Pruning Statistics

| Dataset                    | Cohort               | Min Support Threshold | Kneedle Support Threshold | Raw Crisp Concepts | Raw Fuzzy Concepts | Redundancy Suppressed Concepts (J_max=0.8) | Concepts Removed by Suppression | Mean Concepts per Customer | Median Concepts per Customer | Mean Extent Size (mu > 0.5) | Median Extent Size (mu > 0.5) |
| -------------------------- | -------------------- | --------------------- | ------------------------- | ------------------ | ------------------ | ------------------------------------------ | ------------------------------- | -------------------------- | ---------------------------- | --------------------------- | ----------------------------- |
| Dunnhumby Complete Journey | Observation (n=2499) | 0.04                  | 0.2128851540616246        | 38                 | 502                | 123                                        | 379                             | 56.96748112843377          | 61.59323616612795            | 236.8605577689243           | 220.0                         |

## 6. Geometric Clustering Comparison (k=4,5,6)

| Dataset                    | Method               | Hardening                | k  | Evaluation Space       | n_features | Silhouette            | Davies-Bouldin     |
| -------------------------- | -------------------- | ------------------------ | -- | ---------------------- | ---------- | --------------------- | ------------------ |
| Dunnhumby Complete Journey | K-Means              | Partition (k-means)      | 4  | Raw RFM (Standardized) | 3          | 0.4663271447030586    | 0.7944336377618761 |
| Dunnhumby Complete Journey | Ward (Agglomerative) | Hierarchical (Ward)      | 4  | Raw RFM (Standardized) | 3          | 0.34616967218770084   | 0.8421279462024004 |
| Dunnhumby Complete Journey | FCM (Canonical)      | Argmax Soft Membership   | 4  | Raw RFM (Standardized) | 3          | 0.420441050877825     | 0.7998335377855106 |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA        | Top-k Membership Hardening | 4  | Raw RFM (Standardized) | 3          | -0.08511725507666179  | 2.2278147989728563 |
| Dunnhumby Complete Journey | K-Means              | Partition (k-means)      | 5  | Raw RFM (Standardized) | 3          | 0.4746946163742299    | 0.729979201662086  |
| Dunnhumby Complete Journey | Ward (Agglomerative) | Hierarchical (Ward)      | 5  | Raw RFM (Standardized) | 3          | 0.357076517544361     | 0.8441465456280008 |
| Dunnhumby Complete Journey | FCM (Canonical)      | Argmax Soft Membership   | 5  | Raw RFM (Standardized) | 3          | 0.32271873724162775   | 0.9232890699785165 |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA        | Top-k Membership Hardening | 5  | Raw RFM (Standardized) | 3          | -0.20990336590220812  | 2.5512855767120977 |
| Dunnhumby Complete Journey | K-Means              | Partition (k-means)      | 6  | Raw RFM (Standardized) | 3          | 0.4602505964452054    | 0.7767591200525574 |
| Dunnhumby Complete Journey | Ward (Agglomerative) | Hierarchical (Ward)      | 6  | Raw RFM (Standardized) | 3          | 0.38160885465689576   | 0.7642257651699743 |
| Dunnhumby Complete Journey | FCM (Canonical)      | Argmax Soft Membership   | 6  | Raw RFM (Standardized) | 3          | 0.38798437958610454   | 0.826811149933668  |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA        | Top-k Membership Hardening | 6  | Raw RFM (Standardized) | 3          | -0.23574949394510492  | 3.325766096783338  |
| Dunnhumby Complete Journey | Crisp RFM-FCA        | Natural Hardening Rule   | 6  | Raw RFM (Standardized) | 3          | 0.0008197259756365077 | 1.3502858794160841 |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA        | Natural Hardening Rule   | 12 | Raw RFM (Standardized) | 3          | -0.39989972016926895  | 3.5972215825762937 |

## 7. Out-of-Sample Predictive Metrics (Fixed Split)

| arm                        | n_features | test_repurchase_auc | test_brier_score   | test_spend_r2      | test_spend_mae     | test_spend_spearman | test_invoices_r2   | test_invoices_mae  | test_invoices_spearman |
| -------------------------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | ------------------ | ---------------------- |
| Raw RFM Baseline           | 3          | 0.8694213021403125  | 0.061497070769874  | 0.3604202738478452 | 1.0707438153711142 | 0.7679529255395241  | 0.4744942337676489 | 0.6211976181610656 | 0.7660051268305171     |
| Crisp RFM-FCA              | 40         | 0.8610620213750737  | 0.0548203209635681 | 0.4629245386377572 | 0.970272325538574  | 0.7476991345141422  | 0.5536295851663331 | 0.5611634070495064 | 0.7623348579112877     |
| Fuzzy RFM-FCA (Suppressed) | 114        | 0.862899380066762   | 0.0529104214695956 | 0.5152254538564129 | 0.8973280446387146 | 0.7924057571837513  | 0.6188327718720246 | 0.5132825560039256 | 0.806463220046225      |
| FCM Soft (k=40)            | 40         | 0.8283542315352463  | 0.0557095179157683 | 0.4445004718396745 | 0.9618404187233658 | 0.7505109064874428  | 0.5536796513218485 | 0.5563828995063114 | 0.7678020226977068     |

### Paired Bootstrap 95% Confidence Intervals (N=1,000)

| Comparison                                     | Metric         | Point Delta (M1 - M2) | 95% CI Lower        | 95% CI Upper        | Spans Zero | Empirical p-value |
| ---------------------------------------------- | -------------- | --------------------- | ------------------- | ------------------- | ---------- | ----------------- |
| Fuzzy RFM-FCA (Suppressed) vs Raw RFM Baseline | Repurchase AUC | -0.0065219220735505   | -0.0349289081835564 | 0.019623167488108   | True       | 0.33              |
| Fuzzy RFM-FCA (Suppressed) vs Raw RFM Baseline | Spend R2       | 0.1548051800085677    | 0.1018877638373765  | 0.2136082358177541  | False      | 0.0               |
| Fuzzy RFM-FCA (Suppressed) vs Raw RFM Baseline | Invoice R2     | 0.1443385381043757    | 0.1027651268576387  | 0.1865737728414832  | False      | 0.0               |
| Crisp RFM-FCA vs Raw RFM Baseline              | Repurchase AUC | -0.0083592807652388   | -0.0312582595883781 | 0.0141737431352757  | True       | 0.231             |
| Crisp RFM-FCA vs Raw RFM Baseline              | Spend R2       | 0.1025042647899119    | 0.0544815100696956  | 0.1566844473958947  | False      | 0.0               |
| Crisp RFM-FCA vs Raw RFM Baseline              | Invoice R2     | 0.0791353513986842    | 0.0397602130796613  | 0.1183665274337936  | False      | 0.0               |
| Fuzzy RFM-FCA (Suppressed) vs Crisp RFM-FCA    | Repurchase AUC | 0.0018373586916883    | -0.0199593378592244 | 0.0263549392944287  | True       | 0.425             |
| Fuzzy RFM-FCA (Suppressed) vs Crisp RFM-FCA    | Spend R2       | 0.0523009152186557    | 0.0301815350759035  | 0.0768780255460094  | False      | 0.0               |
| Fuzzy RFM-FCA (Suppressed) vs Crisp RFM-FCA    | Invoice R2     | 0.0652031867056914    | 0.0390978724461161  | 0.0904740438773696  | False      | 0.0               |
| FCM Soft (k=40) vs Raw RFM Baseline            | Repurchase AUC | -0.0410670706050662   | -0.0666995442468831 | -0.0179438383070812 | False      | 0.001             |
| FCM Soft (k=40) vs Raw RFM Baseline            | Spend R2       | 0.0840801979918293    | 0.0414041778664501  | 0.1347270150739695  | False      | 0.001             |
| FCM Soft (k=40) vs Raw RFM Baseline            | Invoice R2     | 0.0791854175541996    | 0.0404479247765745  | 0.1146335307359268  | False      | 0.0               |
| Fuzzy RFM-FCA (Suppressed) vs FCM Soft (k=40)  | Repurchase AUC | 0.0345451485315156    | 0.0052947007042253  | 0.0662572619273968  | False      | 0.011             |
| Fuzzy RFM-FCA (Suppressed) vs FCM Soft (k=40)  | Spend R2       | 0.0707249820167383    | 0.032826142326168   | 0.1075291054383517  | False      | 0.0               |
| Fuzzy RFM-FCA (Suppressed) vs FCM Soft (k=40)  | Invoice R2     | 0.065153120550176     | 0.0357662838081869  | 0.0948303482059286  | False      | 0.0               |

## 8. Multi-Split Temporal Cross-Validation (10 Splits: Seeds 1000-1009)

| arm                        | auc_mean           | auc_std              | spend_mean          | spend_std            | inv_mean           | inv_std              |
| -------------------------- | ------------------ | -------------------- | ------------------- | -------------------- | ------------------ | -------------------- |
| Crisp RFM-FCA              | 0.8540730455272237 | 0.02420767306271801  | 0.48000535980094716 | 0.03513844855601183  | 0.5547797499593543 | 0.03435408595033397  |
| FCM Soft (matched k)       | 0.8378608095598754 | 0.03120577822926389  | 0.4482036316260876  | 0.03384347973114407  | 0.5520299427240206 | 0.025375683875940455 |
| Fuzzy RFM-FCA (Suppressed) | 0.8604701394148504 | 0.029159002093107133 | 0.5028154924952183  | 0.037643216148640884 | 0.6045106006510623 | 0.032972572778388325 |
| Raw RFM Baseline           | 0.8624000673230665 | 0.02746486088623322  | 0.3664099709217208  | 0.045855858179639206 | 0.4515337034783083 | 0.033443203683033766 |

## 9. Validation Checklist

- [x] S does not appear anywhere in the Dunnhumby feature matrix.
- [x] F = distinct BASKET_ID count, not transaction-row count.
- [x] M = sum(SALES_VALUE).
- [x] SALES_VALUE is NOT multiplied by QUANTITY.
- [x] No F* calculation.
- [x] No entropy-based F-weight optimization.
- [x] max_len=None.
- [x] No future information enters representation fitting.
- [x] Kneedle terminology is Kneedle-inspired.
- [x] Stability is described as a proxy.
- [x] Canonical FPC/Xie-Beni are used for FCM only.
- [x] FCA hard clusters use Silhouette/DB.
- [x] Existing Retail II results are not overwritten.
- [x] Olist is not included.
- [x] No synthetic monetary or satisfaction variable is introduced.
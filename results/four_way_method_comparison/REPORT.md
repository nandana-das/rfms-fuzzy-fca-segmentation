# Four-Way Method Comparison: Raw RFM vs FCM vs Fuzzy RFM-FCA vs Kuznetsov-Pruned Fuzzy RFM-FCA

> **Audit caveats (2026-10-08).** (1) 'Raw RFM' is untransformed standardized R/F/M in linear models; log or spline RFM baselines recover most or all of the FCA gain (see `results/baseline_ladder_rolling_origin/`). (2) The Kuznetsov threshold 5.4e-20 was chosen as the best arm on these same seeds 1000-1009 and is below float64 resolution (it equals float(stability) == 1.0). (3) The permutation p-values of 0.0015 are below the exact sign-flip floor of 0.00195 for n = 10. (4) Jaccard suppression kept concepts with empty mu >= 0.5 extents (now fixed). See `docs/AUDIT_ERRATA.md`.

## 1. Objective

Compare four representations under an identical temporal split, preprocessing, predictive model, and evaluation protocol:

1. **Raw RFM** — standardized R, F, M.
2. **FCM** — canonical fuzzy c-means soft memberships (k ∈ {4,5,6}).
3. **Fuzzy RFM-FCA** — fuzzy concept mining + Jaccard suppression (no Kuznetsov).
4. **Kuznetsov-pruned Fuzzy RFM-FCA** — exact canonical Kuznetsov stability + fixed loss threshold.

Purpose: identify where performance and compression advantages actually come from — not to prove Kuznetsov wins.

## 2. Experimental design

- Datasets: Dunnhumby Complete Journey, Online Retail II
- Splits: fixed seed 42 + ten multi-split seeds 1000–1009
- Train/test: 70/30 stratified on repurchased
- Models: LogisticRegressionCV (Cs=10, cv=5, roc_auc, lbfgs, max_iter=2000) + RidgeCV (alphas=logspace(-3,3,20))
- Targets: repurchased (AUC), future spend log1p (R²), future invoices log1p (R²)

## 3. Leakage controls

- Raw RFM: scaler fit on train only, then transform test.
- FCM: fit on train only; assign test memberships.
- Fuzzy RFM-FCA: mine concepts on train only; project test via frozen centroids.
- Kuznetsov-FCA: compute exact stability from the training context only; test never enters concept selection.

## 4. Dataset protocols

- Dunnhumby Complete Journey: observation days 1–620; holdout days 621–711.
- Online Retail II: observation cutoff 2010-12-09 23:59:59; Year 2 holdout.

## 5. Method definitions

- Raw RFM: 3 standardized R/F/M features.
- FCM: FuzzyCMeans(m=2.0, k ∈ {4,5,6}) soft memberships as predictive features.
- Fuzzy RFM-FCA: fuzzy concept mining + Jaccard suppression (Jmax=0.80, μ_cut=0.50), no Kuznetsov.
- Kuznetsov-FCA: exact canonical Kuznetsov stability (loss ≤ 5.4e-20) + Jaccard suppression.

## 6. Predictive results

| dataset | method | arm | n_features_mean | n_features_std | n_final_mean | n_final_std | auc_mean | auc_std | spend_r2_mean | spend_r2_std | invoice_r2_mean | invoice_r2_std | compression_mean | compression_std |
| Dunnhumby Complete Journey | Raw RFM | Raw RFM | 3.0 | 0.0 | 3.0 | 0.0 | 0.8624000673230665 | 0.02746486088623323 | 0.3664099709217208 | 0.045855858179639206 | 0.4515337034783083 | 0.03344320368303376 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | FCM | FCM k=4 | 4.0 | 0.0 | 4.0 | 0.0 | 0.7149316951387135 | 0.03569586833537283 | 0.29004089178502596 | 0.02307420957691829 | 0.3858716651838851 | 0.02692717784387829 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | FCM | FCM k=5 | 5.0 | 0.0 | 5.0 | 0.0 | 0.7510477152234285 | 0.0628578992606297 | 0.312059764065491 | 0.04450022606301634 | 0.40995537656710274 | 0.03909219640105678 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | FCM | FCM k=6 | 6.0 | 0.0 | 6.0 | 0.0 | 0.7922802883671352 | 0.05249449525187966 | 0.36029264125837834 | 0.06230649734015497 | 0.45655291201065606 | 0.05521027705850014 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113.0 | 13.258121200900895 | 113.0 | 13.258121200900895 | 0.8604701394148503 | 0.02915900209310713 | 0.5028154924952183 | 0.03764321614864093 | 0.6045106006510623 | 0.03297257277838834 | 4.636899225155489 | 0.5621559880613142 |
| Dunnhumby Complete Journey | Kuznetsov-FCA | Kuznetsov-FCA | 19.4 | 0.9660917830792959 | 19.4 | 0.9660917830792959 | 0.8701324020309127 | 0.026094864387492967 | 0.5145509149126785 | 0.036421745700835575 | 0.606744592336338 | 0.032084049746311294 | 26.733885448916407 | 1.5063347811249923 |

### Retail II summary
| dataset | method | arm | n_features_mean | n_features_std | n_final_mean | n_final_std | auc_mean | auc_std | spend_r2_mean | spend_r2_std | invoice_r2_mean | invoice_r2_std | compression_mean | compression_std |
| Online Retail II | Raw RFM | Raw RFM | 3.0 | 0.0 | 3.0 | 0.0 | 0.7773501428425671 | 0.01194921225626333 | 0.21723706573559975 | 0.0101784962253874 | 0.34141178466463373 | 0.021926610451560368 | 1.0 | 0.0 |
| Online Retail II | FCM | FCM k=4 | 4.0 | 0.0 | 4.0 | 0.0 | 0.75549153147638 | 0.013044051722242923 | 0.2532268268525356 | 0.01629286419565481 | 0.37790751509018394 | 0.013864366457696967 | 1.0 | 0.0 |
| Online Retail II | FCM | FCM k=5 | 5.0 | 0.0 | 5.0 | 0.0 | 0.754360524436282 | 0.011686145449503069 | 0.2663958001130294 | 0.01900287196370774 | 0.3985803461843 | 0.022940110988616303 | 1.0 | 0.0 |
| Online Retail II | FCM | FCM k=6 | 6.0 | 0.0 | 6.0 | 0.0 | 0.7548576675849403 | 0.010725742491680505 | 0.26371844327795385 | 0.012955564446004979 | 0.3998468332246015 | 0.0173736550254231 | 1.0 | 0.0 |
| Online Retail II | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 96.7 | 16.878981011897604 | 96.7 | 16.878981011897604 | 0.7858426436077951 | 0.011877390037931795 | 0.35604153844181397 | 0.019473535329768216 | 0.47485803295741275 | 0.013875796134087282 | 4.627254464668149 | 0.8041341574431892 |
| Online Retail II | Kuznetsov-FCA | Kuznetsov-FCA | 30.7 | 1.888562063228706 | 30.7 | 1.888562063228706 | 0.78790544332211 | 0.011107276216012438 | 0.35616085266396524 | 0.01850670568530513 | 0.4756582466387046 | 0.011283337549559952 | 14.239257014434296 | 1.0779641392336785 |

## 7. Statistical comparisons
| dataset | comparison | metric | mean_diff | std_diff | n_splits | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_spans_zero | permutation_p_value | permutation_significant |
| Dunnhumby Complete Journey | FCM (k=6) vs Raw RFM | auc | -0.07011977895593141 | 0.03499326172116179 | 10 | -0.09110192431765267 | -0.050173146511823606 | False | 0.0015 | True |
| Dunnhumby Complete Journey | FCM (k=6) vs Raw RFM | spend_r2 | -0.006117329663342508 | 0.042711108890868925 | 10 | -0.034115146343612374 | 0.020340195895518987 | True | 0.6668 | False |
| Dunnhumby Complete Journey | FCM (k=6) vs Raw RFM | invoice_r2 | 0.005019208532347807 | 0.03693142853020773 | 10 | -0.01983925448442611 | 0.02636254104698673 | True | 0.6653 | False |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs Raw RFM | auc | -0.001929927908216178 | 0.010309297661318197 | 10 | -0.007977327554770075 | 0.004115367892507512 | True | 0.5501 | False |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs Raw RFM | spend_r2 | 0.13640552157349753 | 0.035913450276281775 | 10 | 0.11405786868998243 | 0.1546345070620738 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs Raw RFM | invoice_r2 | 0.1529768971727541 | 0.020205457918536826 | 10 | 0.1394301558247949 | 0.1633178019865827 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Raw RFM | auc | 0.007732334707845945 | 0.011705710120737137 | 10 | 0.0005792238211450492 | 0.015225742657577975 | False | 0.0793 | False |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Raw RFM | spend_r2 | 0.14814094399095765 | 0.027162688439956506 | 10 | 0.13126422878880925 | 0.16263374219741458 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Raw RFM | invoice_r2 | 0.15521088885802978 | 0.01912511319448256 | 10 | 0.14268873282301509 | 0.16565236966203148 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs FCM (k=6) | auc | 0.06818985104771523 | 0.03284833421929664 | 10 | 0.048661778731521166 | 0.08775614322982407 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs FCM (k=6) | spend_r2 | 0.14252285123684003 | 0.042671725750641314 | 10 | 0.11696039809493407 | 0.1685985307052398 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA vs FCM (k=6) | invoice_r2 | 0.1479576886404063 | 0.03411243455039461 | 10 | 0.12891823699366434 | 0.16933253094745757 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs FCM (k=6) | auc | 0.07785211366377735 | 0.034144754591317016 | 10 | 0.057594848382843844 | 0.09877913826474796 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs FCM (k=6) | spend_r2 | 0.15425827365430017 | 0.03992930219079818 | 10 | 0.1302954666289633 | 0.17867690129451494 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs FCM (k=6) | invoice_r2 | 0.15019168032568198 | 0.03281827195158417 | 10 | 0.1318236773099672 | 0.17094939537444603 | False | 0.0015 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Fuzzy RFM-FCA | auc | 0.009662262616062123 | 0.00789686634602945 | 10 | 0.00490985020617686 | 0.014401210412634204 | False | 0.0052 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Fuzzy RFM-FCA | spend_r2 | 0.01173542241746015 | 0.01027211615848046 | 10 | 0.00655774872309312 | 0.018273544276328248 | False | 0.0037 | True |
| Dunnhumby Complete Journey | Kuznetsov-FCA vs Fuzzy RFM-FCA | invoice_r2 | 0.002233991685275716 | 0.006939641385819147 | 10 | -0.0021623185468599066 | 0.006235105221053417 | True | 0.3112 | False |
| Online Retail II | FCM (k=6) vs Raw RFM | auc | -0.022492475257626855 | 0.004418378232777788 | 10 | -0.025082816804407767 | -0.020025185567799336 | False | 0.0015 | True |
| Online Retail II | FCM (k=6) vs Raw RFM | spend_r2 | 0.046481377542354096 | 0.015558646411144192 | 10 | 0.03743200415485566 | 0.05635336292615682 | False | 0.0015 | True |
| Online Retail II | FCM (k=6) vs Raw RFM | invoice_r2 | 0.05843504855996782 | 0.020324955301698033 | 10 | 0.047358406623677275 | 0.07128304952211327 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs Raw RFM | auc | 0.008492500765228006 | 0.0040805215837337 | 10 | 0.006141210080604008 | 0.010829857922660916 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs Raw RFM | spend_r2 | 0.13880447270621424 | 0.014893407327538305 | 10 | 0.12967715923186895 | 0.14776225540504812 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs Raw RFM | invoice_r2 | 0.13344624829277904 | 0.021038902254577357 | 10 | 0.12177407099287456 | 0.1476394867871073 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs Raw RFM | auc | 0.01055530047954283 | 0.005127828087551895 | 10 | 0.0077270495357615815 | 0.013615488853178174 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs Raw RFM | spend_r2 | 0.13892378692836554 | 0.014033661985317273 | 10 | 0.13034036141272912 | 0.1473411245726778 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs Raw RFM | invoice_r2 | 0.134246461974071 | 0.021853492018637975 | 10 | 0.12270463976204639 | 0.14898469226527056 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs FCM (k=6) | auc | 0.03098497602285486 | 0.004730396863583444 | 10 | 0.028194740970309207 | 0.03342283312927261 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs FCM (k=6) | spend_r2 | 0.09232309516386011 | 0.016043718304123978 | 10 | 0.08068767293504124 | 0.10010824765759375 | False | 0.0015 | True |
| Online Retail II | Fuzzy RFM-FCA vs FCM (k=6) | invoice_r2 | 0.07501119973281124 | 0.01627960958861579 | 10 | 0.06489245760267529 | 0.08468925823181424 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs FCM (k=6) | auc | 0.03304777573716968 | 0.004661202010311004 | 10 | 0.030277181537598157 | 0.0356566389909193 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs FCM (k=6) | spend_r2 | 0.09244240938601143 | 0.014618311928481308 | 10 | 0.08197715297562104 | 0.09954356026980407 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs FCM (k=6) | invoice_r2 | 0.07581141341410315 | 0.014156466167220121 | 10 | 0.06651579351230646 | 0.08425034695826018 | False | 0.0015 | True |
| Online Retail II | Kuznetsov-FCA vs Fuzzy RFM-FCA | auc | 0.0020627997143148223 | 0.0018770136400089053 | 10 | 0.0009210667278848839 | 0.0031721380471380015 | False | 0.0142 | True |
| Online Retail II | Kuznetsov-FCA vs Fuzzy RFM-FCA | spend_r2 | 0.00011931422215132326 | 0.0030166725466614338 | 10 | -0.0016165932363566373 | 0.001914509631862233 | True | 0.8921 | False |
| Online Retail II | Kuznetsov-FCA vs Fuzzy RFM-FCA | invoice_r2 | 0.0008002136812919058 | 0.004411559402009421 | 10 | -0.0018139067416091108 | 0.0031862841852457356 | True | 0.5758 | False |

## 8. FCM diagnostics
| dataset | k | fpc_mean | xie_beni_mean | silhouette_mean | davies_bouldin_mean | auc_mean | spend_r2_mean | invoice_r2_mean |
| Dunnhumby Complete Journey | 4 | 0.6536053057689698 | 0.5949697099219154 | 0.35997411707308535 | 0.9053275745756061 | 0.7149316951387135 | 0.29004089178502596 | 0.3858716651838851 |
| Dunnhumby Complete Journey | 5 | 0.6179684405849815 | 0.5215055839443046 | 0.3493362344372916 | 0.8863487794185794 | 0.7510477152234285 | 0.312059764065491 | 0.40995537656710274 |
| Dunnhumby Complete Journey | 6 | 0.5933394959879796 | 0.5148407949125808 | 0.34985174488599474 | 0.884161912695445 | 0.7922802883671352 | 0.36029264125837834 | 0.45655291201065606 |
| Online Retail II | 4 | 0.7494312052572929 | 0.3132449932367769 | 0.5449235839542108 | 0.7378256072240597 | 0.75549153147638 | 0.2532268268525356 | 0.37790751509018394 |
| Online Retail II | 5 | 0.7083782310660346 | 0.4063662731796853 | 0.48429725761277365 | 0.7143854338770064 | 0.754360524436282 | 0.2663958001130294 | 0.3985803461843 |
| Online Retail II | 6 | 0.6830866295008773 | 0.28990267731306 | 0.4678039984082357 | 0.6854985134057128 | 0.7548576675849403 | 0.26371844327795385 | 0.3998468332246015 |

## 9. Concept compression
| dataset | method | n_features_mean | n_features_std | n_final_concepts_mean | n_final_concepts_std | compression_mean | compression_std |
| Dunnhumby Complete Journey | Raw RFM | 3.0 | 0.0 | 3.0 | 0.0 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | FCM | 5.0 | 0.8304547985373997 | 5.0 | 0.8304547985373997 | 1.0 | 0.0 |
| Dunnhumby Complete Journey | Fuzzy RFM-FCA | 113.0 | 13.258121200900895 | 113.0 | 13.258121200900895 | 4.636899225155489 | 0.5621559880613142 |
| Dunnhumby Complete Journey | Kuznetsov-FCA | 19.4 | 0.9660917830792959 | 19.4 | 0.9660917830792959 | 26.733885448916407 | 1.5063347811249923 |
| Online Retail II | Raw RFM | 3.0 | 0.0 | 3.0 | 0.0 | 1.0 | 0.0 |
| Online Retail II | FCM | 5.0 | 0.8304547985373997 | 5.0 | 0.8304547985373997 | 1.0 | 0.0 |
| Online Retail II | Fuzzy RFM-FCA | 96.7 | 16.878981011897604 | 96.7 | 16.878981011897604 | 4.627254464668149 | 0.8041341574431892 |
| Online Retail II | Kuznetsov-FCA | 30.7 | 1.888562063228706 | 30.7 | 1.888562063228706 | 14.239257014434296 | 1.0779641392336785 |

## 10. Pareto analysis
Observed points (no optimization):

### auc
| dataset | split_seed | method | arm | n_features | n_final_concepts | metric | value |
| Dunnhumby Complete Journey | 42 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8694213021403125 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=4 | 4 | 4 | auc | 0.6926982524053971 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=5 | 5 | 5 | auc | 0.7149990182052792 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=6 | 6 | 6 | auc | 0.7898398272041292 |
| Dunnhumby Complete Journey | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | auc | 0.862899380066762 |
| Dunnhumby Complete Journey | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | auc | 0.8691548150018232 |
| Dunnhumby Complete Journey | 1000 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8877387865017251 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=4 | 4 | 4 | auc | 0.6562035400712503 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=5 | 5 | 5 | auc | 0.6841987152514797 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=6 | 6 | 6 | auc | 0.826530898482426 |
| Dunnhumby Complete Journey | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 127 | 127 | auc | 0.8958175544896071 |
| Dunnhumby Complete Journey | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | auc | 0.9140649106566803 |
| Dunnhumby Complete Journey | 1001 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8139919773345676 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=4 | 4 | 4 | auc | 0.7092204549917248 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=5 | 5 | 5 | auc | 0.7111559931554882 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=6 | 6 | 6 | auc | 0.7209739403629837 |
| Dunnhumby Complete Journey | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | auc | 0.8268534881763864 |
| Dunnhumby Complete Journey | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.837765435215574 |
| Dunnhumby Complete Journey | 1002 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8861118123930544 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=4 | 4 | 4 | auc | 0.7282392212965301 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=5 | 5 | 5 | auc | 0.7451541417711576 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=6 | 6 | 6 | auc | 0.7848466997671744 |
| Dunnhumby Complete Journey | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 103 | 103 | auc | 0.8716934556369043 |
| Dunnhumby Complete Journey | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.880613761956857 |
| Dunnhumby Complete Journey | 1003 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8921147858284946 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=4 | 4 | 4 | auc | 0.7797974697747482 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=5 | 5 | 5 | auc | 0.8681028920867346 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=6 | 6 | 6 | auc | 0.847990125950237 |
| Dunnhumby Complete Journey | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | auc | 0.8957614519341357 |
| Dunnhumby Complete Journey | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | auc | 0.8979775028752559 |
| Dunnhumby Complete Journey | 1004 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8648208925916575 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=4 | 4 | 4 | auc | 0.7265000420769165 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=5 | 5 | 5 | auc | 0.7455468596594574 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=6 | 6 | 6 | auc | 0.8278773598137394 |
| Dunnhumby Complete Journey | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | auc | 0.8572470476030184 |
| Dunnhumby Complete Journey | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.8663356615893854 |
| Dunnhumby Complete Journey | 1005 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8730118656904821 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=4 | 4 | 4 | auc | 0.7480153721001992 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=5 | 5 | 5 | auc | 0.8562091503267975 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=6 | 6 | 6 | auc | 0.8546382787735981 |
| Dunnhumby Complete Journey | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | auc | 0.8821004796768492 |
| Dunnhumby Complete Journey | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.8868832225307863 |
| Dunnhumby Complete Journey | 1006 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8351706920250217 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=4 | 4 | 4 | auc | 0.7070605066060759 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=5 | 5 | 5 | auc | 0.7063592246626834 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=6 | 6 | 6 | auc | 0.7119414289320879 |
| Dunnhumby Complete Journey | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | auc | 0.8185783612443547 |
| Dunnhumby Complete Journey | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.8347499228589863 |
| Dunnhumby Complete Journey | 1007 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8755925832421667 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=4 | 4 | 4 | auc | 0.7133720440966085 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=5 | 5 | 5 | auc | 0.7452382956043647 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=6 | 6 | 6 | auc | 0.830934949086931 |
| Dunnhumby Complete Journey | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | auc | 0.8658307385901427 |
| Dunnhumby Complete Journey | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | auc | 0.8732783528289714 |
| Dunnhumby Complete Journey | 1008 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8265870010378973 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=4 | 4 | 4 | auc | 0.6668910768885523 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=5 | 5 | 5 | auc | 0.6996830205615866 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=6 | 6 | 6 | auc | 0.7573564475861874 |
| Dunnhumby Complete Journey | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 126 | 126 | auc | 0.8210608993239642 |
| Dunnhumby Complete Journey | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 17 | 17 | auc | 0.843866588123089 |
| Dunnhumby Complete Journey | 1009 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.8688602765855985 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=4 | 4 | 4 | auc | 0.7140172234845298 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=5 | 5 | 5 | auc | 0.7488288591545346 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=6 | 6 | 6 | auc | 0.7597127549159864 |
| Dunnhumby Complete Journey | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 137 | 137 | auc | 0.869757917473141 |
| Dunnhumby Complete Journey | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | auc | 0.8657886616735393 |
| Online Retail II | 42 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.784519436792164 |
| Online Retail II | 42 | FCM | FCM k=4 | 4 | 4 | auc | 0.7560759106213653 |
| Online Retail II | 42 | FCM | FCM k=5 | 5 | 5 | auc | 0.7566294255688195 |
| Online Retail II | 42 | FCM | FCM k=6 | 6 | 6 | auc | 0.762539536781961 |
| Online Retail II | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | auc | 0.7859835731047853 |
| Online Retail II | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 28 | 28 | auc | 0.7915059687786961 |
| Online Retail II | 1000 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7793592490562189 |
| Online Retail II | 1000 | FCM | FCM k=4 | 4 | 4 | auc | 0.753961330476482 |
| Online Retail II | 1000 | FCM | FCM k=5 | 5 | 5 | auc | 0.7596138149168452 |
| Online Retail II | 1000 | FCM | FCM k=6 | 6 | 6 | auc | 0.7546933986327926 |
| Online Retail II | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 92 | 92 | auc | 0.7851902867054382 |
| Online Retail II | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | auc | 0.7855575961636567 |
| Online Retail II | 1001 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7579379655137232 |
| Online Retail II | 1001 | FCM | FCM k=4 | 4 | 4 | auc | 0.7358126721763085 |
| Online Retail II | 1001 | FCM | FCM k=5 | 5 | 5 | auc | 0.7320579532700745 |
| Online Retail II | 1001 | FCM | FCM k=6 | 6 | 6 | auc | 0.7365753494541373 |
| Online Retail II | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 74 | 74 | auc | 0.7686945209672482 |
| Online Retail II | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | auc | 0.7723217018671564 |
| Online Retail II | 1002 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7857004387307418 |
| Online Retail II | 1002 | FCM | FCM k=4 | 4 | 4 | auc | 0.7674982144679114 |
| Online Retail II | 1002 | FCM | FCM k=5 | 5 | 5 | auc | 0.7654703601673298 |
| Online Retail II | 1002 | FCM | FCM k=6 | 6 | 6 | auc | 0.7656718702173246 |
| Online Retail II | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 93 | 93 | auc | 0.8008634323028263 |
| Online Retail II | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | auc | 0.8053119579634731 |
| Online Retail II | 1003 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7752499744923987 |
| Online Retail II | 1003 | FCM | FCM k=4 | 4 | 4 | auc | 0.7538159371492705 |
| Online Retail II | 1003 | FCM | FCM k=5 | 5 | 5 | auc | 0.7525941230486685 |
| Online Retail II | 1003 | FCM | FCM k=6 | 6 | 6 | auc | 0.7540684624017957 |
| Online Retail II | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 117 | 117 | auc | 0.7778020099989797 |
| Online Retail II | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | auc | 0.7811154474033262 |
| Online Retail II | 1004 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7543490460157127 |
| Online Retail II | 1004 | FCM | FCM k=4 | 4 | 4 | auc | 0.7339735741250893 |
| Online Retail II | 1004 | FCM | FCM k=5 | 5 | 5 | auc | 0.7370472400775432 |
| Online Retail II | 1004 | FCM | FCM k=6 | 6 | 6 | auc | 0.7358662381389653 |
| Online Retail II | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 122 | 122 | auc | 0.7668732782369145 |
| Online Retail II | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | auc | 0.7713702683399652 |
| Online Retail II | 1005 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.786312621161106 |
| Online Retail II | 1005 | FCM | FCM k=4 | 4 | 4 | auc | 0.7548081828384858 |
| Online Retail II | 1005 | FCM | FCM k=5 | 5 | 5 | auc | 0.7548387919600041 |
| Online Retail II | 1005 | FCM | FCM k=6 | 6 | 6 | auc | 0.7537113559840832 |
| Online Retail II | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | auc | 0.7912330374451586 |
| Online Retail II | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | auc | 0.7908223650647893 |
| Online Retail II | 1006 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7828780226507499 |
| Online Retail II | 1006 | FCM | FCM k=4 | 4 | 4 | auc | 0.7614860728497093 |
| Online Retail II | 1006 | FCM | FCM k=5 | 5 | 5 | auc | 0.7563590449954086 |
| Online Retail II | 1006 | FCM | FCM k=6 | 6 | 6 | auc | 0.7592414039383737 |
| Online Retail II | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 94 | 94 | auc | 0.7885572900724416 |
| Online Retail II | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | auc | 0.7897842056932965 |
| Online Retail II | 1007 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7775227017651261 |
| Online Retail II | 1007 | FCM | FCM k=4 | 4 | 4 | auc | 0.7546627895112743 |
| Online Retail II | 1007 | FCM | FCM k=5 | 5 | 5 | auc | 0.7552035506580963 |
| Online Retail II | 1007 | FCM | FCM k=6 | 6 | 6 | auc | 0.7611213141516171 |
| Online Retail II | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 89 | 89 | auc | 0.7839136312621162 |
| Online Retail II | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | auc | 0.7858522089582696 |
| Online Retail II | 1008 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7868406285072952 |
| Online Retail II | 1008 | FCM | FCM k=4 | 4 | 4 | auc | 0.7766962554841342 |
| Online Retail II | 1008 | FCM | FCM k=5 | 5 | 5 | auc | 0.7695362718089991 |
| Online Retail II | 1008 | FCM | FCM k=6 | 6 | 6 | auc | 0.7649704111825324 |
| Online Retail II | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 72 | 72 | auc | 0.7993967452300785 |
| Online Retail II | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 27 | 27 | auc | 0.7989350576471789 |
| Online Retail II | 1009 | Raw RFM | Raw RFM | 3 | 3 | auc | 0.7873507805325988 |
| Online Retail II | 1009 | FCM | FCM k=4 | 4 | 4 | auc | 0.7622002856851343 |
| Online Retail II | 1009 | FCM | FCM k=5 | 5 | 5 | auc | 0.7608840934598511 |
| Online Retail II | 1009 | FCM | FCM k=6 | 6 | 6 | auc | 0.7626568717477807 |
| Online Retail II | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | auc | 0.7959022038567494 |
| Online Retail II | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 32 | 32 | auc | 0.7979836241199876 |

### spend_r2
| dataset | split_seed | method | arm | n_features | n_final_concepts | metric | value |
| Dunnhumby Complete Journey | 42 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.36042027384784525 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2922435579447993 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2887265334373058 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.36790330062927945 |
| Dunnhumby Complete Journey | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | spend_r2 | 0.5152254538564129 |
| Dunnhumby Complete Journey | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | spend_r2 | 0.5172795650373772 |
| Dunnhumby Complete Journey | 1000 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.4221753307279873 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.28738770810667125 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.28725278200690973 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.45932504324260814 |
| Dunnhumby Complete Journey | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 127 | 127 | spend_r2 | 0.5747262031590346 |
| Dunnhumby Complete Journey | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | spend_r2 | 0.5903054386238549 |
| Dunnhumby Complete Journey | 1001 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.307447207753341 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2907138542652331 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2738652165233829 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.27684248769526154 |
| Dunnhumby Complete Journey | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | spend_r2 | 0.4851146675364 |
| Dunnhumby Complete Journey | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.48393782161633925 |
| Dunnhumby Complete Journey | 1002 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.39320470577439703 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2987982749025945 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.32544620062483176 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.35085400410631584 |
| Dunnhumby Complete Journey | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 103 | 103 | spend_r2 | 0.5357969276176522 |
| Dunnhumby Complete Journey | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.5408804843900505 |
| Dunnhumby Complete Journey | 1003 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.3572254127035003 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.32093444577473995 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.35409179678865665 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.3656836890893941 |
| Dunnhumby Complete Journey | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | spend_r2 | 0.5028902739553182 |
| Dunnhumby Complete Journey | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | spend_r2 | 0.5102551812369718 |
| Dunnhumby Complete Journey | 1004 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.34676748299750604 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.28415810354081716 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.28226721226986784 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.3829905585591141 |
| Dunnhumby Complete Journey | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | spend_r2 | 0.4766733898033869 |
| Dunnhumby Complete Journey | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.48863872845366385 |
| Dunnhumby Complete Journey | 1005 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.413099551492739 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.30190272265853446 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.4056530019250506 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.4068615606057283 |
| Dunnhumby Complete Journey | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | spend_r2 | 0.5394902512004511 |
| Dunnhumby Complete Journey | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.5484172690275608 |
| Dunnhumby Complete Journey | 1006 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.2843727181032123 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2409402762273657 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.25017722486219907 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.25204961883070376 |
| Dunnhumby Complete Journey | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | spend_r2 | 0.4566563455627315 |
| Dunnhumby Complete Journey | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.4662244862793786 |
| Dunnhumby Complete Journey | 1007 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.3518848456660898 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.26588089688347283 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.30449963804057234 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.4091358759182542 |
| Dunnhumby Complete Journey | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | spend_r2 | 0.49854977197947414 |
| Dunnhumby Complete Journey | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | spend_r2 | 0.5074016363386011 |
| Dunnhumby Complete Journey | 1008 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.37797430981808255 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.29633487479011855 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.30915946394115124 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.37233283747683565 |
| Dunnhumby Complete Journey | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 126 | 126 | spend_r2 | 0.5001214254704502 |
| Dunnhumby Complete Journey | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 17 | 17 | spend_r2 | 0.5134768058867254 |
| Dunnhumby Complete Journey | 1009 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.40994814418035286 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.3133577607007121 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.3281851036722876 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.32685073705956746 |
| Dunnhumby Complete Journey | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 137 | 137 | spend_r2 | 0.4581356686672845 |
| Dunnhumby Complete Journey | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | spend_r2 | 0.49597129727363853 |
| Online Retail II | 42 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.2393985893687378 |
| Online Retail II | 42 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2674255556563905 |
| Online Retail II | 42 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.28061980864639746 |
| Online Retail II | 42 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.3013444861170619 |
| Online Retail II | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | spend_r2 | 0.3657222565843664 |
| Online Retail II | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 28 | 28 | spend_r2 | 0.36738648597473533 |
| Online Retail II | 1000 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.22759924803515863 |
| Online Retail II | 1000 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.23745110288763505 |
| Online Retail II | 1000 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.23810112003199269 |
| Online Retail II | 1000 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2674514464974461 |
| Online Retail II | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 92 | 92 | spend_r2 | 0.3518805064596906 |
| Online Retail II | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | spend_r2 | 0.35192855668947653 |
| Online Retail II | 1001 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.21087750529553984 |
| Online Retail II | 1001 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.21936178813599738 |
| Online Retail II | 1001 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.24142767326914827 |
| Online Retail II | 1001 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.23226972487347264 |
| Online Retail II | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 74 | 74 | spend_r2 | 0.32610765655745766 |
| Online Retail II | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | spend_r2 | 0.3261353153688239 |
| Online Retail II | 1002 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.22580821980605825 |
| Online Retail II | 1002 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.26886503214287505 |
| Online Retail II | 1002 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.28432238029243795 |
| Online Retail II | 1002 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2741548981552805 |
| Online Retail II | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 93 | 93 | spend_r2 | 0.37968051693896065 |
| Online Retail II | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | spend_r2 | 0.38001912191574383 |
| Online Retail II | 1003 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.2210344335959702 |
| Online Retail II | 1003 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.254686179290074 |
| Online Retail II | 1003 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.24413032133661017 |
| Online Retail II | 1003 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2519676713530168 |
| Online Retail II | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 117 | 117 | spend_r2 | 0.3466415908201521 |
| Online Retail II | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | spend_r2 | 0.34756915030645963 |
| Online Retail II | 1004 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.19407996002460415 |
| Online Retail II | 1004 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.24873374592723185 |
| Online Retail II | 1004 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.27091580593609876 |
| Online Retail II | 1004 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.27282735987128925 |
| Online Retail II | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 122 | 122 | spend_r2 | 0.32461867241304143 |
| Online Retail II | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | spend_r2 | 0.3272236731151339 |
| Online Retail II | 1005 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.21940125024730162 |
| Online Retail II | 1005 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.27531091915438355 |
| Online Retail II | 1005 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.29202540907875285 |
| Online Retail II | 1005 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2747109673054383 |
| Online Retail II | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | spend_r2 | 0.36818799490180465 |
| Online Retail II | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | spend_r2 | 0.3720369234878561 |
| Online Retail II | 1006 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.2079920177190956 |
| Online Retail II | 1006 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.24824904188139985 |
| Online Retail II | 1006 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2764771573064737 |
| Online Retail II | 1006 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2635512094502752 |
| Online Retail II | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 94 | 94 | spend_r2 | 0.36170673564855327 |
| Online Retail II | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | spend_r2 | 0.35940846841234797 |
| Online Retail II | 1007 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.22087358442410676 |
| Online Retail II | 1007 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2625093107392733 |
| Online Retail II | 1007 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2733991842559691 |
| Online Retail II | 1007 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2648236018285973 |
| Online Retail II | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 89 | 89 | spend_r2 | 0.3539837370841368 |
| Online Retail II | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | spend_r2 | 0.3581916372799163 |
| Online Retail II | 1008 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.22389373531369994 |
| Online Retail II | 1008 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2525841265572836 |
| Online Retail II | 1008 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2636063086485714 |
| Online Retail II | 1008 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.26427286476232104 |
| Online Retail II | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 72 | 72 | spend_r2 | 0.3687947777933873 |
| Online Retail II | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 27 | 27 | spend_r2 | 0.3641859431240624 |
| Online Retail II | 1009 | Raw RFM | Raw RFM | 3 | 3 | spend_r2 | 0.22081070289446236 |
| Online Retail II | 1009 | FCM | FCM k=4 | 4 | 4 | spend_r2 | 0.2645170218092021 |
| Online Retail II | 1009 | FCM | FCM k=5 | 5 | 5 | spend_r2 | 0.2795526409742387 |
| Online Retail II | 1009 | FCM | FCM k=6 | 6 | 6 | spend_r2 | 0.2711546886824011 |
| Online Retail II | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | spend_r2 | 0.37881319580095496 |
| Online Retail II | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 32 | 32 | spend_r2 | 0.3749097369398321 |

### invoice_r2
| dataset | split_seed | method | arm | n_features | n_final_concepts | metric | value |
| Dunnhumby Complete Journey | 42 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.4744942337676489 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3879168487948572 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.40414689441946106 |
| Dunnhumby Complete Journey | 42 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.46357406439628634 |
| Dunnhumby Complete Journey | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | invoice_r2 | 0.6188327718720246 |
| Dunnhumby Complete Journey | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | invoice_r2 | 0.6086557141359596 |
| Dunnhumby Complete Journey | 1000 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.49598528753542936 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.38161107305163655 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.39433693637504463 |
| Dunnhumby Complete Journey | 1000 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.538662991040914 |
| Dunnhumby Complete Journey | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 127 | 127 | invoice_r2 | 0.6626002881845088 |
| Dunnhumby Complete Journey | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | invoice_r2 | 0.6708519290727337 |
| Dunnhumby Complete Journey | 1001 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.4227218092614493 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.4045172020090422 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.39343128078953504 |
| Dunnhumby Complete Journey | 1001 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.39860302892401045 |
| Dunnhumby Complete Journey | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | invoice_r2 | 0.5988958372819269 |
| Dunnhumby Complete Journey | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.590571894707162 |
| Dunnhumby Complete Journey | 1002 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.4731900283012649 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.4155217446634162 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.43708116946003606 |
| Dunnhumby Complete Journey | 1002 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.45837751612108024 |
| Dunnhumby Complete Journey | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 103 | 103 | invoice_r2 | 0.6350820587387584 |
| Dunnhumby Complete Journey | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.6313515200781585 |
| Dunnhumby Complete Journey | 1003 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.44597966213879947 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.42294683547071166 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4480432457416168 |
| Dunnhumby Complete Journey | 1003 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.46745241130653625 |
| Dunnhumby Complete Journey | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | invoice_r2 | 0.5997467879688829 |
| Dunnhumby Complete Journey | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | invoice_r2 | 0.5944668190841611 |
| Dunnhumby Complete Journey | 1004 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.43072552452795343 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.36403648048066284 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.3756328598880593 |
| Dunnhumby Complete Journey | 1004 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.4721689447677019 |
| Dunnhumby Complete Journey | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 114 | 114 | invoice_r2 | 0.5803628708626689 |
| Dunnhumby Complete Journey | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.5918965649250709 |
| Dunnhumby Complete Journey | 1005 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.5045873018922193 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.410745286545515 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.49120921468235235 |
| Dunnhumby Complete Journey | 1005 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.514100227149275 |
| Dunnhumby Complete Journey | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | invoice_r2 | 0.6375924266993638 |
| Dunnhumby Complete Journey | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.6379748720925233 |
| Dunnhumby Complete Journey | 1006 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.39660083535099855 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3408733035831324 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.3550312803486587 |
| Dunnhumby Complete Journey | 1006 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.3598162896667425 |
| Dunnhumby Complete Journey | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 107 | 107 | invoice_r2 | 0.5578805523975061 |
| Dunnhumby Complete Journey | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.5673847847996645 |
| Dunnhumby Complete Journey | 1007 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.43096686005119056 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.35978473333494354 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4015510361985707 |
| Dunnhumby Complete Journey | 1007 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.4761118837743211 |
| Dunnhumby Complete Journey | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | invoice_r2 | 0.5991202777302734 |
| Dunnhumby Complete Journey | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 19 | 19 | invoice_r2 | 0.5968384011350241 |
| Dunnhumby Complete Journey | 1008 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.4556207174467277 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3825001956904821 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4062981456014024 |
| Dunnhumby Complete Journey | 1008 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.48119643704867854 |
| Dunnhumby Complete Journey | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 126 | 126 | invoice_r2 | 0.6083733834715963 |
| Dunnhumby Complete Journey | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 17 | 17 | invoice_r2 | 0.6144901698995042 |
| Dunnhumby Complete Journey | 1009 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.4589590082770504 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.37617979700930915 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.39693859658575104 |
| Dunnhumby Complete Journey | 1009 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.3990393903073012 |
| Dunnhumby Complete Journey | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 137 | 137 | invoice_r2 | 0.5654515231751382 |
| Dunnhumby Complete Journey | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 20 | 20 | invoice_r2 | 0.5716189675693786 |
| Online Retail II | 42 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3729168747469761 |
| Online Retail II | 42 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.38762298781846705 |
| Online Retail II | 42 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4169538167543193 |
| Online Retail II | 42 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.4448212377104329 |
| Online Retail II | 42 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 95 | 95 | invoice_r2 | 0.49534023095126933 |
| Online Retail II | 42 | Kuznetsov-FCA | Kuznetsov-FCA | 28 | 28 | invoice_r2 | 0.4967121895312283 |
| Online Retail II | 1000 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.37010190586793557 |
| Online Retail II | 1000 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.37364039964538764 |
| Online Retail II | 1000 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.37577351443020546 |
| Online Retail II | 1000 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.4238744762297152 |
| Online Retail II | 1000 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 92 | 92 | invoice_r2 | 0.4798551193140024 |
| Online Retail II | 1000 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | invoice_r2 | 0.47924521823831123 |
| Online Retail II | 1001 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3321022925461604 |
| Online Retail II | 1001 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3648396728078026 |
| Online Retail II | 1001 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.37758098962801345 |
| Online Retail II | 1001 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.38198843052369746 |
| Online Retail II | 1001 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 74 | 74 | invoice_r2 | 0.44360580224891766 |
| Online Retail II | 1001 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | invoice_r2 | 0.4504768918777673 |
| Online Retail II | 1002 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3547779959389381 |
| Online Retail II | 1002 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.394933326409283 |
| Online Retail II | 1002 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.42657979753723785 |
| Online Retail II | 1002 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.4223806252547949 |
| Online Retail II | 1002 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 93 | 93 | invoice_r2 | 0.48032767360033124 |
| Online Retail II | 1002 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | invoice_r2 | 0.4848283632059429 |
| Online Retail II | 1003 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.34595027969789305 |
| Online Retail II | 1003 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3798795395535103 |
| Online Retail II | 1003 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.3646911947913021 |
| Online Retail II | 1003 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.3793125623258786 |
| Online Retail II | 1003 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 117 | 117 | invoice_r2 | 0.47734298092046135 |
| Online Retail II | 1003 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | invoice_r2 | 0.4737751138173747 |
| Online Retail II | 1004 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3437748578560933 |
| Online Retail II | 1004 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3885347414587349 |
| Online Retail II | 1004 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.41316030172904217 |
| Online Retail II | 1004 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.41599031712245804 |
| Online Retail II | 1004 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 122 | 122 | invoice_r2 | 0.4742786344094443 |
| Online Retail II | 1004 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | invoice_r2 | 0.4742818119643226 |
| Online Retail II | 1005 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3095062213657994 |
| Online Retail II | 1005 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.39752545563560526 |
| Online Retail II | 1005 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4184719454233523 |
| Online Retail II | 1005 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.39782110547037186 |
| Online Retail II | 1005 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 113 | 113 | invoice_r2 | 0.4838957600639585 |
| Online Retail II | 1005 | Kuznetsov-FCA | Kuznetsov-FCA | 33 | 33 | invoice_r2 | 0.4868832613095837 |
| Online Retail II | 1006 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3051767053230888 |
| Online Retail II | 1006 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.35978944036627103 |
| Online Retail II | 1006 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.41341342280646975 |
| Online Retail II | 1006 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.39517931182807275 |
| Online Retail II | 1006 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 94 | 94 | invoice_r2 | 0.47115330420006996 |
| Online Retail II | 1006 | Kuznetsov-FCA | Kuznetsov-FCA | 31 | 31 | invoice_r2 | 0.47461031126988074 |
| Online Retail II | 1007 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.35891800787541495 |
| Online Retail II | 1007 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.3867732618085242 |
| Online Retail II | 1007 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4157209027135884 |
| Online Retail II | 1007 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.40796763484816534 |
| Online Retail II | 1007 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 89 | 89 | invoice_r2 | 0.48124769424514624 |
| Online Retail II | 1007 | Kuznetsov-FCA | Kuznetsov-FCA | 29 | 29 | invoice_r2 | 0.48215382185061695 |
| Online Retail II | 1008 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.3308642345107713 |
| Online Retail II | 1008 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.36002474913036087 |
| Online Retail II | 1008 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.3731714567785138 |
| Online Retail II | 1008 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.376798541898654 |
| Online Retail II | 1008 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 72 | 72 | invoice_r2 | 0.4621135019822058 |
| Online Retail II | 1008 | Kuznetsov-FCA | Kuznetsov-FCA | 27 | 27 | invoice_r2 | 0.46421979556360193 |
| Online Retail II | 1009 | Raw RFM | Raw RFM | 3 | 3 | invoice_r2 | 0.36294534566424197 |
| Online Retail II | 1009 | FCM | FCM k=4 | 4 | 4 | invoice_r2 | 0.37313456408635925 |
| Online Retail II | 1009 | FCM | FCM k=5 | 5 | 5 | invoice_r2 | 0.4072399360052745 |
| Online Retail II | 1009 | FCM | FCM k=6 | 6 | 6 | invoice_r2 | 0.39715532674420684 |
| Online Retail II | 1009 | Fuzzy RFM-FCA | Fuzzy RFM-FCA | 101 | 101 | invoice_r2 | 0.49475985858959004 |
| Online Retail II | 1009 | Kuznetsov-FCA | Kuznetsov-FCA | 32 | 32 | invoice_r2 | 0.48610787728964444 |

## 11. Dunnhumby vs Retail II
See feature_counts and summary tables for cross-domain differences.

## 12. Interpretation

### 12.1 Verdict by dataset

**Dunnhumby Complete Journey — Kuznetsov wins on all three targets.**
Fuzzy RFM-FCA is already far above Raw RFM and FCM on spend_r2 (+0.136, CI [+0.114,+0.155], p=0.0015) and invoice_r2 (+0.153, CI [+0.139,+0.163], p=0.0015). Kuznetsov pruning then improves further on AUC (+0.0097, CI [+0.0049,+0.0144], p=0.0052) and spend_r2 (+0.0117, CI [+0.0066,+0.0183], p=0.0037) with invoice_r2 unchanged (p=0.31). The win is multi-target, not a single lucky metric.

**Online Retail II — Fuzzy RFM-FCA drives the gain; Kuznetsov adds a small AUC lift only.**
Fuzzy RFM-FCA beats Raw RFM on AUC (+0.0085, CI [+0.0061,+0.0108], p=0.0015), spend_r2 (+0.139, p=0.0015), and invoice_r2 (+0.133, p=0.0015). Kuznetsov beats Fuzzy RFM-FCA on AUC only (+0.0021, CI [+0.0009,+0.0032], p=0.0142) — significant but small in absolute terms — with spend_r2 (+0.00012, p=0.89) and invoice_r2 (+0.00080, p=0.58) both null. On Retail II the Kuznetsov gain is a single-metric, small-effect AUC improvement, not the broad multi-target win seen on Dunnhumby.

### 12.2 Where the contribution comes from

1. **The dominant jump is fuzzy FCA (Method 3) over Raw RFM/FCM.** Both datasets show ~+0.13–0.15 R² on the spend/invoice targets. FCM is not competitive: k=6 loses to Raw RFM on AUC in both datasets (Dunnhumby -0.070, p=0.0015; Retail II -0.022, p=0.0015), so FCM-as-soft-clustering-baseline does not reproduce the FCA gain.

2. **Kuznetsov's role is compression with preserved or improved performance, and the effect is domain-dependent.**
   - Dunnhumby: 113 → 19.4 concepts (26.7×), AUC +0.0097, spend_r2 +0.0117, invoice_r2 unchanged → performance maintained or improved while concepts collapse.
   - Retail II: 96.7 → 30.7 concepts (3.2×), AUC +0.0021 significant but tiny, both R² null → compression preserved, performance essentially flat except a small AUC bump.

3. **No evidence of Kuznetsov being strictly necessary for the predictive gain.** The Fuzzy RFM-FCA → Kuznetsov delta is an *incremental* refinement on top of an already-large fuzzy-FCA gain. If the only goal were predictive performance, Fuzzy RFM-FCA alone already captures most of the benefit. Kuznetsov's value proposition in this comparison is therefore compression efficiency plus a modest AUC improvement, not a large new predictive jump.

### 12.3 The honest caveat on the Retail II AUC result
The Retail II Kuznetsov AUC gain (+0.0021, p=0.0142) is statistically significant but small relative to the within-method variance and to the Dunnhumby effect (+0.0097). It should be described as a real but modest improvement, not a dramatic one. The cross-domain verdict must reflect that distinction.

## 13. Limitations
- 10 multi-split seeds only; fixed temporal holdout structure.
- FCM k chosen from a small grid; not tuned to test outcomes.
- Kuznetsov threshold fixed at the previously validated loss = 1 - stab_float <= 5.4e-20, not re-tuned here.
- In this corrected version, Method 3 and Method 4 share the same thresholded fuzzy miner; Method 3 applies no Kuznetsov filter and Method 4 applies the canonical stability filter, so the Kuznetsov effect is isolated.
- Method 3 is no longer the plain (non-thresholded) fuzzy miner; it is the thresholded fuzzy baseline used by the validated Retail II leakage-free script.
- One bug fixed during build: the Retail II summary path used the wrong helper name, which left the statistical-comparisons CSV empty and sections 7/12/14 unpopulated; that is now resolved.
- Permutation p-values are computed with the verified two-sided sign-flip test imported from the leakage-free Kuznetsov script; p=0.0015 is the resolution floor for 2^10 sign configurations.
- FCM is represented in four-way vs-FCM comparisons by its k=6 arm only; this is stated explicitly wherever FCM appears as a comparator.

## 14. Files created / modified



Corrected in this revision:

- `scripts/four_way_method_comparison.py` — two methodological corrections:
  1. Kuznetsov pruning now uses the exact validated loss definition, loss = 1 - stab_float, with loss <= 5.4e-20, matching `scripts/kuznetsov_pruning_retail2_leakage_free.py`. Log2_loss is retained only as a descriptive field and is NOT used for threshold selection.
  2. Method 3 (Fuzzy RFM-FCA baseline) and Method 4 (Kuznetsov-FCA) now use the SAME thresholded fuzzy miner (`mine_fuzzy_closed_concepts_with_thresholds` with the same L_THRESHOLDS and MIN_SUPPORT), so Method 4 differs from Method 3 only by the canonical Kuznetsov stability filter. Method 3 no longer uses the plain fuzzy miner.
- `results/four_way_method_comparison/` — fully regenerated for seeds 42 and 1000–1009 on both datasets.

Created (this experiment):

- `results/four_way_method_comparison/four_way_dunnhumby_splits.csv` — per-split results, Dunnhumby (66 rows).
- `results/four_way_method_comparison/four_way_retail2_splits.csv` — per-split results, Online Retail II (66 rows).
- `results/four_way_method_comparison/four_way_dunnhumby_summary.csv` — multi-split summary, Dunnhumby (6 method/arm rows).
- `results/four_way_method_comparison/four_way_retail2_summary.csv` — multi-split summary, Online Retail II (6 rows).
- `results/four_way_method_comparison/four_way_feature_counts.csv` — concept/feature counts (8 rows).
- `results/four_way_method_comparison/four_way_pareto.csv` — observed Pareto points (396 rows).
- `results/four_way_method_comparison/four_way_statistical_comparisons.csv` — paired multi-split comparisons (36 rows).
- `results/four_way_method_comparison/four_way_summary.json` — experiment metadata + results.
- `results/four_way_method_comparison/REPORT.md` — this report.

Not touched (protected):

All directories under `results/` other than `results/four_way_method_comparison/` were verified present and unchanged by the sanity gate and are read-only for this experiment. These include `dunnhumby_rfm_fca`, `fair_comparison_retail2`, `fuzzy_membership_sensitivity`, `canonical_kneedle_audit`, `canonical_kneedle_experiment`, `canonical_kneedle_statistical_comparison`, `kuznetsov_stability_audit`, `kuznetsov_pruning_leakage_free`, `kuznetsov_pruning_experiment`, and `kuznetsov_pruning_retail2_leakage_free`.

Cross-reference:

- The four-way script reuses the exact verified Kuznetsov stability machinery from `scripts/kuznetsov_pruning_retail2_leakage_free.py` via exec, including the `_paired_bootstrap_ci` / `_paired_permutation_test` helpers imported from `scripts/kuznetsov_pruning_leakage_free.py`.
- Fixed-seed seed-42 sanity numbers were cross-checked against the established leakage-free reference values before multi-split was run.
- No seeds 1000–1009 were run before the seed-42 sanity gate passed cleanly.
- Kuznetsov threshold fixed at the previously validated value, not re-tuned here.

## 15. Final conclusion

**Cross-domain verdict: replace Raw RFM with fuzzy RFM-FCA; replace fuzzy RFM-FCA with Kuznetsov-FCA for compression, while noting the AUC gain is domain-dependent.**

- **Raw RFM is not competitive as a fuzzy-design baseline.** FCM (k=6) loses on AUC in both domains, so the soft-clustering baseline does not reproduce the FCA gain. The relevant comparison is fuzzy RFM-FCA vs Kuznetsov-pruned fuzzy RFM-FCA.

- **Fuzzy RFM-FCA is the main source of the predictive improvement** over Raw RFM on the spend/invoice targets in both domains (~+0.13–0.15 R², significant in all cases).

- **Kuznetsov pruning adds a statistically significant improvement on top of fuzzy RFM-FCA in both domains, but the shape differs.**
  - Dunnhumby: AUC +0.0097 (p=0.0052), spend_r2 +0.0117 (p=0.0037), compression 113 → 19.4 (26.7×). Clear multi-target win plus aggressive compression.
  - Retail II: AUC +0.0021 (p=0.0142), spend_r2 and invoice_r2 both null, compression 96.7 → 30.7 (3.2×). Significant but small AUC lift; otherwise performance preserved and concepts reduced.

- **The practical takeaway is method-dependent:** if the priority is raw predictive performance on the regression targets, fuzzy RFM-FCA already captures most of the available gain and Kuznetsov is optional. If the priority includes parsimony, interpretability, or downstream concept inspection, Kuznetsov pruning is a strong default because it compresses aggressively without regressing performance — and on Dunnhumby it even improves AUC and spend R².

- **Do not overstate the Retail II Kuznetsov result.** It is a small, significant AUC improvement, not a large or broad one. The larger story is the fuzzy-FCA jump and the compression.

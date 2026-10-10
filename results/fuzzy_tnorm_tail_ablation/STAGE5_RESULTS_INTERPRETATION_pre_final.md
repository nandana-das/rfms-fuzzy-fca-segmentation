# Stage 5 Results Interpretation & Research Decision Audit (Revised)

**Project:** Fuzzy RFM-FCA Customer Segmentation  
**Evaluation Scope:** Stage 5 Factorial Representation Ablation (M0, M1, M2, M3)  
**Authoritative Evidence Source:** `results/fuzzy_tnorm_tail_ablation/multi_origin_metrics.csv`  
**Evaluation Units:** 9 Pooled Rolling Origins (Dunnhumby: 4 origins; Online Retail II: 5 origins)  
**Audit Status:** Complete, Methodologically Refined, Strictly Descriptive  

---

> [!IMPORTANT]
> **Audit & Methodological Principles:**
> 1. **Descriptive Nature of Summaries:** All summary statistics (means, medians, ranges, sign counts across origins) are **strictly descriptive point summaries** across historical rolling origins. Rolling-origin evaluations evaluate partially overlapping customer cohorts across sequential time windows; they do not represent independent and identically distributed replications, nor do they establish formal statistical significance, hypothesis rejection, or universal temporal generalization.
> 2. **Evaluated Factorial Arms:**
>    - **M0:** Frozen baseline fuzzy RFM-FCA using piecewise-linear memberships and Gödel-min aggregation ($\min$).
>    - **M1:** Tail-sensitive Log-Coordinate Asymptotic Rational Shoulder (LAR-PW) Frequency/Monetary memberships with Gödel-min aggregation ($\min$).
>    - **M2:** Baseline piecewise-linear memberships with Algebraic Product t-norm aggregation ($\prod$).
>    - **M3:** Tail-sensitive LAR-PW memberships with Algebraic Product t-norm aggregation ($\prod$).
> 3. **Metric Definitions:**
>    - Paired differences are calculated as $\Delta = \text{variant} - \text{M0}$ using full floating-point precision from `multi_origin_metrics.csv`.
>    - **ROC AUC** measures binary repurchase ranking discrimination on the held-out 91-day window.
>    - **Spend $R^2$** and **Invoice $R^2$** measure out-of-sample **explained variance** for continuous monetary spend and order frequency, respectively. They do **not** measure probability calibration or continuous prediction calibration.
> 4. **Prediction vs. Segmentation:** Predictive performance on downstream regression and classification tasks serves as an objective empirical proxy for feature expressiveness. It does **not** directly establish customer segmentation quality, cluster interpretability, segment distinctiveness, or managerial actionability.

---

## 1. Executive Summary

This research-methodology audit evaluated whether the additional mathematical complexity introduced by tail-sensitive memberships (LAR-PW) and algebraic product t-norm aggregation provides a consistent, interpretable empirical advantage over the frozen M0 baseline across all 9 rolling origins.

### Core Audit Findings

1. **ROC AUC (Repurchase Ranking Discrimination): No Consistent Variant Improvement on Primary Benchmark.**
   - On the primary confirmatory dataset (**Online Retail II**), the apparent ranking improvement observed during the initial Stage 4 smoke test (+0.0180 AUC on origin `2010-09-10`) proved to be an **isolated temporal outlier**. Across the subsequent four rolling origins (`2010-12-10` to `2011-09-09`), M1 AUC differences collapsed to near-zero ($\pm 0.0005$), resulting in positive $\Delta$ on only 3 of 5 origins (mean $\Delta = +0.003589$, median $\Delta = +0.000258$, range $[-0.000553, +0.017966]$). The positive mean is driven disproportionately by the single first origin.
   - On **Dunnhumby Complete Journey** (secondary replication), M1 showed modest positive differences across all 4 origins (mean $\Delta = +0.001586$, median $\Delta = +0.001603$). However, the product t-norm (M2) caused **uniform ranking degradation across all 4 origins** (mean $\Delta = -0.002870$, 0/4 positive), pulling joint arm M3 into negative territory as well (mean $\Delta = -0.002936$, 1/4 positive).
   - Neither variant satisfies the preregistered consistency standard ($\Delta > 0$ on $\ge 4$ of 5 origins on the primary confirmatory benchmark) for ranking improvement.

2. **Invoice $R^2$ (Continuous Volume Explained Variance): Substantial, Uniform Improvement for LAR-PW (M1).**
   - LAR-PW (M1) achieved higher out-of-sample explained variance across **all 9 evaluated origins in both datasets** (5/5 on Online Retail II, 4/4 on Dunnhumby).
   - On Online Retail II, M1 improved Invoice $R^2$ by $+0.0185$ to $+0.0314$ (mean $\Delta = +0.027647$, median $\Delta = +0.029530$). On Dunnhumby, M1 improved Invoice $R^2$ by $+0.0023$ to $+0.0043$ (mean $\Delta = +0.003379$, median $\Delta = +0.003447$).
   - By contrast, the product t-norm (M2) contributed negligible changes ($+0.001981$ mean on Retail II). Joint arm M3 matched M1 closely ($+0.028375$ mean on Retail II), indicating that the continuous explained variance gain is associated almost entirely with LAR-PW.
   - *Clarification:* These gains reflect higher out-of-sample explained variance ($R^2$) for purchase counts. They do not demonstrate that predictions are calibrated in the formal sense (e.g., matching observed conditional probabilities or empirical count quantiles), as calibration metrics were not evaluated.

3. **Spend $R^2$ (Monetary Value Explained Variance): Modest, Consistent Improvement for LAR-PW (M1).**
   - M1 achieved positive explained variance gains across all 9 origins (mean $\Delta = +0.002768$ on Retail II, $+0.001783$ on Dunnhumby).

4. **Factorial Synergy: Absence of Interaction.**
   - Factorial contrast analysis indicates that the interaction between tail memberships and product t-norms is non-positive or negligible across all metrics (mean interaction contrast for AUC: $-0.004085$ on Retail II, $-0.001652$ on Dunnhumby). M3 provides no synergistic empirical benefit over M1.

### Executive Research Verdict

| Arm | Research Status Recommendation | Justification |
| :--- | :--- | :--- |
| **M1 (LAR-PW Min)** | **Retain only as an exploratory comparison** | Shows a consistent and substantial improvement in out-of-sample explained variance for continuous purchase volume (Invoice $R^2$, 9/9 origins), but fails to provide consistent repurchase ranking gains (AUC, positive on only 3 of 5 origins on the primary benchmark). Retained in the research paper as an exploratory representation ablation for continuous-outcome prediction, **not** as a universally superior Fuzzy RFM-FCA method. |
| **M2 (M0 Product)** | **Do not prioritize for further work** | Secondary comparison only. Does not establish a compelling overall advantage: uniformly degrades AUC on Dunnhumby (4/4 negative), yields near-zero ranking change on Retail II outside the first origin, produces empty-core concepts (up to 14/fold), and provides negligible regression benefit. |
| **M3 (LAR-PW Product)**| **Do not prioritize for further work** | Secondary comparison only. Combines the continuous explained variance gains of M1 with the ranking degradation of M2; interaction contrasts are negative; adds structural complexity without empirical synergy. |

---

## 2. Descriptive Comparison of Factorial Arms against M0

### 2.1 Dunnhumby Complete Journey (4 Rolling Origins, $h = 91\text{d}$ — Secondary Replication)

All customer cohorts evaluate active loyalty cardholders ($N \approx 2{,}498$) under high repurchase prevalence ($\approx 92.2\% - 93.2\%$).

#### Table 1A: Dunnhumby ROC AUC (Repurchase Ranking Discrimination)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **day 347** | 0.89646238 | 0.89741139 | +0.00094901 | 0.89618838 | -0.00027401 | 0.89690125 | +0.00043886 |
| **day 438** | 0.88923988 | 0.89036254 | +0.00112266 | 0.88268244 | -0.00655744 | 0.88357728 | -0.00566260 |
| **day 529** | 0.90054127 | 0.90273129 | +0.00219002 | 0.89985123 | -0.00069004 | 0.89837571 | -0.00216555 |
| **day 620** | 0.87355057 | 0.87563303 | +0.00208245 | 0.86959165 | -0.00395892 | 0.86919726 | -0.00435331 |
| **Summary** | — | — | **Pos: 4, Zero: 0, Neg: 0** | — | **Pos: 0, Zero: 0, Neg: 4** | — | **Pos: 1, Zero: 0, Neg: 3** |
| **Mean $\Delta$** | — | — | **+0.00158603** | — | **-0.00287010** | — | **-0.00293565** |
| **Median $\Delta$**| — | — | **+0.00160256** | — | **-0.00232448** | — | **-0.00325943** |
| **Min $\Delta$** | — | — | **+0.00094901** | — | **-0.00655744** | — | **-0.00566260** |
| **Max $\Delta$** | — | — | **+0.00219002** | — | **-0.00027401** | — | **+0.00043886** |
| **Consistency**| — | — | **Consistent (Positive)** | — | **Consistent (Negative)** | — | **Inconsistent** |

#### Table 1B: Dunnhumby Spend $R^2$ (Monetary Value Out-of-Sample Explained Variance)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **day 347** | 0.54993314 | 0.55072818 | +0.00079503 | 0.55059588 | +0.00066274 | 0.55170424 | +0.00177109 |
| **day 438** | 0.54814528 | 0.54876809 | +0.00062281 | 0.55008426 | +0.00193897 | 0.55209842 | +0.00395313 |
| **day 529** | 0.53781990 | 0.53981108 | +0.00199119 | 0.53920940 | +0.00138950 | 0.54039916 | +0.00257926 |
| **day 620** | 0.51685288 | 0.52057680 | +0.00372392 | 0.52135670 | +0.00450381 | 0.52364472 | +0.00679184 |
| **Summary** | — | — | **Pos: 4, Zero: 0, Neg: 0** | — | **Pos: 4, Zero: 0, Neg: 0** | — | **Pos: 4, Zero: 0, Neg: 0** |
| **Mean $\Delta$** | — | — | **+0.00178324** | — | **+0.00212376** | — | **+0.00377383** |
| **Median $\Delta$**| — | — | **+0.00139311** | — | **+0.00166424** | — | **+0.00326620** |
| **Min $\Delta$** | — | — | **+0.00062281** | — | **+0.00066274** | — | **+0.00177109** |
| **Max $\Delta$** | — | — | **+0.00372392** | — | **+0.00450381** | — | **+0.00679184** |
| **Consistency**| — | — | **Consistent (Positive)** | — | **Consistent (Positive)** | — | **Consistent (Positive)** |

#### Table 1C: Dunnhumby Invoice $R^2$ (Purchase Volume Out-of-Sample Explained Variance)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **day 347** | 0.62443149 | 0.62671214 | +0.00228065 | 0.62492053 | +0.00048904 | 0.62854349 | +0.00411200 |
| **day 438** | 0.62990786 | 0.63263542 | +0.00272756 | 0.63140445 | +0.00149659 | 0.63460371 | +0.00469584 |
| **day 529** | 0.61662655 | 0.62079214 | +0.00416559 | 0.61790384 | +0.00127729 | 0.62161474 | +0.00498820 |
| **day 620** | 0.61969128 | 0.62403321 | +0.00434193 | 0.62317527 | +0.00348399 | 0.62655252 | +0.00686124 |
| **Summary** | — | — | **Pos: 4, Zero: 0, Neg: 0** | — | **Pos: 4, Zero: 0, Neg: 0** | — | **Pos: 4, Zero: 0, Neg: 0** |
| **Mean $\Delta$** | — | — | **+0.00337893** | — | **+0.00168673** | — | **+0.00516432** |
| **Median $\Delta$**| — | — | **+0.00344658** | — | **+0.00138694** | — | **+0.00484202** |
| **Min $\Delta$** | — | — | **+0.00228065** | — | **+0.00048904** | — | **+0.00411200** |
| **Max $\Delta$** | — | — | **+0.00434193** | — | **+0.00348399** | — | **+0.00686124** |
| **Consistency**| — | — | **Consistent (Positive)** | — | **Consistent (Positive)** | — | **Consistent (Positive)** |

---

### 2.2 Online Retail II (5 Rolling Origins, $h = 91\text{d}$ — Primary Confirmatory Benchmark)

Customer cohort sizes expand from $N = 3{,}377$ to $N = 5{,}281$, with repurchase prevalence varying from $30.6\%$ to $57.7\%$.

#### Table 2A: Online Retail II ROC AUC (Repurchase Ranking Discrimination)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-09-10** | 0.70639580 | 0.72436217 | +0.01796637 | 0.72404796 | +0.01765216 | 0.72242625 | +0.01603045 |
| **2010-12-10** | 0.79581982 | 0.79607742 | +0.00025760 | 0.79633098 | +0.00051116 | 0.79579895 | -0.00002087 |
| **2011-03-11** | 0.78565139 | 0.78509824 | -0.00055315 | 0.78523986 | -0.00041153 | 0.78362634 | -0.00202505 |
| **2011-06-10** | 0.81813662 | 0.81784541 | -0.00029120 | 0.81862856 | +0.00049194 | 0.81901541 | +0.00087879 |
| **2011-09-09** | 0.79325824 | 0.79382547 | +0.00056723 | 0.79326247 | +0.00000423 | 0.79416382 | +0.00090559 |
| **Summary** | — | — | **Pos: 3, Zero: 0, Neg: 2** | — | **Pos: 4, Zero: 0, Neg: 1** | — | **Pos: 3, Zero: 0, Neg: 2** |
| **Mean $\Delta$** | — | — | **+0.00358937** | — | **+0.00364959** | — | **+0.00315378** |
| **Median $\Delta$**| — | — | **+0.00025760** | — | **+0.00049194** | — | **+0.00087879** |
| **Min $\Delta$** | — | — | **-0.00055315** | — | **-0.00041153** | — | **-0.00202505** |
| **Max $\Delta$** | — | — | **+0.01796637** | — | **+0.01765216** | — | **+0.01603045** |
| **Consistency**| — | — | **Inconsistent** | — | **Inconsistent** | — | **Inconsistent** |

> [!NOTE]
> **Observation on Online Retail II Ranking Differences:**
> The mean difference for M1 (+0.003589) is driven disproportionately by the single first origin (`2010-09-10`, $\Delta = +0.017966$). Across the remaining four origins, the differences are small and fluctuate around zero (median $\Delta = +0.000258$, range $[-0.000553, +0.000567]$), with two of the five origins showing negative differences. These results do not establish a consistent improvement in repurchase ranking on the primary benchmark.

#### Table 2B: Online Retail II Spend $R^2$ (Monetary Value Out-of-Sample Explained Variance)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-09-10** | 0.23239894 | 0.23355449 | +0.00115555 | 0.23158127 | -0.00081767 | 0.23260207 | +0.00020313 |
| **2010-12-10** | 0.31966139 | 0.32261428 | +0.00295289 | 0.32007293 | +0.00041154 | 0.32362750 | +0.00396612 |
| **2011-03-11** | 0.31636656 | 0.32016074 | +0.00379418 | 0.31713471 | +0.00076815 | 0.31985840 | +0.00349183 |
| **2011-06-10** | 0.36189486 | 0.36511797 | +0.00322311 | 0.36424914 | +0.00235428 | 0.37019955 | +0.00830469 |
| **2011-09-09** | 0.32659932 | 0.32931157 | +0.00271225 | 0.32738021 | +0.00078088 | 0.33115497 | +0.00455565 |
| **Summary** | — | — | **Pos: 5, Zero: 0, Neg: 0** | — | **Pos: 4, Zero: 0, Neg: 1** | — | **Pos: 5, Zero: 0, Neg: 0** |
| **Mean $\Delta$** | — | — | **+0.00276760** | — | **+0.00069944** | — | **+0.00410428** |
| **Median $\Delta$**| — | — | **+0.00295289** | — | **+0.00076815** | — | **+0.00396612** |
| **Min $\Delta$** | — | — | **+0.00115555** | — | **-0.00081767** | — | **+0.00020313** |
| **Max $\Delta$** | — | — | **+0.00379418** | — | **+0.00235428** | — | **+0.00830469** |
| **Consistency**| — | — | **Consistent (Positive)** | — | **Inconsistent** | — | **Consistent (Positive)** |

#### Table 2C: Online Retail II Invoice $R^2$ (Purchase Volume Out-of-Sample Explained Variance)

| Origin | M0 Baseline | M1 (LAR-PW Min) | M1 $\Delta$ | M2 (M0 Product) | M2 $\Delta$ | M3 (LAR-PW Product) | M3 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-09-10** | 0.31574264 | 0.33423071 | +0.01848807 | 0.31624779 | +0.00050515 | 0.33636700 | +0.02062436 |
| **2010-12-10** | 0.36973284 | 0.39926257 | +0.02952973 | 0.37138259 | +0.00164975 | 0.39521152 | +0.02547868 |
| **2011-03-11** | 0.38055974 | 0.41193727 | +0.03137754 | 0.38177068 | +0.00121095 | 0.40993521 | +0.02937548 |
| **2011-06-10** | 0.41193387 | 0.44101132 | +0.02907745 | 0.41582923 | +0.00389537 | 0.44580104 | +0.03386717 |
| **2011-09-09** | 0.39035663 | 0.42011913 | +0.02976250 | 0.39299962 | +0.00264299 | 0.42288541 | +0.03252878 |
| **Summary** | — | — | **Pos: 5, Zero: 0, Neg: 0** | — | **Pos: 5, Zero: 0, Neg: 0** | — | **Pos: 5, Zero: 0, Neg: 0** |
| **Mean $\Delta$** | — | — | **+0.02764706** | — | **+0.00198084** | — | **+0.02837489** |
| **Median $\Delta$**| — | — | **+0.02952973** | — | **+0.00164975** | — | **+0.02937548** |
| **Min $\Delta$** | — | — | **+0.01848807** | — | **+0.00050515** | — | **+0.02062436** |
| **Max $\Delta$** | — | — | **+0.03137754** | — | **+0.00389537** | — | **+0.03386717** |
| **Consistency**| — | — | **Consistent (Positive)** | — | **Consistent (Positive)** | — | **Consistent (Positive)** |

---

## 3. Cross-Dataset Pattern Synthesis

| Metric | Arm M1 (LAR-PW Min) | Arm M2 (M0 Product) | Arm M3 (LAR-PW Product) | Cross-Dataset Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **ROC AUC** | **Mixed / Inconsistent**<br>• Dunnhumby: 4/4 pos ($+0.00159$ mean)<br>• Retail II: 3/5 pos, 2/5 neg ($+0.00026$ median, $+0.00359$ mean) | **Regressive / Harmful**<br>• Dunnhumby: 4/4 neg ($-0.00287$ mean)<br>• Retail II: 4/5 pos ($+0.00049$ median, $+0.00365$ mean) | **Regressive / Mixed**<br>• Dunnhumby: 3/4 neg ($-0.00294$ mean)<br>• Retail II: 3/5 pos, 2/5 neg ($+0.00088$ median, $+0.00315$ mean) | Neither representation alteration reliably improves customer ranking order across datasets. The initial smoke-origin difference on Retail II (+0.0180) did not replicate in subsequent cohorts. Product t-norm harms ranking on Dunnhumby. |
| **Spend $R^2$** | **Consistent Explained Variance Increase**<br>• Dunnhumby: 4/4 pos ($+0.00178$ mean)<br>• Retail II: 5/5 pos ($+0.00277$ mean) | **Weak / Mixed**<br>• Dunnhumby: 4/4 pos ($+0.00212$ mean)<br>• Retail II: 4/5 pos, 1/5 neg ($+0.00070$ mean) | **Consistent Explained Variance Increase**<br>• Dunnhumby: 4/4 pos ($+0.00377$ mean)<br>• Retail II: 5/5 pos ($+0.00410$ mean) | Modest, consistent increases in out-of-sample explained variance for monetary spend across both datasets for LAR-PW arms. |
| **Invoice $R^2$**| **Substantial, Consistent Explained Variance Increase**<br>• Dunnhumby: 4/4 pos ($+0.00338$ mean)<br>• Retail II: 5/5 pos ($+0.02765$ mean) | **Negligible Gain**<br>• Dunnhumby: 4/4 pos ($+0.00169$ mean)<br>• Retail II: 5/5 pos ($+0.00198$ mean) | **Substantial, Consistent Explained Variance Increase**<br>• Dunnhumby: 4/4 pos ($+0.00516$ mean)<br>• Retail II: 5/5 pos ($+0.02837$ mean) | Non-saturating memberships systematically increase explained variance for continuous purchase volume (9/9 origins). Gain is driven almost entirely by LAR-PW, not the product t-norm. |

---

## 4. Factorial Design Analysis: Main Effects & Interaction

The experiment implements a $2 \times 2$ factorial structure:
- **Factor 1:** Membership representation (Baseline Piecewise-Linear vs. LAR-PW).
- **Factor 2:** T-norm aggregation operator (Gödel Minimum vs. Algebraic Product).

```
                 Gödel Minimum (T_min)       Algebraic Product (T_prod)
Baseline (M_bl)          M0                             M2
LAR-PW (M_lar)           M1                             M3
```

### Table 3: Descriptive Factorial Contrasts (Means across Origins)

| Dataset | Metric | Factor 1 at Min<br>$(M1 - M0)$ | Factor 1 at Prod<br>$(M3 - M2)$ | Factor 2 at BL<br>$(M2 - M0)$ | Factor 2 at LAR<br>$(M3 - M1)$ | Interaction Contrast<br>$(M3 - M1) - (M2 - M0)$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dunnhumby** | **ROC AUC** | +0.00158603 | -0.00006555 | -0.00287010 | -0.00452168 | **-0.00165158** |
| | **Spend $R^2$** | +0.00178324 | +0.00165007 | +0.00212376 | +0.00199059 | **-0.00013316** |
| | **Invoice $R^2$** | +0.00337893 | +0.00347759 | +0.00168673 | +0.00178539 | **+0.00009866** |
| **Online Retail II** | **ROC AUC** | +0.00358937 | -0.00049581 | +0.00364959 | -0.00043559 | **-0.00408518** |
| | **Spend $R^2$** | +0.00276760 | +0.00340485 | +0.00069944 | +0.00133669 | **+0.00063725** |
| | **Invoice $R^2$** | +0.02764706 | +0.02639405 | +0.00198084 | +0.00072784 | **-0.00125301** |

### Factorial Interpretations (Plausible Hypotheses, Not Causal Claims)

1. **Factor 1 (LAR-PW Representation): Continuous Volume Association, Ranking Inconsistency.**
   - Regardless of whether aggregation uses Gödel-min or Product t-norm, LAR-PW is associated with higher Invoice $R^2$ across origins ($+0.0276$ at Min, $+0.0264$ at Prod on Retail II).
   - In contrast, for ROC AUC, the observed difference for LAR-PW under Product t-norm is slightly negative on average (mean $-0.00007$ on Dunnhumby, $-0.00050$ on Retail II). Under Gödel-min, the median AUC difference across origins on Online Retail II is near zero ($+0.00026$).
   - *Plausible Interpretation:* Non-saturating memberships preserve quantitative order volume in the feature space, which may provide more linear information for continuous regressions ($R^2$). They do not, however, reliably alter binary customer rank order for repurchase.

2. **Factor 2 (Algebraic Product T-Norm): Ranking Degradation on Dunnhumby, Negligible Calibration Gain.**
   - Product t-norm is associated with lower AUC on Dunnhumby under both membership regimes ($-0.00287$ at BL, $-0.00452$ at LAR). On Retail II, when evaluated with LAR-PW, the t-norm difference is negative across 3 of 5 origins (mean $-0.00044$).
   - On Invoice $R^2$, product t-norm shows small differences ($+0.0020$ on Retail II vs $+0.0276$ for LAR-PW).
   - *Plausible Interpretation:* Multiplicative conjunction contracts feature magnitudes ($x_C \to 0$ as intent size increases). Under uniform L2 regularization, this may attenuate multi-attribute concepts, leading to empty cores and degraded ranking discrimination.

3. **Interaction Contrast: Lack of Compounding Synergy.**
   - The descriptive interaction contrast $(M3 - M1) - (M2 - M0)$ is negative for ROC AUC on both datasets ($-0.00165$ on Dunnhumby, $-0.00409$ on Retail II) and near zero for $R^2$ metrics.
   - *Plausible Interpretation:* The two modifications do not show evidence of positive interaction. Combining product t-norms with tail memberships does not confer an empirical advantage over tail memberships alone.

---

## 5. Diagnostic Interpretation: Representation Differences vs. Predictive and Segmentation Quality

Diagnostics from [fold_diagnostics.csv](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/fuzzy_tnorm_tail_ablation/fold_diagnostics.csv) and [tail_diagnostics.csv](file:///d:/Nandana/MTECH/Semester%203/Projects/MP/rfms_fca_project/results/fuzzy_tnorm_tail_ablation/tail_diagnostics.csv) confirm representation behavior, but do not establish causal links to prediction or segmentation quality:

| Diagnostic Metric | Observed Value | What It Establishes | What It Does NOT Establish |
| :--- | :--- | :--- | :--- |
| **$D_{\text{mem}}$ (Continuous Tail Uniqueness)** | **1.000** across all 9 origins (0 tied raw tail pairs across 436k – 1.68M pairs) | Establishes that the LAR-PW mathematical formulation eliminates numerical saturation in continuous membership space. Distinct raw values yield distinct continuous membership values. | **Does NOT prove predictive or segmentation superiority.** Continuous separation is an arithmetic property of the rational function; it does not prove that this separation caused the observed $R^2$ changes or that downstream models placed useful weight on those distinctions. |
| **$D_{\text{ctx}}$ (Distinct Formal Context Profiles)** | **264 to 307** distinct 45-column binary profiles in the upper tail | Establishes that after threshold quantization ($\alpha \in \{0.3, 0.5, 0.7\}$), substantial combinatorial variety survives in the binary context. | **Does NOT establish segment utility or interpretability.** A higher count of binary configurations may reflect sampling noise or fine-grained variations that do not correspond to meaningful, cohesive, or actionable customer segments. |
| **$H_{\text{ctx}}$ (Formal Context Entropy)** | **6.66 to 7.45 bits** across all origins | Establishes that tail customer profiles are distributed across configurations rather than concentrated in a few dominant patterns. | **Does NOT establish customer segmentation quality.** High attribute entropy does not indicate clearer managerial separation, higher cluster validity, or better segment stability. |
| **Candidate Concepts ($n_{\text{candidates}}$)** | **M1/M3: 388 – 593**<br>M0/M2: 270 – 495 | Demonstrates that LAR-PW produces 25–40% more candidate concepts meeting the support threshold during FP-growth. | **Does NOT establish expanded effective capacity.** Kuznetsov closure and Jaccard suppression ($\ge 0.80$) compress the retained basis back to the same effective dimensionality (70–95 features across all arms). Concept redundancy is a *plausible structural explanation* for the lack of AUC gains, not a proven causal mechanism. |
| **Empty Core Concepts ($n_{\text{empty\_core}}$)** | **M0/M1: 0.0**<br>M2: 2.4 – 8.8<br>M3: 2.8 – 6.6 | Establishes that product t-norms yield concepts where no customer satisfies all intent attributes simultaneously at $\mu_C \ge 0.5$. | Explains the structural shrinkage of multiplicative conjunction, but does not isolate it as the sole causal driver of ranking degradation under regularized linear modeling. |

---

## 6. Research Recommendations

### Variant M1 (Tail-Sensitive LAR-PW Memberships + Gödel-min Aggregation)
- **Recommendation: Retain only as an exploratory comparison.**
- **Research Role:** Retain in the research paper as an exploratory representation ablation for continuous-outcome prediction ($R^2$), **not** as a replacement baseline or universally superior Fuzzy RFM-FCA method.
- **Methodological Justification:**
  1. *Inconsistent Ranking Results:* On the primary confirmatory benchmark (Online Retail II), M1 produces positive AUC differences on only 3 of 5 origins (mean $\Delta = +0.003589$, median $\Delta = +0.000258$). The positive mean is driven disproportionately by the single first origin (+0.0180). This fails the preregistered consistency standard ($\ge 4$ of 5 origins positive) for claiming ranking improvement.
  2. *Consistent Continuous Explained Variance Gain:* Across all 9 origins in both datasets, M1 achieves higher Invoice $R^2$ ($+0.0185$ to $+0.0314$ on Retail II; $+0.0023$ to $+0.0043$ on Dunnhumby). This shows that non-saturating memberships capture additional variance in continuous purchase volume, which is relevant for continuous-outcome ablations.
  3. *Structural Integrity:* M1 retains zero empty-core concepts, preserves Ruspini partitions of unity, and maintains concept lattice stability under Jaccard suppression.
  4. *Separation from Segmentation Quality:* Higher explained variance in downstream regression does not establish that the resulting customer segments are more managerially actionable, distinct, or interpretable.

### Variant M2 (Baseline Memberships + Algebraic Product T-Norm)
- **Recommendation: Do not prioritize for further work.**
- **Research Role:** Secondary comparison only. Report briefly in an ablation table as an aggregation variant that does not provide a compelling advantage.
- **Methodological Justification:**
  1. *Ranking Degradation on Secondary Replication:* Consistently degrades ROC AUC on Dunnhumby across all 4 origins (mean $\Delta = -0.002870$, 0/4 positive). On Online Retail II, median AUC change is near zero ($+0.000492$).
  2. *Negligible Regression Benefit:* Mean Invoice $R^2$ improvement is only $+0.001981$ on Retail II (less than one-tenth the difference observed for M1).
  3. *Feature Shrinkage:* Induces empty-core concepts (up to 14 concepts per fold) due to multiplicative feature attenuation.

### Variant M3 (Tail-Sensitive LAR-PW Memberships + Algebraic Product T-Norm)
- **Recommendation: Do not prioritize for further work.**
- **Research Role:** Secondary comparison only. Report in the factorial table to confirm the lack of compounding empirical synergy.
- **Methodological Justification:**
  1. *No Synergy:* Interaction contrasts are non-positive across all metrics.
  2. *Inherits T-Norm Drawbacks:* Degrades Dunnhumby AUC (negative in 3/4 origins) and produces empty-core concepts.
  3. *Complexity without Distinct Advantage:* The observed explained variance gains in Invoice $R^2$ are driven almost entirely by the main effect of LAR-PW. Adding product t-norm aggregation increases architectural complexity without conferring a clear empirical advantage over M1.

---

## 7. Results Narrative for Research Manuscript

The following concise, defensible narrative reflects the audited findings, qualified claims, and the distinction between primary benchmark and secondary replication:

> *"To evaluate whether upper-tail membership saturation and Gödel minimum aggregation constrain the predictive expressiveness of Fuzzy RFM-FCA, we evaluated a preregistered $2 \times 2$ factorial representation ablation across nine rolling-origin cohorts spanning two independent retail benchmarks: Online Retail II (five rolling origins, designated as the primary benchmark) and Dunnhumby Complete Journey (four rolling origins, evaluated as secondary replication). We compared the frozen M0 baseline against tail-sensitive Log-Coordinate Asymptotic Rational (LAR-PW) memberships (M1), algebraic product t-norm aggregation (M2), and their combination (M3).*
>
> *Across all nine evaluated origins, the tail-sensitive LAR-PW representation (M1) was associated with higher out-of-sample explained variance for continuous purchase volume, increasing Invoice $R^2$ by an average of $+0.0276$ on Online Retail II (median $+0.0295$, range $[+0.0185, +0.0314]$ across five origins) and $+0.0034$ on Dunnhumby (median $+0.0034$, range $[+0.0023, +0.0043]$ across four origins). Modest positive differences were also observed for Monetary Spend $R^2$ ($+0.0028 and $+0.0018$ descriptive means, respectively). These $R^2$ gains reflect improvements in continuous out-of-sample explained variance rather than verified prediction calibration, as formal probability calibration was not assessed.*
>
> *However, this increase in continuous explained variance did not translate into a consistent advantage in customer repurchase ranking order. On the primary Online Retail II benchmark, M1 exceeded M0 in ROC AUC on only three of five rolling origins (mean $\Delta = +0.00359$, median $\Delta = +0.00026$, range $[-0.00055, +0.01797]$). The positive mean difference was driven disproportionately by the single first origin (+0.0180), while differences across the remaining four origins fluctuated narrowly around zero. On the secondary Dunnhumby benchmark, M1 showed modest positive AUC differences across all four origins (mean $\Delta = +0.00159$, range $[+0.00095, +0.00219]$).*
>
> *The algebraic product t-norm (M2) showed no compelling advantage: it degraded Dunnhumby AUC across all four origins (mean $\Delta = -0.00287$) and produced negligible regression differences while generating empty-core concepts. The combined arm (M3) showed no positive factorial synergy, with its performance tracking the main effect of LAR-PW.*
>
> *Diagnostic measurements confirmed that while LAR-PW eliminated upper-tail continuous membership saturation ($D_{\text{mem}} = 1.000$) and increased candidate concept counts by 25–40%, Jaccard suppression ($\ge 0.80$) compressed the retained concept feature matrix to the same effective dimensionality as the baseline (70–95 features). While structural concept redundancy provides a plausible interpretation for the ranking plateau, these representation diagnostics describe pipeline behavior rather than established causal mechanisms. Overall, M1 warrants inclusion only as an exploratory representation ablation for continuous-outcome modeling; the evidence does not support adopting M1 as a universally superior Fuzzy RFM-FCA baseline, nor does predictive accuracy directly imply superior customer segmentation quality."*

---

## 8. Methodological Limitations

1. **Explained Variance vs. Calibration:**
   Invoice $R^2$ and Spend $R^2$ quantify out-of-sample explained variance for continuous outcomes under linear regularized modeling. They do not demonstrate prediction calibration (such as calibration slope, intercept, or reliability diagrams). Claims of "calibrated predictions" are not supported without dedicated calibration testing.

2. **Sequential Temporal Overlap:**
   Rolling-origin cohorts represent sequential historical windows where customer identities partially overlap across origins. Summary statistics across origins are descriptive point measures; they must not be interpreted as independent replications or asymptotic guarantees.

3. **Sensitivity of Averages to Single Origins:**
   On Online Retail II, the first origin (`2010-09-10`) exhibited a large AUC difference (+0.0180) that did not replicate in subsequent time windows (origins 2–5 spanned $[-0.00055, +0.00057]$). Reporting means alongside medians and full ranges is essential to prevent misleading impressions of consistency.

4. **Prediction Proxies vs. Segmentation Quality:**
   Downstream predictive metrics (ROC AUC, $R^2$) evaluate feature expressiveness on supervised regression and classification tasks. They do not directly measure the managerial actionability, clustering validity, stability, or qualitative interpretability of the mined customer segments. High predictive scores do not guarantee better customer segmentation, and vice versa.

5. **Representation Differences vs. Causal Mechanisms:**
   Diagnostic metrics ($D_{\text{mem}}$, $D_{\text{ctx}}$, $n_{\text{candidates}}$, $n_{\text{features}}$) measure structural characteristics of the data pipeline. While concept redundancy and L2 regularization attenuation are plausible explanations for observed performance patterns, descriptive observational point estimates do not prove causal mechanisms.

---

**Report Completion Timestamp:** 2026-10-10T13:45:00Z  
**Audit Finding:** Revised report verified against `multi_origin_metrics.csv`. All numerical values preserved. All claims strictly descriptive and qualified.

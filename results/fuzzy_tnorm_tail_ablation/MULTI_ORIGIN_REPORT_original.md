# Fuzzy RFM-FCA Factorial Ablation: Multi-Origin Validation Report

- **Execution Timestamp:** 2026-10-10T12:48:34.419606+00:00
- **Status:** ALL VERIFICATION CHECKS PASSED
- **Evaluated Origins:** 9 pooled origins (Dunnhumby: 4, Online Retail II: 5)
- **Evaluated Arms:** M0 (Baseline), M1 (LAR-PW Gödel-min), M2 (M0 Product t-norm), M3 (LAR-PW Product t-norm)
- **M0 Parity Max Absolute Error:** 8.33e-17 (Tolerance: <= 1e-12)
- **CDNOW Status:** Deferred (no existing 5-level frozen M0 baseline in `baseline_ladder_rolling_origin/per_origin_metrics.csv`)

> [!IMPORTANT]
> **Methodological Notice:** The metrics below represent multi-origin exploratory validation of the
> preregistered factorial representation arms. In accordance with the preregistration protocol,
> **point differences are presented descriptively and must NOT be interpreted as confirmed statistical
> significance, model superiority, or universal temporal generalization** prior to formal customer-clustered inference.

---

## 1. M0 Parity Verification against Frozen Baseline

| dataset          | origin     | arm | metric     | computed_value      | frozen_value       | abs_difference        | tolerance | passed |
| ---------------- | ---------- | --- | ---------- | ------------------- | ------------------ | --------------------- | --------- | ------ |
| Dunnhumby        | day 347    | M0  | auc        | 0.8964623849940966  | 0.8964623849940966 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | M0  | spend_r2   | 0.5499331432839238  | 0.5499331432839238 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | M0  | invoice_r2 | 0.6244314853034746  | 0.6244314853034746 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 438    | M0  | auc        | 0.8892398820036826  | 0.8892398820036826 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 438    | M0  | spend_r2   | 0.5481452824141043  | 0.5481452824141043 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 438    | M0  | invoice_r2 | 0.6299078614180956  | 0.6299078614180956 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 529    | M0  | auc        | 0.9005412653668468  | 0.9005412653668468 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 529    | M0  | spend_r2   | 0.5378198958615141  | 0.5378198958615141 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 529    | M0  | invoice_r2 | 0.616626548597684   | 0.616626548597684  | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 620    | M0  | auc        | 0.8735505717328832  | 0.8735505717328832 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 620    | M0  | spend_r2   | 0.5168528844206599  | 0.5168528844206599 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 620    | M0  | invoice_r2 | 0.6196912828702221  | 0.6196912828702221 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | auc        | 0.7063957958760724  | 0.7063957958760724 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | spend_r2   | 0.23239893990983218 | 0.2323989399098321 | 8.326672684688674e-17 | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | invoice_r2 | 0.31574263699112215 | 0.3157426369911221 | 5.551115123125783e-17 | 1e-12     | True   |
| Online Retail II | 2010-12-10 | M0  | auc        | 0.795819819801591   | 0.795819819801591  | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-12-10 | M0  | spend_r2   | 0.3196613880614364  | 0.3196613880614364 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-12-10 | M0  | invoice_r2 | 0.3697328385274994  | 0.3697328385274994 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-03-11 | M0  | auc        | 0.7856513875966153  | 0.7856513875966153 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-03-11 | M0  | spend_r2   | 0.31636656237728855 | 0.3163665623772885 | 5.551115123125783e-17 | 1e-12     | True   |
| Online Retail II | 2011-03-11 | M0  | invoice_r2 | 0.3805597355153144  | 0.3805597355153144 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-06-10 | M0  | auc        | 0.8181366171657094  | 0.8181366171657094 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-06-10 | M0  | spend_r2   | 0.36189486279726335 | 0.3618948627972633 | 5.551115123125783e-17 | 1e-12     | True   |
| Online Retail II | 2011-06-10 | M0  | invoice_r2 | 0.4119338681461583  | 0.4119338681461583 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-09-09 | M0  | auc        | 0.793258235402993   | 0.793258235402993  | 0.0                   | 1e-12     | True   |
| Online Retail II | 2011-09-09 | M0  | spend_r2   | 0.32659932497453004 | 0.32659932497453   | 5.551115123125783e-17 | 1e-12     | True   |
| Online Retail II | 2011-09-09 | M0  | invoice_r2 | 0.3903566283455032  | 0.3903566283455032 | 0.0                   | 1e-12     | True   |

---

## 2. Per-Origin Factorial Ablation Metrics

| dataset          | origin     | n    | repurchase_rate     | arm | membership_type     | tnorm     | auc                | spend_r2            | invoice_r2          |
| ---------------- | ---------- | ---- | ------------------- | --- | ------------------- | --------- | ------------------ | ------------------- | ------------------- |
| Dunnhumby        | day 347    | 2497 | 0.921906287545054   | M0  | m0_piecewise_linear | godel_min | 0.8964623849940966 | 0.5499331432839238  | 0.6244314853034746  |
| Dunnhumby        | day 347    | 2497 | 0.921906287545054   | M1  | lar_pw              | godel_min | 0.8974113925460581 | 0.5507281762742595  | 0.6267121378863203  |
| Dunnhumby        | day 347    | 2497 | 0.921906287545054   | M2  | m0_piecewise_linear | product   | 0.8961883757713471 | 0.5505958797699582  | 0.6249205286100592  |
| Dunnhumby        | day 347    | 2497 | 0.921906287545054   | M3  | lar_pw              | product   | 0.8969012452939473 | 0.5517042372028425  | 0.6285434899909708  |
| Dunnhumby        | day 438    | 2498 | 0.9263410728582866  | M0  | m0_piecewise_linear | godel_min | 0.8892398820036826 | 0.5481452824141043  | 0.6299078614180956  |
| Dunnhumby        | day 438    | 2498 | 0.9263410728582866  | M1  | lar_pw              | godel_min | 0.8903625380481756 | 0.5487680939862771  | 0.6326354248603234  |
| Dunnhumby        | day 438    | 2498 | 0.9263410728582866  | M2  | m0_piecewise_linear | product   | 0.8826824433504942 | 0.5500842562226649  | 0.6314044480060568  |
| Dunnhumby        | day 438    | 2498 | 0.9263410728582866  | M3  | lar_pw              | product   | 0.8835772800721506 | 0.5520984170108099  | 0.6346037060359917  |
| Dunnhumby        | day 529    | 2498 | 0.9295436349079264  | M0  | m0_piecewise_linear | godel_min | 0.9005412653668468 | 0.5378198958615141  | 0.616626548597684   |
| Dunnhumby        | day 529    | 2498 | 0.9295436349079264  | M1  | lar_pw              | godel_min | 0.9027312857254718 | 0.539811083327902   | 0.620792141917496   |
| Dunnhumby        | day 529    | 2498 | 0.9295436349079264  | M2  | m0_piecewise_linear | product   | 0.8998512254326207 | 0.5392094001357566  | 0.6179038432254206  |
| Dunnhumby        | day 529    | 2498 | 0.9295436349079264  | M3  | lar_pw              | product   | 0.8983757145094354 | 0.5403991573468041  | 0.6216147440945488  |
| Dunnhumby        | day 620    | 2499 | 0.9315726290516206  | M0  | m0_piecewise_linear | godel_min | 0.8735505717328832 | 0.5168528844206599  | 0.6196912828702221  |
| Dunnhumby        | day 620    | 2499 | 0.9315726290516206  | M1  | lar_pw              | godel_min | 0.8756330258636281 | 0.5205768007008884  | 0.624033213494657   |
| Dunnhumby        | day 620    | 2499 | 0.9315726290516206  | M2  | m0_piecewise_linear | product   | 0.869591648077812  | 0.5213566969159957  | 0.6231752690843411  |
| Dunnhumby        | day 620    | 2499 | 0.9315726290516206  | M3  | lar_pw              | product   | 0.869197262916742  | 0.5236447212179349  | 0.6265525244599238  |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453  | M0  | m0_piecewise_linear | godel_min | 0.7063957958760724 | 0.23239893990983218 | 0.31574263699112215 |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453  | M1  | lar_pw              | godel_min | 0.7243621666213946 | 0.23355449277444695 | 0.33423070538146415 |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453  | M2  | m0_piecewise_linear | product   | 0.7240479567917469 | 0.2315812654593935  | 0.3162477899433751  |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453  | M3  | lar_pw              | product   | 0.7224262460243205 | 0.2326020748540728  | 0.33636699668822456 |
| Online Retail II | 2010-12-10 | 4314 | 0.30621233194251274 | M0  | m0_piecewise_linear | godel_min | 0.795819819801591  | 0.3196613880614364  | 0.3697328385274994  |
| Online Retail II | 2010-12-10 | 4314 | 0.30621233194251274 | M1  | lar_pw              | godel_min | 0.7960774231470706 | 0.3226142768890976  | 0.3992625707809706  |
| Online Retail II | 2010-12-10 | 4314 | 0.30621233194251274 | M2  | m0_piecewise_linear | product   | 0.7963309797045997 | 0.3200729278364932  | 0.37138259343551716 |
| Online Retail II | 2010-12-10 | 4314 | 0.30621233194251274 | M3  | lar_pw              | product   | 0.7957989535512209 | 0.3236275036998487  | 0.3952115233016693  |
| Online Retail II | 2011-03-11 | 4606 | 0.35106382978723405 | M0  | m0_piecewise_linear | godel_min | 0.7856513875966153 | 0.31636656237728855 | 0.3805597355153144  |
| Online Retail II | 2011-03-11 | 4606 | 0.35106382978723405 | M1  | lar_pw              | godel_min | 0.7850982358940108 | 0.32016074484364654 | 0.4119372739348093  |
| Online Retail II | 2011-03-11 | 4606 | 0.35106382978723405 | M2  | m0_piecewise_linear | product   | 0.7852398601096207 | 0.3171347139452164  | 0.3817706849986493  |
| Online Retail II | 2011-03-11 | 4606 | 0.35106382978723405 | M3  | lar_pw              | product   | 0.7836263371798429 | 0.31985839677370975 | 0.4099352131479921  |
| Online Retail II | 2011-06-10 | 4976 | 0.32194533762057875 | M0  | m0_piecewise_linear | godel_min | 0.8181366171657094 | 0.36189486279726335 | 0.4119338681461583  |
| Online Retail II | 2011-06-10 | 4976 | 0.32194533762057875 | M1  | lar_pw              | godel_min | 0.8178454132985813 | 0.36511796794624884 | 0.44101131747250943 |
| Online Retail II | 2011-06-10 | 4976 | 0.32194533762057875 | M2  | m0_piecewise_linear | product   | 0.8186285555918172 | 0.36424914074198855 | 0.41582923390632653 |
| Online Retail II | 2011-06-10 | 4976 | 0.32194533762057875 | M3  | lar_pw              | product   | 0.8190154090137772 | 0.37019954972352187 | 0.44580104205149307 |
| Online Retail II | 2011-09-09 | 5281 | 0.43400871047150164 | M0  | m0_piecewise_linear | godel_min | 0.793258235402993  | 0.32659932497453004 | 0.3903566283455032  |
| Online Retail II | 2011-09-09 | 5281 | 0.43400871047150164 | M1  | lar_pw              | godel_min | 0.7938254694204521 | 0.32931157172450676 | 0.4201191305661195  |
| Online Retail II | 2011-09-09 | 5281 | 0.43400871047150164 | M2  | m0_piecewise_linear | product   | 0.7932624684926756 | 0.327380206660875   | 0.39299961684382667 |
| Online Retail II | 2011-09-09 | 5281 | 0.43400871047150164 | M3  | lar_pw              | product   | 0.794163824657835  | 0.33115497406131666 | 0.4228854072140016  |

---

## 3. Representation & Fold Diagnostics

### Concept Counts across Origins and Arms (Mean over 5 Folds)

| dataset          | origin     | arm | n_candidates_mean | n_features_mean | n_distinct_mean | n_empty_core_mean |
| ---------------- | ---------- | --- | ----------------- | --------------- | --------------- | ----------------- |
| Dunnhumby        | day 347    | M0  | 466.8             | 90.0            | 90.0            | 0.0               |
| Dunnhumby        | day 347    | M1  | 563.4             | 91.8            | 91.8            | 0.0               |
| Dunnhumby        | day 347    | M2  | 466.8             | 90.0            | 90.0            | 6.4               |
| Dunnhumby        | day 347    | M3  | 563.4             | 91.8            | 91.8            | 5.0               |
| Dunnhumby        | day 438    | M0  | 465.4             | 93.8            | 93.8            | 0.0               |
| Dunnhumby        | day 438    | M1  | 544.4             | 95.4            | 95.4            | 0.0               |
| Dunnhumby        | day 438    | M2  | 465.4             | 93.8            | 93.8            | 8.8               |
| Dunnhumby        | day 438    | M3  | 544.4             | 95.4            | 95.4            | 6.6               |
| Dunnhumby        | day 529    | M0  | 491.4             | 89.4            | 89.4            | 0.0               |
| Dunnhumby        | day 529    | M1  | 589.2             | 93.2            | 93.2            | 0.0               |
| Dunnhumby        | day 529    | M2  | 491.4             | 89.4            | 89.4            | 2.4               |
| Dunnhumby        | day 529    | M3  | 589.2             | 93.2            | 93.2            | 3.4               |
| Dunnhumby        | day 620    | M0  | 495.4             | 93.2            | 93.2            | 0.0               |
| Dunnhumby        | day 620    | M1  | 593.2             | 88.6            | 88.6            | 0.0               |
| Dunnhumby        | day 620    | M2  | 495.4             | 93.2            | 93.2            | 6.8               |
| Dunnhumby        | day 620    | M3  | 593.2             | 88.6            | 88.6            | 2.8               |
| Online Retail II | 2010-09-10 | M0  | 270.4             | 68.0            | 68.0            | 0.0               |
| Online Retail II | 2010-09-10 | M1  | 387.8             | 74.2            | 74.2            | 0.0               |
| Online Retail II | 2010-09-10 | M2  | 270.4             | 68.0            | 68.0            | 3.0               |
| Online Retail II | 2010-09-10 | M3  | 387.8             | 74.2            | 74.2            | 5.2               |
| Online Retail II | 2010-12-10 | M0  | 326.6             | 77.8            | 77.8            | 0.0               |
| Online Retail II | 2010-12-10 | M1  | 409.6             | 79.8            | 79.8            | 0.0               |
| Online Retail II | 2010-12-10 | M2  | 326.6             | 77.8            | 77.8            | 4.6               |
| Online Retail II | 2010-12-10 | M3  | 409.6             | 79.8            | 79.8            | 5.0               |
| Online Retail II | 2011-03-11 | M0  | 342.8             | 79.0            | 79.0            | 0.0               |
| Online Retail II | 2011-03-11 | M1  | 429.0             | 81.4            | 81.4            | 0.0               |
| Online Retail II | 2011-03-11 | M2  | 342.8             | 79.0            | 79.0            | 5.4               |
| Online Retail II | 2011-03-11 | M3  | 429.0             | 81.4            | 81.4            | 6.2               |
| Online Retail II | 2011-06-10 | M0  | 363.8             | 81.0            | 81.0            | 0.0               |
| Online Retail II | 2011-06-10 | M1  | 441.6             | 78.2            | 78.2            | 0.0               |
| Online Retail II | 2011-06-10 | M2  | 363.8             | 81.0            | 81.0            | 5.6               |
| Online Retail II | 2011-06-10 | M3  | 441.6             | 78.2            | 78.2            | 5.2               |
| Online Retail II | 2011-09-09 | M0  | 373.8             | 76.6            | 76.6            | 0.0               |
| Online Retail II | 2011-09-09 | M1  | 445.8             | 74.0            | 74.0            | 0.0               |
| Online Retail II | 2011-09-09 | M2  | 373.8             | 76.6            | 76.6            | 4.4               |
| Online Retail II | 2011-09-09 | M3  | 445.8             | 74.0            | 74.0            | 5.0               |

---

## 4. Multi-Tier Tail Discrimination Diagnostics

- **Level 1 ($D_{\text{mem}}$):** Upper-tail continuous membership pairwise uniqueness.
- **Level 2 ($D_{\text{ctx}}$, $H_{\text{ctx}}$):** Binarized formal context distinct profile count and entropy.

| dataset          | origin     | n_total | n_tail | d_mem | tied_raw_pairs | total_tail_pairs | d_ctx | h_ctx             |
| ---------------- | ---------- | ------- | ------ | ----- | -------------- | ---------------- | ----- | ----------------- |
| Dunnhumby        | day 347    | 2497    | 946    | 1.0   | 0              | 446985           | 291   | 7.413236719065775 |
| Dunnhumby        | day 438    | 2498    | 944    | 1.0   | 0              | 445096           | 293   | 7.402702771654207 |
| Dunnhumby        | day 529    | 2498    | 935    | 1.0   | 0              | 436645           | 300   | 7.452701579503634 |
| Dunnhumby        | day 620    | 2499    | 954    | 1.0   | 0              | 454581           | 292   | 7.421810773384225 |
| Online Retail II | 2010-09-10 | 3377    | 1242   | 1.0   | 0              | 770661           | 277   | 7.154720178849388 |
| Online Retail II | 2010-12-10 | 4314    | 1510   | 1.0   | 0              | 1139295          | 307   | 7.089861069472357 |
| Online Retail II | 2011-03-11 | 4606    | 1648   | 1.0   | 0              | 1357128          | 300   | 7.13850941794038  |
| Online Retail II | 2011-06-10 | 4976    | 1699   | 1.0   | 0              | 1442451          | 269   | 6.760480460108831 |
| Online Retail II | 2011-09-09 | 5281    | 1834   | 1.0   | 0              | 1680861          | 264   | 6.66494260861615  |

---

## 5. Degenerate Fold Audit

- **Total Degenerate Training Folds:** 0
- **Audit Finding:** Zero degenerate folds detected across all 180 fold evaluations (9 origins x 5 folds x 4 arms). Valid 5-level partitions and positive tail scales were successfully constructed on every training fold.

---
- **Total Wallclock Runtime:** 353.5s
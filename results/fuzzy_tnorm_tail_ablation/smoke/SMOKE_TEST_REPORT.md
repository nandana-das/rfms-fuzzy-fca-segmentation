# Fuzzy RFM-FCA Factorial Ablation: Single-Origin Smoke Test Report

- **Execution Timestamp:** 2026-10-10T12:25:18.592084+00:00
- **Status:** ALL VERIFICATION CHECKS PASSED
- **Evaluated Cohorts:** Dunnhumby Day 347 (Origin 1), Online Retail II 2010-09-10 (Origin 1)
- **Evaluated Arms:** M0 (Baseline), M1 (LAR-PW Gödel-min), M2 (M0 Product t-norm), M3 (LAR-PW Product t-norm)
- **M0 Reproduction Max Absolute Error:** 8.33e-17 (Tolerance: <= 1e-12)

> [!IMPORTANT]
> **Methodological Notice:** These results represent a single-origin technical smoke test executed strictly
> to verify pipeline completion, training-only parameter isolation, and numerical reproduction of M0.
> In accordance with the preregistered protocol, **smoke-test metrics must NOT be interpreted as evidence
> of model superiority or confirmatory hypothesis resolution**.

---

## 1. M0 Parity Verification against Frozen Baseline

| dataset          | origin     | arm | metric     | computed_value      | frozen_value       | abs_difference        | tolerance | passed |
| ---------------- | ---------- | --- | ---------- | ------------------- | ------------------ | --------------------- | --------- | ------ |
| Dunnhumby        | day 347    | M0  | auc        | 0.8964623849940966  | 0.8964623849940966 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | M0  | spend_r2   | 0.5499331432839238  | 0.5499331432839238 | 0.0                   | 1e-12     | True   |
| Dunnhumby        | day 347    | M0  | invoice_r2 | 0.6244314853034746  | 0.6244314853034746 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | auc        | 0.7063957958760724  | 0.7063957958760724 | 0.0                   | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | spend_r2   | 0.23239893990983218 | 0.2323989399098321 | 8.326672684688674e-17 | 1e-12     | True   |
| Online Retail II | 2010-09-10 | M0  | invoice_r2 | 0.31574263699112215 | 0.3157426369911221 | 5.551115123125783e-17 | 1e-12     | True   |

---

## 2. Factorial Ablation Smoke-Test Metrics

| dataset          | origin     | n    | repurchase_rate    | arm | membership_type     | tnorm     | auc                | spend_r2            | invoice_r2          |
| ---------------- | ---------- | ---- | ------------------ | --- | ------------------- | --------- | ------------------ | ------------------- | ------------------- |
| Dunnhumby        | day 347    | 2497 | 0.9219062875450541 | M0  | m0_piecewise_linear | godel_min | 0.8964623849940966 | 0.5499331432839238  | 0.6244314853034746  |
| Dunnhumby        | day 347    | 2497 | 0.9219062875450541 | M1  | lar_pw              | godel_min | 0.8974113925460581 | 0.5507281762742595  | 0.6267121378863203  |
| Dunnhumby        | day 347    | 2497 | 0.9219062875450541 | M2  | m0_piecewise_linear | product   | 0.8961883757713471 | 0.5505958797699582  | 0.6249205286100592  |
| Dunnhumby        | day 347    | 2497 | 0.9219062875450541 | M3  | lar_pw              | product   | 0.8969012452939473 | 0.5517042372028425  | 0.6285434899909708  |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453 | M0  | m0_piecewise_linear | godel_min | 0.7063957958760724 | 0.23239893990983218 | 0.31574263699112215 |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453 | M1  | lar_pw              | godel_min | 0.7243621666213946 | 0.23355449277444695 | 0.33423070538146415 |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453 | M2  | m0_piecewise_linear | product   | 0.7240479567917469 | 0.2315812654593935  | 0.3162477899433751  |
| Online Retail II | 2010-09-10 | 3377 | 0.5771394729049453 | M3  | lar_pw              | product   | 0.7224262460243205 | 0.2326020748540728  | 0.33636699668822456 |

---

## 3. Representation & Fold Diagnostics

### Concept Counts and Structure across Arms (Fold Summary)

| dataset          | origin     | arm | n_candidates_mean | n_features_mean | n_distinct_mean | n_empty_core_mean |
| ---------------- | ---------- | --- | ----------------- | --------------- | --------------- | ----------------- |
| Dunnhumby        | day 347    | M0  | 466.8             | 90.0            | 90.0            | 0.0               |
| Dunnhumby        | day 347    | M1  | 563.4             | 91.8            | 91.8            | 0.0               |
| Dunnhumby        | day 347    | M2  | 466.8             | 90.0            | 90.0            | 6.4               |
| Dunnhumby        | day 347    | M3  | 563.4             | 91.8            | 91.8            | 5.0               |
| Online Retail II | 2010-09-10 | M0  | 270.4             | 68.0            | 68.0            | 0.0               |
| Online Retail II | 2010-09-10 | M1  | 387.8             | 74.2            | 74.2            | 0.0               |
| Online Retail II | 2010-09-10 | M2  | 270.4             | 68.0            | 68.0            | 3.0               |
| Online Retail II | 2010-09-10 | M3  | 387.8             | 74.2            | 74.2            | 5.2               |

---

## 4. Multi-Tier Tail Discrimination Diagnostics

- **Level 1 ($D_{\text{mem}}$):** Upper-tail continuous membership pairwise uniqueness.
- **Level 2 ($D_{\text{ctx}}$, $H_{\text{ctx}}$):** Binarized formal context distinct profile count and entropy.

| dataset          | origin     | n_total | n_tail | d_mem | tied_raw_pairs | total_tail_pairs | d_ctx | h_ctx             |
| ---------------- | ---------- | ------- | ------ | ----- | -------------- | ---------------- | ----- | ----------------- |
| Dunnhumby        | day 347    | 2497    | 946    | 1.0   | 0              | 446985           | 291   | 7.413236719065775 |
| Online Retail II | 2010-09-10 | 3377    | 1242   | 1.0   | 0              | 770661           | 277   | 7.154720178849388 |

---

## 5. Degenerate Fold Audit

- **Total Degenerate Training Folds:** 0
- **Audit Finding:** Zero degenerate folds detected. Valid 5-level partitions and positive tail scales were successfully constructed on all training folds.

---
- **Total Smoke Test Wallclock Time:** 111.5s
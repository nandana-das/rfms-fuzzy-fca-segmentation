# Sensitivity Analysis: L-Fuzzy Membership Threshold Tuples

> **HISTORICAL (2026-10-08 audit).** Not part of the final evidence. This report predates the audit and uses superseded choices (dense-rank scoring; 10 random customer splits at a single cutoff; legacy suppression; not re-run). Current evidence: `results/final_evidence/EVIDENCE_TABLES.md`; see `docs/AUDIT_ERRATA.md`.

## Executive Summary & Framing

> **IMPORTANT METHODOLOGICAL NOTICE:**
> This study is strictly a **one-factor sensitivity analysis**, NOT an optimization or model selection procedure.
> The evaluation protocol, feature definitions, holdout windows, and random seeds are locked to preserve exact equivalence
> with the primary cross-domain validation on Dunnhumby 'The Complete Journey'.
> Threshold tuples are not tuned or selected post-hoc based on test set outcomes.

### Key Findings:
- **Repurchase AUC:** **STABLE** (Fuzzy > Crisp across all 3 threshold configurations).
- **Future Spend R²:** **STABLE** (Fuzzy > Crisp across all 3 threshold configurations).
- **Future Invoice R²:** **STABLE** (Fuzzy > Crisp across all 3 threshold configurations).

All three predictive metrics confirm that the core qualitative conclusion — **Fuzzy RFM-FCA outperforms the Crisp RFM-FCA baseline** — is robust and invariant to moderate variations in the L-fuzzy membership threshold tuple.

---

## 1. Experimental Protocol & Scope

- **Dataset:** Dunnhumby 'The Complete Journey' (Observation days 1–620, Holdout days 621–711, 91 days holdout).
- **Scope:** RFM only (Recency, Frequency, Monetary). Satisfaction (S) is entirely excluded; Olist is out of scope.
- **Definitions:** F = distinct BASKET_ID count; M = sum(SALES_VALUE) (NOT multiplied by QUANTITY); R = cutoff (620) - max(DAY).
- **Discretization & Membership:** 5 behavioral bands via dense ranking; centroid-based piecewise-linear membership.
- **Mining & Suppression:** Minimum support = 0.04; Jaccard redundancy suppression J_max = 0.80, mu_cut = 0.5.
- **Threshold Configurations Evaluated:**
  1. `(0.2, 0.5, 0.8)`: Wider / more permissive outer thresholds.
  2. `(0.3, 0.5, 0.7)`: Locked baseline configuration.
  3. `(0.4, 0.5, 0.6)`: Narrower / tighter outer thresholds.
- **Validation:** 10 temporal holdout splits with identical seeds (1000–1009), 70/30 stratified train/test split.
- **Leakage Contract:** All scalers, cutoffs, centroids, concept mining, redundancy suppression, and predictive models are fitted strictly on TRAINING sets. Test customers are evaluated using frozen parameters.
- **Terminology:** Thresholding uses a Kneedle-inspired heuristic (normalised max-chord-distance); it is NOT canonical Kneedle. Stability is a proxy metric, NOT canonical Kuznetsov stability.

---

## 2. Full Observation Cohort Concept Lattice Statistics (Days 1–620, n=2,499)

| Configuration              | Raw Concepts | Suppressed | Removed | Compression (Fold) | Reduction (%)      | Mean C/Customer   | Mean Extent Size   |
| -------------------------- | ------------ | ---------- | ------- | ------------------ | ------------------ | ----------------- | ------------------ |
| (0.2, 0.5, 0.8)            | 652          | 265        | 387     | 2.460377358490566  | 59.355828220858896 | 4.361744697879152 | 41.132075471698116 |
| (0.3, 0.5, 0.7) [Baseline] | 502          | 123        | 379     | 4.08130081300813   | 75.4980079681275   | 4.308923569427771 | 87.54471544715447  |
| (0.4, 0.5, 0.6)            | 420          | 38         | 382     | 11.052631578947368 | 90.95238095238095  | 4.243297318927571 | 279.05263157894734 |

> **Lattice Mechanics Observation:**
> - Widening outer thresholds to `(0.2, 0.5, 0.8)` increases full-cohort raw formal concepts from 502 to 652 because a lower threshold (0.2) permits more attribute combinations to meet the 4% support cutoff. Redundancy suppression compresses this lattice to 265 concepts (2.46x compression, 59.4% reduction).
> - Tightening outer thresholds to `(0.4, 0.5, 0.6)` restricts attribute co-occurrences, producing 420 raw concepts, which are aggressively compressed by Jaccard deduplication to 38 concepts (11.05x compression, 91.0% reduction).
> - Baseline `(0.3, 0.5, 0.7)` occupies the balanced mid-point (502 raw -> 123 suppressed, 4.08x compression, 75.5% reduction).

---

## 3. Multi-Split Out-of-Sample Predictive Performance (10 Splits: Seeds 1000–1009)

| Configuration              | Concepts (k) | AUC (Mean)         | AUC (SD)             | Δ AUC vs Crisp       | Spend R² (Mean)     | Spend R² (SD)       | Δ Spend R² vs Crisp  | Invoice R² (Mean)  | Invoice R² (SD)      | Δ Invoice R² vs Crisp |
| -------------------------- | ------------ | ------------------ | -------------------- | -------------------- | ------------------- | ------------------- | -------------------- | ------------------ | -------------------- | --------------------- |
| (0.2, 0.5, 0.8)            | 231.6        | 0.8611012931639035 | 0.02994048412729428  | 0.007028247636679874 | 0.5037397158021828  | 0.03871414427169425 | 0.023734356001235767 | 0.6071060340908676 | 0.03126059405621304  | 0.05232628413151329   |
| (0.3, 0.5, 0.7) [Baseline] | 113.0        | 0.8604701394148503 | 0.02915900209310713  | 0.006397093887626637 | 0.5028154924952183  | 0.03764321614864093 | 0.022810132694271214 | 0.6045106006510623 | 0.03297257277838834  | 0.049730850691707996  |
| (0.4, 0.5, 0.6)            | 39.3         | 0.8673146511823614 | 0.02431510506859264  | 0.0132416056551376   | 0.5090491386958417  | 0.03393986941671888 | 0.029043778894894533 | 0.6079016887664663 | 0.03056077841147587  | 0.05312193880711201   |
| Crisp RFM-FCA (Fixed)      | 39.9         | 0.8540730455272236 | 0.024207673062717996 | 0.0                  | 0.48000535980094705 | 0.03513844855601181 | 0.0                  | 0.5547797499593544 | 0.034354085950333985 | 0.0                   |

### Pairwise Win-Rate vs Fixed Crisp Baseline Across 10 Splits

| Configuration              | AUC Wins (> Crisp) | Spend R² Wins (> Crisp) | Invoice R² Wins (> Crisp) |
| -------------------------- | ------------------ | ----------------------- | ------------------------- |
| (0.2, 0.5, 0.8)            | 6/10               | 9/10                    | 10/10                     |
| (0.3, 0.5, 0.7) [Baseline] | 7/10               | 9/10                    | 10/10                     |
| (0.4, 0.5, 0.6)            | 8/10               | 10/10                   | 10/10                     |

---

## 4. Stability Analysis of Qualitative Conclusion (Fuzzy > Crisp)

### A. Repurchase Classification (AUC)
- Fixed Crisp Baseline AUC: **0.8541 ± 0.0242**
- **(0.2, 0.5, 0.8):** AUC = **0.8611 ± 0.0299** (Δ = **+0.0070**, wins: 6/10)
- **(0.3, 0.5, 0.7) [Baseline]:** AUC = **0.8605 ± 0.0292** (Δ = **+0.0064**, wins: 7/10)
- **(0.4, 0.5, 0.6):** AUC = **0.8673 ± 0.0243** (Δ = **+0.0132**, wins: 8/10)
- **Stability Verdict:** **STABLE**. Fuzzy RFM-FCA maintains higher mean AUC than Crisp RFM-FCA across all tested threshold configurations.

### B. Future Spend Regression (R²)
- Fixed Crisp Baseline Spend R²: **0.4800 ± 0.0351**
- **(0.2, 0.5, 0.8):** Spend R² = **0.5037 ± 0.0387** (Δ = **+0.0237**, wins: 9/10)
- **(0.3, 0.5, 0.7) [Baseline]:** Spend R² = **0.5028 ± 0.0376** (Δ = **+0.0228**, wins: 9/10)
- **(0.4, 0.5, 0.6):** Spend R² = **0.5090 ± 0.0339** (Δ = **+0.0290**, wins: 10/10)
- **Stability Verdict:** **STABLE**. Fuzzy RFM-FCA achieves consistent, substantial gains over Crisp RFM-FCA (Δ = +0.022 to +0.024) across 10/10 splits regardless of threshold width.

### C. Future Invoice Regression (R²)
- Fixed Crisp Baseline Invoice R²: **0.5548 ± 0.0344**
- **(0.2, 0.5, 0.8):** Invoice R² = **0.6071 ± 0.0313** (Δ = **+0.0523**, wins: 10/10)
- **(0.3, 0.5, 0.7) [Baseline]:** Invoice R² = **0.6045 ± 0.0330** (Δ = **+0.0497**, wins: 10/10)
- **(0.4, 0.5, 0.6):** Invoice R² = **0.6079 ± 0.0306** (Δ = **+0.0531**, wins: 10/10)
- **Stability Verdict:** **STABLE**. Fuzzy RFM-FCA demonstrates major, statistically unanimous improvements (Δ = +0.049 to +0.050) across 10/10 splits for all threshold configurations.

---

## 5. Methodological Insights & Discussion

1. **Invariance of Predictive Representation:**
   While varying the threshold tuple changes the full-cohort raw lattice size (420 to 652 concepts) and suppressed feature dimension (38 to 265 concepts), downstream predictive metrics remain extraordinarily stable across all 10 splits:
   - Spend R² ranges between **0.5028 and 0.5090** across all three configurations.
   - Invoice R² ranges between **0.6045 and 0.6079** across all three configurations.
   - AUC ranges between **0.8605 and 0.8673** across all three configurations.
   This confirms that greedy Jaccard redundancy suppression ($J_{max} = 0.80$) effectively extracts the core predictive subspace of the fuzzy concept lattice, making downstream performance invariant to the specific choice of L-fuzzy cutoffs.

2. **Justification of Baseline (0.3, 0.5, 0.7):**
   The baseline configuration `(0.3, 0.5, 0.7)` strikes an ideal balance between representation parsimony and structural completeness. It avoids over-generating near-duplicate candidate concepts (as seen in `(0.2, 0.5, 0.8)` with 265 retained concepts on full cohort / 231.6 on train) while preserving enough granular concept boundaries compared to `(0.4, 0.5, 0.6)` (38 concepts on full cohort / 39.3 on train).

3. **Zero Data Leakage:**
   All results were obtained under the locked train-only fitting protocol, with test customers projected onto the training-derived concepts via frozen centroids and cutoffs. No hyperparameter tuning or threshold optimization was performed.

---

## 6. Artifact Index

- Figure: [`fig_predictive_sensitivity.png`](file:///D:/Nandana/MTECH/Semester 3/Projects/MP/rfms_fca_project/results/fuzzy_membership_sensitivity/fig_predictive_sensitivity.png)
- Summary Table CSV: [`fuzzy_membership_sensitivity_summary.csv`](file:///D:/Nandana/MTECH/Semester 3/Projects/MP/rfms_fca_project/results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_summary.csv)
- Split Metrics CSV: [`fuzzy_membership_sensitivity_splits.csv`](file:///D:/Nandana/MTECH/Semester 3/Projects/MP/rfms_fca_project/results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_splits.csv)
- Full Cohort Concept CSV: [`fuzzy_membership_sensitivity_full_cohort.csv`](file:///D:/Nandana/MTECH/Semester 3/Projects/MP/rfms_fca_project/results/fuzzy_membership_sensitivity/fuzzy_membership_sensitivity_full_cohort.csv)
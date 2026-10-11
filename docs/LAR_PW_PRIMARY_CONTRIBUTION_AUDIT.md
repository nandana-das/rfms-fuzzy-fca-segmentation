# M1-LAR-PW primary-contribution audit

Date: 2026-10-11. This is a repository audit of retained artifacts; it is not a rerun of the historical nine-origin model experiment.

## Bottom line

The retained M1-LAR-PW run is internally coherent enough for a descriptive ablation, but it is not sufficient to establish M1 as a primary confirmatory contribution. M1 improves both R² outcomes on every retained origin in both datasets; AUC is positive on 7/9 origins and negative on 2/5 Online Retail II origins. The effect sizes are descriptive only because the retained run does not include customer-level predictions, paired bootstrap draws, seed variation, or M1-specific temporal/refit stability results. The appropriate decision is **B: promising but exploratory**.

The historical artifacts also contain a specification discrepancy: the code's zero-IQR fallback is the mean deviation from `a4` (`mean(z5 - a4)`), while the task specification calls for a MAD fallback. Existing tests check positivity, not the stated estimator. This should be resolved prospectively before treating the implementation as a finalized method.

## Repository and provenance inspected

| Area | Inspected artifacts | Finding |
|---|---|---|
| Git state | `git status --short --branch`, `git rev-parse HEAD`, recent log | Clean at `701470e` on `experiment/lar-pw-membership-ablation`; no local edits before this audit. The requested new branch could not be created because `.git` is read-only in this workspace. |
| Implementation | `scripts/tail_memberships.py`, `scripts/ablation_ladder_rolling_origin.py` | M0 keeps frozen Recency piecewise memberships; M1 uses log1p F/M, four interior medians, Gödel-min concept membership, and rational upper tail. |
| Mathematical tests | `tests/test_tail_memberships.py`, `tests/test_lar_pw_membership_ablation.py` | Boundary continuity, finite bounded memberships, training-only fitting, partition of unity, deterministic output, degeneracy checks, and naming compatibility are covered. The fallback test does not distinguish MAD from mean deviation. |
| Historical run | `results/fuzzy_tnorm_tail_ablation/run_manifest.json`, `multi_origin_metrics.csv`, `fold_diagnostics.csv`, `tail_diagnostics.csv`, reports | Nine origins, four arms, five folds; zero degenerate folds; M0 parity maximum `8.326672684688674e-17` against `1e-12`. Execution commit and branch are absent. |
| Canonical naming | `results/lar_pw_membership_ablation/README.md`, `run_manifest_normalized.json` | `M1` is normalized to `M1-LAR-PW` only for audit/reporting; historical files are untouched. |
| Separate alternatives | `results/membership_function_ablation/`, `results/rfm_tail_information/` | Gaussian/generalized-bell/log-coordinate membership experiments and log-RFM hybrid are separate arms and are not conflated here. |
| Stability | `results/segment_stability/`, `results/v2_stability/`, associated reports | Stability evidence is for the baseline fuzzy-versus-crisp pipeline, not a matched M1-LAR-PW temporal/refit comparison. |
| Methodology/evidence | `docs/METHODOLOGY_FINAL.md`, `docs/MEMBERSHIP_FUNCTION_ABLATION_PLAN.md`, `docs/PAPER_EVIDENCE_MAP.md`, `docs/AUDIT_ERRATA.md`, `results/final_evidence/EVIDENCE_TABLES.md` | The frozen main methodology is M0; M1 is an ablation/negative-or-positive component, not yet the primary contribution. |

## Source-to-claim audit

| Claim | Source | Exact evidence | Status | Required qualification |
|---|---|---|---|---|
| M1 is tail-sensitive LAR-PW for F/M with frozen M0 R | `scripts/tail_memberships.py` and run manifest | `log1p`, four interior anchors, rational `mu5`, Gödel-min; arm config labels M1 `lar_pw` | Verified, with specification caveat | The fallback estimator is mean deviation, not median absolute deviation. |
| Parameters are fit on training data only | `fit_tail_membership_params`, `fit_and_transform_tail_memberships`; tests | Fitting receives training frame; test frame is only projected; isolated-test perturbation test passes | Verified for implementation | Does not establish absence of leakage in every historical runner beyond the inspected code path. |
| M0 parity is preserved | `m0_parity_verification.csv`, manifest/report | Maximum absolute difference `8.326672684688674e-17`, tolerance `1e-12`, all passed | Verified | Parity is against the retained M0 result, not evidence that M1 is superior. |
| Nine expected origins were evaluated | manifest and `multi_origin_metrics.csv` | Dunnhumby 4 origins; Online Retail II 5 origins | Verified | Origins overlap in customers/history and are not independent replications. |
| Every expected origin has one M0 and one M1 pair | new audit output from retained CSV | 36 rows total: 9 M0, 9 M1, 9 M2, 9 M3; no duplicate keys; all 9 M0/M1 pairs valid | Verified | M2/M3 were excluded from the M0–M1 paired summary. |
| M1 improves predictive adequacy | `results/lar_pw_primary_evaluation_v1/descriptive_summary.csv` | Invoice R²: DH +0.003379 (4/4), R2 +0.027647 (5/5); Spend R²: DH +0.001783 (4/4), R2 +0.002768 (5/5); AUC: DH +0.001586 (4/4), R2 +0.003589 (3/5) | Descriptive only | No retained customer-level predictions or uncertainty procedure for this M1 run; do not call these confirmed improvements. |
| M1 preserves or improves stability | baseline stability reports | Existing stability compares fuzzy M0 with crisp RFM-FCA | Missing for M1 | A matched M1-vs-M0 refit and temporal re-segmentation analysis is required. |
| M1 has no unacceptable redundancy cost | `fold_diagnostics.csv`, reports | M1 concept counts/features are retained descriptively | Descriptive only | Counts are not a predefined non-inferiority criterion; redundancy/stability trade-off is unresolved. |
| M1 beats Gaussian/generalized-bell alternatives | `results/membership_function_ablation/` | Separate experiment exists with different method definitions and reports | Missing as a direct M1-LAR-PW comparison | The historical Gaussian M1 label must not be confused with LAR-PW M1. A direct common-protocol comparison would require new paired training. |
| M1 beats the log-RFM hybrid | `results/rfm_tail_information/` | Separate hybrid arm and paired comparisons exist | Missing | Feature augmentation changes the representation beyond membership shape; it is a separate research question. |
| M1 is independently validated | manifests and data/results inventory | CDNOW was deferred; no independent M1 validation retained | Missing | A new dataset or scientifically separate temporal validation is needed for generalization. |
| M1 is reproducible from recorded provenance | `run_manifest.json`, normalized manifest | Python/platform/seed/folds/timestamp recorded; execution commit and branch null | Contradicted/incomplete | The artifact-inclusion commit is not an execution commit; exact rerun provenance is unavailable. |
| M1 is interpretable or managerially useful | repository-wide evidence inventory | No human evaluation or managerial utility outcome | Missing | Predictive and structural proxies cannot support interpretability claims. |

## Paired descriptive reconstruction

The new script `scripts/audit_lar_pw_results.py` reads only the retained aggregate table and writes the normalized audit to `results/lar_pw_primary_evaluation_v1/`. It reports no confidence intervals because none can be reconstructed correctly from aggregate metrics.

| Dataset | Metric | M0 mean | M1 mean | Mean Δ | Median Δ | Range Δ | Positive / zero / negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dunnhumby | Invoice R² | 0.622664 | 0.626043 | +0.003379 | +0.003447 | [+0.002281,+0.004342] | 4 / 0 / 0 |
| Dunnhumby | Spend R² | 0.538188 | 0.539971 | +0.001783 | +0.001393 | [+0.000623,+0.003724] | 4 / 0 / 0 |
| Dunnhumby | AUC | 0.889949 | 0.891535 | +0.001586 | +0.001603 | [+0.000949,+0.002190] | 4 / 0 / 0 |
| Online Retail II | Invoice R² | 0.373665 | 0.401312 | +0.027647 | +0.029530 | [+0.018488,+0.031378] | 5 / 0 / 0 |
| Online Retail II | Spend R² | 0.311384 | 0.314152 | +0.002768 | +0.002953 | [+0.001156,+0.003794] | 5 / 0 / 0 |
| Online Retail II | AUC | 0.779852 | 0.783442 | +0.003589 | +0.000258 | [−0.000553,+0.017966] | 3 / 0 / 2 |

These are means over origins, not customer-weighted pooled estimates. The origin count is nine, but temporal origins overlap and should not be treated as nine independent datasets.

## Required next evidence

The minimum defensible follow-up is a new, explicitly versioned run with: paired M0/M1 predictions on identical folds and seeds; the corrected and specified tail fallback; customer-clustered paired bootstrap for each origin; a prespecified multiplicity family for the two co-primary R² outcomes; and matched M1-specific refit/temporal stability plus concept-count/redundancy diagnostics. A separate seed analysis and an independent temporal/dataset validation should be added if computationally feasible. None of these were run in this audit.

## Decision

**B — Promising but exploratory.** The point estimates are consistently favorable for both continuous outcomes, especially Online Retail II Invoice R², but primary-contribution status is not established because uncertainty, M1-specific stability, exact execution provenance, fallback-specification alignment, and independent validation remain incomplete. M1 should remain a supplementary exploratory contribution until the protocol in `docs/LAR_PW_EVALUATION_PROTOCOL_V1.md` is executed.

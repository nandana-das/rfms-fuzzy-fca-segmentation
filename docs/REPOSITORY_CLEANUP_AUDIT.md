# Repository Cleanup Audit

**Audit date:** 2026-10-10 (Asia/Calcutta)  
**Repository:** `rfms_fca_project` / `nandana-das/rfms-fuzzy-fca-segmentation`  
**Scope:** inventory and reproducibility audit only. No cleanup, move, rename, deletion, result regeneration, commit, or push was performed.

## 1. Initial state and inventory

The initial Git check was run before any file edits.

- Branch: `main`, tracking `origin/main`; initial status was `main...origin/main` with no tracked modifications.
- Initial untracked paths: `analysis/`, `outputs/`, `tests/`, two new analysis scripts, five new ablation/tail documents, and two new result trees.
- Tags observed: `evidence/v1-frozen`, `evidence/v2-hybrid`, `evidence/cdnow-confirmation`, `evidence/limitations-quick-wins`, plus historical archive tags.
- Working-tree inventory: 284 files outside `.git/` and `.venv/`, including ignored raw data, logs, caches, generated outputs, tracked evidence, and untracked research additions. The row-level inventory is [repository_inventory.csv](repository_inventory.csv).
- `.venv/` was excluded as an installed environment, not repository content. Git history, tags, and branches were inspected read-only.

The inventory is conservative: `KEEP` means retain pending normal review; `REVIEW_REQUIRED` means provenance or scientific status is unresolved; `REMOVE_CANDIDATE` is limited to regenerable local caches and does not authorize deletion.

## 2. Canonical research pipeline

1. Build RFM cohorts from Online Retail II and Dunnhumby with training-only, tie-preserving quintile cut points.
2. Construct crisp and baseline fuzzy RFM-FCA representations, threshold-scaled concepts, Gödel-min memberships, and greedy redundancy suppression.
3. Evaluate rolling-origin prediction with fold-local transformations and paired customer bootstrap inference.
4. Evaluate refit and temporal structural stability, then run the pre-specified matched-count diagnostic.
5. Aggregate evidence tables and validate the paper evidence map.

Authoritative v1 files:

- `scripts/baseline_ladder_rolling_origin.py` → `results/baseline_ladder_rolling_origin/`;
- `scripts/segment_stability.py` → `results/segment_stability/`;
- `scripts/matched_count_stability_diagnostic.py` → `results/segment_stability/matched_count_diagnostic/`;
- `scripts/final_evidence_tables.py` → `results/final_evidence/`;
- `scripts/base_paper_comparison.py` and `scripts/fair_comparison_retail2.py` → comparator and reconstruction outputs;
- `scripts/build_paper_evidence_map.py` and `scripts/validate_paper_evidence_map.py` → evidence-map generation/checking;
- `docs/METHODOLOGY_FINAL.md`, `docs/RESULTS_SUMMARY.md`, `docs/AUDIT_ERRATA.md`, `docs/PAPER_EVIDENCE_MAP.md`, and `docs/base_paper_methodology_audit.md`.

The protected v1 evidence is anchored by tag `evidence/v1-frozen`; its files and history were not modified.

## 3. Evidence map by study variant

| Study/status | Authoritative code/plan | Outputs | Status documentation |
|---|---|---|---|
| v1 crisp vs fuzzy primary | `baseline_ladder_rolling_origin.py`, `final_evidence_tables.py` | `results/baseline_ladder_rolling_origin/`, `results/final_evidence/` | `RESULTS_SUMMARY.md`, `METHODOLOGY_FINAL.md` |
| v1 stability and matched-count | `segment_stability.py`, `matched_count_stability_diagnostic.py` | `results/segment_stability/` | errata and stability reports |
| Base-paper comparator | `fair_comparison_retail2.py`, `base_paper_comparison.py`, `replicate_base_paper.py` | `results/fair_comparison_retail2/`, `results/base_paper_comparison/`, `results/base_paper_replication/` | base-paper audit/reports |
| v2 hybrid, exploratory/post hoc | `v2_hybrid_ladder.py` | `results/v2_hybrid/` | `V2_HYBRID_ANALYSIS_PLAN.md`, `V2_HYBRID_RESULTS.md`; tag `evidence/v2-hybrid` |
| CDNOW confirmation | `cdnow_confirmation.py` | `results/cdnow_confirmation/` | `CDNOW_CONFIRMATION_PLAN.md`, `CDNOW_CONFIRMATION_RESULTS.md`; tag `evidence/cdnow-confirmation` |
| CDNOW independent verification | `verify_cdnow_confirmation.py` | `results/cdnow_confirmation_verification/` | 59-check verification report |
| Limitation follow-ups | `qw_cdnow_stability.py`, `qw_inference_robustness.py`, `qw_interpretability_proxies.py` | `results/qw_*` | `QUICK_WINS_PLAN.md`, `QUICK_WINS_RESULTS.md`; tag `evidence/limitations-quick-wins` |
| New membership-function ablation | `membership_function_ablation.py`, tests | `results/membership_function_ablation/` | new plans/reports say smoke-only, but `full_run/` exists; review required |
| New tail-information hybrid | `rfm_tail_information.py`, tests | `results/rfm_tail_information/` | new plan/results; exploratory/post hoc |

## 4. Retention decisions

Must remain: frozen v1 scripts/results/evidence map; v2, CDNOW, verification, and limitation follow-ups; non-FCA baselines and comparator rows; historical/negative evidence; and data needed for applicable reproduction.

Safe only as candidates, not actions taken:

- `__pycache__/`, `.pytest_cache/`, and `.freebuff/`: regenerable local state, ignored by Git, and not evidence. These are the only clear removal candidates.
- `logs/stage2_run*.log`: likely historical logs, but provenance is not established; `REVIEW_REQUIRED`.
- `outputs/figures/unified_method_comparison/`: regenerable but its audit CSVs/report preserve figure provenance; keep until a reviewed manifest confirms no dependency.

No tracked result, script, document, dataset, or generated evidence table was identified as safe to delete.

## 5. Clutter and superseded experiments

Historical membership-function alternatives, threshold sensitivity, Kuznetsov, Kneedle, LRFM, older clustering/hierarchy/Hasse/overlap/stability work, and early Olist/RFMS material are not automatically disposable. Current documents use some as negative evidence or provenance and retain superseded documents under `docs/superseded_2026-10-08/`.

The current tree contains `.pyc` names for removed historical scripts; these are caches, not source or evidence. The new membership-function tree contains smoke, corrected smoke, corrected pooling smoke, and a `full_run/` directory, while its report says full evaluation is blocked. This status inconsistency requires review; it must not be silently promoted to publication evidence or deleted.

The untracked tail-information artifacts are documented as exploratory and hybrid, with output-isolation tests and frozen-arm reproduction guards. Keep them separate from frozen v1 evidence.

## 6. Risks and unresolved questions

1. Was `results/membership_function_ablation/full_run/` intentionally run, and should its report/status be updated?
2. Should the untracked analysis, output, test, ablation, and tail-information files be kept locally, added to Git, or archived?
3. Are `logs/stage2_run*.log` required for historical provenance?
4. May raw CDNOW and Dunnhumby files remain locally under their licensing/redistribution constraints?
5. README mentions an LRFM branch/plan not present in the current working-tree inventory; verify remote branch/tag state before editing documentation.
6. `pytest` has a host temp-directory setup failure in one test; this is described below.

## 7. Validation performed

- `python scripts/validate_paper_evidence_map.py` — PASS: 0 discrepancies; predictive rows 81, stability 33, provenance 21, cited IDs 92, paths 31, robustness 36, interpretability 6.
- `python -m compileall -q scripts analysis tests` — PASS.
- `pytest -q` — 15 passed, 1 setup error, 6 warnings. The error occurred while pytest created numbered temp directories under the host sandbox temp path, before the affected test body; it was not a repository assertion failure.
- CSV result schema inspection — PASS for readable result CSVs; headers and row counts were enumerated. No results were regenerated or replaced.
- `git status --short --branch` — initial status recorded above; no cleanup changes were made.

These checks do not establish full reproducibility: raw data, dependencies, compute time, and full experiment execution remain prerequisites.

## 8. Staged cleanup plan (approval required)

Stage 0: reconcile the membership-ablation full-run status, confirm untracked ownership, and decide treatment of ignored logs/raw data.

Stage 1: after approval, remove only caches using a manifest and restoration note; do not touch evidence or logs.

Stage 2: if approved, add provenance for exploratory analyses and optionally archive clearly historical generated outputs without breaking references. Prefer a dedicated branch/commit.

Stage 3: run a temp-directory-configured test suite, import/path/link checks, evidence-map validation, CSV schema checks, and `git diff --check`; review accidental deletions.

## Conclusion

The repository is consolidated around a frozen v1 evidence path with protected tags for subsequent studies. The only clear disposable material is ignored local cache state. Because the new untracked research additions include a status inconsistency and cleanup requires approval, this audit stops after producing the report. No files were removed, moved, renamed, overwritten, regenerated, committed, or pushed.


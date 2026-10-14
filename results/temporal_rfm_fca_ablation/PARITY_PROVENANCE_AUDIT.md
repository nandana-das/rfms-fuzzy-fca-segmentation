# Temporal ablation parity and provenance audit

Date: 2026-10-14

## R/F/M parity verification

The temporal scoring path was compared with the verified M0 builder on one real Dunnhumby origin/fold using the existing cohort and fold protocol. Before the fix, F and M scores matched but R scores were reversed: the maximum R-score difference was 4, and R membership differences reached 1.0. The cause was a second R inversion in `scripts/temporal_rfm_fca_ablation.py`; `quantile_scores` already performs the M0 recency inversion.

After removing the second R inversion, the narrow parity check reported:

- R/F/M score maximum absolute difference: `0`
- Training R/F/M membership maximum absolute difference: `0.0`
- Test R/F/M membership maximum absolute difference: `0.0`

The temporal test projection and the stability diagnostic were updated consistently. Regression coverage is in `tests/test_temporal_rfm_parity.py`; the targeted test run passed 3 tests.

## Historical result provenance

The original temporal-ablation CSVs were written at approximately 22:06:49 on 2026-10-13, before the current source file's later edit. The retained artifacts do not contain a source commit, source hash, or run manifest linking them to an exact script version. Their generating script version is therefore **not determinable** from the preserved evidence.

The existing temporal CSVs, report, and parameters were not overwritten or silently corrected. They remain historical outputs from an unresolved source state and should be treated as provisional until the temporal ablation is intentionally rerun under the corrected code.

The current corrected source hash is recorded for future provenance:

```text
scripts/temporal_rfm_fca_ablation.py
SHA256 5A4F0D65CBBE7CBA9BB946860C26D0C27CDC048B85661847CD6A158ED3E473DD
```

The prior matched-complexity and stability diagnostics likewise lack an exact historical source hash. Their outputs were preserved; the matched/stability scripts now import the corrected temporal scoring path for any future intentional rerun.

No temporal ablation, matched-complexity experiment, or stability experiment was rerun during this audit.

## R/F/M-only control and statistical audit

The corrected temporal R/F/M control was checked on an actual Dunnhumby origin/fold against `bl._fuzzy_context` without fitting downstream learners or regenerating experiment outputs. The score and membership results were exact:

- R/F/M score maximum absolute difference: `0`
- Training membership maximum absolute difference: `0.0`
- Test membership maximum absolute difference: `0.0`

The control also exposed a pre-fix interface limitation: `_score_temporal` is designed for temporal dimensions and cannot be called with R/F/M alone because it expects T-level metadata. The control therefore used the same quantile and membership primitives with only R/F/M, which is sufficient to verify parity but does not alter the experiment implementation.

The statistical/report-generation audit found and corrected three future-run issues without regenerating historical outputs:

1. Pooled point deltas and positive-origin counts now use an explicit pooled-origin map rather than positional slicing of the full origin list.
2. The generated report now documents the actual Holm family of nine pooled tests per dataset: three arm comparisons by three metrics.
3. The generated conclusion is built from the comparison dataframe rather than hardcoded delta and p-value text.

Targeted regression coverage now includes five passing tests across R/F/M parity, recency orientation, fuzzy-membership parity, pooled-origin indexing, and the nine-test Holm family. The existing CSVs and reports remain unchanged.

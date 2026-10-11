# M1-LAR-PW tail-scale discrepancy audit

Starting commit: `cac5454460e39c3210a954b0b17fc21a52c1b0a3` (`codex/lar-pw-primary-audit`).

## Finding

The discrepancy is an implementation/specification mismatch, not merely a label mismatch. The historical function `fit_lar_pw_parameters` computes the upper-tail scale at `scripts/tail_memberships.py:206–218`. When band-5 IQR is positive it uses that IQR. Otherwise, when `max(z5) > a4`, the executed formula is:

`s_tail = mean(z5 - a4)`.

The surrounding docstring and test call this “MAD,” but they do not implement or assert median absolute deviation. The requested protocol specifies an MAD fallback. Since `s_tail` enters the rational tail memberships, the two fallbacks produce different membership values whenever the fallback branch is reached; this is a mathematical method difference.

The original manifest records zero degenerate folds but no per-fold IQR, fallback flag, tail scale, or source hash. The retained fold diagnostics likewise do not record these quantities. Therefore the repository cannot establish from retained outputs alone whether the fallback branch was reached in any historical fold. The raw datasets and deterministic split configuration are present, so a lightweight prospective usage audit can determine branch reach without fitting downstream models; it cannot change what the historical result means.

## Source-to-specification table

| Item | Implemented behavior | Documented behavior | Evidence | Consequence |
|---|---|---|---|---|
| Primary scale | `IQR(z5)` from training band 5, only if positive | IQR-based tail scale | `tail_memberships.py:201–212`; original M1 docstring | Same when IQR > 0. |
| Fallback trigger | IQR ≤ 0 and `max(z5) > a4` | IQR unsuitable, with fallback | `tail_memberships.py:213–218`; audit task | Branch condition is explicit, but historical branch usage is not recorded. |
| Historical fallback formula | `mean(z5-a4)` | Documentation says “MAD” / task requires MAD | `tail_memberships.py:214–216`; `tests/test_tail_memberships.py:276–289` | Mean deviation is sensitive to extreme tail values and differs from MAD. |
| Corrected v2 formula | `median(abs(z5-a4))` | Exact v2 protocol clarification | `scripts/tail_memberships_v2.py`; `docs/LAR_PW_V2_EVALUATION_PROTOCOL.md` | Robust scale relative to the transition anchor; no outcome-based choice. |
| Degenerate scale | Raise `DegenerateTrainingFoldError` if scale ≤ 0 | Reject degenerate folds | `tail_memberships.py:220–223`; v2 implementation | No silent epsilon or test-data fallback. |
| Input validation | Reject NaN, infinity, and negative raw RFM values | Finite, nonnegative inputs | `check_rfm_values`; old and v2 tests | Preserves bounded finite projection assumptions. |
| Tail membership | `mu5=dz/(dz+s)`, `mu4=s/(dz+s)` for `z≥a4` | Same rational equation | `tail_memberships.py:236–298`; v2 reuses projection | Difference is entirely in fitted `s_tail` when fallback is used. |
| Historical provenance | Python/platform/seed/folds recorded; execution commit/branch absent | Reproducible execution provenance expected | `results/fuzzy_tnorm_tail_ablation/run_manifest.json`, normalized manifest | Cannot attribute historical results to a source revision with certainty. |
| Historical outputs | Aggregate metrics, diagnostics, reports retained | Preserve historical artifacts | `results/fuzzy_tnorm_tail_ablation/` | They remain valid historical M1 results, not v2 results. |

## Methodological choice

The corrected v2 choice is a protocol clarification, not a claim to restore the exact historical implementation. The fallback is defined as the median absolute deviation from `a4`:

`MAD_a4(z5) = median_i |z5_i - a4|`.

This is appropriate for the rational tail because the scale is attached to the excess over the transition anchor, and it is less influenced by a single extreme observation than the mean deviation. IQR remains the preferred scale when it is positive because it is a familiar robust dispersion measure with a wider central-spread interpretation. A non-finite or non-positive resulting scale is a degenerate training fold and is rejected. No downstream metric is used to select between scale estimators.

The choice is deliberately explicit because “MAD” can also mean deviation around `median(z5)`. The v2 protocol fixes the center at `a4`, matching the rational tail’s displacement variable `z-a4`; this is the smallest defensible change consistent with the intended formula and the existing code structure.

## Historical fallback reachability

The lightweight usage audit was run over the original nine origins and five training folds, fitting only membership parameters and not downstream models. It recorded IQR, `a4`, maximum band-5 value, fallback flag, historical scale, and v2 scale for F and M. The result is **0 fallback-reached rows out of 90 dimension-level training fits**; historical and v2 scales are numerically identical in all 90 rows. This establishes that the fallback discrepancy did not affect the retained nine-origin metrics under the reconstructed split boundary. The audit output and manifest are in `results/lar_pw_v2_tail_scale_audit/`.

This result does not prove the unavailable historical execution used the same source revision; it establishes reachability under the recorded data/origin/fold configuration. It also does not make historical results v2 results.

## Preservation decision

The historical implementation is left untouched. The corrected implementation is separately named `M1-LAR-PW-v2` in `scripts/tail_memberships_v2.py`, and its tests explicitly distinguish it from historical `M1-LAR-PW`. No historical result is relabeled, overwritten, or regenerated.

# Tail-scale correction decision memo

## Decision

Proceed with a separately named `M1-LAR-PW-v2` implementation and protocol clarification, but do not run a full corrected predictive experiment in this task and do not promote M1 to the primary contribution.

## Why

The historical implementation uses mean deviation from `a4`, while the documented specification says MAD. These formulas are mathematically different and must not be silently conflated. The corrected v2 implementation defines the fallback as `median(abs(z5-a4))`, while preserving the historical code and outputs.

The lightweight reachability audit covered all 9 rolling origins, 5 folds, and F/M dimensions. The fallback was reached 0/90 times; historical and v2 scales were identical in all 90 fits. A full v2 rerun on the same design would therefore reproduce the historical membership parameters and would not resolve the remaining uncertainty or add independent evidence.

## Paper recommendation

M1-LAR-PW remains a supplementary exploratory ablation. The existing nine-origin effects remain descriptive. A future full v2 evaluation should be run only if the study expands to settings where the fallback can be exercised or if customer-level prediction/seed/stability evidence is needed under the corrected protocol.

## Remaining limitations

- The historical execution commit and branch are absent.
- Historical output files do not retain per-fold scale values or fallback flags.
- Customer-level historical out-of-fold predictions and seed-level results remain unavailable.
- The choice of MAD centered at `a4` is a prospective protocol clarification, not a claim to reconstruct an undocumented historical intent.

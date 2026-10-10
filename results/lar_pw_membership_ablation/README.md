# LAR-PW Membership Ablation — Canonical Compatibility Record

This directory is a provenance and naming layer for the existing experiment.
It is **not** a new evaluation and contains no replacement metric values.

## Canonical identity

| Canonical label | Historical internal identifier | Membership implementation | T-norm |
|---|---|---|---|
| `M1-LAR-PW` | `M1` | `lar_pw` in `scripts/tail_memberships.py` | Gödel minimum |

The historical `M1` identifier remains unchanged in
`results/fuzzy_tnorm_tail_ablation/`.  In newly created reports and metadata,
`M1` means `M1-LAR-PW` only for that existing tail-sensitive F/M arm.  It must
not be confused with the separate Gaussian/generalized-bell/log-coordinate
membership-function ablation in `results/membership_function_ablation/`, or
with the separate log-RFM hybrid feature augmentation in
`results/rfm_tail_information/`.

## Source and verification

The authoritative historical artifacts remain under
`results/fuzzy_tnorm_tail_ablation/`.  The normalized manifest lists every
source file and records the available run metadata without changing those
files.  The run reported 9 pooled rolling-origin evaluations across Dunnhumby
and Online Retail II, with M0 parity reported as a maximum absolute difference
of `8.326672684688674e-17` against a tolerance of `1e-12` and zero degenerate
folds.  CDNOW was explicitly deferred in the original manifest.

The reconciliation itself performed no new evaluation.  Focused compatibility
tests check only the canonical mapping and manifest consistency; the existing
mathematical tests remain in `tests/test_tail_memberships.py`.

Execution commit and execution branch were not recorded in the historical run
manifest and are therefore left unavailable.  Git identifies `ec9f3f3` as the
commit that currently contains the source artifacts; this is an artifact
inclusion commit, not an asserted execution commit.

# Membership-function ablation smoke results

**Branch:** `experiment/membership-function-ablation`  
**Status:** smoke evaluation only; full evaluation awaits approval.

## What was run

The isolated runner `scripts/membership_function_ablation.py --smoke` evaluated the first Dunnhumby rolling origin (`day 347`, 2,497 customers), with five stratified customer folds shared by M0–M3. It preserved the audited five-level scoring, threshold scaling `(0.3, 0.5, 0.7)`, minimum support `0.04`, empty-core/Jaccard suppression, and downstream models. Smoke inference used 200 paired bootstrap draws instead of the full protocol's 2,000 only to bound this preliminary run.

All output files are under `results/membership_function_ablation/`. Frozen results and the locally completed tail-information experiment were not modified.

## Smoke metrics

| Method | ROC AUC | Spend R² | Invoice R² |
|---|---:|---:|---:|
| M0 frozen piecewise | 0.8965 | 0.5499 | 0.6244 |
| M1 normalized Gaussian | 0.8912 | 0.5502 | 0.6187 |
| M2 normalized generalized bell | 0.8911 | 0.5441 | 0.6130 |
| M3 log-coordinate Gaussian | 0.8953 | 0.5470 | 0.6225 |

Against M0, every 95% bootstrap interval included zero and every Holm-adjusted p-value was 1.0 in this one-origin smoke run. This is not evidence for or against a method in the planned full comparison.

## Representation diagnostics

Mean retained concepts over the five folds were 90.0 (M0), 55.2 (M1), 55.0 (M2), and 83.8 (M3). Mean normalized membership entropy was approximately 0.235 (M0), 0.476 (M1), 0.407 (M2), and 0.378 (M3). These are descriptive diagnostics and are not segmentation-quality criteria.

The fold-level retained concept-set Jaccard diagnostic was 0.862 (M0), 0.958 (M1), 0.819 (M2), and 0.922 (M3). It matches concepts by their underlying band sets across folds; it does not establish customer-level temporal stability and should not be interpreted as a predictive result.

## Verification status

- Python syntax compilation passed.
- Focused ablation suite passed: 7 tests.
- Existing repository regression suite passed: 31 tests.
- M0 feature matrices and downstream predictions matched the frozen implementation within exact/`1e-12` numerical tolerance on the regression fixture.
- Tests cover finite bounded memberships, normalized sums, coincident centers, extreme and nonpositive log inputs, training-only fitting, identical folds, and output isolation.
- Full rolling-origin evaluation was intentionally not started.

The exact equations, width floor, edge-case policy, frozen protocol, and planned full-evaluation outputs are in `docs/MEMBERSHIP_FUNCTION_ABLATION_PLAN.md`.

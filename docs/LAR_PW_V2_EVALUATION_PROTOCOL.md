# M1-LAR-PW-v2 corrected evaluation protocol

Protocol version: `M1-LAR-PW-v2 / 2026-10-11`.

This protocol is written before any corrected model evaluation. Historical M1-LAR-PW results remain descriptive and are never relabeled as v2.

## Method definition

M0 is unchanged: frozen Recency piecewise-linear memberships, the existing threshold-scaled binary context, concept mining, Gödel-min customer–concept membership, redundancy suppression, and downstream models.

M1-LAR-PW-v2 changes only the F/M tail-scale fallback:

1. For each F/M dimension, fit on training customers only using `z=log1p(x)`.
2. Use the four training-band medians `a1<a2<a3<a4` as interior anchors.
3. Use linear interpolation between adjacent anchors and a left shoulder below `a1`.
4. In the upper tail `z≥a4`, use `mu5=(z-a4)/(z-a4+s_tail)` and `mu4=s_tail/(z-a4+s_tail)`.
5. Set `s_tail=IQR(z5)` for training band 5 when IQR is positive.
6. If IQR is non-positive, set `s_tail=median(abs(z5-a4))`. If this is non-finite or non-positive, fail the training fold as degenerate.
7. Reject non-finite or negative raw inputs. Require all memberships finite, in [0,1], and summing to one.

The MAD center at `a4` is a prospective protocol clarification. It is not asserted to be the historical estimator because the historical execution revision and fallback-use records are unavailable.

## Data and temporal design

Use the same nine rolling origins as the historical experiment: Dunnhumby days 347, 438, 529, and 620; Online Retail II origins 2010-09-10, 2010-12-10, 2011-03-11, 2011-06-10, and 2011-09-09. Use the established 91-day holdout and existing RFM/outcome definitions. Origins overlap in customers and history; they are temporal evaluations, not independent replications.

## Paired fitting and leakage safeguards

At every origin, use identical customer indices, stratified five-fold splits, preprocessing, concept-mining settings, model classes, and seed for M0 and M1-LAR-PW-v2. Fit quantiles, anchors, tail scales, concepts, suppression, and downstream models inside each training fold. Project test customers only with training-fitted parameters. A failed or degenerate fold is recorded and causes the origin/seed evaluation to be marked incomplete; it is not silently dropped.

The pre-specified CV seeds are `0, 1, 2, 3, 4`. Within each seed, downstream model seeds are derived exactly as `seed + fold`, as in the established runner. No origin or seed may be selected after inspecting metrics.

## Outcomes and decision rule

Primary co-primary outcomes: Future Invoice R² and Future Spend R². Secondary outcome: Repurchase ROC AUC. Diagnostics: refit and temporal segment stability, concept count/redundancy, tail occupancy and uniqueness, and runtime.

There is no defensible domain-derived minimum practical effect margin in the repository, so the v2 evaluation is exploratory unless a margin is supplied and versioned before execution. Report paired effect sizes and intervals for each dataset and metric. If formal testing is used, apply Holm correction to the two co-primary outcomes within each dataset family; AUC is secondary and cannot rescue a primary failure.

Retain M1-v2 as a supplementary contribution if either co-primary outcome is inconsistent across datasets/seeds, if uncertainty spans practically negligible effects, or if stability/redundancy has a material adverse trade-off. Promote it to the primary methodological contribution only if both co-primary outcomes show robust, practically justified improvements across prespecified seeds and datasets, M0 parity passes, no material stability/redundancy penalty is found, and at least one validation setting is temporally or dataset-wise separate from the original development evidence. Reject the corrected variant if effects are negligible/inconsistent or adverse structural consequences outweigh predictive gains.

## Predictions, uncertainty, and missing data

For every origin × seed × fold × customer, retain customer ID, dataset, origin, seed, fold, outcomes, and paired M0/v2 predictions for repurchase, spend, and invoices. Also retain training/test membership parameters, tail IQR, MAD fallback flag, anchors, concept counts, and failure status.

Compute customer-clustered paired bootstrap intervals from the retained paired out-of-fold predictions within each origin. These intervals quantify test-customer sampling conditional on fitted folds; seed variation, refitting variation, and temporal shifts are reported separately. Do not manufacture prediction-level uncertainty from aggregate historical metrics.

Missing predictions, non-finite metrics, failed folds, or degenerate scales produce an incomplete run status and are reported in the manifest. No imputation or selective origin removal is permitted.

## Reproducibility manifest

Save a manifest containing protocol version, Git commit, branch, Python/platform/dependency versions, dataset file hashes, source-file hashes, random seeds, origins, fold counts, metric definitions, configuration constants, timestamps, output checksums, and completion/failure counts. Save outputs under `results/lar_pw_v2_primary_evaluation_v1/`, never under the historical result directories.

## Execution status

At protocol creation, no corrected model evaluation has been run. The lightweight fallback-reachability audit is required first. The full corrected evaluation may run only after the audit and protocol are committed and the required data/dependencies are verified.

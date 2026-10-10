# Membership-function ablation plan

**Branch:** `experiment/membership-function-ablation`  
**Status:** exploratory; this plan is written before implementation or evaluation.

## Question and scope

This ablation asks whether changing only the five-level RFM membership function changes predictive performance, information retention, or structural stability in the frozen RFM-FCA pipeline. The study is RFM-only. It remains separate from the LRFM extension and from the continuous-tail augmentation experiment.

The four pre-specified arms are:

| Arm | Membership representation | Purpose |
|---|---|---|
| M0 | Frozen centroid-based piecewise-linear memberships with saturating outer shoulders | Reproduction control |
| M1 | Normalized Gaussian memberships | Smooth, globally overlapping alternative |
| M2 | Normalized generalized-bell memberships | Smooth alternative with fixed shape |
| M3 | Normalized Gaussian memberships on `log1p` RFM coordinates | Tail-compressed exploratory alternative |

No arm is selected using held-out outcomes. Any full-evaluation conclusions will be descriptive and will report all four arms.

## Frozen protocol

The current audited pipeline is the reference implementation. The following remain unchanged initially:

- RFM construction, tie-preserving training quantile cutoffs, five levels per dimension, and R inversion;
- training-only fitting and projection of every data-dependent parameter;
- threshold-scaled binary context at 0.3, 0.5, and 0.7;
- FP-growth mining with no itemset length cap, exact-extent grouping, minimum support 0.04, and the existing closed-concept schema;
- Gödel-min customer-to-concept membership;
- greedy redundancy suppression with `J_max = 0.80`, `mu_cut = 0.50`, and empty-core removal;
- rolling origins, customer folds, outcomes, LogisticRegressionCV settings, RidgeCV settings, bootstrap inference, and Holm adjustment used by `baseline_ladder_rolling_origin.py`.

The existing tail-information artifacts are not reused as an implementation dependency and will not be modified.

## Exact membership equations

For each dimension `d`, training customers are assigned to five tie-preserving quantile bands. Let `c_{d,k}` be the median raw training value in band `k`, for `k = 1,...,5`. The band order is the project order: recency is inverted during scoring, while frequency and monetary value increase with the score. Test observations use the training cutoffs and fitted parameters unchanged.

### M0 — frozen piecewise-linear shoulders

After sorting the five centroids for the raw-coordinate projection, values at or below the first centroid receive membership 1 in level 1; values at or above the fifth centroid receive membership 1 in level 5. Between adjacent centroids `c_j < x < c_{j+1}`:

`mu_j(x) = (c_{j+1} - x)/(c_{j+1} - c_j)`  
`mu_{j+1}(x) = (x - c_j)/(c_{j+1} - c_j)`

All other memberships are zero. The implementation uses a deterministic `1e-9` denominator fallback for coincident centroids, while the audited training data has strictly increasing centroids after raw-coordinate sorting. Columns are restored to project band order after sorting.

### M1 — normalized Gaussian

For each training-derived center `c_k`,

`g_k(x) = exp(-0.5 * ((x - c_k)/sigma_k)^2)`  
`mu_k(x) = g_k(x) / sum_j g_j(x)`.

Widths are estimated without outcomes: let `g_j = |c_{j+1} - c_j|`. Zero gaps are replaced by the median of the positive gaps; if no positive gap exists, the gap fallback is `1e-12 * max(1, max_k |c_k|)`. The endpoint width is half its adjacent effective gap. An interior width is half the mean of its two adjacent effective gaps. Every width is floored at the same numerical fallback. Thus coincident centers are finite and receive equal Gaussian contributions when their widths and centers coincide.

### M2 — normalized generalized bell

Using the same training-only centers and widths as M1, with the fixed, non-tuned shape parameter `b = 2`:

`g_k(x) = 1 / (1 + |(x - c_k)/sigma_k|^(2b))`  
`mu_k(x) = g_k(x) / sum_j g_j(x)`.

The shape is fixed before evaluation and is not chosen from validation or test outcomes.

### M3 — log-coordinate Gaussian

For each nonnegative raw value, use `z = log1p(x)` and fit the five centers and M1 widths from the transformed training values in the training bands. The normalized Gaussian equation is then applied to transformed test values using the frozen training centers and widths. R, F, and M are all transformed; the recency direction remains encoded by the existing R score/band order. Because the project RFM variables are nonnegative, any negative value encountered by a defensive test/projection path is clipped deterministically to zero before `log1p`; non-finite values raise an error. No transformation parameter is fitted on test data.

## Fairness, diagnostics, and inference

Every arm receives identical customer indices for every fold. Metrics are ROC AUC for repurchase, R² for log1p future spend, and R² for log1p future invoices. Paired customer-level bootstrap deltas are computed against M0 with the existing per-origin pooling and Holm-adjustment convention. Concept candidates and retained concepts are recorded per fold. Membership overlap/entropy and concept stability are descriptive only; neither concept count nor entropy is evidence of segmentation quality by itself.

The primary stability diagnostic, if supported by the existing local pipeline without changing fit rules, is core-profile Jaccard at membership `>= 0.5`, matched by underlying band set. If matching is not valid for an alternative, the limitation will be recorded rather than patched with an outcome-informed rule.

## Safeguards

Focused tests will verify finite `[0,1]` memberships, normalized row sums, coincident-centroid behavior, constant/extreme values, training-only parameter fitting, identical folds, M0 feature/prediction reproduction, and result-path isolation. Syntax checks and the full existing regression suite run before the smoke evaluation.

The smoke evaluation will use the same code path but a bounded subset of the frozen cohort/origin protocol. It will write only under `results/membership_function_ablation/`. Full evaluation is intentionally not launched in this phase.

## Planned outputs

- `scripts/membership_function_ablation.py`
- `tests/test_membership_function_ablation.py`
- `docs/MEMBERSHIP_FUNCTION_ABLATION_RESULTS.md` (after smoke evaluation)
- `results/membership_function_ablation/` (smoke artifacts only until approval)

No frozen methodology document or historical result directory will be edited.

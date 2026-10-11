# M1-LAR-PW prospective evaluation protocol v1

Status: prespecified for any future additional evaluation; written before new model training. The retained nine-origin results remain descriptive and are not retroactively converted into confirmatory evidence.

## Research question and estimands

Does replacing only the M0 Frequency/Monetary membership representation with the specified log-coordinate asymptotic rational piecewise-linear (LAR-PW) representation improve out-of-sample predictive adequacy while avoiding unacceptable structural costs?

For each dataset and rolling origin, the estimand is the expected test-set metric difference for the same customers and same future holdout under paired M1-LAR-PW and M0 fits:

`Δ(metric) = metric(M1-LAR-PW) − metric(M0)`.

The primary outcomes are Future Invoice R² and Future Spend R², treated as co-primary. Secondary outcome is Repurchase ROC AUC. Diagnostics are refit/temporal segment stability, concept count and redundancy, upper-tail membership behavior, and comparable runtime.

Because the project has no domain-derived minimum practically meaningful R²/AUC margins, no numerical non-inferiority or superiority threshold is invented here. Until a domain justification is supplied, conclusions are exploratory: report effect sizes and intervals, and do not label M1 confirmatory merely because an interval excludes zero. A future protocol revision may add a margin only before inspecting new outcomes.

## Comparison arms

1. Crisp RFM-FCA versus M0 answers the foundational fuzzification question.
2. M1-LAR-PW versus M0 answers the membership-function question.
3. Gaussian/generalized-bell membership experiments are separate membership alternatives and must use their own canonical names.
4. The log-RFM hybrid changes the feature representation and is a separate augmentation question.

Only item 2 is the primary comparison in this protocol. M0 and M1 use the same rolling origins, customer folds, seeds, model classes, hyperparameter grids, concept-mining settings, t-norm, redundancy rule, and outcome definitions.

## Membership specification to lock before execution

- R membership remains the existing frozen M0 piecewise-linear construction.
- F and M use `z = log1p(x)`.
- Four strictly increasing training-only interior anchors are the training-band medians in log coordinates.
- Interior transitions are linear between adjacent anchors.
- For `z >= a4`, `mu5(z) = (z-a4)/((z-a4)+s_tail)` and `mu4(z) = s_tail/((z-a4)+s_tail)`; at `z=a4`, `mu5=0`, `mu4=1`.
- `s_tail` is the training-band-5 IQR. If IQR is nonpositive, the fallback must be defined exactly before the run. The current code uses mean deviation from `a4`, while the task specification says MAD; these are not interchangeable. Resolve this mismatch and test the exact chosen estimator before execution.
- No test observation may affect cutoffs, anchors, centroids, scaling, tail scale, concept mining, or redundancy suppression.
- Memberships must be finite, within [0,1], and sum to one.

## Data, splits, and pairing

Use the existing four Dunnhumby and five Online Retail II rolling origins only if the source data and protocol remain unchanged. At each origin, use the established 91-day holdout and stratified five-fold customer CV. Pair M0 and M1 within every fold and origin, reusing the same customer indices and random seed. If new seeds are run, use a prespecified seed set and pair the arms under each seed. Do not rerun the historical nine-origin experiment solely to make its result more favorable.

If a new dataset or genuinely later temporal window is available, declare it before execution as validation, keep it separate from the original origins, and do not pool it into the original claim without a planned hierarchical summary.

## Inference and multiplicity

For each origin, compute customer-level out-of-fold predictions for M0 and M1 and form paired metric differences. Use a customer-clustered paired bootstrap within origin, resampling customer rows with replacement and recomputing the metric difference from the paired predictions. Report the actual bootstrap method, number of draws, random seed, and interval method.

The two co-primary outcomes form one multiplicity family. If formal testing is later justified, use two-sided tests and Holm adjustment across the two primary outcomes within each prespecified dataset-level family; report each dataset separately. AUC is secondary and is not allowed to rescue a failure on a co-primary outcome. Stability and redundancy are diagnostics, not silently added to the predictive multiplicity family.

Bootstrap intervals describe test-customer sampling conditional on fitted models, folds, and the observed temporal origins. They do not capture variation from refitting, seed choice, or temporal shifts. Report seed variation and temporal/refit stability separately. Origins share customers and history, so pooled origin summaries are descriptive or model-based summaries, not independent-replication inference.

## Stability and redundancy diagnostics

Run the established stability definitions under matched conditions:

- refit stability: M1 versus M0 under the same 80% subsamples and reference fits;
- temporal re-segmentation: consecutive origins, all customers and the established unchanged-behavior subset;
- core-profile Jaccard and concept recurrence, matched by underlying band set;
- retained concept count, raw candidate count, duplicate/redundant concepts, empty-core count, and core concepts per customer;
- tail diagnostics: tail size, continuous membership uniqueness, binary profile diversity, and upper-tail occupancy.

Do not call a higher concept count “better” or lower redundancy “more interpretable” without a separate validity criterion. A materially worse stability pattern must qualify any predictive improvement.

## Proposed minimal additional analyses

| Analysis | Question | Inputs | Estimand/output | Cost and limitation |
|---|---|---|---|---|
| Paired customer bootstrap | How uncertain are M1−M0 metrics conditional on the fitted folds? | Customer-level paired OOF predictions and outcomes | Per-origin Δ and customer-clustered interval; dataset summary | Moderate; unavailable from current aggregate artifacts and not a refit-uncertainty analysis. |
| Paired seed analysis | Are results sensitive to CV/model seeds? | Raw data, same origins, prespecified seed set | Distribution of paired Δ across seeds | Requires new training; seeds are not temporal replications. |
| M1 stability/refit analysis | Does M1 change stability or redundancy relative to M0? | Saved concepts/memberships or new matched refits | Jaccard, recurrence, counts, redundancy deltas | Requires M1-specific fits; current baseline stability files do not answer it. |
| Tail-scale sensitivity | Does the result depend on the estimator? | Training folds and fixed, prospectively chosen estimators | Paired performance and tail diagnostics by estimator | Exploratory unless estimator choice is locked before outcomes. |
| Independent validation | Does the effect transfer? | New dataset or later non-overlapping window | Separate dataset-level Δ | Highest cost; absent in current retained M1 artifacts. |

The smallest necessary set for a primary-contribution decision is the paired bootstrap (if predictions can be recovered), M1 stability analysis, and a locked fallback specification. Seed and independent validation are needed for a stronger generalization claim.

## Decision rule

- **A, supported primary contribution:** both co-primary outcomes show robust, practically justified improvements under the prespecified inference, with no material stability/redundancy trade-off and appropriate validation.
- **B, promising but exploratory:** point effects are encouraging but uncertainty, practical margins, independent validation, or stability evidence remain insufficient.
- **C, not supported:** effects are inconsistent/negligible under the prespecified criteria or are offset by important adverse outcomes.

With the current evidence, the decision is B. This protocol does not authorize a large training run; it defines what must be approved and executed next.

Internal design note — optimized_fuzzy_fca_experiment
=====================================================

1. WHAT EXISTS AND WILL BE REUSED (read-only, no modification)
----------------------------------------------------------------
- project_paths.py — DATA_DIR / RESULTS_DIR paths.
- fair_comparison_retail2.py (read-only):
    * load_cleaned_transactions, aggregate_rfm  (Retail II cohort)
    * dense_rank_scores, extract_dense_rank_cutoffs, apply_dense_rank_cutoffs
    * band_centroids, piecewise_membership  (the centroid+pwlin machinery — we
      will NOT modify these; we will write a parallel generalized constructor
      in the new script that follows the same idea but allows variable levels)
    * compute_customer_concept_memberships  (Gödel-min t-norm; reused as-is)
    * mine_crisp_closed_concepts  (reused as-is for crisp reference)
    * mine_fuzzy_closed_concepts  (reused as-is; accepts l_thresholds param)
- fuzzy_membership_sensitivity.py (read-only):
    * load_and_prepare_cohorts  (Dunnhumby cohort)
    * mine_fuzzy_closed_concepts_with_thresholds  (accepts l_thresholds +
      min_support; reused as-is)
    * DIMS, SUPPORT_CUTOFF  constants
- concept_redundancy.py (read-only):
    * suppress_redundant_concepts  (J_max, mu_cut params; reused as-is)
- kuznetsov_pruning_leakage_free.py (read-only):
    * compute_split_stability  (exact canonical Kuznetsov; reused via exec,
      same pattern as four_way_method_comparison.py)
    * _paired_bootstrap_ci, _paired_permutation_test  (imported directly)
- fcm.py (read-only):
    * FuzzyCMeans  (kept as a baseline comparator arm only)


2. PARAMETERS THAT BECOME OPTIMIZATION VARIABLES
-------------------------------------------------
Stage 1 — fuzzy representation only (FCA selection fixed at validated baseline):

  A. per-dimension level counts:
      nR in {3,4,5,6,7}, nF in {3,4,5,6,7}, nM in {3,4,5,6,7}
      (asymmetric allowed, e.g. R=4,F=6,M=5)
  B. overlap / width of membership functions:
      The existing pwlin membership is fully determined by the band centroids
      and the score-band boundaries; there is no separate "sigma/width" knob
      in the current implementation. The natural parameter we CAN expose is
      the number of bands per dimension (which controls granularity/overlap
      indirectly) plus the L-fuzzy thresholds that binarize the memberships for
      FCA mining. We do NOT invent a synthetic width parameter. If a genuine
      width parameter exists in the codebase it would be in
      piecewise_membership/band_centroids, but inspecting those shows they are
      fully determined by centroids; therefore overlap is operationalized via
      (levels per dim, L-thresholds).
  C. fuzzy boundary locations:
      Boundaries are derived from training data via dense-rank score bands +
      centroid medians. We do NOT optimize raw quantile cutoffs directly
      (that would be a different discretization); instead we optimize the
      number of bands, which is the code's native knob. This respects the
      constraint that boundaries stay training-derived and ordered.
  D. R/F/M-specific granularity: supported via (nR,nF,nM) tuple.

  FCA selection kept fixed at Stage 1 baseline:
      min_support = 0.04, L_THRESHOLDS = (0.3,0.5,0.7),
      J_MAX = 0.80, MU_CUT = 0.50, no Kuznetsov (loss_threshold = 1.0 i.e.
      keep all valid-stability concepts, to isolate fuzzy representation).

Stage 2 — FCA selection only (fuzzy representation fixed at a representative
         good Stage-1 config, e.g. the baseline 5/5/5 or the best Stage-1
         config):
      min_support in [0.01, 0.10]
      Jaccard J_max in [0.60, 0.95]
      mu_cut in [0.40, 0.70]
      Kuznetsov loss_threshold in the validated set {1.0, 6.2e-2, 3.9e-3,
        1.5e-5, 5.4e-20}

Stage 3 — joint NSGA-II only if Stage 1/2 indicate a real opening and runtime
         budget allows.


3. LEAKAGE PREVENTION
---------------------
For each outer seed split:
  - RFM aggregation: train rows only (already cohort-level; holdout outcomes
    are separate columns; no leakage there).
  - dense-rank score bands + cutoffs: fitted on TRAIN only; test projected via
    frozen cutoffs (apply_dense_rank_cutoffs).
  - fuzzy memberships: centroids fitted on TRAIN only; test projected via
    frozen centroids (compute_fuzzy_memberships with trained_centroids=).
  - FCA mining: train fuzzy memberships only.
  - Kuznetsov stability: compute_split_stability on TRAIN fuzzy memberships
    only (same as validated leakage-free path).
  - Jaccard suppression: mu_matrix from TRAIN only.
  - predictive models: fit on TRAIN only; evaluate once on TEST.
  - optimizer: every configuration's fitness is the OUT-OF-SAMPLE test metric
    from that split, but the optimizer only SEES the training-fitted
    representation. To avoid optimizing to a single test split, Stage 1/2
    fitness uses an inner validation split inside TRAIN (e.g. 70/30 within
    train, stratified on repurchased). The outer 30% test is held out for the
    final per-configuration reporting and for the Stage-3 / repeated eval.
    Concretely:
      outer split: seed -> train_outer (70%) / test (30%)
      inner split: train_outer -> train_inner (70% of outer train) /
                   val (30% of outer train), stratified, fixed inner seed.
      Optimizer fitness = val metrics from inner split.
      Final reported metrics per config = test metrics from outer split.
    This is the critical separation: optimizer never sees the outer test set.

  The existing scripts already implement the outer train-only fitting. We add
  the inner validation split as the optimizer's objective source.

  IMPORTANT: configuration selection uses inner validation; the outer test is
  reported but NOT used to choose configurations. Final repeated evaluation
  (seeds 1000-1009) will be done with a SINGLE selected configuration (or a
  small set of Pareto configs) to avoid selection-on-test.


4. CANDIDATE CONFIGURATION EVALUATION
-------------------------------------
For one (config, split) evaluation:
  1. aggregate train RFM (already in cohort).
  2. dense-rank score train + fit cutoffs (train only).
  3. build generalized fuzzy memberships on train with (nR,nF,nM) bands;
     fit centroids on train; project train memberships.
  4. mine fuzzy closed concepts on train fuzzy memberships with given
     l_thresholds + min_support.
  5. (Stage 2+) compute_split_stability on train memberships; filter by
     loss_threshold; then Jaccard suppress with J_max/mu_cut.
  6. (Stage 1: skip stability filter; use loss_threshold=1.0 keep-all, then
     Jaccard with baseline J_max/mu_cut.)
  7. build customer-concept membership matrix (Gödel-min) on train.
  8. inner split: train_inner vs val within train_outer.
  9. fit LogisticRegressionCV + RidgeCV on train_inner; evaluate on val.
  10. fitness = normalized predictive composite from val:
        For each metric m in {AUC, SpendR2, InvR2}:
          norm_m = (m - ref_min_m) / (ref_max_m - ref_min_m)
        where ref_min/ref_max are FIXED constants derived from the existing
        validated results (NOT from the test set of this experiment). Use the
        approximate existing result ranges as anchors:
          AUC   ref in [0.75, 0.88]
          SpR2  ref in [0.20, 0.52]
          InvR2 ref in [0.34, 0.62]
        These anchors are fixed before evaluation and documented. They are
        derived from existing validated experiments, not from this experiment's
        test outcomes.
        PredictiveScore = mean(norm_AUC, norm_SpendR2, norm_InvR2)
     Complexity score = retained_concept_count (lower is better), possibly
        also recorded: candidate_concept_count, n_final, features dim.
     Objectives for NSGA-II:
        obj1 = -PredictiveScore   (minimize = maximize predictive)
        obj2 = retained_concept_count  (minimize complexity)
     Both objectives are scalar and cheap.

  For the smoke test we evaluate a small grid / tiny population and confirm
  each stage runs end-to-end on one split, one dataset (Dunnhumby), with
  deterministic seeds, and that Pareto solutions are produced.


5. SMOKE TEST SCOPE
-------------------
- Dataset: Dunnhumby Complete Journey (smaller, faster).
- One outer split: seed 42, 70/30 stratified.
- Inner split: 70/30 within train, fixed inner seed 123.
- Population = 6, generations = 3 (toy NSGA-II), or just a tiny explicit grid
  of ~12 configs if NSGA-II setup is not ready yet — the goal is to validate
  the pipeline, not to find a good config.
- Variables for smoke: restrict nR,nF,nM in {3,4,5} each; min_support fixed at
  0.04; L_THRESHOLDS fixed at baseline; J_MAX/MU_CUT fixed; no Kuznetsov
  (loss<=1.0). This is a pure Stage-1 fuzzy-representation smoke.
- Verify:
    * generalized fuzzy membership builder produces valid memberships (all in
      [0,1], each customer has at least one band with mu>=0.5, dimensions
      sum reasonable).
    * FCA mining completes (no exceptions), candidate count > 0.
    * (Stage 2 path) stability computation completes for a few concepts.
    * Jaccard suppression completes, retained > 0.
    * predictive models fit and evaluate; metrics in sane ranges.
    * Pareto front produced (non-empty), with at least 2 non-dominated points.
    * No leakage: assert test data never passed to membership/centroid/mining/
      stability/suppression/model-fit.
- Record runtime.


6. REUSABILITY / ISOLATION NOTES
---------------------------------
- We do NOT modify fair_comparison_retail2.py's compute_fuzzy_memberships,
  band_centroids, or piecewise_membership. We write a new isolated function
  generalized_fuzzy_memberships(scored, n_levels_per_dim, dims) in the new
  script that follows the same centroid+pwlin logic. Import the existing
  helpers (band_centroids, piecewise_membership) but call them in a loop over
  per-dimension level counts — this reuses verified logic without changing it.
  Actually: band_centroids and piecewise_membership assume 5 bands. To support
  variable levels without modifying them, the new builder replicates their
  logic generically. This is acceptable because it is new code in the new
  script, not a modification of existing validated code. We will cross-check
  that for (5,5,5) it reproduces the existing memberships closely (sanity).
- We do NOT modify concept_redundancy.py, fuzzy_membership_sensitivity.py,
  kuznetsov_pruning_leakage_free.py, fcm.py.
- The canonical Kuznetsov path is reused via exec of
  kuznetsov_pruning_retail2_leakage_free.py is Retail-II-specific; for
  Dunnhumby we reuse kuznetsov_pruning_leakage_free.py's compute_split_stability
  via exec (same pattern as four_way_method_comparison.py). We must be careful
  to exec the DUNNHUMBY leakage-free script for Dunnhumby and the RETAIL2 one
  for Retail II. For the smoke test we only do Dunnhumby, so we exec
  kuznetsov_pruning_leakage_free.py.


7. OUTPUTS (per the experiment spec)
-------------------------------------
- optimized_fuzzy_fca_configurations.csv  (every evaluated config)
- optimized_fuzzy_fca_pareto.csv
- optimized_fuzzy_fca_summary.csv  (best predictive / best compact / balanced)
- optimized_fuzzy_fca_comparison.csv  (vs Raw RFM, existing fuzzy FCA,
  existing Kuznetsov FCA, FCM)
- figures: pareto.png, sensitivity plots (levels vs perf, support vs perf,
  stability threshold vs perf) — only if the experiment actually produces the
  data to support them.
- README-ish report in results/optimized_fuzzy_fca/report.md


8. SUCCESS CRITERIA (verbatim from spec)
-----------------------------------------
A. Optimized FCA significantly beats raw RFM on predictive + reasonable
   complexity -> strong evidence.
B. Similar predictive to raw RFM but richer representation + compression ->
   representation/segmentation contribution, NOT predictive superiority.
C. Still fails to outperform raw RFM and no compelling complexity advantage
   -> report explicitly: optimization did not establish sufficient empirical
   advantage. Do NOT force positive.

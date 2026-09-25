# RFMS-Fuzzy FCA Segmentation — Results Summary (Olist + Online Retail II, REVISED v3)

## Revision note (v3 — second methodological review)
v2 fixed the F* objective and a documentation/denominator mismatch. This version (v3) fixes the
largest remaining issue flagged by a second independent review: **`max_len=6` was hardcoded in the
FP-growth call**, silently truncating the concept lattice at itemset length 6. That cap was verified
to be actively binding (9,618/10,283 Olist concepts, 704/1,531 Retail II concepts sat exactly at the
size-6 wall). Removing it (uncapped mining runs in ~1s; true max itemset length is 12) revealed the
previously-reported concept counts were substantially inflated by spurious "closed" itemsets that are
not actually closed once larger supersets with identical support are allowed to be found. All
downstream steps (4, 5, 7) were rerun on the corrected lattice, and the mandatory Kneedle
sensitivity analysis (both audits flagged this as required before treating any concept count as
final) was performed for the first time. Terminology and wording fixes from the second review are
also applied throughout.

### Fixes applied in v3
1. **Removed `max_len=6` cap** in `step2_fuzzy_fca.py` and `step7_cross_domain_retail2.py`
   (`max_len=None`). Olist raw closed concepts: **10,283 → 1,369** (corrected). Retail II: **1,531 →
   1,064**. The previous larger counts were an artifact of the length cap, not a real lattice.
2. **Terminology: "triangular fuzzy membership" → "centroid-based piecewise-linear fuzzy membership."**
   The implementation linearly interpolates between adjacent band centroids, saturating to 1.0 beyond
   the outermost centroids. Middle bands are triangular; the two outer bands are shoulder/trapezoidal.
   Calling all five bands "triangular" was imprecise.
3. **Wording: do not call the entropy-derived F\* weights "the optimal purchase-intensity
   formulation."** Per review, the correct framing is: *"the entropy-maximizing weight combination
   within the specified grid and scoring procedure"* — entropy-maximization is a criterion chosen to
   investigate sparsity mitigation, not an independently validated ground truth for "purchase
   intensity" in general.
4. **Kneedle sensitivity analysis performed** (Step 8, new) — see below. This was listed as
   "outstanding work item 1" in v2 and flagged as mandatory by both reviews.

## Step 1 — RFMS Feature Engineering (unchanged from v2 — max_len fix does not affect Step 1)
- 96,096 true unique customers (via `customer_unique_id`) vs 99,441 order-scoped `customer_id`.
- After delivered-order + valid-payment filtering: **93,357 customers** retained.
- Repeat-buyer rate: **3.00%** (single-order rate 97.00%).
- F\* grid search selects **α=0.20, β=0.05, γ=0.75** (band entropy 0.1457 nats, 9.1% of the ln(5)
  maximum) — the entropy-maximizing blend within the searched grid, not claimed as an independently
  optimal purchase-intensity formulation. f_score=1 share = **97.00%**, matching the literal
  single-order rate exactly.
- Within the f_score=1 group (n=90,553), F\* retains a small but non-zero spread (std=0.008, 15
  distinct values) — some continuous signal survives into Step 2; whether it becomes useful downstream
  structure is evaluated empirically (Steps 5-6), not assumed.
- **Review-missingness limitation:** 603 customers (0.65%) had no observed customer-level review score
  and received the observed median, S=5.0. Since 5.0 is also the observed mode (54,449/92,754; 58.70%),
  imputation increases the S=5.0 count to 55,052 (58.97% of the full population, +0.27 percentage
  points). This may modestly increase measured support for concepts containing the highest S band;
  the imputed values should not be interpreted as evidence that those customers were highly satisfied.

## Step 2 — Fuzzy Formal Context (v3: max_len fix + terminology fix)
- Fuzzy membership matrix (93,357 × 20 attributes); row-sums = 1.0 per dimension confirms correct
  partition construction.
- **Centroid-based piecewise-linear fuzzy membership** (not "triangular" — see fix #2 above).
- L-fuzzy scaling at thresholds {0.3, 0.5, 0.7} (Belohlavek threshold reduction) into a crisp
  multi-level context; closed itemsets enumerated via uncapped FP-growth on that scaled context. This
  is a scalable computational realization of the thresholded fuzzy context, not a literal
  implementation of exact fuzzy Next-Closure.
- Raw frequent itemsets (uncapped, min_support=0.02): **40,906** (size distribution: 1:51, 2:584,
  3:2586, 4:6189, 5:9339, 6:9618, 7:7038, 8:3705, 9:1385, 10:352, 11:55, 12:4; true max length 12).
- **Closed itemsets (fuzzy formal concepts): 1,369** (corrected down from the max_len=6-truncated
  figure of 10,283 — the truncated run mistook thousands of non-maximal size-6 itemsets for closed
  concepts because it never searched for the larger supersets that actually subsume them).

## Step 3 — Stability + Iceberg Pruning (v3: rerun on corrected 1,369-concept lattice)
- Kneedle-derived thresholds: **supp_min\* = 0.1025**, **theta\* = 0.9876**.
- Surviving concepts: **153** (from 1,369), with **304 Hasse covering edges**.
- Stability proxy (1 − unique_profiles/extent_size) is a tractable diagnostic, not the canonical
  exponential-cost Kuznetsov stability index; it depends on the same 4-dimensional score profiles used
  for scoring, creating a dependency chain (`scoring → profiles → stability → pruning`).

### Kneedle sensitivity analysis (new, Step 8 — mandatory per both reviews)
Ran on both datasets: support threshold and stability threshold each perturbed at 0.9×/1.0×/1.1×,
independently and jointly, against the full pre-pruning concept set (1,369 for Olist, 1,064 for
Retail II).

**Support-threshold-only** (theta fixed at theta\*) — well-behaved, monotonic, no cliffs:

| | 0.9× | 1.0× (base) | 1.1× |
|---|---|---|---|
| Olist | 184 (+20.3%) | **153** | 144 (−5.9%) |
| Retail II | 139 (+20.9%) | **115** | 97 (−15.7%) |

**Stability-threshold-only, multiplicative** (support fixed at supp_min\*): both datasets' theta\* is
already close to the theoretical ceiling of 1.0 (0.9876 and 0.9315 respectively), so a ×1.1
multiplicative perturbation pushes the threshold to exactly 1.0 and collapses survivors to 0. This is
a **boundedness artifact of multiplying a value already near 1.0**, not evidence of real instability —
flagged and re-tested additively:

| Δtheta\* | Olist survivors | Retail II survivors |
|---|---|---|
| −0.05 | 153 (+0.0%) | 115 (+0.0%) |
| −0.02 | 153 (+0.0%) | 115 (+0.0%) |
| 0 (base) | **153** | **115** |
| +0.02 | 0 (ceiling hit; theta clips to 1.0) | 113 (−1.7%) |
| +0.05 | 0 (ceiling hit) | 70 (−39.1%) |

**Reading:** downward perturbation of theta\* is completely stable on both datasets (no change at
−0.02/−0.05). Upward perturbation is not informative for Olist because theta\* is already close enough
to 1.0 that any positive additive shift saturates the bound; Retail II (theta\*=0.9315, more headroom)
shows a real and non-trivial sensitivity upward (−1.7% at +0.02, −39.1% at +0.05), meaning the knee is
in a genuinely steep part of the stability curve there. Support-threshold sensitivity is mild and
monotonic on both datasets (≤21% either direction under ±10%). **Overall: the pruning choice is
reasonably but not perfectly stable** — support threshold is robust, theta threshold is robust
downward and untestable-upward for Olist (ceiling artifact) but shows real upward sensitivity for
Retail II. This should be reported as a genuine open point, not glossed over as fully robust.
Full grids saved to `kneedle_sensitivity_olist.csv`, `kneedle_sensitivity_retail_ii.csv`, and the
additive-theta supplements.

## Step 4 — Alpha-Cut Hard Cluster Assignment (v3: rerun on corrected 153-concept lattice)
- **12 top-level single-dimension concepts** (F1, S5, M1, M2, R4, R5, R3, M3, R2, S4, M4, R1) — up from
  8 in v2 (the corrected, smaller-but-more-accurate lattice surfaces more distinct top-level bands).
- Hard-cluster distribution still dominated by F1 (84,199 / 93,357 = 90.2%).
- **99.86%** of customers satisfy the alpha-cut in more than one top-level concept (up from 97.72%).
- This overlap remains a property of the fuzzy representation by construction; it demonstrates the
  representation *is* overlapping, not that the overlap is behaviorally meaningful. Concrete
  interpretable profile examples (e.g., an F1/M4/S5 customer) are still not produced — outstanding.

## Steps 5-6 — Quantitative Benchmarking (v3: rerun on corrected Step 4 output)

| method | k | silhouette | davies_bouldin |
|---|---|---|---|
| Fuzzy-FCA (natural k) | 12 | 0.166 | 5.947 |
| Fuzzy-FCA (k-matched) | 4 | 0.259 | 2.220 |
| Fuzzy-FCA (k-matched) | 5 | 0.247 | 2.392 |
| Fuzzy-FCA (k-matched) | 6 | 0.231 | 2.592 |
| K-means | 4/5/6 | 0.349 / 0.387 / 0.392 | 0.995 / 0.839 / 0.867 |
| Hierarchical (Ward) | 4/5/6 | 0.302 / 0.348 / 0.351 | 1.145 / 0.912 / 0.912 |
| K-means (outlier-removed) | 4/5/6 | 0.320 / 0.344 / 0.358 | 0.945 / 1.053 / 0.994 |
| Hierarchical (outlier-removed) | 4/5/6 | 0.309 / 0.332 / 0.337 | 0.919 / 1.076 / 1.050 |

**Fuzzy Partition Coefficient (FCA-only): 0.2370** (v2 reported 0.3039 under the max_len-truncated,
larger lattice — the corrected, smaller lattice produces a lower FPC; the direction of this change
should be noted honestly rather than cited from the stale v2 number).

**k-matched Silhouette/DB values are essentially unchanged from v2** (0.23–0.26 vs 0.259/0.247/0.231
before) — the max_len fix changed the underlying lattice substantially (1,369 vs 10,283 raw concepts,
153 vs 423 pruned) but the *top-level single-dimension bands* used for k-matched comparison were
already dominated by the same few large bands (F1, S5, M1, M2, R-bands) in both versions, so this
particular metric is robust to the fix. Natural-k comparison is not directly comparable across
versions since natural k itself changed (8 → 12).

### Interpretation (unchanged framing from v2, still applies)
> The crisp conversion of FCA memberships produces weaker compactness/separation scores than
> distance-based clustering. This indicates the two approaches encode different structural objectives;
> FCA should not be evaluated solely through partition compactness.

FPC is reported descriptively, not as proof that Silhouette/DB are "wrong" or that segmentation
quality is thereby validated. Additional fuzzy-validity diagnostics (membership entropy, resampling
stability, sensitivity to α and to the {0.3,0.5,0.7} thresholds beyond what Step 8 covers, concept
persistence) remain outstanding.

Base-paper-style outlier removal retains 68,001/93,357 (72.84%) customers (unaffected by the max_len
fix, since it operates on raw RFMS features, not the lattice).

## Step 7 — Cross-Domain Validation (Online Retail II, v3: max_len fix applied and rerun)

Dataset: 1,067,371 raw rows → 805,549 after cleaning → **5,878 unique customers** (exact match to base
paper's Table 3). Same entropy-based F\* objective (selects α=0.05, β=0.15, γ=0.80; band entropy
1.6087/1.6094 — near-ideal, since this domain is not degenerate).

| Metric | Olist (RFMS, 20 attrs) | Online Retail II (RFM, 15 attrs) |
|---|---|---|
| Customers | 93,357 | 5,878 |
| Repeat-buyer rate | 3.00% | 72.39% |
| Raw closed concepts (uncapped) | 1,369 | 1,064 |
| Pruned concepts (Kneedle) | 153 | 115 |
| Hasse edges | 304 | — |
| Lattice density ρ=\|L_f\|/2^\|M\| | 0.001306 (2^20 universe) | 0.032471 (2^15 universe) |
| Mean concept-membership/customer | — | 60.07 |
| Top-level band concepts | 12 | 14 |
| ARI (FCA hard vs K-means, k=14) | — | **0.2021** |
| FCA Silhouette / DB | 0.166 / 5.947 (k=12) | −0.171 / 13.216 (k=14) |
| K-means Silhouette / DB | 0.349–0.392 / 0.84–1.00 | 0.514 / 0.923 |

**Note on comparability:** the pipeline is not literally "identical" across datasets — the common RFM
component was applied identically, with the Olist-only Satisfaction dimension omitted for Retail II
(no review data exists there). Lattice density figures are computed over different attribute-universe
sizes (2^20 vs 2^15); the density gap (25× rather than v2's reported 4.5×, since Olist's corrected
concept count fell much further than Retail II's) should be read as a property of the differently-sized
universes and the differing degree of max_len-cap correction on each dataset, not interpreted as
"Retail II's lattice is unconditionally richer."

### Cross-domain interpretation (unchanged conclusions, updated numbers)
1. Sparsity is domain-specific, not a pipeline artifact: the identical scoring pipeline produces
   near-ideal quintile balance (entropy 99.96% of maximum) on the repeat-heavy domain and a genuinely
   low-entropy result (9.1% of maximum) on the marketplace domain, using the same code.
2. **ARI=0.2021** (was 0.1986 pre-fix — materially unchanged) indicates low correspondence between the
   FCA-derived and K-means partitions on Retail II. Stated as: *"consistent with the hypothesis that
   the methods capture different structural relationships"* — not confirmation that they answer
   genuinely different questions, which the ARI number alone does not establish.
3. The FCA-vs-K-means Silhouette/DB gap persists on both datasets and is larger on Retail II. Consistent
   with the gap being a structural property of crisp-converting overlapping fuzzy concepts rather than
   an Olist-specific sparsity artifact, though this remains an interpretation the numbers support rather
   than strictly prove.

## Step 9 — f_score=1 group structure-recovery test (new, resolves outstanding item 2)
Question: within the dominant f_score=1 group (90,553 customers, 97.00% of the population, all
crisp-tied on frequency), does the fuzzy/FCA pipeline recover any usable differentiation from R/M/S
and from F\*'s tiny residual spread, or does the group collapse into one indistinguishable mass?

- **Hard-cluster fan-out:** 6,354/90,553 (7.02%) of f_score=1 customers are assigned to a hard cluster
  *other than* F1, driven entirely by R/M/S since F is tied for the whole group. Real, but modest —
  92.98% still land in F1.
- **Concept-membership fan-out:** within-group std of concept-membership count (14.49) is 99.0% of the
  population-wide std (14.63) — the f_score=1 group is essentially as internally differentiated by the
  lattice as the population at large; it is not a homogeneous blob. 58 distinct membership-count values
  and 111 distinct alpha-cut band signatures occur within the group.
- **F\* residual correlation:** corr(F\*, concept-membership-count) = **−0.151**, corr(F\*,
  n_bands_satisfying_alpha) = **−0.163**, both within the f_score=1 group. These are weak. **This is an
  honest negative-ish finding, not fully favorable**: the tiny F\* spread inside the group (std=0.008,
  15 distinct values, per Step 1) is not the source of the differentiation found above — the
  differentiation is coming from R, M, S, not from F\* itself. F\*'s within-band residual signal appears
  close to dead weight for downstream structure, even though it technically survives as numeric
  variation.
- **Reading:** the fuzzy representation *does* recover real, non-trivial structure among nominally
  frequency-tied customers (via R/M/S), which is a genuine positive result for the fuzzy-FCA framework
  generally. But the specific hoped-for mechanism — F\*'s continuous residual smoothing over the
  frequency dimension itself — is not what is doing that work. The paper should credit the
  multi-dimensional fuzzy lattice, not F\* specifically, for f_score=1 differentiation, and should not
  claim F\*'s entropy-optimized weighting "solves" frequency sparsity at the individual-customer level.

Full detail (90,553 rows) saved locally as `f1_group_recovery_detail.csv`; a 36-row sample
(top-3 by concept-membership count per hard_cluster) pushed to the Project as
`f1_group_recovery_sample.csv` (the full file was too large for the Project's knowledge cap).

## Step 10 — Interpretable overlap-segment profiles (new, resolves outstanding item 3)
Question: what does the overlapping fuzzy representation say about a customer that a single crisp
hard_cluster label (or a plain RFMS quintile score) would not? Selected systematically, not by
cherry-picking: ranked all 218 distinct alpha-cut top-level band signatures by frequency, restricted to
the 169 "nontrivial" signatures touching ≥3 of the 4 RFMS dimensions (91,210 customers, 97.7% of the
population), took the top-6 most frequent nontrivial signatures, and pulled 3 representative customers
per signature (closest to that signature's median F\*).

**The clearest finding is the collapse test, not the top-6 table:** the single hard_cluster="F1" label
(84,199 customers, the dominant crisp assignment) spans **R (recency) = 0 to 694 days** (median 219,
IQR 115–345), **M (monetary) = $9.59 to $13,664.08** (median $101.37, IQR $59.99–$169.82), and the full
S range 1–5 — and contains **111 distinct overlap signatures**. A plain crisp RFMS system that
hard-assigns one label per customer reports all of these as indistinguishably "F1." The fuzzy overlap
representation preserves which R-band, M-band and S-band each of these customers *also* satisfies,
recovering exactly the differentiation a single-label system discards. This directly operationalizes
the abstract "97.72%/99.86% of customers satisfy >1 band" overlap statistic (§Step 4) into a concrete,
reviewer-checkable claim.

The top-6 nontrivial-signature table (e.g. `('F1','S5','M2','R5')` n=4,379: R=[0,121]d median 61,
M=[$82.76,$144.78] vs `('F1','S5','M1','R2')` n=3,384: R=[356,476]d median 408, M=[$11.63,$82.74]) shows
the same pattern at finer grain: signatures sharing the same F/S/M-tier bands but different R-bands
correspond to genuinely different recency/monetary profiles, all mapped to the same F1 or S5 hard label
by the crisp assignment. Full example table: `overlap_profile_examples.csv` (18 representative
customers across the top 6 signatures).

## Step 11 — Additional fuzzy-validity diagnostics (new, resolves outstanding item 4)
Four diagnostics beyond FPC, all on Olist:

**(a) Membership entropy** (per-dimension, mean across customers; max possible = ln(5) = 1.609):
| Dim | Mean entropy | % fully crisp (H=0) | % meaningfully fuzzy (H>0.1) |
|---|---|---|---|
| R | 0.410 | 18.1% | 78.9% |
| F | 0.014 | 89.8% | 10.2% |
| M | 0.383 | 22.5% | 74.3% |
| S | 0.005 | 99.3% | 0.7% |

R and M carry genuine, substantial fuzziness (~75-79% of customers meaningfully fuzzy). **F and S are
nearly crisp in practice** (89.8% and 99.3% fully crisp respectively) — F because F\*'s within-band
spread is tiny (consistent with the Step 1/9 finding that F\* carries little independent signal), S
because Olist review scores are themselves discrete 1-5 integers with little room to fall between
centroids. This is an honest finding, not fully favorable: two of the four fuzzy dimensions are doing
close to no fuzzy work. The value of the fuzzy representation is concentrated in R and M.

**(b) Alpha-cut sensitivity** (top-level overlap % vs α, 12 fixed top-level bands):
| α | % satisfying >1 band | % satisfying 0 bands | mean bands/customer |
|---|---|---|---|
| 0.4 | 99.91% | 0.00% | 4.02 |
| 0.5 (used) | 99.86% | 0.00% | 3.71 |
| 0.6 | 98.69% | 0.07% | 3.38 |

Smooth, monotonic response with no discontinuity — α=0.5 is not a knife-edge choice.

**(c) L-fuzzy threshold sensitivity** (closed-concept count under alternate threshold sets):
| Threshold set | Raw itemsets | Closed concepts |
|---|---|---|
| {0.3,0.5,0.7} (baseline) | 40,906 | 1,369 |
| {0.2,0.4,0.6} | 74,462 | 2,322 |
| {0.25,0.5,0.75} | 39,936 | 1,506 |

Closed-concept count moves smoothly with the threshold set (not collapsing or exploding), though it is
not threshold-invariant — {0.2,0.4,0.6} gives 70% more concepts. The lattice size is threshold-set
dependent, as expected for any threshold-scaling reduction; this should be stated plainly rather than
implied to be an intrinsic property of the data.

**(d) Resampling stability / concept persistence** (10× 80% subsamples, no replacement, seed=42):
**100% of the 153 baseline pruned concepts reappear as frequent itemsets in all 10 subsamples**
(mean persistence fraction = 1.000). Raw closed-concept counts across subsamples: 1280-1361 (baseline
full-population: 1369), a tight ±3% band. This is a strong, unambiguous robustness result — the
pruned lattice is not an artifact of the specific 93,357-customer sample.

Detail: `concept_persistence.csv` (per-concept persistence fraction, all = 1.000 in this run).

## Step 12 — Retail II parity tests (new — Steps 9-11 replicated cross-domain)
Olist's dominant crisp-tie group was f_score=1 (97.00%, frequency-tied — sparse marketplace). Retail II
is not frequency-sparse (72.39% repeat rate); its dominant crisp-tie group is instead **r_score=5**
(3,219/5,878 = 54.76% — most customers in this repeat-heavy retailer ordered recently). Running the
same three tests here checks whether the Olist findings generalize or are domain-specific.

**Structure recovery (analog of Step 9):** 1,038/3,219 (32.25%) of r_score=5 customers get a non-R
hard cluster (vs Olist's 7.02% for f_score=1) — a **much stronger** fan-out than Olist. Within-group
concept-membership std is 96.4% of population-wide std (vs Olist's 99.0% — comparable). corr(R,
n_concepts) within the group = **−0.33** (vs Olist's F\* correlation of −0.15) — stronger, and still
negative, meaning within-band recency variation does not positively drive differentiation either;
again the differentiation is supplied by the *other* dimensions (F, M), not by residual variation in
the tied dimension itself. This cross-domain agreement (negative/weak correlation with the tied
dimension's own residual, both times) strengthens the Step 9 conclusion rather than being an
Olist-specific artifact.

**Overlap profiles (analog of Step 10):** hard_cluster="R5" (2,178 customers) spans **M = \$20.80 to
\$608,821.65** (median \$1,750.79), **n_orders = 1 to 398** (median 6), and **21 distinct overlap
signatures**. Same collapse pattern as Olist's F1 finding, on the opposite dominant dimension —
confirms the phenomenon is a property of the fuzzy-FCA framework, not specific to Olist's frequency
sparsity. Top-5 nontrivial signatures follow the same pattern (e.g. `R5+F5+M5` n=685: M up to
\$608,821.65 vs `R5+F1+M1` n=225: M up to \$292.20, both partly hard-assigned to R5).

**Fuzzy-validity diagnostics (analog of Step 11):**
| Dim | Mean entropy | % fully crisp | % meaningfully fuzzy |
|---|---|---|---|
| R | 0.316 | 32.4% | 63.6% |
| F | 0.392 | 20.6% | 75.6% |
| M | 0.388 | 20.2% | 75.7% |

Unlike Olist (where F and S were nearly crisp), **all three Retail II dimensions carry substantial
fuzziness** (63.6-75.7% meaningfully fuzzy) — consistent with Retail II not having Olist's degenerate,
near-constant F\* problem. Alpha-cut sensitivity is smooth (100%→92.87% overlap, α=0.4→0.6, no
knife-edge). L-threshold sensitivity: {0.3,0.5,0.7}(baseline)=1,064 closed, {0.25,0.5,0.75}=1,216,
{0.2,0.4,0.6}=2,228 — same smooth-but-not-invariant pattern as Olist. **Resampling stability: 100% of
the 115 baseline pruned concepts persist across all 10×80% subsamples** (raw closed-concept counts
1043-1071 vs baseline 1064, a tight ±2.5% band — even tighter than Olist's ±3%).

**Overall cross-domain reading:** every Olist finding from Steps 9-11 replicates on Retail II, several
more strongly (fan-out 32% vs 7%, persistence band tighter). Fuzzy membership is more uniformly
meaningful across dimensions in Retail II (no near-crisp dimension, unlike Olist's F/S). This is a
genuine strength of the cross-domain validation — the differentiation and robustness findings are not
Olist-specific artifacts.

Files: `retail2_overlap_profile_examples.csv`, `retail2_concept_persistence.csv`.

## Step 13 — Publication figures (new)
Four figures generated (`scripts/step13_figures.py`, PNG output in `results/figures/`, delivered
directly to the user — the Project's docs tool is text-only and rejects images, same restriction
already hit with zip archives):
1. `fig1_lattice_size.png` — raw vs. Kneedle-pruned concept counts, Olist vs. Retail II.
2. `fig2_benchmark_k5.png` — Silhouette / Davies-Bouldin at k=5, Fuzzy-FCA vs. K-means vs. Hierarchical
   (Olist), color-coded, showing the FCA gap honestly (not hidden).
3. `fig3_hasse_diagram_top40.png` — Hasse diagram of the top-40 (by support) of the 153 pruned Olist
   concepts, node size/color = support, 60 covering edges among the subset. Full 153-node diagram would
   be too dense to read; this is a legible representative subset.
4. `fig4_kneedle_curves.png` — support and stability curves with the Kneedle knee point marked, for
   both Olist and Retail II (4 panels), visually confirming the elbow-detection is picking a genuine
   inflection point, not an arbitrary rank.

## Outstanding work (updated)
1. ~~Sensitivity analysis of the Kneedle-derived thresholds~~ **DONE (Step 8)** — support threshold
   robust, stability threshold robust downward, ceiling-limited upward for Olist, genuinely sensitive
   upward for Retail II (report both findings, do not overclaim full robustness).
2. ~~Quantitative validation of f_score=1 structure recovery~~ **DONE (Step 9, this revision)** — real
   differentiation exists (7.02% non-F1 hard-cluster fan-out, near-population-level concept-membership
   variance) but it comes from R/M/S, not from F\*'s own residual spread (weak/negative correlation,
   r=-0.15). State this precisely; do not credit F\* for an effect that belongs to the other dimensions.
3. ~~Concrete, interpretable segment profiles for overlapping customers~~ **DONE (Step 10, this
   revision)** — systematic (not cherry-picked) top-6 nontrivial-signature selection, plus the
   collapse test: hard_cluster="F1" alone spans R=0-694 days, M=$9.59-$13,664, and 111 distinct
   overlap signatures, concretely supporting the "overlap is useful" claim.
4. ~~Additional fuzzy-validity diagnostics~~ **DONE (Step 11, this revision)** — membership entropy
   (R/M genuinely fuzzy, F/S nearly crisp — honest mixed finding), α-sensitivity (smooth, no
   knife-edge), L-threshold sensitivity (smooth but not invariant — 70% swing at {0.2,0.4,0.6}),
   concept persistence (100% of pruned concepts survive 10×80% resampling — strong robustness result).
5. Precision pass on terminology throughout the eventual paper draft: "L-fuzzy-scaled formal context
   with closed-itemset enumeration" rather than an unqualified "FP-growth = fuzzy formal concepts";
   "purchase-intensity index" rather than "frequency" for F\*; stability proxy explicitly labeled as an
   approximation; "centroid-based piecewise-linear fuzzy membership" rather than "triangular"; the F\*
   weights described as "entropy-maximizing within the specified grid," never as "optimal" in general.
6. ~~Remove the `max_len=6` restriction~~ **DONE (this revision)** — but note the resulting lattice
   sizes are now materially smaller than previously reported (1,369 vs 10,283 raw Olist concepts), so
   any prior draft text, slides, or abstract citing the old numbers must be updated, not just the
   scripts.

## Files in this delivery
```
rfms_project/
├── scripts/
│   ├── step1_rfms_prep.py             (v2: entropy-based F* objective)
│   ├── step2_fuzzy_fca.py             (v3: max_len=None, terminology fix)
│   ├── step3_stability_pruning.py     (v3: rerun on corrected lattice)
│   ├── step4_alpha_cut_clusters.py    (v3: rerun on corrected lattice)
│   ├── step5_benchmark.py             (v3: rerun on corrected Step 4 output)
│   ├── step7_cross_domain_retail2.py  (v3: max_len=None, stale comparison number fixed)
│   ├── step8_kneedle_sensitivity.py   (new: mandatory sensitivity analysis)
│   ├── step9_f1_group_recovery.py     (new: f_score=1 structure-recovery test)
│   ├── step10_overlap_profiles.py     (new: interpretable overlap-segment profiles)
│   ├── step11_fuzzy_validity_diagnostics.py  (new: entropy, alpha/L sensitivity, persistence)
│   ├── step12_retail2_parity.py       (new: Steps 9-11 replicated on Retail II)
│   └── step13_figures.py              (new: publication figures)
├── data/
│   ├── raw/                              (9 Olist CSVs)
│   ├── raw_retail2/                      (2 Online Retail II CSVs)
│   └── processed/
│       ├── olist_rfms_features.csv          (v2, unaffected by v3 fix)
│       ├── olist_rfms_with_hard_clusters.csv  (v3: rerun)
│       ├── soft_membership_top_level.csv      (v3: rerun)
│       └── retail2_rfm_features.csv           (v2, unaffected by v3 fix)
├── results/
│   ├── fuzzy_concepts_raw.pkl            (1,369 concepts, Olist, v3)
│   ├── pruned_fuzzy_concepts.pkl         (153 concepts, Olist, v3)
│   ├── hasse_edges.pkl                   (304 edges, Olist, v3)
│   ├── retail2_fuzzy_concepts_raw.pkl    (1,064 concepts, Retail II, v3)
│   ├── retail2_pruned_fuzzy_concepts.pkl (115 concepts, Retail II, v3)
│   ├── benchmark_comparison.csv          (v3: rerun)
│   ├── cross_domain_comparison.csv       (v3: rerun)
│   ├── fpc_result.txt                    (v3: FPC=0.2370)
│   ├── kneedle_sensitivity_olist.csv               (new)
│   ├── kneedle_sensitivity_retail_ii.csv           (new)
│   ├── kneedle_sensitivity_olist_theta_additive.csv     (new)
│   ├── kneedle_sensitivity_retail_ii_theta_additive.csv (new)
│   ├── f1_group_recovery_detail.csv      (new, local only, 90,553 rows)
│   ├── f1_group_recovery_sample.csv      (new, pushed to Project, 36-row sample)
│   ├── f1_chunks/f1_detail_*.csv         (new, pushed to Project, per-hard_cluster split of the
│   │                                       full detail file -- F1 downsampled to 2,000, others full)
│   ├── overlap_profile_examples.csv      (new: Step 10, 18 representative customers)
│   ├── concept_persistence.csv           (new: Step 11, 153 concepts, persistence fraction)
│   ├── retail2_overlap_profile_examples.csv  (new: Step 12)
│   ├── retail2_concept_persistence.csv       (new: Step 12, 115 concepts, persistence fraction)
│   ├── figures/fig1_lattice_size.png     (new: Step 13, local + delivered to user)
│   ├── figures/fig2_benchmark_k5.png     (new: Step 13, local + delivered to user)
│   ├── figures/fig3_hasse_diagram_top40.png  (new: Step 13, local + delivered to user)
│   ├── figures/fig4_kneedle_curves.png   (new: Step 13, local + delivered to user)
│   └── RESULTS_SUMMARY.md                (this file)
└── PROJECT_DOCUMENT.md                   (v3)
```

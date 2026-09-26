# RFMS-Fuzzy FCA Customer Segmentation: Research Gap, Objectives, Methodology, Results (REVISED v3)

**Revision note (v3):** a first independent review fixed two real issues (F* objective was degenerate;
F* was mislabeled as "frequency"). A second review found the single largest remaining issue: a
hardcoded `max_len=6` FP-growth cap was silently truncating the concept lattice (verified binding —
9,618/10,283 Olist concepts sat exactly at the size-6 wall). Removing it corrected the closed-concept
count from 10,283 to **1,369** for Olist (1,531 → 1,064 for Retail II) — the earlier, larger counts
included thousands of itemsets that are not actually closed once the larger (7-12-attribute)
supersets the cap prevented from being found are included. All downstream steps were rerun on the
corrected lattice, and the mandatory Kneedle sensitivity analysis (required by both reviews before
treating any concept count as final) was performed. Terminology fixes ("triangular" →
"centroid-based piecewise-linear"; F* weights framed as "entropy-maximizing within the specified grid,"
not "optimal") are applied throughout.

## 1. Research Gap

Base study (Rungruang et al., "RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA")
combines RFM with binary Formal Concept Analysis (FCA) on UK single-retailer data (Online Retail II).
Gaps in that work, feasible for a Q1-level extension:

1. **Binary/crisp FCA context** — hard quintile cutoffs; author's own future-work note flags non-binary
   context as unexplored.
2. **Only 3 RFM variables** — no satisfaction/engagement dimension; author flags variable extension
   (RFMT, LRFM, RFMTC) as future work.
3. **Arbitrary concept-lattice pruning** — support threshold (0.04) chosen manually, no derivation method.
4. **Single-domain validation** — only tested on one repeat-buyer-heavy UK retailer; generalizability
   to other domains (e.g. sparse-frequency marketplaces) untested.
5. **Asymmetric evaluation** — FCA judged narratively; K-means/hierarchical judged with Silhouette/DB.
   No shared quantitative yardstick.
6. **Untested on marketplace data** — classic RFM assumes repeat purchase behavior; multi-seller
   marketplaces (e.g. Olist) are dominated by one-time buyers, breaking standard frequency scoring.
   Not addressed anywhere in RFM-FCA literature.

## 2. Research Objectives

1. Replace binary FCA with **fuzzy (non-binary) FCA** using centroid-based piecewise-linear membership
   functions, resolving the hard-cutoff limitation.
2. Extend RFM to **RFMS** (add Satisfaction, from review scores), motivated by Olist having review data
   unavailable in the base paper's dataset.
3. Investigate whether **frequency-sparsity** in marketplace data (Olist: 97% one-time buyers) can be
   mitigated via a composite purchase-intensity measure (F*) and dense-rank scoring, and quantify how
   much of that sparsity is a genuine domain property vs an artifact of naive scoring.
4. Replace arbitrary support thresholds with a **principled, data-driven pruning method** (Kneedle
   elbow algorithm on support + stability distributions).
5. **Validate cross-domain generalizability** by running the identical (common-component) pipeline on
   both a marketplace dataset (Olist) and a single-retailer dataset (Online Retail II).
6. Establish **symmetric quantitative benchmarking** (Silhouette, Davies-Bouldin, plus a
   fuzzy-only Fuzzy Partition Coefficient) placing FCA and K-means/hierarchical on equal evaluative
   footing, reporting results honestly even where FCA does not "win."

*(Objective 3 rephrased per review: the original framing — "fix frequency-sparsity via F*" — overstated
what was demonstrated. F* is a purchase-intensity index, not literal purchase frequency once its
optimal weights lean on item-row and repeat-flag signal; whether it meaningfully mitigates sparsity is
an empirical question this project investigates, not a result it assumes.)*

## 3. Methodology

### 3.1 Data preparation & RFMS feature engineering
- Join orders to true unique customers via `customer_unique_id` (Olist) / `CustomerID` (Retail II) —
  not the order-scoped ID.
- Filter to delivered/valid transactions; drop nulls, cancellations, non-positive quantity/price.
- Aggregate per customer: Recency (R), order count (n — the literal frequency variable, reported
  separately), Monetary (M), Satisfaction (S, Olist only — mean review score).
- For Satisfaction, review scores are first averaged within order; observed order-level means are then
  averaged per customer. Customers with no observed customer-level score (603/93,357; 0.65%) receive
  the median of observed customer-level S (5.0). Among observed customers, 5.0 is also the mode
  (54,449/92,754; 58.70%). Imputation therefore raises the S=5.0 count to 55,052 (58.97% of the
  full population, +0.27 percentage points). This may modestly increase measured support for concepts
  containing the highest satisfaction band; it does not establish the missing customers' true scores.
- **Composite purchase-intensity index**: F* = α·n + β·Σlog(1+qty) + γ·1[repeat]. Weights fit via grid
  search maximizing the **Shannon entropy of the resulting 5-band quintile histogram** — this
  criterion was revised from an original variance-ratio objective, which was degenerate (trivially
  maximized by whichever raw term has the highest variance alone, never producing a genuine blend).
  The entropy objective directly targets the actual goal: spreading customers across score bands as
  evenly as the data allows. The resulting weights (Olist: α=0.20, β=0.05, γ=0.75) are reported as
  **"the entropy-maximizing weight combination within the specified grid and scoring procedure,"**
  not as an independently optimal purchase-intensity formulation in general — entropy-maximization is
  the criterion chosen to investigate sparsity mitigation, not a validated behavioral ground truth.
- **Dense-rank fractional scoring**: score = ceil(5 · dense_rank(x) / K), where **K = the number of
  distinct raw values** (max of the dense rank), not the total customer count. This denominator choice
  is what the implementation has always used; an earlier written draft of this methodology incorrectly
  stated the denominator as total customer count, which would collapse nearly all scores to band 1
  whenever K ≪ m — that was a documentation error, corrected here.

### 3.2 Fuzzy formal context construction
- Band centroids computed per dimension (median raw value within each crisp score group).
- **Centroid-based piecewise-linear fuzzy membership**: linear interpolation between adjacent centroids
  gives each customer a continuous [0,1] membership in each of 5 bands per dimension. *(Terminology
  note, per review: middle bands are triangular, but the two outer bands saturate to 1.0 beyond their
  centroid — technically shoulder/trapezoidal, not pure triangular. "Triangular" alone is imprecise.)*
- **L-fuzzy scaling** (Belohlavek threshold reduction) at thresholds {0.3, 0.5, 0.7} converts the fuzzy
  context into a crisp multi-level context.
- **Closed frequent itemset mining** (FP-growth, uncapped — `max_len=None`) on the scaled context.
  *(Terminology note, per review: this is a scalable computational realization of the thresholded fuzzy*
  *context via closed-itemset enumeration — not a literal implementation of the exact fuzzy*
  *Next-Closure algorithm. The two are related through the threshold-scaling reduction, and the*
  *equivalence should be stated at that precision rather than as "FP-growth = fuzzy formal concepts.")*
  *(Implementation note: an earlier version of this code capped `max_len=6`, which was found to be*
  *actively binding — most concepts sat exactly at that size wall — and was removed. Uncapped mining*
  *is still fast (~1s at this scale, true max itemset length 12) and revealed the true closed-concept*
  *count is substantially smaller than the capped run reported, since many apparently-closed size-6*
  *itemsets are subsumed by larger supersets once those are allowed to be found.)*

### 3.3 Stability-based iceberg pruning
- **Stability proxy**: 1 − (unique customer score-profiles in a concept's extent / extent size). This
  is a tractable diagnostic, explicitly *not* the canonical (exponential-cost) Kuznetsov stability
  index. It is computed from the same 4-dimensional score profiles whose discretization is itself a
  modeling choice, creating a dependency chain (scoring → profiles → stability → pruning) that should
  be read as such rather than as an independent validation.
- **Kneedle elbow algorithm** applied to the support and stability distributions separately, deriving
  `supp_min*` and `theta*` without manual tuning. A sensitivity analysis (0.9×/1.0×/1.1× perturbation
  of both thresholds, plus an additive check for the stability threshold since it sits close to its
  1.0 ceiling on both datasets) was run — see §4.3a. Result: support-threshold sensitivity is mild and
  monotonic on both datasets; stability-threshold sensitivity is robust downward on both, and upward is
  either ceiling-limited (Olist) or genuinely non-trivial (Retail II, −39% at +0.05). This is reported
  as a partial, not complete, robustness result.

### 3.4 Cluster assignment & benchmarking
- **α-cut (α=0.5)** on top-level single-dimension concepts, mode assignment → hard clusters; full soft
  membership retained separately.
- **K-means** and **agglomerative hierarchical (Ward)** on the identical standardized score space at
  k=4,5,6, plus a base-paper-style branch with IQR outlier removal + log transform + min-max scaling.
- **Silhouette** and **Davies-Bouldin** computed identically across all methods; **Fuzzy Partition
  Coefficient (FPC)** for FCA only.

### 3.5 Cross-domain validation
- Common RFM component of the pipeline applied identically to Online Retail II, with the Olist-only
  Satisfaction dimension omitted (no review data exists there) — this is a partial, not literal,
  identity between the two runs, and is described as such.
- Domain-invariant metrics: lattice density ρ=|L_f|/2^|M| (computed over different attribute-universe
  sizes per dataset — 2^20 vs 2^15 — so cross-dataset density comparisons should be read with that
  caveat), mean concept-membership count per customer, Adjusted Rand Index (ARI) between FCA hard
  clusters and K-means clusters.

## 4. Results

### 4.1 Data scale (validation against base paper)
| | Olist | Online Retail II |
|---|---|---|
| Raw transaction rows | 96,477 (post-clean) | 805,549 (post-clean) |
| Unique customers | **93,357** | **5,878** (exact match to base paper's Table 3) |
| Repeat-buyer rate | 3.00% | 72.39% |

### 4.2 Sparsity finding (revised)
Under the corrected entropy objective, F* selects a genuine 3-way blend on Olist (α=0.20, β=0.05,
γ=0.75), achieving only 9.1% of the maximum possible band entropy — 97.00% of customers land in the
bottom frequency band, exactly matching the raw single-order rate. Cross-domain contrast: the identical
scoring pipeline on Retail II (repeat-heavy) achieves 99.96% of maximum entropy with the same code,
selecting near-perfectly balanced weights (α=0.05, β=0.15, γ=0.80). This supports that Olist's sparsity
is a genuine domain property rather than a pipeline defect.

Whether fuzzy membership smoothing "solves" that sparsity was tested directly (§4.4a, Step 9): the
fuzzy lattice **does** recover real downstream differentiation among f_score=1 customers (7.02%
non-F1 hard-cluster fan-out; concept-membership variance within the group at 99.0% of the
population-wide level) — but that differentiation is driven by R/M/S, not by F*'s own residual spread
(corr(F*, concept count) = −0.151, weak). The honest conclusion: the fuzzy-FCA *framework* mitigates
frequency-tie collapse, but the specific mechanism is multi-dimensional fuzzy overlap, not F*'s
entropy-optimized weighting acting alone. Claims should credit the framework, not F* in isolation.

### 4.3 Fuzzy concept lattice (revised — v3, max_len=6 cap removed)
| | Olist (RFMS, 20 attrs) | Retail II (RFM, 15 attrs) |
|---|---|---|
| Raw closed concepts (uncapped) | **1,369** (was 10,283 under the length cap) | **1,064** (was 1,531) |
| Pruned concepts (Kneedle) | **153** (was 423) | **115** (was 129) |
| Hasse edges | 304 (was 561) | — |
| Lattice density ρ | 0.001306 | 0.032471 |
| supp_min* / theta* | 0.1025 / 0.9876 | 0.1170 / 0.9315 |

The pre-fix counts above are not a real lattice — they were inflated by the `max_len=6` cap, which
prevented discovery of larger (7-12-attribute) supersets that subsume many apparently-closed size-6
itemsets. The corrected counts are smaller but methodologically sound.

### 4.3a Kneedle threshold sensitivity (new — mandatory per both reviews)
Support-threshold perturbation (0.9×/1.0×/1.1×, theta fixed) is monotonic and mild: Olist 184/**153**/144
(+20%/−6%), Retail II 139/**115**/97 (+21%/−16%). Stability-threshold perturbation is robust downward on
both datasets (no change at −0.02/−0.05 additive) but upward behaves differently per dataset: for Olist,
theta\*=0.9876 is close enough to the 1.0 ceiling that any positive additive shift saturates it (0
survivors, an artifact of the bound, not evidence of instability); for Retail II, theta\*=0.9315 has more
headroom and shows real sensitivity (113 at +0.02, 70 at +0.05 — a genuine steep region of the curve).
**Conclusion stated honestly:** the pruning choice is reasonably, not perfectly, stable — robust to
moderate perturbation in most directions tested, with one dataset (Retail II) showing genuine upward
sensitivity in the stability threshold that should be disclosed rather than hidden.

### 4.4a f_score=1 structure-recovery test (new, Step 9)
Within the 90,553-customer (97.00%) f_score=1 group, all crisp-tied on frequency: 6,354 (7.02%) get a
hard cluster other than F1 (driven by R/M/S); within-group concept-membership-count std is 99.0% of the
population-wide std (not a homogeneous blob); 111 distinct alpha-cut band signatures occur within the
group. F*'s own residual spread correlates weakly/negatively with downstream structure (r=−0.15 to
−0.16), meaning the differentiation found is attributable to R/M/S, not to F* itself. See §4.2.

### 4.4 Overlapping structure (v3: rerun on corrected lattice)
99.86% of Olist customers satisfy the α-cut in more than one top-level concept (12 top-level bands, up
from 8). This demonstrates the representation *is* overlapping by construction. §4.4b turns this
abstract statistic into a concrete, checkable claim about interpretable, behaviorally distinct segments.

### 4.4b Interpretable overlap-segment profiles (new, Step 10 — resolves §4.7 item 3)
Systematic (not cherry-picked) selection: ranked all 218 distinct alpha-cut top-level signatures by
frequency, restricted to 169 "nontrivial" signatures touching ≥3 of 4 RFMS dimensions (91,210
customers, 97.7% of the population), took the top-6 most frequent, pulled 3 representative customers
per signature (nearest the signature's median F\*).

**Collapse test (the clearest evidence):** hard_cluster="F1" (84,199 customers, single crisp label)
spans **R=0–694 days** (median 219), **M=\$9.59–\$13,664.08** (median \$101.37), the full S=1–5 range,
and **111 distinct overlap signatures**. A hard-assignment crisp system reports all of these
identically as "F1"; the fuzzy overlap representation preserves which R/M/S band each customer also
satisfies. This is the concrete evidence for the "overlap is useful" claim — not just that overlap
exists (§4.4), but that it carries real, checkable behavioral differentiation a single label discards.

Top-6 table (finer detail, same pattern): signatures sharing F1+S5+M-tier bands but different R-bands
map to genuinely different recency/monetary ranges (e.g. `F1+S5+M2+R5`: R=[0,121]d, M=[\$82.76,\$144.78]
vs `F1+S5+M1+R2`: R=[356,476]d, M=[\$11.63,\$82.74]) while both hard-assign to F1 or S5. Full table:
`overlap_profile_examples.csv`.

### 4.4c Additional fuzzy-validity diagnostics (new, Step 11 — resolves §4.7 item 4)
**Membership entropy** (per-dim, max=ln(5)=1.609): R mean H=0.410 (78.9% meaningfully fuzzy), M
mean H=0.383 (74.3%), F mean H=0.014 (89.8% fully crisp), S mean H=0.005 (99.3% fully crisp). Honest
mixed result: R and M carry genuine fuzziness; F and S are nearly crisp in practice (F because F\*'s
within-band spread is tiny, consistent with §4.4a; S because review scores are discrete integers with
little room between centroids). The fuzzy representation's value is concentrated in R and M, not
uniform across all four dimensions.

**Alpha-cut sensitivity:** overlap % moves smoothly with α (99.91%→99.86%→98.69% at α=0.4/0.5/0.6,
mean bands/customer 4.02→3.71→3.38) — α=0.5 is not a knife-edge choice.

**L-fuzzy threshold sensitivity:** closed-concept count is smooth but not invariant across nearby
threshold sets — {0.3,0.5,0.7}(baseline)=1,369, {0.25,0.5,0.75}=1,506, {0.2,0.4,0.6}=2,322 (a 70% swing).
Lattice size is threshold-set dependent, as expected for a threshold-scaling reduction; stated plainly,
not implied to be an intrinsic data property.

**Resampling stability:** 10×80% subsamples (no replacement, seed=42) — **100% of the 153 baseline
pruned concepts reappear as frequent itemsets in all 10 subsamples** (persistence fraction = 1.000
across the board); raw closed-concept counts stay in a tight 1280-1361 band (baseline: 1369, ±3%). A
strong, unambiguous robustness result.

### 4.5 Quantitative benchmark (v3: rerun on corrected Step 4 output)
| Method | Olist Silhouette (k=4-6) | Olist DB (k=4-6) |
|---|---|---|
| Fuzzy-FCA (k-matched) | 0.23 – 0.26 | 2.2 – 2.6 |
| K-means | 0.35 – 0.39 | 0.84 – 1.00 |
| Hierarchical (Ward) | 0.30 – 0.35 | 0.91 – 1.14 |

These k-matched values are essentially unchanged from before the max_len fix (0.23-0.26 both times) —
the top-level single-dimension bands used for k-matched comparison were already dominated by the same
few large bands (F1, S5, M1, M2, R-bands) under both the truncated and corrected lattices, so this
particular metric happened to be robust to the fix. FCA's Silhouette is **positive at every k**, and
FCA still trails both baselines on these metrics on both datasets. Stated framing: *"the crisp
conversion of FCA memberships produces weaker compactness/separation scores than distance-based
clustering, indicating the two approaches encode different structural objectives — FCA should not be
evaluated solely through partition compactness."*
**FPC = 0.2370** (Olist; was 0.3039 under the max_len-truncated, larger, pre-fix lattice — the
corrected, smaller lattice yields a lower FPC, reported honestly here rather than citing the stale
higher number). FPC is reported descriptively as an additional fuzzy-partition diagnostic, not as
proof that Silhouette/DB are the "wrong" metric or that FCA's segmentation is thereby validated.

### 4.6 Cross-domain structural agreement (v3: rerun with max_len fix)
**ARI = 0.2021** (was 0.1986 pre-fix — materially unchanged) between FCA hard clusters and K-means
clusters on Retail II. Stated framing: this
indicates limited correspondence between the two partitions, *consistent with* the hypothesis that the
methods capture different structural relationships — not confirmation that they "answer genuinely
different questions," which the ARI number alone does not establish.

### 4.6a Retail II parity tests (new, Step 12 — Steps 9-11 replicated cross-domain)
Retail II's dominant crisp-tie group (not frequency, since it isn't sparse) is **r_score=5** (54.76%,
recency-tied). Replicating Steps 9-11 there: **1,038/3,219 (32.25%)** of r_score=5 customers get a
non-R hard cluster (stronger than Olist's 7.02%); corr(R, concept count) within the group = **−0.33**
(same direction as Olist's F\* result, −0.15 — the tied dimension's own residual does not drive
differentiation in either domain). Overlap collapse test: hard_cluster="R5" (2,178 customers) spans
M=\$20.80–\$608,821.65 and 21 distinct signatures — same pattern as Olist's F1 finding, on the opposite
dominant dimension. Validity diagnostics: unlike Olist, **all three Retail II dimensions are
meaningfully fuzzy** (63.6–75.7%, no near-crisp dimension); resampling stability gives **100% concept
persistence** (115/115 across 10×80% subsamples, raw counts 1043–1071 vs baseline 1064, ±2.5%).
**Every Olist finding replicates cross-domain, several more strongly** — the differentiation and
robustness results are not Olist-specific artifacts.

### 4.7 Outstanding work before this is Q1-submission-ready
1. ~~Sensitivity analysis around the Kneedle-derived thresholds.~~ **DONE (§4.3a).** Support threshold
   robust; stability threshold robust downward, ceiling-limited upward for Olist, genuinely sensitive
   upward for Retail II — report as a partial robustness result, not full robustness.
2. ~~Quantitative demonstration that fuzzy membership recovers usable structure within f_score=1~~
   **DONE (§4.4a, Step 9).** Real differentiation exists (7.02% non-F1 fan-out, near-population
   concept-membership variance), but is attributable to R/M/S, not to F*'s residual spread — report
   precisely, do not credit F* for an effect that belongs to the other dimensions.
3. ~~Concrete interpretable segment profiles for overlapping customers~~ **DONE (§4.4b, Step 10).**
   Systematic top-6 nontrivial-signature selection plus a collapse test: hard_cluster="F1" alone spans
   R=0-694d, M=$9.59-$13,664, and 111 distinct overlap signatures — concrete evidence overlap carries
   real behavioral differentiation a single crisp label discards.
4. ~~Additional fuzzy-validity diagnostics~~ **DONE (§4.4c, Step 11).** Membership entropy (R/M
   genuinely fuzzy, F/S nearly crisp), α-sensitivity (smooth, no knife-edge), L-threshold sensitivity
   (smooth but not invariant), concept persistence (100% survive 10×80% resampling — strong result).
5. Consistent terminology pass across the eventual paper draft (purchase-intensity index, not
   frequency; L-fuzzy-scaled closed-itemset enumeration, not "FP-growth = fuzzy concepts"; stability
   proxy explicitly labeled as approximate; centroid-based piecewise-linear membership, not
   "triangular"; F* weights as "entropy-maximizing within the grid," not "optimal").
6. ~~Remove the `max_len=6` restriction~~ **DONE.** Closed-concept counts corrected substantially
   downward (Olist 10,283→1,369 raw, 423→153 pruned; Retail II 1,531→1,064 raw, 129→115 pruned). Any
   earlier draft text, slide, or abstract citing the old numbers must be updated.

**Overall assessment:** the architecture (fuzzy FCA, RFMS, Kneedle pruning, cross-domain validation,
symmetric benchmarking) is sound and worth keeping. Three real issues have now been found and fixed
across two review rounds: the degenerate F* objective, the score-denominator documentation mismatch,
and the max_len=6 lattice truncation — the last being the most consequential, since it means every
concept-lattice-size number in the pre-v3 draft was wrong. The mandatory sensitivity analysis is now
done and shows the corrected pruning is reasonably but not perfectly stable. The f_score=1
structure-recovery test (item 2) is also done and gives an honest, partially-favorable result: real
differentiation exists but is attributable to R/M/S, not F*. The overlap-profile test (item 3) is done
and gives concrete, systematically-selected evidence that overlap carries real information a single
crisp label discards. The extra fuzzy-validity diagnostics (item 4) are done: concept persistence is
a strong, unambiguous positive (100% of pruned concepts survive resampling); membership entropy gives
an honest mixed result (fuzziness is real for R/M, nearly absent for F/S). Only item 5 (a terminology
pass across the eventual paper draft) remains open; items 1, 2, 3, 4, and 6 are resolved as of this
revision.

## 5. v4 improvements line (2026-09-26; full record in docs/fuzzy_improvements_retail2.md)

Three methodological additions were designed, tested, and validated on both datasets after v3.
Scripts: `scripts/concept_redundancy.py` (adopted stage), `scripts/fcm.py` (baseline),
`scripts/fuzzy_improvements_retail2.py` / `scripts/fcm_baseline_retail2.py` /
`scripts/olist_rfms_comparison.py` / `scripts/multisplit_validation.py` (experiments).

### 5.1 Concept redundancy suppression (ADOPTED)

Greedy extent-Jaccard pruning: order non-trivial concepts by support (desc), intent size (asc),
stability proxy (desc); keep a concept iff its μ≥0.5-cut extent has Jaccard < 0.8 against every
already-kept extent. Leakage contract: extents from training customers only under any train/test
protocol.

- **Retail II:** 445 → 95 concepts (4.7× compression) with zero predictive loss — spend R2 and
  invoices R2 actually rise (0.3664/0.4952 vs 0.3647/0.4933 full). Near-duplicate pairs (extent
  Jaccard ≥ 0.8) 3.3% → **0%**; mean concepts/customer at μ≥0.5: 41.3 → **4.1**. vs crisp: invoices
  R2 delta **+0.0310** [+0.0130, +0.0494] — the only parsimonious arm clearing the +0.030
  exploratory threshold. Dominates Kneedle pruning as a dimensionality reducer (95 features at
  invoices R2 0.4952 vs Kneedle's 64 at 0.4739).
- **Olist:** 633 → 627. The marketplace lattice is nearly redundancy-free; the stage is harmless
  but not needed there. Cross-split replication: 374–383 concepts, suppression removes ≤6 on every
  split.

**Framing:** this upgrades the fair-comparison dedup finding (445 → 73 unique profiles at zero
loss) into an adopted, interpretable pipeline stage. It also resolves the v3 tension where fuzzy's
gains required a 15× larger feature set than crisp.

### 5.2 Fuzzy C-Means baseline (improvement 4 — FCA structure wins)

All prior benchmarks compared FCA against *hard* K-means/hierarchical, confounding representation
with algorithm. Canonical FCM (m=2, k-means++ seeded) at matched k isolates the two:

- **Retail II (k=30):** FCA-suppressed beats FCM soft significantly on spend R2 (−0.0381, CI
  excludes zero) and invoices R2 (−0.0227); AUC delta −0.0088 spans zero. K-means one-hot loses on
  all three. Decomposition: fuzziness itself adds ~+0.003 AUC (soft vs hardened FCM); the prototype
  geometry costs ~−0.011 vs FCA.
- **Validity/utility inversion:** FCM at natural k=5 has the cleanest partition geometry (FPC 0.740,
  silhouette 0.524) but the worst holdout performance; FCA-suppressed has the worst hardened
  geometry (silhouette −0.337) and the best prediction. Consistent with the v3 framing that FCA and
  distance-based clustering encode different structural objectives.
- **Olist:** replicates at matched k (FCM ≈ raw, far below fuzzy-FCA).

**Claim enabled:** the value of the proposed method is the *lattice structure*, not fuzziness per se
— previously untestable because no fuzzy baseline existed.

### 5.3 Olist RFMS cross-domain check + stored-band label leak

`scripts/olist_rfms_comparison.py` applies the adopted stage and the FCM baseline to the full
4-dimension RFMS context, with a validation gate: bitset mining with the **stored Step-1 bands**
reproduces the 1,369-concept lattice exactly (PASS). Re-deriving bands from raw values does not
(float-tie shift: 1 F row, 776 M rows; lattice 1,369 → 1,374) — stored bands are the source of truth
for full-population structure.

**Stored-band label leak (found during 10-split validation, fixed before adoption):** the temporal
obs cohort includes customers whose later order falls in the holdout window; stored bands are
computed over ALL orders, so stored f_score≥2 practically announces the label (P(label=1 | f_score≥2)
= 0.52 vs 0.00 for f_score=1) — a leaked fixed-split crisp arm scored AUC 0.9996. Re-deriving bands
from pre-cutoff features only removes it. **Rule adopted: stored bands are used for full-population
structure only; holdout features are always re-derived from pre-cutoff data.**

### 5.4 Multi-split validation (10 random customer splits per dataset)

All fixed-split results were conditional on one 70/30 seed. Re-running both holdouts over 10 seeds
(seeds 1000–1009), customer-level splitting within the same observation windows:

- **Retail II:** fuzzy-suppressed beats crisp on AUC in **10/10 splits** (mean delta +0.0089,
  paired t p=2.3e-04) and raw RFM 10/10 (p=1.0e-04); spend R2 (+0.0161) and invoices R2 (+0.0332)
  advantages replicate on every split.
- **Olist (leak-fixed):** beats re-derived crisp **10/10** (mean delta +0.128 AUC, p=4.1e-08), raw
  RFMS 10/10 (+0.083), FCM 10/10 (+0.064). Absolute levels are modest (AUC ~0.64) because Year-2
  repurchase is a 2.6% rare event; the ordering fuzzy-FCA > FCM ≥ raw > crisp is split-robust.
- **Cross-domain contrast:** fuzzy FCA's advantage over raw features is ~10× larger on the sparse
  marketplace (+0.11 AUC) than on repeat-heavy Retail II (+0.01 AUC) — the method earns its keep
  exactly where classical RFM breaks down.

### 5.5 Updated outstanding work

1. ~~Terminology pass~~ — now also covering v4 additions (suppression, FCM, label leak).
2. ~~Olist crisp-arm diagnosis~~ **DONE (2026-09-26; appendix in docs/fuzzy_improvements_retail2.md).**
   Mechanism profiled on split seed 1000: the obs cohort's F dimension is degenerate (constant
   F*=0.20 → universal F5 band, support 1.000), so all 44 crisp concepts at the support-0.04
   cutoff are F5-embedded conjunctions and no pure-R concept survives; the predictive R gradient
   is sub-band (label rate by raw-R decile 3.5% → 1.3–1.9% vs weak/non-monotone 1.9–3.0% by
   crisp band) and 239/382 fuzzy features correlate with raw R vs 18/44 crisp. Decisive ablation
   (test AUC): fuzzy concepts 0.6292 > raw R 0.5643 ≈ crisp concepts 0.5662 > raw RFMS 0.5529.
   Framed as a finding: fuzzy FCA interpolates across band boundaries, preserving sub-band
   signal that crisp banding discards — the collapse is a domain-representation property, not
   a pipeline bug.
3. Paper draft: v3 structure plus §5 material; all quantitative claims are now cross-split
   validated except where explicitly noted (single-split artifacts remain in
   results/fuzzy_improvements_retail2/ and results/fcm_baseline_retail2/).

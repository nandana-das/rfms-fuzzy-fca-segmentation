# Adopted Improvement: Concept Redundancy Suppression (Online Retail II)

**Status: ADOPTED (improvement 1).** Experiment script: `scripts/fuzzy_improvements_retail2.py`
(single fixed split, 2026-09-26 run; since simplified to cover only the adopted experiment).
Reusable implementation: `scripts/concept_redundancy.py`. FCM baseline script:
`scripts/fcm_baseline_retail2.py` (+ `scripts/fcm.py`). Artifacts:
`results/fuzzy_improvements_retail2/`, `results/fcm_baseline_retail2/`.

**Repository state (post-cleanup 2026-09-26):** the kept code is the suppression module, the
FCM baseline, and the suppression experiment script. The not-adopted mechanisms below
(continuous memberships, adaptive alpha-level mining) are no longer carried in code — their
numbers and rationale are documented in this file, and their one-time artifacts
(`threshold_robustness.csv`, `adaptive_level_diagnostics.csv`, `percentile_centroids.csv`)
remain under `results/fuzzy_improvements_retail2/` as evidence.

## Decision

The greedy extent-Jaccard redundancy suppression pass is adopted as a standard stage of the
Retail II fuzzy pipeline, applied after support filtering and before (or in place of) Kneedle
pruning when concepts are used as downstream features. Two other candidate improvements were
tested on the same fixed holdout and are **not adopted**:

1. **Percentile-anchored continuous memberships** (P10/P30/P50/P70/P90 centroids replacing
   band medians) — neutral: AUC 0.7901 vs 0.7915 baseline; invoices R2 slightly worse
   (0.4807 vs 0.4933). Kept in the experiment script for the record.
2. **Threshold-free adaptive alpha-level mining** (70 support-anchored levels replacing the
   fixed {0.3, 0.5, 0.7} set) — best invoices-R2 arms on this split (adaptive full:
   0.5002; adaptive+suppression: 0.4986, both clearing the +0.030 exploratory threshold vs
   crisp) but at 6x the feature count of suppression alone for ~+0.003 invoices R2, and its
   principal value turned out to be a structural finding rather than a predictive one (see
   below). Not adopted as a default; revisit if feature budget is not a constraint.

## Evidence for adoption (fixed 70/30 temporal holdout, cutoff 2010-12-09, seed 42)

Test set N=1,294; models identical across arms (LogisticRegressionCV / RidgeCV); B=1,000
paired customer bootstrap, conditional on the fixed split and fitted models.

| Arm | Features | Test AUC | Spend R2 | Invoices R2 |
|---|---:|---:|---:|---:|
| Crisp RFM-FCA (control) | 30 | 0.7753 | 0.3476 | 0.4642 |
| Fuzzy baseline | 445 | 0.7915 | 0.3647 | 0.4933 |
| **Fuzzy + redundancy suppression** | **95** | 0.7875 | **0.3664** | **0.4952** |
| Fuzzy continuous | 378 | 0.7901 | 0.3624 | 0.4807 |
| Fuzzy continuous + suppression | 97 | 0.7900 | 0.3648 | 0.4813 |
| Fuzzy adaptive alpha (full) | 2,398 | 0.7911 | 0.3698 | 0.5002 |
| Fuzzy adaptive + suppression | 580 | 0.7888 | 0.3644 | 0.4986 |

Suppression vs crisp deltas (95% paired bootstrap CIs): invoices R2 **+0.0310**
[+0.0130, +0.0494] — the only parsimonious arm to clear the +0.030 exploratory decision
threshold; spend R2 +0.0189 [+0.0060, +0.0318]; AUC +0.0122 [+0.0034, +0.0203]. All CIs
exclude zero.

Structural effect (train set): 445 -> 95 concepts (4.7x compression); near-duplicate concept
pairs (extent Jaccard >= 0.8) 3.3% -> **0%**; mean pairwise extent Jaccard 0.069 -> **0.006**;
mean concepts-per-customer at mu >= 0.5: 41.3 -> **4.1** (crisp control: 5.2).

Interpretation: the fuzzy representation's predictive value survives a 4.7x redundancy
reduction, confirming the fair-comparison dedup finding (445 -> 73 unique profiles at zero
loss) at the concept level. Suppression also dominates Kneedle pruning as a dimensionality
reducer on this split (95 features at invoices R2 0.4952 vs 64 features at 0.4739 under
Kneedle on the continuous lattice; the Kneedle-pruned baseline set was not competitive).

## Cross-domain check (2026-09-26): Olist RFMS with the adopted improvements

`scripts/olist_rfms_comparison.py` applies the adopted suppression stage and the FCM
baseline to the full 4-dimension RFMS context (20 attributes) on Olist. **Validation gate
PASSED**: with the stored Step-1 score bands, the bitset miner reproduces the stored
Step-2 lattice exactly (1,369 closed concepts at support 0.02). Re-deriving bands from raw
values does NOT reproduce them (float-tie shift: 1 F row, 776 M rows move across band
boundaries; lattice 1,369 -> 1,374) - stored bands are the source of truth.

Full population (93,357): crisp support>0.04 = 76 concepts; fuzzy = 633, reduced to only
**627 after suppression** (J>=0.8). Contrast with Retail II (445 -> 95, 4.7x compression):
**Olist's RFMS lattice is nearly redundancy-free** - consistent with its sparser,
lower-support concept structure (mining support cutoff here is 0.04 of 93k, a much tighter
filter than Retail II's 0.04 of 4.3k).

Temporal holdout (cutoff 2017-08-31; obs cohort 21,664 first-or-early customers, 2.56%
Year-2 repurchase rate): fuzzy+suppression (409 features) **dominates every baseline** -
vs raw RFMS: AUC +0.111, spend R2 +0.088, invoices R2 +0.090; vs crisp RFMS-FCA: AUC
+0.135, spend R2 +0.108; vs FCM at matched k=76: AUC +0.111, spend R2 +0.089 (all 95%
paired bootstrap CIs exclude zero). Absolute levels are modest (AUC 0.667) because Year-2
repurchase is a 2.6% rare event in a growth-phase marketplace - but the **ordering
replicates Retail II exactly: FCA structure > FCM fuzziness > raw features**, and the FCA
advantage is *much larger* on the marketplace domain than on Retail II (AUC +0.111 vs
+0.012 over raw).

Caveats: single fixed split; observation features use pre-cutoff orders only, with
F=0.20 (single-order F*) and stored customer-level M/S (S may include post-cutoff reviews
for ~2% of the obs cohort - flagged in run_parameters.csv); suppression removed only 6
concepts here, so Part B's fuzzy arm is effectively the unpruned lattice.

## Improvement 4 (tested 2026-09-26): Fuzzy C-Means baseline — FCA structure wins

**Question:** does fuzzy FCA's lattice structure add value over generic fuzzy clustering? All
prior benchmarks compared FCA against *hard* K-means/hierarchical, confounding representation
with algorithm. Canonical FCM (Bezdek, m=2, k-means++ seeded, `scripts/fcm.py`; no FCM package
in the pinned environment) was slotted into the identical holdout protocol at matched k=30
(crisp concept count) and natural k=5 (WSS knee).

| Arm (test N=1,294) | Features | AUC | Spend R2 | Invoices R2 |
|---|---:|---:|---:|---:|
| FCA-suppressed (adopted) | 95 | **0.7875** | **0.3664** | **0.4952** |
| FCM soft (k=30) | 30 | 0.7787 | 0.3283 | 0.4725 |
| FCM hardened (k=30) | 30 | 0.7758 | 0.3384 | 0.4879 |
| K-means one-hot (k=30) | 30 | 0.7642 | 0.3128 | 0.4519 |
| FCM soft (k=5, natural) | 5 | 0.7536 | 0.2812 | 0.4152 |

Paired bootstrap vs FCA-suppressed: FCM-soft loses significantly on spend R2
(−0.0381 [−0.0581, −0.0187]) and invoices R2 (−0.0227 [−0.0379, −0.0072]); the AUC delta
(−0.0088) spans zero. K-means one-hot loses significantly on all three. **Verdict: the FCA
lattice structure — not fuzziness per se — carries the predictive value.**

Fuzziness decomposition (matched k=30): FCM soft 0.7787 vs FCM hardened 0.7758 vs K-means
0.7642 — fuzziness itself adds ~+0.003 AUC over hardening the same prototypes, while the
prototype geometry costs ~−0.011 vs FCA.

Validity indices expose the recurring metric/utility mismatch: FCM at natural k=5 has the
cleanest partition geometry (FPC 0.740, silhouette 0.524, DB 0.651) but the *worst* holdout
performance (AUC 0.7536); FCA-suppressed has the worst hardened geometry (silhouette −0.337,
DB 8.29, NaN FPC/PE — concept memberships are not a partition and must not be scored with
FPC/PE) yet the best downstream prediction. Consistent with the project's standing framing
that FCA and distance-based clustering encode different structural objectives.

Limitations: single fixed split (as everywhere); FCM matched on k=30 (crisp count), not on
FCA's 95 features — an FCM k=95 arm would equalize feature budget and is the natural
follow-up. Artifacts: `results/fcm_baseline_retail2/`.

## Method definition (as adopted)

Order non-trivial concepts by support (desc), intent size (asc), stability proxy (desc),
original row order (final deterministic tie-break). Sweep in that order, keeping a concept
iff its mu >= 0.5-cut extent has Jaccard < 0.8 against every already-kept extent. Leakage
contract: extents must be computed from training customers only under any train/test
protocol (the suppressor never sees outcomes, but the extent cut must remain fit-free with
respect to test data).

## Secondary finding exposed by the (not adopted) adaptive-mining arm

Mining only at the fixed levels {0.3, 0.5, 0.7} simultaneously, rather than level by level,
means most retained intents mix attributes realized at different levels. On the Retail II
train lattice, only 29.2% of the 445 fixed-set concepts appear as single-level closed
concepts when 70 support-anchored levels are enumerated; the remaining ~71% exist only as
multi-level mixing products of that specific threshold set. Combined with the known
threshold-set sensitivity of lattice size (445 vs 759 vs 495 concepts for
{0.3,0.5,0.7} / {0.2,0.4,0.6} / {0.25,0.5,0.75} on identical memberships), this strengthens
the case for reporting threshold-set sensitivity wherever L-fuzzy scaling is used. Caveat:
completeness of the support-anchored level grid was verified empirically (baseline
containment check), not proven exactly.

## Limitations

- Single fixed 70/30 split; bootstrap CIs are conditional on that split and the fitted
  models. Cross-split repetition is the obvious next hardening step.
- J_MAX = 0.8 and the mu >= 0.5 extent cut are exploratory choices, recorded in
  `run_parameters.csv`; no sensitivity sweep over J_MAX has been run yet.
- Suppression quality order is unsupervised (support/intent size/stability); a
  downstream-utility-ordered variant (e.g., stability selection on train) is untested.

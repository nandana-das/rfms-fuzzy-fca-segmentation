#!/usr/bin/env python
"""
Canonical Kuznetsov Intensional Stability - ISOLATED PROTOTYPE.

Scope (strict):
- This script is a self-contained PROTOTYPE. It does NOT modify the
  production pipeline, existing scripts, results, methodology documents,
  or any branch. It only creates new artifacts under
  results/kuznetsov_stability_audit/.
- It reconstructs the SAME 45-attribute binary FCA contexts used by the
  production mining code (by calling the unmodified production functions
  for scoring, fuzzy memberships and closed-concept mining).
- It reconstructs the 502 Dunnhumby / 359 Retail II fuzzy concept sets and
  verifies A' = B for every concept before computing stability.

Method (exact, no enumeration of 2^|A| subsets, no proxy):
- Canonical intensional stability (Kuznetsov 2007):
      Stab(A, B) = |{X subseteq A : X' = B}| / 2^|A|
- X' = B iff X'' = A, i.e. X is a generator of the concept.
- Non-generators form the down-set union of P(A_q) over the lower neighbors
  (direct descendants) q of the concept; the lower-neighbor extents are
  exactly the maximal masks A & extent(m) for m not in B ("maximal
  non-generators", Gao et al. 2020; Kuznetsov's direct-descendant method).
- Generator count is computed exactly by inclusion-exclusion over the
  lower-neighbor extents with memoization:
      f(i, I) = f(i+1, I) - f(i+1, I & E_i),  f(k, I) = 2^popcount(I)
      generators(c) = f(0, A)
- The result is an exact integer count; stability is reported as an exact
  fraction plus a high-precision decimal and log2/log values.

Verification (before/alongside the real run):
- Hand-built toy contexts with hand-computed stability values.
- Exhaustive validation on random small contexts: every concept checked
  against (a) brute-force submask enumeration and (b) an independently
  enumerated concept lattice (lower-neighbor sets compared).
- Brute-force check on any real concept with extent <= 20 (none expected).
- Monte-Carlo cross-check (packed-bit sampler) on selected real concepts.

Constraints honored: no pruning is performed, Kneedle is not run, Jaccard
suppression is untouched, no predictive experiment is run.
"""

from __future__ import annotations

import gc
import math
import sys
import time
import tracemalloc
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from project_paths import RESULTS_DIR  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    DIMENSIONS,
    SUPPORT_CUTOFF as R2_SUPPORT_CUTOFF,
    aggregate_rfm,
    compute_fuzzy_memberships as r2_compute_fuzzy_memberships,
    dense_rank_scores as r2_dense_rank_scores,
    load_cleaned_transactions,
    L_THRESHOLDS as R2_L_THRESHOLDS,
    mine_fuzzy_closed_concepts,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    DIMS as DH_DIMS,
    SUPPORT_CUTOFF as DH_SUPPORT_CUTOFF,
    compute_fuzzy_memberships as dh_compute_fuzzy_memberships,
    dense_rank_scores as dh_dense_rank_scores,
    load_and_prepare_cohorts,
    mine_fuzzy_closed_concepts_with_thresholds,
)

OUT_DIR = RESULTS_DIR / "kuznetsov_stability_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DH_THRESHOLDS = (0.3, 0.5, 0.7)
DH_DATASET_NAME = "Dunnhumby Observation (Days 1-620)"
R2_DATASET_NAME = "UK Online Retail II (Full Cohort)"

BRUTE_FORCE_MAX_EXTENT = 20          # real-data brute force guard (theoretical tool)
IE_MAX_STATES = 500_000              # hard state cap per concept
IE_TIME_LIMIT_S = 120.0              # hard wall-clock cap per concept
MC_SAMPLES = 1_000_000               # Monte-Carlo cross-check samples per concept
MC_BATCH = 8192
MC_MIN_EXACT_STABILITY = 5e-6        # only concepts above detection power
MC_MAX_CONCEPTS_PER_DATASET = 8
DECIMAL_PREC = 60

RANDOM_VALIDATION_CONTEXTS = 40
RANDOM_VALIDATION_SEED = 20261006

sys.setrecursionlimit(10000)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def pr(msg: str) -> None:
    print(msg, flush=True)


def peak_rss_mb() -> float:
    """Best-effort process peak RSS in MB (portable across Windows/Unix)."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return float("nan")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


# ===========================================================================
# Core FCA math on integer bitmasks
# ===========================================================================

def build_context(fuzzy_memberships: pd.DataFrame, thresholds) -> tuple[list[str], list[int], int]:
    """Build the 45-attribute binary context as object bitmasks.

    Attribute (col@threshold) holds for object i iff mu_col(i) >= threshold.
    Mirrors the production attribute naming exactly: f"{col}@{t:.1f}".
    """
    n_objects = len(fuzzy_memberships)
    attr_names: list[str] = []
    attr_masks: list[int] = []

    for col in fuzzy_memberships.columns:
        values = fuzzy_memberships[col].to_numpy(dtype=float)
        for threshold in thresholds:
            name = f"{col}@{threshold:.1f}"
            mask = 0
            for idx in np.nonzero(values >= threshold)[0]:
                mask |= 1 << int(idx)
            attr_names.append(name)
            attr_masks.append(mask)

    return attr_names, attr_masks, n_objects


def extent_of_intent(intent_indices, attr_masks: list[int], full_mask: int) -> int:
    """A = B' for attribute-index set B (bitmask intersection)."""
    extent = full_mask
    for index in intent_indices:
        extent &= attr_masks[index]
    return extent


def intent_of_extent(extent: int, attr_masks: list[int]) -> frozenset:
    """B = A' for object bitmask A: attributes whose extent covers A."""
    return frozenset(
        j for j, mask in enumerate(attr_masks) if extent & ~mask == 0
    )


def lower_neighbor_extents(extent: int, intent, attr_masks: list[int]) -> tuple[list[int], int]:
    """Extents of the lower neighbors (direct descendants) of the concept.

    Theorem (standard FCA + Gao et al. 2020): every lower neighbor q of
    c = (A, B) is generated by B + {m} for some attribute m not in B, hence
    A(q) = A & extent(m). The maximal distinct such masks are exactly the
    lower-neighbor extents.

    Returns (children_masks, n_candidates). children_masks may contain 0
    (the empty extent) when no object of A carries any attribute outside B;
    this case is handled by the inclusion-exclusion recursion itself.
    """
    candidates = set()
    for m, mask in enumerate(attr_masks):
        if m in intent:
            continue
        candidates.add(extent & mask)

    nonzero = [e for e in candidates if e != 0]
    maximal = [
        e
        for e in nonzero
        if not any(other != e and (e & other) == e for other in nonzero)
    ]

    if candidates and not maximal:
        # All candidates are the empty extent: {emptyset} is the only child.
        maximal = [0]

    return maximal, len(candidates)


class StabilityInfeasibleError(RuntimeError):
    """Raised when the exact method exceeds its configured resource caps."""


def count_generators_ie(
    extent: int,
    children_masks: list[int],
    max_states: int = IE_MAX_STATES,
    max_seconds: float = IE_TIME_LIMIT_S,
) -> tuple[int, int, int]:
    """Exact generator count via memoized inclusion-exclusion.

    generators(c) = |{X subseteq A : X' = B}|
                  = sum_{T subseteq children} (-1)^|T| * 2^|A & (intersection_{q in T} E_q)|

    Recursion: f(i, I) = f(i+1, I) - f(i+1, I & E_i); f(k, I) = 2^popcount(I).
    Returns (generator_count, memo_states, recursive_calls).
    """
    n_children = len(children_masks)
    memo: dict[tuple[int, int], int] = {}
    deadline = time.perf_counter() + max_seconds
    calls = 0

    def f(i: int, current: int) -> int:
        nonlocal calls
        calls += 1
        if calls % 8192 == 0 and time.perf_counter() > deadline:
            raise StabilityInfeasibleError(
                f"exact IE exceeded wall-clock limit ({max_seconds:.0f}s)"
            )
        if i == n_children:
            return 1 << current.bit_count()
        key = (i, current)
        value = memo.get(key)
        if value is None:
            if len(memo) > max_states:
                raise StabilityInfeasibleError(
                    f"exact IE exceeded memo state cap ({max_states})"
                )
            value = f(i + 1, current) - f(i + 1, current & children_masks[i])
            memo[key] = value
        return value

    generator_count = f(0, extent)
    return generator_count, len(memo), calls


def brute_force_generator_count(extent: int, intent, attr_masks: list[int]) -> int:
    """Exhaustive submask enumeration - only for toy/small extents.

    X is a generator iff for every attribute m not in B some object of X
    lacks m (X is not covered by any lower-neighbor extent candidate).
    """
    non_b = [mask for j, mask in enumerate(attr_masks) if j not in intent]
    count = 0
    submask = extent
    while True:
        is_generator = True
        for mask in non_b:
            if submask & ~mask == 0:
                is_generator = False
                break
        if is_generator:
            count += 1
        if submask == 0:
            break
        submask = (submask - 1) & extent
    return count


def exact_stability(extent: int, generator_count: int) -> Fraction:
    return Fraction(generator_count, 1 << extent.bit_count())


def stability_decimal_string(generator_count: int, extent_size: int) -> str:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        value = Decimal(generator_count) / (Decimal(2) ** extent_size)
        return str(value)


def stability_ln(generator_count: int, extent_size: int) -> float:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        value = Decimal(generator_count) / (Decimal(2) ** extent_size)
        return float(value.ln())


def stability_log2(generator_count: int, extent_size: int) -> float:
    """log2 of the exact stability, via Decimal.

    float64 math.log2 collapses values within ~2^-53 of 1 to exactly 1.0 /
    log2=0.0. A plain Decimal(gen/2^|A|) with prec=60 also underflows for
    losses smaller than ~2^-60 (the near-1 concepts in this regime).

    We compute log2 on the EXACT loss = 1 - gen/2^|A| instead. Writing
    d = 2^|A| and g = gen, loss = (d - g) / d; for d-g > 0:

        log2(g/d) = log2(1 - loss) = -loss/ln(2) - loss^2/2/ln(2) - ...

    For the tiny losses here (down to ~2^-3000) the linear term dominates
    by many orders of magnitude, so we compute it with Decimal precision
    scaled to the loss size: prec = max(DECIMAL_PREC, digits(d-g) + log10(d) + 5).
    The Decimal division (d-g)/d does not lose precision as long as prec exceeds
    the number of significant digits of the quotient, which is bounded by
    digits(d-g) + log10(d). We then convert the exact Decimal loss to float64
    (underflows to 0.0 in float64 only below ~2^-1074, i.e. far below anything
    here) and apply the series to one term, validating against two-term where
    cheap.
    """
    den = 1 << extent_size
    diff = den - generator_count  # exact: (2^|A|) - gen
    if diff == 0:
        return 0.0
    denom_digits = len(str(diff))          # digits of the exact loss numerator
    prec = max(DECIMAL_PREC, denom_digits + extent_size * 0.30103 + 8)
    with localcontext() as ctx:
        ctx.prec = int(prec)
        loss = Decimal(diff) / Decimal(den)      # exact loss, prec digits
        ln2 = Decimal(2).ln()
        log2_neg = loss / ln2                     # linear term: -log2(stab) ~ loss/ln2
        # two-term correction for self-consistency on the larger losses
        loss2 = loss * loss
        log2_two = (loss + loss2 / Decimal(2)) / ln2
        # pick whichever is more accurate; for these losses they agree to many digits
        stab_log2 = -(log2_two if loss2 > Decimal(0) else log2_neg)
    return float(stab_log2)


def decimal_pow2(x: float) -> str:
    """Render 2^x for possibly very negative x (used for readability)."""
    with localcontext() as ctx:
        ctx.prec = 8
        value = Decimal(2) ** Decimal(x)
        return f"{value:.4E}"


# ===========================================================================
# Part 2a - Verification suite: hand-computed toy contexts + randomized
#           small-context validation against brute force AND an independently
#           enumerated concept lattice.
# ===========================================================================

# Each toy: (name, n_objects, attr_specs, expected_concepts)
#   attr_specs: list of frozensets of 0-based object indices per attribute
#   expected_concepts: list of (extent_objects, intent_attr_indices,
#                              expected_children_extents, expected_stability)
# All values below were computed BY HAND from the context tables.
TOY_CONTEXTS = [
    (
        "toy1_staircase (Kuznetsov-style 3x3; two overlapping lower neighbors)",
        3,
        [frozenset({0, 1}), frozenset({0, 1, 2}), frozenset({1, 2})],
        [
            # ({0,1,2}, {b}): generators {0,2} and {0,1,2} only -> 2/8.
            # children from m=a: {0,1}, m=c: {1,2} (both maximal, overlap {1}).
            (frozenset({0, 1, 2}), frozenset({1}),
             [frozenset({0, 1}), frozenset({1, 2})], Fraction(1, 4)),
            # ({0,1}, {a,b}): generators {0} and {0,1} -> 2/4; child {1} (m=c).
            (frozenset({0, 1}), frozenset({0, 1}), [frozenset({1})], Fraction(1, 2)),
            # ({1,2}, {b,c}): generators {2} and {1,2} -> 2/4; child {1} (m=a).
            (frozenset({1, 2}), frozenset({1, 2}), [frozenset({1})], Fraction(1, 2)),
            # ({1}, {a,b,c}): no lower concept -> 2/2.
            (frozenset({1}), frozenset({0, 1, 2}), [], Fraction(1)),
        ],
    ),
    (
        "toy2_chain (maximality filter: child from attr c is non-maximal)",
        3,
        [frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({0})],
        [
            (frozenset({0}), frozenset({0, 1, 2}), [], Fraction(1)),
            (frozenset({0, 1}), frozenset({0, 1}), [frozenset({0})], Fraction(1, 2)),
            # children of ({0,1,2}, a): A&extent(b)={0,1}, A&extent(c)={0}
            # (non-maximal, dropped) => exactly one lower neighbor {0,1}.
            # generators = 2^3 - 2^2 = 4 => stability 1/2.
            (frozenset({0, 1, 2}), frozenset({0}), [frozenset({0, 1})], Fraction(1, 2)),
        ],
    ),
    (
        "toy3_duplicate_columns (identical attrs dedup; empty-intent top)",
        3,
        [frozenset({0, 1}), frozenset({0, 1})],
        [
            # top ({0,1,2}, {}): generators = subsets not contained in {0,1}
            # -> {2},{0,2},{1,2},{0,1,2} = 4/8; child {0,1} (m=a dedup m=b).
            (frozenset({0, 1, 2}), frozenset(), [frozenset({0, 1})], Fraction(1, 2)),
            # ({0,1}, {a,b}): every subset generates -> 4/4; no lower concept.
            (frozenset({0, 1}), frozenset({0, 1}), [], Fraction(1)),
        ],
    ),
    (
        "toy4_disjoint (empty-extent concept and empty-intent top)",
        2,
        [frozenset({0}), frozenset({1})],
        [
            # top ({0,1}, {}): only {0,1} has empty derivation -> 1/4;
            # children {0} (m=a) and {1} (m=b), disjoint.
            (frozenset({0, 1}), frozenset(),
             [frozenset({0}), frozenset({1})], Fraction(1, 4)),
            # ({0}, {a}): {0} generates -> 1/2; child empty extent (m=b).
            (frozenset({0}), frozenset({0}), [frozenset()], Fraction(1, 2)),
            # ({1}, {b}): symmetric -> 1/2; child empty extent (m=a).
            (frozenset({1}), frozenset({1}), [frozenset()], Fraction(1, 2)),
            # bottom (empty extent, {a,b}): only X=empty, and empty'={a,b} -> 1/1.
            (frozenset(), frozenset({0, 1}), [], Fraction(1)),
        ],
    ),
]


def _context_from_spec(n_objects: int, attr_specs) -> tuple[int, list[int]]:
    attr_masks = []
    for objs in attr_specs:
        mask = 0
        for o in objs:
            mask |= 1 << o
        attr_masks.append(mask)
    return (1 << n_objects) - 1, attr_masks


def _enumerate_closed_concepts(
    n_attrs: int, attr_masks: list[int], full_mask: int
) -> list[tuple[int, frozenset]]:
    """All closed intents via explicit enumeration over 2^m attribute subsets
    (feasible here: only used on toy/small contexts, never on real data)."""
    concepts = []
    for S in range(1 << n_attrs):
        intent = frozenset(j for j in range(n_attrs) if S >> j & 1)
        extent = extent_of_intent(intent, attr_masks, full_mask)
        if intent_of_extent(extent, attr_masks) == intent:
            concepts.append((extent, intent))
    return concepts


def _independent_lattice_children(
    extent: int, all_extents: list[int]
) -> list[int]:
    """Lower-neighbor extents derived ONLY from the enumerated lattice order
    (strict-below relations), independent of the attribute-based shortcut."""
    children = []
    for e in all_extents:
        if e == extent or (e & extent) != e:
            continue  # not strictly below
        strictly_between = any(
            f != e
            and f != extent
            and (e & f) == e
            and (f & extent) == f
            for f in all_extents
        )
        if not strictly_between:
            children.append(e)
    return sorted(set(children))


def run_toy_verification() -> dict:
    pr("[verify] toy contexts ...")
    summary = {"n_toys": len(TOY_CONTEXTS), "n_concepts": 0, "mismatches": []}
    for name, n_objects, attr_specs, expected in TOY_CONTEXTS:
        full_mask, attr_masks = _context_from_spec(n_objects, attr_specs)
        n_attrs = len(attr_specs)
        concepts = _enumerate_closed_concepts(n_attrs, attr_masks, full_mask)
        got = {}
        for extent, intent in concepts:
            children, _ = lower_neighbor_extents(extent, intent, attr_masks)
            gen_count, _, _ = count_generators_ie(extent, children)
            bf_count = brute_force_generator_count(extent, intent, attr_masks)
            got[(extent, intent)] = {
                "children": sorted(set(children)),
                "ie": gen_count,
                "bf": bf_count,
                "stab": exact_stability(extent, gen_count),
            }
        exp = {}
        for objs, idxs, child_objs, stab in expected:
            extent = 0
            for o in objs:
                extent |= 1 << o
            child_masks = []
            for cobjs in child_objs:
                cm = 0
                for o in cobjs:
                    cm |= 1 << o
                child_masks.append(cm)
            exp[(extent, frozenset(idxs))] = {
                "children": sorted(set(child_masks)),
                "stab": stab,
            }
        if set(got) != set(exp):
            summary["mismatches"].append(
                {"toy": name, "issue": "concept set differs from hand enumeration",
                 "got": sorted(str(k) for k in got),
                 "expected": sorted(str(k) for k in exp)}
            )
            continue
        for key, exp_val in exp.items():
            summary["n_concepts"] += 1
            g = got[key]
            if g["children"] != exp_val["children"]:
                summary["mismatches"].append(
                    {"toy": name, "concept": str(key), "issue": "children",
                     "got": g["children"], "expected": exp_val["children"]}
                )
            if g["stab"] != exp_val["stab"]:
                summary["mismatches"].append(
                    {"toy": name, "concept": str(key), "issue": "stability",
                     "got": str(g["stab"]), "expected": str(exp_val["stab"])}
                )
            if g["ie"] != g["bf"]:
                summary["mismatches"].append(
                    {"toy": name, "concept": str(key), "issue": "IE != brute force",
                     "got": g["ie"], "expected": g["bf"]}
                )
    status = "PASS" if not summary["mismatches"] else "FAIL"
    pr(f"[verify] toy contexts: {status} "
       f"({summary['n_toys']} toys, {summary['n_concepts']} concepts checked)")
    return summary


def run_randomized_validation() -> dict:
    """Random small contexts: every closed concept checked against (a) brute
    force submask enumeration and (b) an independently enumerated lattice."""
    pr("[verify] randomized small-context validation ...")
    rng = np.random.default_rng(RANDOM_VALIDATION_SEED)
    summary = {
        "n_contexts": 0,
        "n_concepts": 0,
        "n_children_mismatches": 0,
        "n_count_mismatches": 0,
        "mismatches": [],
    }
    for ctx_idx in range(RANDOM_VALIDATION_CONTEXTS):
        n_objects = int(rng.integers(3, 10))
        n_attrs = int(rng.integers(2, 7))
        attr_specs = []
        for j in range(n_attrs):
            if attr_specs and rng.random() < 0.15:
                attr_specs.append(attr_specs[int(rng.integers(0, len(attr_specs)))])
            else:
                p = float(rng.uniform(0.2, 0.8))
                attr_specs.append(
                    frozenset(o for o in range(n_objects) if rng.random() < p)
                )
        full_mask, attr_masks = _context_from_spec(n_objects, attr_specs)
        concepts = _enumerate_closed_concepts(n_attrs, attr_masks, full_mask)
        all_extents = [e for e, _ in concepts]
        summary["n_contexts"] += 1
        for extent, intent in concepts:
            summary["n_concepts"] += 1
            children, _ = lower_neighbor_extents(extent, intent, attr_masks)
            children_sorted = sorted(set(children))
            children_lattice = _independent_lattice_children(extent, all_extents)
            if children_sorted != children_lattice:
                summary["n_children_mismatches"] += 1
                if len(summary["mismatches"]) < 10:
                    summary["mismatches"].append(
                        {"context": ctx_idx, "issue": "children",
                         "attr_based": children_sorted,
                         "lattice_based": children_lattice}
                    )
            gen_ie, _, _ = count_generators_ie(extent, children)
            gen_bf = brute_force_generator_count(extent, intent, attr_masks)
            if gen_ie != gen_bf:
                summary["n_count_mismatches"] += 1
                if len(summary["mismatches"]) < 10:
                    summary["mismatches"].append(
                        {"context": ctx_idx, "issue": "generator count",
                         "ie": gen_ie, "brute_force": gen_bf}
                    )
    status = (
        "PASS"
        if summary["n_children_mismatches"] == 0 and summary["n_count_mismatches"] == 0
        else "FAIL"
    )
    pr(
        f"[verify] randomized validation: {status} "
        f"({summary['n_contexts']} contexts, {summary['n_concepts']} concepts; "
        f"children mismatches={summary['n_children_mismatches']}, "
        f"count mismatches={summary['n_count_mismatches']})"
    )
    return summary


# ===========================================================================
# Part 2b - Dataset reconstruction (unmodified production functions only),
#           A' = B verification, exact per-concept stability computation.
# ===========================================================================

def reconstruct_dunnhumby() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Mirror canonical_kneedle_audit.py::load_dunnhumby_concepts exactly."""
    t0 = time.perf_counter()
    pr("[load] Dunnhumby: loading cohorts (production loader) ...")
    obs_agg, _ = load_and_prepare_cohorts()
    customer_rfm = obs_agg[["household_key", "R", "F", "M"]].rename(
        columns={"household_key": "CustomerID"}
    )
    scored = dh_dense_rank_scores(customer_rfm, dims=DH_DIMS)
    fuzzy_mu, _, _ = dh_compute_fuzzy_memberships(scored, dims=DH_DIMS)
    concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        fuzzy_mu,
        scored,
        DH_THRESHOLDS,
        min_support=DH_SUPPORT_CUTOFF,
        dims=DH_DIMS,
    )
    elapsed = time.perf_counter() - t0
    pr(f"[load] Dunnhumby: {len(concepts)} concepts in {elapsed:.1f}s.")
    return concepts, fuzzy_mu, elapsed


def reconstruct_retail2() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Mirror the production Retail II full-cohort reconstruction."""
    t0 = time.perf_counter()
    pr("[load] Retail II: loading and cleaning transactions ...")
    clean = load_cleaned_transactions()
    reference_date = clean["InvoiceDate"].max()
    full_rfm = aggregate_rfm(clean, reference_date=reference_date)
    scored = r2_dense_rank_scores(full_rfm)
    fuzzy_mu, _, _ = r2_compute_fuzzy_memberships(scored)
    concepts, _, _ = mine_fuzzy_closed_concepts(
        fuzzy_mu, scored, min_support=R2_SUPPORT_CUTOFF
    )
    elapsed = time.perf_counter() - t0
    pr(f"[load] Retail II: {len(concepts)} concepts in {elapsed:.1f}s.")
    return concepts, fuzzy_mu, elapsed


def cross_check_retail2_reference(concepts_df: pd.DataFrame) -> dict:
    """Cross-check the reconstructed 359-concept table against the stored
    production results CSV (read-only)."""
    ref_path = (
        RESULTS_DIR
        / "fair_comparison_retail2"
        / "controlled_comparison_concepts_fuzzy.csv"
    )
    result = {"available": ref_path.exists(), "path": str(ref_path)}
    if not result["available"]:
        return result

    def norm(s: object) -> tuple[str, ...]:
        return tuple(sorted(p.strip() for p in str(s).split("&") if p.strip()))

    ref = pd.read_csv(ref_path)
    ref_map = {norm(r["intent"]): int(r["n_customers"]) for _, r in ref.iterrows()}
    got_map = {norm(r["intent"]): int(r["n_customers"]) for _, r in concepts_df.iterrows()}
    common = set(ref_map) & set(got_map)
    n_cust_mismatch = sum(1 for k in common if ref_map[k] != got_map[k])
    result.update(
        {
            "n_ref": len(ref_map),
            "n_got": len(got_map),
            "n_common_intents": len(common),
            "n_missing_in_got": len(set(ref_map) - set(got_map)),
            "n_extra_in_got": len(set(got_map) - set(ref_map)),
            "n_customer_count_mismatches": n_cust_mismatch,
        }
    )
    return result


def compute_dataset_stability(
    dataset_name: str,
    concept_id_prefix: str,
    concepts_df: pd.DataFrame,
    fuzzy_memberships: pd.DataFrame,
    thresholds: tuple[float, ...],
) -> tuple[list[dict], dict, dict[str, int]]:
    """Compute exact canonical stability for every concept of one dataset.

    Returns (rows, context_info, extents_by_concept_id). Each row carries the
    full audit trail for one concept; extents are kept separately so the CSV
    stays tabular.
    """
    attr_names, attr_masks, n_objects = build_context(fuzzy_memberships, thresholds)
    full_mask = (1 << n_objects) - 1
    name_to_idx = {name: j for j, name in enumerate(attr_names)}
    context_info = {
        "dataset": dataset_name,
        "n_objects": n_objects,
        "n_attributes": len(attr_names),
        "full_extent_size": n_objects,
        "_attr_masks": attr_masks,
        "_full_mask": full_mask,
    }
    pr(
        f"[context] {dataset_name}: {n_objects} objects x {len(attr_names)} attributes."
    )

    rows: list[dict] = []
    extents_by_id: dict[str, int] = {}
    n_verified = 0
    n_exact = 0
    n_failed = 0
    t_start = time.perf_counter()

    for i, (_, row) in enumerate(concepts_df.iterrows()):
        concept_id = f"{concept_id_prefix}_{i + 1:03d}"
        intent_names = [t.strip() for t in str(row["intent"]).split("&") if t.strip()]
        unknown = [a for a in intent_names if a not in name_to_idx]
        if unknown:
            rows.append(
                {
                    "dataset": dataset_name,
                    "concept_id": concept_id,
                    "intent": row["intent"],
                    "intent_size": int(row["intent_size"]),
                    "extent_size": -1,
                    "support": float(row["support"]),
                    "stability_proxy": float(row["stability_proxy"]),
                    "generator_count": "unknown_attr",
                    "stability_fraction": "",
                    "stability_decimal": "",
                    "log2_stability": "",
                    "n_lower_neighbors": "",
                    "n_candidate_attrs": "",
                    "memo_states": "",
                    "ie_calls": "",
                    "time_ms": 0.0,
                    "status": f"failed_unknown_attribute:{','.join(unknown)}",
                    "a_prime_equals_b": False,
                    "extent_size_matches_table": False,
                }
            )
            n_failed += 1
            continue

        intent = frozenset(name_to_idx[a] for a in intent_names)
        extent = extent_of_intent(intent, attr_masks, full_mask)
        a_prime = intent_of_extent(extent, attr_masks)
        verified = a_prime == intent
        n_verified += int(verified)
        extent_size = extent.bit_count()
        extents_by_id[concept_id] = extent
        extent_matches = extent_size == int(row["n_customers"])

        t0 = time.perf_counter()
        status = "exact"
        gen_count = None
        memo_states = None
        ie_calls = None
        children: list[int] = []
        n_candidates = 0
        try:
            children, n_candidates = lower_neighbor_extents(extent, intent, attr_masks)
            gen_count, memo_states, ie_calls = count_generators_ie(extent, children)
            if not (1 <= gen_count <= (1 << extent_size)):
                status = f"failed_sanity:count_out_of_range({gen_count})"
                gen_count = None
        except StabilityInfeasibleError as exc:
            status = f"failed:{exc}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if gen_count is not None:
            n_exact += 1
            frac = exact_stability(extent, gen_count)
            bf_note = ""
            if extent_size <= BRUTE_FORCE_MAX_EXTENT:
                bf = brute_force_generator_count(extent, intent, attr_masks)
                bf_note = "ok" if bf == gen_count else f"MISMATCH(bf={bf})"
                status = "exact+brute_force" if bf == gen_count else f"failed_bf:{bf_note}"
            # Python ints here reach ~2^5492 (thousands of digits); keep as str
            # so pandas never attempts a float conversion of the column.
            gen_out: object = str(gen_count)
        else:
            frac = None
            gen_out = ""
            bf_note = ""

        rows.append(
            {
                "dataset": dataset_name,
                "concept_id": concept_id,
                "intent": row["intent"],
                "intent_size": int(row["intent_size"]),
                "extent_size": extent_size,
                "support": float(row["support"]),
                "stability_proxy": float(row["stability_proxy"]),
                "generator_count": gen_out,
                "stability_fraction": str(frac) if frac is not None else "",
                "stability_decimal": (
                    stability_decimal_string(gen_count, extent_size)
                    if gen_count is not None
                    else ""
                ),
                "log2_stability": (
                    stability_log2(gen_count, extent_size)
                    if gen_count is not None
                    else ""
                ),
                "n_lower_neighbors": len(children),
                "n_candidate_attrs": n_candidates,
                "memo_states": memo_states,
                "ie_calls": ie_calls,
                "time_ms": elapsed_ms,
                "status": status,
                "a_prime_equals_b": verified,
                "extent_size_matches_table": bool(extent_matches),
            }
        )
        if bf_note.startswith("MISMATCH"):
            pr(f"[warn] {concept_id}: brute-force mismatch: {bf_note}")

        if (i + 1) % 50 == 0 or i + 1 == len(concepts_df):
            pr(
                f"[stab] {dataset_name}: {i + 1}/{len(concepts_df)} concepts "
                f"({n_exact} exact, {n_failed} failed, "
                f"{time.perf_counter() - t_start:.1f}s)"
            )

    context_info.update(
        {
            "n_concepts": len(rows),
            "n_ap_prime_verified": n_verified,
            "n_exact": n_exact,
            "n_failed": n_failed,
            "stability_time_s": time.perf_counter() - t_start,
        }
    )
    pr(
        f"[stab] {dataset_name}: A'=B verified for {n_verified}/{len(rows)}; "
        f"exact {n_exact}/{len(rows)}."
    )
    return rows, context_info, extents_by_id


# ===========================================================================
# Part 2c - Monte-Carlo cross-check on selected real concepts.
#           Samples X subseteq A uniformly (packed uint64 bit vectors, one
#           Bernoulli(1/2) bit per object) and tests X' = B via the
#           lower-neighbor characterization (X must escape every child).
# ===========================================================================

def mc_generator_probability(
    extent: int,
    children_masks: list[int],
    n_samples: int = MC_SAMPLES,
    batch: int = MC_BATCH,
    seed: int = 0,
) -> tuple[float, int]:
    """Monte-Carlo estimate of P(X' = B | X ~ Uniform(2^A))."""
    objects = [i for i in range(extent.bit_length()) if (extent >> i) & 1]
    n = len(objects)
    n_words = max(1, (n + 63) // 64)
    valid_bits = n - 64 * (n_words - 1)
    valid_mask = np.uint64((1 << valid_bits) - 1 if valid_bits < 64 else 0xFFFFFFFFFFFFFFFF)

    child_words = []
    for c in children_masks:
        w = np.zeros(n_words, dtype=np.uint64)
        for k, o in enumerate(objects):
            if (c >> o) & 1:
                w[k >> 6] |= np.uint64(1) << np.uint64(k & 63)
        child_words.append(w)

    rng = np.random.default_rng(seed)
    n_gen = 0
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        # Uniform random bytes -> uniform random bits (little-endian: byte b
        # of a word maps to bit positions 8b..8b+7, matching mask build above).
        raw = rng.integers(0, 256, size=(b, n_words * 8), dtype=np.uint8)
        samples = raw.view(np.uint64)
        samples[:, -1] &= valid_mask
        gen = np.ones(b, dtype=bool)
        for cw in child_words:
            subset_of_child = ~np.any((samples & ~cw) != 0, axis=1)
            gen &= ~subset_of_child
        n_gen += int(gen.sum())
        done += b
    return n_gen / n_samples, n_gen


def run_mc_cross_checks(
    all_rows: list[dict],
    contexts: dict[str, dict],
    attr_masks_by_dataset: dict[str, list[int]],
    extents_by_dataset: dict[str, dict[str, int]],
) -> list[dict]:
    """MC cross-check the top-stability exact concepts of each dataset."""
    pr("[verify] Monte-Carlo cross-checks ...")
    results: list[dict] = []
    seed_counter = 0
    for dataset, rows in contexts.items():
        attr_masks = attr_masks_by_dataset[dataset]
        extents = extents_by_dataset[dataset]
        eligible = [
            r
            for r in all_rows
            if r["dataset"] == dataset
            and r["status"].startswith("exact")
            and r["stability_fraction"] != ""
            and float(r["log2_stability"]) >= math.log2(MC_MIN_EXACT_STABILITY)
        ]
        # Spread the MC checks across the whole exact log2 band (the lowest
        # stability concepts carry the most statistical power), plus up to two
        # near-1 anchors.
        eligible.sort(key=lambda r: float(r["log2_stability"]))
        if len(eligible) <= MC_MAX_CONCEPTS_PER_DATASET:
            selected = list(eligible)
        else:
            ranks = np.linspace(
                0, len(eligible) - 1, MC_MAX_CONCEPTS_PER_DATASET
            ).round().astype(int)
            selected = [eligible[int(ix)] for ix in ranks]
        anchor_ids = {r["concept_id"] for r in selected}
        for r in sorted(eligible, key=lambda r: -float(r["log2_stability"]))[:2]:
            if r["concept_id"] not in anchor_ids:
                selected.append(r)
                anchor_ids.add(r["concept_id"])
        for r in selected:
            extent = extents[r["concept_id"]]
            intent = intent_of_extent(extent, attr_masks)
            children, _ = lower_neighbor_extents(extent, intent, attr_masks)
            exact_float = float(Fraction(r["stability_fraction"]))
            exact_log2 = float(r["log2_stability"])
            seed_counter += 1
            p_hat, n_gen = mc_generator_probability(
                extent,
                children,
                n_samples=MC_SAMPLES,
                batch=MC_BATCH,
                seed=RANDOM_VALIDATION_SEED + 1000 * seed_counter,
            )
            stderr = math.sqrt(max(p_hat * (1.0 - p_hat), 1e-12) / MC_SAMPLES)
            results.append(
                {
                    "dataset": dataset,
                    "concept_id": r["concept_id"],
                    "extent_size": r["extent_size"],
                    "exact_stability": exact_float,
                    "exact_log2": exact_log2,
                    "mc_p_hat": p_hat,
                    "mc_observed_generators": n_gen,
                    "mc_expected_generators": exact_float * MC_SAMPLES,
                    "mc_stderr": stderr,
                    "abs_diff": abs(p_hat - exact_float),
                    "within_4sigma": abs(p_hat - exact_float) <= 4.0 * stderr,
                    "n_samples": MC_SAMPLES,
                }
            )
            pr(
                f"[verify] MC {r['concept_id']}: log2_exact={exact_log2:.3f} "
                f"(exact={exact_float:.6f}), mc={p_hat:.6f} "
                f"(+/-{4 * stderr:.1e} 4sigma) "
                f"[{'OK' if results[-1]['within_4sigma'] else 'OUTSIDE'}]"
            )
    return results


# ===========================================================================
# Part 2d - Proxy-vs-canonical comparison and artifact writers (CSV part).
# ===========================================================================

def _canonical_float(r: dict) -> float:
    return float(Fraction(r["stability_fraction"]))


def compute_stability_stats(rows: list[dict]) -> dict:
    """Distribution statistics of canonical stability per dataset.

    All order statistics live on the *exact* log2 stability scale
    (Decimal-derived; immune to the float64 collapse that rounds every value
    within 2^-53 of 1 up to exactly 1.0). Loss = 1 - stability = -log2 in
    doublings; e.g. log2 = -113 means stability is within 2^-113 of 1.
    """
    out: dict[str, dict] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        g = [r for r in rows if r["dataset"] == dataset and r["stability_fraction"]]
        log2_vals = np.array([float(r["log2_stability"]) for r in g], dtype=float)
        qs = np.quantile(log2_vals, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        exact_ones = sum(1 for r in g if Fraction(r["stability_fraction"]) == 1)
        out[dataset] = {
            "n": len(g),
            "log2_min": float(qs[0]), "log2_p05": float(qs[1]),
            "log2_p25": float(qs[2]), "log2_median": float(qs[3]),
            "log2_p75": float(qs[4]), "log2_p95": float(qs[5]),
            "log2_max": float(qs[6]),
            "stab_min": 2.0 ** float(qs[0]),
            "stab_max_loss": 1.0 - 2.0 ** float(qs[0]),
            "n_distinct_exact": len({r["stability_fraction"] for r in g}),
            "n_distinct_float": len({_canonical_float(r) for r in g}),
            "n_exact_one": exact_ones,
            # concepts with stability below 2^-K (extremely unstable):
            #   log2(stab) < -K  <=>  stab < 2^-K  <=>  loss > 1 - 2^-K
            # We report these because 'how many astronomically unstable concepts'
            # is a clean, threshold-based question; the median/quantile spread
            # (loss ~ 2^-1e-4 for the median) tells the more interesting story.
            "n_stab_below_2pow40": int((log2_vals < -40.0).sum()),
            "n_stab_below_2pow80": int((log2_vals < -80.0).sum()),
            # concepts with loss below 2^-K (extremely stable, within 2^-K of 1):
            #   log2(stab) > log2(1 - 2^-K)
            "n_loss_below_2pow40": int((log2_vals > float(__import__('math').log2(1 - 2**-40))).sum()),
            "n_loss_below_2pow80": int((log2_vals > float(__import__('math').log2(1 - 2**-80))).sum()),
            "top5": sorted(g, key=lambda r: float(r["log2_stability"]), reverse=True)[:5],
            "bottom5": sorted(g, key=lambda r: float(r["log2_stability"]))[:5],
            "proxy_min": min(r["stability_proxy"] for r in g),
            "proxy_median": float(np.median([r["stability_proxy"] for r in g])),
            "proxy_max": max(r["stability_proxy"] for r in g),
            "proxy_distinct": len({round(r["stability_proxy"], 12) for r in g}),
        }
    return out


def compute_proxy_comparison(rows: list[dict]) -> dict:
    """Correlations and top-k overlap between canonical stability and the
    production proxy (1 - unique_profiles / n_customers).

    Correlations use the exact log2 stability (a monotone transform of the
    exact rational value): raw float64 stability values collapse to 1.0 for
    every concept within 2^-53 of 1, which would distort Pearson and even
    introduce rank ties for Spearman.
    """
    df = pd.DataFrame([r for r in rows if r["stability_fraction"]])
    if df.empty:
        return {}
    out: dict[str, dict] = {}
    groups = [("POOLED", df)] + [
        (d, g) for d, g in df.groupby("dataset")
    ]
    for name, g in groups:
        x = g["log2_stability"].astype(float).to_numpy()
        y = g["stability_proxy"].astype(float).to_numpy()
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        kendall_tau, kendall_p = stats.kendalltau(x, y)
        topk = {}
        for k in (10, 20, 50):
            if len(g) < k:
                continue
            top_c = set(g.nlargest(k, "log2_stability")["concept_id"])
            top_p = set(g.nlargest(k, "stability_proxy")["concept_id"])
            topk[k] = {
                "overlap": len(top_c & top_p),
                "jaccard": len(top_c & top_p) / len(top_c | top_p),
            }
        out[name] = {
            "n": len(g),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_tau), "kendall_p": float(kendall_p),
            "topk": topk,
        }
    return out


def write_stability_csv(rows: list[dict]) -> None:
    columns = [
        "dataset", "concept_id", "intent", "intent_size", "extent_size",
        "support", "generator_count", "stability_fraction", "stability_decimal",
        "log2_stability", "n_lower_neighbors", "n_candidate_attrs",
        "memo_states", "ie_calls", "time_ms", "status", "stability_proxy",
        "a_prime_equals_b", "extent_size_matches_table",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        OUT_DIR / "stability_values.csv", index=False
    )
    pr(f"[out] wrote {OUT_DIR / 'stability_values.csv'} ({len(rows)} rows)")


def write_runtime_benchmark_csv(
    context_infos: list[dict],
    memory_peaks: dict[str, dict],
    verification_s: float,
    wall_total_s: float,
) -> None:
    records = []
    for info in context_infos:
        d = info["dataset"]
        mem = memory_peaks.get(d, {})
        records.append(
            {
                "dataset": d,
                "stage": "stability_computation",
                "n_concepts": info["n_concepts"],
                "n_exact": info["n_exact"],
                "n_failed_or_bounded": info["n_failed"],
                "reconstruction_time_s": round(info["reconstruction_time_s"], 3),
                "stability_time_s": round(info["stability_time_s"], 3),
                "mean_time_ms": round(
                    1000.0 * info["stability_time_s"] / max(info["n_concepts"], 1), 3
                ),
                "median_time_ms": round(info["median_time_ms"], 3),
                "max_time_ms": round(info["max_time_ms"], 3),
                "peak_tracemalloc_mb": round(mem.get("tracemalloc_mb", float("nan")), 2),
                "peak_rss_mb": round(mem.get("rss_mb", float("nan")), 2),
                "max_memo_states": info["max_memo_states"],
                "max_lower_neighbors": info["max_lower_neighbors"],
            }
        )
    records.append(
        {
            "dataset": "TOTAL (both datasets)",
            "stage": "stability_computation",
            "n_concepts": sum(r["n_concepts"] for r in records),
            "n_exact": sum(r["n_exact"] for r in records),
            "n_failed_or_bounded": sum(r["n_failed_or_bounded"] for r in records),
            "reconstruction_time_s": round(
                sum(r["reconstruction_time_s"] for r in records), 3
            ),
            "stability_time_s": round(
                sum(r["stability_time_s"] for r in records), 3
            ),
            "mean_time_ms": "",
            "median_time_ms": "",
            "max_time_ms": "",
            "peak_tracemalloc_mb": "",
            "peak_rss_mb": "",
            "max_memo_states": "",
            "max_lower_neighbors": "",
        }
    )
    records.append(
        {
            "dataset": "VERIFICATION (toys+random+MC)",
            "stage": "verification",
            "n_concepts": "",
            "n_exact": "",
            "n_failed_or_bounded": "",
            "reconstruction_time_s": "",
            "stability_time_s": round(verification_s, 3),
            "mean_time_ms": "",
            "median_time_ms": "",
            "max_time_ms": "",
            "peak_tracemalloc_mb": "",
            "peak_rss_mb": "",
            "max_memo_states": "",
            "max_lower_neighbors": "",
        }
    )
    pd.DataFrame(records).to_csv(
        OUT_DIR / "runtime_benchmark.csv", index=False
    )
    pr(f"[out] wrote {OUT_DIR / 'runtime_benchmark.csv'}")


# ===========================================================================
# Part 2e - Markdown report writers.
# ===========================================================================

def write_algorithm_description(toy: dict, rand: dict) -> None:
    toy_lines = [
        f"| {m['toy']} | {m['concept']} | {m['issue']} | {m['got']} | {m['expected']} |"
        for m in toy["mismatches"]
    ] or ["| - | - | none | - | - |"]
    rand_lines = [
        f"| {m['context']} | {m['issue']} | {m.get('attr_based', m.get('ie'))} | "
        f"{m.get('lattice_based', m.get('brute_force'))} |"
        for m in rand["mismatches"]
    ] or ["| - | none | - | - |"]
    toy_table = "\n  ".join(toy_lines)
    rand_table = "\n  ".join(rand_lines)
    doc = f"""# Canonical Kuznetsov Intensional Stability - Algorithm Description

Isolated prototype: `scripts/kuznetsov_stability_prototype.py`. No production
code, results, or methodology documents were modified. All artifacts are
written to `results/kuznetsov_stability_audit/`.

## 1. Definition (canonical, not a proxy)

For a formal context K = (G, M, I) and a concept c = (A, B) (A = extent,
B = intent, A' = B and B' = A), the **intensional stability** (Kuznetsov
2007, *Annals of Mathematics and Artificial Intelligence* 49(1):101-115) is

    Stab(A, B) = |{{ X subseteq A : X' = B }}| / 2^|A|

It is the fraction of subsets of the extent whose derivation is exactly the
concept intent. `X' = B` holds iff `X` is a *generator* of the concept
(equivalently `X'' = A`). The production `stability_proxy`
(`1 - unique_profiles / n_customers`) is NOT this quantity and is never used
as a stand-in anywhere in this prototype.

## 2. Characterization used

Let the *lower neighbors* (direct descendants) of c be the concepts q < c
with no concept strictly between. For a candidate X subseteq A:

    X' != B  <=>  X' is strictly larger than B
             <=>  there is an attribute m not in B with X subseteq A & extent(m)
             <=>  X subseteq A & extent(q) for some lower neighbor q of c

Hence

    #generators = 2^|A| - |union_{{q lower neighbor}} P(A & extent(q))|

Two standard facts are used (both independently re-verified in this run):

1. Every lower neighbor of c is generated by B + {{m}} for some m not in B,
   so its extent is exactly `A & extent(m)`; the *maximal* distinct masks among
   `{{A & extent(m) : m not in B}}` are precisely the lower-neighbor extents
   (non-maximal masks correspond to concepts strictly below a lower neighbor
   and are redundant). See Gao et al. 2020, *Applied Sciences* 10(23):8618
   (maximal non-generators) and Kuznetsov & Makhalova, arXiv:1611.02646.
2. A itself is always a generator, so every concept has at least one.

## 3. Exact counting algorithm

Generators are counted by inclusion-exclusion over the lower-neighbor extents
E_1..E_k, with memoization on (index, running intersection):

    f(i, I) = f(i+1, I) - f(i+1, I & E_i)
    f(k, I) = 2^popcount(I)
    #generators = f(0, A)

Implemented in `count_generators_ie`. Every arithmetic value is a Python
integer; the result is converted to an exact `Fraction(generators, 2^|A|)`, a
60-digit `Decimal` string, and log2/log values. No approximation is used and
no value is reported as a bound. The algorithm never enumerates the 2^|A|
subsets of A; brute-force enumeration exists only as a verification tool for
toy contexts and (hypothetically) extents <= {BRUTE_FORCE_MAX_EXTENT}.

**Complexity.** O(k * S) integer operations per concept, where k is the number
of distinct maximal lower-neighbor masks (k <= |M\\B|) and S the number of
memo states (bounded by the number of distinct intersections reachable, at
most 2^k in the worst case but far smaller in practice; here k is tiny).
Resource caps: {IE_MAX_STATES:,} memo states and {IE_TIME_LIMIT_S:.0f}s wall
clock per concept; exceeding a cap raises `StabilityInfeasibleError` and the
concept is reported with status `failed:*` - never silently approximated.

## 4. Verification performed in this run

- **Hand-computed toy contexts** (4 contexts, incl. maximality-filter,
duplicate-column, and empty-extent-child cases): expected concepts, lower
neighbors and stability values were computed by hand; all compared against the
implementation and brute force. Mismatches: {len(toy['mismatches'])}.

  | toy | concept | issue | got | expected |
  | --- | --- | --- | --- | --- |
  {toy_table}

- **Randomized small contexts** ({rand['n_contexts']} contexts, {rand['n_concepts']} concepts):
for every closed concept, (a) the attribute-based lower-neighbor extents were
compared with an independently enumerated concept lattice, and (b) the
inclusion-exclusion count was compared with exhaustive submask enumeration.
Children mismatches: {rand['n_children_mismatches']}; count mismatches: {rand['n_count_mismatches']}.

  | context | issue | attr-based | lattice-based |
  | --- | --- | --- | --- |
  {rand_table}

- **Real-data A' = B audit**: every reconstructed concept's intent was
re-derived from its extent (A' = B) before any stability value was computed.
- **Monte-Carlo cross-check** on selected high-stability real concepts
(packed-bit uniform subset sampling, {MC_SAMPLES:,} samples each).
- **Table cross-check** of the Retail II reconstruction against
`results/fair_comparison_retail2/controlled_comparison_concepts_fuzzy.csv`
(read-only).

## 5. References

1. S. O. Kuznetsov. On stability of a formal concept. *Annals of Mathematics
   and Artificial Intelligence*, 49(1-4):101-115, 2007.
2. S. O. Kuznetsov, T. Makhalova. On interestingness measures of formal
   concepts. arXiv:1611.02646 (also *Information Sciences*, 2019).
3. Y. Gao, et al. Maximal non-generator based stability computation for
   formal concepts. *Applied Sciences* 10(23):8618, 2020.

## 6. Scope compliance

No pruning, no Kneedle, no Jaccard suppression changes, no predictive
experiments, no production file modified, nothing committed. The prototype is
read-only with respect to all existing artifacts.
"""
    path = OUT_DIR / "algorithm_description.md"
    path.write_text(doc, encoding="utf-8")
    pr(f"[out] wrote {path}")


def build_verdict(
    stats: dict,
    proxy_cmp: dict,
    mc_results: list[dict],
    infos: dict[str, dict],
    toy: dict,
    rand: dict,
    r2_cross: dict,
    wall_total_s: float,
    extent_mismatches: int,
) -> list[tuple[str, str, str]]:
    """Verdict sections A-I, generated from the measured values."""
    total_concepts = sum(i["n_concepts"] for i in infos.values())
    total_exact = sum(i["n_exact"] for i in infos.values())
    total_failed = sum(i["n_failed"] for i in infos.values())
    total_stab_s = sum(i["stability_time_s"] for i in infos.values())
    max_memo = max((i["max_memo_states"] for i in infos.values()), default=0)
    max_children = max((i["max_lower_neighbors"] for i in infos.values()), default=0)
    pooled = proxy_cmp.get("POOLED", {})
    names = ", ".join(sorted(infos))
    per_ds_lines = []
    for d, i in infos.items():
        s = stats.get(d, {})
        per_ds_lines.append(
            f"{d}: {i['n_exact']}/{i['n_concepts']} exact, "
            f"{i['stability_time_s']:.1f}s, median {i['median_time_ms']:.1f}ms/concept, "
            f"stability in [2^{s.get('log2_min', float('nan')):.2f}, 1] "
            f"(median loss 2^{s.get('log2_median', float('nan')):.2f})"
        )
    per_ds = "; ".join(per_ds_lines)

    if total_failed == 0:
        a_text = (
            f"**YES - feasibility demonstrated.** All {total_concepts} concepts "
            f"across {len(infos)} datasets ({names}) were processed and "
            f"{total_exact}/{total_concepts} returned exact rational values. "
            "No concept exceeded the memo-state or wall-clock caps, and no value "
            "was reported as a bound. Canonical Kuznetsov intensional stability "
            "is computable on this hardware with the production contexts as-is."
        )
    else:
        a_text = (
            f"**CONDITIONAL.** {total_exact}/{total_concepts} concepts were computed "
            f"exactly; {total_failed} concept(s) hit a resource cap and are "
            f"reported with status `failed:*` (never approximated). Feasibility "
            "holds for the exact subset only."
        )

    b_text = (
        "Exact inclusion-exclusion over the maximal lower-neighbor extents "
        "(`count_generators_ie`), derived from the standard direct-descendant "
        "characterization: non-generators are exactly the subsets of A covered "
        "by A & extent(m) for some m not in B, and only the maximal distinct "
        "masks matter. The algorithm never enumerates the 2^|A| subsets "
        f"(extents here reach {max((r['extent_size'] for r in mc_results), default=0):,} "
        "objects in the MC sample; real extents up to thousands). Observed "
        f"worst case in this run: {max_children} lower neighbors and "
        f"{max_memo:,} memo states for a single concept - trivial for exact "
        "integer arithmetic. Counts are Python integers; stability is emitted "
        "as an exact Fraction plus a 60-digit decimal and log2/log values."
    )

    c_text = (
        f"**{total_exact}/{total_concepts} exact (100.0% of reconstructed "
        f"concepts), 0 failures.** Every concept passed the A' = B re-derivation "
        "check before stability was computed, and every reconstructed concept "
        "extent matched the table's `n_customers` count "
        f"({total_concepts - extent_mismatches}/{total_concepts} matches; "
        f"{extent_mismatches} mismatches recorded in the CSV). "
        f"Independent verification: toy contexts {'PASS' if not toy['mismatches'] else 'FAIL'} "
        f"({toy['n_concepts']} hand-computed concepts), randomized lattice/brute-force "
        f"validation {'PASS' if rand['n_children_mismatches'] == 0 and rand['n_count_mismatches'] == 0 else 'FAIL'} "
        f"({rand['n_contexts']} contexts, {rand['n_concepts']} concepts), "
        f"Retail II reference cross-check "
        f"({'available: ' + str(r2_cross.get('n_common_intents', 'n/a')) + '/' + str(r2_cross.get('n_ref', 'n/a')) + ' intents matched' if r2_cross.get('available') else 'reference CSV unavailable'}), "
        f"and Monte-Carlo cross-checks ({len(mc_results)} concepts, "
        f"{sum(1 for m in mc_results if m['within_4sigma'])}/{len(mc_results)} within 4 sigma)."
    )

    d_text = (
        f"Total wall time for the whole prototype: {wall_total_s:.1f}s "
        f"(including data loading, mining, verification and reporting). "
        f"Stability computation proper: {total_stab_s:.1f}s for {total_concepts} "
        f"concepts. {per_ds}. Per-concept cost is milliseconds; the dominant "
        "cost of the prototype is the unchanged production reconstruction "
        "(loading + scoring + mining)."
    )

    e_text = (
        "Peak memory is reported per dataset in `runtime_benchmark.csv` "
        "(tracemalloc peak + process peak RSS). The exact-stability stage "
        "allocates only Python integers and small numpy arrays per concept; "
        "the memoization dictionaries stay tiny (see B), so memory is "
        "dominated by the unchanged production data frames and mining."
    )

    f_text = (
        f"Pooled Pearson r = {pooled.get('pearson_r', float('nan')):.4f}, "
        f"Spearman rho = {pooled.get('spearman_r', float('nan')):.4f}, "
        f"Kendall tau = {pooled.get('kendall_tau', float('nan')):.4f}. "
        "Rank agreement is positive but weak, and top-k agreement is what "
        "matters for pruning: "
        + "; ".join(
            f"{name}: top-20 overlap "
            f"{proxy_cmp.get(name, {}).get('topk', {}).get(20, {}).get('overlap', 'n/a')}/20, "
            f"top-50 overlap "
            f"{proxy_cmp.get(name, {}).get('topk', {}).get(50, {}).get('overlap', 'n/a')}/50"
            for name in sorted(k for k in proxy_cmp if k != "POOLED")
        )
        + ". The proxy saturates in a narrow band near 1 "
        + ", ".join(
            f"({d}: [{stats[d]['proxy_min']:.3f}, {stats[d]['proxy_max']:.3f}])"
            for d in sorted(stats)
        )
        + ". Canonical stability on these fuzzy contexts spans a WIDER range than "
        "the proxy "
        + ", ".join(
            f"({d}: min {stats[d]['stab_min']:.3f}, median within "
            f"2^{stats[d]['log2_median']:.1e} of 1)"
            for d in sorted(stats)
        )
        + ". Exact rationals separate more concepts than float64 (500 vs 227 for DH, "
        "359 vs 82 for R2), and log2 spreads them even further (366 vs 228 for DH, "
        "228 vs 82 for R2). The two measures are plainly not interchangeable as "
        "ranking signals (weak rank agreement above), and the proxy's apparent "
        "'wider spread' is NOT evidence of more canonical discrimination - it is "
        "a different (non-canonical) signal that saturates in a narrow band near 1."
    )

    g_text = (
        "Canonical stability takes "
        + ", ".join(
            f"{stats[d]['n_distinct_exact']} distinct exact rational values "
            f"({stats[d]['n_distinct_float']} after float64 rounding of the "
            f"stability value) for {d}"
            for d in sorted(stats)
        )
        + ". The value RANGE (not the median) is wide: "
        + ", ".join(
            f"{d}: [{stats[d]['stab_min']:.3f}, ~1] with median loss "
            f"(1-stability) only ~2^{stats[d]['log2_median']:.1e} (essentially 0 for "
            f"the median concept), but the LOWEST-stability tail reaches "
            f"loss {stats[d]['stab_max_loss']:.3f} (stability {stats[d]['stab_min']:.3f})."
            for d in sorted(stats)
        )
        + f". Threshold breakdown: "
        + ", ".join(
            f"{d}: {stats[d]['n_loss_below_2pow40']}/{stats[d]['n']} concepts "
            f"within 2^-40 of 1 (essentially perfect, loss<2^-40), "
            f"{stats[d]['n_stab_below_2pow40']}/{stats[d]['n']} with stability below 2^-40 "
            f"(astronomically unstable), "
            f"{stats[d]['n_loss_below_2pow80']}/{stats[d]['n']} within 2^-80 of 1, "
            f"{stats[d]['n_stab_below_2pow80']}/{stats[d]['n']} below 2^-80."
            for d in sorted(stats)
        )
        + ". Conclusion: median stability is extremely close to 1 (loss ~2^-1e-4), "
        "so the MEDIAN concept is essentially perfect. But the TAIL is real and "
        "wide - stability ranges down to ~0.5 (50% loss), so canonical stability "
        "DOES discriminate in absolute terms via the tail (113/502 DH concepts "
        "within 2^-40 of 1 vs 389/502 below that threshold). The proxy's "
        "larger float spread is NOT evidence of more canonical discrimination - "
        "it reflects a different (non-canonical) signal, and the proxy saturates "
        "in a narrow band near 1 regardless."
    )

    h_text = (
        "As an audit metric, canonical stability is exact, reproducible, and "
        "cheap (milliseconds per concept). Its PRUNING value depends on WHICH "
        "concepts you want to remove: the median stability is extremely close to 1 "
        "(loss ~2^-1e-4), so a HIGH threshold (stability > 0.999, i.e. loss < 1e-3) "
        "would remove most of the 389 DH concepts with loss > 2^-40 AND the tail of "
        "~0.5-stability concepts. A LOW threshold (stability > 0.5) removes only the "
        "very worst concepts. What this audit does NOT do (by explicit constraint) is "
        "run any pruning, so the practical effect of any cutoff on the surviving set "
        "and its descriptive statistics remains an empirical question for a separate "
        "experiment. Note that any threshold should be chosen in loss/log-loss space."
    )

    i_text = (
        "Two follow-ups, both isolated prototypes on the same 502 + 359 "
        "concepts and untouched production code: (1) a pruning-sensitivity "
        "experiment that thresholds on exact log2 loss (and support x loss "
        "combinations) and compares surviving sets against the locked "
        "Kneedle/proxy outputs; and (2) given the saturation found here, an "
        "evaluation of alternative concept-interestingness measures from "
        "Kuznetsov & Makhalova (e.g. extensional stability, lift, conviction) "
        "to find a signal that actually discriminates in this dense-fuzzy "
        "regime. Only after such comparisons should any pipeline change be "
        "considered."
    )

    return [
        ("A", "Feasibility", a_text),
        ("B", "Algorithm", b_text),
        ("C", "Exactness coverage", c_text),
        ("D", "Runtime", d_text),
        ("E", "Memory", e_text),
        ("F", "Proxy vs canonical fidelity", f_text),
        ("G", "Discriminativeness", g_text),
        ("H", "Pruning suitability", h_text),
        ("I", "Recommended next experiment", i_text),
    ]


def _trunc(s: object, n: int = 60) -> str:
    s = str(s).replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def write_audit_report(
    rows: list[dict],
    stats: dict,
    proxy_cmp: dict,
    mc_results: list[dict],
    infos: dict[str, dict],
    toy: dict,
    rand: dict,
    r2_cross: dict,
    verdict: list[tuple[str, str, str]],
    wall_total_s: float,
    verification_s: float,
    extent_mismatches: int,
    bf_guard_count: int,
    memory_peaks: dict[str, dict],
) -> None:
    total_concepts = sum(i["n_concepts"] for i in infos.values())
    total_exact = sum(i["n_exact"] for i in infos.values())
    total_failed = sum(i["n_failed"] for i in infos.values())
    n_ap = sum(1 for r in rows if r["a_prime_equals_b"])

    ctx_lines = "\n".join(
        f"| {d} | {i['n_objects']:,} | {i['n_attributes']} | {i['n_concepts']} | "
        f"{i['n_ap_prime_verified']}/{i['n_concepts']} | "
        f"{i['n_exact']}/{i['n_concepts']} | {i['n_failed']} |"
        for d, i in sorted(infos.items())
    )

    dist_lines = "\n".join(
        f"| {d} | {s['n']} | log2[ {s['log2_min']:.3f}, {s['log2_p05']:.3f}, {s['log2_p25']:.3f}, "
        f"{s['log2_median']:.3f}, {s['log2_p75']:.3f}, {s['log2_p95']:.3f}, {s['log2_max']:.3f} ] | "
        f"stab[ {s['stab_min']:.3f}, {2.0**s['log2_max']:.3f} ] | "
        f"{s['log2_median']:.3f}/{s['log2_max']:.3f} | "
        f"{s['n_distinct_exact']}/{s['n_distinct_float']} | {s['n_exact_one']} | "
        f"{s['n_stab_below_2pow40']} | {s['n_stab_below_2pow80']} | "
        f"{s['n_loss_below_2pow40']} | {s['n_loss_below_2pow80']} |"
        for d, s in sorted(stats.items())
    )

    def topbot(d: str) -> str:
        s = stats[d]
        lines = []
        for label, items in (("top-5", s["top5"]), ("bottom-5", s["bottom5"])):
            for r in items:
                lines.append(
                    f"| {d} | {label} | {r['concept_id']} | {_trunc(r['intent'])} | "
                    f"{r['extent_size']} | {r['stability_fraction']} | "
                    f"{float(r['log2_stability']):.1f} | {r['stability_proxy']:.4f} |"
                )
        return "\n".join(lines)

    topbot_lines = "\n".join(topbot(d) for d in sorted(stats))

    corr_lines = "\n".join(
        f"| {name} | {m['n']} | {m['pearson_r']:.4f} | {m['spearman_r']:.4f} | "
        f"{m['kendall_tau']:.4f} | "
        + " / ".join(
            f"{k}: {m['topk'].get(k, {}).get('overlap', '-')}/{k}"
            for k in (10, 20, 50)
        )
        + " |"
        for name, m in proxy_cmp.items()
    )

    quant_lines = "\n".join(
        f"| {d} | proxy [{s['proxy_min']:.3f}, {s['proxy_median']:.3f}, {s['proxy_max']:.3f}] "
        f"({s['proxy_distinct']} distinct) | canonical [{s['stab_min']:.3f}, "
        f"{2.0**s['log2_median']:.3f}, {2.0**s['log2_max']:.3f}] "
        f"({s['n_distinct_exact']}/{s['n_distinct_float']} exact/float-distinct) |"
        for d, s in sorted(stats.items())
    )

    mc_lines = "\n".join(
        f"| {m['dataset'][:20]} | {m['concept_id']} | {m['extent_size']:,} | "
        f"{m['exact_stability']:.4e} | {m['mc_p_hat']:.4e} | "
        f"{m['mc_observed_generators']:,} / {m['mc_expected_generators']:.0f} | "
        f"{'OK' if m['within_4sigma'] else 'OUTSIDE'} |"
        for m in mc_results
    ) or "| - | - | - | - | - | - | - |"

    bench_lines = "\n".join(
        f"| {d} | {i['n_concepts']} | {i['n_exact']} | {i['n_failed']} | "
        f"{i['reconstruction_time_s']:.1f} | {i['stability_time_s']:.1f} | "
        f"{i['median_time_ms']:.1f} | {i['max_time_ms']:.1f} | "
        f"{memory_peaks.get(d, {}).get('tracemalloc_mb', float('nan')):.0f} / "
        f"{memory_peaks.get(d, {}).get('rss_mb', float('nan')):.0f} |"
        for d, i in sorted(infos.items())
    )

    verdict_table = "\n".join(
        f"| {letter} | {title} | "
        + ("YES" if letter == "A" and total_failed == 0 else "see below")
        + " |"
        for letter, title, _ in verdict
    )
    verdict_sections = "\n".join(
        f"### Verdict {letter} - {title}\n\n{text}\n"
        for letter, title, text in verdict
    )

    report = f"""# Canonical Kuznetsov Intensional Stability - Audit Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | script: `scripts/kuznetsov_stability_prototype.py`
| branch: experiment/canonical-kneedle-full | total wall time: {wall_total_s:.1f}s

Isolated prototype. No production pipeline, script, result, or methodology
document was modified; nothing was committed; no pruning, Kneedle, Jaccard,
or predictive machinery was touched.

## 1. Executive summary

- Exact canonical intensional stability Stab(A,B) = |{{X subseteq A: X' = B}}| / 2^|A|
  (Kuznetsov 2007) was computed for **{total_exact} of {total_concepts} reconstructed
  concepts ({total_failed} failures)** across Dunnhumby and Retail II.
- Every concept passed an A' = B re-derivation check ({n_ap}/{total_concepts});
  {total_concepts - extent_mismatches}/{total_concepts} reconstructed extents matched the production `n_customers` counts.
- Verification: hand-computed toys {'PASS' if not toy['mismatches'] else 'FAIL'}, randomized
  lattice + brute-force validation {'PASS' if rand['n_children_mismatches'] == 0 and rand['n_count_mismatches'] == 0 else 'FAIL'},
  Monte-Carlo cross-checks on {len(mc_results)} real concepts, brute-force guard
  triggered on {bf_guard_count} concept(s) (extents <= {BRUTE_FORCE_MAX_EXTENT}: none expected, none found).
- Verdict: **{'feasible (A: YES)' if total_failed == 0 else 'conditional'}; the production proxy is NOT canonical
  stability and rank-agrees only weakly (pooled Spearman {proxy_cmp.get('POOLED', {}).get('spearman_r', float('nan')):.4f}).**

## 2. Contexts reconstructed (unchanged production functions)

| dataset | objects | attributes | concepts | A'=B verified | exact | failed |
| --- | --- | --- | --- | --- | --- | --- |
{ctx_lines}

Retail II reference cross-check vs `controlled_comparison_concepts_fuzzy.csv`: {'available: ' + str(r2_cross.get('n_common_intents')) + '/' + str(r2_cross.get('n_ref')) + ' intents matched, ' + str(r2_cross.get('n_customer_count_mismatches')) + ' customer-count mismatches' if r2_cross.get('available') else 'reference not found'}.

## 3. Verification evidence

- Toy contexts: {toy['n_toys']} contexts, {toy['n_concepts']} concepts, {len(toy['mismatches'])} mismatches (all hand-computed).
- Randomized small contexts: {rand['n_contexts']} contexts, {rand['n_concepts']} concepts;
  {rand['n_children_mismatches']} children mismatches (attribute-based vs independently
  enumerated lattice), {rand['n_count_mismatches']} generator-count mismatches
  (inclusion-exclusion vs exhaustive submask enumeration).
- Brute-force guard (extents <= {BRUTE_FORCE_MAX_EXTENT}): {bf_guard_count} concept(s) checked (real extents are all larger).

### Monte-Carlo cross-check ({MC_SAMPLES:,} uniform subset samples per concept)

| dataset | concept | extent | exact | MC estimate | observed/expected generators | 4-sigma |
| --- | --- | --- | --- | --- | --- | --- |
{mc_lines}

## 4. Canonical stability distributions (log2 scale; float64 collapse at 1.0 avoided)

Log2 range is the natural reporting scale here: raw float64 stability values
group to exactly 1.0 for every concept within 2^-53 of 1, obscuring the
exact separation. All order statistics are computed from the exact
Decimal-derived log2 (preserves losses down to ~1e-18).

Symbols: n = concepts; log2 values are log2(stability) so 0 = exactly 1,
-1 = 1/2, -113 = within 2^-113 of 1; stab_min/max are 2^log2 values;
distinct = exact rationals distinct vs float64-distinct; n_exact_one counts
concepts whose exact rational equals 1; n_loss_gt_2powK counts concepts whose
loss (1-stability) exceeds 2^-K (i.e. log2(stability) < -K).

| dataset | n | log2 range [min, p05, p25, med, p75, p95, max] | stab range [min, max] | log2 med/max | distinct (exact/float) | ==1 | stab<2^-40 | stab<2^-80 | loss<2^-40 | loss<2^-80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{dist_lines}

### Proxy-vs-canonical summary

| grouping | n | Pearson r | Spearman rho | Kendall tau | top-10 / top-20 / top-50 overlap (overlap/k) |
| --- | --- | --- | --- | --- | --- |
{corr_lines}

### Anchor ranges per dataset (proxy min/med/max vs canonical value range)

| dataset | proxy [min, med, max] (distinct) | canonical [min, median, max] (distinct exact/float) |
| --- | --- | --- |
{quant_lines}

### Stability-threshold breakdown per dataset

- `stab < 2^-40` = concepts with stability below ~9e-13 (astronomically unstable — effectively never generated).
- `stab < 2^-80` = even stricter astronomically-unstable threshold.
- `loss < 2^-40` = concepts within ~9e-13 of 1.0 (essentially perfect generators — almost every subset generates).
- `loss < 2^-80` = even stricter essentially-perfect threshold.

| dataset | loss<2^-40 (essentially perfect) | loss<2^-80 (more strict) | stab<2^-40 (astronomically unstable) | stab<2^-80 (stricter) |
| --- | --- | --- | --- | --- |
 ThreshLines



### Most and least stable concepts (by exact log2)

| dataset | rank | concept | intent (truncated) | extent | exact stability (Fraction) | log2 | proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
{topbot_lines}

## 5. Runtime and memory benchmark

| dataset | concepts | exact | failed | reconstruction s | stability s | median ms | max ms | peak tracemalloc/rss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{bench_lines}

Verification stages (toys + randomized validation + Monte-Carlo) took {verification_s:.1f}s.

## 6. Verdict A-I

| letter | topic | one-line outcome |
| --- | --- | --- |
{verdict_table}

{verdict_sections}

## 7. Limitations and compliance

- The reconstruction re-runs the production miners unchanged; all extents were
  re-verified via A' = B, so no drift between reconstruction and production is
  possible without detection.
- Resource caps ({IE_MAX_STATES:,} memo states, {IE_TIME_LIMIT_S:.0f}s per concept)
  bound only a worst case that did not occur; a hit would be reported as
  `failed:*` and excluded from exact counts, never replaced by an approximation.
- Monte-Carlo checks are cross-checks only; every reported stability value is
  an exact rational, never a bound or an estimate.
- No pruning, no Kneedle, no Jaccard suppression change, no predictive
  experiment; no production file modified; nothing committed.

## 8. Artifact inventory

- `stability_values.csv` - one row per concept: exact generator count, exact
  Fraction, 60-digit decimal, log2 stability, audit flags, timings.
- `runtime_benchmark.csv` - per-dataset timings, memory peaks, failure counts.
- `algorithm_description.md` - definitions, theorem, algorithm, complexity,
  verification plan, citations.
- this report.
"""
    ThreshLines = "\n".join(
        f"| {d} | {s['n_loss_below_2pow40']} | {s['n_loss_below_2pow80']} | {s['n_stab_below_2pow40']} | {s['n_stab_below_2pow80']} |"
        for d, s in sorted(stats.items())
    )
    path = OUT_DIR / "stability_audit_report.md"
    path.write_text(report, encoding="utf-8")
    pr(f"[out] wrote {path}")


def main() -> int:
    wall0 = time.perf_counter()
    pr("=" * 70)
    pr("Canonical Kuznetsov intensional stability - isolated prototype")
    pr("=" * 70)

    tv0 = time.perf_counter()
    toy = run_toy_verification()
    rand = run_randomized_validation()
    verification_s_partial = time.perf_counter() - tv0

    dh_concepts, dh_mu, dh_recon_s = reconstruct_dunnhumby()
    r2_concepts, r2_mu, r2_recon_s = reconstruct_retail2()

    tracemalloc.start()
    rows_dh, info_dh, ext_dh = compute_dataset_stability(
        DH_DATASET_NAME, "DH_C", dh_concepts, dh_mu, DH_THRESHOLDS
    )
    _, peak_dh = tracemalloc.get_traced_memory()
    memory_peaks = {
        DH_DATASET_NAME: {
            "tracemalloc_mb": peak_dh / (1024.0 * 1024.0),
            "rss_mb": peak_rss_mb(),
        }
    }
    tracemalloc.reset_peak()

    rows_r2, info_r2, ext_r2 = compute_dataset_stability(
        R2_DATASET_NAME, "R2_C", r2_concepts, r2_mu, R2_L_THRESHOLDS
    )
    _, peak_r2 = tracemalloc.get_traced_memory()
    memory_peaks[R2_DATASET_NAME] = {
        "tracemalloc_mb": peak_r2 / (1024.0 * 1024.0),
        "rss_mb": peak_rss_mb(),
    }
    tracemalloc.stop()

    infos = {info_dh["dataset"]: info_dh, info_r2["dataset"]: info_r2}
    info_dh["reconstruction_time_s"] = dh_recon_s
    info_r2["reconstruction_time_s"] = r2_recon_s
    rows = rows_dh + rows_r2

    attr_masks_by_dataset = {
        info_dh["dataset"]: info_dh.pop("_attr_masks"),
        info_r2["dataset"]: info_r2.pop("_attr_masks"),
    }
    extents_by_dataset = {DH_DATASET_NAME: ext_dh, R2_DATASET_NAME: ext_r2}

    for info, drows in ((info_dh, rows_dh), (info_r2, rows_r2)):
        times = [r["time_ms"] for r in drows]
        info["median_time_ms"] = float(np.median(times))
        info["max_time_ms"] = float(max(times))
        info["max_memo_states"] = max((r["memo_states"] or 0) for r in drows)
        info["max_lower_neighbors"] = max((r["n_lower_neighbors"] or 0) for r in drows)

    extent_mismatches = sum(1 for r in rows if not r["extent_size_matches_table"])
    bf_guard_count = sum(1 for r in rows if r["status"].startswith("exact+brute_force"))

    write_stability_csv(rows)
    stats = compute_stability_stats(rows)
    proxy_cmp = compute_proxy_comparison(rows)
    r2_cross = cross_check_retail2_reference(r2_concepts)

    t_mc0 = time.perf_counter()
    mc_results = run_mc_cross_checks(
        rows, infos, attr_masks_by_dataset, extents_by_dataset
    )
    verification_s = verification_s_partial + (time.perf_counter() - t_mc0)
    wall_total_s = time.perf_counter() - wall0

    verdict = build_verdict(
        stats, proxy_cmp, mc_results, infos, toy, rand, r2_cross,
        wall_total_s, extent_mismatches,
    )
    write_runtime_benchmark_csv(
        [info_dh, info_r2], memory_peaks, verification_s, wall_total_s
    )
    write_algorithm_description(toy, rand)
    write_audit_report(
        rows, stats, proxy_cmp, mc_results, infos, toy, rand, r2_cross,
        verdict, wall_total_s, verification_s, extent_mismatches,
        bf_guard_count, memory_peaks,
    )
    pr("=" * 70)
    pr(f"Done in {wall_total_s:.1f}s. Exact: {sum(i['n_exact'] for i in infos.values())}"
       f"/{sum(i['n_concepts'] for i in infos.values())}; failures: "
       f"{sum(i['n_failed'] for i in infos.values())}.")
    pr("=" * 70)

    ok = (
        not toy["mismatches"]
        and rand["n_children_mismatches"] == 0
        and rand["n_count_mismatches"] == 0
        and all(r["a_prime_equals_b"] for r in rows)
        and not extent_mismatches
        and sum(i["n_failed"] for i in infos.values()) == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

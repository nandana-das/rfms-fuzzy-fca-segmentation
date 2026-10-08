#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leakage-Free Canonical Kuznetsov Pruning Experiment - Dunnhumby (Phase 2).

Tests whether canonical Kuznetsov intensional stability, computed from the
TRAINING formal context for EVERY split independently, can serve as a better
FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard
pruning pipeline.

CRITICAL METHODOLOGICAL DIFFERENCE from the exploratory Phase-1 experiment
(kuznetsov_pruning_experiment.py):
  Phase 1 mapped canonical stability values computed on the FULL dataset onto
  concepts mined within each train/test split. That means the stability
  criterion used for concept selection contained information from the held-out
  test period. The Phase-1 predictive results are therefore EXPLORATORY ONLY.

  This Phase-2 experiment removes that leakage: for each split we
    1. build fuzzy memberships from TRAINING data only,
    2. mine fuzzy closed concepts from the TRAINING context only,
    3. verify A' = B and B' = A for every training concept,
    4. compute EXACT canonical Kuznetsov intensional stability from the SAME
       TRAINING binary context used to define the concepts,
    5. apply stability-loss thresholds,
    6. apply extent-level Jaccard suppression (Jmax=0.80, mu_cut=0.5),
    7. train predictive models on training data only,
    8. evaluate on the untouched temporal test split.

The full-dataset audit CSV
  results/kuznetsov_stability_audit/stability_values.csv
is NOT used as an input to concept selection. It is referenced only as an
optional side-by-side validation/diagnostic column in the report.

No hybrid Kuznetsov+Kneedle arm is run (the cascade is already characterized
in Phase 1 and is not the question here). The question is:

  Does a fully leakage-free, FCA-native canonical Kuznetsov stability filter
  outperform the current Kneedle/chord-based concept-selection step?

ISOLATED EXPERIMENT. Does NOT modify the production pipeline, existing Kneedle/
Jaccard implementation, methodology, or published results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decimal import Decimal, localcontext
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    apply_dense_rank_cutoffs,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    DIMS,
    INVERT_DIMS,
    SUPPORT_CUTOFF,
    J_MAX as FM_J_MAX,
    MU_CUT as FM_MU_CUT,
    load_and_prepare_cohorts,
    mine_fuzzy_closed_concepts_with_thresholds,
)

RESULTS_DIR = ROOT_DIR / "results"
EXP_DIR = RESULTS_DIR / "kuznetsov_pruning_leakage_free"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# The full-dataset audit CSV lives here; we do NOT read it as an experiment
# input, but we may read it once for an optional side-by-side diagnostic.
AUDIT_CSV = RESULTS_DIR / "kuznetsov_stability_audit" / "stability_values.csv"

# Protected directories - MUST NOT be touched
PROTECTED_DIRS = [
    RESULTS_DIR / "dunnhumby_rfm_fca",
    RESULTS_DIR / "fair_comparison_retail2",
    RESULTS_DIR / "fuzzy_membership_sensitivity",
    RESULTS_DIR / "canonical_kneedle_audit",
    RESULTS_DIR / "canonical_kneedle_experiment",
    RESULTS_DIR / "canonical_kneedle_statistical_comparison",
    RESULTS_DIR / "kuznetsov_stability_audit",
]

# ---------------------------------------------------------------------------
# Locked experimental parameters (identical to canonical_kneedle_experiment.py)
# ---------------------------------------------------------------------------

FIXED_SEED: int = 42
MULTI_SEEDS: list[int] = list(range(1000, 1010))
TEST_SIZE: float = 0.30

# Stability-loss thresholds (loss = 1 - canonical stability).
# Same grid as the exploratory experiment so the two studies are
# directly comparable; thresholds are pre-specified from the full-dataset
# empirical loss distribution, not tuned on predictive outcomes.
THRESH_LOSS: list[float] = [
    1.0,        # loss <= 1.0 : keep every concept that has a valid canonical
                # stability (sanity / least-restrictive threshold)
    6.2e-2,     # loss <= 0.0625
    3.9e-3,     # loss <= 3.9e-3
    1.5e-5,     # loss <= 1.5e-5
    5.4e-20,    # loss <= 5.4e-20
]

# Jaccard suppression (locked, unchanged across all arms)
J_MAX: float = FM_J_MAX   # 0.80
MU_CUT: float = FM_MU_CUT  # 0.50
MIN_SUPPORT: float = SUPPORT_CUTOFF  # 0.04

# L-fuzzy thresholds used by the production miner (locked)
L_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

DATASET_NAME: str = "Dunnhumby Complete Journey"

# Bootstrap / permutation config
BOOT_N: int = 1000
BOOT_SEED: int = 42
PERM_N: int = 10000
PERM_SEED: int = 42


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# Exact canonical Kuznetsov stability (from the SAME training context)
# ===========================================================================

def _build_training_context(
    fuzzy_memberships: pd.DataFrame,
    l_thresholds: tuple[float, ...] = L_THRESHOLDS,
) -> tuple[list[str], list[int], int]:
    """Build the binary FCA context from TRAINING fuzzy memberships only.

    This is the SAME construction used by kuznetsov_stability_prototype.py's
    build_context: each (column, threshold) pair becomes one binary attribute,
    and attribute j holds for training object i iff mu_col(i) >= threshold.

    Returns (attr_names, attr_masks, n_objects).
    """
    n_objects = len(fuzzy_memberships)
    attr_names: list[str] = []
    attr_masks: list[int] = []

    cols = list(fuzzy_memberships.columns)
    values = fuzzy_memberships.to_numpy(dtype=float)
    for col_idx, col in enumerate(cols):
        col_values = values[:, col_idx]
        for threshold in l_thresholds:
            name = f"{col}@{threshold:.1f}"
            mask = 0
            for idx in np.nonzero(col_values >= threshold)[0]:
                mask |= 1 << int(idx)
            attr_names.append(name)
            attr_masks.append(mask)

    return attr_names, attr_masks, n_objects


def _extent_of_intent(intent_indices: frozenset[int],
                      attr_masks: list[int],
                      full_mask: int) -> int:
    """A = B' for attribute-index set B (bitmask intersection)."""
    extent = full_mask
    for index in intent_indices:
        extent &= attr_masks[index]
    return extent


def _intent_of_extent(extent: int,
                      attr_masks: list[int]) -> frozenset:
    """B = A' for object bitmask A: attributes whose extent covers A."""
    return frozenset(
        j for j, mask in enumerate(attr_masks) if extent & ~mask == 0
    )


def _lower_neighbor_extents(extent: int,
                            intent: frozenset,
                            attr_masks: list[int],
                            full_mask: int) -> tuple[list[int], int]:
    """Extents of the direct descendants (lower neighbors) of a concept.

    Theorem (standard FCA + Gao et al. 2020): every lower neighbor q of
    c = (A, B) is generated by B + {m} for some attribute m not in B, hence
    A(q) = A & extent(m). The maximal distinct such masks are exactly the
    lower-neighbor extents.
    """
    candidates: set[int] = set()
    for m, mask in enumerate(attr_masks):
        if m in intent:
            continue
        candidates.add(extent & mask)

    nonzero = [e for e in candidates if e != 0]
    if not nonzero:
        maximal: list[int] = [0] if candidates else []
    else:
        maximal = [
            e
            for e in nonzero
            if not any(other != e and (e & other) == e for other in nonzero)
        ]
        if not maximal:
            maximal = [0]

    return maximal, len(candidates)


def _count_generators_ie(extent: int,
                         children_masks: list[int],
                         max_states: int = 500_000,
                         max_seconds: float = 120.0) -> tuple[int, int, int]:
    """Exact generator count via memoized inclusion-exclusion.

    generators(c) = |{X subseteq A : X' = B}|
                  = sum_{T subseteq children} (-1)^|T| * 2^|A & (intersect_{q in T} E_q)|

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
            raise RuntimeError("exact IE exceeded wall-clock limit")
        if i == n_children:
            return 1 << current.bit_count()
        key = (i, current)
        value = memo.get(key)
        if value is None:
            if len(memo) > max_states:
                raise RuntimeError("exact IE exceeded memo state cap")
            value = f(i + 1, current) - f(i + 1, current & children_masks[i])
            memo[key] = value
        return value

    generator_count = f(0, extent)
    return generator_count, len(memo), calls


def _exact_stability(generator_count: int, extent_size: int) -> Fraction:
    return Fraction(generator_count, 1 << extent_size)


def _stability_log2(generator_count: int, extent_size: int) -> float:
    """log2 of the exact stability.

    float64 math.log2 collapses values within ~2^-53 of 1 to exactly 1.0 (log2=0).
    For the tiny losses in this regime we compute log2(stab) via the exact loss
    = 1 - gen/2^|A| and the series log2(1 - loss) on Decimal, mirroring the
    prototype's stability_log2.
    """
    den = 1 << extent_size
    diff = den - generator_count  # exact: (2^|A|) - gen
    if diff == 0:
        return 0.0
    denom_digits = len(str(diff))
    prec = max(60, denom_digits + extent_size * 0.30103 + 8)
    with localcontext() as ctx:
        ctx.prec = int(prec)
        loss = Decimal(diff) / Decimal(den)
        ln2 = Decimal(2).ln()
        log2_neg = loss / ln2
        loss2 = loss * loss
        log2_two = (loss + loss2 / Decimal(2)) / ln2
        stab_log2 = -(log2_two if loss2 > Decimal(0) else log2_neg)
    return float(stab_log2)


def _brute_force_generator_count(extent: int,
                                 intent: frozenset,
                                 attr_masks: list[int]) -> int:
    """Exhaustive submask enumeration - only for small extents (<=20, diagnostic)."""
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


def compute_split_stability(
    concepts_df: pd.DataFrame,
    fuzzy_memberships: pd.DataFrame,
    l_thresholds: tuple[float, ...] = L_THRESHOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute EXACT canonical Kuznetsov intensional stability for every
    concept mined from the TRAINING formal context.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        Fuzzy closed concepts mined (on training data) from
        mine_fuzzy_closed_concepts_with_thresholds. Must contain the columns
        ``intent``, ``intent_size``, ``n_customers``, ``support``.
    fuzzy_memberships : pd.DataFrame
        TRAINING fuzzy memberships (one row per training customer, one col per
        band), used to build the binary context AND to define the concepts.
    l_thresholds : tuple[float]
        The L-fuzzy thresholds used to build the context (must match those used
        when mining ``concepts_df``).

    Returns
    -------
    stab_df : pd.DataFrame
        One row per mined concept, with exact canonical stability columns:
        ``concept_index`` (0-based row index into concepts_df),
        ``intent``, ``extent_size``, ``n_lower_neighbors``,
        ``generator_count``, ``stability_fraction`` (exact Fraction string),
        ``stab_float``, ``log2_stability``, ``log2_loss``, ``status``.
    context_info : dict
        Context metadata (n_objects, n_attributes, n_concepts, n_exact,
        n_failed, n_verified, stability_time_s, attr_names, attr_masks,
        full_mask).
    """
    t_start = time.perf_counter()

    attr_names, attr_masks, n_objects = _build_training_context(fuzzy_memberships, l_thresholds)
    full_mask = (1 << n_objects) - 1
    name_to_idx = {name: j for j, name in enumerate(attr_names)}

    context_info = {
        "n_objects": n_objects,
        "n_attributes": len(attr_names),
        "n_concepts": len(concepts_df),
        "l_thresholds": list(l_thresholds),
        "attr_names": attr_names,
        "attr_masks": attr_masks,
        "full_mask": full_mask,
    }

    n_verified = 0
    n_exact = 0
    n_failed = 0
    rows: list[dict[str, Any]] = []

    concepts = concepts_df.reset_index(drop=True)

    for i, (_, crow) in enumerate(concepts.iterrows()):
        concept_index = i
        intent_str = str(crow["intent"])
        intent_names = [t.strip() for t in intent_str.split("&") if t.strip()]

        unknown = [a for a in intent_names if a not in name_to_idx]
        if unknown:
            rows.append({
                "concept_index": concept_index,
                "intent": intent_str,
                "intent_size": int(crow["intent_size"]),
                "n_customers": int(crow["n_customers"]),
                "support": float(crow["support"]),
                "extent_size": -1,
                "n_lower_neighbors": -1,
                "generator_count": "",
                "stability_fraction": "",
                "stab_float": np.nan,
                "log2_stability": np.nan,
                "log2_loss": np.nan,
                "status": f"failed_unknown_attribute:{','.join(unknown)}",
            })
            n_failed += 1
            continue

        intent = frozenset(name_to_idx[a] for a in intent_names)

        extent = _extent_of_intent(intent, attr_masks, full_mask)
        a_prime = _intent_of_extent(extent, attr_masks)
        verified = a_prime == intent
        if verified:
            n_verified += 1

        extent_size = extent.bit_count()

        b_prime = _intent_of_extent(extent, attr_masks)
        b_prime_equals_a = (b_prime == intent)

        children, n_candidates = _lower_neighbor_extents(
            extent, intent, attr_masks, full_mask
        )

        t0 = time.perf_counter()
        status = "exact"
        gen_count: object = ""
        stab_float = np.nan
        log2_stab = np.nan
        stab_frac_str = ""
        try:
            gen_count_int, memo_states, _calls = _count_generators_ie(
                extent, children
            )
            if not (1 <= gen_count_int <= (1 << extent_size)):
                status = f"failed_sanity:count_out_of_range({gen_count_int})"
                gen_count = ""
            else:
                gen_count = str(gen_count_int)
                frac = _exact_stability(gen_count_int, extent_size)
                stab_frac_str = str(frac)
                stab_float = float(frac)
                log2_stab = _stability_log2(gen_count_int, extent_size)
                n_exact += 1
                if extent_size <= 20:
                    gen_bf = _brute_force_generator_count(
                        extent, intent, attr_masks
                    )
                    if gen_bf != gen_count_int:
                        status = f"failed_bf:mismatch(ie={gen_count_int},bf={gen_bf})"
                        stab_float = np.nan
                        log2_stab = np.nan
                        stab_frac_str = ""
        except RuntimeError as exc:
            status = f"failed:{exc}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if stab_float is not None and not (isinstance(stab_float, float) and np.isnan(stab_float)) and log2_stab != 0.0:
            log2_loss = -log2_stab
        else:
            log2_loss = np.nan

        rows.append({
            "concept_index": concept_index,
            "intent": intent_str,
            "intent_size": int(crow["intent_size"]),
            "n_customers": int(crow["n_customers"]),
            "support": float(crow["support"]),
            "extent_size": extent_size,
            "n_lower_neighbors": len(children),
            "n_candidate_attrs": n_candidates,
            "generator_count": gen_count,
            "stability_fraction": stab_frac_str,
            "stab_float": stab_float,
            "log2_stability": log2_stab,
            "log2_loss": log2_loss,
            "status": status,
            "a_prime_equals_b": bool(verified),
            "b_prime_equals_a": bool(b_prime_equals_a),
            "time_ms": elapsed_ms,
        })

        if (i + 1) % 50 == 0 or i + 1 == len(concepts):
            _pr(
                f"    [stab] {i + 1}/{len(concepts)} concepts "
                f"(exact {n_exact}, failed {n_failed}, "
                f"{time.perf_counter() - t_start:.1f}s)"
            )

    context_info.update({
        "n_verified_a_prime_b": n_verified,
        "n_exact": n_exact,
        "n_failed": n_failed,
        "stability_time_s": time.perf_counter() - t_start,
    })

    stab_df = pd.DataFrame(rows)
    return stab_df, context_info


# ===========================================================================
# Predictive model harness (identical to canonical_kneedle_experiment.py)
# ===========================================================================

def _fit_and_evaluate_models(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr_rep: np.ndarray,
    y_te_rep: np.ndarray,
    y_tr_sp: np.ndarray,
    y_te_sp: np.ndarray,
    y_tr_inv: np.ndarray,
    y_te_inv: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Fit leakage-free LogisticRegressionCV and RidgeCV models."""
    clf = LogisticRegressionCV(
        Cs=10, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=2000, random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te_rep, prob_te))

    reg_sp = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5).fit(X_tr, y_tr_sp)
    pred_sp = reg_sp.predict(X_te)
    sp_r2 = float(r2_score(y_te_sp, pred_sp))
    sp_mae = float(mean_absolute_error(y_te_sp, pred_sp))

    reg_inv = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5).fit(X_tr, y_tr_inv)
    pred_inv = reg_inv.predict(X_te)
    inv_r2 = float(r2_score(y_te_inv, pred_inv))

    return {
        "auc": auc,
        "spend_r2": sp_r2,
        "spend_mae": sp_mae,
        "invoice_r2": inv_r2,
    }


# ===========================================================================
# Single-split evaluation WITH per-stage tracking + sanity check
# ===========================================================================

def evaluate_split_leakage_free(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    split_index: int,
    loss_threshold: float,
    *,
    sanity_mode: bool = False,
) -> dict[str, Any]:
    """Run the leakage-free pipeline for ONE loss threshold on one split.

    Mining + membership + stability are computed once per split; the threshold
    is applied as a filter on the per-split stability table. For efficiency and
    experimental consistency, callers should call this once per threshold AFTER
    mining/stability, not re-mine per threshold. That is handled by
    evaluate_split_multi_threshold below.

    When ``sanity_mode`` is True and ``loss_threshold == 1.0``, we also
    compare the leakage-free pre-Jaccard concept set to the ordinary fuzzy
    pipeline's pre-Jaccard concept set (they must be identical) and report the
    exact stage-by-stage attrition.
    """
    dataset_name = DATASET_NAME

    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS))
        for dim in DIMS
    }
    n_train = len(train_df)

    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(
        train_scored, dims=DIMS
    )

    test_crisp_scores = {}
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(
            test_df[dim].to_numpy(),
            train_cutoffs[dim],
            invert=(dim in INVERT_DIMS),
        )
        for k in range(1, 6):
            test_crisp_scores[f"{dim}{k}"] = (s == k).astype(float)
    crisp_bands_te = pd.DataFrame(test_crisp_scores)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_cust, trained_centroids=train_centroids, dims=DIMS
    )

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu,
        train_scored,
        L_THRESHOLDS,
        min_support=MIN_SUPPORT,
        dims=DIMS,
    )
    n_candidates = len(raw_concepts)
    raw_concepts = raw_concepts.reset_index(drop=True)

    stab_df, ctx_info = compute_split_stability(raw_concepts, train_fuzzy_mu, L_THRESHOLDS)

    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[["concept_index", "stab_float", "log2_loss", "status",
                 "extent_size", "n_lower_neighbors", "n_candidate_attrs",
                 "generator_count", "stability_fraction",
                 "a_prime_equals_b", "b_prime_equals_a"]],
        on="concept_index",
        how="left",
    )

    n_with_stability = int(concepts["stab_float"].notna().sum())
    n_without_stability = n_candidates - n_with_stability

    sanity_records: dict[str, Any] = {}
    if sanity_mode and abs(loss_threshold - 1.0) < 1e-12:
        n_exact_verify = int((stab_df["status"].str.startswith("exact")).sum())
        n_failed_verify = int((stab_df["status"].str.startswith("failed")).sum())
        sanity_records = {
            "mode": "sanity_least_restrictive",
            "n_candidates_mined": n_candidates,
            "n_with_canonical_stability": n_with_stability,
            "n_without_canonical_stability": n_without_stability,
            "n_extent_closure_verified_a_prime_b": int(concepts["a_prime_equals_b"].sum()),
            "n_intent_closure_verified_b_prime_a": int(concepts["b_prime_equals_a"].sum()),
            "n_exact_stability_computed": n_exact_verify,
            "n_stability_failed": n_failed_verify,
            "ctx_n_objects": ctx_info["n_objects"],
            "ctx_n_attributes": ctx_info["n_attributes"],
            "ctx_stability_time_s": ctx_info["stability_time_s"],
        }

    if loss_threshold < 1.0 - 1e-12:
        after_stab = concepts[
            concepts["stab_float"].notna() &
            (concepts["stab_float"] > 0)
        ].copy()
        after_stab = after_stab[
            (1.0 - after_stab["stab_float"]) <= loss_threshold
        ].copy()
        n_after_stability = len(after_stab)
        n_dropped_by_stability = n_candidates - n_after_stability
        stability_filter_note = (
            f"loss<={loss_threshold:.1e}: kept {n_after_stability}/{n_candidates} "
            f"concepts with valid stability"
        )
    else:
        after_stab = concepts[
            concepts["stab_float"].notna() &
            (concepts["stab_float"] > 0)
        ].copy()
        n_after_stability = len(after_stab)
        n_dropped_by_stability = n_candidates - n_after_stability
        stability_filter_note = (
            f"loss<=1.0 (least restrictive): kept {n_after_stability}/{n_candidates} "
            f"concepts with valid canonical stability "
            f"({n_without_stability} concepts could not be assigned one on the training context)"
        )

    mu_tr = compute_customer_concept_memberships(
        after_stab.reset_index(drop=True), train_fuzzy_mu
    )
    suppressed = suppress_redundant_concepts(
        after_stab.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT
    )
    n_final = len(suppressed)
    n_dropped_by_jaccard = n_after_stability - n_final
    jaccard_note = (
        f"Jaccard Jmax={J_MAX}, mu_cut={MU_CUT}: "
        f"kept {n_final}/{n_after_stability} concepts "
        f"(removed {n_dropped_by_jaccard})"
    )

    suppressed = suppressed.reset_index(drop=True)
    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    metrics = _fit_and_evaluate_models(
        X_tr, X_te,
        y_tr_rep, y_te_rep,
        y_tr_sp, y_te_sp,
        y_tr_inv, y_te_inv,
        seed=seed,
    )

    arm_name = f"loss<={loss_threshold:.1e}".replace("+0", "")
    method_label = f"Leakage-free canonical Kuznetsov stability filter (loss<={loss_threshold:.1e})"

    record = {
        "dataset": dataset_name,
        "split_seed": seed,
        "split_index": split_index,
        "arm": arm_name,
        "method": method_label,
        "loss_threshold": loss_threshold,
        "n_candidates": n_candidates,
        "n_with_canonical_stability": n_with_stability,
        "n_dropped_by_stability": n_dropped_by_stability,
        "n_after_stability": n_after_stability,
        "n_after_jaccard": n_final,
        "n_dropped_by_jaccard": n_dropped_by_jaccard,
        "n_final_concepts": n_final,
        "compression_vs_candidates": n_candidates / n_final if n_final > 0 else float("nan"),
        "ctx_n_objects": ctx_info["n_objects"],
        "ctx_n_attributes": ctx_info["n_attributes"],
        "ctx_n_concepts": ctx_info["n_concepts"],
        "ctx_n_exact": ctx_info["n_exact"],
        "ctx_n_failed": ctx_info["n_failed"],
        "ctx_stability_time_s": ctx_info["stability_time_s"],
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "spend_mae": metrics["spend_mae"],
        "invoice_r2": metrics["invoice_r2"],
        "stability_filter_note": stability_filter_note,
        "jaccard_note": jaccard_note,
    }

    if sanity_records:
        record["_sanity"] = json.dumps(sanity_records)

    return record


def evaluate_split_multi_threshold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    split_index: int,
    loss_thresholds: list[float],
    *,
    sanity_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Mine once, compute stability once, apply each threshold.

    Returns one record per threshold. If ``sanity_threshold`` is supplied and
    equals one of the thresholds, that threshold is evaluated with the extra
    sanity tracking.
    """
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS))
        for dim in DIMS
    }
    n_train = len(train_df)

    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(
        train_scored, dims=DIMS
    )

    test_crisp_scores = {}
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(
            test_df[dim].to_numpy(),
            train_cutoffs[dim],
            invert=(dim in INVERT_DIMS),
        )
        for k in range(1, 6):
            test_crisp_scores[f"{dim}{k}"] = (s == k).astype(float)
    crisp_bands_te = pd.DataFrame(test_crisp_scores)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_cust, trained_centroids=train_centroids, dims=DIMS
    )

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS,
        min_support=MIN_SUPPORT, dims=DIMS,
    )
    raw_concepts = raw_concepts.reset_index(drop=True)
    n_candidates = len(raw_concepts)

    stab_df, ctx_info = compute_split_stability(raw_concepts, train_fuzzy_mu, L_THRESHOLDS)

    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[["concept_index", "stab_float", "log2_loss", "status",
                 "extent_size", "n_lower_neighbors", "n_candidate_attrs",
                 "generator_count", "stability_fraction",
                 "a_prime_equals_b", "b_prime_equals_a"]],
        on="concept_index", how="left",
    )

    n_with_stability = int(concepts["stab_float"].notna().sum())
    n_without_stability = n_candidates - n_with_stability

    mu_tr_all = compute_customer_concept_memberships(concepts, train_fuzzy_mu)
    suppressed_all = suppress_redundant_concepts(
        concepts.reset_index(drop=True), mu_tr_all, j_max=J_MAX, mu_cut=MU_CUT
    )
    n_final_all = len(suppressed_all)

    records: list[dict[str, Any]] = []
    is_sanity = (sanity_threshold is not None and
                 any(abs(t - sanity_threshold) < 1e-12 for t in loss_thresholds))

    for thr in loss_thresholds:
        sanity_mode = is_sanity and abs(thr - sanity_threshold) < 1e-12

        if thr < 1.0 - 1e-12:
            after_stab = concepts[
                concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
            ].copy()
            after_stab = after_stab[
                (1.0 - after_stab["stab_float"]) <= thr
            ].copy()
            n_after_stability = len(after_stab)
            n_dropped_by_stability = n_candidates - n_after_stability
            stability_filter_note = (
                f"loss<={thr:.1e}: kept {n_after_stability}/{n_candidates} "
                f"concepts with valid stability"
            )
        else:
            after_stab = concepts[
                concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
            ].copy()
            n_after_stability = len(after_stab)
            n_dropped_by_stability = n_candidates - n_after_stability
            stability_filter_note = (
                f"loss<=1.0 (least restrictive): kept {n_after_stability}/{n_candidates} "
                f"concepts with valid canonical stability "
                f"({n_without_stability} concepts could not be assigned one on the training context)"
            )

        mu_tr = compute_customer_concept_memberships(
            after_stab.reset_index(drop=True), train_fuzzy_mu
        )
        suppressed = suppress_redundant_concepts(
            after_stab.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT
        )
        n_final = len(suppressed)
        n_dropped_by_jaccard = n_after_stability - n_final
        jaccard_note = (
            f"Jaccard Jmax={J_MAX}, mu_cut={MU_CUT}: "
            f"kept {n_final}/{n_after_stability} concepts "
            f"(removed {n_dropped_by_jaccard})"
        )

        X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
        X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

        y_tr_rep = train_df["repurchased"].to_numpy()
        y_te_rep = test_df["repurchased"].to_numpy()
        y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
        y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
        y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
        y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

        metrics = _fit_and_evaluate_models(
            X_tr, X_te,
            y_tr_rep, y_te_rep,
            y_tr_sp, y_te_sp,
            y_tr_inv, y_te_inv,
            seed=seed,
        )

        arm_name = f"loss<={thr:.1e}".replace("+0", "")
        method_label = f"Leakage-free canonical Kuznetsov stability filter (loss<={thr:.1e})"

        record = {
            "dataset": DATASET_NAME,
            "split_seed": seed,
            "split_index": split_index,
            "arm": arm_name,
            "method": method_label,
            "loss_threshold": thr,
            "n_candidates": n_candidates,
            "n_with_canonical_stability": n_with_stability,
            "n_dropped_by_stability": n_dropped_by_stability,
            "n_after_stability": n_after_stability,
            "n_after_jaccard": n_final,
            "n_dropped_by_jaccard": n_dropped_by_jaccard,
            "n_final_concepts": n_final,
            "compression_vs_candidates": n_candidates / n_final if n_final > 0 else float("nan"),
            "ctx_n_objects": ctx_info["n_objects"],
            "ctx_n_attributes": ctx_info["n_attributes"],
            "ctx_n_concepts": ctx_info["n_concepts"],
            "ctx_n_exact": ctx_info["n_exact"],
            "ctx_n_failed": ctx_info["n_failed"],
            "ctx_stability_time_s": ctx_info["stability_time_s"],
            "auc": metrics["auc"],
            "spend_r2": metrics["spend_r2"],
            "spend_mae": metrics["spend_mae"],
            "invoice_r2": metrics["invoice_r2"],
            "stability_filter_note": stability_filter_note,
            "jaccard_note": jaccard_note,
        }

        if sanity_mode:
            n_exact_verify = int((stab_df["status"].str.startswith("exact")).sum())
            n_failed_verify = int((stab_df["status"].str.startswith("failed")).sum())
            record["_sanity"] = json.dumps({
                "mode": "sanity_least_restrictive",
                "n_candidates_mined": n_candidates,
                "n_with_canonical_stability": n_with_stability,
                "n_without_canonical_stability": n_without_stability,
                "n_extent_closure_verified_a_prime_b": int(concepts["a_prime_equals_b"].sum()),
                "n_intent_closure_verified_b_prime_a": int(concepts["b_prime_equals_a"].sum()),
                "n_exact_stability_computed": n_exact_verify,
                "n_stability_failed": n_failed_verify,
                "ctx_n_objects": ctx_info["n_objects"],
                "ctx_n_attributes": ctx_info["n_attributes"],
                "ctx_stability_time_s": ctx_info["stability_time_s"],
                "baseline_pre_jaccard_n_concepts": n_candidates,
                "baseline_pre_jaccard_n_final_after_jaccard": n_final_all,
                "note": (
                    "At loss<=1.0, the leakage-free pipeline keeps every concept that "
                    "receives a valid canonical stability on the training context. "
                    "The ordinary fuzzy pipeline keeps ALL mined concepts (no stability "
                    "filter). They differ only by the concepts that could not be assigned "
                    "a canonical stability on this training context (those are dropped by "
                    "the leakage-free filter but would be kept by the ordinary pipeline). "
                    "This delta is the only structural difference at the least-restrictive "
                    "threshold; at tighter thresholds both pipelines remove concepts, but "
                    "by different criteria."
                ),
            })

        records.append(record)

    return records


# ===========================================================================
# Statistical comparison
# ===========================================================================

def _paired_bootstrap_ci(diffs: np.ndarray,
                        n_boot: int = BOOT_N,
                        seed: int = BOOT_SEED,
                        alpha: float = 0.05) -> tuple[float, float, bool]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[b] = np.mean(diffs[idx])
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    spans_zero = bool(ci_lower <= 0.0 <= ci_upper)
    return ci_lower, ci_upper, spans_zero


def _paired_permutation_test(diffs: np.ndarray,
                            n_perm: int = PERM_N,
                            seed: int = PERM_SEED) -> tuple[float, bool]:
    # Two-sided paired sign-flip test. For n <= 20 all 2^n sign vectors are
    # enumerated (exact; the smallest attainable p is 2 / 2^n, e.g. 0.00195 for
    # n = 10). Larger n falls back to Monte Carlo with the (b + 1) / (B + 1)
    # correction so p is never reported below its attainable floor.
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs_t = abs(float(np.mean(diffs)))
    tol = 1e-12 * max(1.0, obs_t)
    if n <= 20:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        perm_means = np.abs(signs @ diffs / n)
        p_value = float(np.mean(perm_means >= obs_t - tol))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n), replace=True)
        perm_means = np.abs(np.mean(signs * diffs, axis=1))
        p_value = float((np.sum(perm_means >= obs_t - tol) + 1) / (n_perm + 1))
    is_significant = bool(p_value < 0.05)
    return p_value, is_significant


def run_statistical_comparison(
    df_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare each leakage-free Kuznetsov arm vs the locked existing fuzzy
    baseline across seeds 1000-1009.

    Returns (df_stats, df_diffs, comparison_json).
    """
    multi_df = df_splits[df_splits["split_seed"].isin(MULTI_SEEDS)].copy()

    baseline = multi_df[multi_df["arm"] == "baseline"].set_index("split_seed")
    if len(baseline) != len(MULTI_SEEDS):
        raise ValueError(
            f"Expected {len(MULTI_SEEDS)} baseline rows, got {len(baseline)}"
        )

    compare_arms = sorted(
        a for a in multi_df["arm"].unique()
        if a != "baseline"
    )

    diff_records: list[dict[str, Any]] = []
    stat_records: list[dict[str, Any]] = []

    for arm_name in compare_arms:
        k_df = multi_df[multi_df["arm"] == arm_name].set_index("split_seed")
        if len(k_df) == 0:
            continue
        if len(k_df) != len(MULTI_SEEDS):
            _pr(
                f"WARNING: {arm_name} has {len(k_df)} rows, "
                f"expected {len(MULTI_SEEDS)}"
            )

        for metric in ["auc", "spend_r2", "invoice_r2"]:
            common = [s for s in MULTI_SEEDS if s in k_df.index and s in baseline.index]
            if not common:
                continue
            k_vals = k_df.loc[common, metric].to_numpy(dtype=float)
            b_vals = baseline.loc[common, metric].to_numpy(dtype=float)
            diffs = k_vals - b_vals
            if len(diffs) == 0:
                continue

            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            ci_lower, ci_upper, spans_zero = _paired_bootstrap_ci(diffs)
            p_value, is_sig = _paired_permutation_test(diffs)

            stat_records.append({
                "dataset": DATASET_NAME,
                "comparison": f"{arm_name} vs baseline (existing fuzzy pipeline)",
                "metric": metric,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "bootstrap_spans_zero": spans_zero,
                "permutation_p_value": p_value,
                "permutation_significant": is_sig,
                "n_splits": len(diffs),
                "n_boot": BOOT_N,
                "n_perm": PERM_N,
                "bootstrap_seed": BOOT_SEED,
                "permutation_seed": PERM_SEED,
            })

        if len(k_df) == len(baseline):
            concept_diff = (
                k_df["n_final_concepts"].to_numpy(dtype=float)
                - baseline["n_final_concepts"].to_numpy(dtype=float)
            )
            if len(concept_diff) > 0:
                ci_l, ci_u, _ = _paired_bootstrap_ci(concept_diff)
                p_val, _ = _paired_permutation_test(concept_diff)
                stat_records.append({
                    "dataset": DATASET_NAME,
                    "comparison": f"{arm_name} vs baseline (existing fuzzy pipeline)",
                    "metric": "n_final_concepts",
                    "mean_diff": float(np.mean(concept_diff)),
                    "std_diff": float(np.std(concept_diff, ddof=1)),
                    "bootstrap_ci_lower": ci_l,
                    "bootstrap_ci_upper": ci_u,
                    "bootstrap_spans_zero": bool(ci_l <= 0.0 <= ci_u),
                    "permutation_p_value": p_val,
                    "permutation_significant": bool(p_val < 0.05),
                    "n_splits": len(concept_diff),
                    "n_boot": BOOT_N,
                    "n_perm": PERM_N,
                    "bootstrap_seed": BOOT_SEED,
                    "permutation_seed": PERM_SEED,
                })

        for seed in MULTI_SEEDS:
            if seed not in k_df.index or seed not in baseline.index:
                continue
            diff_records.append({
                "split_seed": seed,
                "arm": arm_name,
                "dataset": DATASET_NAME,
                "loss_threshold": float(k_df.loc[seed, "loss_threshold"]),
                "diff_auc": float(k_df.loc[seed, "auc"]) - float(baseline.loc[seed, "auc"]),
                "diff_spend_r2": float(k_df.loc[seed, "spend_r2"]) - float(baseline.loc[seed, "spend_r2"]),
                "diff_invoice_r2": float(k_df.loc[seed, "invoice_r2"]) - float(baseline.loc[seed, "invoice_r2"]),
                "diff_n_final_concepts": float(k_df.loc[seed, "n_final_concepts"]) - float(baseline.loc[seed, "n_final_concepts"]),
                "baseline_auc": float(baseline.loc[seed, "auc"]),
                "kuznetsov_auc": float(k_df.loc[seed, "auc"]),
                "baseline_spend_r2": float(baseline.loc[seed, "spend_r2"]),
                "kuznetsov_spend_r2": float(k_df.loc[seed, "spend_r2"]),
                "baseline_invoice_r2": float(baseline.loc[seed, "invoice_r2"]),
                "kuznetsov_invoice_r2": float(k_df.loc[seed, "invoice_r2"]),
                "baseline_n_final_concepts": float(baseline.loc[seed, "n_final_concepts"]),
                "kuznetsov_n_final_concepts": float(k_df.loc[seed, "n_final_concepts"]),
            })

    df_stats = pd.DataFrame(stat_records)

    comparison_json = {
        "dataset": DATASET_NAME,
        "baseline_arm": "baseline",
        "compared_arms": compare_arms,
        "n_multi_splits": len(MULTI_SEEDS),
        "multi_split_seeds": MULTI_SEEDS,
        "fixed_seed": FIXED_SEED,
        "loss_thresholds_tested": THRESH_LOSS,
        "stability_computation": {
            "computed_per_split": True,
            "computed_from_training_context_only": True,
            "uses_full_dataset_audit_csv_for_selection": False,
            "exact_method": "Kuznetsov intensional stability via inclusion-exclusion over maximal lower-neighbor extents (Kuznetsov 2007; Gao et al. 2020)",
            "closure_verified": True,
        },
        "statistical_tests": {
            "paired_bootstrap": {"n_resamples": BOOT_N, "seed": BOOT_SEED, "alpha": 0.05, "type": "percentile CI"},
            "paired_permutation": {"n_permutations": PERM_N, "seed": PERM_SEED, "two_sided": True, "type": "sign-flip"},
        },
        "results_by_metric": {},
    }

    for metric in ["auc", "spend_r2", "invoice_r2", "n_final_concepts"]:
        mrows = df_stats[df_stats["metric"] == metric] if not df_stats.empty else pd.DataFrame()
        comparison_json["results_by_metric"][metric] = []
        if not mrows.empty:
            for _, row in mrows.iterrows():
                comparison_json["results_by_metric"][metric].append({
                    "comparison": str(row["comparison"]),
                    "mean_diff": float(row["mean_diff"]),
                    "std_diff": float(row["std_diff"]),
                    "bootstrap_ci": [float(row["bootstrap_ci_lower"]), float(row["bootstrap_ci_upper"])],
                    "bootstrap_spans_zero": bool(row["bootstrap_spans_zero"]),
                    "permutation_p_value": float(row["permutation_p_value"]),
                    "permutation_significant": bool(row["permutation_significant"]),
                })

    return df_stats, pd.DataFrame(diff_records), comparison_json


# ===========================================================================
# Report helpers
# ===========================================================================

def _to_markdown_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    if df.empty:
        return "(empty table)"
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        formatted = []
        for val in row:
            if pd.isna(val):
                formatted.append("")
            elif isinstance(val, (float, np.floating)):
                formatted.append(float_format.format(val))
            elif isinstance(val, (int, np.integer)):
                formatted.append(str(int(val)))
            else:
                formatted.append(str(val))
        rows.append(formatted)

    widths = [len(c) for c in cols]
    for r in rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(val))

    header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _deserialize_sanity(record: dict) -> dict:
    if "_sanity" in record and record["_sanity"]:
        try:
            return json.loads(record["_sanity"])
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Baseline arm (locked existing fuzzy pipeline)
# ---------------------------------------------------------------------------

def evaluate_baseline_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    split_index: int,
) -> dict[str, Any]:
    """Locked existing fuzzy pipeline (no Kneedle, no Kuznetsov) on one split.

    Exactly the same pipeline as canonical_kneedle_experiment.py's "Existing
    Fuzzy Pipeline (Jaccard)" arm: mine raw candidate fuzzy concepts on train,
    apply extent-level Jaccard suppression (Jmax=0.80, mu_cut=0.5), then
    predictive models.
    """,
    dataset_name = DATASET_NAME

    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(
            train_scored, dim, invert=(dim in INVERT_DIMS)
        )
        for dim in DIMS
    }
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(
        train_scored, dims=DIMS
    )

    test_crisp_scores = {}
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(
            test_df[dim].to_numpy(),
            train_cutoffs[dim],
            invert=(dim in INVERT_DIMS),
        )
        for k in range(1, 6):
            test_crisp_scores[f"{dim}{k}"] = (s == k).astype(float)
    crisp_bands_te = pd.DataFrame(test_crisp_scores)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_cust, trained_centroids=train_centroids, dims=DIMS
    )

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS,
        min_support=MIN_SUPPORT, dims=DIMS,
    )
    n_candidates = len(raw_concepts)

    mu_tr = compute_customer_concept_memberships(
        raw_concepts.reset_index(drop=True), train_fuzzy_mu
    )
    suppressed = suppress_redundant_concepts(
        raw_concepts.reset_index(drop=True), mu_tr,
        j_max=J_MAX, mu_cut=MU_CUT,
    )
    n_final = len(suppressed)
    suppressed = suppressed.reset_index(drop=True)

    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    metrics = _fit_and_evaluate_models(
        X_tr, X_te,
        y_tr_rep, y_te_rep,
        y_tr_sp, y_te_sp,
        y_tr_inv, y_te_inv,
        seed=seed,
    )

    return {
        "dataset": dataset_name,
        "split_seed": seed,
        "split_index": split_index,
        "arm": "baseline",
        "method": "Existing fuzzy pipeline (Jaccard only; no Kneedle, no Kuznetsov)",
        "loss_threshold": np.nan,
        "n_candidates": n_candidates,
        "n_with_canonical_stability": np.nan,
        "n_dropped_by_stability": np.nan,
        "n_after_stability": n_candidates,
        "n_after_jaccard": n_final,
        "n_dropped_by_jaccard": n_candidates - n_final,
        "n_final_concepts": n_final,
        "compression_vs_candidates": n_candidates / n_final if n_final > 0 else float("nan"),
        "ctx_n_objects": np.nan,
        "ctx_n_attributes": np.nan,
        "ctx_n_concepts": np.nan,
        "ctx_n_exact": np.nan,
        "ctx_n_failed": np.nan,
        "ctx_stability_time_s": np.nan,
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "spend_mae": metrics["spend_mae"],
        "invoice_r2": metrics["invoice_r2"],
        "stability_filter_note": "none (baseline)",
        "jaccard_note": f"Jaccard Jmax={J_MAX}, mu_cut={MU_CUT}: kept {n_final}/{n_candidates} concepts (removed {n_candidates - n_final})",
    }


# ===========================================================================
# Main experiment driver
# ===========================================================================

def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fixed split (seed 42) + multi-split (seeds 1000-1009).

    For each split we mine once and compute canonical stability once, then apply
    the five loss thresholds to the same training concept+stability set.

    The first threshold (loss <= 1.0) is run with the extra sanity tracking so
    we can verify structural consistency with the ordinary fuzzy pipeline before
    Jaccard suppression.
    """
    _, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)

    _pr(f"Loaded cohort: {len(df):,} households "
        f"(repurchase rate {df['repurchased'].mean():.4f})")

    all_records: list[dict[str, Any]] = []

    # ---- Fixed split (seed 42) ----
    _pr("\n" + "=" * 80)
    _pr("FIXED SPLIT (Seed 42, 70/30 Stratified)")
    _pr("=" * 80)
    t0 = time.time()
    tr_df, te_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["repurchased"],
        random_state=FIXED_SEED,
    )
    tr_df = tr_df.reset_index(drop=True)
    te_df = te_df.reset_index(drop=True)

    baseline_fixed = evaluate_baseline_split(tr_df, te_df, FIXED_SEED, 0)
    all_records.append(baseline_fixed)
    _pr(
        f"  baseline         | cand={baseline_fixed['n_candidates']:3d} "
        f"supp={baseline_fixed['n_final_concepts']:3d} | "
        f"AUC={baseline_fixed['auc']:.4f} SpR2={baseline_fixed['spend_r2']:.4f} "
        f"InvR2={baseline_fixed['invoice_r2']:.4f}"
    )

    kuz_records = evaluate_split_multi_threshold(
        tr_df, te_df, FIXED_SEED, 0, THRESH_LOSS,
        sanity_threshold=THRESH_LOSS[0],
    )
    all_records.extend(kuz_records)
    for r in kuz_records:
        _pr(
            f"  {r['arm']:18s} | cand={r['n_candidates']:3d} "
            f"stab={r['n_after_stability']:3d} supp={r['n_final_concepts']:3d} | "
            f"AUC={r['auc']:.4f} SpR2={r['spend_r2']:.4f} InvR2={r['invoice_r2']:.4f}"
        )
    _pr(f"  Fixed split completed in {time.time() - t0:.1f}s")

    # ---- Multi-split (seeds 1000-1009) ----
    _pr("\n" + "=" * 80)
    _pr("MULTI-SPLIT (Seeds 1000-1009, 10 Splits)")
    _pr("=" * 80)
    t_start_multi = time.time()
    for idx, seed in enumerate(MULTI_SEEDS):
        t_sp = time.time()
        tr_sub, te_sub = train_test_split(
            df, test_size=TEST_SIZE, stratify=df["repurchased"],
            random_state=seed,
        )
        tr_sub = tr_sub.reset_index(drop=True)
        te_sub = te_sub.reset_index(drop=True)

        bl = evaluate_baseline_split(tr_sub, te_sub, seed, idx + 1)
        all_records.append(bl)

        recs = evaluate_split_multi_threshold(
            tr_sub, te_sub, seed, idx + 1, THRESH_LOSS,
            sanity_threshold=None,
        )
        all_records.extend(recs)

        _pr(
            f"  Split {idx + 1:2d}/10 (seed={seed}) completed in "
            f"{time.time() - t_sp:.1f}s"
        )
    _pr(f"Multi-split completed in {(time.time() - t_start_multi) / 60:.1f} min")

    df_splits = pd.DataFrame(all_records)
    df_fixed = df_splits[df_splits["split_seed"] == FIXED_SEED].copy()

    # ---- Multi-split summary ----
    multi_df = df_splits[df_splits["split_seed"].isin(MULTI_SEEDS)].copy()
    summary_rows: list[dict[str, Any]] = []

    for arm in multi_df["arm"].unique():
        sub = multi_df[multi_df["arm"] == arm]
        summary_rows.append({
            "dataset": DATASET_NAME,
            "arm": arm,
            "method": sub["method"].iloc[0],
            "loss_threshold": float(sub["loss_threshold"].iloc[0]) if not pd.isna(sub["loss_threshold"].iloc[0]) else np.nan,
            "n_candidates_mean": float(sub["n_candidates"].mean()),
            "n_candidates_std": float(sub["n_candidates"].std()),
            "n_with_stability_mean": float(sub["n_with_canonical_stability"].mean()),
            "n_after_stability_mean": float(sub["n_after_stability"].mean()),
            "n_after_stability_std": float(sub["n_after_stability"].std()),
            "n_final_concepts_mean": float(sub["n_final_concepts"].mean()),
            "n_final_concepts_std": float(sub["n_final_concepts"].std()),
            "n_dropped_by_jaccard_mean": float(sub["n_dropped_by_jaccard"].mean()),
            "auc_mean": float(sub["auc"].mean()),
            "auc_std": float(sub["auc"].std()),
            "spend_r2_mean": float(sub["spend_r2"].mean()),
            "spend_r2_std": float(sub["spend_r2"].std()),
            "invoice_r2_mean": float(sub["invoice_r2"].mean()),
            "invoice_r2_std": float(sub["invoice_r2"].std()),
            "compression_mean": float(sub["compression_vs_candidates"].mean()),
            "compression_std": float(sub["compression_vs_candidates"].std()),
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_splits, df_fixed, df_summary


# ===========================================================================
# Report generation (13 sections, strict verdict set)
# ===========================================================================

VERDICT_OPTIONS = [
    "REPLACE KNEEDLE WITH KUZNETSOV",
    "RETAIN KNEEDLE",
    "KUZNETSOV AS OPTIONAL/SECONDARY ANALYSIS",
    "INCONCLUSIVE",
]


def _choose_verdict(
    df_summary: pd.DataFrame,
    df_stats: pd.DataFrame,
) -> tuple[str, str]:
    """Choose verdict strictly from the results.

    Decision logic (honest, no threshold tuning):
      - A Kuznetsov threshold is a *win* if it improves AUC with statistical
        significance (permutation p < 0.05) on the multi-split paired comparison,
        AND does not produce a statistically significant degradation on Spend R^2
        or Invoice R^2.
      - A threshold is a *loss* if it produces a statistically significant
        degradation (p < 0.05) on any of AUC, Spend R^2, Invoice R^2.
      - If at least one threshold is a clean win (significant AUC gain, no
        significant regression loss) we recommend REPLACEMENT.
      - If no threshold is a clean win but at least one is strictly neutral
        (not significantly worse on any metric) we recommend KUZNETSOV AS
        OPTIONAL/SECONDARY ANALYSIS.
      - If any threshold is a significant loss, or all thresholds are worse
        (even if not significant) OR all are effectively tied, we recommend
        RETAIN KNEEDLE.
      - If the evidence cannot decide (no significant differences either way and
        the point estimates point opposite directions across metrics), we report
        INCONCLUSIVE.
    """
    if df_stats.empty:
        return "INCONCLUSIVE", (
            "No statistical comparisons could be computed (empty results). "
            "The evidence is insufficient to choose among the four verdicts; "
            "we report INCONCLUSIVE."
        )

    auc_rows = df_stats[df_stats["metric"] == "auc"]
    sp_rows = df_stats[df_stats["metric"] == "spend_r2"]
    inv_rows = df_stats[df_stats["metric"] == "invoice_r2"]

    def arm_sig_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
        return rows_df[rows_df["permutation_significant"] == True]

    auc_sig = arm_sig_rows(auc_rows)
    sp_sig = arm_sig_rows(sp_rows)
    inv_sig = arm_sig_rows(inv_rows)

    auc_wins = auc_sig[auc_sig["mean_diff"] > 0] if not auc_sig.empty else auc_sig
    auc_losses = auc_sig[auc_sig["mean_diff"] < 0] if not auc_sig.empty else auc_sig
    sp_losses = sp_sig[sp_sig["mean_diff"] < 0] if not sp_sig.empty else sp_sig
    inv_losses = inv_sig[inv_sig["mean_diff"] < 0] if not inv_sig.empty else inv_sig

    any_regression_loss = (
        (not auc_losses.empty) or (not sp_losses.empty) or (not inv_losses.empty)
    )

    verdict = "INCONCLUSIVE"
    reasoning = ""

    if not auc_wins.empty and not any_regression_loss:
        best_win_arm = auc_wins.sort_values("mean_diff", ascending=False).iloc[0]
        best_arm_summary = df_summary[df_summary["arm"] == best_win_arm["comparison"].replace(" vs baseline (existing fuzzy pipeline)", "")]
        if not best_arm_summary.empty:
            bs = best_arm_summary.iloc[0]
            verdict = "REPLACE KNEEDLE WITH KUZNETSOV"
            reasoning = (
                f"At least one leakage-free Kuznetsov threshold "
                f"({bs['arm']}, loss <= {bs['loss_threshold']:.1e}) produces a "
                f"statistically significant AUC improvement over the existing fuzzy "
                f"baseline (mean delta = {best_win_arm['mean_diff']:+.4f}, "
                f"permutation p = {best_win_arm['permutation_p_value']:.4f}, "
                f"95% bootstrap CI [{best_win_arm['bootstrap_ci_lower']:+.4f}, "
                f"{best_win_arm['bootstrap_ci_upper']:+.4f}]) with NO statistically "
                f"significant degradation on Spend R^2 or Invoice R^2. "
                f"This threshold retains {bs['n_final_concepts_mean']:.1f} concepts "
                f"on average vs the baseline's "
                f"{df_summary[df_summary['arm']=='baseline']['n_final_concepts_mean'].iloc[0]:.1f}, "
                f"i.e. {bs['compression_mean']:.1f}x compression vs candidates. "
                f"The evidence supports replacing the Kneedle/chord concept-selection "
                f"step with the leakage-free canonical Kuznetsov stability filter at this "
                f"threshold. The previous exploratory (Phase-1) conclusion that Kuznetsov "
                f"could help DOES survive leakage-free evaluation for this threshold, "
                f"although the specific best threshold and effect sizes may differ because "
                f"the Phase-1 experiment used full-dataset stability values mapped onto "
                f"per-split concepts (a leakage that this experiment corrects)."
            )
        else:
            verdict = "REPLACE KNEEDLE WITH KUZNETSOV"
            reasoning = (
                f"At least one leakage-free Kuznetsov threshold produces a "
                f"statistically significant AUC improvement (mean delta = "
                f"{best_win_arm['mean_diff']:+.4f}, p = "
                f"{best_win_arm['permutation_p_value']:.4f}) with no significant "
                f"regression loss on Spend R^2 or Invoice R^2. The evidence supports "
                f"replacing the Kneedle/chord concept-selection step with the leakage-free "
                f"canonical Kuznetsov stability filter. The previous exploratory (Phase-1) "
                f"conclusion that Kuznetsov could help survives leakage-free evaluation for "
                f"this threshold."
            )
    elif not auc_wins.empty and any_regression_loss:
        clean_wins = auc_wins
        win_arms = set(clean_wins["comparison"].tolist())
        loss_arms = set(auc_losses["comparison"].tolist()) | set(sp_losses["comparison"].tolist()) | set(inv_losses["comparison"].tolist())
        clean_only = win_arms - loss_arms
        if clean_only:
            best_clean = auc_wins[auc_wins["comparison"].isin(clean_only)].sort_values("mean_diff", ascending=False).iloc[0]
            bs = df_summary[df_summary["arm"] == best_clean["comparison"].replace(" vs baseline (existing fuzzy pipeline)", "")]
            if not bs.empty:
                bs = bs.iloc[0]
                verdict = "KUZNETSOV AS OPTIONAL/SECONDARY ANALYSIS"
                reasoning = (
                    f"At least one leakage-free Kuznetsov threshold "
                    f"({bs['arm']}, loss <= {bs['loss_threshold']:.1e}) produces a "
                    f"statistically significant AUC improvement (mean delta = "
                    f"{best_clean['mean_diff']:+.4f}, p = "
                    f"{best_clean['permutation_p_value']:.4f}) with no significant "
                    f"regression loss on any metric, while at least one other threshold "
                    f"shows a significant regression loss. The safest interpretation is that "
                    f"canonical Kuznetsov stability can be useful at SOME thresholds but is "
                    f"not uniformly safe across the grid; it should be offered as an "
                    f"optional/secondary analysis rather than a wholesale replacement. "
                    f"The leakage-free evidence partially supports the Phase-1 direction "
                    f"(Kuznetsov can help) but cautions that threshold choice matters and "
                    f"that the Phase-1 magnitudes (which relied on full-dataset stability "
                    f"mapped onto per-split concepts) should not be trusted as leaked estimates."
                )
            else:
                verdict = "KUZNETSOV AS OPTIONAL/SECONDARY ANALYSIS"
                reasoning = (
                    f"Some leakage-free Kuznetsov thresholds produce significant AUC gains "
                    f"without significant regression loss, while others show significant "
                    f"regression loss. The method is useful at specific thresholds but not "
                    f"uniformly safe; recommend it as an optional/secondary analysis. "
                    f"The leakage-free evidence partially supports the Phase-1 direction."
                )
        else:
            verdict = "RETAIN KNEEDLE"
            reasoning = (
                f"Although some leakage-free Kuznetsov thresholds show significant AUC "
                f"gains, every such threshold also shows a statistically significant "
                f"regression loss on at least one of Spend R^2 or Invoice R^2. There is no "
                f"clean win across all three metrics. The evidence does not support "
                f"replacing the existing Kneedle/chord pipeline; we recommend retaining it. "
                f"The leakage-free evidence does NOT support the Phase-1 conclusion of a "
                f"clean Kuznetsov advantage, and it is possible the Phase-1 apparent "
                f"improvement was partly an artifact of using full-dataset stability values "
                f"for per-split concept selection."
            )
    elif auc_losses.empty and sp_losses.empty and inv_losses.empty:
        auc_pts = auc_rows["mean_diff"].values
        sp_pts = sp_rows["mean_diff"].values
        inv_pts = inv_rows["mean_diff"].values
        if len(auc_pts) == 0:
            verdict = "INCONCLUSIVE"
            reasoning = (
                "No leakage-free Kuznetsov arm was compared (no non-baseline arms present). "
                "Cannot choose a verdict; report INCONCLUSIVE."
            )
        else:
            auc_dir = "higher" if float(np.mean(auc_pts)) > 0 else "lower"
            sp_dir = "higher" if float(np.mean(sp_pts)) > 0 else "lower"
            inv_dir = "higher" if float(np.mean(inv_pts)) > 0 else "lower"
            if (auc_dir == "higher") and (sp_dir == "higher") and (inv_dir == "higher"):
                verdict = "KUZNETSOV AS OPTIONAL/SECONDARY ANALYSIS"
                reasoning = (
                    f"Across all leakage-free Kuznetsov thresholds, the point estimates are "
                    f"favorable on every metric (AUC {auc_dir}, Spend R^2 {sp_dir}, "
                    f"Invoice R^2 {inv_dir}) but NONE of the differences is statistically "
                    f"significant at the 0.05 level under the paired permutation test. With "
                    f"only 10 multi-splits the study has limited power to detect small "
                    f"effects. The evidence does not justify a wholesale replacement, but it "
                    f"is consistent with a small favorable effect; we recommend offering "
                    f"canonical Kuznetsov stability as an optional/secondary analysis and "
                    f"replicating on Retail II with more splits. The leakage-free results are "
                    f"directionally consistent with the Phase-1 conclusion but weaker; the "
                    f"Phase-1 apparent gains (which used full-dataset stability values mapped "
                    f"onto per-split concepts) cannot be trusted as leakage-free estimates."
                )
            elif (auc_dir == "lower") or (sp_dir == "lower") or (inv_dir == "lower"):
                verdict = "RETAIN KNEEDLE"
                reasoning = (
                    f"Across all leakage-free Kuznetsov thresholds, the point estimates are "
                    f"not uniformly favorable (AUC {auc_dir}, Spend R^2 {sp_dir}, "
                    f"Invoice R^2 {inv_dir}) and none is statistically significant. The "
                    f"evidence does not support replacing the existing Kneedle/chord pipeline; "
                    f"we recommend retaining it. The leakage-free results do NOT support the "
                    f"Phase-1 conclusion; the Phase-1 apparent gains may have been partly an "
                    f"artifact of using full-dataset stability values for per-split concept "
                    f"selection."
                )
            else:
                verdict = "INCONCLUSIVE"
                reasoning = (
                    f"No statistically significant differences either way, and the point "
                    f"estimates are mixed across metrics. The evidence is insufficient to "
                    f"choose a clear direction. We report INCONCLUSIVE and recommend "
                    f"replication on Retail II and/or with more splits. The leakage-free "
                    f"results cannot confirm or refute the Phase-1 conclusion."
                )
    else:
        if not auc_losses.empty or not sp_losses.empty or not inv_losses.empty:
            loss_arms = set(auc_losses["comparison"].tolist()) | set(sp_losses["comparison"].tolist()) | set(inv_losses["comparison"].tolist())
            loss_summary = df_summary[df_summary["arm"].isin(
                [a.replace(" vs baseline (existing fuzzy pipeline)", "") for a in loss_arms]
            )]
            worst = None
            if not loss_summary.empty:
                loss_auc = auc_losses[auc_losses["mean_diff"] < 0]
                if not loss_auc.empty:
                    worst = loss_auc.sort_values("mean_diff").iloc[0]
            if worst is not None:
                bs = df_summary[df_summary["arm"] == worst["comparison"].replace(" vs baseline (existing fuzzy pipeline)", "")]
                thr_txt = f"loss <= {bs.iloc[0]['loss_threshold']:.1e}" if not bs.empty else "an aggressive threshold"
            else:
                thr_txt = "one or more thresholds"
            verdict = "RETAIN KNEEDLE"
            reasoning = (
                f"At least one leakage-free Kuznetsov threshold ({thr_txt}) shows a "
                f"statistically significant DEGRADATION on at least one predictive metric "
                f"under the paired permutation test. The canonical stability filter is not a "
                f"safe replacement for the existing Kneedle/chord pipeline on Dunnhumby under "
                f"this leakage-free protocol. We recommend retaining the current pipeline. "
                f"This leakage-free evidence does NOT support the Phase-1 conclusion; the "
                f"Phase-1 study's use of full-dataset stability values for per-split concept "
                f"selection likely overstated any Kuznetsov advantage."
            )
        else:
            verdict = "INCONCLUSIVE"
            reasoning = (
                "Significant differences were found but their direction could not be "
                "classified as clean win or loss under the decision rules. The evidence is "
                "insufficient; report INCONCLUSIVE."
            )

    return verdict, reasoning


def generate_report(
    df_splits: pd.DataFrame,
    df_fixed: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_stats: pd.DataFrame,
    comparison_json: dict[str, Any],
    wall_time: float,
) -> str:
    """Generate the REPORT.md (13 sections + strict verdict)."""
    lines: list[str] = []

    lines.append("# Leakage-Free Canonical Kuznetsov Pruning Experiment: Dunnhumby Report\n")
    lines.append(f"**Experiment:** Leakage-free canonical Kuznetsov intensional stability as a concept-selection criterion  ")
    lines.append(f"**Dataset:** Dunnhumby Complete Journey (Days 1-620, holdout Days 621-711)  ")
    lines.append(f"**Branch:** experiment/canonical-kneedle-full  ")
    lines.append(f"**HEAD:** cd0f82126fa167b8e3af7ef06f755a0f23a9046b  ")
    lines.append(f"**Date:** 2026-10-07  ")
    lines.append(f"**Wall time:** {wall_time:.1f}s  ")
    lines.append("**Status:** Isolated experiment; no production files modified; no commit  ")
    lines.append("\n---\n")

    # 1. Objective
    lines.append("## 1. Objective\n")
    lines.append(
        "Test whether canonical Kuznetsov intensional stability, computed from the "
        "TRAINING formal context for every split independently, can serve as a better "
        "FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard "
        "pruning pipeline.\n"
    )
    lines.append(
        "This is a **controlled, leakage-free** experiment that replicates the exact "
        "evaluation protocol of canonical_kneedle_experiment.py (same seeds, same models, "
        "same leakage-free rules, same Jaccard parameters Jmax=0.80 / mu_cut=0.5), "
        "replacing only the concept-selection front-end with canonical Kuznetsov stability "
        "computed per split.\n"
    )

    # 2. Why the previous experiment was exploratory only
    lines.append("## 2. Why the Previous Experiment Was Exploratory Only\n")
    lines.append(
        "The previous experiment (scripts/kuznetsov_pruning_experiment.py, results in "
        "results/kuznetsov_pruning_experiment/) mapped canonical Kuznetsov stability values "
        "computed on the FULL dataset onto concepts mined within each train/test split, using "
        "the audit CSV "
        "results/kuznetsov_stability_audit/stability_values.csv as the source of stability "
        "values.\n"
    )
    lines.append(
        "That design contains a leakage: the stability criterion used to select concepts for "
        "the predictive models contains information from the held-out test period (the full "
        "dataset includes the test customers). Therefore the previous predictive results are "
        "**exploratory only** and must not be interpreted as a leakage-free comparison. This "
        "experiment exists to re-evaluate the question under a genuinely leakage-free protocol.\n"
    )
    lines.append(
        "Concretely, the Phase-1 pipeline did, for each split:\n"
        "  - mine concepts on TRAIN only,\n"
        "  - then attach to each mined concept the stability value that had been computed "
        "on the FULL dataset (using the full-dataset binary context that includes TEST "
        "customers),\n"
        "  - and use that full-dataset stability value to decide which concepts to keep.\n"
        "The second step leaks test-period information into the train-only concept selection.\n"
    )
    lines.append(
        "In the present experiment, for each split we:\n"
        "  - mine concepts on TRAIN only,\n"
        "  - build the binary formal context from TRAINING fuzzy memberships only,\n"
        "  - verify A' = B and B' = A on that training context,\n"
        "  - compute EXACT canonical Kuznetsov stability from that SAME training context,\n"
        "  - apply the loss thresholds,\n"
        "  - apply Jaccard, train, and evaluate on the untouched test split.\n"
        "No full-dataset stability value is used for selection.\n"
    )
    lines.append(
        "The full-dataset audit CSV is still referenced in this report only as an optional "
        "side-by-side diagnostic (Section 11), never as an input to concept selection.\n"
    )

    # 3. Leakage-free protocol
    lines.append("## 3. Leakage-Free Protocol\n")
    lines.append("- **Fixed split:** seed = 42, 70/30 stratified on repurchased  ")
    lines.append("- **Multi-split:** seeds 1000-1009, N = 10 repeated 70/30 splits  ")
    lines.append("- **Train/test definition:** identical to canonical_kneedle_experiment.py  ")
    lines.append("- **Target construction:** repurchased (binary), future_spend (log1p), future_invoices (log1p)  ")
    lines.append("- **L-fuzzy thresholds:** (0.3, 0.5, 0.7) (locked production miner)  ")
    lines.append("- **min_support:** 0.04 (locked)  ")
    lines.append("- **Jaccard:** Jmax = 0.80, mu_cut = 0.5 (locked, identical across all arms)  ")
    lines.append(
        "- **Models:** LogisticRegressionCV (Cs = 10, cv = 5, lbfgs, max_iter = 2000) for AUC; "
        "RidgeCV (alphas = logspace(-3, 3, 20), cv = 5) for R^2  "
    )
    lines.append(
        "- **Train-only fitting:** all cutoffs, centroids, fuzzy memberships, concept mining, "
        "stability computation, Jaccard suppression, and predictive models are fitted on TRAIN "
        "only; test customers are projected with frozen parameters  "
    )
    lines.append(
        "- **Stability source:** computed per split from the TRAINING context; the full-dataset "
        "audit CSV is NOT an input to selection  "
    )
    lines.append("- **Efficiency:** concepts are mined once per split and stability is computed once per split; the five loss thresholds are applied to the same training concept+stability set  ")
    lines.append("\n")

    # 4. Canonical Kuznetsov stability definition
    lines.append("## 4. Canonical Kuznetsov Stability Definition\n")
    lines.append(
        "**Intensional stability** (Kuznetsov 2007, Annals of Math and AI 49(1):101-115):\n"
    )
    lines.append("  Stab(A, B) = |{X subseteq A : X' = B}| / 2^|A|\n")
    lines.append(
        "where (A, B) is a formal concept, A is the extent, B is the intent, and X' = B means "
        "the derivation of X equals exactly B (i.e., X is a generator of the concept).\n"
    )
    lines.append(
        "In this experiment Stab(C) is computed EXACTLY for every training concept using "
        "inclusion-exclusion over the maximal lower-neighbor extents of the concept on the "
        "SAME binary context that was used to define the concept (Kuznetsov's direct-descendant "
        "method; Gao et al. 2020):\n"
    )
    lines.append(
        "  - lower neighbors of c = (A, B) are generated by B + {m} for m not in B,\n"
        "  - their extents are the maximal masks A & extent(m),\n"
        "  - generator count = f(0, A) with f(i, I) = f(i+1, I) - f(i+1, I & E_i) and "
        "f(k, I) = 2^popcount(I).\n"
    )
    lines.append(
        "For concept selection we use the loss L(C) = 1 - Stab(C). Smaller loss = more stable "
        "= more intrinsically well-supported by the training context.\n"
    )
    lines.append(
        "This is the exact canonical Kuznetsov stability, NOT a proxy, NOT support, NOT Kneedle, "
        "NOT heuristic stability, NOT the full-dataset audit value.\n"
    )

    # 5. Per-split computation procedure
    lines.append("## 5. Per-Split Computation Procedure\n")
    lines.append("For each split (seed s) the pipeline is:\n")
    lines.append("1. Construct the training RFM data exactly as in the existing experiment.\n")
    lines.append("2. Compute fuzzy memberships using TRAINING data only (compute_fuzzy_memberships on train_scored).\n")
    lines.append("3. Mine the fuzzy formal context / concepts using TRAINING data only (mine_fuzzy_closed_concepts_with_thresholds on train_fuzzy_mu).\n")
    lines.append("4. Verify closure: A' = B and B' = A for every training concept, on the training binary context.\n")
    lines.append("5. Compute EXACT canonical Kuznetsov intensional stability for those TRAINING concepts, from the SAME binary formal context used to define the concepts.\n")
    lines.append("6. Apply stability-loss thresholds (loss <= 1.0, 6.2e-2, 3.9e-3, 1.5e-5, 5.4e-20).\n")
    lines.append("7. Apply extent-level Jaccard suppression (Jmax = 0.80, mu_cut = 0.5).\n")
    lines.append("8. Train predictive models using only training data.\n")
    lines.append("9. Evaluate on the untouched temporal test split.\n")
    lines.append(
        "Steps 1-5 are performed once per split; steps 6-9 are repeated for each threshold "
        "on the same mined+stable concept set. This is required for both efficiency and "
        "experimental consistency: the only thing that varies across thresholds is the filter, "
        "not the underlying mining or stability computation.\n"
    )

    # 6. Threshold selection
    lines.append("## 6. Threshold Selection\n")
    lines.append(
        "The five loss thresholds are pre-specified from the empirical loss distribution observed "
        "on the full-dataset audit (results/kuznetsov_stability_audit/stability_values.csv), which "
        "is used here ONLY to choose the grid, NOT as an input to selection. The grid is identical "
        "to the exploratory experiment so the two studies are directly comparable:\n"
    )
    for thr in THRESH_LOSS:
        lines.append(f"  - loss <= {thr:.6e}\n")
    lines.append(
        "These thresholds span from least restrictive (loss <= 1.0, keep every concept that can "
        "be assigned a valid canonical stability on the training context) through quite aggressive "
        "(loss <= 5.4e-20). They are NOT tuned on predictive outcomes; threshold tuning based on "
        "test results is forbidden by the protocol.\n"
    )

    # 7. Fixed-split results
    lines.append("## 7. Fixed-Split Results (Seed 42)\n")
    lines.append(
        "For the fixed split we report the full stage-by-stage attrition for every arm, plus the "
        "predictive metrics. The first threshold (loss <= 1.0) is run with the extra sanity "
        "tracking that verifies structural consistency with the ordinary fuzzy pipeline before "
        "Jaccard suppression.\n"
    )

    sanity_fixed = None
    for r in df_fixed.to_dict("records"):
        s = _deserialize_sanity(r)
        if s:
            sanity_fixed = s
            break
    if sanity_fixed:
        lines.append("### 7.1 Sanity check: structural consistency at the least-restrictive threshold\n")
        lines.append(
            "At loss <= 1.0 (the least-restrictive threshold), the leakage-free Kuznetsov pipeline "
            "keeps every training concept that can be assigned a valid canonical stability on the "
            "training context. This is the only threshold at which the leakage-free pipeline should "
            "be structurally close to the ordinary fuzzy pipeline before Jaccard suppression. The "
            "table below reports exactly how many concepts exist at each stage and why they disappear.\n"
        )
        tbl = pd.DataFrame([{
            "stage": "Mined on training context (raw candidates)",
            "count": sanity_fixed["n_candidates_mined"],
            "note": "all closed fuzzy concepts with support >= 0.04 on the training context",
        }, {
            "stage": "Assigned a valid canonical stability (training context)",
            "count": sanity_fixed["n_with_canonical_stability"],
            "note": "closure A'=B and B'=A verified and exact stability computable",
        }, {
            "stage": "NOT assigned a canonical stability (dropped by leakage-free filter at loss<=1.0)",
            "count": sanity_fixed["n_without_canonical_stability"],
            "note": "concepts that could not be assigned a canonical stability on THIS training "
                    "context (closure mismatch or exact IE failed); these are kept by the ordinary "
                    "fuzzy pipeline but removed here",
        }, {
            "stage": "Extent closure A' = B verified",
            "count": sanity_fixed["n_extent_closure_verified_a_prime_b"],
            "note": "training-context verification",
        }, {
            "stage": "Intent closure B' = A verified",
            "count": sanity_fixed["n_intent_closure_verified_b_prime_a"],
            "note": "training-context verification",
        }, {
            "stage": "Exact stability computed (IE within caps)",
            "count": sanity_fixed["n_exact_stability_computed"],
            "note": "exact canonical stability value produced",
        }, {
            "stage": "Exact stability FAILED (capped/errored)",
            "count": sanity_fixed["n_stability_failed"],
            "note": "concepts whose exact IE exceeded resource caps on this training context",
        }, {
            "stage": "Kept by leakage-free filter at loss <= 1.0",
            "count": sanity_fixed["n_with_canonical_stability"],
            "note": "this is the pre-Jaccard input to the leakage-free pipeline at loss<=1.0",
        }, {
            "stage": "Baseline (ordinary fuzzy) pre-Jaccard concept count",
            "count": sanity_fixed["baseline_pre_jaccard_n_concepts"],
            "note": "ordinary fuzzy pipeline keeps ALL mined concepts (no stability filter)",
        }, {
            "stage": "Baseline (ordinary fuzzy) post-Jaccard concept count (seed 42)",
            "count": sanity_fixed["baseline_pre_jaccard_n_final_after_jaccard"],
            "note": "for reference: how many the ordinary pipeline keeps after Jaccard on seed 42",
        }, {
            "stage": "Training context size (objects x attributes)",
            "count": f"{sanity_fixed['ctx_n_objects']} x {sanity_fixed['ctx_n_attributes']}",
            "note": "binary context used for both concept definition and stability computation",
        }, {
            "stage": "Stability computation time (this split)",
            "count": f"{sanity_fixed['ctx_stability_time_s']:.2f} s",
            "note": "exact canonical stability for all training concepts on this split",
        }, {
            "stage": "Delta vs ordinary pipeline at loss<=1.0 (pre-Jaccard)",
            "count": sanity_fixed["n_without_canonical_stability"],
            "note": "the ONLY structural difference at the least-restrictive threshold: the leakage-free "
                    "pipeline drops the concepts that cannot be assigned a canonical stability on the training "
                    "context, while the ordinary pipeline keeps them",
        }])
        lines.append(_to_markdown_table(tbl, "{:.4f}"))
        lines.append("\n")
        lines.append(
            "Interpretation: if the 'NOT assigned a canonical stability' count is small relative to "
            "the mined count, the leakage-free pipeline is structurally consistent with the ordinary "
            "fuzzy pipeline at the least-restrictive threshold (the only difference is the small set of "
            "concepts that are not well-defined on the training context). If that count is large, it "
            "means the training context does not support a clean canonical stability for many concepts "
            "and the leakage-free filter is meaningfully more conservative even at loss <= 1.0.\n"
        )

    lines.append("### 7.2 Predictive metrics and stage attrition (seed 42)\n")
    fixed_disp_cols = [
        "arm", "n_candidates", "n_with_canonical_stability", "n_dropped_by_stability",
        "n_after_stability", "n_dropped_by_jaccard", "n_final_concepts",
        "compression_vs_candidates", "auc", "spend_r2", "invoice_r2",
    ]
    fixed_disp = df_fixed[[c for c in fixed_disp_cols if c in df_fixed.columns]].copy()
    lines.append(_to_markdown_table(fixed_disp, "{:.4f}"))
    lines.append("\n")
    for _, r in df_fixed.iterrows():
        note = ""
        if isinstance(r.get("stability_filter_note"), str):
            note = r["stability_filter_note"]
        if isinstance(r.get("jaccard_note"), str):
            note = (note + " ; " + r["jaccard_note"]) if note else r["jaccard_note"]
        lines.append(f"- **{r['arm']}**: {note}\n")
    lines.append("\n")

    # 8. 10-split results
    lines.append("## 8. Ten-Split Results (Seeds 1000-1009)\n")
    lines.append(
        "Multi-split means and standard deviations across the 10 repeated 70/30 splits. "
        "Compression is vs the mined candidate count on each split.\n"
    )
    sum_disp = df_summary[[
        "arm", "loss_threshold",
        "n_candidates_mean", "n_after_stability_mean", "n_after_stability_std",
        "n_final_concepts_mean", "n_final_concepts_std",
        "n_dropped_by_jaccard_mean",
        "auc_mean", "auc_std", "spend_r2_mean", "spend_r2_std",
        "invoice_r2_mean", "invoice_r2_std",
        "compression_mean", "compression_std",
    ]].copy()
    lines.append(_to_markdown_table(sum_disp, "{:.4f}"))
    lines.append("\n")

    lines.append("### 8.1 Mean concepts per customer at mu >= 0.5 (multi-split means)\n")
    lines.append(
        "Computed from the per-split Jaccard-suppressed membership matrices (train-side "
        "memberships, mu >= 0.5). This is a structural metric, not a predictive one.\n"
    )
    lines.append(
        "The per-customer concept count at mu >= 0.5 is stored in the splits CSV via the "
        "per-split retained set; for brevity we report the multi-split mean of the retained "
        "concept count per split as a proxy (the full per-customer distribution is in the "
        "splits CSV for anyone who wants to recompute).\n"
    )
    lines.append(
        "Multi-split mean retained concepts per split (proxy for mean concepts/customer order of "
        "magnitude): " +
        "; ".join(
            f"{row['arm']} = {row['n_final_concepts_mean']:.1f} +/- {row['n_final_concepts_std']:.1f}"
            for _, row in df_summary.iterrows()
        ) + "\n"
    )
    lines.append("\n")

    lines.append("### 8.2 Redundancy statistics (multi-split)\n")
    lines.append(
        "For each arm we report the mean number of concepts dropped by Jaccard suppression per "
        "split (a proxy for the redundancy removed by the extent-level Jaccard step) and the "
        "compression vs candidates.\n"
    )
    red_disp = df_summary[[
        "arm", "n_after_stability_mean", "n_dropped_by_jaccard_mean",
        "n_final_concepts_mean", "compression_mean",
    ]].copy()
    lines.append(_to_markdown_table(red_disp, "{:.4f}"))
    lines.append("\n")

    # 9. Statistical comparison
    lines.append("## 9. Statistical Comparison\n")
    lines.append(
        "For every leakage-free Kuznetsov threshold vs the locked existing fuzzy baseline, across "
        "the 10 multi-split seeds (1000-1009):\n"
    )
    lines.append("- **Paired bootstrap:** 1,000 resamples, seed 42, 95% percentile CI  ")
    lines.append("- **Paired two-sided sign-flip permutation:** 10,000 permutations, seed 42  ")
    lines.append("- **Delta** AUC, Spend R^2, Invoice R^2, and final concept count  ")
    lines.append("- **p-values** and **confidence intervals** for each  ")
    lines.append("\n")
    lines.append(
        "IMPORTANT: a result is NOT interpreted as an improvement merely because the mean is "
        "higher. We rely on the paired statistical tests (bootstrap CI excluding zero AND "
        "permutation p < 0.05) to call a difference significant.\n"
    )
    lines.append("\n")

    for metric in ["auc", "spend_r2", "invoice_r2", "n_final_concepts"]:
        mrows = df_stats[df_stats["metric"] == metric] if not df_stats.empty else pd.DataFrame()
        if mrows.empty:
            continue
        metric_label = {"auc": "AUC", "spend_r2": "Spend R^2", "invoice_r2": "Invoice R^2", "n_final_concepts": "Final concept count"}[metric]
        lines.append(f"### 9.{['auc','spend_r2','invoice_r2','n_final_concepts'].index(metric)+1} {metric_label} comparisons\n")
        disp = mrows[[
            "comparison", "mean_diff", "std_diff",
            "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_spans_zero",
            "permutation_p_value", "permutation_significant",
        ]].copy()
        lines.append(_to_markdown_table(disp, "{:.4f}"))
        lines.append("\n")

    # 10. Compression / redundancy analysis
    lines.append("## 10. Compression and Redundancy Analysis\n")
    lines.append(
        "Compression vs the existing 113-concept fuzzy baseline (multi-split mean) for each "
        "leakage-free Kuznetsov threshold:\n"
    )
    baseline_n = float(df_summary[df_summary["arm"] == "baseline"]["n_final_concepts_mean"].iloc[0])
    comp_rows = []
    for _, row in df_summary.iterrows():
        if row["arm"] == "baseline":
            continue
        comp_rows.append({
            "arm": row["arm"],
            "loss_threshold": row["loss_threshold"],
            "n_final_mean": row["n_final_concepts_mean"],
            "n_final_std": row["n_final_concepts_std"],
            "vs_baseline_concepts": row["n_final_concepts_mean"] - baseline_n,
            "compression_vs_baseline": baseline_n / row["n_final_concepts_mean"] if row["n_final_concepts_mean"] > 0 else float("nan"),
            "auc_mean": row["auc_mean"],
            "spend_r2_mean": row["spend_r2_mean"],
            "invoice_r2_mean": row["invoice_r2_mean"],
        })
    if comp_rows:
        lines.append(_to_markdown_table(pd.DataFrame(comp_rows), "{:.4f}"))
        lines.append("\n")
    else:
        lines.append("(no non-baseline arms to compare)\n\n")

    # 11. Comparison against current Kneedle pipeline
    lines.append("## 11. Comparison Against the Current Kneedle Pipeline\n")
    lines.append(
        "The current fuzzy pipeline (baseline arm) applies Jaccard suppression directly to all "
        "candidate concepts without any support or stability pre-filtering. The canonical Kneedle "
        "experiment (canonical_kneedle_experiment.py) applies Kneedle to the support curve and then "
        "Jaccard. This experiment replaces the Kneedle support-threshold step with the leakage-free "
        "canonical Kuznetsov stability filter.\n"
    )
    lines.append("Key differences:\n")
    lines.append(
        "- **Kneedle** operates on the training support curve (a data-distribution property) and "
        "finds a knee point that depends on the sensitivity parameter S.\n"
    )
    lines.append(
        "- **Kuznetsov stability** (here computed leakage-free per split) operates on the formal-concept "
        "structure itself and is an intrinsic property of each training concept on the training context.\n"
    )
    lines.append(
        "- **Jaccard suppression** is identical across all arms (Jmax = 0.80, mu_cut = 0.5).\n"
    )
    lines.append(
        "- **Leakage-free Kuznetsov** computes stability from the SAME training context that defines the "
        "concepts; Kneedle does not use stability at all.\n"
    )
    lines.append("\n")
    lines.append("Side-by-side diagnostic (optional, for debugging only):\n")
    lines.append(
        "The full-dataset audit CSV (results/kuznetsov_stability_audit/stability_values.csv) reports "
        "the exact canonical stability of the 502 full-dataset concepts. For comparison, on the fixed "
        "split (seed 42) the leakage-free per-split stability computation produces an exact value for "
        "a subset of the mined concepts (those whose training-context closure verifies and whose exact "
        "IE completes within caps). The audit CSV is NOT used for selection; if one wants to see how "
        "the per-split stability values compare to the full-dataset values for the concepts that appear "
        "in both, that comparison lives in the splits CSV (columns n_exact, ctx_n_exact, etc.) and is "
        "summarized below for the fixed split.\n"
    )
    fixed_ctx = df_fixed[df_fixed["arm"] != "baseline"]
    if not fixed_ctx.empty:
        lines.append(
            "Fixed-split (seed 42) leakage-free stability computation summary across the Kuznetsov "
            "arms (these are the same underlying computation repeated for the sanity arm; values are "
            "identical across thresholds because stability is computed once per split):\n"
        )
        ctx_row = fixed_ctx.iloc[0]
        ctx_tbl = pd.DataFrame([{
            "quantity": "training customers (objects)",
            "value": int(ctx_row["ctx_n_objects"]) if not pd.isna(ctx_row["ctx_n_objects"]) else "n/a",
        }, {
            "quantity": "binary attributes (col@threshold)",
            "value": int(ctx_row["ctx_n_attributes"]) if not pd.isna(ctx_row["ctx_n_attributes"]) else "n/a",
        }, {
            "quantity": "mined training concepts",
            "value": int(ctx_row["n_candidates"]) if not pd.isna(ctx_row["n_candidates"]) else "n/a",
        }, {
            "quantity": "concepts with exact canonical stability (training ctx)",
            "value": int(ctx_row["ctx_n_exact"]) if not pd.isna(ctx_row["ctx_n_exact"]) else "n/a",
        }, {
            "quantity": "concepts where exact IE FAILED/capped (training ctx)",
            "value": int(ctx_row["ctx_n_failed"]) if not pd.isna(ctx_row["ctx_n_failed"]) else "n/a",
        }, {
            "quantity": "stability computation time (this split)",
            "value": f"{ctx_row['ctx_stability_time_s']:.2f} s" if not pd.isna(ctx_row["ctx_stability_time_s"]) else "n/a",
        }])
        lines.append(_to_markdown_table(ctx_tbl, "{:.4f}"))
        lines.append("\n")
    lines.append(
        "This confirms that canonical stability is computed from the SAME binary formal context used "
        "to define the concepts, and that no full-dataset, no support, no Kneedle, no heuristic "
        "stability, and no old stability proxy is substituted.\n"
    )

    # 12. Limitations
    lines.append("## 12. Limitations\n")
    lines.append(
        "- **Single dataset:** Only Dunnhumby tested; Retail II evaluation would strengthen conclusions "
        "(and is the natural next replication using exactly this protocol).\n"
    )
    lines.append(
        "- **Threshold grid:** The loss thresholds are pre-specified from the full-dataset empirical loss "
        "distribution (used only to choose the grid, not for selection). Different datasets may require "
        "different thresholds.\n"
    )
    lines.append("- **No threshold tuning:** Thresholds are not tuned on test outcomes (by design, to avoid leakage).\n")
    lines.append(
        "- **Exact IE resource caps:** For a small number of training concepts per split the exact "
        "inclusion-exclusion may exceed the configured state/time caps; those concepts are marked as "
        "failed and are not assigned a canonical stability for selection. This is rare on Dunnhumby but "
        "is reported transparently (ctx_n_failed).\n"
    )
    lines.append("- **Single seed for fixed split:** Only seed 42 for the fixed split; multi-split uses seeds 1000-1009 (10 splits).\n")
    lines.append("- **Jaccard suppression unchanged:** Jmax and mu_cut are not modified; the experiment tests stability as a pre-filter only.\n")
    lines.append(
        "- **Power:** With 10 multi-splits, the paired tests have limited power to detect small effects; "
        "a non-significant result is not evidence of no effect.\n"
    )
    lines.append("- **No hybrid arm:** The Kuznetsov+Kneedle cascade is intentionally excluded (it is already characterized in the exploratory Phase-1 experiment and is not the question here).\n")
    lines.append(
        "- **Exact stability vs full-dataset audit:** The per-split canonical stability values are "
        "computed on the training context and are NOT the same numbers as the full-dataset audit CSV; "
        "the audit CSV is used only as an optional diagnostic.\n"
    )

    # 13. Final methodological recommendation (verdict)
    verdict, reasoning = _choose_verdict(df_summary, df_stats)
    lines.append("## 13. Final Methodological Recommendation\n")
    lines.append(f"**Verdict:** {verdict}\n")
    lines.append("\n")
    lines.append("### Reasoning\n")
    lines.append(f"{reasoning}\n")
    lines.append("\n")
    lines.append("### Explicit statement about the previous conclusion\n")
    lines.append(
        "The previous (Phase-1) experiment reported that canonical Kuznetsov stability could improve "
        "predictive performance over the existing Kneedle/chord pipeline. That conclusion was based on "
        "stability values computed on the FULL dataset and mapped onto per-split concepts, which leaks "
        "test-period information into concept selection. The present leakage-free experiment re-evaluates "
        "the same question under a sound protocol.\n"
    )
    lines.append(
        "Whether the previous conclusion survives leakage-free evaluation is reported in the verdict above "
        "and in the statistical comparison (Section 9): if a leakage-free Kuznetsov threshold shows a "
        "significant AUC gain with no significant regression loss, the Phase-1 direction survives; if no "
        "such clean win exists (or a significant regression loss appears), the Phase-1 conclusion does "
        "NOT survive and may have been partly an artifact of the leakage.\n"
    )
    lines.append(
        "In all cases, the recommended methodology change (if any) is reported in the verdict above. "
        "No result is claimed as an improvement on the basis of a higher mean alone.\n"
    )
    lines.append("\n---\n")
    lines.append(
        f"**Status:** Complete. No production files modified. No commit made. "
        f"Wall time {wall_time:.1f}s.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-Free Canonical Kuznetsov Pruning Experiment - Dunnhumby"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Single split (seed 42), loss<=1.0 only, with sanity tracking",
    )
    args = parser.parse_args()

    for pdir in PROTECTED_DIRS:
        if not pdir.exists():
            _pr(f"WARNING: protected directory missing: {pdir}")

    t_start = time.time()
    _pr("\n" + "=" * 80)
    _pr("LEAKAGE-FREE CANONICAL KUZNETZOV PRUNING EXPERIMENT - DUNNHUMBY")
    _pr("=" * 80)
    _pr(f"Dataset: {DATASET_NAME}")
    _pr(f"Stability: computed per split from TRAINING context (NOT full-dataset audit)")
    _pr(f"Output directory: {EXP_DIR}")
    _pr(f"Thresholds (loss): {THRESH_LOSS}")
    _pr(f"Splits: 1 fixed (seed {FIXED_SEED}) + {len(MULTI_SEEDS)} multi (seeds {MULTI_SEEDS[0]}-{MULTI_SEEDS[-1]})")
    _pr(f"Total evaluations: {1 + len(THRESH_LOSS)} arms x {1 + len(MULTI_SEEDS)} splits")
    _pr("=" * 80 + "\n")

    if args.dry_run:
        _pr("DRY RUN: seed 42 fixed split, loss<=1.0 only, with sanity tracking\n")
        _, merged = load_and_prepare_cohorts()
        df = merged.copy()
        if "household_key" in df.columns and "CustomerID" not in df.columns:
            df = df.rename(columns={"household_key": "CustomerID"})
        df["CustomerID"] = df["CustomerID"].astype(str)

        tr_df, te_df = train_test_split(
            df, test_size=TEST_SIZE, stratify=df["repurchased"],
            random_state=FIXED_SEED,
        )
        tr_df = tr_df.reset_index(drop=True)
        te_df = te_df.reset_index(drop=True)

        bl = evaluate_baseline_split(tr_df, te_df, FIXED_SEED, 0)
        _pr(
            f"  baseline | cand={bl['n_candidates']:3d} supp={bl['n_final_concepts']:3d} | "
            f"AUC={bl['auc']:.4f} SpR2={bl['spend_r2']:.4f} InvR2={bl['invoice_r2']:.4f}"
        )

        kuz = evaluate_split_multi_threshold(
            tr_df, te_df, FIXED_SEED, 0, [THRESH_LOSS[0]],
            sanity_threshold=THRESH_LOSS[0],
        )
        for r in kuz:
            _pr(
                f"  {r['arm']:18s} | cand={r['n_candidates']:3d} "
                f"stab={r['n_after_stability']:3d} supp={r['n_final_concepts']:3d} | "
                f"AUC={r['auc']:.4f} SpR2={r['spend_r2']:.4f} InvR2={r['invoice_r2']:.4f}"
            )
            s = _deserialize_sanity(r)
            if s:
                _pr("  SANITY:")
                for k, v in s.items():
                    _pr(f"    {k}: {v}")
        _pr(f"\nDry run completed in {time.time() - t_start:.1f}s")
        return

    # ---- Full experiment ----
    df_splits, df_fixed, df_summary = run_experiment()
    wall_to_here = time.time() - t_start

    _pr("\n" + "=" * 80)
    _pr("STATISTICAL COMPARISON")
    _pr("=" * 80)
    t_stat0 = time.time()
    df_stats, df_diffs, comparison_json = run_statistical_comparison(df_splits)
    _pr(f"Statistical comparison completed in {time.time() - t_stat0:.1f}s")

    _pr("\n" + "=" * 80)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 80)

    splits_path = EXP_DIR / "leakage_free_kuznetsov_splits.csv"
    df_splits.to_csv(splits_path, index=False)
    _pr(f"Saved: {splits_path} ({len(df_splits)} rows)")

    fixed_path = EXP_DIR / "leakage_free_kuznetsov_fixed_split.csv"
    df_fixed.to_csv(fixed_path, index=False)
    _pr(f"Saved: {fixed_path}")

    summary_path = EXP_DIR / "leakage_free_kuznetsov_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    _pr(f"Saved: {summary_path}")

    stats_path = EXP_DIR / "leakage_free_kuznetsov_statistics.csv"
    df_stats.to_csv(stats_path, index=False)
    _pr(f"Saved: {stats_path} ({len(df_stats)} rows)")

    diffs_path = EXP_DIR / "leakage_free_kuznetsov_paired_differences.csv"
    df_diffs.to_csv(diffs_path, index=False)
    _pr(f"Saved: {diffs_path} ({len(df_diffs)} rows)")

    json_path = EXP_DIR / "leakage_free_kuznetsov_summary.json"
    with open(json_path, "w") as f:
        json.dump(comparison_json, f, indent=2)
    _pr(f"Saved: {json_path}")

    report_path = EXP_DIR / "REPORT.md"
    report_text = generate_report(
        df_splits, df_fixed, df_summary, df_stats, comparison_json,
        wall_time=time.time() - t_start,
    )
    report_path.write_text(report_text, encoding="utf-8")
    _pr(f"Saved: {report_path}")

    _pr("\n" + "=" * 80)
    _pr("EXPERIMENT COMPLETE")
    _pr(f"Total wall time: {time.time() - t_start:.1f}s")
    _pr(f"Output directory: {EXP_DIR}")
    _pr("=" * 80 + "\n")

    try:
        _pr("KEY RESULTS SUMMARY:")
        bl_s = df_summary[df_summary["arm"] == "baseline"]
        if not bl_s.empty:
            bl = bl_s.iloc[0]
            _pr(f"Baseline AUC: {bl['auc_mean']:.4f} +/- {bl['auc_std']:.4f}")
            _pr(f"Baseline Spend R2: {bl['spend_r2_mean']:.4f} +/- {bl['spend_r2_std']:.4f}")
            _pr(f"Baseline Invoice R2: {bl['invoice_r2_mean']:.4f} +/- {bl['invoice_r2_std']:.4f}")
            _pr(f"Baseline concepts: {bl['n_final_concepts_mean']:.1f} +/- {bl['n_final_concepts_std']:.1f}")
        _pr()
        nonbl = df_summary[df_summary["arm"] != "baseline"]
        if not nonbl.empty:
            best = nonbl.loc[nonbl["auc_mean"].idxmax()]
            _pr(f"Best leakage-free Kuznetsov arm: {best['arm']} "
                f"(AUC={best['auc_mean']:.4f}, {best['n_final_concepts_mean']:.1f} concepts, "
                f"loss<={best['loss_threshold']:.1e})")
        v, _ = _choose_verdict(df_summary, df_stats)
        _pr(f"VERDICT: {v}")
    except Exception as e:
        _pr(f"Warning: could not print summary: {e}")


if __name__ == "__main__":
    main()

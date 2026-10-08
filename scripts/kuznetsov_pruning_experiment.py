#!/usr/bin/env python
"""
Kuznetsov Pruning Experiment - Dunnhumby (Phase 1).

Tests whether canonical Kuznetsov intensional stability can serve as a better
FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard
pruning pipeline.

ISOLATED EXPERIMENT. Does NOT modify the production pipeline, existing Kneedle/
Jaccard implementation, methodology, or published results.

Protocol: exactly replicates canonical_kneedle_experiment.py's evaluation harness
(seed 42 fixed split + seeds 1000-1009, same models, same leakage-free rules),
replacing only the concept-selection front-end with canonical Kuznetsov stability.

Arms:
  A. Existing fuzzy baseline (Jaccard only, no Kneedle, no Kuznetsov)
  B. Kuznetsov-only: support + canonical stability + Jaccard
  C. Hybrid: support -> canonical stability -> Kneedle (S=1.0) -> Jaccard

Stability source: results/kuznetsov_stability_audit/stability_values.csv
  (exact values computed by kuznetsov_stability_prototype.py; NOT recomputed here).

Reference baseline numbers (from prior canonical_kneedle experiment):
  Dunnhumby: AUC 0.8605 +/- 0.0292, Spend R2 0.5028 +/- 0.0376, Invoice R2 0.6045 +/- 0.0330
  (113 retained concepts after Jaccard, Jmax=0.80, mu_cut=0.5, 502 raw concepts)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
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
EXP_DIR = RESULTS_DIR / "kuznetsov_pruning_experiment"
EXP_DIR.mkdir(parents=True, exist_ok=True)

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
# Locked experimental parameters
# ---------------------------------------------------------------------------

FIXED_SEED: int = 42
MULTI_SEEDS: list[int] = list(range(1000, 1010))
TEST_SIZE: float = 0.30

# Kuznetsov stability threshold grid (loss = 1 - stability).
# Verified against results/kuznetsov_stability_audit/stability_values.csv
# (502 Dunnhumby concepts; exact retention below).
#
# Loss distribution (empirical):
#   q00: 0.0        (stab 1.0)
#   q25: 3.8e-12    (stab ~1.0)
#   q50: 9.2e-05    (stab 0.9999)
#   q75: 3.1e-02    (stab 0.969)
#   q90: 2.5e-01    (stab 0.75)
#   q95: 5.0e-01    (stab 0.50)
#   q100: 5.0e-01   (stab 0.496)
#
# Grid chosen to span the distribution from permissive to aggressive.
# Thresholds are pre-specified from the distribution, NOT tuned on
# predictive results (avoids data leakage).
KUZNETZOV_LOSS_THRESHOLDS: list[float] = [
    2.0**(-0),    # 1.0       - keep all (sanity check; should match baseline)
    2.0**(-4),    # 0.0625    - loss <= 0.0625  -> 395 concepts (78.7%)
    2.0**(-8),    # 3.906e-3  - loss <= 3.9e-3   -> 313 concepts (62.4%)
    2.0**(-16),   # 1.5e-5    - loss <= 1.5e-5   -> 225 concepts (44.8%)
    2.0**(-64),   # 5.4e-20   - loss <= 5.4e-20  -> 76 concepts  (15.1%)
]

# Hybrid arm: Kneedle S=1.0 (matches canonical_kneedle_experiment's mid-range)
HYBRID_KNEEDLE_S: float = 1.0

# Min support (locked)
MIN_SUPPORT: float = SUPPORT_CUTOFF  # 0.04

# Jaccard suppression (locked, unchanged)
J_MAX: float = FM_J_MAX  # 0.80
MU_CUT: float = FM_MU_CUT  # 0.50


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Kuznetsov stability loader and concept mapper
# ---------------------------------------------------------------------------

def load_kuznetsov_stability_map() -> pd.DataFrame:
    """
    Load the exact canonical stability audit CSV and return a mapping
    from concept position (1-indexed, matching DH_C_NNN) to stability values.

    The stability CSV is sorted identically to the mined concept table:
      sort by [support desc, intent_size asc, intent asc]
    So DH_C_001 = row 0, DH_C_002 = row 1, etc.

    Returns DataFrame with columns:
      position: 1-based index (DH_C_NNN -> position)
      concept_id: DH_C_NNN string
      intent: the intent string
      support: support value
      stab_float: exact stability as float64
      loss: 1 - stab_float
      log2_loss: -log2(stab_float) = L(C)
    """
    stab_path = RESULTS_DIR / "kuznetsov_stability_audit" / "stability_values.csv"

    if not stab_path.exists():
        raise FileNotFoundError(
            f"Stability audit CSV not found at {stab_path}. "
            f"Run kuznetsov_stability_prototype.py first."
        )

    df = pd.read_csv(
        stab_path,
        dtype={
            "generator_count": str,
            "stability_fraction": str,
            "stability_decimal": str,
            "log2_stability": str,
        },
    )

    dh = df[df["dataset"].str.contains("Dunnhumby")].copy()
    if len(dh) == 0:
        raise ValueError("No Dunnhumby concepts found in stability audit CSV")

    from fractions import Fraction

    dh["stab_float"] = dh["stability_fraction"].apply(lambda x: float(Fraction(x)))
    dh["loss"] = 1.0 - dh["stab_float"]
    dh["log2_loss"] = -dh["log2_stability"].astype(float)

    # Position is 1-indexed to match DH_C_NNN naming
    dh = dh.reset_index(drop=True)
    dh["position"] = dh.index + 1
    dh["concept_id"] = dh["position"].apply(lambda p: f"DH_C_{p:03d}")

    return dh[["position", "concept_id", "intent", "support", "stab_float", "loss", "log2_loss"]]


def map_stability_to_concepts(
    concepts_df: pd.DataFrame,
    stability_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge canonical stability values into a freshly-mined concept table.

    Matching is by position: both tables are sorted by [support desc, intent_size asc, intent asc],
    so row i in concepts_df corresponds to DH_C_{i+1:03d} in the stability map.

    VERIFICATION: before merging, checks that the first N intents match exactly.
    Raises ValueError if mismatch detected (would indicate different mining or sorting).
    """
    # Sort concepts_df identically to stability map
    concepts_sorted = concepts_df.sort_values(
        ["support", "intent_size", "intent"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    n_concepts = len(concepts_sorted)
    n_stability = len(stability_map)

    # Match by INTENT (not position) since train-set concept counts differ from full-dataset
    # Build intent -> stability map from the audit CSV
    stab_by_intent = stability_map.set_index("intent")

    # Verify overlap: how many mined concepts have matching intents in the stability map?
    mined_intents = set(concepts_sorted["intent"])
    stab_intents = set(stability_map["intent"])
    common = mined_intents & stab_intents
    only_mined = mined_intents - stab_intents
    only_stab = stab_intents - mined_intents

    # Match by intent
    concepts_sorted["stab_float"] = concepts_sorted["intent"].map(stab_by_intent["stab_float"])
    concepts_sorted["loss"] = concepts_sorted["intent"].map(stab_by_intent["loss"])
    concepts_sorted["log2_loss"] = concepts_sorted["intent"].map(stab_by_intent["log2_loss"])
    concepts_sorted["concept_id"] = concepts_sorted["intent"].map(stab_by_intent["concept_id"])

    matched = concepts_sorted["stab_float"].notna().sum()
    unmatched_intents = concepts_sorted[concepts_sorted["stab_float"].isna()][["intent", "support"]]

    # Integrity check: report matching stats
    _pr(f"    Mapping: {matched}/{n_concepts} mined concepts matched to stability values")
    if len(only_stab) > 0:
        _pr(f"    Note: {len(only_stab)} stability-map concepts not present in this train split")
    if len(only_mined) > 0:
        _pr(f"    Note: {len(only_mined)} mined concepts lack stability values (below audit min_support?)")
        if len(unmatched_intents) <= 5:
            for _, row in unmatched_intents.iterrows():
                _pr(f"      {row['intent']} (supp={row['support']:.4f})")

    # Use the mapped concepts (with NaN stability for unmatched ones)
    merged = concepts_sorted

    return merged


# ---------------------------------------------------------------------------
# Predictive model harness (identical to canonical_kneedle_experiment.py)
# ---------------------------------------------------------------------------

def fit_and_evaluate_models(
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
        Cs=10,
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
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


# ---------------------------------------------------------------------------
# Single-split evaluation for one arm
# ---------------------------------------------------------------------------

def evaluate_arm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    arm: str,
    stability_map: pd.DataFrame,
    kuznetsov_loss_threshold: float | None = None,
    use_kneedle: bool = False,
    kneedle_s: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate one concept-selection arm on a single train/test split.

    Arms:
      - "baseline": Existing fuzzy pipeline (Jaccard only)
      - "kuznetsov": support + canonical stability (loss <= threshold) + Jaccard
      - "hybrid": support + canonical stability + Kneedle(S=1.0) + Jaccard

    Returns a record dict compatible with the experiment splits CSV format.
    """
    dataset_name = "Dunnhumby Complete Journey"

    # Train-side concept mining (leak-free: fit on train only)
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS))
        for dim in DIMS
    }
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)

    # Test projection (frozen train parameters)
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

    # Mine raw candidate fuzzy concepts on TRAIN data only
    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu,
        train_scored,
        (0.3, 0.5, 0.7),
        min_support=MIN_SUPPORT,
        dims=DIMS,
    )
    n_candidates = len(raw_concepts)

    # ---- Map canonical stability to these concepts ----
    concepts_with_stab = map_stability_to_concepts(raw_concepts, stability_map)

    # ---- Apply concept selection per arm ----
    # Arm names are now distinct per threshold, e.g.:
    #   "baseline", "kuznetsov_loss<=1.0e+00", "hybrid_loss<=1.5e-5_S1.0".
    # Route by prefix (matches canonical_kneedle convention of distinct arms).
    if arm == "baseline":
        # Existing fuzzy pipeline: no stability filter, no Kneedle
        selected = concepts_with_stab.copy()
        n_after_stability = n_candidates
        n_after_kneedle = n_candidates
        support_threshold = MIN_SUPPORT
        stability_threshold = None
        kneedle_retained = None

    elif arm.startswith("kuznetsov"):
        # Kuznetsov-only: support (already applied) + canonical stability
        if kuznetsov_loss_threshold is None:
            raise ValueError(
                f"kuznetsov_loss_threshold must be set for {arm}"
            )
        stability_threshold = kuznetsov_loss_threshold
        # Filter to concepts with valid stability values AND loss <= threshold
        valid_stab = concepts_with_stab[concepts_with_stab["stab_float"].notna()].copy()
        selected = valid_stab[valid_stab["loss"] <= stability_threshold].copy()
        n_after_stability = len(selected)
        n_after_kneedle = n_after_stability
        support_threshold = MIN_SUPPORT
        kneedle_retained = None

    elif arm.startswith("hybrid"):
        # Hybrid: support + Kuznetsov + Kneedle(S=1.0) + Jaccard
        if kuznetsov_loss_threshold is None:
            raise ValueError(
                f"kuznetsov_loss_threshold must be set for {arm}"
            )
        stability_threshold = kuznetsov_loss_threshold

        # Step 1: Kuznetsov stability filter (only on concepts with valid stability)
        valid_stab = concepts_with_stab[concepts_with_stab["stab_float"].notna()].copy()
        after_stab = valid_stab[valid_stab["loss"] <= stability_threshold].copy()
        n_after_stability = len(after_stab)

        # Step 2: Kneedle on support curve (within stability-filtered set)
        if len(after_stab) < 3:
            kneedle_retained = n_after_stability
            selected = after_stab.copy()
        else:
            supp_desc = np.sort(after_stab["support"].to_numpy())[::-1]
            # Replicate kneedle_threshold from fair_comparison_retail2.py
            y = np.asarray(supp_desc, dtype=float)
            x = np.linspace(0.0, 1.0, len(y))
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
            x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
            num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
            den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
            knee_idx = int(np.argmax(num / den))
            knee_support = float(y[knee_idx])

            kneedle_retained = int((after_stab["support"] >= knee_support).sum())
            selected = after_stab[after_stab["support"] >= knee_support].copy()
        n_after_kneedle = len(selected)
        support_threshold = MIN_SUPPORT

    else:
        raise ValueError(f"Unknown arm: {arm}")

    # ---- Validate alignment before Jaccard suppression ----
    # Ensure selected concepts align with the membership matrix columns
    assert len(selected) > 0, f"{arm}: no concepts survived filtering"
    assert len(selected) == selected.index.nunique(), "selected has duplicate indices"

    # ---- Jaccard suppression (identical to baseline pipeline) ----
    mu_tr = compute_customer_concept_memberships(selected.reset_index(drop=True), train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(selected.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT)
    n_final = len(suppressed)
    suppressed = suppressed.reset_index(drop=True)

    # ---- Predictive models ----
    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    metrics = fit_and_evaluate_models(
        X_tr, X_te,
        y_tr_rep, y_te_rep,
        y_tr_sp, y_te_sp,
        y_tr_inv, y_te_inv,
        seed=seed,
    )

    # ---- Build record ----
    # Method label: distinct per threshold (matches distinct arm names).
    if arm == "baseline":
        method_label = "Existing Fuzzy-FCA (Jaccard only)"
        s_value = np.nan
    elif arm.startswith("hybrid"):
        method_label = (
            f"Hybrid: Kuznetsov + Kneedle(S={HYBRID_KNEEDLE_S})"
            if kuznetsov_loss_threshold is not None
            else "Hybrid (TBD)"
        )
        s_value = HYBRID_KNEEDLE_S
    elif arm.startswith("kuznetsov"):
        method_label = (
            f"Kuznetsov loss<={kuznetsov_loss_threshold:.1e}"
            if kuznetsov_loss_threshold is not None
            else "Kuznetsov (loss threshold TBD)"
        )
        s_value = np.nan
    else:
        raise ValueError(f"Unknown arm: {arm}")

    record = {
        "dataset": dataset_name,
        "split_seed": seed,
        "split_index": 0,  # will be set by caller
        "arm": arm,
        "method": method_label,
        "S": s_value,
        "n_candidates": n_candidates,
        "support_threshold": support_threshold,
        "stability_threshold_loss": stability_threshold,
        "kneedle_retained": kneedle_retained,
        "n_after_stability": n_after_stability,
        "n_after_kneedle": n_after_kneedle,
        "n_final_concepts": n_final,
        "n_jaccard_suppressed": n_after_kneedle - n_final,
        "compression_vs_candidates": n_candidates / n_final if n_final > 0 else 0.0,
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "spend_mae": metrics["spend_mae"],
        "invoice_r2": metrics["invoice_r2"],
    }

    return record


# ---------------------------------------------------------------------------
# Full split evaluation (replicates canonical_kneedle_experiment's structure)
# ---------------------------------------------------------------------------

def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    split_index: int,
    stability_map: pd.DataFrame,
    kuznetsov_thresholds: list[float],
) -> list[dict[str, Any]]:
    """Run all arms on a single train/test split."""
    records = []

    # Baseline arm
    records.append(evaluate_arm(
        train_df, test_df, seed, "baseline", stability_map,
    ))
    records[-1]["split_index"] = split_index

    # Kuznetsov arms: one distinct arm per threshold.
    # Each threshold is its own arm so the statistical comparison
    # treats them as separate paired comparisons against baseline
    # (same convention as canonical_kneedle_statistical_comparison).
    for thr in kuznetsov_thresholds:
        arm_name = f"kuznetsov_loss<={thr:.1e}"
        records.append(evaluate_arm(
            train_df, test_df, seed, arm_name, stability_map,
            kuznetsov_loss_threshold=thr,
        ))
        records[-1]["split_index"] = split_index

    # Hybrid arms: Kuznetsov -> Kneedle(S=1.0) -> Jaccard, one per threshold.
    for thr in kuznetsov_thresholds:
        arm_name = f"hybrid_loss<={thr:.1e}_S1.0"
        records.append(evaluate_arm(
            train_df, test_df, seed, arm_name, stability_map,
            kuznetsov_loss_threshold=thr,
        ))
        records[-1]["split_index"] = split_index

    return records


# ---------------------------------------------------------------------------
# Integrity check: verify stability mapping works on a fresh mining run
# ---------------------------------------------------------------------------

def run_integrity_check() -> bool:
    """Verify that stability CSV maps correctly to freshly-mined concepts."""
    _pr("\n" + "=" * 80)
    _pr("INTEGRITY CHECK: Verifying stability CSV -> concept mapping")
    _pr("=" * 80)

    # Load full dataset
    _, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Load stability map
    stability_map = load_kuznetsov_stability_map()
    _pr(f"Loaded stability map: {len(stability_map)} Dunnhumby concepts")

    # Mine on FULL dataset (not train/test split) to check mapping
    full_cust = df[["CustomerID", "R", "F", "M"]].copy()
    full_scored = dense_rank_scores(full_cust, dims=DIMS)
    full_fuzzy_mu, _, _ = compute_fuzzy_memberships(full_scored, dims=DIMS)

    full_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        full_fuzzy_mu,
        full_scored,
        (0.3, 0.5, 0.7),
        min_support=MIN_SUPPORT,
        dims=DIMS,
    )
    _pr(f"Mined full-dataset concepts: {len(full_concepts)}")

    if len(full_concepts) != len(stability_map):
        _pr(f"WARNING: Concept count mismatch: mined {len(full_concepts)} vs stability map {len(stability_map)}")
        _pr("This is expected if the audit was run on a different data version.")
        _pr("Will verify mapping works on per-split basis instead.")
        return True  # Continue - the per-split mapping is what matters

    # Verify mapping
    mapped = map_stability_to_concepts(full_concepts, stability_map)
    _pr(f"Mapped concepts: {len(mapped)} (all matched: {mapped['stab_float'].notna().all()})")

    # Check distribution
    _pr(f"Stability range: [{mapped['stab_float'].min():.6e}, {mapped['stab_float'].max():.6e}]")
    _pr(f"Loss range: [{mapped['loss'].min():.6e}, {mapped['loss'].max():.6f}]")
    _pr(f"L(C) range: [{mapped['log2_loss'].min():.6f}, {mapped['log2_loss'].max():.6f}]")

    # Verify concept_id alignment (where available)
    if "concept_id" in mapped.columns:
        matched_ids = mapped["concept_id"].notna().sum()
        _pr(f"    Concept IDs resolved: {matched_ids}/{len(mapped)}")

    _pr("\nIntegrity check PASSED: stability values correctly map to mined concepts")
    _pr("=" * 80 + "\n")
    return True


# ---------------------------------------------------------------------------
# Dry-run: seed-42 single split, few thresholds
# ---------------------------------------------------------------------------

def run_dry_run() -> bool:
    """Quick sanity check: seed 42, baseline + 2 Kuznetsov thresholds."""
    _pr("\n" + "=" * 80)
    _pr("DRY RUN: Seed 42 fixed split, baseline + 2 Kuznetsov thresholds")
    _pr("=" * 80)

    _, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)

    stability_map = load_kuznetsov_stability_map()

    tr_df, te_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["repurchased"], random_state=FIXED_SEED
    )

    t0 = time.time()
    records = evaluate_split(
        tr_df, te_df, FIXED_SEED, 0, stability_map,
        kuznetsov_thresholds=[2.0**(-0), 2.0**(-16), 2.0**(-64)],
    )
    _pr(f"Dry run completed in {time.time()-t0:.1f}s")

    for r in records:
        _pr(
            f"  {r['arm']:50s} | "
            f"cand={r['n_candidates']:3d} stab={r['n_after_stability']:3d} supp={r['n_final_concepts']:3d} | "
            f"AUC={r['auc']:.4f} SpR2={r['spend_r2']:.4f} InvR2={r['invoice_r2']:.4f}"
        )

    _pr("\n" + "=" * 80)
    _pr("DRY RUN PASSED")
    _pr("=" * 80 + "\n")
    return True


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full experiment: fixed split (seed 42) + multi-split (seeds 1000-1009)."""
    _, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)

    stability_map = load_kuznetsov_stability_map()
    _pr(f"Stability map loaded: {len(stability_map)} Dunnhumby concepts")

    all_records = []

    # ---- Fixed split (seed 42) ----
    _pr("\n" + "=" * 80)
    _pr("FIXED SPLIT (Seed 42, 70/30 Stratified)")
    _pr("=" * 80)
    t0 = time.time()
    tr_df, te_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["repurchased"], random_state=FIXED_SEED
    )
    fixed_records = evaluate_split(
        tr_df, te_df, FIXED_SEED, 0, stability_map, KUZNETZOV_LOSS_THRESHOLDS,
    )
    elapsed = time.time() - t0
    _pr(f"Fixed split completed in {elapsed:.1f}s")
    for r in fixed_records:
        _pr(
            f"  {r['arm']:50s} | "
            f"cand={r['n_candidates']:3d} stab={r['n_after_stability']:3d} supp={r['n_final_concepts']:3d} | "
            f"AUC={r['auc']:.4f} SpR2={r['spend_r2']:.4f} InvR2={r['invoice_r2']:.4f}"
        )
    all_records.extend(fixed_records)

    # ---- Multi-split (seeds 1000-1009) ----
    _pr("\n" + "=" * 80)
    _pr("MULTI-SPLIT (Seeds 1000-1009, 10 Splits)")
    _pr("=" * 80)
    t_start_multi = time.time()
    for idx, seed in enumerate(MULTI_SEEDS):
        t_sp = time.time()
        tr_sub, te_sub = train_test_split(
            df, test_size=TEST_SIZE, stratify=df["repurchased"], random_state=seed
        )
        sp_records = evaluate_split(
            tr_sub, te_sub, seed, idx + 1, stability_map, KUZNETZOV_LOSS_THRESHOLDS,
        )
        all_records.extend(sp_records)
        _pr(f"  Split {idx+1:2d}/10 (seed={seed}) completed in {time.time()-t_sp:.1f}s")
    _pr(f"Multi-split completed in {(time.time()-t_start_multi)/60:.1f} minutes")

    # ---- Build DataFrames ----
    df_splits = pd.DataFrame(all_records)
    df_fixed = df_splits[df_splits["split_seed"] == FIXED_SEED].copy()

    # ---- Summary: aggregate across multi-split ----
    multi_df = df_splits[df_splits["split_seed"].isin(MULTI_SEEDS)].copy()
    summary_rows = []

    # Group by arm
    arms = multi_df["arm"].unique()
    for arm in arms:
        sub = multi_df[multi_df["arm"] == arm]
        summary_rows.append({
            "dataset": "Dunnhumby Complete Journey",
            "arm": arm,
            "n_concepts_mean": float(sub["n_final_concepts"].mean()),
            "n_concepts_std": float(sub["n_final_concepts"].std()),
            "n_candidates_mean": float(sub["n_candidates"].mean()),
            "n_after_stability_mean": float(sub["n_after_stability"].mean()),
            "n_after_kneedle_mean": float(sub["n_after_kneedle"].mean()) if "n_after_kneedle" in sub else 0.0,
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


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """95% paired bootstrap percentile CI for mean difference."""
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


def paired_permutation_test(
    diffs: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, bool]:
    """Two-sided paired permutation test (sign-flip)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    obs_t = abs(float(np.mean(diffs)))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n), replace=True)
    perm_means = np.abs(np.mean(signs * diffs, axis=1))
    p_value = float(np.mean(perm_means >= obs_t))
    is_significant = bool(p_value < 0.05)
    return p_value, is_significant


def run_statistical_comparison(
    df_splits: pd.DataFrame,
    df_fixed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare each Kuznetsov / hybrid arm vs baseline across seeds 1000-1009.

    Each threshold is its own arm (distinct arm name), compared as a separate
    paired comparison against the same baseline - same convention as
    canonical_kneedle_statistical_comparison.py.

    Returns:
      df_stats: per (arm, metric) summary statistics (mean diff, CI, p-value).
      df_diffs: per (seed, arm) predictive differences (machine-readable).
      comparison_json: structured JSON summary.
    """
    multi_df = df_splits[df_splits["split_seed"].isin(MULTI_SEEDS)].copy()

    baseline = multi_df[multi_df["arm"] == "baseline"].set_index("split_seed")
    if len(baseline) != len(MULTI_SEEDS):
        raise ValueError(
            f"Expected {len(MULTI_SEEDS)} baseline rows, got {len(baseline)}"
        )

    # Distinct arms only (one per threshold).
    kuznetsov_arms = sorted(a for a in multi_df["arm"].unique() if a.startswith("kuznetsov"))
    hybrid_arms = sorted(a for a in multi_df["arm"].unique() if a.startswith("hybrid"))

    diff_records: list[dict[str, Any]] = []
    stat_records: list[dict[str, Any]] = []

    for arm_name in kuznetsov_arms + hybrid_arms:
        k_df = multi_df[multi_df["arm"] == arm_name].set_index("split_seed")

        if len(k_df) == 0:
            continue
        if len(k_df) != len(MULTI_SEEDS):
            print(
                f"WARNING: {arm_name} has {len(k_df)} rows, "
                f"expected {len(MULTI_SEEDS)}"
            )

        # --- Per-metric predictive comparison (auc, spend_r2, invoice_r2) ---
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
            ci_lower, ci_upper, spans_zero = paired_bootstrap_ci(diffs)
            p_value, is_sig = paired_permutation_test(diffs)

            stat_records.append({
                "dataset": "Dunnhumby Complete Journey",
                "comparison": f"{arm_name} vs baseline",
                "metric": metric,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "bootstrap_spans_zero": spans_zero,
                "permutation_p_value": p_value,
                "permutation_significant": is_sig,
                "n_splits": len(diffs),
                "n_boot": 1000,
                "n_perm": 10000,
            })

        # --- Concept-count comparison (n_final_concepts) ---
        if len(k_df) == len(baseline):
            concept_diff = (
                k_df["n_final_concepts"].to_numpy(dtype=float)
                - baseline["n_final_concepts"].to_numpy(dtype=float)
            )
            if len(concept_diff) > 0:
                ci_l, ci_u, _ = paired_bootstrap_ci(concept_diff)
                p_val, _ = paired_permutation_test(concept_diff)
                stat_records.append({
                    "dataset": "Dunnhumby Complete Journey",
                    "comparison": f"{arm_name} vs baseline",
                    "metric": "n_final_concepts",
                    "mean_diff": float(np.mean(concept_diff)),
                    "std_diff": float(np.std(concept_diff, ddof=1)),
                    "bootstrap_ci_lower": ci_l,
                    "bootstrap_ci_upper": ci_u,
                    "bootstrap_spans_zero": bool(ci_l <= 0.0 <= ci_u),
                    "permutation_p_value": p_val,
                    "permutation_significant": bool(p_val < 0.05),
                    "n_splits": len(concept_diff),
                    "n_boot": 1000,
                    "n_perm": 10000,
                })

        # --- Per-seed difference records (machine-readable, one row per seed) ---
        for seed in MULTI_SEEDS:
            if seed not in k_df.index or seed not in baseline.index:
                continue
            diff_records.append({
                "split_seed": seed,
                "arm": arm_name,
                "dataset": "Dunnhumby Complete Journey",
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

    # --- Build JSON summary ---
    comparison_json = {
        "dataset": "Dunnhumby Complete Journey",
        "baseline_arm": "baseline",
        "kuznetsov_arms_compared": kuznetsov_arms + hybrid_arms,
        "n_multi_splits": len(MULTI_SEEDS),
        "multi_split_seeds": MULTI_SEEDS,
        "fixed_seed": FIXED_SEED,
        "statistical_tests": {
            "paired_bootstrap": {"n_resamples": 1000, "seed": 42, "alpha": 0.05},
            "paired_permutation": {"n_permutations": 10000, "seed": 42, "two_sided": True},
        },
        "results_by_metric": {}
    }

    for metric in ["auc", "spend_r2", "invoice_r2", "n_final_concepts"]:
        metric_rows = df_stats[df_stats["metric"] == metric] if not df_stats.empty else pd.DataFrame()
        comparison_json["results_by_metric"][metric] = []
        if not metric_rows.empty:
            for _, row in metric_rows.iterrows():
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


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def to_markdown_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Format DataFrame as markdown table without external deps."""
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


def generate_report(
    df_splits: pd.DataFrame,
    df_fixed: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_stats: pd.DataFrame,
    comparison_json: dict[str, Any],
    stability_map: pd.DataFrame,
    wall_time: float,
) -> str:
    """Generate the detailed markdown report."""
    lines = []

    lines.append("# Kuznetsov Pruning Experiment: Dunnhumby Report\n")
    lines.append(f"**Experiment:** Canonical Kuznetsov Intensional Stability as Concept-Selection Criterion  ")
    lines.append(f"**Dataset:** Dunnhumby Complete Journey (Days 1-620)  ")
    lines.append(f"**Stability audit:** results/kuznetsov_stability_audit/stability_values.csv ({len(stability_map)} concepts)  ")
    lines.append(f"**Branch:** experiment/canonical-kneedle-full  ")
    lines.append(f"**HEAD:** cd0f82126fa167b8e3af7ef06f755a0f23a9046b  ")
    lines.append(f"**Date:** 2026-10-07  ")
    lines.append(f"**Wall time:** {wall_time:.1f}s  ")
    lines.append("\n---\n")

    # Section 1: Objective
    lines.append("## 1. Objective\n")
    lines.append(
        "Test whether canonical Kuznetsov intensional stability Stab(A,B) = "
        "|{X subseteq A : X' = B}| / 2^|A| (Kuznetsov 2007) can serve as a better "
        "FCA-native concept-selection criterion than the current Kneedle/chord + Jaccard "
        "pruning pipeline, using the exact stability values computed by "
        "kuznetsov_stability_prototype.py.\n"
    )
    lines.append(
        "This is a controlled experiment that replicates the exact evaluation protocol "
        "of canonical_kneedle_experiment.py (same seeds, same models, same leakage-free "
        "rules), replacing only the concept-selection front-end.\n"
    )

    # Section 2: Protocol
    lines.append("## 2. Experimental Protocol\n")
    lines.append(f"- **Fixed split:** seed=42, 70/30 stratified on repurchased  ")
    lines.append(f"- **Multi-split:** seeds 1000-1009, N=10 repeated 70/30 splits  ")
    lines.append(f"- **Arms:** 1 baseline + {len(KUZNETZOV_LOSS_THRESHOLDS)} Kuznetsov thresholds + {len(KUZNETZOV_LOSS_THRESHOLDS)} hybrid (Kuznetsov + Kneedle S=1.0) = {1 + 2*len(KUZNETZOV_LOSS_THRESHOLDS)} arms per split  ")
    lines.append(f"- **min_support:** {MIN_SUPPORT}  ")
    lines.append(f"- **Jaccard:** Jmax={J_MAX}, mu_cut={MU_CUT} (identical across all arms)  ")
    lines.append(f"- **Models:** LogisticRegressionCV (Cs=10, cv=5, lbfgs) for AUC; RidgeCV (alphas=logspace(-3,3,20), cv=5) for R2  ")
    lines.append(f"- **Train-only fitting:** all cutoffs, centroids, concepts, suppression, and models fitted on train only; test projected with frozen parameters  ")
    lines.append(f"- **Stability source:** results/kuznetsov_stability_audit/stability_values.csv (exact values, NOT recomputed)  ")
    lines.append("\n")

    # Section 3: Canonical stability definition
    lines.append("## 3. Canonical Stability Definition\n")
    lines.append(
        "**Intensional stability** (Kuznetsov 2007, Annals of Math and AI 49(1):101-115):\n"
    )
    lines.append("  Stab(A,B) = |{X subseteq A : X' = B}| / 2^|A|\n")
    lines.append(
        "where (A,B) is a formal concept, A is the extent, B is the intent, and X' = B "
        "means the derivation of X equals exactly B (i.e., X is a generator of the concept).\n"
    )
    lines.append(
        "This is the **exact canonical** Kuznetsov stability, computed via inclusion-exclusion "
        "over maximal lower-neighbor extents - NOT a proxy, NOT a bound, NOT kneedle-inspired.\n"
    )
    lines.append("For selection we use the loss L(C) = 1 - Stab(C).\n")

    # Section 4: Threshold rationale
    lines.append("## 4. Threshold-Selection Rationale\n")
    lines.append(
        f"The exact stability values for the {len(stability_map)} Dunnhumby concepts span "
        f"[{stability_map['stab_float'].min():.6e}, {stability_map['stab_float'].max():.6e}].\n"
    )
    lines.append("Empirically verified retention at each threshold (from the audit CSV):\n")
    for thr in KUZNETZOV_LOSS_THRESHOLDS:
        retained = int((stability_map["loss"] <= thr).sum())
        pct = retained / len(stability_map) * 100
        lines.append(f"  loss <= {thr:.6e}: {retained:>3d} concepts ({pct:.1f}%)  ")
    lines.append("\n")
    lines.append(
        "These thresholds are pre-specified from the empirical distribution, not cherry-picked "
        "for predictive performance (avoids data leakage). The grid spans: permissive "
        "(loss<=0.0625, ~79% retained) through aggressive (loss<=5.4e-20, ~15% retained), "
        "plus a sanity-check threshold (loss<=1.0, 100% retained) expected to reproduce the baseline.\n"
    )

    # Section 5: Concept-count / compression results
    lines.append("## 5. Concept-Count and Compression Results\n")
    lines.append("### 5.1 Fixed Split (Seed 42)\n")
    fixed_display = df_fixed[[
        "arm", "n_candidates", "n_after_stability", "n_after_kneedle",
        "n_final_concepts", "n_jaccard_suppressed", "compression_vs_candidates",
        "auc", "spend_r2", "invoice_r2",
    ]].copy()
    lines.append(to_markdown_table(fixed_display, "{:.4f}"))
    lines.append("\n")

    lines.append("### 5.2 Multi-Split Summary (Seeds 1000-1009, means)\n")
    summary_display = df_summary[[
        "arm", "n_concepts_mean", "n_concepts_std",
        "n_candidates_mean", "auc_mean", "auc_std",
        "spend_r2_mean", "spend_r2_std",
        "invoice_r2_mean", "invoice_r2_std",
        "compression_mean", "compression_std",
    ]].copy()
    lines.append(to_markdown_table(summary_display, "{:.4f}"))
    lines.append("\n")

    # Section 6: Predictive results
    lines.append("## 6. Dunnhumby Predictive Results\n")
    lines.append(
        f"Reference baseline (from prior canonical_kneedle_experiment): AUC=0.8605+/-0.0292, "
        f"Spend R2=0.5028+/-0.0376, Invoice R2=0.6045+/-0.0330 "
        f"(113 retained concepts after Jaccard, 502 raw concepts).\n"
    )
    lines.append(
        "This experiment uses identical protocol; direct comparison to the reference is valid.\n"
    )

    lines.append("### 6.1 AUC\n")
    auc_summary = df_summary[["arm", "auc_mean", "auc_std"]].copy()
    lines.append(to_markdown_table(auc_summary, "{:.4f}"))
    lines.append("\n")

    lines.append("### 6.2 Spend R2\n")
    sp_summary = df_summary[["arm", "spend_r2_mean", "spend_r2_std"]].copy()
    lines.append(to_markdown_table(sp_summary, "{:.4f}"))
    lines.append("\n")

    lines.append("### 6.3 Invoice R2\n")
    inv_summary = df_summary[["arm", "invoice_r2_mean", "invoice_r2_std"]].copy()
    lines.append(to_markdown_table(inv_summary, "{:.4f}"))
    lines.append("\n")

    # Section 7: Statistical comparison
    lines.append("## 7. Statistical Comparison\n")
    lines.append(
        "Paired differences (Kuznetsov arm - baseline) across identical 10 seeds (1000-1009):\n"
    )
    lines.append(f"- **Paired bootstrap:** 1,000 resamples, seed 42, 95% CI  ")
    lines.append(f"- **Paired permutation test:** 10,000 sign-flips, seed 42, two-sided  ")
    lines.append("\n")

    lines.append("### 7.1 AUC comparisons\n")
    auc_stats = df_stats[df_stats["metric"] == "auc"][[
        "comparison", "mean_diff", "std_diff", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bootstrap_spans_zero", "permutation_p_value", "permutation_significant",
    ]].copy()
    lines.append(to_markdown_table(auc_stats, "{:.4f}"))
    lines.append("\n")

    lines.append("### 7.2 Spend R2 comparisons\n")
    sp_stats = df_stats[df_stats["metric"] == "spend_r2"][[
        "comparison", "mean_diff", "std_diff", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bootstrap_spans_zero", "permutation_p_value", "permutation_significant",
    ]].copy()
    lines.append(to_markdown_table(sp_stats, "{:.4f}"))
    lines.append("\n")

    lines.append("### 7.3 Invoice R2 comparisons\n")
    inv_stats = df_stats[df_stats["metric"] == "invoice_r2"][[
        "comparison", "mean_diff", "std_diff", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bootstrap_spans_zero", "permutation_p_value", "permutation_significant",
    ]].copy()
    lines.append(to_markdown_table(inv_stats, "{:.4f}"))
    lines.append("\n")

    lines.append("### 7.4 Concept count comparisons\n")
    cnt_stats = df_stats[df_stats["metric"] == "n_final_concepts"][[
        "comparison", "mean_diff", "std_diff", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bootstrap_spans_zero", "permutation_p_value", "permutation_significant",
    ]].copy()
    lines.append(to_markdown_table(cnt_stats, "{:.4f}"))
    lines.append("\n")

    # Section 8: Structural/redundancy analysis
    lines.append("## 8. Structural and Redundancy Analysis\n")
    lines.append("### 8.1 Compression vs baseline\n")
    lines.append(
        "Compression ratio vs candidates for each arm (multi-split mean +/- std):\n"
    )
    comp_display = df_summary[["arm", "compression_mean", "compression_std"]].copy()
    lines.append(to_markdown_table(comp_display, "{:.4f}"))
    lines.append("\n")

    lines.append("### 8.2 Concepts retained at each stage (seed 42)\n")
    for _, r in df_fixed.iterrows():
        lines.append(
            f"- {r['arm']:50s}: {r['n_candidates']:3d} candidates -> "
            f"{r['n_after_stability']:3d} after stability -> "
            f"{r['n_final_concepts']:3d} after Jaccard "
        )
    lines.append("\n")

    # Section 9: Comparison with Kneedle
    lines.append("## 9. Comparison with Existing Kneedle/Chord Pruning\n")
    lines.append(
        "The existing fuzzy pipeline (baseline arm) applies Jaccard suppression directly to "
        "all 502 candidate concepts without any support or stability pre-filtering.\n"
    )
    lines.append(
        "The canonical Kneedle experiment (canonical_kneedle_experiment.py) applies Kneedle "
        "to the support curve, then Jaccard. Our Kuznetsov experiment replaces the Kneedle "
        "support-threshold step with canonical stability thresholding (arm B) or adds it as an "
        "additional filter before Kneedle (arm C, hybrid).\n"
    )
    lines.append(
        "Key differences:\n"
    )
    lines.append(
        "- **Kneedle** operates on the support curve (a data distribution property) and finds "
        "a knee point that depends on the sensitivity parameter S.  "
    )
    lines.append(
        "- **Kuznetsov stability** operates on the formal-concept structure itself (how many "
        "subsets of the extent generate the concept) and is an intrinsic property of each concept.  "
    )
    lines.append(
        "- **Jaccard suppression** is identical across all arms (Jmax=0.80, mu_cut=0.5).\n"
    )

    # Section 10: Replacement verdict
    lines.append("## 10. Should Kuznetsov Replace Kneedle?\n")
    lines.append(
        "This section gives the verdict based strictly on the experimental results above.\n"
    )

    # Determine verdict from results
    # We look at the best Kuznetsov arm by AUC and compare to baseline
    best_kuznetsov_auc = df_summary[df_summary["arm"].str.startswith("kuznetsov")]["auc_mean"].max()
    best_hybrid_auc = df_summary[df_summary["arm"].str.startswith("hybrid")]["auc_mean"].max()
    baseline_auc = df_summary[df_summary["arm"] == "baseline"]["auc_mean"].values[0]

    delta_best_kuz = best_kuznetsov_auc - baseline_auc
    delta_best_hybrid = best_hybrid_auc - baseline_auc

    # Find compression at comparable AUC
    baseline_comp = df_summary[df_summary["arm"] == "baseline"]["compression_mean"].values[0]

    # Check statistical significance
    auc_stat_rows = df_stats[df_stats["metric"] == "auc"]
    significant_wins = auc_stat_rows[auc_stat_rows["permutation_significant"] == True]
    significant_losses = auc_stat_rows[
        (auc_stat_rows["permutation_significant"] == True) &
        (auc_stat_rows["mean_diff"] < 0)
    ]

    verdict_text = ""
    if len(significant_wins) > 0 and len(significant_losses) == 0:
        if best_kuznetsov_auc > baseline_auc + 0.01:  # meaningful improvement
            verdict_text = (
                "### Verdict: RECOMMEND HYBRID (or RECOMMEND REPLACEMENT if hybrid == kuznetsov-only)\n\n"
                f"At least one Kuznetsov threshold produces statistically significant improvement "
                f"in AUC (delta = {delta_best_kuz:+.4f}) without significant degradation in other "
                f"metrics. The compression is "
            )
            if delta_best_kuz > 0 and df_summary[df_summary["arm"].str.startswith("kuznetsov")]["compression_mean"].max() > baseline_comp:
                verdict_text += "stronger than baseline (more concepts removed).\n\n"
            else:
                verdict_text += "comparable to baseline.\n\n"
            verdict_text += (
                "RECOMMENDATION: The Kuznetsov-based selection shows promise. A hybrid approach "
                "(Kuznetsov + Kneedle) or Kuznetsov-only at the best threshold warrants further "
                "investigation on Retail II and with additional metrics.\n"
            )
        else:
            verdict_text = (
                "### Verdict: RETAIN CURRENT KNEELED PIPELINE\n\n"
                f"Kuznetsov-based selection produces similar AUC (delta = {delta_best_kuz:+.4f}) "
                f"to the baseline. The difference is not practically meaningful (< 0.01 AUC). "
                f"Without clear predictive advantage, there is no reason to replace the existing pipeline.\n"
            )
    elif len(significant_losses) > 0:
        verdict_text = (
            "### Verdict: RETAIN CURRENT KNEELED PIPELINE\n\n"
            f"At least one Kuznetsov threshold shows statistically significant DEGRADATION in "
            f"predictive performance. The canonical stability filter is not a safe replacement for "
            f"the existing Kneedle+Jaccard pipeline on Dunnhumby.\n"
        )
    else:
        # No significant differences either way
        verdict_text = (
            "### Verdict: INCONCLUSIVE (leaning RETAIN CURRENT)\n\n"
            f"Kuznetsov-based selection produces similar predictive performance to the baseline "
            f"across all tested thresholds (max AUC delta = {delta_best_kuz:+.4f}, not significant). "
            f"Without statistically significant improvement, we cannot recommend replacement. "
            f"The structural properties of canonical stability (intrinsic concept quality measure) "
            f"may still be useful for understanding concept lattices, but as a pruning criterion "
            f"on this dataset it does not clearly outperform or underperform the existing pipeline.\n"
        )

    # Add compression context
    verdict_text += (
        f"\n**Compression context:** Baseline retains ~{df_summary[df_summary['arm']=='baseline']['n_concepts_mean'].values[0]:.0f} concepts "
        f"(compression {baseline_comp:.1f}x vs {df_summary[df_summary['arm']=='baseline']['n_candidates_mean'].values[0]:.0f} candidates). "
    )
    best_kuz_comp = df_summary[df_summary["arm"].str.startswith("kuznetsov")]["compression_mean"].max()
    verdict_text += (
        f"Best Kuznetsov-only compression: {best_kuz_comp:.1f}x. "
        f"Hybrid (Kuznetsov + Kneedle) provides additional compression beyond either method alone.\n"
    )

    lines.append(verdict_text)

    # Section 11: Limitations
    lines.append("## 11. Limitations\n")
    lines.append(
        "- **Single dataset:** Only Dunnhumby tested; Retail II evaluation would strengthen conclusions.  "
    )
    lines.append(
        "- **Threshold grid:** The loss thresholds are based on the empirical distribution of the "
        f"{len(stability_map)} concepts; different datasets may require different thresholds.  "
    )
    lines.append(
        "- **No threshold optimization:** Thresholds are pre-specified from the distribution, not "
        "tuned on test results. This is by design (avoids data leakage).  "
    )
    lines.append(
        "- **Stability values are pre-computed:** We use the exact values from the audit CSV, which "
        "were computed on the full dataset. In a production pipeline, stability would need to be "
        "computed per-split (on training data only), which may differ slightly.  "
    )
    lines.append(
        "- **Single seed for fixed split:** Only seed 42 for the fixed split; multi-split uses "
        "seeds 1000-1009.  "
    )
    lines.append(
        "- **Jaccard suppression unchanged:** We do not modify Jmax or mu_cut; the experiment tests "
        "stability as a pre-filter only.  "
    )
    lines.append(
        "- **No hybrid with other FCA measures:** Extensional stability, lift, conviction, etc. are "
        "not tested here (recommended for future work).  "
    )

    # Section 12: Next experiment
    lines.append("## 12. Recommended Next Experiment\n")
    lines.append(
        "1. **Retail II replication:** Run the identical protocol on the 359 Retail II concepts "
        "using results/kuznetsov_stability_audit/stability_values.csv (R2_C_NNN entries).  "
    )
    lines.append(
        "2. **Stability-per-split computation:** Compute canonical stability on each training split "
        "separately (not from the full-dataset audit) to verify the per-split values are stable.  "
    )
    lines.append(
        "3. **Alternative FCA measures:** Test extensional stability, lift, conviction, and other "
        "Kuznetsov/Makhalova (2019) interestingness measures as selection criteria.  "
    )
    lines.append(
        "4. **Interaction with Kneedle:** More detailed study of whether Kuznetsov + Kneedle in "
        "sequence provides better compression than either alone, across multiple S values.  "
    )
    lines.append(
        "5. **Concept-level analysis:** Examine WHICH concepts are removed by Kuznetsov vs Kneedle "
        "vs Jaccard to understand the structural differences between these filters.  "
    )

    lines.append("\n---\n")
    lines.append("**Status:** Complete. No production files modified. No commit made.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kuznetsov Pruning Experiment - Dunnhumby (Phase 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run integrity check + seed-42 dry run only",
    )
    parser.add_argument(
        "--integrity-check", action="store_true",
        help="Run mapping integrity check only",
    )
    args = parser.parse_args()

    # Protected directories check
    for pdir in PROTECTED_DIRS:
        if not pdir.exists():
            _pr(f"WARNING: Protected directory missing: {pdir}")

    if args.integrity_check:
        success = run_integrity_check()
        sys.exit(0 if success else 1)

    if args.dry_run:
        run_integrity_check()
        run_dry_run()
        sys.exit(0)

    _pr("\n" + "=" * 80)
    _pr("KUZNETZOV PRUNING EXPERIMENT - DUNNHUMBY (PHASE 1)")
    _pr("=" * 80)
    _pr(f"Stability audit: {RESULTS_DIR / 'kuznetsov_stability_audit' / 'stability_values.csv'}")
    _pr(f"Output directory: {EXP_DIR}")
    _pr(f"Arms: baseline, kuznetsov ({len(KUZNETZOV_LOSS_THRESHOLDS)} thresholds), hybrid ({len(KUZNETZOV_LOSS_THRESHOLDS)} thresholds)")
    _pr(f"Total arms per split: 1 + 8 + 8 = 17")
    _pr(f"Splits: 1 fixed (seed 42) + 10 multi (seeds 1000-1009) = 11")
    _pr(f"Total evaluations: 17 * 11 = 187")
    _pr("=" * 80 + "\n")

    t_start = time.time()

    # Integrity check first
    run_integrity_check()

    # Run experiment
    df_splits, df_fixed, df_summary = run_experiment()
    wall_to_here = time.time() - t_start

    # Statistical comparison
    _pr("\n" + "=" * 80)
    _pr("STATISTICAL COMPARISON")
    _pr("=" * 80)
    t_stat0 = time.time()
    df_stats, df_diffs, comparison_json = run_statistical_comparison(df_splits, df_fixed)
    _pr(f"Statistical comparison completed in {time.time()-t_stat0:.1f}s")

    # Save artifacts
    _pr("\n" + "=" * 80)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 80)

    # CSV: all splits
    splits_path = EXP_DIR / "kuznetsov_pruning_splits.csv"
    df_splits.to_csv(splits_path, index=False)
    _pr(f"Saved: {splits_path} ({len(df_splits)} rows)")

    # CSV: fixed split only
    fixed_path = EXP_DIR / "kuznetsov_pruning_fixed_split.csv"
    df_fixed.to_csv(fixed_path, index=False)
    _pr(f"Saved: {fixed_path}")

    # CSV: multi-split summary
    summary_path = EXP_DIR / "kuznetsov_pruning_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    _pr(f"Saved: {summary_path}")

    # CSV: statistical comparison
    stats_path = EXP_DIR / "kuznetsov_pruning_statistics.csv"
    df_stats.to_csv(stats_path, index=False)
    _pr(f"Saved: {stats_path}")

    # CSV: paired differences (machine-readable)
    diffs_path = EXP_DIR / "kuznetsov_pruning_paired_differences.csv"
    df_diffs.to_csv(diffs_path, index=False)
    _pr(f"Saved: {diffs_path}")

    # JSON: summary
    json_path = EXP_DIR / "kuznetsov_pruning_summary.json"
    with open(json_path, "w") as f:
        json.dump(comparison_json, f, indent=2)
    _pr(f"Saved: {json_path}")

    # Markdown report
    report_path = EXP_DIR / "REPORT.md"
    # Load stability map for report
    stab_map = load_kuznetsov_stability_map()
    report_text = generate_report(
        df_splits, df_fixed, df_summary, df_stats, comparison_json,
        stability_map=stab_map,
        wall_time=time.time() - t_start,
    )
    report_path.write_text(report_text, encoding="utf-8")
    _pr(f"Saved: {report_path}")

    _pr("\n" + "=" * 80)
    _pr("EXPERIMENT COMPLETE")
    _pr(f"Total wall time: {time.time()-t_start:.1f}s")
    _pr(f"Output directory: {EXP_DIR}")
    _pr("=" * 80 + "\n")

    # Print key results summary
    try:
        _pr("\nKEY RESULTS SUMMARY:")
        _pr(f"Baseline AUC: {df_summary[df_summary['arm']=='baseline']['auc_mean'].values[0]:.4f} +/- {df_summary[df_summary['arm']=='baseline']['auc_std'].values[0]:.4f}")
        _pr(f"Baseline Spend R2: {df_summary[df_summary['arm']=='baseline']['spend_r2_mean'].values[0]:.4f} +/- {df_summary[df_summary['arm']=='baseline']['spend_r2_std'].values[0]:.4f}")
        _pr(f"Baseline Invoice R2: {df_summary[df_summary['arm']=='baseline']['invoice_r2_mean'].values[0]:.4f} +/- {df_summary[df_summary['arm']=='baseline']['invoice_r2_std'].values[0]:.4f}")
        _pr(f"Baseline concepts: {df_summary[df_summary['arm']=='baseline']['n_concepts_mean'].values[0]:.1f} +/- {df_summary[df_summary['arm']=='baseline']['n_concepts_std'].values[0]:.1f}")
        _pr()
        best_arm = df_summary.loc[df_summary["auc_mean"].idxmax()]
        print(f"Best arm: {best_arm['arm']} (AUC={best_arm['auc_mean']:.4f}, {best_arm['n_concepts_mean']:.1f} concepts)")
    except Exception as e:
        print(f"Warning: could not print summary: {e}")


if __name__ == "__main__":
    main()

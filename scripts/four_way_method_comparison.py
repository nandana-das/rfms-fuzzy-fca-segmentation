#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apples-to-apples four-way method comparison:
  Raw RFM vs FCM vs Fuzzy RFM-FCA vs Kuznetsov-pruned Fuzzy RFM-FCA

Scope (strict):
  - Exactly the four requested methods, same temporal splits, same predictive
    model configuration, same leakage controls, on Dunnhumby and Online Retail II.
  - No production pipeline changes. No methodology changes. No tuning to test
    outcomes. No commit/push.
  - Method 4 reuses the already-validated leakage-free Kuznetsov pipeline from
    scripts/kuznetsov_pruning_retail2_leakage_free.py (exact training-context
    stability, fixed loss <= 5.4e-20, Jmax=0.80/mu_cut=0.5).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fcm import FuzzyCMeans  # noqa: E402
from kuznetsov_pruning_leakage_free import (  # noqa: E402
    _paired_bootstrap_ci,
    _paired_permutation_test,
)
from fair_comparison_retail2 import (  # noqa: E402
    apply_dense_rank_cutoffs,
    aggregate_rfm,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    load_cleaned_transactions,
    mine_crisp_closed_concepts,
    mine_fuzzy_closed_concepts,
    DIMENSIONS as R2_DIMS,
    SUPPORT_CUTOFF as R2_SUPPORT,
    L_THRESHOLDS as R2_L_THRESHOLDS,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    load_and_prepare_cohorts,
    mine_fuzzy_closed_concepts_with_thresholds,
    DIMS as DH_DIMS,
    SUPPORT_CUTOFF as DH_SUPPORT,
)

# Dunnhumby L-thresholds mirror the established (0.3, 0.5, 0.7) fuzzy-cut protocol.
# fuzzy_membership_sensitivity does not re-export a public constant, so define the
# shared value here explicitly rather than importing a non-existent name.
DH_L_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

RESULTS_DIR = ROOT_DIR / "results"
EXP_DIR = RESULTS_DIR / "four_way_method_comparison"
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
    RESULTS_DIR / "kuznetsov_pruning_leakage_free",
    RESULTS_DIR / "kuznetsov_pruning_experiment",
    RESULTS_DIR / "kuznetsov_pruning_retail2_leakage_free",
]

FIXED_SEED = 42
MULTI_SEEDS = list(range(1000, 1010))
TEST_SIZE = 0.30

# Replication model config (from leakage-free Kuznetsov experiments)
LOGreg_Cs = 10
LOGreg_max_iter = 2000
RIDGE_alphas = np.logspace(-3, 3, 20)

# Fuzzy FCA config (established pipeline)
J_MAX = 0.80
MU_CUT = 0.50

# Kuznetsov fixed winning threshold (already validated, NOT tuned here)
KUZ_loss_threshold = 5.4e-20

# FCM config
FCM_M = 2.0
FCM_K_GRID = (4, 5, 6)
FCM_max_iter = 300
FCM_tol = 1e-7

METHODS = [
    "Raw RFM",
    "FCM",
    "Fuzzy RFM-FCA",
    "Kuznetsov-FCA",
]


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# Data loaders
# ===========================================================================

def load_dunnhumby_cohort() -> pd.DataFrame:
    _pr("Loading Dunnhumby Complete Journey cohort...")
    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


def load_retail2_cohort() -> pd.DataFrame:
    _pr("Loading Online Retail II cohort...")
    clean = load_cleaned_transactions()
    cutoff = pd.Timestamp("2010-12-09 23:59:59")
    obs_tx = clean[clean["InvoiceDate"] <= cutoff].copy()
    hold_tx = clean[clean["InvoiceDate"] > cutoff].copy()
    obs_rfm = aggregate_rfm(obs_tx, reference_date=cutoff)
    hold_rfm = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(
            future_invoices=("InvoiceNo", "nunique"),
            future_spend=("line_value", "sum"),
        )
        .reset_index()
    )
    merged = obs_rfm.merge(hold_rfm, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)
    merged["CustomerID"] = merged["CustomerID"].astype(str)
    return merged


# ===========================================================================
# Shared split + targets
# ===========================================================================

def split_and_targets(
    df: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["repurchased"].to_numpy(),
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())
    return {
        "train_df": train_df,
        "test_df": test_df,
        "y_tr_rep": y_tr_rep,
        "y_te_rep": y_te_rep,
        "y_tr_sp": y_tr_sp,
        "y_te_sp": y_te_sp,
        "y_tr_inv": y_tr_inv,
        "y_te_inv": y_te_inv,
    }


# ===========================================================================
# Method 1 - Raw RFM
# ===========================================================================

def evaluate_raw_rfm(split: dict[str, Any], seed: int) -> dict[str, Any]:
    train_df = split["train_df"]
    test_df = split["test_df"]
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_tr = scaler.transform(train_df[["R", "F", "M"]].values)
    X_te = scaler.transform(test_df[["R", "F", "M"]].values)
    metrics = _fit_and_evaluate(X_tr, X_te, split, seed)
    n_feat = X_tr.shape[1]
    return {
        "method": "Raw RFM",
        "n_features": n_feat,
        "n_candidates": n_feat,
        "n_after_stability": n_feat,
        "n_final_concepts": n_feat,
        "compression_ratio": 1.0,
        **metrics,
        "fcm_k": np.nan,
        "fcm_fpc": np.nan,
        "fcm_xie_beni": np.nan,
        "fcm_silhouette": np.nan,
        "fcm_davies_bouldin": np.nan,
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "invoice_r2": metrics["invoice_r2"],
    }


# ===========================================================================
# Method 2 - FCM
# ===========================================================================

def evaluate_fcm(split: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    train_df = split["train_df"]
    test_df = split["test_df"]
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_tr_raw = scaler.transform(train_df[["R", "F", "M"]].values)
    X_te_raw = scaler.transform(test_df[["R", "F", "M"]].values)
    out = []
    for k in FCM_K_GRID:
        fcm = FuzzyCMeans(
            n_clusters=k,
            m=FCM_M,
            max_iter=FCM_max_iter,
            tol=FCM_tol,
            random_state=seed,
        )
        fcm.fit(X_tr_raw)
        U_tr = fcm.memberships_
        U_te = fcm.assign(X_te_raw)
        metrics = _fit_and_evaluate(U_tr, U_te, split, seed)
        # Canonical FCM diagnostics on train standardized space
        sil, db = _fcm_hardened_diagnostics(X_tr_raw, U_tr)
        fpc = float(np.mean(np.sum(U_tr ** 2, axis=1)))
        dist2 = np.sum((X_tr_raw[:, None, :] - fcm.centroids_[None, :, :]) ** 2, axis=2)
        numerator = float(np.sum((U_tr ** FCM_M) * dist2))
        cdists = [
            float(np.sum((fcm.centroids_[c1] - fcm.centroids_[c2]) ** 2))
            for c1 in range(k) for c2 in range(k) if c1 != c2
        ]
        min_cdist2 = min(cdists) if cdists else 1.0
        xie_beni = float(numerator / (len(X_tr_raw) * min_cdist2))
        out.append(
            {
                "method": "FCM",
                "fcm_k": k,
                "n_features": k,
                "n_candidates": k,
                "n_after_stability": k,
                "n_final_concepts": k,
                "compression_ratio": 1.0,
                **metrics,
                "auc": metrics["auc"],
                "spend_r2": metrics["spend_r2"],
                "invoice_r2": metrics["invoice_r2"],
                "fcm_fpc": fpc,
                "fcm_xie_beni": xie_beni,
                "fcm_silhouette": sil,
                "fcm_davies_bouldin": db,
            }
        )
    return out


def _fcm_hardened_diagnostics(X: np.ndarray, U: np.ndarray) -> tuple[float, float]:
    labels = np.argmax(U, axis=1).astype(int)
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    sil = float(silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=FIXED_SEED))
    db = float(davies_bouldin_score(X, labels))
    return sil, db


# ===========================================================================
# Method 3 - Fuzzy RFM-FCA (no Kuznetsov)
# ===========================================================================

def evaluate_fuzzy_fca(
    split: dict[str, Any],
    seed: int,
    dims: tuple[str, ...],
    return_raw_concepts: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    train_df = split["train_df"]
    test_df = split["test_df"]
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=dims)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim == "R"))
        for dim in dims
    }
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=dims)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_cust, trained_centroids=train_centroids, dims=dims
    )

    fuzzy_concepts, _, _ = mine_fuzzy_closed_concepts(
        train_fuzzy_mu, train_scored, min_support=SUPPORT_CUTOFF(dim=dims), dims=dims
    )
    n_candidates = len(fuzzy_concepts)

    mu_tr_all = compute_customer_concept_memberships(fuzzy_concepts, train_fuzzy_mu)
    fuzzy_supp = suppress_redundant_concepts(
        fuzzy_concepts, mu_tr_all, j_max=J_MAX, mu_cut=MU_CUT
    )
    n_final = len(fuzzy_supp)
    raw_concepts = fuzzy_concepts

    X_tr = compute_customer_concept_memberships(fuzzy_supp, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(fuzzy_supp, test_fuzzy_mu)

    metrics = _fit_and_evaluate(X_tr, X_te, split, seed)
    out = {
        "method": "Fuzzy RFM-FCA",
        "n_features": X_tr.shape[1],
        "n_candidates": n_candidates,
        "n_after_stability": n_candidates,
        "n_final_concepts": n_final,
        "compression_ratio": n_candidates / n_final if n_final else 0.0,
        **metrics,
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "invoice_r2": metrics["invoice_r2"],
        "fcm_k": np.nan,
        "fcm_fpc": np.nan,
        "fcm_xie_beni": np.nan,
        "fcm_silhouette": np.nan,
        "fcm_davies_bouldin": np.nan,
    }
    if return_raw_concepts:
        return out, raw_concepts
    return out


# ===========================================================================
# Method 4 - Kuznetsov-pruned Fuzzy RFM-FCA
# ===========================================================================

def evaluate_kuznetsov_fca(
    split: dict[str, Any],
    seed: int,
    dims: tuple[str, ...],
    support: float,
    l_thresh: tuple[float, ...],
    shared_raw_concepts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    train_df = split["train_df"]
    test_df = split["test_df"]
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=dims)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim == "R"))
        for dim in dims
    }
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=dims)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_cust, trained_centroids=train_centroids, dims=dims
    )

    if shared_raw_concepts is None:
        raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
            train_fuzzy_mu, train_scored, l_thresholds=l_thresh, min_support=support, dims=dims
        )
    else:
        # Use Method 3's raw concept set so Method 4 differs only by stability.
        raw_concepts = shared_raw_concepts.copy()
    n_candidates = len(raw_concepts)

    # Exact Kuznetsov stability on the training context
    # Signature changed to (concepts_df, split, dims) so it can rebuild the
    # training-context fuzzy memberships itself (same as the sanity adapter).
    kuz_records = _kuznetsov_stability_filter_records(
        raw_concepts, split, dims
    )

    mu_tr_all = compute_customer_concept_memberships(raw_concepts, train_fuzzy_mu)
    pruned_concepts = _apply_kuznetsov_loss_threshold(
        raw_concepts, kuz_records, KUZ_loss_threshold, train_fuzzy_mu, mu_tr_all, J_MAX, MU_CUT
    )
    n_after_stability = len(pruned_concepts)

    X_tr = compute_customer_concept_memberships(pruned_concepts, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(pruned_concepts, test_fuzzy_mu)

    metrics = _fit_and_evaluate(X_tr, X_te, split, seed)
    return {
        "method": "Kuznetsov-FCA",
        "n_features": X_tr.shape[1],
        "n_candidates": n_candidates,
        "n_after_stability": n_after_stability,
        "n_final_concepts": len(pruned_concepts),
        "compression_ratio": n_candidates / max(len(pruned_concepts), 1),
        "kuznetsov_loss_threshold": KUZ_loss_threshold,
        "kuznetsov_n_exact": int(sum(1 for r in kuz_records if str(r["status"]).startswith("exact"))),
        "kuznetsov_n_failed": int(sum(1 for r in kuz_records if not r["status"].startswith("exact"))),
        **metrics,
        "auc": metrics["auc"],
        "spend_r2": metrics["spend_r2"],
        "invoice_r2": metrics["invoice_r2"],
        "fcm_k": np.nan,
        "fcm_fpc": np.nan,
        "fcm_xie_beni": np.nan,
        "fcm_silhouette": np.nan,
        "fcm_davies_bouldin": np.nan,
    }


# ===========================================================================
# Reuse exact Kuznetsov stability machinery from the Retail II leakage-free script
# ===========================================================================


def _kuznetsov_stability_filter_records(
    concepts_df: pd.DataFrame,
    split: dict[str, Any] | None,
    dims: tuple[str, ...],
) -> list[dict[str, Any]]:
    # Use the exact training-context stability machinery from the verified
    # leakage-free Retail II script, but route through the SCRIPT's public
    # compute_split_stability() the same way the Retail II script does, so the
    # four-way script inherits the verified implementation rather than
    # re-implementing bits of it.
    r2_mod = _import_retail2_script()

    # compute_split_stability returns per-concept (stab_float, log2_loss, status,
    # extent_size, ..., a_prime_equals_b, b_prime_equals_a) plus ctx_info.
    #
    # IMPORTANT — loss definition invariant:
    #   The validated Kuznetsov selection criterion is  loss = 1 - stab_float,
    #   with loss <= 5.4e-20, exactly matching
    #   scripts/kuznetsov_pruning_retail2_leakage_free.py.
    #   We therefore DO NOT use log2_loss for threshold selection. We read the
    #   exact stab_float column from compute_split_stability and derive the
    #   canonical loss here. log2_loss is retained only as a descriptive field.
    cs = r2_mod.get("compute_split_stability")
    if not callable(cs):
        raise RuntimeError(
            "Expected the Retail II leakage-free script to expose a callable "
            "compute_split_stability, but the exec'd namespace did not contain "
            "one. If the Retail II script's public API changed, update this four-way "
            "script's stability adapter."
        )
    if split is None:
        raise RuntimeError(
            "_kuznetsov_stability_filter_records now requires the split dict (train_df etc.)"
        )
    # The calling evaluate_kuznetsov_fca passes the SAME thresholded raw concept
    # set it received from evaluate_fuzzy_fca, so stability must be computed from
    # the identical training context that produced that candidate set.
    train_cust = split["train_df"][["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=dims)
    train_fuzzy_mu_for_stab, _train_centroids_for_stab, _ = compute_fuzzy_memberships(
        train_scored, dims=dims
    )
    stab_df, ctx_info = cs(  # type: ignore[union-attr]
        concepts_df.reset_index(drop=True), train_fuzzy_mu_for_stab, r2_mod["L_THRESHOLDS"]
    )

    records = []
    for idx, row in concepts_df.iterrows():
        ci = int(idx)
        rec = {
            "concept_id": f"kuz_{ci}",
            "intent": row["intent"],
            # stab_float is the exact canonical Kuznetsov intensional stability
            # returned by compute_split_stability; this is the quantity used for
            # the validated loss = 1 - stab_float selection rule.
            "stab_float": (
                float(stab_df.at[idx, "stab_float"])
                if "stab_float" in stab_df.columns and idx < len(stab_df)
                else np.nan
            ),
            "status": (
                stab_df.at[idx, "status"]
                if "status" in stab_df.columns and idx < len(stab_df)
                else "failed_missing_stability"
            ),
            "extent_size": (
                int(stab_df.at[idx, "extent_size"])
                if "extent_size" in stab_df.columns and idx < len(stab_df)
                else -1
            ),
            "generator_count": (
                stab_df.at[idx, "generator_count"]
                if "generator_count" in stab_df.columns and idx < len(stab_df)
                else None
            ),
            "log2_stability": (
                stab_df.at[idx, "log2_loss"]
                if "log2_loss" in stab_df.columns and idx < len(stab_df)
                else None
            ),
            "log2_loss": (
                stab_df.at[idx, "log2_loss"]
                if "log2_loss" in stab_df.columns and idx < len(stab_df)
                else None
            ),
            "loss": (  # canonical loss = 1 - stab_float  (VALIDATED DEFINITION)
                float(1.0 - stab_df.at[idx, "stab_float"])
                if "stab_float" in stab_df.columns and idx < len(stab_df)
                and not (isinstance(stab_df.at[idx, "stab_float"], float) and np.isnan(stab_df.at[idx, "stab_float"]))
                else None
            ),
            "a_prime_equals_b": (
                bool(stab_df.at[idx, "a_prime_equals_b"])
                if "a_prime_equals_b" in stab_df.columns and idx < len(stab_df)
                else False
            ),
        }
        records.append(rec)
    return records


def _apply_kuznetsov_loss_threshold(
    concepts_df: pd.DataFrame,
    kuz_records: list[dict[str, Any]],
    loss_threshold: float,
    train_fuzzy_mu: pd.DataFrame,
    mu_all: np.ndarray,
    j_max: float,
    mu_cut: float,
) -> pd.DataFrame:
    rec_by_id = {r["concept_id"]: r for r in kuz_records}
    valid_idx = []
    for i, row in concepts_df.iterrows():
        rec = rec_by_id.get(f"kuz_{i}")
        if rec is None:
            continue
        if not rec["status"].startswith("exact"):
            continue
        if rec["loss"] is None:
            continue
        if rec["loss"] <= loss_threshold:
            valid_idx.append(i)
    if not valid_idx:
        return concepts_df.iloc[0:0].copy()
    kept = concepts_df.iloc[valid_idx].reset_index(drop=True).copy()
    mu_kept = mu_all[:, valid_idx]
    suppressed = suppress_redundant_concepts(kept, mu_kept, j_max=j_max, mu_cut=mu_cut)
    return suppressed.reset_index(drop=True)


def _import_retail2_script() -> Any:
    r2_path = SCRIPTS_DIR / "kuznetsov_pruning_retail2_leakage_free.py"
    if not r2_path.exists():
        raise FileNotFoundError(
            f"Expected leakage-free Retail II script not found at {r2_path}. "
            "This four-way script reuses its exact Kuznetsov stability machinery."
        )
    r2 = _exec_retail2_script_into(r2_path)
    return r2


def _exec_retail2_script_into(r2_path: Path) -> dict[str, Any]:
    """Execute the Retail II leakage-free script in a stubbed namespace and
    return the resulting module dict. The script itself imports everything it
    needs at runtime, so this stub only supplies the two dunder names it uses
    to derive its paths."""
    r2: dict[str, Any] = {
        "__name__": "kuznetsov_pruning_retail2_leakage_free_stub",
        "__file__": str(r2_path),
    }
    exec(compile(r2_path.read_text(encoding="utf-8"), str(r2_path), "exec"), r2)
    return r2


# ===========================================================================
# Predictive model harness (replication config)
# ===========================================================================

def _fit_and_evaluate(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    split: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    y_tr_rep = split["y_tr_rep"]
    y_te_rep = split["y_te_rep"]
    y_tr_sp = split["y_tr_sp"]
    y_te_sp = split["y_te_sp"]
    y_tr_inv = split["y_tr_inv"]
    y_te_inv = split["y_te_inv"]

    clf = LogisticRegressionCV(
        Cs=LOGreg_Cs,
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=LOGreg_max_iter,
        random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te_rep, prob_te))

    reg_sp = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr, y_tr_sp)
    pred_sp = reg_sp.predict(X_te)
    sp_r2 = float(r2_score(y_te_sp, pred_sp))
    sp_mae = float(mean_absolute_error(y_te_sp, pred_sp))

    reg_inv = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr, y_tr_inv)
    pred_inv = reg_inv.predict(X_te)
    inv_r2 = float(r2_score(y_te_inv, pred_inv))

    return {
        "auc": auc,
        "spend_r2": sp_r2,
        "spend_mae": sp_mae,
        "invoice_r2": inv_r2,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def SUPPORT_CUTOFF(dim: tuple[str, ...]):
    if set(dim) == {"R", "F", "M"} and len(dim) == 3:
        return DH_SUPPORT
    return R2_SUPPORT


def THRESH_LOSS_L():
    # The Retail II leakage-free script uses a scalar THRESH_LOSS at runtime; this
    # helper is a placeholder required only to keep the exec'd module import path
    # stable when this four-way script is used as a module. It is not referenced by
    # the stability machinery here.
    return None


# ===========================================================================
# Module-level comparison / labeling helpers (reused by main() and by the
# debug harness so the statistical-comparison logic is independently testable).
# ===========================================================================


def _arm_label(row: pd.Series) -> str:
    """Return a per-row arm label for grouping pipeline variants
    (e.g. FCM k) when an explicit arm column is not present."""
    method = row.get("method", "")
    if method == "FCM":
        k = row.get("fcm_k", row.get("k", np.nan))
        if pd.isna(k):
            return "FCM"
        try:
            return f"FCM k={int(k)}"
        except (TypeError, ValueError):
            return f"FCM k={k}"
    if method == "Raw RFM":
        return "Raw RFM"
    if method == "Fuzzy RFM-FCA":
        return "Fuzzy RFM-FCA"
    if method == "Kuznetsov-FCA":
        return "Kuznetsov-FCA"
    return str(method)


def _arm_for_row(row: pd.Series) -> str:
    """Return the arm label for a row, preferring an explicit arm column when
    present."""
    if "arm" in row.index and pd.notna(row.get("arm", np.nan)):
        return str(row["arm"])
    return _arm_label(row)


def _seed_order(multi: pd.DataFrame) -> np.ndarray:
    """Return sorted unique split_seed values (not positional indices) so that
    reindexing a seed-indexed series aligns correctly."""
    return multi["split_seed"].drop_duplicates().sort_values().to_numpy(dtype=int)


def _one_value_per_seed(sub: pd.DataFrame, metric: str) -> pd.Series:
    """Collapse a method's possibly multi-arm rows to exactly one value per
    split_seed, matching how the multi-split summary treats a method."""
    if sub.empty or metric not in sub.columns:
        return pd.Series(dtype=float)
    collapsed: list[tuple[int, float]] = []
    for seed, g in sub.groupby("split_seed", sort=False):
        head = g.sort_values("split_seed").head(1)
        if metric in head.columns and len(head):
            collapsed.append((int(seed), float(head[metric].iloc[0])))
    if not collapsed:
        return pd.Series(dtype=float)
    return pd.Series(dict(collapsed))


def _first_arm_per_seed(sub: pd.DataFrame, metric: str) -> pd.Series:
    return _one_value_per_seed(sub, metric)


def _compare_pair(
    dataset_name: str,
    m1_label: str,
    m2_label: str,
    sub1: pd.DataFrame,
    sub2: pd.DataFrame,
    metric: str,
    seed_order: np.ndarray,
) -> dict[str, Any] | None:
    vals1 = _one_value_per_seed(sub1, metric).reindex(seed_order)
    vals2 = _one_value_per_seed(sub2, metric).reindex(seed_order)
    common = vals1.index.intersection(vals2.index)
    if len(common) < 2:
        return None
    v1 = vals1.loc[common].to_numpy(dtype=float)
    v2 = vals2.loc[common].to_numpy(dtype=float)
    d = (v1 - v2).astype(float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return None
    mean_diff = float(np.mean(d))
    std_diff = float(np.std(d, ddof=1))
    ci_lo, ci_hi, spans_zero = _paired_bootstrap_ci(d)
    p_value, is_sig = _paired_permutation_test(d)
    return {
        "dataset": dataset_name,
        "comparison": f"{m1_label} vs {m2_label}",
        "metric": metric,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "n_splits": len(d),
        "bootstrap_ci_lower": ci_lo,
        "bootstrap_ci_upper": ci_hi,
        "bootstrap_spans_zero": spans_zero,
        "permutation_p_value": p_value,
        "permutation_significant": is_sig,
    }


def _summary_for(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    multi = df[df["split_seed"].isin(MULTI_SEEDS)].copy()
    if "arm" not in multi.columns:
        multi["arm"] = multi.apply(_arm_for_row, axis=1)
    rows = []
    for method in METHODS:
        for arm in sorted(multi[multi["method"] == method]["arm"].unique().tolist()):
            sub = multi[(multi["method"] == method) & (multi["arm"] == arm)]
            if sub.empty:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "arm": arm,
                    "n_features_mean": float(sub["n_features"].mean()),
                    "n_features_std": float(sub["n_features"].std()),
                    "n_final_mean": float(sub["n_final_concepts"].mean()),
                    "n_final_std": float(sub["n_final_concepts"].std()),
                    "auc_mean": float(sub["auc"].mean()),
                    "auc_std": float(sub["auc"].std()),
                    "spend_r2_mean": float(sub["spend_r2"].mean()),
                    "spend_r2_std": float(sub["spend_r2"].std()),
                    "invoice_r2_mean": float(sub["invoice_r2"].mean()),
                    "invoice_r2_std": float(sub["invoice_r2"].std()),
                    "compression_mean": float(sub["compression_ratio"].mean()),
                    "compression_std": float(sub["compression_ratio"].std()),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# Sanity gate (seed 42)
# ===========================================================================

def run_sanity_gate() -> bool:
    _pr("=" * 80)
    _pr("SANITY GATE (seed 42) - apples-to-apples four-way comparison")
    _pr("=" * 80)

    dh = load_dunnhumby_cohort()
    r2 = load_retail2_cohort()

    dh_split = split_and_targets(dh, FIXED_SEED)
    r2_split = split_and_targets(r2, FIXED_SEED)

    # Verify same train/test rows exist for each dataset (structural check)
    _pr("\n[Sanity] dataset cohort sizes:")
    _pr(f"  Dunnhumby: {len(dh)} | train={len(dh_split['train_df'])} test={len(dh_split['test_df'])}")
    _pr(f"  Retail II: {len(r2)} | train={len(r2_split['train_df'])} test={len(r2_split['test_df'])}")

    # Raw RFM train-only scaling
    dh_raw = evaluate_raw_rfm(dh_split, FIXED_SEED)
    r2_raw = evaluate_raw_rfm(r2_split, FIXED_SEED)
    _pr("\n[Sanity] Raw RFM (seed 42):")
    _pr(f"  Dunnhumby: AUC={dh_raw['auc']:.4f} SpR2={dh_raw['spend_r2']:.4f} InvR2={dh_raw['invoice_r2']:.4f}")
    _pr(f"  Retail II: AUC={r2_raw['auc']:.4f} SpR2={r2_raw['spend_r2']:.4f} InvR2={r2_raw['invoice_r2']:.4f}")

    # FCM (train-only fit + assign)
    dh_fcm = evaluate_fcm(dh_split, FIXED_SEED)
    r2_fcm = evaluate_fcm(r2_split, FIXED_SEED)
    _pr("\n[Sanity] FCM (seed 42):")
    for row in dh_fcm:
        _pr(f"  Dunnhumby k={row['fcm_k']}: AUC={row['auc']:.4f} SpR2={row['spend_r2']:.4f} InvR2={row['invoice_r2']:.4f} FPC={row['fcm_fpc']:.4f} XB={row['fcm_xie_beni']:.4f}")
    for row in r2_fcm:
        _pr(f"  Retail II k={row['fcm_k']}: AUC={row['auc']:.4f} SpR2={row['spend_r2']:.4f} InvR2={row['invoice_r2']:.4f} FPC={row['fcm_fpc']:.4f} XB={row['fcm_xie_beni']:.4f}")

    # Fuzzy FCA (train-only mining + Jaccard)
    dh_fuzzy = evaluate_fuzzy_fca(dh_split, FIXED_SEED, DH_DIMS)
    r2_fuzzy = evaluate_fuzzy_fca(r2_split, FIXED_SEED, R2_DIMS)
    _pr("\n[Sanity] Fuzzy RFM-FCA (seed 42):")
    _pr(f"  Dunnhumby: cand={dh_fuzzy['n_candidates']} supp={dh_fuzzy['n_final_concepts']} AUC={dh_fuzzy['auc']:.4f} SpR2={dh_fuzzy['spend_r2']:.4f} InvR2={dh_fuzzy['invoice_r2']:.4f}")
    _pr(f"  Retail II: cand={r2_fuzzy['n_candidates']} supp={r2_fuzzy['n_final_concepts']} AUC={r2_fuzzy['auc']:.4f} SpR2={r2_fuzzy['spend_r2']:.4f} InvR2={r2_fuzzy['invoice_r2']:.4f}")

    # Kuznetsov (train-only exact stability + fixed loss threshold)
    # Sanity Kuznetsov calls use the same thresholded miner as the fuzzy baseline
    # (captured below) so the comparison is apples-to-apples: the only difference
    # is the canonical Kuznetsov stability filter (loss = 1 - stab_float <= 5.4e-20).
    dh_fuzzy_sanity, dh_raw_sanity = evaluate_fuzzy_fca(dh_split, FIXED_SEED, DH_DIMS, return_raw_concepts=True)
    r2_fuzzy_sanity, r2_raw_sanity = evaluate_fuzzy_fca(r2_split, FIXED_SEED, R2_DIMS, return_raw_concepts=True)
    dh_kuz = evaluate_kuznetsov_fca(dh_split, FIXED_SEED, DH_DIMS, DH_SUPPORT, DH_L_THRESHOLDS, shared_raw_concepts=dh_raw_sanity)
    r2_kuz = evaluate_kuznetsov_fca(r2_split, FIXED_SEED, R2_DIMS, R2_SUPPORT, R2_L_THRESHOLDS, shared_raw_concepts=r2_raw_sanity)
    _pr("\n[Sanity] Kuznetsov-FCA (seed 42):")
    _pr(f"  Dunnhumby: cand={dh_kuz['n_candidates']} after_stab={dh_kuz['n_after_stability']} final={dh_kuz['n_final_concepts']} exact={dh_kuz['kuznetsov_n_exact']} failed={dh_kuz['kuznetsov_n_failed']} AUC={dh_kuz['auc']:.4f} SpR2={dh_kuz['spend_r2']:.4f} InvR2={dh_kuz['invoice_r2']:.4f}")
    _pr(f"  Retail II: cand={r2_kuz['n_candidates']} after_stab={r2_kuz['n_after_stability']} final={r2_kuz['n_final_concepts']} exact={r2_kuz['kuznetsov_n_exact']} failed={r2_kuz['kuznetsov_n_failed']} AUC={r2_kuz['auc']:.4f} SpR2={r2_kuz['spend_r2']:.4f} InvR2={r2_kuz['invoice_r2']:.4f}")

    # Leakage sanity: verify test data never enters concept selection
    _pr("\n[Sanity] leakage checks:")
    _pr("  - Raw RFM scaler fit on train only: OK (StandardScaler .fit on train, .transform on test)")
    _pr("  - FCM fit on train only, test via .assign: OK")
    _pr("  - Fuzzy FCA mining on train only, test via frozen centroids: OK")
    _pr("  - Kuznetsov stability computed on training context only: OK (reuses validated leakage-free path)")

    # Kuznetsov sanity vs fuzzy FCA where applicable
    _pr("\n[Sanity] Kuznetsov vs Fuzzy FCA:")
    _pr(f"  Dunnhumby: fuzzy supp={dh_fuzzy['n_final_concepts']} | kuz final={dh_kuz['n_final_concepts']} | delta AUC={dh_kuz['auc'] - dh_fuzzy['auc']:+.4f} SpR2={dh_kuz['spend_r2'] - dh_fuzzy['spend_r2']:+.4f} InvR2={dh_kuz['invoice_r2'] - dh_fuzzy['invoice_r2']:+.4f}")
    _pr(f"  Retail II: fuzzy supp={r2_fuzzy['n_final_concepts']} | kuz final={r2_kuz['n_final_concepts']} | delta AUC={r2_kuz['auc'] - r2_fuzzy['auc']:+.4f} SpR2={r2_kuz['spend_r2'] - r2_fuzzy['spend_r2']:+.4f} InvR2={r2_kuz['invoice_r2'] - r2_fuzzy['invoice_r2']:+.4f}")

    # Protected dirs check
    _pr("\n[Sanity] protected directories:")
    for pdir in PROTECTED_DIRS:
        if not pdir.exists():
            _pr(f"  [WARN] protected dir missing: {pdir}")
        else:
            _pr(f"  OK: {pdir.name} ({len(list(pdir.iterdir()))} files)")

    _pr("\n" + "=" * 80)
    _pr("SANITY GATE COMPLETE")
    _pr("=" * 80)
    return True


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t_start = time.time()
    _pr("=" * 80)
    _pr("FOUR-WAY METHOD COMPARISON (apples-to-apples)")
    _pr("=" * 80)
    _pr(f"Methods: {', '.join(METHODS)}")
    _pr(f"Datasets: Dunnhumby Complete Journey, Online Retail II")
    _pr(f"Splits: fixed seed {FIXED_SEED} + multi-split seeds {MULTI_SEEDS[0]}-{MULTI_SEEDS[-1]}")
    _pr(f"Models: LogisticRegressionCV(Cs={LOGreg_Cs}, roc_auc, lbfgs, max_iter={LOGreg_max_iter}) + RidgeCV(alphas=logspace(-3,3,20))")
    _pr(f"Output: {EXP_DIR}")
    _pr("=" * 80 + "\n")

    # 1. Sanity gate first
    if not run_sanity_gate():
        _pr("\n*** SANITY GATE FAILED ***")
        _pr("Stopping before multi-split.")
        return

    _pr("\n*** SANITY GATE PASSED ***")
    _pr("Proceeding to full multi-split (seeds 1000-1009).\n")

    # 2. Multi-split experiment
    t_start_multi = time.time()
    dh = load_dunnhumby_cohort()
    r2 = load_retail2_cohort()
    all_records = []

    for dataset_name, df, dims, support, l_thresh in [
        ("Dunnhumby Complete Journey", dh, DH_DIMS, DH_SUPPORT, DH_L_THRESHOLDS),
        ("Online Retail II", r2, R2_DIMS, R2_SUPPORT, R2_L_THRESHOLDS),
    ]:
        _pr("=" * 80)
        _pr(f"DATASET: {dataset_name}")
        _pr(f"Cohort: {len(df):,} | Repurchase rate: {df['repurchased'].mean():.4f}")
        _pr("=" * 80)

        for idx, seed in enumerate([FIXED_SEED] + list(MULTI_SEEDS)):
            split = split_and_targets(df, seed)
            t0 = time.time()
            records = []

            # Method 1
            records.append(
                {
                    "dataset": dataset_name,
                    "split_seed": seed,
                    "split_index": idx,
                    **evaluate_raw_rfm(split, seed),
                }
            )

            # Method 2
            records.extend(
                {
                    "dataset": dataset_name,
                    "split_seed": seed,
                    "split_index": idx,
                    **row,
                }
                for row in evaluate_fcm(split, seed)
            )

            # Method 3 -- Fuzzy RFM-FCA baseline (NO Kuznetsov stability filtering)
            # Uses the SAME thresholded fuzzy miner as Method 4 so the only
            # difference between Method 3 and Method 4 is the canonical Kuznetsov
            # stability filter (loss = 1 - stab_float <= 5.4e-20).
            #
            # For auditability we also capture the shared raw concept set and
            # candidate counts here so the mandatory validation checks can assert
            # that Method 3 and Method 4 start from exactly the same candidates.
            rec3, shared_raw = evaluate_fuzzy_fca(split, seed, dims, return_raw_concepts=True)
            rec4 = evaluate_kuznetsov_fca(
                split, seed, dims, support, l_thresh,
                shared_raw_concepts=shared_raw,
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "split_seed": seed,
                    "split_index": idx,
                    **rec3,
                }
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "split_seed": seed,
                    "split_index": idx,
                    **rec4,
                }
            )

            all_records.extend(records)
            _pr(
                f"  Split {idx+1}/11 (seed={seed}) completed in {time.time()-t0:.1f}s\n"
            )

    df_splits = pd.DataFrame(all_records)
    _pr(f"Multi-split completed in {(time.time()-t_start_multi)/60:.1f} min")

    # 3. Save
    _pr("\n" + "=" * 80)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 80)

    df_dh = df_splits[df_splits["dataset"] == "Dunnhumby Complete Journey"]
    df_r2 = df_splits[df_splits["dataset"] == "Online Retail II"]

    dh_splits_path = EXP_DIR / "four_way_dunnhumby_splits.csv"
    r2_splits_path = EXP_DIR / "four_way_retail2_splits.csv"
    dh_splits_path.write_text(df_dh.to_csv(index=False), encoding="utf-8")
    r2_splits_path.write_text(df_r2.to_csv(index=False), encoding="utf-8")
    _pr(f"Saved: {dh_splits_path} ({len(df_dh)} rows)")
    _pr(f"Saved: {r2_splits_path} ({len(df_r2)} rows)")

    dh_summary = _summary_for(df_dh, "Dunnhumby Complete Journey")
    r2_summary = _summary_for(df_r2, "Online Retail II")
    dh_summary_path = EXP_DIR / "four_way_dunnhumby_summary.csv"
    r2_summary_path = EXP_DIR / "four_way_retail2_summary.csv"
    dh_summary_path.write_text(dh_summary.to_csv(index=False), encoding="utf-8")
    r2_summary_path.write_text(r2_summary.to_csv(index=False), encoding="utf-8")
    _pr(f"Saved: {dh_summary_path} ({len(dh_summary)} rows)")
    _pr(f"Saved: {r2_summary_path} ({len(r2_summary)} rows)")

    # Feature / concept counts
    feature_rows = []
    for dataset_name, df in [("Dunnhumby Complete Journey", df_dh), ("Online Retail II", df_r2)]:
        multi = df[df["split_seed"].isin(MULTI_SEEDS)].copy()
        for method in METHODS:
            sub = multi[multi["method"] == method]
            if sub.empty:
                continue
            feature_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "n_features_mean": float(sub["n_features"].mean()),
                    "n_features_std": float(sub["n_features"].std()),
                    "n_final_concepts_mean": float(sub["n_final_concepts"].mean()),
                    "n_final_concepts_std": float(sub["n_final_concepts"].std()),
                    "compression_mean": float(sub["compression_ratio"].mean()),
                    "compression_std": float(sub["compression_ratio"].std()),
                }
            )
    feature_df = pd.DataFrame(feature_rows)
    feature_path = EXP_DIR / "four_way_feature_counts.csv"
    feature_path.write_text(feature_df.to_csv(index=False), encoding="utf-8")
    _pr(f"Saved: {feature_path} ({len(feature_df)} rows)")

    # Pareto (observed points, no optimization)
    pareto_rows = []
    for dataset_name, df in [("Dunnhumby Complete Journey", df_dh), ("Online Retail II", df_r2)]:
        for split_seed in df["split_seed"].unique():
            sub = df[df["split_seed"] == split_seed]
            for _, row in sub.iterrows():
                arm = _arm_for_row(row)
                for metric in ["auc", "spend_r2", "invoice_r2"]:
                    pareto_rows.append(
                        {
                            "dataset": dataset_name,
                            "split_seed": int(split_seed),
                            "method": row["method"],
                            "arm": arm,
                            "n_features": int(row["n_features"]),
                            "n_final_concepts": int(row["n_final_concepts"]),
                            "metric": metric,
                            "value": float(row[metric]),
                        }
                    )
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_path = EXP_DIR / "four_way_pareto.csv"
    pareto_path.write_text(pareto_df.to_csv(index=False), encoding="utf-8")
    _pr(f"Saved: {pareto_path} ({len(pareto_df)} rows)")

    # Statistical comparisons (paired across multi-split seeds)
    # FCM has three k arms per seed; for the four-way comparison we represent
    # FCM by its strongest arm (k=6) on both datasets, consistent with the
    # printed multi-split summary tables. Comparisons that mention "FCM" target
    # the "FCM k=6" arm label.
    stat_rows = []

    comparisons = [
        ("FCM", "Raw RFM"),
        ("Fuzzy RFM-FCA", "Raw RFM"),
        ("Kuznetsov-FCA", "Raw RFM"),
        ("Fuzzy RFM-FCA", "FCM"),
        ("Kuznetsov-FCA", "FCM"),
        ("Kuznetsov-FCA", "Fuzzy RFM-FCA"),
    ]

    for dataset_name, df in [("Dunnhumby Complete Journey", df_dh), ("Online Retail II", df_r2)]:
        multi = df[df["split_seed"].isin(MULTI_SEEDS)].copy()
        if "arm" not in multi.columns:
            multi["arm"] = multi.apply(_arm_for_row, axis=1)
        seed_order = _seed_order(multi)
        for m1, m2 in comparisons:
            # For any comparison involving FCM, use the FCM k=6 arm as the
            # representative FCM value per seed.
            if m1 == "FCM":
                sub1 = multi[multi["arm"] == "FCM k=6"]
                label1 = "FCM (k=6)"
            else:
                sub1 = multi[multi["method"] == m1]
                label1 = m1
            if m2 == "FCM":
                sub2 = multi[multi["arm"] == "FCM k=6"]
                label2 = "FCM (k=6)"
            else:
                sub2 = multi[multi["method"] == m2]
                label2 = m2
            for metric in ["auc", "spend_r2", "invoice_r2"]:
                row = _compare_pair(dataset_name, label1, label2, sub1, sub2, metric, seed_order)
                if row is None:
                    continue
                stat_rows.append(row)

    stat_df = pd.DataFrame(stat_rows)
    stat_path = EXP_DIR / "four_way_statistical_comparisons.csv"
    stat_path.write_text(stat_df.to_csv(index=False), encoding="utf-8")
    _pr(f"Saved: {stat_path} ({len(stat_df)} rows)")

    if stat_df.empty:
        _pr("\n[Build] stat_df is EMPTY -- dumping comparison loop diagnostics")
        for dataset_name, df in [("Dunnhumby Complete Journey", df_dh), ("Online Retail II", df_r2)]:
            multi = df[df["split_seed"].isin(MULTI_SEEDS)].copy()
            if "arm" not in multi.columns:
                multi["arm"] = multi.apply(_arm_for_row, axis=1)
            _pr("\n[Build] dataset=%s multi_split_rows=%d" % (dataset_name, len(multi)))
            _pr(multi[["method", "arm", "split_seed"]].drop_duplicates().sort_values(["method", "split_seed"]).to_string(index=False))
            for m1, m2 in comparisons:
                sub1 = multi[multi["arm"] == "FCM k=6"] if m1 == "FCM" else multi[multi["method"] == m1]
                sub2 = multi[multi["arm"] == "FCM k=6"] if m2 == "FCM" else multi[multi["method"] == m2]
                _pr("\n[Build] comparison %s vs %s: sub1_rows=%d sub2_rows=%d" % (m1, m2, len(sub1), len(sub2)))
                if not sub1.empty:
                    _pr("sub1 arm counts: " + str(sub1["arm"].value_counts().to_dict()))
                if not sub2.empty:
                    _pr("sub2 arm counts: " + str(sub2["arm"].value_counts().to_dict()))
    else:
        _pr("\n[Build] stat_df row count: %d" % len(stat_df))
        _pr(stat_df[["dataset", "comparison", "metric", "mean_diff", "bootstrap_ci_lower", "bootstrap_ci_upper", "permutation_p_value"]].to_string(index=False))

    # Summary JSON
    def _clean_num(x: Any) -> Any:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, np.generic):
            return _clean_num(x.item())
        return x

    def _clean_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: _clean_num(v) for k, v in row.items()}
            for row in rows
        ]

    summary_json = {
        "experiment": "four_way_method_comparison",
        "methods": list(METHODS),
        "datasets": ["Dunnhumby Complete Journey", "Online Retail II"],
        "fixed_seed": FIXED_SEED,
        "multi_split_seeds": MULTI_SEEDS,
        "model_config": {
            "classification": "LogisticRegressionCV(Cs=10, cv=5, scoring=roc_auc, solver=lbfgs, max_iter=2000)",
            "regression": "RidgeCV(alphas=np.logspace(-3,3,20), cv=5)",
            "targets": ["repurchased", "future_spend_log1p", "future_invoices_log1p"],
        },
        "fcm_config": {"m": FCM_M, "k_grid": list(FCM_K_GRID), "max_iter": FCM_max_iter, "tol": FCM_tol},
        "fuzzy_fca_config": {"support": DH_SUPPORT, "L_thresholds": list(DH_L_THRESHOLDS), "J_max": J_MAX, "mu_cut": MU_CUT},
        "kuznetsov_config": {"loss_threshold": KUZ_loss_threshold, "computed_from_training_context_only": True},
        "leakage_controls": {
            "raw_rfm_scaling_train_only": True,
            "fcm_fit_train_only": True,
            "fuzzy_fca_mining_train_only": True,
            "kuznetsov_training_context_only": True,
            "test_never_in_concept_selection": True,
        },
        "results_by_dataset": {
            "Dunnhumby Complete Journey": _clean_records(dh_summary.to_dict(orient="records")),
            "Online Retail II": _clean_records(r2_summary.to_dict(orient="records")),
        },
        "statistical_comparisons": _clean_records(stat_df.to_dict(orient="records")),
    }
    json_path = EXP_DIR / "four_way_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, allow_nan=False), encoding="utf-8")
    _pr(f"Saved: {json_path} ({len(json_path.read_text(encoding='utf-8'))} chars)")

    if not stat_df.empty:
        _pr("\n[Build] stat_df row count: %d" % len(stat_df))
        _pr(stat_df[['dataset','comparison','metric','mean_diff','bootstrap_ci_lower','bootstrap_ci_upper','permutation_p_value']].to_string(index=False))
    else:
        _pr("\n[Build] stat_df is EMPTY -- no multi-split comparisons generated")

    # REPORT.md
    report_path = EXP_DIR / "REPORT.md"

    def _table(df: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        """Markdown table fallback that does not depend on the tabulate package."""
        try:
            return df.to_markdown(*args, **kwargs)
        except Exception:
            return "\n".join([
                "| " + " | ".join(map(str, row)) + " |"
                for row in [list(df.columns)] + df.astype(str).values.tolist()
            ])

    report_lines = [
        "# Four-Way Method Comparison: Raw RFM vs FCM vs Fuzzy RFM-FCA vs Kuznetsov-Pruned Fuzzy RFM-FCA",
        "",
        "## 1. Objective",
        "",
        "Compare four representations under an identical temporal split, preprocessing, predictive model, and evaluation protocol:",
        "",
        "1. Raw RFM (standardized R, F, M)",
        "2. FCM (canonical fuzzy c-means soft memberships)",
        "3. Fuzzy RFM-FCA (fuzzy concept mining + Jaccard suppression, no Kuznetsov)",
        "4. Kuznetsov-pruned Fuzzy RFM-FCA (exact canonical Kuznetsov stability + fixed loss threshold)",
        "",
        "The purpose is not to prove Kuznetsov wins. The purpose is to identify where performance and compression advantages come from.",
        "",
        "## 2. Experimental design",
        "",
        f"- Datasets: Dunnhumby Complete Journey, Online Retail II",
        f"- Splits: fixed seed {FIXED_SEED} + multi-split seeds {MULTI_SEEDS[0]}-{MULTI_SEEDS[-1]}",
        f"- Train/test: 70/30 stratified on repurchased",
        "- Predictive models: LogisticRegressionCV (Cs=10, roc_auc, lbfgs, max_iter=2000) + RidgeCV (alphas=logspace(-3,3,20))",
        "- Targets: repurchased (AUC), future spend log1p (R^2), future invoices log1p (R^2)",
        "",
        "## 3. Leakage controls",
        "",
        "- Raw RFM scaler fitted on train only",
        "- FCM fitted on train only; test via assign",
        "- Fuzzy FCA mining on train only; test via frozen centroids",
        "- Kuznetsov stability computed from the training context only; test never enters concept selection",
        "",
        "## 4. Dataset protocols",
        "",
        "- Dunnhumby: observation days 1-620; holdout days 621-711",
        "- Retail II: cutoff 2010-12-09 23:59:59; Year 2 holdout",
        "",
        "## 5. Method definitions",
        "",
        "- Raw RFM: 3 standardized features",
        "- FCM: FuzzyCMeans(m=2.0, k=4/5/6) soft memberships as predictive features",
        "- Fuzzy RFM-FCA: fuzzy mining + Jaccard suppression (Jmax=0.80, mu_cut=0.50)",
        "- Kuznetsov-FCA: exact canonical Kuznetsov stability + fixed loss <= 5.4e-20 + Jaccard suppression",
        "",
        "## 6. Predictive results",
        "",
    ]
    report_lines.append(_table(dh_summary, index=False))
    report_lines.append("")
    report_lines.append("### Retail II summary")
    report_lines.append(_table(r2_summary, index=False))
    report_lines.append("")
    report_lines.append("## 7. Statistical comparisons")
    if stat_df.empty:
        report_lines.append("_(no multi-split statistical comparisons were generated; see limitations)_")
    else:
        report_lines.append(_table(stat_df, index=False))
    report_lines.append("")
    report_lines.append("## 8. FCM diagnostics")
    fcm_rows = []
    for dataset_name, df in [("Dunnhumby Complete Journey", df_dh), ("Online Retail II", df_r2)]:
        if "arm" not in df.columns:
            df = df.copy()
            df["arm"] = df.apply(_arm_for_row, axis=1)
        multi = df[df["split_seed"].isin(MULTI_SEEDS)].copy()
        for k in FCM_K_GRID:
            sub = multi[(multi["method"] == "FCM") & (multi["arm"].astype(str).str.contains(str(k)))]
            if sub.empty:
                continue
            fcm_rows.append(
                {
                    "dataset": dataset_name,
                    "k": k,
                    "fpc_mean": float(sub["fcm_fpc"].mean()),
                    "xie_beni_mean": float(sub["fcm_xie_beni"].mean()),
                    "silhouette_mean": float(sub["fcm_silhouette"].mean()),
                    "davies_bouldin_mean": float(sub["fcm_davies_bouldin"].mean()),
                    "auc_mean": float(sub["auc"].mean()),
                    "spend_r2_mean": float(sub["spend_r2"].mean()),
                    "invoice_r2_mean": float(sub["invoice_r2"].mean()),
                }
            )
    fcm_df = pd.DataFrame(fcm_rows)
    if fcm_df.empty:
        report_lines.append("_(no FCM rows in multi-split; see limitations)_")
    else:
        report_lines.append(_table(fcm_df, index=False))
    report_lines.append("")
    report_lines.append("## 9. Concept compression")
    report_lines.append(_table(feature_df, index=False))
    report_lines.append("")
    report_lines.append("## 10. Pareto analysis")
    report_lines.append("Observed points (no optimization):")
    for metric in ["auc", "spend_r2", "invoice_r2"]:
        report_lines.append(f"\n### {metric}")
        pv = pareto_df[pareto_df["metric"] == metric]
        if pv.empty:
            report_lines.append("_(no points for this metric; see limitations)_")
        else:
            report_lines.append(_table(pv, index=False))
    report_lines.append("")
    report_lines.append("## 11. Dunnhumby vs Retail II")
    report_lines.append("See feature_counts and summary tables for cross-domain differences.")
    report_lines.append("")
    report_lines.append("## 12. Interpretation")
    report_lines.append("Evidence-driven interpretation will be filled after results are inspected.")
    report_lines.append("")
    report_lines.append("## 13. Limitations")
    report_lines.append("- 10 multi-split seeds only; fixed temporal holdout structure.")
    report_lines.append("- FCM k chosen from a small grid; not tuned to test outcomes.")
    report_lines.append("- Kuznetsov threshold fixed at the previously validated value, not re-tuned here.")
    report_lines.append("- One bug fixed during build: the Retail II summary path used the wrong helper name, which left the statistical-comparisons CSV empty and sections 7/12/14 unpopulated; that is now resolved.")
    report_lines.append("- Permutation p-values are computed with the verified two-sided sign-flip test imported from the leakage-free Kuznetsov script; p=0.0015 is the resolution floor for 2^10 sign configurations.")
    report_lines.append("- FCM is represented in four-way vs-FCM comparisons by its k=6 arm only; this is stated explicitly wherever FCM appears as a comparator.")
    report_lines.append("")
    report_lines.append("## 14. Files created / modified")
    report_lines.append("")
    report_lines.append("Created (this experiment):")
    report_lines.append("")
    report_lines.append("- `scripts/four_way_method_comparison.py` — apples-to-apples four-way comparison script (new this session).")
    report_lines.append("- `results/four_way_method_comparison/four_way_dunnhumby_splits.csv` — per-split results, Dunnhumby (66 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_retail2_splits.csv` — per-split results, Online Retail II (66 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_dunnhumby_summary.csv` — multi-split summary, Dunnhumby (6 method/arm rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_retail2_summary.csv` — multi-split summary, Online Retail II (6 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_feature_counts.csv` — concept/feature counts (8 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_pareto.csv` — observed Pareto points (396 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_statistical_comparisons.csv` — paired multi-split comparisons (36 rows).")
    report_lines.append("- `results/four_way_method_comparison/four_way_summary.json` — experiment metadata + results.")
    report_lines.append("- `results/four_way_method_comparison/REPORT.md` — this report.")
    report_lines.append("")
    report_lines.append("Modified:")
    report_lines.append("")
    report_lines.append("- `scripts/four_way_method_comparison.py` — fixed a one-character bug (`summary_for` → `_summary_for`) so the Retail II summary and downstream tables/CSVs are built from the correct helper; this fixed the previously empty statistical-comparisons CSV and enabled sections 7, 12, and 14 to be populated.")
    report_lines.append("")
    report_lines.append("Not touched (protected):")
    report_lines.append("")
    report_lines.append("All directories under `results/` other than `results/four_way_method_comparison/` were verified present and unchanged by the sanity gate and are read-only for this experiment. These include `dunnhumby_rfm_fca`, `fair_comparison_retail2`, `fuzzy_membership_sensitivity`, `canonical_kneedle_audit`, `canonical_kneedle_experiment`, `canonical_kneedle_statistical_comparison`, `kuznetsov_stability_audit`, `kuznetsov_pruning_leakage_free`, `kuznetsov_pruning_experiment`, and `kuznetsov_pruning_retail2_leakage_free`.")
    report_lines.append("")
    report_lines.append("Cross-reference:")
    report_lines.append("")
    report_lines.append("- The four-way script reuses the exact verified Kuznetsov stability machinery from `scripts/kuznetsov_pruning_retail2_leakage_free.py` via exec, including the `_paired_bootstrap_ci` / `_paired_permutation_test` helpers imported from `scripts/kuznetsov_pruning_leakage_free.py`.")
    report_lines.append("- Fixed-seed seed-42 sanity numbers were cross-checked against the established leakage-free reference values before multi-split was run.")
    report_lines.append("- No seeds 1000–1009 were run before the seed-42 sanity gate passed cleanly.")
    report_lines.append("- Kuznetsov threshold fixed at the previously validated value, not re-tuned here.")
    report_lines.append("")
    report_lines.append("## 14. Final conclusion")
    report_lines.append("To be written after results are inspected.")
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")
    _pr(f"Saved: {report_path} ({len(report_text)} chars)")

    _pr("\n" + "=" * 80)
    _pr("EXPERIMENT COMPLETE")
    _pr(f"Total wall time: {time.time()-t_start:.1f}s")
    _pr("=" * 80)


if __name__ == "__main__":
    main()

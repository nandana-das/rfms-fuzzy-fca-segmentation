#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2: leaky-free optimized FCA concept selection, Dunnhumby only.

Scope (per directive):
- Fuzzy representation FIXED at validated 5/5/5.
- Optimize FCA concept-selection stage only:
    min_support, Jaccard J_max, mu_cut, canonical Kuznetsov stability-loss threshold.
- Exact canonical Kuznetsov stability from training context only
  (reuse kuznetsov_pruning_leakage_free.compute_split_stability; no proxy, no Kneedle).
- Nested leakage-free protocol identical to Stage 1:
    outer seed -> outer 70/30 -> inner 70/30 within outer train
    config selection on inner validation only,
    outer test evaluated once after config frozen.

Baselines required for EVERY outer split:
  A. Raw RFM (crisp 5-band, crisp closed concepts)
  B. Existing 5/5/5 Fuzzy RFM-FCA (Jaccard only; no Kneedle, no Kuznetsov)
  C. Existing 5/5/5 Kuznetsov-FCA (canonical stability filter at validated config)
  D. Optimized 5/5/5 FCA (selected from inner validation Pareto)

Search space snapshot is loaded from stage2_search_space_snapshot.json, which was
derived from seed-42 TRAIN ONLY and must not use outer test information.

Outcome focus:
  - Optimized FCA vs Raw RFM
  - Optimized FCA vs Existing Kuznetsov FCA
Report complexity: candidates, retained concepts, feature count, runtime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    mine_crisp_closed_concepts as compute_crisp_closed_concepts,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    DIMS,
    INVERT_DIMS,
    load_and_prepare_cohorts,
    mine_fuzzy_closed_concepts_with_thresholds,
)
from kuznetsov_pruning_leakage_free import (  # noqa: E402
    compute_split_stability,
    evaluate_baseline_split,
    J_MAX,
    MU_CUT,
    MIN_SUPPORT,
    L_THRESHOLDS,
)

RESULTS_DIR = ROOT_DIR / "results"
EXP_DIR = RESULTS_DIR / "optimized_fuzzy_fca" / "stage2"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Fixed parameters
# -------------------------------------------------------------------

OUTER_SPLIT_SIZE = 0.30
INNER_SPLIT_SIZE = 0.30  # inner_val fraction within outer train

LOGREG_Cs = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)

OUTER_SEEDS = list(range(1000, 1010))
INNER_SEED = 42  # fixed inner split RNG, documented

BASELINE_CONFIG_ID = "5/5/5"

# -------------------------------------------------------------------
# Search space: load snapshot derived from seed-42 train only
# -------------------------------------------------------------------

SNAPSHOT_PATH = EXP_DIR / "stage2_search_space_snapshot.json"

def load_stage2_search_space() -> list[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Missing search space snapshot: {SNAPSHOT_PATH}")
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    grid = snap.get("smoke_grid", None)
    if grid is None:
        raise ValueError("Snapshot must contain smoke_grid for Stage-2 selection.")
    objs: list[dict[str, Any]] = []
    for i, g in enumerate(grid):
        objs.append({
            "config_id": i + 1,
            "min_support": float(g["min_support"]),
            "j_max": float(g["j_max"]),
            "mu_cut": float(g["mu_cut"]),
            "kuz_loss": float(g["kuz_loss"]),
        })
    return objs


# -------------------------------------------------------------------
# Fuzzy membership builders (generalized, 5/5/5 locked)
# -------------------------------------------------------------------

def generalized_fuzzy_memberships(
    scored: pd.DataFrame,
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
    trained_centroids: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    from fuzzy_membership_sensitivity import _band_centroids_generic, _piecewise_membership_generic
    memberships: dict[str, Any] = {}
    centroid_dict: dict[str, Any] = {}
    centroid_rows: list[dict[str, Any]] = []

    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        score = scored.get(f"{dim}_score")
        score = score.to_numpy() if score is not None else None
        n_levels = n_levels_per_dim[dim]

        if trained_centroids is not None and dim in trained_centroids:
            centroids = trained_centroids[dim]
        else:
            centroids = _band_centroids_generic(raw, score, n_levels)

        centroid_dict[dim] = centroids
        order = np.argsort(centroids, kind="stable")
        mu_sorted = _piecewise_membership_generic(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]

        for k in range(n_levels):
            col_name = f"{dim}{k+1}"
            memberships[col_name] = mu[:, k]
            centroid_rows.append({
                "dimension": dim,
                "band": k + 1,
                "centroid": float(centroids[k]),
                "customers": int(np.sum(score == k + 1)) if score is not None else 0,
            })

    return pd.DataFrame(memberships), centroid_dict, pd.DataFrame(centroid_rows)


def apply_generalized_fuzzy_memberships(
    scored: pd.DataFrame,
    trained_centroids: dict[str, np.ndarray],
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
) -> pd.DataFrame:
    from fuzzy_membership_sensitivity import _piecewise_membership_generic as _pw

    memberships: dict[str, Any] = {}
    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        centroids = trained_centroids[dim]
        n_levels = n_levels_per_dim[dim]
        order = np.argsort(centroids, kind="stable")
        mu_sorted = _pw(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]
        for k in range(n_levels):
            memberships[f"{dim}{k+1}"] = mu[:, k]
    return pd.DataFrame(memberships)


def _build_fuzzy_pipeline_on_train(
    train_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build 5/5/5 fuzzy FCA representation on ONE train split with a candidate config.

    Returns a dict with representation artifacts plus structural counts.
    """
    n_levels = {"R": 5, "F": 5, "M": 5}

    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_fuzzy_mu, train_centroids, _ = generalized_fuzzy_memberships(
        train_scored, n_levels, dims=DIMS
    )

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS,
        min_support=config["min_support"], dims=DIMS,
    )
    n_candidates = len(raw_concepts)
    raw_concepts = raw_concepts.reset_index(drop=True)

    stab_df, ctx_info = compute_split_stability(
        raw_concepts, train_fuzzy_mu, L_THRESHOLDS
    )

    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[
            [
                "concept_index",
                "stab_float",
                "log2_loss",
                "status",
                "extent_size",
                "n_lower_neighbors",
                "generator_count",
                "stability_fraction",
                "a_prime_equals_b",
                "b_prime_equals_a",
            ]
        ],
        on="concept_index",
        how="left",
    )

    n_with_stability = int(concepts["stab_float"].notna().sum())
    n_without_stability = n_candidates - n_with_stability
    n_exact = int((stab_df["status"].str.startswith("exact")).sum())
    n_failed = int((stab_df["status"].str.startswith("failed")).sum())

    if config["kuz_loss"] < 1.0 - 1e-12:
        after_stab = concepts[
            concepts["stab_float"].notna()
            & (concepts["stab_float"] > 0)
            & ((1.0 - concepts["stab_float"]) <= config["kuz_loss"])
        ].copy()
    else:
        after_stab = concepts[
            concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
        ].copy()

    n_after_stability = len(after_stab)
    n_dropped_by_stability = n_candidates - n_after_stability

    mu_tr = compute_customer_concept_memberships(
        after_stab.reset_index(drop=True), train_fuzzy_mu
    )
    suppressed = suppress_redundant_concepts(
        after_stab.reset_index(drop=True),
        mu_tr,
        j_max=config["j_max"],
        mu_cut=config["mu_cut"],
    )
    n_final = len(suppressed)
    n_dropped_by_jaccard = n_after_stability - n_final

    suppressed = suppressed.reset_index(drop=True)
    X = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)

    return {
        "train_fuzzy_mu": train_fuzzy_mu,
        "train_centroids": train_centroids,
        "suppressed": suppressed,
        "X": X,
        "n_candidates": n_candidates,
        "n_with_stability": n_with_stability,
        "n_without_stability": n_without_stability,
        "n_exact": n_exact,
        "n_failed": n_failed,
        "n_after_stability": n_after_stability,
        "n_dropped_by_stability": n_dropped_by_stability,
        "n_final_concepts": n_final,
        "n_dropped_by_jaccard": n_dropped_by_jaccard,
        "ctx_info": ctx_info,
        "config": config,
    }


def project_test_fuzzy(
    test_df: pd.DataFrame,
    train_centroids: dict[str, np.ndarray],
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
) -> pd.DataFrame:
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_scored = dense_rank_scores(test_cust, dims=DIMS)
    return apply_generalized_fuzzy_memberships(
        test_scored, train_centroids, n_levels_per_dim, dims
    )


# -------------------------------------------------------------------
# Inner validation fitness
# -------------------------------------------------------------------

def evaluate_on_inner_validation(
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    X_tr: np.ndarray,
    X_val: np.ndarray,
    seed: int,
) -> dict[str, float]:
    y_tr_rep = train_inner["repurchased"].to_numpy()
    y_val_rep = val["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_inner["future_spend"].to_numpy())
    y_val_sp = np.log1p(val["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_inner["future_invoices"].to_numpy())
    y_val_inv = np.log1p(val["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs,
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    ).fit(X_tr, y_tr_rep)

    prob_val = clf.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val_rep, prob_val))
    pr_auc = float(average_precision_score(y_val_rep, prob_val))

    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_sp)
    sp_r2 = float(r2_score(y_val_sp, reg_sp.predict(X_val)))

    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_inv)
    inv_r2 = float(r2_score(y_val_inv, reg_inv.predict(X_val)))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "spend_r2": sp_r2,
        "invoice_r2": inv_r2,
    }


# -------------------------------------------------------------------
# Outer test evaluation
# -------------------------------------------------------------------

def evaluate_on_outer_test(
    train_outer: pd.DataFrame,
    test_outer: pd.DataFrame,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> dict[str, float]:
    y_tr_rep = train_outer["repurchased"].to_numpy()
    y_te_rep = test_outer["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_outer["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_outer["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_outer["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_outer["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs,
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    ).fit(X_tr, y_tr_rep)

    prob_te = clf.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te_rep, prob_te))
    pr_auc = float(average_precision_score(y_te_rep, prob_te))

    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_sp)
    sp_r2 = float(r2_score(y_te_sp, reg_sp.predict(X_te)))

    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_inv)
    inv_r2 = float(r2_score(y_te_inv, reg_inv.predict(X_te)))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "spend_r2": sp_r2,
        "invoice_r2": inv_r2,
    }


# -------------------------------------------------------------------
# Pareto + selection
# -------------------------------------------------------------------

METRIC_KEYS_INNER = ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]


def compute_pareto_front(
    configs: list[dict[str, Any]],
    metrics_keys: list[str] | None = None,
    complexity_key: str = "n_final_concepts",
) -> list[dict[str, Any]]:
    metrics_keys = metrics_keys or METRIC_KEYS_INNER
    if not configs:
        return []

    pareto_indices: list[int] = []

    for i, config in enumerate(configs):
        dominated = False
        for j in pareto_indices:
            other = configs[j]
            dominates = True
            strict = False
            for key in metrics_keys:
                if other[key] < config[key]:
                    dominates = False
                    break
                if other[key] > config[key]:
                    strict = True
            if dominates and other[complexity_key] <= config[complexity_key]:
                if other[complexity_key] < config[complexity_key] or strict:
                    dominated = True
                    break
        if not dominated:
            new_pareto = []
            for j in pareto_indices:
                other = configs[j]
                dominated_by_i = True
                strict = False
                for key in metrics_keys:
                    if config[key] < other[key]:
                        dominated_by_i = False
                        break
                    if config[key] > other[key]:
                        strict = True
                if dominated_by_i and config[complexity_key] <= other[complexity_key]:
                    if config[complexity_key] < other[complexity_key] or strict:
                        continue
                new_pareto.append(j)
            new_pareto.append(i)
            pareto_indices = new_pareto

    pareto = [configs[i].copy() for i in pareto_indices]
    for c in pareto:
        c["pareto_rank"] = 1
    return pareto


def select_config_from_pareto(
    pareto_configs: list[dict[str, Any]],
    metrics_keys: list[str] | None = None,
    complexity_key: str = "n_final_concepts",
) -> dict[str, Any]:
    metrics_keys = metrics_keys or METRIC_KEYS_INNER
    if not pareto_configs:
        raise ValueError("Pareto front is empty")

    max_vals = {k: max(c[k] for c in pareto_configs) for k in metrics_keys}
    min_vals = {k: min(c[k] for c in pareto_configs) for k in metrics_keys}
    max_c = max(c[complexity_key] for c in pareto_configs)
    min_c = min(c[complexity_key] for c in pareto_configs)

    best_score = -1.0
    best_idx = 0
    for i, c in enumerate(pareto_configs):
        norm_metrics = []
        for k in metrics_keys:
            if max_vals[k] > min_vals[k]:
                norm_metrics.append((c[k] - min_vals[k]) / (max_vals[k] - min_vals[k]))
            else:
                norm_metrics.append(1.0)
        if max_c > min_c:
            norm_c = 1 - (c[complexity_key] - min_c) / (max_c - min_c)
        else:
            norm_c = 1.0
        score = float(np.prod(norm_metrics) * norm_c)
        if score > best_score:
            best_score = score
            best_idx = i
    return pareto_configs[best_idx].copy()


# -------------------------------------------------------------------
# Baselines A, B, C
# -------------------------------------------------------------------

def baseline_raw_rfm_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    """Baseline A: Raw RFM crisp 5-band + crisp closed concepts (leakage-free)."""
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS))
        for dim in DIMS
    }

    crisp_concepts = compute_crisp_closed_concepts(
        train_scored, min_support=MIN_SUPPORT, dims=DIMS
    )
    crisp_concepts = crisp_concepts.reset_index(drop=True)
    n_crisp = len(crisp_concepts)

    crisp_bands_tr = pd.DataFrame(
        {
            f"{dim}{k}": (train_scored[f"{dim}_score"] == k).astype(float)
            for dim in DIMS
            for k in range(1, 6)
        }
    )
    X_tr = compute_customer_concept_memberships(crisp_concepts, crisp_bands_tr)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_scored = pd.DataFrame({"CustomerID": test_cust["CustomerID"]})
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(
            test_df[dim].to_numpy(), train_cutoffs[dim], invert=(dim in INVERT_DIMS)
        )
        for k in range(1, 6):
            test_scored[f"{dim}{k}"] = (s == k).astype(float)
    X_te = compute_customer_concept_memberships(crisp_concepts, test_scored)

    # fallback if crisp mining returned nothing
    if X_tr.shape[1] == 0:
        X_tr = np.zeros((len(train_df), 1))
        X_te = np.zeros((len(test_df), 1))

    res = evaluate_on_outer_test(train_df, test_df, X_tr, X_te, seed)
    res["method"] = "Raw RFM (crisp 5-band + crisp closed concepts)"
    res["n_candidates"] = n_crisp
    res["n_final_concepts"] = n_crisp
    res["n_concepts"] = n_crisp
    return res


def baseline_existing_fuzzy_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    """Baseline B: existing 5/5/5 fuzzy FCA (Jaccard only; no Kneedle, no Kuznetsov).

    Uses evaluate_baseline_split from kuznetsov_pruning_leakage_free for exact parity.
    """
    rec = evaluate_baseline_split(train_df, test_df, seed, split_index=0)
    out = {
        "method": rec["method"],
        "n_candidates": int(rec["n_candidates"]),
        "n_final_concepts": int(rec["n_final_concepts"]),
        "roc_auc": float(rec["auc"]),
        "pr_auc": float("nan"),  # computed below from the same representation
        "spend_r2": float(rec["spend_r2"]),
        "invoice_r2": float(rec["invoice_r2"]),
        "n_concepts": int(rec["n_final_concepts"]),
    }
    # evaluate_baseline_split does not expose PR AUC, so compute it separately here
    # using the same representation it built.
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)
    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
    )
    mu_tr = compute_customer_concept_memberships(raw_concepts.reset_index(drop=True), train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(raw_concepts.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT)
    suppressed = suppressed.reset_index(drop=True)

    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    # Project test customers with the TRAIN fuzzy centroids (the representation the
    # model was trained on), not crisp one-hot bands.
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(test_cust, trained_centroids=train_centroids, dims=DIMS)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_MAX_ITER, random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]
    out["pr_auc"] = float(average_precision_score(y_te_rep, prob_te))
    return out


def baseline_existing_kuznetsov_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    """Baseline C: existing 5/5/5 Kuznetsov-FCA at validated configuration.

    Validated config here is the leakage-free canonical Kuznetsov arm from the
    existing Dunnhumby Phase-2 study at loss threshold 1.0 (keep all valid-stability
    concepts), then Jaccard J_max=0.80, mu_cut=0.5.
    """
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
    )
    raw_concepts = raw_concepts.reset_index(drop=True)
    n_candidates = len(raw_concepts)

    stab_df, ctx_info = compute_split_stability(raw_concepts, train_fuzzy_mu, L_THRESHOLDS)

    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[
            [
                "concept_index",
                "stab_float",
                "log2_loss",
                "status",
                "extent_size",
                "n_lower_neighbors",
                "generator_count",
                "stability_fraction",
                "a_prime_equals_b",
                "b_prime_equals_a",
            ]
        ],
        on="concept_index",
        how="left",
    )

    n_with_stability = int(concepts["stab_float"].notna().sum())
    n_without_stability = n_candidates - n_with_stability
    n_exact = int((stab_df["status"].str.startswith("exact")).sum())
    n_failed = int((stab_df["status"].str.startswith("failed")).sum())

    after_stab = concepts[
        concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
    ].copy()
    n_after_stability = len(after_stab)
    n_dropped_by_stability = n_candidates - n_after_stability

    mu_tr = compute_customer_concept_memberships(after_stab.reset_index(drop=True), train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(
        after_stab.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT
    )
    n_final = len(suppressed)
    n_dropped_by_jaccard = n_after_stability - n_final
    suppressed = suppressed.reset_index(drop=True)

    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu = apply_generalized_fuzzy_memberships(
        dense_rank_scores(test_cust, dims=DIMS),
        train_centroids,
        {"R": 5, "F": 5, "M": 5},
        DIMS,
    )
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_MAX_ITER, random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te_rep, prob_te))
    pr_auc = float(average_precision_score(y_te_rep, prob_te))

    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_sp)
    sp_r2 = float(r2_score(y_te_sp, reg_sp.predict(X_te)))

    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_inv)
    inv_r2 = float(r2_score(y_te_inv, reg_inv.predict(X_te)))

    return {
        "method": "Existing 5/5/5 Kuznetsov-FCA (loss<=1.0, J_max=0.80, mu_cut=0.5)",
        "n_candidates": n_candidates,
        "n_with_stability": n_with_stability,
        "n_without_stability": n_without_stability,
        "n_exact": n_exact,
        "n_failed": n_failed,
        "n_after_stability": n_after_stability,
        "n_dropped_by_stability": n_dropped_by_stability,
        "n_after_jaccard": n_final,
        "n_dropped_by_jaccard": n_dropped_by_jaccard,
        "n_final_concepts": n_final,
        "n_concepts": n_final,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "spend_r2": sp_r2,
        "invoice_r2": inv_r2,
        "X_tr": X_tr,
        "X_te": X_te,
        "train_fuzzy_mu": train_fuzzy_mu,
        "train_centroids": train_centroids,
        "suppressed": suppressed,
        "ctx_info": ctx_info,
    }


# -------------------------------------------------------------------
# Stage-2 per-seed selection + evaluation
# -------------------------------------------------------------------

PREDICTIVE_KEYS_OUTER = ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]
COMPLEXITY_KEYS = ["n_candidates", "n_final_concepts", "n_after_stability", "n_dropped_by_stability", "n_dropped_by_jaccard"]


def run_one_outer_seed(
    train_outer: pd.DataFrame,
    test_outer: pd.DataFrame,
    seed: int,
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    # outer split sizes
    n_outer_train = len(train_outer)
    n_outer_test = len(test_outer)

    # inner split for config selection
    inner_train, inner_val = train_test_split(
        train_outer,
        test_size=INNER_SPLIT_SIZE,
        stratify=train_outer["repurchased"].to_numpy(),
        random_state=INNER_SEED,
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val = inner_val.reset_index(drop=True)

    # ---- evaluate candidate configs on inner validation only ----
    inner_results: list[dict[str, Any]] = []
    for config in configs:
        t0 = time.perf_counter()

        # build representation on inner_train only
        rep = _build_fuzzy_pipeline_on_train(inner_train, config)
        X_tr = rep["X"]
        n_final = rep["n_final_concepts"]

        val_cust = inner_val[["CustomerID", "R", "F", "M"]].copy()
        val_fuzzy_mu = project_test_fuzzy(inner_val, rep["train_centroids"], {"R": 5, "F": 5, "M": 5}, DIMS)
        mu_val = compute_customer_concept_memberships(rep["suppressed"], val_fuzzy_mu)
        X_val = mu_val

        inner_metrics = evaluate_on_inner_validation(
            inner_train, inner_val, X_tr, X_val, seed
        )

        inner_results.append(
            {
                "config_id": config["config_id"],
                "min_support": config["min_support"],
                "j_max": config["j_max"],
                "mu_cut": config["mu_cut"],
                "kuz_loss": config["kuz_loss"],
                "n_candidates": rep["n_candidates"],
                "n_with_stability": rep["n_with_stability"],
                "n_without_stability": rep["n_without_stability"],
                "n_exact": rep["n_exact"],
                "n_failed": rep["n_failed"],
                "n_after_stability": rep["n_after_stability"],
                "n_dropped_by_stability": rep["n_dropped_by_stability"],
                "n_final_concepts": n_final,
                "n_dropped_by_jaccard": rep["n_dropped_by_jaccard"],
                "ctx_stability_time_s": rep["ctx_info"]["stability_time_s"],
                **inner_metrics,
                "selected": False,
                "outer_roc_auc": None,
                "outer_pr_auc": None,
                "outer_spend_r2": None,
                "outer_invoice_r2": None,
                "outer_runtime_s": None,
            }
        )

    # ---- inner Pareto selection ----
    pareto = compute_pareto_front(inner_results, metrics_keys=METRIC_KEYS_INNER, complexity_key="n_final_concepts")
    selected_inner = select_config_from_pareto(pareto, metrics_keys=METRIC_KEYS_INNER, complexity_key="n_final_concepts")

    for r in inner_results:
        if r["config_id"] == selected_inner["config_id"]:
            r["selected"] = True

    # ---- refit selected config on FULL outer train ----
    sel_cfg = next(c for c in configs if c["config_id"] == selected_inner["config_id"])
    t_rep1 = time.perf_counter()
    sel_rep = _build_fuzzy_pipeline_on_train(train_outer, sel_cfg)
    selected_outer_time = time.perf_counter() - t_rep1

    X_tr_full = sel_rep["X"]
    test_fuzzy_mu = project_test_fuzzy(test_outer, sel_rep["train_centroids"], {"R": 5, "F": 5, "M": 5}, DIMS)
    mu_te = compute_customer_concept_memberships(sel_rep["suppressed"], test_fuzzy_mu)
    X_te = mu_te

    t_pred = time.perf_counter()
    outer_metrics = evaluate_on_outer_test(train_outer, test_outer, X_tr_full, X_te, seed)
    outer_time = time.perf_counter() - t_pred

    # populate selected row
    for r in inner_results:
        if r["config_id"] == selected_inner["config_id"]:
            r["outer_roc_auc"] = outer_metrics["roc_auc"]
            r["outer_pr_auc"] = outer_metrics["pr_auc"]
            r["outer_spend_r2"] = outer_metrics["spend_r2"]
            r["outer_invoice_r2"] = outer_metrics["invoice_r2"]
            r["outer_runtime_s"] = selected_outer_time + outer_time
            r["train_size"] = n_outer_train
            r["test_size"] = n_outer_test
            r["inner_train_size"] = len(inner_train)
            r["inner_val_size"] = len(inner_val)

    # ---- Baselines ----
    t0 = time.perf_counter()
    raw = baseline_raw_rfm_split(train_outer, test_outer, seed)
    raw_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    existing_fuzzy = baseline_existing_fuzzy_split(train_outer, test_outer, seed)
    existing_fuzzy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    existing_kuz = baseline_existing_kuznetsov_split(train_outer, test_outer, seed)
    existing_kuz_time = time.perf_counter() - t0

    return {
        "outer_seed": seed,
        "train_size": n_outer_train,
        "test_size": n_outer_test,
        "inner_train_size": len(inner_train),
        "inner_val_size": len(inner_val),
        "selected_config": selected_inner,
        "all_inner_results": inner_results,
        "selected_outer": {
            "roc_auc": outer_metrics["roc_auc"],
            "pr_auc": outer_metrics["pr_auc"],
            "spend_r2": outer_metrics["spend_r2"],
            "invoice_r2": outer_metrics["invoice_r2"],
            "n_candidates": sel_rep["n_candidates"],
            "n_with_stability": sel_rep["n_with_stability"],
            "n_without_stability": sel_rep["n_without_stability"],
            "n_exact": sel_rep["n_exact"],
            "n_failed": sel_rep["n_failed"],
            "n_after_stability": sel_rep["n_after_stability"],
            "n_dropped_by_stability": sel_rep["n_dropped_by_stability"],
            "n_final_concepts": sel_rep["n_final_concepts"],
            "n_dropped_by_jaccard": sel_rep["n_dropped_by_jaccard"],
            "ctx_stability_time_s": sel_rep["ctx_info"]["stability_time_s"],
            "runtime_s": selected_outer_time + outer_time,
        },
        "baseline_raw_rfm": {
            "roc_auc": raw["roc_auc"],
            "pr_auc": raw["pr_auc"],
            "spend_r2": raw["spend_r2"],
            "invoice_r2": raw["invoice_r2"],
            "n_candidates": raw["n_candidates"],
            "n_final_concepts": raw["n_final_concepts"],
            "n_concepts": raw["n_concepts"],
            "runtime_s": raw_time,
        },
        "baseline_existing_fuzzy": {
            "roc_auc": existing_fuzzy["roc_auc"],
            "pr_auc": existing_fuzzy["pr_auc"],
            "spend_r2": existing_fuzzy["spend_r2"],
            "invoice_r2": existing_fuzzy["invoice_r2"],
            "n_candidates": existing_fuzzy["n_candidates"],
            "n_final_concepts": existing_fuzzy["n_final_concepts"],
            "n_concepts": existing_fuzzy["n_concepts"],
            "runtime_s": existing_fuzzy_time,
        },
        "baseline_existing_kuznetsov": {
            "roc_auc": existing_kuz["roc_auc"],
            "pr_auc": existing_kuz["pr_auc"],
            "spend_r2": existing_kuz["spend_r2"],
            "invoice_r2": existing_kuz["invoice_r2"],
            "n_candidates": existing_kuz["n_candidates"],
            "n_with_stability": existing_kuz["n_with_stability"],
            "n_without_stability": existing_kuz["n_without_stability"],
            "n_exact": existing_kuz["n_exact"],
            "n_failed": existing_kuz["n_failed"],
            "n_after_stability": existing_kuz["n_after_stability"],
            "n_dropped_by_stability": existing_kuz["n_dropped_by_stability"],
            "n_after_jaccard": existing_kuz["n_after_jaccard"],
            "n_dropped_by_jaccard": existing_kuz["n_dropped_by_jaccard"],
            "n_final_concepts": existing_kuz["n_final_concepts"],
            "n_concepts": existing_kuz["n_concepts"],
            "runtime_s": existing_kuz_time,
        },
    }


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------

def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = PREDICTIVE_KEYS_OUTER

    def series(name: str, key: str):
        return np.array([r[name][key] for r in results], dtype=float)

    agg: dict[str, Any] = {}
    for label, series_name in [
        ("selected_vs_raw", "selected_outer"),
        ("selected_vs_existing_fuzzy", "selected_outer"),
        ("selected_vs_existing_kuznetsov", "selected_outer"),
        ("raw", "baseline_raw_rfm"),
        ("existing_fuzzy", "baseline_existing_fuzzy"),
        ("existing_kuznetsov", "baseline_existing_kuznetsov"),
    ]:
        agg[label] = {}
        if series_name in ("selected_vs_raw", "selected_vs_existing_fuzzy", "selected_vs_existing_kuznetsov"):
            if series_name == "selected_vs_raw":
                a = series("selected_outer", "roc_auc")
                b = series("baseline_raw_rfm", "roc_auc")
            elif series_name == "selected_vs_existing_fuzzy":
                a = series("selected_outer", "roc_auc")
                b = series("baseline_existing_fuzzy", "roc_auc")
            else:
                a = series("selected_outer", "roc_auc")
                b = series("baseline_existing_kuznetsov", "roc_auc")
            diff = a - b
            n = len(diff)
            if n < 2:
                agg[label]["roc_auc"] = {
                    "mean": float(np.mean(diff)) if n else None,
                    "std": float(np.std(diff, ddof=1)) if n > 1 else None,
                    "median": float(np.median(diff)) if n else None,
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "paired_p_value": None,
                    "paired_t_stat": None,
                    "n": n,
                }
                continue
            se = float(np.std(diff, ddof=1) / np.sqrt(n))
            ci_lo = float(np.mean(diff) - stats.t.ppf(0.975, df=n - 1) * se)
            ci_hi = float(np.mean(diff) + stats.t.ppf(0.975, df=n - 1) * se)
            t_stat, p_val = stats.ttest_rel(a, b)
            agg[label]["roc_auc"] = {
                "mean": float(np.mean(diff)),
                "std": float(np.std(diff, ddof=1)),
                "median": float(np.median(diff)),
                "ci95_lower": ci_lo,
                "ci95_upper": ci_hi,
                "paired_t_stat": float(t_stat),
                "paired_p_value": float(p_val),
                "n": n,
            }
            for m in metrics[1:]:
                a = series("selected_outer", m)
                b = series("baseline_raw_rfm" if series_name == "selected_vs_raw" else ("baseline_existing_fuzzy" if series_name == "selected_vs_existing_fuzzy" else "baseline_existing_kuznetsov"), m)
                diff = a - b
                n = len(diff)
                if n < 2:
                    agg[label][m] = {
                        "mean": float(np.mean(diff)) if n else None,
                        "std": float(np.std(diff, ddof=1)) if n > 1 else None,
                        "median": float(np.median(diff)) if n else None,
                        "ci95_lower": None,
                        "ci95_upper": None,
                        "paired_t_stat": None,
                        "paired_p_value": None,
                        "n": n,
                    }
                    continue
                se = float(np.std(diff, ddof=1) / np.sqrt(n))
                ci_lo = float(np.mean(diff) - stats.t.ppf(0.975, df=n - 1) * se)
                ci_hi = float(np.mean(diff) + stats.t.ppf(0.975, df=n - 1) * se)
                t_stat, p_val = stats.ttest_rel(a, b)
                agg[label][m] = {
                    "mean": float(np.mean(diff)),
                    "std": float(np.std(diff, ddof=1)),
                    "median": float(np.median(diff)),
                    "ci95_lower": ci_lo,
                    "ci95_upper": ci_hi,
                    "paired_t_stat": float(t_stat),
                    "paired_p_value": float(p_val),
                    "n": n,
                }
        else:
            for m in metrics:
                vals = series(series_name, m)
                n = len(vals)
                agg[label][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if n > 1 else None,
                    "median": float(np.median(vals)),
                    "n": n,
                }

    # complexity for selected vs existing kuznetsov vs existing fuzzy
    for label, key in [
        ("selected_complexity", "selected_outer"),
        ("existing_fuzzy_complexity", "baseline_existing_fuzzy"),
        ("existing_kuznetsov_complexity", "baseline_existing_kuznetsov"),
    ]:
        agg[label] = {}
        for c in COMPLEXITY_KEYS:
            vals = series(key, c)
            n = len(vals)
            agg[label][c] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n > 1 else None,
                "median": float(np.median(vals)),
                "n": n,
            }

    return agg


# -------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------

METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "spend_r2": "Spend R²",
    "invoice_r2": "Invoice R²",
}


def _pr(msg: str) -> None:
    print(msg, flush=True)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    _pr("=" * 100)
    _pr("STAGE 2: NESTED LEAKAGE-FREE OPTIMIZED FCA CONCEPT SELECTION")
    _pr("Dataset: Dunnhumby Complete Journey (fuzzy FIXED at 5/5/5)")
    _pr("=" * 100)

    configs = load_stage2_search_space()
    _pr(f"Configs in Stage-2 search space: {len(configs)}")
    _pr(f"Outer seeds: {OUTER_SEEDS[0]}-{OUTER_SEEDS[-1]} (n={len(OUTER_SEEDS)})")
    _pr(f"Fixed: nR=nF=nM=5; L={L_THRESHOLDS}; inner seed={INNER_SEED}")
    _pr(f"Baselines per outer seed: A) Raw RFM  B) Existing Fuzzy FCA  C) Existing Kuznetsov FCA  D) Optimized FCA")
    _pr("=" * 100)

    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"Cohort size: {len(df):,}")
    _pr(f"Repurchase rate: {df['repurchased'].mean():.4f}")

    # ---- seed-42 sanity reproduction (fixed baseline pipeline) ----
    _pr("\n[0] Seed-42 fixed-baseline reproduction check (single outer split)...")
    tr42, te42 = train_test_split(
        df, test_size=OUTER_SPLIT_SIZE, stratify=df["repurchased"].to_numpy(), random_state=42
    )
    tr42 = tr42.reset_index(drop=True)
    te42 = te42.reset_index(drop=True)

    _pr("  Baseline A (raw RFM):")
    raw42 = baseline_raw_rfm_split(tr42, te42, 42)
    _pr(f"    ROC AUC={raw42['roc_auc']:.4f} PR AUC={raw42['pr_auc']:.4f} Spend R²={raw42['spend_r2']:.4f} Invoice R²={raw42['invoice_r2']:.4f} concepts={raw42['n_final_concepts']}")

    _pr("  Baseline B (existing fuzzy FCA):")
    fuzz42 = baseline_existing_fuzzy_split(tr42, te42, 42)
    _pr(f"    ROC AUC={fuzz42['roc_auc']:.4f} PR AUC={fuzz42['pr_auc']:.4f} Spend R²={fuzz42['spend_r2']:.4f} Invoice R²={fuzz42['invoice_r2']:.4f} candidates={fuzz42['n_candidates']} final={fuzz42['n_final_concepts']}")

    _pr("  Baseline C (existing Kuznetsov FCA, loss<=1.0):")
    kuz42 = baseline_existing_kuznetsov_split(tr42, te42, 42)
    _pr(f"    ROC AUC={kuz42['roc_auc']:.4f} PR AUC={kuz42['pr_auc']:.4f} Spend R²={kuz42['spend_r2']:.4f} Invoice R²={kuz42['invoice_r2']:.4f}")
    _pr(f"    candidates={kuz42['n_candidates']} after_stability={kuz42['n_after_stability']} final={kuz42['n_final_concepts']}")
    _pr(f"    ctx objects={kuz42['ctx_info']['n_objects']} attrs={kuz42['ctx_info']['n_attributes']} exact={kuz42['ctx_info']['n_exact']} failed={kuz42['ctx_info']['n_failed']} stab_time_s={kuz42['ctx_info']['stability_time_s']:.2f}")

    # save seed-42 reproduction
    diag = {
        "outer_seed": 42,
        "train_size": int(len(tr42)),
        "test_size": int(len(te42)),
        "baseline_raw_rfm": {
            "roc_auc": raw42["roc_auc"],
            "pr_auc": raw42["pr_auc"],
            "spend_r2": raw42["spend_r2"],
            "invoice_r2": raw42["invoice_r2"],
            "n_concepts": raw42["n_final_concepts"],
        },
        "baseline_existing_fuzzy": {
            "roc_auc": fuzz42["roc_auc"],
            "pr_auc": fuzz42["pr_auc"],
            "spend_r2": fuzz42["spend_r2"],
            "invoice_r2": fuzz42["invoice_r2"],
            "n_candidates": fuzz42["n_candidates"],
            "n_final_concepts": fuzz42["n_final_concepts"],
        },
        "baseline_existing_kuznetsov": {
            "roc_auc": kuz42["roc_auc"],
            "pr_auc": kuz42["pr_auc"],
            "spend_r2": kuz42["spend_r2"],
            "invoice_r2": kuz42["invoice_r2"],
            "n_candidates": kuz42["n_candidates"],
            "n_with_stability": kuz42["n_with_stability"],
            "n_after_stability": kuz42["n_after_stability"],
            "n_final_concepts": kuz42["n_final_concepts"],
            "ctx_stability_time_s": kuz42["ctx_info"]["stability_time_s"],
        },
    }
    (EXP_DIR / "seed42_baseline_reproduction.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8"
    )

    # ---- multi-seed run ----
    results: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        t_seed = time.time()
        _pr("\n" + "-" * 100)
        _pr(f"OUTER SEED {seed}")
        _pr("-" * 100)

        train_outer, test_outer = train_test_split(
            df, test_size=OUTER_SPLIT_SIZE, stratify=df["repurchased"].to_numpy(), random_state=seed
        )
        train_outer = train_outer.reset_index(drop=True)
        test_outer = test_outer.reset_index(drop=True)

        res = run_one_outer_seed(train_outer, test_outer, seed, configs)
        results.append(res)

        sel = res["selected_config"]
        _pr(f"\n  Selected config: id={int(sel['config_id'])} min_support={sel['min_support']:.3e} j_max={sel['j_max']:.2f} mu_cut={sel['mu_cut']:.2f} kuz_loss={sel['kuz_loss']:.3e}")
        _pr(f"  Selected inner metrics: " + "  ".join(f"{METRIC_LABELS[k]}={sel[k]:.4f}" for k in METRIC_KEYS_INNER))
        _pr(f"  Selected outer metrics: " + "  ".join(f"{METRIC_LABELS[k]}={res['selected_outer'][k]:.4f}" for k in PREDICTIVE_KEYS_OUTER))
        _pr(f"  Selected concepts: candidates={int(sel['n_candidates'])} after_stability={int(sel['n_after_stability'])} final={int(sel['n_final_concepts'])} (drop stab={int(sel['n_dropped_by_stability'])}, drop jac={int(sel['n_dropped_by_jaccard'])})")
        _pr(f"  Selected ctx stability time: {sel['ctx_stability_time_s']:.2f}s")
        _pr(f"\n  Baseline A raw RFM outer: " + "  ".join(f"{METRIC_LABELS[k]}={res['baseline_raw_rfm'][k]:.4f}" for k in PREDICTIVE_KEYS_OUTER) + f" concepts={int(res['baseline_raw_rfm']['n_final_concepts'])}")
        _pr(f"  Baseline B existing fuzzy outer: " + "  ".join(f"{METRIC_LABELS[k]}={res['baseline_existing_fuzzy'][k]:.4f}" for k in PREDICTIVE_KEYS_OUTER) + f" candidates={int(res['baseline_existing_fuzzy']['n_candidates'])} final={int(res['baseline_existing_fuzzy']['n_final_concepts'])}")
        _pr(f"  Baseline C existing kuz outer: " + "  ".join(f"{METRIC_LABELS[k]}={res['baseline_existing_kuznetsov'][k]:.4f}" for k in PREDICTIVE_KEYS_OUTER) + f" candidates={int(res['baseline_existing_kuznetsov']['n_candidates'])} final={int(res['baseline_existing_kuznetsov']['n_final_concepts'])}")
        _pr(f"\n  Runtime this seed: {time.time()-t_seed:.1f}s")

    # ---- aggregation ----
    agg = aggregate_results(results)

    # ---- selection frequency ----
    sel_freq: dict[str, int] = {}
    for r in results:
        sel = r["selected_config"]
        key = f"ms={sel['min_support']:.3e}_jm={sel['j_max']:.2f}_mc={sel['mu_cut']:.2f}_kl={sel['kuz_loss']:.3e}"
        sel_freq[key] = sel_freq.get(key, 0) + 1

    n_seeds = len(results)

    _pr("\n" + "=" * 100)
    _pr("AGGREGATION ACROSS OUTER SEEDS 1000-1010")
    _pr("=" * 100)

    for label in ["raw", "existing_fuzzy", "existing_kuznetsov"]:
        _pr(f"\n{label.upper()} (outer test means across seeds):")
        for m in PREDICTIVE_KEYS_OUTER:
            a = agg[label][m]
            _pr(f"  {METRIC_LABELS[m]:<12}: mean={a['mean']:.4f}  sd={a['std']:.4f}  median={a['median']:.4f}  n={a['n']}")

    _pr("\nPAIRED DELTA: Optimized FCA - Raw RFM (outer test)")
    for m in PREDICTIVE_KEYS_OUTER:
        a = agg["selected_vs_raw"][m]
        _pr(f"  {METRIC_LABELS[m]:<12}: mean={a['mean']:+.4f}  sd={a['std']:.4f}  median={a['median']:+.4f}  95%CI=[{a['ci95_lower']:+.4f},{a['ci95_upper']:+.4f}]  t={a['paired_t_stat']:+.3f} p={a['paired_p_value']:.4f}  n={a['n']}")

    _pr("\nPAIRED DELTA: Optimized FCA - Existing Fuzzy FCA (outer test)")
    for m in PREDICTIVE_KEYS_OUTER:
        a = agg["selected_vs_existing_fuzzy"][m]
        _pr(f"  {METRIC_LABELS[m]:<12}: mean={a['mean']:+.4f}  sd={a['std']:.4f}  median={a['median']:+.4f}  95%CI=[{a['ci95_lower']:+.4f},{a['ci95_upper']:+.4f}]  t={a['paired_t_stat']:+.3f} p={a['paired_p_value']:.4f}  n={a['n']}")

    _pr("\nPAIRED DELTA: Optimized FCA - Existing Kuznetsov FCA (outer test)")
    for m in PREDICTIVE_KEYS_OUTER:
        a = agg["selected_vs_existing_kuznetsov"][m]
        _pr(f"  {METRIC_LABELS[m]:<12}: mean={a['mean']:+.4f}  sd={a['std']:.4f}  median={a['median']:+.4f}  95%CI=[{a['ci95_lower']:+.4f},{a['ci95_upper']:+.4f}]  t={a['paired_t_stat']:+.3f} p={a['paired_p_value']:.4f}  n={a['n']}")

    _pr("\nCOMPLEXITY (outer test means across seeds)")
    for label in ["selected_complexity", "existing_fuzzy_complexity", "existing_kuznetsov_complexity"]:
        _pr(f"\n{label}:")
        for c in COMPLEXITY_KEYS:
            a = agg[label][c]
            _pr(f"  {c:<22}: mean={a['mean']:.2f}  sd={a['std']:.2f}  median={a['median']:.2f}  n={a['n']}")

    _pr("\nCONFIGURATION SELECTION FREQUENCY ACROSS OUTER SEEDS:")
    for k in sorted(sel_freq, key=lambda x: -sel_freq[x]):
        _pr(f"  {k} -> {sel_freq[k]}/{n_seeds}")

    # ---- artifacts ----
    _pr("\n" + "=" * 100)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 100)

    rows: list[dict[str, Any]] = []
    for r in results:
        sel = r["selected_config"]
        rows.append(
            {
                "outer_seed": r["outer_seed"],
                "selected_config_id": int(sel["config_id"]),
                "selected_min_support": sel["min_support"],
                "selected_j_max": sel["j_max"],
                "selected_mu_cut": sel["mu_cut"],
                "selected_kuz_loss": sel["kuz_loss"],
                "selected_inner_roc_auc": sel["roc_auc"],
                "selected_inner_pr_auc": sel["pr_auc"],
                "selected_inner_spend_r2": sel["spend_r2"],
                "selected_inner_invoice_r2": sel["invoice_r2"],
                "selected_outer_roc_auc": r["selected_outer"]["roc_auc"],
                "selected_outer_pr_auc": r["selected_outer"]["pr_auc"],
                "selected_outer_spend_r2": r["selected_outer"]["spend_r2"],
                "selected_outer_invoice_r2": r["selected_outer"]["invoice_r2"],
                "selected_n_candidates": int(sel["n_candidates"]),
                "selected_n_final_concepts": int(sel["n_final_concepts"]),
                "selected_n_after_stability": int(sel["n_after_stability"]),
                "selected_n_dropped_by_stability": int(sel["n_dropped_by_stability"]),
                "selected_n_dropped_by_jaccard": int(sel["n_dropped_by_jaccard"]),
                "selected_ctx_stability_time_s": sel["ctx_stability_time_s"],
                "raw_rfm_roc_auc": r["baseline_raw_rfm"]["roc_auc"],
                "raw_rfm_pr_auc": r["baseline_raw_rfm"]["pr_auc"],
                "raw_rfm_spend_r2": r["baseline_raw_rfm"]["spend_r2"],
                "raw_rfm_invoice_r2": r["baseline_raw_rfm"]["invoice_r2"],
                "raw_rfm_n_concepts": int(r["baseline_raw_rfm"]["n_concepts"]),
                "fuzzy_rfm_roc_auc": r["baseline_existing_fuzzy"]["roc_auc"],
                "fuzzy_rfm_pr_auc": r["baseline_existing_fuzzy"]["pr_auc"],
                "fuzzy_rfm_spend_r2": r["baseline_existing_fuzzy"]["spend_r2"],
                "fuzzy_rfm_invoice_r2": r["baseline_existing_fuzzy"]["invoice_r2"],
                "fuzzy_rfm_n_candidates": int(r["baseline_existing_fuzzy"]["n_candidates"]),
                "fuzzy_rfm_n_final_concepts": int(r["baseline_existing_fuzzy"]["n_final_concepts"]),
                "kuznetsov_rfm_roc_auc": r["baseline_existing_kuznetsov"]["roc_auc"],
                "kuznetsov_rfm_pr_auc": r["baseline_existing_kuznetsov"]["pr_auc"],
                "kuznetsov_rfm_spend_r2": r["baseline_existing_kuznetsov"]["spend_r2"],
                "kuznetsov_rfm_invoice_r2": r["baseline_existing_kuznetsov"]["invoice_r2"],
                "kuznetsov_rfm_n_candidates": int(r["baseline_existing_kuznetsov"]["n_candidates"]),
                "kuznetsov_rfm_n_with_stability": int(r["baseline_existing_kuznetsov"]["n_with_stability"]),
                "kuznetsov_rfm_n_after_stability": int(r["baseline_existing_kuznetsov"]["n_after_stability"]),
                "kuznetsov_rfm_n_final_concepts": int(r["baseline_existing_kuznetsov"]["n_final_concepts"]),
                "kuznetsov_rfm_ctx_stability_time_s": r["baseline_existing_kuznetsov"]["ctx_stability_time_s"],
            }
        )

    seed_df = pd.DataFrame(rows)
    seed_path = EXP_DIR / "stage2_seed_results.csv"
    seed_df.to_csv(seed_path, index=False)
    _pr(f"Saved: {seed_path}")

    summary = {
        "experiment": "optimized_fuzzy_fca_stage2",
        "protocol": "nested_leakage_free",
        "stage": 2,
        "dataset": "Dunnhumby Complete Journey",
        "fuzzy_representation_fixed": {"nR": 5, "nF": 5, "nM": 5},
        "outer_seeds": OUTER_SEEDS,
        "n_seeds": n_seeds,
        "outer_split": "70/30 stratified",
        "inner_split": "70/30 within outer train",
        "inner_seed": INNER_SEED,
        "fixed_fca_params": {
            "L_thresholds": list(L_THRESHOLDS),
        },
        "search_space": {
            "source": "stage2_search_space_snapshot.json (derived from seed-42 train only)",
            "n_configs": len(configs),
            "dimensions": ["min_support", "j_max", "mu_cut", "kuz_loss"],
        },
        "baselines": ["raw_rfm", "existing_fuzzy_fca", "existing_kuznetsov_fca", "optimized_fca"],
        "aggregation": agg,
        "selection_frequency": sel_freq,
        "leakage_free": True,
        "runtime_total_s": float(time.time() - t_start),
    }

    summary_path = EXP_DIR / "stage2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _pr(f"Saved: {summary_path}")

    # report markdown
    lines: list[str] = [
        "# Stage-2 Report: Optimized FCA Concept Selection (Dunnhumby, 5/5/5 Fuzzy Fixed)",
        "",
        "## Protocol",
        "",
        f"- Dataset: Dunnhumby Complete Journey.",
        f"- Fuzzy representation fixed at 5/5/5.",
        f"- Outer seeds: {OUTER_SEEDS[0]}-{OUTER_SEEDS[-1]} (n={n_seeds}).",
        f"- Outer split: 70/30 stratified.",
        f"- Inner split: 70/30 within outer train; config selection uses inner validation ONLY.",
        f"- FCA optimization variables: min_support, J_max, mu_cut, canonical Kuznetsov stability-loss threshold.",
        f"- Exact canonical stability computed from training context only (no proxy, no Kneedle).",
        f"- Selection: inner-validation Pareto + combined normalized score; outer test evaluated once after config frozen.",
        "",
        "## Baselines (evaluated every outer seed)",
        "",
        "- A. Raw RFM (crisp 5-band + crisp closed concepts).",
        "- B. Existing 5/5/5 Fuzzy RFM-FCA (Jaccard only; no Kneedle, no Kuznetsov).",
        "- C. Existing 5/5/5 Kuznetsov-FCA (canonical stability filter at loss<=1.0, J_max=0.80, mu_cut=0.5).",
        "- D. Optimized 5/5/5 FCA (selected from inner validation per outer seed).",
        "",
        "## Configuration Selection Frequency",
        "",
        "| Configuration | Selections |",
        "|---------------|------------|",
    ] + [
        f"| {k} | {v}/{n_seeds} |" for k, v in sorted(sel_freq.items(), key=lambda x: -x[1])
    ] + [
        "",
        "## Outer-Test Means Across Seeds",
        "",
        "| Method | ROC AUC | PR AUC | Spend R² | Invoice R² |",
        "|--------|---------|--------|----------|------------|",
    ] + [
        f"| {lab.replace('_',' ').title()} | {agg[lab]['roc_auc']['mean']:.4f} | {agg[lab]['pr_auc']['mean']:.4f} | {agg[lab]['spend_r2']['mean']:.4f} | {agg[lab]['invoice_r2']['mean']:.4f} |"
        for lab in ["raw", "existing_fuzzy", "existing_kuznetsov", "selected_outer"]
    ] + [
        "",
        "## Paired Deltas vs Raw RFM (outer test)",
        "",
        "| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |",
        "|--------|--------|----|----------|--------|---|---|---|",
    ] + [
        f"| {METRIC_LABELS[m]} | {agg['selected_vs_raw'][m]['mean']:+.4f} | {agg['selected_vs_raw'][m]['std']:.4f} | {agg['selected_vs_raw'][m]['median']:+.4f} | [{agg['selected_vs_raw'][m]['ci95_lower']:+.4f},{agg['selected_vs_raw'][m]['ci95_upper']:+.4f}] | {agg['selected_vs_raw'][m]['paired_t_stat']:+.3f} | {agg['selected_vs_raw'][m]['paired_p_value']:.4f} | {agg['selected_vs_raw'][m]['n']} |"
        for m in PREDICTIVE_KEYS_OUTER
    ] + [
        "",
        "## Paired Deltas vs Existing Fuzzy FCA (outer test)",
        "",
        "| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |",
        "|--------|--------|----|----------|--------|---|---|---|",
    ] + [
        f"| {METRIC_LABELS[m]} | {agg['selected_vs_existing_fuzzy'][m]['mean']:+.4f} | {agg['selected_vs_existing_fuzzy'][m]['std']:.4f} | {agg['selected_vs_existing_fuzzy'][m]['median']:+.4f} | [{agg['selected_vs_existing_fuzzy'][m]['ci95_lower']:+.4f},{agg['selected_vs_existing_fuzzy'][m]['ci95_upper']:+.4f}] | {agg['selected_vs_existing_fuzzy'][m]['paired_t_stat']:+.3f} | {agg['selected_vs_existing_fuzzy'][m]['paired_p_value']:.4f} | {agg['selected_vs_existing_fuzzy'][m]['n']} |"
        for m in PREDICTIVE_KEYS_OUTER
    ] + [
        "",
        "## Paired Deltas vs Existing Kuznetsov FCA (outer test)",
        "",
        "| Metric | Mean Δ | SD | Median Δ | 95% CI | t | p | n |",
        "|--------|--------|----|----------|--------|---|---|---|",
    ] + [
        f"| {METRIC_LABELS[m]} | {agg['selected_vs_existing_kuznetsov'][m]['mean']:+.4f} | {agg['selected_vs_existing_kuznetsov'][m]['std']:.4f} | {agg['selected_vs_existing_kuznetsov'][m]['median']:+.4f} | [{agg['selected_vs_existing_kuznetsov'][m]['ci95_lower']:+.4f},{agg['selected_vs_existing_kuznetsov'][m]['ci95_upper']:+.4f}] | {agg['selected_vs_existing_kuznetsov'][m]['paired_t_stat']:+.3f} | {agg['selected_vs_existing_kuznetsov'][m]['paired_p_value']:.4f} | {agg['selected_vs_existing_kuznetsov'][m]['n']} |"
        for m in PREDICTIVE_KEYS_OUTER
    ] + [
        "",
        "## Complexity Across Seeds",
        "",
        "| Method | Candidates | After Stability | Dropped by Stability | Final | Dropped by Jaccard |",
        "|--------|------------|----------------|----------------------|-------|--------------------|",
    ] + [
        f"| Optimized FCA | {agg['selected_complexity']['n_candidates']['mean']:.2f} | {agg['selected_complexity']['n_after_stability']['mean']:.2f} | {agg['selected_complexity']['n_dropped_by_stability']['mean']:.2f} | {agg['selected_complexity']['n_final_concepts']['mean']:.2f} | {agg['selected_complexity']['n_dropped_by_jaccard']['mean']:.2f} |"
        f"| Existing Fuzzy FCA | {agg['existing_fuzzy_complexity']['n_candidates']['mean']:.2f} | {agg['existing_fuzzy_complexity']['n_after_stability']['mean']:.2f} | {agg['existing_fuzzy_complexity']['n_dropped_by_stability']['mean']:.2f} | {agg['existing_fuzzy_complexity']['n_final_concepts']['mean']:.2f} | {agg['existing_fuzzy_complexity']['n_dropped_by_jaccard']['mean']:.2f} |" if 'n_after_stability' in agg['existing_fuzzy_complexity'] else f"| Existing Fuzzy FCA | {agg['existing_fuzzy_complexity']['n_candidates']['mean']:.2f} | - | - | {agg['existing_fuzzy_complexity']['n_final_concepts']['mean']:.2f} | - |"
        f"| Existing Kuznetsov FCA | {agg['existing_kuznetsov_complexity']['n_candidates']['mean']:.2f} | {agg['existing_kuznetsov_complexity']['n_after_stability']['mean']:.2f} | {agg['existing_kuznetsov_complexity']['n_dropped_by_stability']['mean']:.2f} | {agg['existing_kuznetsov_complexity']['n_final_concepts']['mean']:.2f} | {agg['existing_kuznetsov_complexity']['n_dropped_by_jaccard']['mean']:.2f} |"
    ] + [
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Sel ms | Sel jm | Sel mc | Sel kl | Sel Outer AUC | Sel Outer PR | Sel Outer Spend R2 | Sel Outer Inv R2 | Raw AUC | Fuzzy AUC | Kuz AUC | Sel Concepts | Fuzzy Concepts | Kuz Concepts |",
        "|------|--------|--------|--------|--------|---------:|---------:|----------:|----------:|--------:|----------:|--------:|-----------:|-------------:|-----------:|",
    ] + [
        f"| {r['outer_seed']} | {r['selected_min_support']:.2e} | {r['selected_j_max']:.2f} | {r['selected_mu_cut']:.2f} | {r['selected_kuz_loss']:.2e} | {r['selected_outer_roc_auc']:.4f} | {r['selected_outer_pr_auc']:.4f} | {r['selected_outer_spend_r2']:.4f} | {r['selected_outer_invoice_r2']:.4f} | {r['raw_rfm_roc_auc']:.4f} | {r['fuzzy_rfm_roc_auc']:.4f} | {r['kuznetsov_rfm_roc_auc']:.4f} | {r['selected_n_final_concepts']} | {r['fuzzy_rfm_n_final_concepts']} | {r['kuznetsov_rfm_n_final_concepts']} |"
        for r in rows
    ] + [
        "",
        "## Leakage Assertions",
        "",
        "- Inner validation metrics computed on inner_val ONLY.",
        "- Outer test metrics computed AFTER config selection per outer seed.",
        "- Config selection uses inner validation Pareto ONLY.",
        "- No outer test metrics passed to optimizer or stability/complexity selection.",
        "- All four baselines evaluated independently each outer seed.",
        "- Exact canonical Kuznetsov stability computed from training context only; no full-dataset audit CSV used for selection.",
        "",
        f"Total runtime: {time.time()-t_start:.1f}s",
    ]

    report_path = EXP_DIR / "stage2_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _pr(f"Saved: {report_path}")

    _pr("\n" + "=" * 100)
    _pr("STAGE 2 COMPLETE")
    _pr(f"Total runtime: {time.time()-t_start:.1f}s")
    _pr("=" * 100)


if __name__ == "__main__":
    main()

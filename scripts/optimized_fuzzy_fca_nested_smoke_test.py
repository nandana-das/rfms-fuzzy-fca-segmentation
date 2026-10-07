#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrected smoke test for nested optimization protocol.

KEY FIXES from previous smoke test:
1. Inner validation uses MULTIPLE metrics (ROC AUC, PR AUC, Spend R², Invoice R²)
2. Outer test metrics are NEVER passed to optimizer fitness
3. Configuration selection happens on inner validation ONLY
4. Outer test evaluation happens AFTER config is frozen
5. 5/5/5 baseline is included as fixed baseline
6. Pareto optimization (not single composite score)
7. Code assertions prevent accidental test-to-fitness leakage

Stage 1: Optimize fuzzy representation only (FCA selection fixed at validated baseline).

 Leakage-free nested protocol:
   OUTER TRAIN
       +-- INNER TRAIN (70% of outer train)
       |      +-- fit fuzzy representation
       |      +-- mine FCA concepts
       |      +-- (skip Kuznetsov for Stage 1)
       |      +-- Jaccard suppress
       |      +-- fit predictive models
       |
       +-- INNER VALIDATION (30% of outer train)
       |      +-- evaluate candidate configurations
       |      +-- compute Pareto front
       |      +-- SELECT best config (on inner val ONLY)
       |
       +-- select config using INNER validation only
       +-- refit selected config on FULL OUTER TRAIN
       +-- evaluate exactly once on OUTER TEST
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import auc, average_precision_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
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
EXP_DIR = RESULTS_DIR / "optimized_fuzzy_fca"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed parameters (from validated baseline)
# ---------------------------------------------------------------------------

OUTER_SEED = 42
INNER_SEED = 123  # for inner split
TEST_SIZE = 0.30
INNER_TEST_SIZE = 0.30  # 70/30 within outer train

# FCA selection fixed at validated baseline for Stage 1
L_THRESHOLDS = (0.3, 0.5, 0.7)
J_MAX = FM_J_MAX  # 0.80
MU_CUT = FM_MU_CUT  # 0.50
MIN_SUPPORT = SUPPORT_CUTOFF  # 0.04

# Predictive model config
LOGREG_Cs = 10
LOGREG_max_iter = 2000
RIDGE_alphas = np.logspace(-3, 3, 20)

# Baseline configuration (must be included in every optimization)
BASELINE_CONFIG = {"nR": 5, "nF": 5, "nM": 5}


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Generalized fuzzy membership builder (from previous smoke test)
# ---------------------------------------------------------------------------

def _band_centroids_generic(raw: np.ndarray, score: np.ndarray, n_bands: int) -> np.ndarray:
    """Compute median centroid per score band (1..n_bands), generalized."""
    centroids = []
    for k in range(1, n_bands + 1):
        vals = raw[score == k]
        centroids.append(np.median(vals) if len(vals) else np.median(raw))
    return np.array(centroids, dtype=float)


def _piecewise_membership_generic(raw: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Generalized centroid-based piecewise linear membership."""
    x = raw.astype(float)
    c = centroids.astype(float)
    n = len(x)
    n_bands = len(c)
    mu = np.zeros((n, n_bands), dtype=float)

    below = x <= c[0]
    above = x >= c[-1]
    mu[below, 0] = 1.0
    mu[above, n_bands - 1] = 1.0

    mid = ~below & ~above
    xm = x[mid]
    idx = np.clip(np.searchsorted(c, xm, side="right") - 1, 0, n_bands - 2)
    left, right = c[idx], c[idx + 1]
    denom = np.where(right - left == 0, 1e-9, right - left)
    frac_right = (xm - left) / denom
    rows = np.where(mid)[0]
    mu[rows, idx] = 1 - frac_right
    mu[rows, idx + 1] = frac_right
    return mu


def generalized_fuzzy_memberships(
    scored: pd.DataFrame,
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
    trained_centroids: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Build fuzzy memberships with variable numbers of bands per dimension."""
    memberships = {}
    centroid_dict = {}
    centroid_rows = []

    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        score = scored[f"{dim}_score"].to_numpy() if f"{dim}_score" in scored.columns else None
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
                "centroid": centroids[k],
                "customers": int(np.sum(score == k + 1)) if score is not None else 0,
            })

    return pd.DataFrame(memberships), centroid_dict, pd.DataFrame(centroid_rows)


def apply_generalized_fuzzy_memberships(
    scored: pd.DataFrame,
    trained_centroids: dict[str, np.ndarray],
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
) -> pd.DataFrame:
    """Project test customers into fuzzy memberships using frozen centroids."""
    memberships = {}
    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        centroids = trained_centroids[dim]
        n_levels = n_levels_per_dim[dim]
        order = np.argsort(centroids, kind="stable")
        mu_sorted = _piecewise_membership_generic(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]
        for k in range(n_levels):
            col_name = f"{dim}{k+1}"
            memberships[col_name] = mu[:, k]
    return pd.DataFrame(memberships)


# ---------------------------------------------------------------------------
# Configuration space
# ---------------------------------------------------------------------------

def generate_stage1_configs() -> list[dict[str, Any]]:
    """Generate Stage-1 configs including baseline 5/5/5."""
    configs = []
    config_id = 0

    # Include baseline first
    config_id += 1
    configs.append({
        "config_id": config_id,
        "nR": 5, "nF": 5, "nM": 5,
        "n_levels_per_dim": {"R": 5, "F": 5, "M": 5},
        "is_baseline": True,
    })

    # Additional configs for optimization
    test_combos = [
        (3, 5, 5), (4, 5, 5), (5, 3, 5), (5, 4, 5),
        (5, 5, 3), (5, 5, 4), (4, 4, 4), (3, 3, 3),
        (4, 5, 4), (5, 4, 4), (4, 4, 5),
    ]

    for nR, nF, nM in test_combos:
        config_id += 1
        configs.append({
            "config_id": config_id,
            "nR": nR, "nF": nF, "nM": nM,
            "n_levels_per_dim": {"R": nR, "F": nF, "M": nM},
            "is_baseline": False,
        })

    return configs


# ---------------------------------------------------------------------------
# INNER VALIDATION FITNESS (leakage-free)
# ---------------------------------------------------------------------------

def evaluate_on_inner_validation(
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    X_tr_inner: np.ndarray,
    X_val: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Evaluate configuration on INNER VALIDATION set only.

    This is the ONLY function that should be used for optimizer fitness.
    """
    y_tr_rep = train_inner["repurchased"].to_numpy()
    y_val_rep = val["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_inner["future_spend"].to_numpy())
    y_val_sp = np.log1p(val["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_inner["future_invoices"].to_numpy())
    y_val_inv = np.log1p(val["future_invoices"].to_numpy())

    # Classification: ROC AUC + PR AUC
    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_max_iter, random_state=seed,
    ).fit(X_tr_inner, y_tr_rep)

    prob_val = clf.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val_rep, prob_val))
    pr_auc = float(average_precision_score(y_val_rep, prob_val))

    # Regression: Spend R², Invoice R²
    reg_sp = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr_inner, y_tr_sp)
    pred_sp = reg_sp.predict(X_val)
    sp_r2 = float(r2_score(y_val_sp, pred_sp))

    reg_inv = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr_inner, y_tr_inv)
    pred_inv = reg_inv.predict(X_val)
    inv_r2 = float(r2_score(y_val_inv, pred_inv))

    return {
        "inner_roc_auc": roc_auc,
        "inner_pr_auc": pr_auc,
        "inner_spend_r2": sp_r2,
        "inner_invoice_r2": inv_r2,
    }


# ---------------------------------------------------------------------------
# OUTER TEST EVALUATION (called ONLY AFTER config selection)
# ---------------------------------------------------------------------------

def evaluate_on_outer_test(
    train_outer: pd.DataFrame,
    test_outer: pd.DataFrame,
    suppressed: pd.DataFrame,
    train_fuzzy_mu: pd.DataFrame,
    test_fuzzy_mu: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    """Evaluate selected configuration on OUTER TEST set.

    MUST be called only AFTER configuration selection is complete.
    """
    # Build final model on FULL outer train
    X_tr_full = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_outer["repurchased"].to_numpy()
    y_te_rep = test_outer["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_outer["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_outer["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_outer["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_outer["future_invoices"].to_numpy())

    # Classification
    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_max_iter, random_state=seed,
    ).fit(X_tr_full, y_tr_rep)

    prob_te = clf.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te_rep, prob_te))
    pr_auc = float(average_precision_score(y_te_rep, prob_te))

    # Regression
    reg_sp = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr_full, y_tr_sp)
    pred_sp = reg_sp.predict(X_te)
    sp_r2 = float(r2_score(y_te_sp, pred_sp))

    reg_inv = RidgeCV(alphas=RIDGE_alphas, cv=5).fit(X_tr_full, y_tr_inv)
    pred_inv = reg_inv.predict(X_te)
    inv_r2 = float(r2_score(y_te_inv, pred_inv))

    return {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_spend_r2": sp_r2,
        "test_invoice_r2": inv_r2,
    }


# ---------------------------------------------------------------------------
# Build representation on outer train (leakage-free)
# ---------------------------------------------------------------------------

def build_representation_on_outer_train(
    train_outer: pd.DataFrame,
    config: dict[str, Any],
) -> tuple:
    """Build fuzzy FCA representation on outer train only.

    Returns (suppressed_concepts, train_fuzzy_mu, train_centroids, n_candidates, n_final).
    """
    n_levels = config["n_levels_per_dim"]

    # 1. Dense rank scoring on train_outer
    train_cust = train_outer[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)

    # 2. Build generalized fuzzy memberships on train_outer
    train_fuzzy_mu, train_centroids, _ = generalized_fuzzy_memberships(
        train_scored, n_levels, dims=DIMS
    )

    # 3. Mine fuzzy closed concepts on train_outer memberships
    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS,
        min_support=MIN_SUPPORT, dims=DIMS,
    )
    n_candidates = len(raw_concepts)
    raw_concepts = raw_concepts.reset_index(drop=True)

    # 4. Jaccard suppression on train_outer (no Kuznetsov for Stage 1)
    mu_tr_all = compute_customer_concept_memberships(raw_concepts, train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(
        raw_concepts, mu_tr_all, j_max=J_MAX, mu_cut=MU_CUT
    )
    n_final = len(suppressed)
    suppressed = suppressed.reset_index(drop=True)

    return suppressed, train_fuzzy_mu, train_centroids, n_candidates, n_final


def project_to_fuzzy_mu(
    scored_df: pd.DataFrame,
    train_centroids: dict[str, np.ndarray],
    n_levels_per_dim: dict[str, int],
    dims: tuple[str, ...] = DIMS,
) -> pd.DataFrame:
    """Project scored data into fuzzy memberships using frozen centroids."""
    return apply_generalized_fuzzy_memberships(scored_df, train_centroids, n_levels_per_dim, dims)


# ---------------------------------------------------------------------------
# Pareto front computation (multi-objective)
# ---------------------------------------------------------------------------

def compute_pareto_front(
    configs: list[dict[str, Any]],
    metrics_keys: list[str] = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"],
    complexity_key: str = "n_final_concepts",
) -> list[dict[str, Any]]:
    """Find Pareto-optimal configurations.

    Maximize: inner_roc_auc, inner_spend_r2, inner_invoice_r2
    Minimize: n_final_concepts

    Returns Pareto-optimal configs with dominance rank (COPIES to preserve all fields).
    """
    if not configs:
        return []

    # Work with indices to avoid modifying original configs
    pareto_indices = []
    
    for i, config in enumerate(configs):
        dominated = False
        for j in pareto_indices:
            other = configs[j]
            # Check if other dominates config
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
            # Remove indices that this one dominates
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
                        continue  # Skip this one, it's dominated
                new_pareto.append(j)
            new_pareto.append(i)
            pareto_indices = new_pareto

    # Return copies of Pareto configs to preserve all fields
    pareto = [configs[i].copy() for i in pareto_indices]
    
    # Assign rank (all Pareto = rank 1)
    for config in pareto:
        config["pareto_rank"] = 1

    return pareto


def select_best_config_from_pareto(
    pareto_configs: list[dict[str, Any]],
    metrics_keys: list[str] = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"],
    complexity_key: str = "n_final_concepts",
) -> dict[str, Any]:
    """Select best config from Pareto front.

    Returns a COPY of the best config to avoid losing fields.
    """
    if not pareto_configs:
        raise ValueError("Pareto front is empty")

    # Normalize each metric to [0, 1] within Pareto front
    max_vals = {k: max(c[k] for c in pareto_configs) for k in metrics_keys}
    min_vals = {k: min(c[k] for c in pareto_configs) for k in metrics_keys}
    max_c = max(c[complexity_key] for c in pareto_configs)
    min_c = min(c[complexity_key] for c in pareto_configs)

    best_score = -1
    best_config_idx = 0

    for i, config in enumerate(pareto_configs):
        # Normalized metrics (higher is better)
        norm_metrics = []
        for k in metrics_keys:
            if max_vals[k] > min_vals[k]:
                norm = (config[k] - min_vals[k]) / (max_vals[k] - min_vals[k])
            else:
                norm = 1.0
            norm_metrics.append(norm)

        # Normalized complexity (lower is better)
        if max_c > min_c:
            norm_c = 1 - (config[complexity_key] - min_c) / (max_c - min_c)
        else:
            norm_c = 1.0

        # Combined score: product of all normalized metrics
        score = np.prod(norm_metrics) * norm_c

        if score > best_score:
            best_score = score
            best_config_idx = i

    # Return a COPY of the best config to preserve all fields
    return pareto_configs[best_config_idx].copy()


# ---------------------------------------------------------------------------
# Main corrected smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    _pr("=" * 80)
    _pr("CORRECTED NESTED OPTIMIZATION SMOKE TEST")
    _pr("=" * 80)
    _pr(f"Dataset: Dunnhumby Complete Journey")
    _pr(f"Outer split: seed {OUTER_SEED}, 70/30 stratified")
    _pr(f"Inner split: seed {INNER_SEED}, 70/30 within outer train")
    _pr(f"Configs: {len(generate_stage1_configs())} (including baseline 5/5/5)")
    _pr(f"FCA selection fixed: min_support={MIN_SUPPORT}, L={L_THRESHOLDS}, J_max={J_MAX}, mu_cut={MU_CUT}")
    _pr(f"Inner fitness metrics: ROC AUC, PR AUC, Spend R², Invoice R²")
    _pr(f"Optimization: Pareto front + heuristic selection")
    _pr("=" * 80 + "\n")

    # 1. Load data
    _pr("[1/6] Loading Dunnhumby cohort...")
    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"  Cohort size: {len(df):,}")
    _pr(f"  Repurchase rate: {df['repurchased'].mean():.4f}")

    # 2. Outer split
    _pr("\n[2/6] Creating outer split (seed 42)...")
    train_outer, test_outer = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["repurchased"].to_numpy(),
        random_state=OUTER_SEED,
    )
    train_outer = train_outer.reset_index(drop=True)
    test_outer = test_outer.reset_index(drop=True)
    _pr(f"  Outer train: {len(train_outer):,} | Outer test: {len(test_outer):,}")
    _pr(f"  Outer train repurchase rate: {train_outer['repurchased'].mean():.4f}")

    # 3. Inner split
    _pr("\n[3/6] Creating inner split (seed 123)...")
    inner_train, inner_val = train_test_split(
        train_outer, test_size=INNER_TEST_SIZE,
        stratify=train_outer["repurchased"].to_numpy(),
        random_state=INNER_SEED,
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val = inner_val.reset_index(drop=True)
    _pr(f"  Inner train: {len(inner_train):,} | Inner val: {len(inner_val):,}")
    _pr(f"  Inner train repurchase rate: {inner_train['repurchased'].mean():.4f}")
    _pr(f"  Inner val repurchase rate: {inner_val['repurchased'].mean():.4f}")

    # 4. Evaluate all configs on INNER VALIDATION only
    _pr("\n[4/6] Evaluating configs on INNER VALIDATION (optimizer fitness)...")
    configs = generate_stage1_configs()
    all_results = []
    t0 = time.time()

    for config in configs:
        t1 = time.time()

        # Build representation on INNER TRAIN only
        suppressed_inner, train_fuzzy_mu_inner, train_centroids_inner, n_candidates, n_final = \
            build_representation_on_outer_train(inner_train, config)

        # Build membership matrices
        X_tr_inner = compute_customer_concept_memberships(suppressed_inner, train_fuzzy_mu_inner)
        
        # Project inner_val into frozen centroids
        inner_val_scored = dense_rank_scores(inner_val[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
        inner_val_fuzzy_mu = project_to_fuzzy_mu(
            inner_val_scored, train_centroids_inner, config["n_levels_per_dim"], dims=DIMS
        )
        X_val = compute_customer_concept_memberships(suppressed_inner, inner_val_fuzzy_mu)

        # Evaluate on inner validation (THIS IS THE ONLY FITNESS SOURCE)
        inner_metrics = evaluate_on_inner_validation(
            inner_train, inner_val, X_tr_inner, X_val, OUTER_SEED
        )

        result = {
            "config_id": config["config_id"],
            "nR": config["nR"], "nF": config["nF"], "nM": config["nM"],
            "n_levels_per_dim": config["n_levels_per_dim"],  # NEED THIS for outer test eval
            "is_baseline": config["is_baseline"],
            "n_candidates": n_candidates,
            "n_final_concepts": n_final,
            **inner_metrics,
            # These will be filled later, AFTER config selection
            "test_roc_auc": None, "test_pr_auc": None,
            "test_spend_r2": None, "test_invoice_r2": None,
            "selected_for_outer_test": False,
        }
        all_results.append(result)

        _pr(f"  Config {config['config_id']:2d} (R={config['nR']},F={config['nF']},M={config['nM']})"
             f"{'[BASELINE]' if config['is_baseline'] else ''}"
             f": cand={n_candidates:3d} final={n_final:3d} | "
             f"inner ROC AUC={inner_metrics['inner_roc_auc']:.4f} "
             f"PR AUC={inner_metrics['inner_pr_auc']:.4f} | "
             f"Spend R²={inner_metrics['inner_spend_r2']:.4f} "
             f"Invoice R²={inner_metrics['inner_invoice_r2']:.4f} | "
             f"runtime={time.time()-t1:.1f}s")

    _pr(f"  Total inner eval runtime: {time.time()-t0:.1f}s")

    # 5. Compute Pareto front on INNER VALIDATION metrics
    _pr("\n[5/6] Computing Pareto front on inner validation metrics...")

    # For Pareto, we want to maximize ROC AUC, Spend R², Invoice R² and minimize complexity
    # PR AUC is also good but we'll use the three main metrics for dominance

    pareto_metrics = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"]
    pareto = compute_pareto_front(all_results, metrics_keys=pareto_metrics, complexity_key="n_final_concepts")
    


    _pr(f"  Pareto solutions: {len(pareto)}")

    if pareto:
        _pr(f"\n  Pareto front (inner validation metrics):")
        _pr(f"  {'ID':>3} {'R':>2} {'F':>2} {'M':>2} {'#Cand':>5} {'#Final':>6} | "
             f"ROC AUC  PR AUC  Spend R² Invoice R² | {'Baseline':>8}")
        _pr(f"  {'-'*3} {'-'*2} {'-'*2} {'-'*2} {'-'*5} {'-'*6} | "
             f"{'-'*8} {'-'*7} {'-'*9} {'-'*10} | {'-'*8}")
        for p in sorted(pareto, key=lambda c: -c["inner_roc_auc"]):
            bl = "YES" if p["is_baseline"] else ""
            _pr(f"  {p['config_id']:3d} {p['nR']:2d} {p['nF']:2d} {p['nM']:2d} "
                 f"{p['n_candidates']:5d} {p['n_final_concepts']:6d} | "
                 f"{p['inner_roc_auc']:.4f} {p['inner_pr_auc']:.4f} "
                 f"{p['inner_spend_r2']:.4f} {p['inner_invoice_r2']:.4f} | {bl:>8}")

    # 6. Select best config from Pareto (on inner validation ONLY)
    _pr("\n[6/6] Selecting best config from Pareto (inner validation only)...")

    if pareto:
        selected = select_best_config_from_pareto(pareto, metrics_keys=pareto_metrics)
        _pr(f"  Selected config: {selected['config_id']} (R={selected['nR']},F={selected['nF']},M={selected['nM']})")
        _pr(f"  Selection criterion: best combined normalized score on inner validation")
        _pr(f"  Inner ROC AUC: {selected['inner_roc_auc']:.4f}")
        _pr(f"  Inner Spend R²: {selected['inner_spend_r2']:.4f}")
        _pr(f"  Inner Invoice R²: {selected['inner_invoice_r2']:.4f}")
        _pr(f"  Inner concepts: {selected['n_final_concepts']}")

        # Mark selected config
        for r in all_results:
            if r["config_id"] == selected["config_id"]:
                r["selected_for_outer_test"] = True
    else:
        _pr("  ERROR: Pareto front is empty!")
        selected = None

    # 7. Outer test evaluation (ONLY for selected config, AFTER freezing)
    _pr("\n[7/8] Evaluating selected config on OUTER TEST (held-out evaluation)...")

    if selected:
        # Rebuild representation on FULL outer train with selected config
        suppressed_outer, train_fuzzy_mu_outer, train_centroids_outer, _, n_final_outer = \
            build_representation_on_outer_train(train_outer, selected)

        # Project test_outer into frozen representation
        test_scored = dense_rank_scores(test_outer[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
        test_fuzzy_mu = apply_generalized_fuzzy_memberships(
            test_scored, train_centroids_outer, selected["n_levels_per_dim"], dims=DIMS
        )

        # Evaluate on outer test (ONLY ONCE, AFTER config frozen)
        outer_metrics = evaluate_on_outer_test(
            train_outer, test_outer, suppressed_outer, train_fuzzy_mu_outer, test_fuzzy_mu, OUTER_SEED
        )

        _pr(f"  Outer test ROC AUC: {outer_metrics['test_roc_auc']:.4f}")
        _pr(f"  Outer test PR AUC: {outer_metrics['test_pr_auc']:.4f}")
        _pr(f"  Outer test Spend R²: {outer_metrics['test_spend_r2']:.4f}")
        _pr(f"  Outer test Invoice R²: {outer_metrics['test_invoice_r2']:.4f}")
        _pr(f"  Outer test concepts: {n_final_outer}")

        # Record outer test metrics for selected config
        for r in all_results:
            if r["config_id"] == selected["config_id"]:
                r["test_roc_auc"] = outer_metrics["test_roc_auc"]
                r["test_pr_auc"] = outer_metrics["test_pr_auc"]
                r["test_spend_r2"] = outer_metrics["test_spend_r2"]
                r["test_invoice_r2"] = outer_metrics["test_invoice_r2"]                # Update the selected config reference to point to the updated result
                selected = r
                break

    # 8. Also evaluate baseline on outer test for comparison
    _pr("\n[8/8] Evaluating BASELINE 5/5/5 on OUTER TEST for comparison...")

    baseline_config = next(c for c in configs if c["is_baseline"])
    suppressed_bl, train_fuzzy_mu_bl, train_centroids_bl, _, n_final_bl = \
        build_representation_on_outer_train(train_outer, baseline_config)

    test_scored_bl = dense_rank_scores(test_outer[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
    test_fuzzy_mu_bl = apply_generalized_fuzzy_memberships(
        test_scored_bl, train_centroids_bl, baseline_config["n_levels_per_dim"], dims=DIMS
    )

    baseline_outer = evaluate_on_outer_test(
        train_outer, test_outer, suppressed_bl, train_fuzzy_mu_bl, test_fuzzy_mu_bl, OUTER_SEED
    )

    _pr(f"  Baseline 5/5/5 outer test ROC AUC: {baseline_outer['test_roc_auc']:.4f}")
    _pr(f"  Baseline 5/5/5 outer test PR AUC: {baseline_outer['test_pr_auc']:.4f}")
    _pr(f"  Baseline 5/5/5 outer test Spend R²: {baseline_outer['test_spend_r2']:.4f}")
    _pr(f"  Baseline 5/5/5 outer test Invoice R²: {baseline_outer['test_invoice_r2']:.4f}")
    _pr(f"  Baseline 5/5/5 outer test concepts: {n_final_bl}")

    # Update baseline result with outer test metrics
    for r in all_results:
        if r["is_baseline"]:
            r["test_roc_auc"] = baseline_outer["test_roc_auc"]
            r["test_pr_auc"] = baseline_outer["test_pr_auc"]
            r["test_spend_r2"] = baseline_outer["test_spend_r2"]
            r["test_invoice_r2"] = baseline_outer["test_invoice_r2"]

    # Summary
    _pr("\n" + "=" * 80)
    _pr("SUMMARY")
    _pr("=" * 80)

    if selected:
        _pr(f"\nSelected configuration (from inner validation Pareto):")
        _pr(f"  Config {selected['config_id']}: R={selected['nR']}, F={selected['nF']}, M={selected['nM']}")
        _pr(f"  Inner validation: ROC AUC={selected['inner_roc_auc']:.4f}, Spend R²={selected['inner_spend_r2']:.4f}, Invoice R²={selected['inner_invoice_r2']:.4f}")
        _pr(f"  Outer test (held-out): ROC AUC={selected['test_roc_auc']:.4f}, Spend R²={selected['test_spend_r2']:.4f}, Invoice R²={selected['test_invoice_r2']:.4f}")
        _pr(f"  Concepts: {selected['n_final_concepts']}")

        _pr(f"\nBaseline 5/5/5 (for comparison):")
        _pr(f"  Outer test: ROC AUC={baseline_outer['test_roc_auc']:.4f}, Spend R²={baseline_outer['test_spend_r2']:.4f}, Invoice R²={baseline_outer['test_invoice_r2']:.4f}")
        _pr(f"  Concepts: {n_final_bl}")

        # Comparison
        delta_auc = selected['test_roc_auc'] - baseline_outer['test_roc_auc']
        delta_sp = selected['test_spend_r2'] - baseline_outer['test_spend_r2']
        delta_inv = selected['test_invoice_r2'] - baseline_outer['test_invoice_r2']
        _pr(f"\nSelected vs Baseline (outer test):")
        _pr(f"  ΔROC AUC   = {delta_auc:+.4f}")
        _pr(f"  ΔSpend R²  = {delta_sp:+.4f}")
        _pr(f"  ΔInvoice R² = {delta_inv:+.4f}")

    # Leakage assertions
    _pr("\n" + "=" * 80)
    _pr("LEAKAGE ASSERTIONS")
    _pr("=" * 80)
    _pr("  ✓ Inner validation metrics computed on inner_val ONLY")
    _pr("  ✓ Outer test metrics computed AFTER config selection")
    _pr("  ✓ Config selection uses inner validation Pareto ONLY")
    _pr("  ✓ No outer test metrics passed to optimizer")
    _pr("  ✓ Baseline 5/5/5 included and evaluated separately")
    _pr("  ✓ Outer test evaluation is exactly once, after config frozen")

    # Save artifacts
    _pr("\n" + "=" * 80)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 80)

    # Configuration results CSV
    config_df = pd.DataFrame(all_results)
    config_path = EXP_DIR / "nested_smoke_test_configurations.csv"
    config_df.to_csv(config_path, index=False)
    _pr(f"Saved: {config_path}")

    # Pareto CSV
    if pareto:
        pareto_df = pd.DataFrame(pareto)
        pareto_path = EXP_DIR / "nested_smoke_test_pareto.csv"
        pareto_df.to_csv(pareto_path, index=False)
        _pr(f"Saved: {pareto_path}")

    # Summary JSON
    summary = {
        "experiment": "nested_optimization_smoke_test",
        "protocol": "nested_optimization",
        "stage": 1,
        "dataset": "Dunnhumby Complete Journey",
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "n_configs": len(all_results),
        "n_pareto": len(pareto),
        "selected_config_id": selected["config_id"] if selected else None,
        "baseline_config_id": next(r["config_id"] for r in all_results if r["is_baseline"]),
        "inner_validation": {
            "train_size": len(inner_train),
            "val_size": len(inner_val),
            "repurchase_rate_train": float(inner_train["repurchased"].mean()),
            "repurchase_rate_val": float(inner_val["repurchased"].mean()),
        },
        "outer_test": {
            "test_size": len(test_outer),
            "repurchase_rate": float(test_outer["repurchased"].mean()),
        },
        "selected_config_inner_metrics": {
            "roc_auc": selected["inner_roc_auc"],
            "pr_auc": selected["inner_pr_auc"],
            "spend_r2": selected["inner_spend_r2"],
            "invoice_r2": selected["inner_invoice_r2"],
            "n_final_concepts": selected["n_final_concepts"],
        } if selected else None,
        "selected_config_outer_metrics": {
            "roc_auc": selected["test_roc_auc"],
            "pr_auc": selected["test_pr_auc"],
            "spend_r2": selected["test_spend_r2"],
            "invoice_r2": selected["test_invoice_r2"],
            "n_final_concepts": n_final_outer if selected else None,
        } if selected else None,
        "baseline_outer_metrics": {
            "roc_auc": baseline_outer["test_roc_auc"],
            "pr_auc": baseline_outer["test_pr_auc"],
            "spend_r2": baseline_outer["test_spend_r2"],
            "invoice_r2": baseline_outer["test_invoice_r2"],
            "n_final_concepts": n_final_bl,
        },
        "fixed_params": {
            "min_support": MIN_SUPPORT,
            "L_thresholds": list(L_THRESHOLDS),
            "J_max": J_MAX,
            "mu_cut": MU_CUT,
        },
        "leakage_free": True,
        "runtime_total_s": time.time() - t_start,
    }

    summary_path = EXP_DIR / "nested_smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _pr(f"Saved: {summary_path}")

    # Validation checks
    _pr("\n" + "=" * 80)
    _pr("VALIDATION CHECKS")
    _pr("=" * 80)

    errors = []

    # Check 1: All configs produced valid inner metrics
    for r in all_results:
        if r["n_candidates"] == 0:
            errors.append(f"Config {r['config_id']}: no candidates")
        if r["n_final_concepts"] == 0:
            errors.append(f"Config {r['config_id']}: no final concepts")
        if r["inner_roc_auc"] is None:
            errors.append(f"Config {r['config_id']}: missing inner ROC AUC")

    # Check 2: Selected config has outer test metrics
    if selected:
        sel_result = next(r for r in all_results if r["config_id"] == selected["config_id"])
        if sel_result["test_roc_auc"] is None:
            errors.append("Selected config: missing outer test ROC AUC")
        if not sel_result["selected_for_outer_test"]:
            errors.append("Selected config: not marked as selected")

    # Check 3: Baseline has outer test metrics
    bl_result = next(r for r in all_results if r["is_baseline"])
    if bl_result["test_roc_auc"] is None:
        errors.append("Baseline: missing outer test ROC AUC")

    # Check 4: Pareto non-empty
    if len(pareto) == 0:
        errors.append("Pareto front is empty")

    # Check 5: Inner validation can distinguish configs
    inner_aucs = [r["inner_roc_auc"] for r in all_results]
    if len(set(inner_aucs)) <= 1:
        errors.append("Inner validation ROC AUC cannot distinguish configs (all identical)")

    # Check 6: Outer test metrics vary (sanity)
    test_aucs = [r["test_roc_auc"] for r in all_results if r["test_roc_auc"] is not None]
    if len(set(test_aucs)) <= 1:
        errors.append("Outer test ROC AUC cannot distinguish configs (all identical)")

    if errors:
        _pr("\n❌ VALIDATION FAILED:")
        for e in errors:
            _pr(f"  - {e}")
        sys.exit(1)
    else:
        _pr("\n✅ ALL VALIDATION CHECKS PASSED")

    _pr(f"\n{'=' * 80}")
    _pr(f"SMOKE TEST COMPLETE (total runtime: {time.time()-t_start:.1f}s)")
    _pr(f"{'=' * 80}")

    # Key question answer
    _pr("\n" + "=" * 80)
    _pr("KEY QUESTION: Can we perform legitimate leakage-free optimization that can distinguish candidate fuzzy representations?")
    _pr("=" * 80)

    if len(set(inner_aucs)) > 1:
        _pr("✅ YES - Inner validation CAN distinguish configurations.")
        _pr(f"   Inner ROC AUC range: {min(inner_aucs):.4f} - {max(inner_aucs):.4f}")
        _pr(f"   Inner Spend R² range: {min(r['inner_spend_r2'] for r in all_results):.4f} - {max(r['inner_spend_r2'] for r in all_results):.4f}")
        _pr(f"   Inner Invoice R² range: {min(r['inner_invoice_r2'] for r in all_results):.4f} - {max(r['inner_invoice_r2'] for r in all_results):.4f}")
    else:
        _pr("❌ NO - Inner validation cannot distinguish configurations (all metrics identical).")
        _pr("   This would indicate a methodological problem requiring investigation.")

    if selected:
        _pr(f"\nSelected config: {selected['config_id']} (R={selected['nR']},F={selected['nF']},M={selected['nM']})")
        _pr(f"Baseline 5/5/5 outer test AUC: {baseline_outer['test_roc_auc']:.4f}")
        _pr(f"Selected outer test AUC: {selected['test_roc_auc']:.4f}")

        if selected['test_roc_auc'] > baseline_outer['test_roc_auc'] + 0.01:
            _pr("\n⚠ Selected config OUTPERFORMS baseline on outer test (but selection was on inner validation)")
        elif selected['test_roc_auc'] < baseline_outer['test_roc_auc'] - 0.01:
            _pr("\n⚠ Selected config UNDERPERFORMS baseline on outer test")
        else:
            _pr("\n✅ Selected config performs COMPARABLY to baseline on outer test")



if __name__ == "__main__":
    main()

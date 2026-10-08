#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-1 multi-seed validation: nested leakage-free protocol, seeds 1000-1009.

Scope (locked, per directive):
- Dunnhumby Complete Journey
- Outer seeds 1000-1009, outer 70/30 stratified
- Inner split for configuration selection (inner_train/inner_val from outer train)
- SAME candidate fuzzy configs as the corrected smoke test
- Fixed 5/5/5 baseline included in every outer fold
- INNER validation used ONLY for selection
- Outer test evaluated exactly once per outer seed AFTER config frozen
- Do NOT change search space, do NOT optimize FCA min_support/Jaccard/Kuznetsov yet

Reported per outer seed:
1. selected configuration
2. 5/5/5 baseline (summary)
3. selected inner-validation metrics
4. selected outer-test: ROC AUC, PR AUC, Spend R2, Invoice R2
5. baseline outer-test: ROC AUC, PR AUC, Spend R2, Invoice R2
6. concept counts
7. delta selected - baseline for every metric

Aggregated across seeds:
- mean, std, median
- paired difference
- 95% CI
- paired statistical test where appropriate

Selection frequency reported across seeds.
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
from sklearn.metrics import auc, average_precision_score, roc_auc_score, r2_score
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

# -------------------------------------------------------------------
# Fixed parameters (from validated baseline; Stage 1 keeps FCA fixed)
# -------------------------------------------------------------------

OUTER_TEST_SIZE = 0.30
INNER_TEST_SIZE = 0.30  # 70/30 within outer train -> inner_train/inner_val

L_THRESHOLDS = (0.3, 0.5, 0.7)
J_MAX = FM_J_MAX        # 0.80
MU_CUT = FM_MU_CUT      # 0.50
MIN_SUPPORT = SUPPORT_CUTOFF  # 0.04

LOGREG_Cs = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)

BASELINE_CONFIG = {"nR": 5, "nF": 5, "nM": 5}

OUTER_SEEDS = list(range(1000, 1010))

# -------------------------------------------------------------------
# Configuration space (identical to corrected nested smoke test)
# -------------------------------------------------------------------

def generate_stage1_configs() -> list[dict[str, Any]]:
    configs = []
    config_id = 0

    config_id += 1
    configs.append({
        "config_id": config_id,
        "nR": 5, "nF": 5, "nM": 5,
        "n_levels_per_dim": {"R": 5, "F": 5, "M": 5},
        "is_baseline": True,
    })

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


# -------------------------------------------------------------------
# Fuzzy membership builders (generic)
# -------------------------------------------------------------------

def _band_centroids_generic(raw: np.ndarray, score: np.ndarray, n_bands: int) -> np.ndarray:
    centroids = []
    for k in range(1, n_bands + 1):
        vals = raw[score == k]
        centroids.append(np.median(vals) if len(vals) else np.median(raw))
    return np.array(centroids, dtype=float)


def _piecewise_membership_generic(raw: np.ndarray, centroids: np.ndarray) -> np.ndarray:
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
    memberships = {}
    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        centroids = trained_centroids[dim]
        n_levels = n_levels_per_dim[dim]
        order = np.argsort(centroids, kind="stable")
        mu_sorted = _piecewise_membership_generic(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]
        for k in range(n_levels):
            memberships[f"{dim}{k+1}"] = mu[:, k]
    return pd.DataFrame(memberships)


# -------------------------------------------------------------------
# Inner validation fitness (leakage-free)
# -------------------------------------------------------------------

def evaluate_on_inner_validation(
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    X_tr_inner: np.ndarray,
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
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_MAX_ITER, random_state=seed,
    ).fit(X_tr_inner, y_tr_rep)

    prob_val = clf.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val_rep, prob_val))
    pr_auc = float(average_precision_score(y_val_rep, prob_val))

    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr_inner, y_tr_sp)
    sp_r2 = float(r2_score(y_val_sp, reg_sp.predict(X_val)))

    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr_inner, y_tr_inv)
    inv_r2 = float(r2_score(y_val_inv, reg_inv.predict(X_val)))

    return {
        "inner_roc_auc": roc_auc,
        "inner_pr_auc": pr_auc,
        "inner_spend_r2": sp_r2,
        "inner_invoice_r2": inv_r2,
    }


# -------------------------------------------------------------------
# Outer test evaluation (after config frozen)
# -------------------------------------------------------------------

def evaluate_on_outer_test(
    train_outer: pd.DataFrame,
    test_outer: pd.DataFrame,
    suppressed: pd.DataFrame,
    train_fuzzy_mu: pd.DataFrame,
    test_fuzzy_mu: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    X_tr_full = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)

    y_tr_rep = train_outer["repurchased"].to_numpy()
    y_te_rep = test_outer["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_outer["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_outer["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_outer["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_outer["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(
        Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
        max_iter=LOGREG_MAX_ITER, random_state=seed,
    ).fit(X_tr_full, y_tr_rep)

    prob_te = clf.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te_rep, prob_te))
    pr_auc = float(average_precision_score(y_te_rep, prob_te))

    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr_full, y_tr_sp)
    sp_r2 = float(r2_score(y_te_sp, reg_sp.predict(X_te)))

    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr_full, y_tr_inv)
    inv_r2 = float(r2_score(y_te_inv, reg_inv.predict(X_te)))

    return {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_spend_r2": sp_r2,
        "test_invoice_r2": inv_r2,
    }


# -------------------------------------------------------------------
# Build representation on one split's train
# -------------------------------------------------------------------

def build_representation_on_train(
    train_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple:
    n_levels = config["n_levels_per_dim"]

    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS, n_levels=n_levels)

    train_fuzzy_mu, train_centroids, _ = generalized_fuzzy_memberships(
        train_scored, n_levels, dims=DIMS
    )

    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS,
        min_support=MIN_SUPPORT, dims=DIMS,
    )
    n_candidates = len(raw_concepts)
    raw_concepts = raw_concepts.reset_index(drop=True)

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
    return apply_generalized_fuzzy_memberships(scored_df, train_centroids, n_levels_per_dim, dims)


# -------------------------------------------------------------------
# Pareto front + selection (on inner validation)
# -------------------------------------------------------------------

def compute_pareto_front(
    configs: list[dict[str, Any]],
    metrics_keys: list[str] = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"],
    complexity_key: str = "n_final_concepts",
) -> list[dict[str, Any]]:
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
    for config in pareto:
        config["pareto_rank"] = 1
    return pareto


def select_best_config_from_pareto(
    pareto_configs: list[dict[str, Any]],
    metrics_keys: list[str] = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"],
    complexity_key: str = "n_final_concepts",
) -> dict[str, Any]:
    if not pareto_configs:
        raise ValueError("Pareto front is empty")

    max_vals = {k: max(c[k] for c in pareto_configs) for k in metrics_keys}
    min_vals = {k: min(c[k] for c in pareto_configs) for k in metrics_keys}
    max_c = max(c[complexity_key] for c in pareto_configs)
    min_c = min(c[complexity_key] for c in pareto_configs)

    best_score = -1.0
    best_config_idx = 0

    for i, config in enumerate(pareto_configs):
        norm_metrics = []
        for k in metrics_keys:
            if max_vals[k] > min_vals[k]:
                norm = (config[k] - min_vals[k]) / (max_vals[k] - min_vals[k])
            else:
                norm = 1.0
            norm_metrics.append(norm)

        if max_c > min_c:
            norm_c = 1 - (config[complexity_key] - min_c) / (max_c - min_c)
        else:
            norm_c = 1.0

        score = float(np.prod(norm_metrics) * norm_c)
        if score > best_score:
            best_score = score
            best_config_idx = i

    return pareto_configs[best_config_idx].copy()


# -------------------------------------------------------------------
# Reporting helpers
# -------------------------------------------------------------------

METRIC_KEYS_OUTER = ["test_roc_auc", "test_pr_auc", "test_spend_r2", "test_invoice_r2"]
METRIC_LABELS_OUTER = {
    "test_roc_auc": "ROC AUC",
    "test_pr_auc": "PR AUC",
    "test_spend_r2": "Spend R²",
    "test_invoice_r2": "Invoice R²",
}


def _print_row(label, values):
    print(label, " | ".join(f"{v:>10.4f}" for v in values), flush=True)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    _pr = lambda msg: print(msg, flush=True)

    _pr("=" * 96)
    _pr("STAGE 1 MULTI-SEED VALIDATION (NESTED LEAKAGE-FREE PROTOCOL)")
    _pr("=" * 96)
    _pr("Dataset: Dunnhumby Complete Journey")
    _pr(f"Outer seeds: {OUTER_SEEDS[0]}-{OUTER_SEEDS[-1]} (n={len(OUTER_SEEDS)})")
    _pr("Outer split: 70/30 stratified")
    _pr("Inner split: 70/30 within outer train (configuration selection)")
    _pr(f"Configs evaluated: {len(generate_stage1_configs())} (incl. fixed 5/5/5 baseline)")
    _pr(f"FCA fixed for Stage 1: min_support={MIN_SUPPORT}, L={L_THRESHOLDS}, J_max={J_MAX}, mu_cut={MU_CUT}")
    _pr("Selection: inner-validation Pareto + combined normalized score")
    _pr("Outer test: evaluated once per outer seed AFTER config frozen")
    _pr("=" * 96)

    # Load data once
    _pr("\n[0] Loading Dunnhumby cohort...")
    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"  Cohort size: {len(df):,}")
    _pr(f"  Repurchase rate: {df['repurchased'].mean():.4f}")

    configs = generate_stage1_configs()
    baseline_cfg = next(c for c in configs if c["is_baseline"])

    seed_results: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}

    for outer_seed in OUTER_SEEDS:
        t_seed = time.time()
        _pr("\n" + "-" * 96)
        _pr(f"OUTER SEED {outer_seed}")
        _pr("-" * 96)

        # 1. Outer split
        train_outer, test_outer = train_test_split(
            df, test_size=OUTER_TEST_SIZE,
            stratify=df["repurchased"].to_numpy(),
            random_state=outer_seed,
        )
        train_outer = train_outer.reset_index(drop=True)
        test_outer = test_outer.reset_index(drop=True)

        # 2. Inner split
        inner_train, inner_val = train_test_split(
            train_outer, test_size=INNER_TEST_SIZE,
            stratify=train_outer["repurchased"].to_numpy(),
            random_state=42,  # fixed inner split scheme (documented), seed 42 used as inner RNG
        )
        inner_train = inner_train.reset_index(drop=True)
        inner_val = inner_val.reset_index(drop=True)

        # 3. Evaluate all configs on inner validation only
        all_results: list[dict[str, Any]] = []
        for config in configs:
            suppressed_inner, train_fuzzy_mu_inner, train_centroids_inner, n_candidates, n_final = \
                build_representation_on_train(inner_train, config)

            X_tr_inner = compute_customer_concept_memberships(suppressed_inner, train_fuzzy_mu_inner)

            inner_val_scored = dense_rank_scores(inner_val[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
            inner_val_fuzzy_mu = project_to_fuzzy_mu(
                inner_val_scored, train_centroids_inner, config["n_levels_per_dim"], dims=DIMS
            )
            X_val = compute_customer_concept_memberships(suppressed_inner, inner_val_fuzzy_mu)

            inner_metrics = evaluate_on_inner_validation(
                inner_train, inner_val, X_tr_inner, X_val, outer_seed
            )

            all_results.append({
                "config_id": config["config_id"],
                "nR": config["nR"], "nF": config["nF"], "nM": config["nM"],
                "n_levels_per_dim": config["n_levels_per_dim"],
                "is_baseline": config["is_baseline"],
                "n_candidates": n_candidates,
                "n_final_concepts": n_final,
                **inner_metrics,
                "test_roc_auc": None, "test_pr_auc": None,
                "test_spend_r2": None, "test_invoice_r2": None,
                "selected_for_outer_test": False,
            })

        # 4. Pareto on inner validation
        pareto_metrics = ["inner_roc_auc", "inner_spend_r2", "inner_invoice_r2"]
        pareto = compute_pareto_front(all_results, metrics_keys=pareto_metrics, complexity_key="n_final_concepts")
        selected = select_best_config_from_pareto(pareto, metrics_keys=pareto_metrics)

        for r in all_results:
            if r["config_id"] == selected["config_id"]:
                r["selected_for_outer_test"] = True

        sel_key = f"{selected['nR']}/{selected['nF']}/{selected['nM']}"
        selection_counts[sel_key] = selection_counts.get(sel_key, 0) + 1

        # 5. Outer test for selected config
        suppressed_outer, train_fuzzy_mu_outer, train_centroids_outer, _, n_final_outer = \
            build_representation_on_train(train_outer, selected)

        test_scored = dense_rank_scores(test_outer[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
        test_fuzzy_mu = apply_generalized_fuzzy_memberships(
            test_scored, train_centroids_outer, selected["n_levels_per_dim"], dims=DIMS
        )
        selected_outer = evaluate_on_outer_test(
            train_outer, test_outer, suppressed_outer, train_fuzzy_mu_outer, test_fuzzy_mu, outer_seed
        )

        # 6. Outer test for baseline (always evaluated for comparison)
        suppressed_bl, train_fuzzy_mu_bl, train_centroids_bl, _, n_final_bl = \
            build_representation_on_train(train_outer, baseline_cfg)

        test_scored_bl = dense_rank_scores(test_outer[["CustomerID", "R", "F", "M"]].copy(), dims=DIMS)
        test_fuzzy_mu_bl = apply_generalized_fuzzy_memberships(
            test_scored_bl, train_centroids_bl, baseline_cfg["n_levels_per_dim"], dims=DIMS
        )
        baseline_outer = evaluate_on_outer_test(
            train_outer, test_outer, suppressed_bl, train_fuzzy_mu_bl, test_fuzzy_mu_bl, outer_seed
        )

        # 7. Record
        sel_result = next(r for r in all_results if r["config_id"] == selected["config_id"])
        bl_result = next(r for r in all_results if r["is_baseline"])

        bl_result["test_roc_auc"] = baseline_outer["test_roc_auc"]
        bl_result["test_pr_auc"] = baseline_outer["test_pr_auc"]
        bl_result["test_spend_r2"] = baseline_outer["test_spend_r2"]
        bl_result["test_invoice_r2"] = baseline_outer["test_invoice_r2"]

        sel_result["test_roc_auc"] = selected_outer["test_roc_auc"]
        sel_result["test_pr_auc"] = selected_outer["test_pr_auc"]
        sel_result["test_spend_r2"] = selected_outer["test_spend_r2"]
        sel_result["test_invoice_r2"] = selected_outer["test_invoice_r2"]

        seed_results.append({
            "outer_seed": outer_seed,
            "selected_config_id": int(selected["config_id"]),
            "selected_config": f"{selected['nR']}/{selected['nF']}/{selected['nM']}",
            "baseline_config": "5/5/5",
            "inner_train_size": int(len(inner_train)),
            "inner_val_size": int(len(inner_val)),
            "outer_train_size": int(len(train_outer)),
            "outer_test_size": int(len(test_outer)),
            "selected_inner_metrics": {
                "roc_auc": float(selected["inner_roc_auc"]),
                "pr_auc": float(selected["inner_pr_auc"]),
                "spend_r2": float(selected["inner_spend_r2"]),
                "invoice_r2": float(selected["inner_invoice_r2"]),
            },
            "selected_outer_metrics": {
                "roc_auc": float(selected_outer["test_roc_auc"]),
                "pr_auc": float(selected_outer["test_pr_auc"]),
                "spend_r2": float(selected_outer["test_spend_r2"]),
                "invoice_r2": float(selected_outer["test_invoice_r2"]),
            },
            "baseline_outer_metrics": {
                "roc_auc": float(baseline_outer["test_roc_auc"]),
                "pr_auc": float(baseline_outer["test_pr_auc"]),
                "spend_r2": float(baseline_outer["test_spend_r2"]),
                "invoice_r2": float(baseline_outer["test_invoice_r2"]),
            },
            "selected_concepts_final": int(n_final_outer),
            "baseline_concepts_final": int(n_final_bl),
            "selected_candidates": int(selected["n_candidates"]),
            "baseline_candidates": int(bl_result["n_candidates"]),
        })

        # 8. Per-seed report
        _pr(f"\n  Selected configuration (inner-val Pareto): {selected['config_id']} -> {sel_key}")
        _pr(f"  Selected inner validation metrics:")
        _pr(f"    ROC AUC        = {selected['inner_roc_auc']:.4f}")
        _pr(f"    PR AUC         = {selected['inner_pr_auc']:.4f}")
        _pr(f"    Spend R²       = {selected['inner_spend_r2']:.4f}")
        _pr(f"    Invoice R²     = {selected['inner_invoice_r2']:.4f}")

        _pr(f"\n  Selected outer-test metrics:")
        _pr(f"    ROC AUC        = {selected_outer['test_roc_auc']:.4f}")
        _pr(f"    PR AUC         = {selected_outer['test_pr_auc']:.4f}")
        _pr(f"    Spend R²       = {selected_outer['test_spend_r2']:.4f}")
        _pr(f"    Invoice R²     = {selected_outer['test_invoice_r2']:.4f}")
        _pr(f"    Concepts (final) = {n_final_outer} (candidates {selected['n_candidates']})")

        _pr(f"\n  Baseline 5/5/5 outer-test metrics:")
        _pr(f"    ROC AUC        = {baseline_outer['test_roc_auc']:.4f}")
        _pr(f"    PR AUC         = {baseline_outer['test_pr_auc']:.4f}")
        _pr(f"    Spend R²       = {baseline_outer['test_spend_r2']:.4f}")
        _pr(f"    Invoice R²     = {baseline_outer['test_invoice_r2']:.4f}")
        _pr(f"    Concepts (final) = {n_final_bl} (candidates {bl_result['n_candidates']})")

        deltas = {}
        for k in METRIC_KEYS_OUTER:
            d = selected_outer[k] - baseline_outer[k]
            deltas[k] = d
            _pr(f"\n  Δ {METRIC_LABELS_OUTER[k]:<12} (selected - baseline) = {d:+.4f}")

        _pr(f"\n  Runtime this seed: {time.time()-t_seed:.1f}s")

    # -------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------
    _pr("\n" + "=" * 96)
    _pr("AGGREGATION ACROSS OUTER SEEDS 1000-1010")
    _pr("=" * 96)

    agg = {}
    metric_name_map = {
        "test_roc_auc": "roc_auc",
        "test_pr_auc": "pr_auc",
        "test_spend_r2": "spend_r2",
        "test_invoice_r2": "invoice_r2",
    }
    for metric in METRIC_KEYS_OUTER:
        inner = metric_name_map[metric]
        sel_vals = np.array([r["selected_outer_metrics"][inner] for r in seed_results])
        bl_vals = np.array([r["baseline_outer_metrics"][inner] for r in seed_results])
        diff = sel_vals - bl_vals

        mean_sel, std_sel, med_sel = float(sel_vals.mean()), float(sel_vals.std(ddof=1)), float(np.median(sel_vals))
        mean_bl, std_bl, med_bl = float(bl_vals.mean()), float(bl_vals.std(ddof=1)), float(np.median(bl_vals))
        mean_diff, std_diff = float(diff.mean()), float(diff.std(ddof=1))
        n = len(diff)
        se = std_diff / np.sqrt(n)
        ci_lo = mean_diff - stats.t.ppf(0.975, df=n-1) * se
        ci_hi = mean_diff + stats.t.ppf(0.975, df=n-1) * se
        t_stat, p_val = stats.ttest_rel(sel_vals, bl_vals)

        agg[metric] = {
            "selected_mean": mean_sel,
            "selected_std": std_sel,
            "selected_median": med_sel,
            "baseline_mean": mean_bl,
            "baseline_std": std_bl,
            "baseline_median": med_bl,
            "paired_diff_mean": mean_diff,
            "paired_diff_std": std_diff,
            "paired_diff_median": float(np.median(diff)),
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_val),
            "n_seeds": n,
        }

    # Selection frequency
    total_seeds = len(seed_results)
    sel_freq = {k: {"count": v, "freq": f"{v}/{total_seeds}"} for k, v in selection_counts.items()}

    # -------------------------------------------------------------------
    # Print aggregate
    # -------------------------------------------------------------------
    _pr("\nPer-metric aggregation (selected vs baseline 5/5/5):")
    for metric in METRIC_KEYS_OUTER:
        a = agg[metric]
        _pr(f"\n  {METRIC_LABELS_OUTER[metric]}:")
        _pr(f"    Selected : mean={a['selected_mean']:.4f}  std={a['selected_std']:.4f}  median={a['selected_median']:.4f}")
        _pr(f"    Baseline : mean={a['baseline_mean']:.4f}  std={a['baseline_std']:.4f}  median={a['baseline_median']:.4f}")
        _pr(f"    Paired Δ : mean={a['paired_diff_mean']:+.4f}  std={a['paired_diff_std']:.4f}  median={a['paired_diff_median']:+.4f}")
        _pr(f"    95% CI   : [{a['ci95_lower']:+.4f}, {a['ci95_upper']:+.4f}]")
        _pr(f"    Paired t : t={a['paired_t_stat']:+.4f}, p={a['paired_p_value']:.4f}, n={a['n_seeds']}")

    _pr("\nConfiguration selection frequency across outer seeds:")
    for k in sorted(sel_freq, key=lambda x: -sel_freq[x]["count"]):
        _pr(f"  {k:>10s} -> {sel_freq[k]['count']}/{total_seeds}")

    # -------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------
    _pr("\n" + "=" * 96)
    _pr("SAVING ARTIFACTS")
    _pr("=" * 96)

    seed_df = pd.DataFrame(seed_results)
    seed_path = EXP_DIR / "stage1_multiseed_seed_results.csv"
    seed_df.to_csv(seed_path, index=False)
    _pr(f"Saved: {seed_path}")

    summary = {
        "experiment": "stage1_multiseed_validation",
        "protocol": "nested_leakage_free",
        "stage": 1,
        "dataset": "Dunnhumby Complete Journey",
        "outer_seeds": OUTER_SEEDS,
        "n_seeds": len(OUTER_SEEDS),
        "outer_split": "70/30 stratified",
        "inner_split": "70/30 within outer train",
        "configs_evaluated": len(configs),
        "config_space": [
            {"config_id": c["config_id"], "nR": c["nR"], "nF": c["nF"], "nM": c["nM"], "is_baseline": c["is_baseline"]}
            for c in configs
        ],
        "fixed_fca_params": {
            "min_support": MIN_SUPPORT,
            "L_thresholds": list(L_THRESHOLDS),
            "J_max": J_MAX,
            "mu_cut": MU_CUT,
        },
        "selection_frequency": sel_freq,
        "aggregation": agg,
        "leakage_free": True,
        "runtime_total_s": float(time.time() - t_start),
    }

    summary_path = EXP_DIR / "stage1_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _pr(f"Saved: {summary_path}")

    # Report markdown
    lines = [
        "# Stage-1 Multi-Seed Validation Report",
        "",
        "## Protocol",
        "",
        f"- Dataset: Dunnhumby Complete Journey.",
        f"- Outer seeds: {OUTER_SEEDS[0]}-{OUTER_SEEDS[-1]} (n={len(OUTER_SEEDS)}).",
        f"- Outer split: 70/30 stratified.",
        f"- Inner split: 70/30 within outer train; configuration selection uses inner validation ONLY.",
        f"- Configs evaluated: {len(configs)} (fixed 5/5/5 baseline included).",
        f"- FCA selection fixed for Stage 1: min_support={MIN_SUPPORT}, L={L_THRESHOLDS}, J_max={J_MAX}, mu_cut={MU_CUT}.",
        f"- Selection: inner-validation Pareto + combined normalized score.",
        f"- Outer test evaluated once per outer seed after config frozen.",
        "",
        "## Configuration Selection Frequency",
        "",
        "| Configuration | Selections |",
        "|---------------|------------|",
    ] + [
        f"| {k} | {sel_freq[k]['count']}/{total_seeds} |"
        for k in sorted(sel_freq, key=lambda x: -sel_freq[x]["count"])
    ] + [
        "",
        "## Per-Metric Aggregation (Selected vs Baseline 5/5/5)",
        "",
        "| Metric | Selected Mean | Selected SD | Baseline Mean | Baseline SD | Paired Δ Mean | Paired Δ SD | 95% CI | Paired t | p-value |",
        "|--------|---------------|-------------|---------------|-------------|---------------|-------------|--------|----------|--------|",
    ] + [
        f"| {METRIC_LABELS_OUTER[m]} | {a['selected_mean']:.4f} | {a['selected_std']:.4f} | {a['baseline_mean']:.4f} | {a['baseline_std']:.4f} | {a['paired_diff_mean']:+.4f} | {a['paired_diff_std']:.4f} | [{a['ci95_lower']:+.4f}, {a['ci95_upper']:+.4f}] | {a['paired_t_stat']:+.4f} | {a['paired_p_value']:.4f} |"
        for m in METRIC_KEYS_OUTER
        for a in [agg[m]]
    ] + [
        "",
        "## Per-Seed Results",
        "",
        "| Outer Seed | Selected | Baseline | Sel. Inner AUC | Sel. Inner Spend R2 | Sel. Inner Invoice R2 | Sel. Test AUC | Sel. Test PR AUC | Sel. Test Spend R2 | Sel. Test Invoice R2 | Baseline Test AUC | Baseline Test PR AUC | Baseline Test Spend R2 | Baseline Test Invoice R2 | Δ AUC | Δ PR AUC | Δ Spend R2 | Δ Invoice R2 | Sel. Concepts | Baseline Concepts |",
        "|------------|----------|----------|----------------|---------------------|------------------------|---------------|------------------|--------------------|----------------------|-------------------|----------------------|-----------------------|-------------------------|-------|----------|------------|---------------|---------------|-------------------|",
    ] + [
        f"| {r['outer_seed']} | {r['selected_config']} | {r['baseline_config']} | {r['selected_inner_metrics']['roc_auc']:.4f} | {r['selected_inner_metrics']['spend_r2']:.4f} | {r['selected_inner_metrics']['invoice_r2']:.4f} | {r['selected_outer_metrics']['roc_auc']:.4f} | {r['selected_outer_metrics']['pr_auc']:.4f} | {r['selected_outer_metrics']['spend_r2']:.4f} | {r['selected_outer_metrics']['invoice_r2']:.4f} | {r['baseline_outer_metrics']['roc_auc']:.4f} | {r['baseline_outer_metrics']['pr_auc']:.4f} | {r['baseline_outer_metrics']['spend_r2']:.4f} | {r['baseline_outer_metrics']['invoice_r2']:.4f} | {r['selected_outer_metrics']['roc_auc']-r['baseline_outer_metrics']['roc_auc']:+.4f} | {r['selected_outer_metrics']['pr_auc']-r['baseline_outer_metrics']['pr_auc']:+.4f} | {r['selected_outer_metrics']['spend_r2']-r['baseline_outer_metrics']['spend_r2']:+.4f} | {r['selected_outer_metrics']['invoice_r2']-r['baseline_outer_metrics']['invoice_r2']:+.4f} | {r['selected_concepts_final']} | {r['baseline_concepts_final']} |"
        for r in seed_results
    ] + [
        "",
        "## Leakage Assertions",
        "",
        "- Inner validation metrics computed on inner_val ONLY.",
        "- Outer test metrics computed AFTER config selection per outer seed.",
        "- Config selection uses inner validation Pareto ONLY.",
        "- No outer test metrics passed to optimizer.",
        "- Baseline 5/5/5 included and evaluated independently each outer seed.",
        "",
        f"Total runtime: {time.time()-t_start:.1f}s",
    ]

    report_path = EXP_DIR / "stage1_multiseed_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _pr(f"Saved: {report_path}")

    _pr("\n" + "=" * 96)
    _pr("STAGE 1 MULTI-SEED VALIDATION COMPLETE")
    _pr(f"Total runtime: {time.time()-t_start:.1f}s")
    _pr("=" * 96)


if __name__ == "__main__":
    main()

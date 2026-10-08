#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 FAST smoke test: single outer seed (42), nested leakage-free,
all baselines A/B/C/D, reduced grid spanning the full validated search space.

The reduced grid keeps the key degrees of freedom:
- min_support: 0.02, 0.04, 0.05
- j_max:       0.70, 0.80, 0.90
- mu_cut:      0.45, 0.50, 0.60
- kuz_loss:    1e-300, 1e-5, 0.062, 1.0   (spanning the observed distribution)

That is 3*3*3*4 = 108 configs, far fewer than the 2080 full grid.
This verifies the pipeline runs end-to-end without leakage before the 10-seed run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy_membership_sensitivity import load_and_prepare_cohorts, DIMS, INVERT_DIMS, mine_fuzzy_closed_concepts_with_thresholds, _band_centroids_generic, _piecewise_membership_generic
from fair_comparison_retail2 import (
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    apply_dense_rank_cutoffs,
    compute_fuzzy_memberships,
    compute_customer_concept_memberships,
    mine_crisp_closed_concepts,
)
from concept_redundancy import suppress_redundant_concepts
from kuznetsov_pruning_leakage_free import (
    compute_split_stability,
    evaluate_baseline_split,
    J_MAX,
    MU_CUT,
    MIN_SUPPORT,
    L_THRESHOLDS,
)

RESULTS_DIR = ROOT_DIR.parent / "results"  # ROOT_DIR is scripts/ here (sys.path)
EXP_DIR = RESULTS_DIR / "optimized_fuzzy_fca" / "stage2"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def _pr(msg: str) -> None:
    print(msg, flush=True)


# -------------------------------------------------------------------
# Small grid spanning the full validated search space
# -------------------------------------------------------------------

def load_smoke_grid() -> list[dict]:
    min_supports = [0.02, 0.04, 0.05]
    j_maxs = [0.70, 0.80, 0.90]
    mu_cuts = [0.45, 0.50, 0.60]
    kuz_losses = [1e-300, 1e-5, 0.062, 1.0]
    grid = []
    idx = 1
    for ms in min_supports:
        for jm in j_maxs:
            for mc in mu_cuts:
                for kl in kuz_losses:
                    grid.append({
                        "config_id": idx,
                        "min_support": float(ms),
                        "j_max": float(jm),
                        "mu_cut": float(mc),
                        "kuz_loss": float(kl),
                    })
                    idx += 1
    return grid


# -------------------------------------------------------------------
# Fuzzy pipeline on one train split
# -------------------------------------------------------------------

def generalized_fuzzy_memberships(
    scored: 'pd.DataFrame',
    n_levels_per_dim: dict,
    dims=DIMS,
    trained_centroids=None,
) -> tuple:
    memberships = {}
    centroid_dict = {}
    centroid_rows = []
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
            memberships[f"{dim}{k+1}"] = mu[:, k]
            centroid_rows.append({
                "dimension": dim, "band": k+1,
                "centroid": float(centroids[k]),
                "customers": int(np.sum(score == k+1)) if score is not None else 0,
            })
    import pandas as _pd
    return _pd.DataFrame(memberships), centroid_dict, _pd.DataFrame(centroid_rows)


def apply_generalized_fuzzy_memberships(
    scored, trained_centroids, n_levels_per_dim, dims=DIMS,
):
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
    import pandas as _pd
    return _pd.DataFrame(memberships)


def build_rep_on_train(train_df, config):
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

    stab_df, ctx_info = compute_split_stability(raw_concepts, train_fuzzy_mu, L_THRESHOLDS)
    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[
            ["concept_index", "stab_float", "log2_loss", "status",
             "extent_size", "n_lower_neighbors", "generator_count",
             "stability_fraction", "a_prime_equals_b", "b_prime_equals_a"]
        ],
        on="concept_index", how="left",
    )
    n_with_stability = int(concepts["stab_float"].notna().sum())
    n_after_stability = len(concepts[concepts["stab_float"].notna() & (concepts["stab_float"] > 0)])
    n_dropped_by_stability = n_candidates - n_after_stability

    if config["kuz_loss"] < 1.0 - 1e-12:
        after_stab = concepts[
            concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
            & ((1.0 - concepts["stab_float"]) <= config["kuz_loss"])
        ].copy()
    else:
        after_stab = concepts[
            concepts["stab_float"].notna() & (concepts["stab_float"] > 0)
        ].copy()

    n_after_stability = len(after_stab)
    n_dropped_by_stability = n_candidates - n_after_stability
    mu_tr = compute_customer_concept_memberships(after_stab.reset_index(drop=True), train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(after_stab.reset_index(drop=True), mu_tr,
                                              j_max=config["j_max"], mu_cut=config["mu_cut"])
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
        "n_after_stability": n_after_stability,
        "n_dropped_by_stability": n_dropped_by_stability,
        "n_final_concepts": n_final,
        "n_dropped_by_jaccard": n_dropped_by_jaccard,
        "ctx_info": ctx_info,
        "config": config,
    }


def project_test_fuzzy(test_df, train_centroids, n_levels_per_dim, dims=DIMS):
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_scored = dense_rank_scores(test_cust, dims=DIMS)
    return apply_generalized_fuzzy_memberships(test_scored, train_centroids, n_levels_per_dim, dims)


# -------------------------------------------------------------------
# Predictive evaluation
# -------------------------------------------------------------------

from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

LOGREG_Cs = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)


def evaluate_on_split(train_df, test_df, X_tr, X_te, seed):
    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(Cs=LOGREG_Cs, cv=5, scoring="roc_auc",
                                solver="lbfgs", max_iter=LOGREG_MAX_ITER, random_state=seed).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te_rep, prob_te))
    pr_auc = float(average_precision_score(y_te_rep, prob_te))
    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_sp)
    sp_r2 = float(r2_score(y_te_sp, reg_sp.predict(X_te)))
    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr_inv)
    inv_r2 = float(r2_score(y_te_inv, reg_inv.predict(X_te)))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "spend_r2": sp_r2, "invoice_r2": inv_r2}


# -------------------------------------------------------------------
# Baselines A, B, C
# -------------------------------------------------------------------

def baseline_raw_rfm_split(train_df, test_df, seed):
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_cutoffs = {dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS)) for dim in DIMS}
    crisp_concepts = mine_crisp_closed_concepts(train_scored, min_support=MIN_SUPPORT, dims=DIMS)
    crisp_concepts = crisp_concepts.reset_index(drop=True)
    n_crisp = len(crisp_concepts)
    crisp_bands_tr = _pd.DataFrame({f"{dim}{k}": (train_scored[f"{dim}_score"] == k).astype(float)
                                     for dim in DIMS for k in range(1, 6)})
    X_tr = compute_customer_concept_memberships(crisp_concepts, crisp_bands_tr)
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_scored = _pd.DataFrame({"CustomerID": test_cust["CustomerID"]})
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(test_df[dim].to_numpy(), train_cutoffs[dim], invert=(dim in INVERT_DIMS))
        for k in range(1, 6):
            test_scored[f"{dim}{k}"] = (s == k).astype(float)
    X_te = compute_customer_concept_memberships(crisp_concepts, test_scored)
    if X_tr.shape[1] == 0:
        X_tr = np.zeros((len(train_df), 1))
        X_te = np.zeros((len(test_df), 1))
    res = evaluate_on_split(train_df, test_df, X_tr, X_te, seed)
    res.update({"method": "Raw RFM", "n_candidates": n_crisp, "n_final_concepts": n_crisp, "n_concepts": n_crisp})
    return res


def baseline_existing_fuzzy_split(train_df, test_df, seed):
    rec = evaluate_baseline_split(train_df, test_df, seed, split_index=0)
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
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_scored = _pd.DataFrame({"CustomerID": test_cust["CustomerID"]})
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(test_df[dim].to_numpy(),
                                      extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS)),
                                      invert=(dim in INVERT_DIMS))
        for k in range(1, 6):
            test_scored[f"{dim}{k}"] = (s == k).astype(float)
    X_te = compute_customer_concept_memberships(suppressed, test_scored)
    y_te_rep = test_df["repurchased"].to_numpy()
    clf = LogisticRegressionCV(Cs=LOGREG_Cs, cv=5, scoring="roc_auc", solver="lbfgs",
                                max_iter=LOGREG_MAX_ITER, random_state=seed).fit(X_tr, train_df["repurchased"].to_numpy())
    prob_te = clf.predict_proba(X_te)[:, 1]
    pr_auc = float(average_precision_score(y_te_rep, prob_te))
    return {
        "method": rec["method"],
        "n_candidates": int(rec["n_candidates"]),
        "n_final_concepts": int(rec["n_final_concepts"]),
        "roc_auc": float(rec["auc"]),
        "pr_auc": pr_auc,
        "spend_r2": float(rec["spend_r2"]),
        "invoice_r2": float(rec["invoice_r2"]),
        "n_concepts": int(rec["n_final_concepts"]),
    }


def baseline_existing_kuznetsov_split(train_df, test_df, seed):
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
        stab_df[["concept_index", "stab_float", "log2_loss", "status",
                 "extent_size", "n_lower_neighbors", "generator_count",
                 "stability_fraction", "a_prime_equals_b", "b_prime_equals_a"]],
        on="concept_index", how="left",
    )
    n_with_stability = int(concepts["stab_float"].notna().sum())
    after_stab = concepts[concepts["stab_float"].notna() & (concepts["stab_float"] > 0)].copy()
    n_after_stability = len(after_stab)
    n_dropped_by_stability = n_candidates - n_after_stability
    mu_tr = compute_customer_concept_memberships(after_stab.reset_index(drop=True), train_fuzzy_mu)
    suppressed = suppress_redundant_concepts(after_stab.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT)
    n_final = len(suppressed)
    n_dropped_by_jaccard = n_after_stability - n_final
    suppressed = suppressed.reset_index(drop=True)
    X_tr = compute_customer_concept_memberships(suppressed, train_fuzzy_mu)
    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu = apply_generalized_fuzzy_memberships(dense_rank_scores(test_cust, dims=DIMS),
                                                        train_centroids, {"R": 5, "F": 5, "M": 5}, DIMS)
    X_te = compute_customer_concept_memberships(suppressed, test_fuzzy_mu)
    res = evaluate_on_split(train_df, test_df, X_tr, X_te, seed)
    res.update({
        "method": "Existing Kuznetsov-FCA (loss<=1.0)",
        "n_candidates": n_candidates,
        "n_with_stability": n_with_stability,
        "n_after_stability": n_after_stability,
        "n_dropped_by_stability": n_dropped_by_stability,
        "n_final_concepts": n_final,
        "n_dropped_by_jaccard": n_dropped_by_jaccard,
        "n_concepts": n_final,
        "ctx_info": ctx_info,
    })
    return res


# -------------------------------------------------------------------
# Pareto selection
# -------------------------------------------------------------------

METRIC_KEYS_INNER = ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]


def compute_pareto_front(configs, metrics_keys=None, complexity_key="n_final_concepts"):
    metrics_keys = metrics_keys or METRIC_KEYS_INNER
    if not configs:
        return []
    pareto_indices = []
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


def select_config_from_pareto(pareto_configs, metrics_keys=None, complexity_key="n_final_concepts"):
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
# Main
# -------------------------------------------------------------------

import pandas as _pd


def main() -> None:
    t_start = time.time()
    _pr("=" * 100)
    _pr("STAGE 2 FAST SMOKE TEST (OUTER SEED 42, REDUCED GRID)")
    _pr("=" * 100)

    configs = load_smoke_grid()
    _pr(f"Reduced smoke grid: {len(configs)} configs (3*3*3*4)")
    _pr(f"Outer seed: 42, inner seed: 42, fuzzy fixed 5/5/5")
    _pr(f"Baselines: A) Raw RFM  B) Existing Fuzzy FCA  C) Existing Kuznetsov FCA  D) Optimized FCA")

    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"Cohort: {len(df):,}  repurchase rate={df['repurchased'].mean():.4f}")

    _pr("\n[1] Outer split seed=42 (70/30 stratified)...")
    train_outer, test_outer = train_test_split(
        df, test_size=0.30, stratify=df["repurchased"].to_numpy(), random_state=42
    )
    train_outer = train_outer.reset_index(drop=True)
    test_outer = test_outer.reset_index(drop=True)
    _pr(f"  outer train: {len(train_outer):,}  outer test: {len(test_outer):,}")

    _pr("\n[2] Baseline A: Raw RFM")
    t0 = time.time()
    raw42 = baseline_raw_rfm_split(train_outer, test_outer, 42)
    _pr(f"  ROC AUC={raw42['roc_auc']:.4f} PR AUC={raw42['pr_auc']:.4f} Spend R2={raw42['spend_r2']:.4f} "
        f"Invoice R2={raw42['invoice_r2']:.4f} concepts={raw42['n_concepts']} runtime={time.time()-t0:.1f}s")

    _pr("\n[3] Baseline B: Existing Fuzzy FCA")
    t0 = time.time()
    fuzz42 = baseline_existing_fuzzy_split(train_outer, test_outer, 42)
    _pr(f"  ROC AUC={fuzz42['roc_auc']:.4f} PR AUC={fuzz42['pr_auc']:.4f} Spend R2={fuzz42['spend_r2']:.4f} "
        f"Invoice R2={fuzz42['invoice_r2']:.4f} candidates={fuzz42['n_candidates']} final={fuzz42['n_final_concepts']} "
        f"runtime={time.time()-t0:.1f}s")

    _pr("\n[4] Baseline C: Existing Kuznetsov FCA (loss<=1.0)")
    t0 = time.time()
    kuz42 = baseline_existing_kuznetsov_split(train_outer, test_outer, 42)
    _pr(f"  ROC AUC={kuz42['roc_auc']:.4f} PR AUC={kuz42['pr_auc']:.4f} Spend R2={kuz42['spend_r2']:.4f} "
        f"Invoice R2={kuz42['invoice_r2']:.4f} candidates={kuz42['n_candidates']} after_stab={kuz42['n_after_stability']} "
        f"final={kuz42['n_final_concepts']} stab_time={kuz42['ctx_info']['stability_time_s']:.2f}s "
        f"runtime={time.time()-t0:.1f}s")

    _pr(f"\n[5] Inner split for config selection (70/30 within outer train, seed=42)...")
    inner_train, inner_val = train_test_split(
        train_outer, test_size=0.30, stratify=train_outer["repurchased"].to_numpy(), random_state=42
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val = inner_val.reset_index(drop=True)
    _pr(f"  inner train: {len(inner_train):,}  inner val: {len(inner_val):,}")

    _pr(f"\n[6] Evaluate {len(configs)} configs on INNER VALIDATION ONLY...")
    inner_results = []
    t0 = time.time()
    for config in configs:
        rep = build_rep_on_train(inner_train, config)
        val_fuzzy_mu = project_test_fuzzy(inner_val, rep["train_centroids"], {"R": 5, "F": 5, "M": 5}, DIMS)
        X_val = compute_customer_concept_memberships(rep["suppressed"], val_fuzzy_mu)
        inner_metrics = evaluate_on_split(inner_train, inner_val, rep["X"], X_val, 42)
        inner_results.append({
            "config_id": config["config_id"],
            "min_support": config["min_support"],
            "j_max": config["j_max"],
            "mu_cut": config["mu_cut"],
            "kuz_loss": config["kuz_loss"],
            "n_candidates": rep["n_candidates"],
            "n_after_stability": rep["n_after_stability"],
            "n_dropped_by_stability": rep["n_dropped_by_stability"],
            "n_final_concepts": rep["n_final_concepts"],
            "n_dropped_by_jaccard": rep["n_dropped_by_jaccard"],
            "ctx_stability_time_s": rep["ctx_info"]["stability_time_s"],
            **inner_metrics,
            "selected": False,
            "outer_roc_auc": None, "outer_pr_auc": None, "outer_spend_r2": None, "outer_invoice_r2": None,
        })
    _pr(f"  evaluated in {time.time()-t0:.1f}s")

    _pr("\n[7] Inner-validation Pareto selection (frozen before outer test)...")
    pareto = compute_pareto_front(inner_results, metrics_keys=METRIC_KEYS_INNER, complexity_key="n_final_concepts")
    selected_inner = select_config_from_pareto(pareto, metrics_keys=METRIC_KEYS_INNER, complexity_key="n_final_concepts")
    for r in inner_results:
        if r["config_id"] == selected_inner["config_id"]:
            r["selected"] = True
    _pr(f"  Pareto front size: {len(pareto)}")
    _pr(f"  Selected: id={int(selected_inner['config_id'])} ms={selected_inner['min_support']:.3e} "
        f"jm={selected_inner['j_max']:.2f} mc={selected_inner['mu_cut']:.2f} kl={selected_inner['kuz_loss']:.3e}")
    _pr(f"  Inner metrics: " + "  ".join(f"{k}={selected_inner[k]:.4f}" for k in METRIC_KEYS_INNER))
    _pr(f"  Inner concepts: candidates={int(selected_inner['n_candidates'])} after_stab={int(selected_inner['n_after_stability'])} "
        f"final={int(selected_inner['n_final_concepts'])} (drop stab={int(selected_inner['n_dropped_by_stability'])}, "
        f"drop jac={int(selected_inner['n_dropped_by_jaccard'])})")

    _pr("\n[8] Refit selected config on FULL OUTER TRAIN, evaluate OUTER TEST ONCE...")
    sel_cfg = next(c for c in configs if c["config_id"] == selected_inner["config_id"])
    sel_rep = build_rep_on_train(train_outer, sel_cfg)
    test_fuzzy_mu = project_test_fuzzy(test_outer, sel_rep["train_centroids"], {"R": 5, "F": 5, "M": 5}, DIMS)
    X_te = compute_customer_concept_memberships(sel_rep["suppressed"], test_fuzzy_mu)
    outer_metrics = evaluate_on_split(train_outer, test_outer, sel_rep["X"], X_te, 42)
    selected_outer = {
        **outer_metrics,
        "n_candidates": sel_rep["n_candidates"],
        "n_with_stability": sel_rep["n_with_stability"],
        "n_after_stability": sel_rep["n_after_stability"],
        "n_dropped_by_stability": sel_rep["n_dropped_by_stability"],
        "n_final_concepts": sel_rep["n_final_concepts"],
        "n_dropped_by_jaccard": sel_rep["n_dropped_by_jaccard"],
        "ctx_stability_time_s": sel_rep["ctx_info"]["stability_time_s"],
    }
    _pr(f"  Outer test: " + "  ".join(f"{k}={outer_metrics[k]:.4f}" for k in METRIC_KEYS_INNER))
    _pr(f"  Concepts: candidates={int(sel_rep['n_candidates'])} after_stab={int(sel_rep['n_after_stability'])} "
        f"final={int(sel_rep['n_final_concepts'])}")
    _pr(f"  stab_time={sel_rep['ctx_info']['stability_time_s']:.2f}s")

    # deltas
    delta_raw = {k: outer_metrics[k] - raw42[k] for k in METRIC_KEYS_INNER}
    delta_fuzz = {k: outer_metrics[k] - fuzz42[k] for k in METRIC_KEYS_INNER}
    delta_kuz = {k: outer_metrics[k] - kuz42[k] for k in METRIC_KEYS_INNER}

    _pr("\n[9] OUTER TEST DELTAS (optimized - baseline):")
    _pr(f"  vs Raw RFM   : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_raw.items()))
    _pr(f"  vs Fuzzy FCA : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_fuzz.items()))
    _pr(f"  vs Kuz FCA   : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_kuz.items()))

    _pr("\n[10] LEAKAGE ASSERTIONS:")
    _pr("  - inner_val used for config selection ONLY")
    _pr("  - outer test evaluated ONCE after config frozen")
    _pr("  - baselines A/B/C computed independently")
    _pr("  - exact canonical Kuznetsov stability from training context only")
    _pr("  - no outer test metrics passed to optimizer")

    # save artifact
    artifact = {
        "outer_seed": 42,
        "grid_size": len(configs),
        "baseline_raw_rfm": raw42,
        "baseline_existing_fuzzy": fuzz42,
        "baseline_existing_kuznetsov": {
            "roc_auc": kuz42["roc_auc"], "pr_auc": kuz42["pr_auc"],
            "spend_r2": kuz42["spend_r2"], "invoice_r2": kuz42["invoice_r2"],
            "n_candidates": kuz42["n_candidates"], "n_after_stability": kuz42["n_after_stability"],
            "n_final_concepts": kuz42["n_final_concepts"],
            "ctx_stability_time_s": kuz42["ctx_info"]["stability_time_s"],
        },
        "selected_config": {
            "config_id": int(selected_inner["config_id"]),
            "min_support": selected_inner["min_support"],
            "j_max": selected_inner["j_max"],
            "mu_cut": selected_inner["mu_cut"],
            "kuz_loss": selected_inner["kuz_loss"],
            "inner_roc_auc": selected_inner["roc_auc"],
            "inner_pr_auc": selected_inner["pr_auc"],
            "inner_spend_r2": selected_inner["spend_r2"],
            "inner_invoice_r2": selected_inner["invoice_r2"],
            "n_candidates": int(selected_inner["n_candidates"]),
            "n_after_stability": int(selected_inner["n_after_stability"]),
            "n_final_concepts": int(selected_inner["n_final_concepts"]),
            "n_dropped_by_stability": int(selected_inner["n_dropped_by_stability"]),
            "n_dropped_by_jaccard": int(selected_inner["n_dropped_by_jaccard"]),
        },
        "selected_outer": selected_outer,
        "delta_vs_raw": delta_raw,
        "delta_vs_fuzzy": delta_fuzz,
        "delta_vs_kuz": delta_kuz,
        "runtime_s": time.time() - t_start,
    }
    path = EXP_DIR / "stage2_fast_smoke_test_seed42.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _pr(f"\nSaved: {path}")
    _pr(f"\n{'='*100}")
    _pr(f"FAST SMOKE TEST COMPLETE (runtime {time.time()-t_start:.1f}s)")
    _pr(f"{'='*100}")

    # Print per-config inner results summary (top 5 by composite)
    _pr("\n[11] Top 5 configs by inner-validation composite (for sanity):")
    sorted_configs = sorted(inner_results, key=lambda r: -(r["roc_auc"] + r["pr_auc"] + r["spend_r2"] + r["invoice_r2"]))
    for r in sorted_configs[:5]:
        _pr(f"  id={int(r['config_id']):>3} ms={r['min_support']:.3e} jm={r['j_max']:.2f} mc={r['mu_cut']:.2f} "
            f"kl={r['kuz_loss']:.3e} | AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} "
            f"SpR2={r['spend_r2']:.4f} InvR2={r['invoice_r2']:.4f} | final={int(r['n_final_concepts'])}")


if __name__ == "__main__":
    main()

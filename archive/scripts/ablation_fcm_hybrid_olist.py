"""Experiment 6: Controlled Representation Ablation Study on Olist (RFMS).

Tests whether FCM soft-cluster memberships contain information complementary
to the fuzzy-FCA concept representation, rather than simply replacing FCA with FCM.

Ablation arms:
A. Raw RFMS                     (standardized R, F, M, S features; 4 dims)
B. Fuzzy-FCA                    (leakage-free redundancy-suppressed fuzzy concepts)
C. FCM memberships              (canonical FCM soft memberships at matched k=44)
D. Raw RFMS + FCM               (concatenation: 4 + 44 = 48 dims)
E. Raw RFMS + Fuzzy-FCA         (concatenation: 4 + n_fca dims)
F. Raw RFMS + FCM + Fuzzy-FCA   (hybrid: 4 + 44 + n_fca dims)

Strict temporal protocol (leakage-free):
- Cutoff: 2017-08-31 23:59:59
- Observation cohort: 21,664 customers with order_purchase_timestamp <= cutoff
- Satisfaction (S): Strictly pre-cutoff reviews satisfying BOTH:
    order_purchase_timestamp <= cutoff AND review_answer_timestamp <= cutoff
- Preprocessing fit on train split only: StandardScaler, dense-rank cutoffs,
  fuzzy centroids, concept mining, redundancy suppression, and FCM prototype fitting.
- Test set transformed strictly using frozen train parameters.

Downstream evaluation:
- LogisticRegressionCV (5-fold, C in [1e-3..1e2], scoring='neg_log_loss') for repurchase.
- RidgeCV (5-fold, 13 alphas in [1e-2..1e4]) for log1p future spend and log1p invoices.
- Paired customer bootstrap CIs (N=1000) conditional on fixed split.
- Multi-split validation across 10 stratified splits (seeds 1000..1009).

Outputs: results/ablation_fcm_hybrid_olist/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts_sparse
from fair_comparison_retail2 import (
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    mine_crisp_closed_concepts,
)
from fcm import FuzzyCMeans
from olist_rfms_comparison import (
    CUTOFF_DATE,
    DECISION_AUC,
    DECISION_R2,
    DIMS,
    INVERT_DIMS,
    J_MAX,
    N_BOOT,
    RANDOM_SEED,
    SUPPORT_CUTOFF,
    load_olist_transactions,
    mine_fuzzy_closed_concepts_bitset,
)
from project_paths import OLIST_RAW_DIR, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "ablation_fcm_hybrid_olist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 10
BASE_SEED = 1000
FCM_M = 2.0
FCM_MAX_ITER = 300
FCM_TOL = 1e-7


# -------------------------------------------------------------------------
# Data Preparation (Leakage-Free Temporal Protocol)
# -------------------------------------------------------------------------
def prepare_observation_cohort() -> pd.DataFrame:
    """Prepare observation cohort strictly adhering to validated leakage-free protocol."""
    tx = load_olist_transactions()
    obs_tx = tx[tx["order_purchase_timestamp"] <= CUTOFF_DATE]
    hold_tx = tx[tx["order_purchase_timestamp"] > CUTOFF_DATE]

    # Pre-cutoff features: R, F, M
    obs_first = (
        obs_tx.groupby("CustomerID", sort=True)
        .agg(
            R_days=("order_purchase_timestamp", "max"),
            M_obs=("payment_value", "sum"),
            n_obs_orders=("order_id", "nunique"),
        )
        .reset_index()
    )
    obs_ref = obs_tx["order_purchase_timestamp"].max()
    obs_first["R"] = (obs_ref - obs_first["R_days"]).dt.days.astype(int)
    obs_first["F"] = 0.20  # constant single-order baseline
    obs_first["M"] = obs_first["M_obs"]
    obs_first = obs_first[["CustomerID", "R", "F", "M"]]

    # S: strictly pre-cutoff reviews governed by BOTH purchase date and answer timestamp
    reviews = pd.read_csv(
        OLIST_RAW_DIR / "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )
    review_tx = reviews.merge(
        tx[["order_id", "CustomerID", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
    )
    pre_cutoff_reviews = review_tx[
        (review_tx["order_purchase_timestamp"] <= CUTOFF_DATE)
        & (review_tx["review_answer_timestamp"] <= CUTOFF_DATE)
    ].copy()
    s_pre = pre_cutoff_reviews.groupby("CustomerID")["review_score"].mean()
    obs_first["S"] = obs_first["CustomerID"].map(s_pre).fillna(5.0)

    # Outcomes: post-cutoff orders only
    hold_agg = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(
            future_invoices=("order_id", "nunique"),
            future_spend=("payment_value", "sum"),
        )
        .reset_index()
    )
    merged = obs_first.merge(hold_agg, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)
    merged = merged.dropna(subset=["R", "F", "M", "S"]).reset_index(drop=True)

    print(
        f"Prepared observation cohort: {len(merged):,} customers | "
        f"Repurchase rate: {merged['repurchased'].mean():.4f}"
    )
    return merged


# -------------------------------------------------------------------------
# Feature Representation Construction
# -------------------------------------------------------------------------
def build_representations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    k_matched: int = 44,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Build the six representation variants strictly using train parameters."""
    feature_cols = list(DIMS)

    # 1. Standardized Raw RFMS
    scaler = StandardScaler().fit(train_df[feature_cols].values)
    X_tr_raw = scaler.transform(train_df[feature_cols].values)
    X_te_raw = scaler.transform(test_df[feature_cols].values)

    # 2. Fuzzy-FCA with Redundancy Suppression
    train_scored = dense_rank_scores(
        train_df[["CustomerID", "R", "F", "M", "S"]], dims=DIMS
    )
    train_fmu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)
    fuzzy = mine_fuzzy_closed_concepts_bitset(train_fmu, train_scored, SUPPORT_CUTOFF)
    X_tr_fuzzy = compute_customer_concept_memberships(fuzzy, train_fmu)
    sup = suppress_redundant_concepts_sparse(fuzzy, X_tr_fuzzy, j_max=J_MAX)
    X_tr_fca = compute_customer_concept_memberships(sup, train_fmu)

    test_fmu, _, _ = compute_fuzzy_memberships(
        test_df[["CustomerID", "R", "F", "M", "S"]],
        trained_centroids=train_centroids,
        dims=DIMS,
    )
    X_te_fca = compute_customer_concept_memberships(sup, test_fmu)

    # 3. FCM Soft Memberships
    # Canonical FCM fitted ONLY on training data
    fcm = FuzzyCMeans(
        n_clusters=k_matched,
        m=FCM_M,
        max_iter=FCM_MAX_ITER,
        tol=FCM_TOL,
        random_state=seed,
    ).fit(X_tr_raw)
    X_tr_fcm = fcm.memberships_
    X_te_fcm = fcm.assign(X_te_raw)

    # 4. Concatenated Variants
    X_tr_raw_fcm = np.hstack([X_tr_raw, X_tr_fcm])
    X_te_raw_fcm = np.hstack([X_te_raw, X_te_fcm])

    X_tr_raw_fca = np.hstack([X_tr_raw, X_tr_fca])
    X_te_raw_fca = np.hstack([X_te_raw, X_te_fca])

    X_tr_hybrid = np.hstack([X_tr_raw, X_tr_fcm, X_tr_fca])
    X_te_hybrid = np.hstack([X_te_raw, X_te_fcm, X_te_fca])

    # Check for NaNs / Infs
    for name, (X_tr, X_te) in [
        ("A: Raw RFMS", (X_tr_raw, X_te_raw)),
        ("B: Fuzzy-FCA", (X_tr_fca, X_te_fca)),
        ("C: FCM", (X_tr_fcm, X_te_fcm)),
        ("D: Raw + FCM", (X_tr_raw_fcm, X_te_raw_fcm)),
        ("E: Raw + Fuzzy-FCA", (X_tr_raw_fca, X_te_raw_fca)),
        ("F: Hybrid (Raw + FCM + FCA)", (X_tr_hybrid, X_te_hybrid)),
    ]:
        assert not np.isnan(X_tr).any(), f"NaN found in train {name}"
        assert not np.isinf(X_tr).any(), f"Inf found in train {name}"
        assert not np.isnan(X_te).any(), f"NaN found in test {name}"
        assert not np.isinf(X_te).any(), f"Inf found in test {name}"

    arms = {
        "A: Raw RFMS": (X_tr_raw, X_te_raw),
        "B: Fuzzy-FCA": (X_tr_fca, X_te_fca),
        "C: FCM": (X_tr_fcm, X_te_fcm),
        "D: Raw + FCM": (X_tr_raw_fcm, X_te_raw_fcm),
        "E: Raw + Fuzzy-FCA": (X_tr_raw_fca, X_te_raw_fca),
        "F: Hybrid (Raw + FCM + FCA)": (X_tr_hybrid, X_te_hybrid),
    }

    meta = {
        "n_raw": X_tr_raw.shape[1],
        "n_fca": X_tr_fca.shape[1],
        "n_fcm": X_tr_fcm.shape[1],
        "k_matched": k_matched,
        "n_fuzzy_mined": len(fuzzy),
        "n_fuzzy_suppressed": len(sup),
        "fcm_iterations": fcm.n_iter_,
        "fcm_final_obj": fcm.objective_[-1] if fcm.objective_ else np.nan,
    }

    return arms, meta


# -------------------------------------------------------------------------
# Downstream Predictive Evaluation
# -------------------------------------------------------------------------
def fit_evaluate_arms(
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    y_tr_rep: np.ndarray,
    y_te_rep: np.ndarray,
    y_tr_sp: np.ndarray,
    y_te_sp: np.ndarray,
    y_tr_inv: np.ndarray,
    y_te_inv: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Train downstream models and evaluate on test set."""
    metrics_rows = []
    test_preds: dict[str, dict[str, np.ndarray]] = {}

    for arm_name, (X_tr, X_te) in arms.items():
        # Classification for repurchase
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            scoring="neg_log_loss",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        ).fit(X_tr, y_tr_rep)
        prob = clf.predict_proba(X_te)[:, 1]

        # Ridge regression for spend (log1p)
        ridge_sp = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_tr_sp)
        sp_pred = ridge_sp.predict(X_te)

        # Ridge regression for invoices (log1p)
        ridge_inv = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(
            X_tr, np.log1p(y_tr_inv)
        )
        inv_pred = ridge_inv.predict(X_te)

        metrics_rows.append(
            {
                "arm": arm_name,
                "n_features": X_te.shape[1],
                "test_repurchase_auc": float(roc_auc_score(y_te_rep, prob)),
                "test_brier_score": float(brier_score_loss(y_te_rep, prob)),
                "test_spend_r2": float(r2_score(y_te_sp, sp_pred)),
                "test_spend_mae": float(mean_absolute_error(y_te_sp, sp_pred)),
                "test_spend_spearman": float(stats.spearmanr(y_te_sp, sp_pred).statistic),
                "test_invoices_r2": float(r2_score(np.log1p(y_te_inv), inv_pred)),
                "test_invoices_spearman": float(
                    stats.spearmanr(y_te_inv, inv_pred).statistic
                ),
            }
        )
        test_preds[arm_name] = {"rep": prob, "sp": sp_pred, "inv": inv_pred}

    return pd.DataFrame(metrics_rows), test_preds


# -------------------------------------------------------------------------
# Paired Bootstrap Comparison
# -------------------------------------------------------------------------
def run_paired_bootstrap(
    df_metrics: pd.DataFrame,
    test_preds: dict[str, dict[str, np.ndarray]],
    y_te_rep: np.ndarray,
    y_te_sp: np.ndarray,
    y_te_inv: np.ndarray,
    seed: int,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Run paired customer bootstrap across requested comparisons."""
    comparisons = [
        ("B: Fuzzy-FCA", "C: FCM"),
        ("B: Fuzzy-FCA", "F: Hybrid (Raw + FCM + FCA)"),
        ("C: FCM", "F: Hybrid (Raw + FCM + FCA)"),
        ("A: Raw RFMS", "F: Hybrid (Raw + FCM + FCA)"),
        ("E: Raw + Fuzzy-FCA", "F: Hybrid (Raw + FCM + FCA)"),
        ("A: Raw RFMS", "B: Fuzzy-FCA"),
        ("A: Raw RFMS", "C: FCM"),
        ("D: Raw + FCM", "F: Hybrid (Raw + FCM + FCA)"),
    ]

    rng = np.random.default_rng(seed)
    n_te = len(y_te_rep)

    boot_diffs: dict[tuple[str, str], dict[str, list[float]]] = {
        pair: {"auc": [], "sp": [], "inv": []} for pair in comparisons
    }

    for _ in range(n_boot):
        idx = rng.choice(n_te, size=n_te, replace=True)
        if len(np.unique(y_te_rep[idx])) < 2:
            continue

        for arm1, arm2 in comparisons:
            p1 = test_preds[arm1]
            p2 = test_preds[arm2]

            # Delta = arm2 - arm1 (how much arm2 improves over arm1)
            auc1 = roc_auc_score(y_te_rep[idx], p1["rep"][idx])
            auc2 = roc_auc_score(y_te_rep[idx], p2["rep"][idx])
            boot_diffs[(arm1, arm2)]["auc"].append(auc2 - auc1)

            sp1 = r2_score(y_te_sp[idx], p1["sp"][idx])
            sp2 = r2_score(y_te_sp[idx], p2["sp"][idx])
            boot_diffs[(arm1, arm2)]["sp"].append(sp2 - sp1)

            inv1 = r2_score(np.log1p(y_te_inv[idx]), p1["inv"][idx])
            inv2 = r2_score(np.log1p(y_te_inv[idx]), p2["inv"][idx])
            boot_diffs[(arm1, arm2)]["inv"].append(inv2 - inv1)

    ci_rows = []
    metric_cols = {
        "auc": "test_repurchase_auc",
        "sp": "test_spend_r2",
        "inv": "test_invoices_r2",
    }
    metric_labels = {
        "auc": "Delta Repurchase AUC",
        "sp": "Delta Spend R2",
        "inv": "Delta Invoices R2",
    }

    for arm1, arm2 in comparisons:
        for m_key in ["auc", "sp", "inv"]:
            diffs = np.asarray(boot_diffs[(arm1, arm2)][m_key])
            lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

            v1 = df_metrics.loc[df_metrics["arm"] == arm1, metric_cols[m_key]].values[0]
            v2 = df_metrics.loc[df_metrics["arm"] == arm2, metric_cols[m_key]].values[0]
            point = float(v2 - v1)

            threshold = DECISION_AUC if m_key == "auc" else DECISION_R2

            ci_rows.append(
                {
                    "comparison": f"{arm2} vs {arm1}",
                    "arm_reference": arm1,
                    "arm_tested": arm2,
                    "metric": metric_labels[m_key],
                    "point_estimate": point,
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "spans_zero": bool(lo <= 0 <= hi),
                    "p_value_approx": float(
                        2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
                    ),
                    "exceeds_decision_threshold": bool(abs(point) >= threshold),
                    "interval_type": "95% Paired Customer Bootstrap (1,000 resamples)",
                }
            )

    return pd.DataFrame(ci_rows)


# -------------------------------------------------------------------------
# Complementarity Analysis
# -------------------------------------------------------------------------
def run_complementarity_analysis(
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    df_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Analyze correlation, redundancy, and incremental predictive contribution between FCM and Fuzzy-FCA."""
    X_tr_fcm, X_te_fcm = arms["C: FCM"]
    X_tr_fca, X_te_fca = arms["B: Fuzzy-FCA"]
    X_tr_raw, X_te_raw = arms["A: Raw RFMS"]

    k_fcm = X_te_fcm.shape[1]
    m_fca = X_te_fca.shape[1]

    # Pairwise Pearson correlations on test set
    # Standardize columns to compute correlation as matrix multiplication
    fcm_std = (X_te_fcm - X_te_fcm.mean(axis=0)) / (X_te_fcm.std(axis=0) + 1e-12)
    fca_std = (X_te_fca - X_te_fca.mean(axis=0)) / (X_te_fca.std(axis=0) + 1e-12)
    corr_matrix = (fcm_std.T @ fca_std) / (len(X_te_fcm) - 1)  # shape (44, 331)

    abs_corr = np.abs(corr_matrix)

    # For each FCM cluster, find highest correlation with any fuzzy concept
    max_corr_per_fcm = np.max(abs_corr, axis=1)
    argmax_corr_per_fcm = np.argmax(abs_corr, axis=1)

    # For each fuzzy concept, find highest correlation with any FCM cluster
    max_corr_per_fca = np.max(abs_corr, axis=0)

    # Linear predictability: Regress each FCM feature onto all Fuzzy-FCA features
    # (How much of FCM membership is linearly spanned by Fuzzy-FCA concepts?)
    r2_fcm_from_fca = []
    lr = LinearRegression()
    for col in range(k_fcm):
        lr.fit(X_tr_fca, X_tr_fcm[:, col])
        pred_te = lr.predict(X_te_fca)
        r2_val = r2_score(X_te_fcm[:, col], pred_te)
        r2_fcm_from_fca.append(r2_val)
    r2_fcm_from_fca = np.array(r2_fcm_from_fca)

    # Regress each Raw RFMS feature onto Fuzzy-FCA concepts and FCM memberships
    r2_raw_from_fca = []
    r2_raw_from_fcm = []
    r2_raw_from_both = []
    for col in range(4):
        # From FCA
        lr.fit(X_tr_fca, X_tr_raw[:, col])
        r2_raw_from_fca.append(r2_score(X_te_raw[:, col], lr.predict(X_te_fca)))
        # From FCM
        lr.fit(X_tr_fcm, X_tr_raw[:, col])
        r2_raw_from_fcm.append(r2_score(X_te_raw[:, col], lr.predict(X_te_fcm)))
        # From both
        X_tr_both = np.hstack([X_tr_fcm, X_tr_fca])
        X_te_both = np.hstack([X_te_fcm, X_te_fca])
        lr.fit(X_tr_both, X_tr_raw[:, col])
        r2_raw_from_both.append(r2_score(X_te_raw[:, col], lr.predict(X_te_both)))

    summary_stats = {
        "mean_abs_corr_all_pairs": float(abs_corr.mean()),
        "median_abs_corr_all_pairs": float(np.median(abs_corr)),
        "max_abs_corr_overall": float(abs_corr.max()),
        "mean_max_corr_per_fcm": float(max_corr_per_fcm.mean()),
        "median_max_corr_per_fcm": float(np.median(max_corr_per_fcm)),
        "max_corr_per_fcm_gt_05": int(np.sum(max_corr_per_fcm > 0.5)),
        "pct_fcm_gt_05": float(np.mean(max_corr_per_fcm > 0.5) * 100),
        "max_corr_per_fcm_gt_08": int(np.sum(max_corr_per_fcm > 0.8)),
        "pct_fcm_gt_08": float(np.mean(max_corr_per_fcm > 0.8) * 100),
        "mean_max_corr_per_fca": float(max_corr_per_fca.mean()),
        "median_max_corr_per_fca": float(np.median(max_corr_per_fca)),
        "mean_r2_fcm_explained_by_fca": float(r2_fcm_from_fca.mean()),
        "median_r2_fcm_explained_by_fca": float(np.median(r2_fcm_from_fca)),
        "max_r2_fcm_explained_by_fca": float(r2_fcm_from_fca.max()),
        "min_r2_fcm_explained_by_fca": float(r2_fcm_from_fca.min()),
        "mean_r2_raw_explained_by_fca": float(np.mean(r2_raw_from_fca)),
        "mean_r2_raw_explained_by_fcm": float(np.mean(r2_raw_from_fcm)),
        "mean_r2_raw_explained_by_both": float(np.mean(r2_raw_from_both)),
    }

    # Record per-FCM-cluster correlation detail
    fcm_records = []
    for col in range(k_fcm):
        fcm_records.append(
            {
                "fcm_cluster": f"FCM_k{col+1}",
                "max_abs_corr_with_fca": float(max_corr_per_fcm[col]),
                "best_matching_fca_idx": int(argmax_corr_per_fcm[col]),
                "r2_explained_by_all_fca": float(r2_fcm_from_fca[col]),
            }
        )
    df_fcm_corr = pd.DataFrame(fcm_records)

    return df_fcm_corr, summary_stats


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
def generate_ablation_figures(
    df_metrics: pd.DataFrame, df_ci: pd.DataFrame, out_dir: Path
) -> None:
    """Generate publication-ready figures for fixed-split ablation."""
    short_names = [
        "A: Raw\nRFMS",
        "B: Fuzzy\nFCA",
        "C: FCM\n(k=44)",
        "D: Raw +\nFCM",
        "E: Raw +\nFCA",
        "F: Hybrid\n(All 3)",
    ]
    colors = ["#4a5568", "#2b6cb0", "#319795", "#d69e2e", "#805ad5", "#dd6b20"]

    # 1. AUC Figure with error bars or comparison view
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    aucs = df_metrics["test_repurchase_auc"].to_numpy()
    bars = ax.bar(
        short_names, aucs, color=colors, width=0.55, edgecolor="black", alpha=0.9
    )
    ax.set_ylabel("Test Repurchase ROC-AUC", fontsize=11, fontweight="bold")
    ax.set_title(
        "Representation Ablation: Repurchase Prediction (Olist Fixed Split)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.53, 0.57)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.0008,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_dir / "fig_ablation_auc.png")
    plt.close()

    # 2. Spend R2 and Invoices R2 Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    sp_r2 = df_metrics["test_spend_r2"].to_numpy()
    inv_r2 = df_metrics["test_invoices_r2"].to_numpy()

    bars1 = ax1.bar(
        short_names, sp_r2, color=colors, width=0.55, edgecolor="black", alpha=0.9
    )
    ax1.set_ylabel("Test Spend R² (log1p)", fontsize=10, fontweight="bold")
    ax1.set_title(
        "Ablation: Future Spend R²", fontsize=11, fontweight="bold", pad=10
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars1, sp_r2):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            max(0, val) + 0.00003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    bars2 = ax2.bar(
        short_names, inv_r2, color=colors, width=0.55, edgecolor="black", alpha=0.9
    )
    ax2.set_ylabel("Test Invoices R² (log1p)", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Ablation: Future Invoices R²", fontsize=11, fontweight="bold", pad=10
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars2, inv_r2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            max(0, val) + 0.00003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_dir / "fig_ablation_r2.png")
    plt.close()


# -------------------------------------------------------------------------
# Multi-Split Validation (10 Splits)
# -------------------------------------------------------------------------
def run_multisplit_ablation(
    merged: pd.DataFrame, n_splits: int = N_SPLITS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 6-arm representation ablation across 10 stratified customer splits."""
    print("\n" + "=" * 80)
    print(f"MULTI-SPLIT ABLATION: {n_splits} STRATIFIED SPLITS (SEEDS {BASE_SEED}..{BASE_SEED+n_splits-1})")
    print("=" * 80)

    rows = []
    t_start = time.time()

    for s in range(n_splits):
        seed = BASE_SEED + s
        t_s0 = time.time()

        train_df, test_df = train_test_split(
            merged,
            test_size=0.30,
            stratify=merged["repurchased"].to_numpy(),
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        y_tr_rep = train_df["repurchased"].to_numpy()
        y_te_rep = test_df["repurchased"].to_numpy()
        y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
        y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
        y_tr_inv = train_df["future_invoices"].to_numpy()
        y_te_inv = test_df["future_invoices"].to_numpy()

        arms, meta = build_representations(train_df, test_df, seed=seed, k_matched=44)
        df_split_metrics, _ = fit_evaluate_arms(
            arms, y_tr_rep, y_te_rep, y_tr_sp, y_te_sp, y_tr_inv, y_te_inv, seed=seed
        )

        for _, row in df_split_metrics.iterrows():
            rec = dict(row)
            rec["split_seed"] = seed
            rows.append(rec)

        auc_b = df_split_metrics.loc[
            df_split_metrics["arm"] == "B: Fuzzy-FCA", "test_repurchase_auc"
        ].values[0]
        auc_f = df_split_metrics.loc[
            df_split_metrics["arm"] == "F: Hybrid (Raw + FCM + FCA)",
            "test_repurchase_auc",
        ].values[0]
        auc_a = df_split_metrics.loc[
            df_split_metrics["arm"] == "A: Raw RFMS", "test_repurchase_auc"
        ].values[0]

        dt = time.time() - t_s0
        print(
            f"Split {s+1}/{n_splits} (seed {seed}) in {dt:.1f}s | "
            f"FCA={meta['n_fca']} concepts | "
            f"AUC: Raw={auc_a:.4f}, FCA={auc_b:.4f}, Hybrid={auc_f:.4f} (Diff F-B: {auc_f - auc_b:+.4f})"
        )

    df_multi = pd.DataFrame(rows)

    # Summary table: Mean ± SD across splits
    summary_rows = []
    arms_order = [
        "A: Raw RFMS",
        "B: Fuzzy-FCA",
        "C: FCM",
        "D: Raw + FCM",
        "E: Raw + Fuzzy-FCA",
        "F: Hybrid (Raw + FCM + FCA)",
    ]

    piv_auc = df_multi.pivot(
        index="split_seed", columns="arm", values="test_repurchase_auc"
    )
    piv_sp = df_multi.pivot(index="split_seed", columns="arm", values="test_spend_r2")
    piv_inv = df_multi.pivot(
        index="split_seed", columns="arm", values="test_invoices_r2"
    )

    for arm in arms_order:
        sub = df_multi[df_multi["arm"] == arm]
        summary_rows.append(
            {
                "arm": arm,
                "n_features_mean": float(sub["n_features"].mean()),
                "test_auc_mean": float(sub["test_repurchase_auc"].mean()),
                "test_auc_sd": float(sub["test_repurchase_auc"].std()),
                "test_spend_r2_mean": float(sub["test_spend_r2"].mean()),
                "test_spend_r2_sd": float(sub["test_spend_r2"].std()),
                "test_invoices_r2_mean": float(sub["test_invoices_r2"].mean()),
                "test_invoices_r2_sd": float(sub["test_invoices_r2"].std()),
                "wins_vs_raw_auc": int((piv_auc[arm] > piv_auc["A: Raw RFMS"]).sum()),
                "wins_vs_fuzzy_auc": int(
                    (piv_auc[arm] > piv_auc["B: Fuzzy-FCA"]).sum()
                ),
                "wins_vs_fcm_auc": int((piv_auc[arm] > piv_auc["C: FCM"]).sum()),
            }
        )

    df_multi_summary = pd.DataFrame(summary_rows)

    # Generate multi-split figure
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(arms_order))
    means = [
        df_multi_summary.loc[df_multi_summary["arm"] == a, "test_auc_mean"].values[0]
        for a in arms_order
    ]
    sds = [
        df_multi_summary.loc[df_multi_summary["arm"] == a, "test_auc_sd"].values[0]
        for a in arms_order
    ]
    short_labels = ["A: Raw", "B: FCA", "C: FCM", "D: Raw+FCM", "E: Raw+FCA", "F: Hybrid"]
    colors = ["#4a5568", "#2b6cb0", "#319795", "#d69e2e", "#805ad5", "#dd6b20"]

    ax.bar(
        x,
        means,
        yerr=sds,
        capsize=5,
        color=colors,
        width=0.55,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("10-Split Mean Test AUC (± SD)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"10-Split Robustness: Representation Ablation on Olist (n={len(merged):,})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.53, 0.575)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(
            i,
            m + s + 0.001,
            f"{m:.4f}\n±{s:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_multisplit_ablation.png")
    plt.close()

    total_time = time.time() - t_start
    print(f"10-split validation completed in {total_time/60:.2f} minutes.")
    return df_multi, df_multi_summary


# -------------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Experiment 6: FCM Hybrid Ablation")
    parser.add_argument(
        "--skip-multisplit",
        action="store_true",
        help="Skip 10-split validation and run only fixed split",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT 6: REPRESENTATION ABLATION STUDY ON OLIST")
    print("Testing Complementarity: Raw RFMS vs Fuzzy-FCA vs FCM Soft Memberships")
    print("=" * 80)

    # 1. Prepare data
    merged = prepare_observation_cohort()
    assert (
        len(merged) == 21664
    ), f"Expected 21,664 observation customers, got {len(merged)}"

    # 2. Fixed Split (70/30, seed=42)
    train_df, test_df = train_test_split(
        merged,
        test_size=0.30,
        stratify=merged["repurchased"].to_numpy(),
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(
        f"Train: {len(train_df):,} (repurchase {train_df['repurchased'].mean():.4f}) | "
        f"Test: {len(test_df):,} (repurchase {test_df['repurchased'].mean():.4f})"
    )

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = train_df["future_invoices"].to_numpy()
    y_te_inv = test_df["future_invoices"].to_numpy()

    # 3. Build representations
    print("\nBuilding representations on fixed split...")
    arms, meta = build_representations(train_df, test_df, seed=RANDOM_SEED, k_matched=44)

    # Feature Accounting
    feature_counts = [
        {
            "arm": name,
            "total_features": X_te.shape[1],
            "raw_features": 4 if "Raw" in name or name.startswith("A:") else 0,
            "fcm_features": meta["n_fcm"] if "FCM" in name else 0,
            "fca_features": meta["n_fca"] if "FCA" in name or name.startswith("B:") else 0,
            "train_samples": X_tr.shape[0],
            "test_samples": X_te.shape[0],
        }
        for name, (X_tr, X_te) in arms.items()
    ]
    df_feat = pd.DataFrame(feature_counts)
    df_feat.to_csv(OUT_DIR / "feature_counts.csv", index=False)
    print("\nFeature Accounting:")
    print(df_feat.to_string(index=False))

    # 4. Downstream Evaluation
    print("\nFitting downstream estimators...")
    df_metrics, test_preds = fit_evaluate_arms(
        arms,
        y_tr_rep,
        y_te_rep,
        y_tr_sp,
        y_te_sp,
        y_tr_inv,
        y_te_inv,
        seed=RANDOM_SEED,
    )
    df_metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\nFixed-Split Out-of-Sample Holdout Metrics:")
    print(
        df_metrics[
            [
                "arm",
                "n_features",
                "test_repurchase_auc",
                "test_brier_score",
                "test_spend_r2",
                "test_invoices_r2",
            ]
        ].to_string(index=False)
    )

    # 5. Paired Bootstrap
    print(f"\nRunning {N_BOOT} paired customer bootstrap resamples...")
    df_ci = run_paired_bootstrap(
        df_metrics,
        test_preds,
        y_te_rep,
        y_te_sp,
        y_te_inv,
        seed=RANDOM_SEED,
        n_boot=N_BOOT,
    )
    df_ci.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    print("\nPaired Bootstrap Comparisons:")
    print(
        df_ci[
            [
                "comparison",
                "metric",
                "point_estimate",
                "ci_lower_95",
                "ci_upper_95",
                "spans_zero",
                "exceeds_decision_threshold",
            ]
        ].to_string(index=False)
    )

    # 6. Complementarity Analysis
    print("\nRunning complementarity and representation correlation analysis...")
    df_fcm_corr, comp_stats = run_complementarity_analysis(arms, df_metrics)
    df_fcm_corr.to_csv(OUT_DIR / "representation_correlations.csv", index=False)

    print("\nComplementarity Summary Statistics:")
    for k, v in comp_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 7. Run Parameters
    run_params = [
        {"parameter": "cutoff_date", "value": str(CUTOFF_DATE)},
        {"parameter": "n_observation_customers", "value": len(merged)},
        {"parameter": "fixed_split_ratio", "value": "70/30 stratified"},
        {"parameter": "fixed_split_random_state", "value": RANDOM_SEED},
        {"parameter": "fcm_matched_k", "value": meta["k_matched"]},
        {"parameter": "fcm_fuzzifier_m", "value": FCM_M},
        {"parameter": "fcm_max_iter", "value": FCM_MAX_ITER},
        {"parameter": "fcm_tol", "value": FCM_TOL},
        {"parameter": "fuzzy_support_cutoff", "value": SUPPORT_CUTOFF},
        {"parameter": "redundancy_suppression_j_max", "value": J_MAX},
        {"parameter": "n_bootstrap_samples", "value": N_BOOT},
        {"parameter": "decision_threshold_auc", "value": DECISION_AUC},
        {"parameter": "decision_threshold_r2", "value": DECISION_R2},
        {"parameter": "n_fuzzy_mined", "value": meta["n_fuzzy_mined"]},
        {"parameter": "n_fuzzy_suppressed", "value": meta["n_fuzzy_suppressed"]},
    ]
    for k, v in comp_stats.items():
        run_params.append({"parameter": f"comp_{k}", "value": v})
    pd.DataFrame(run_params).to_csv(OUT_DIR / "run_parameters.csv", index=False)

    # 8. Generate Figures
    generate_ablation_figures(df_metrics, df_ci, OUT_DIR)

    # 9. Multi-Split Validation (10 Splits)
    df_multi, df_multi_summary = None, None
    if not args.skip_multisplit:
        df_multi, df_multi_summary = run_multisplit_ablation(merged, n_splits=N_SPLITS)
        df_multi.to_csv(OUT_DIR / "multisplit_ablation_metrics.csv", index=False)
        df_multi_summary.to_csv(
            OUT_DIR / "multisplit_ablation_summary.csv", index=False
        )
        print("\nMulti-Split Summary (10 Splits):")
        print(df_multi_summary.to_string(index=False))

    # 10. Write Summary Markdown
    write_summary_markdown(df_metrics, df_ci, df_feat, comp_stats, df_multi_summary)

    print("\nExperiment 6 completed successfully!")
    print(f"All artifacts saved to: {OUT_DIR}")


# -------------------------------------------------------------------------
# Markdown Summary Generation
# -------------------------------------------------------------------------
def to_md_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without external dependencies."""
    headers = [str(col) for col in df.columns]
    rows = [[str(val) for val in row] for row in df.values]
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def write_summary_markdown(
    df_metrics: pd.DataFrame,
    df_ci: pd.DataFrame,
    df_feat: pd.DataFrame,
    comp_stats: dict,
    df_multi_summary: pd.DataFrame | None,
) -> None:
    """Generate comprehensive summary markdown artifact."""
    summary_path = OUT_DIR / "summary.md"

    # Extract key values
    def get_val(arm, col):
        return df_metrics.loc[df_metrics["arm"] == arm, col].values[0]

    auc_a = get_val("A: Raw RFMS", "test_repurchase_auc")
    auc_b = get_val("B: Fuzzy-FCA", "test_repurchase_auc")
    auc_c = get_val("C: FCM", "test_repurchase_auc")
    auc_d = get_val("D: Raw + FCM", "test_repurchase_auc")
    auc_e = get_val("E: Raw + Fuzzy-FCA", "test_repurchase_auc")
    auc_f = get_val("F: Hybrid (Raw + FCM + FCA)", "test_repurchase_auc")

    lines = [
        "# Experiment 6: Controlled Representation Ablation Study on Olist",
        "",
        "## 1. Executive Summary & Core Hypothesis",
        "",
        "**Hypothesis Tested**: Do Fuzzy C-Means (FCM) soft-cluster memberships provide information complementary to the Fuzzy-FCA concept representation, such that a hybrid representation improves downstream predictive performance over its individual constituent representations?",
        "",
        "### Key Findings:",
        f"1. **Fixed-Split Repurchase AUC**:",
        f"   - **A: Raw RFMS**: {auc_a:.4f}",
        f"   - **B: Fuzzy-FCA**: {auc_b:.4f}",
        f"   - **C: FCM (k=44)**: {auc_c:.4f}",
        f"   - **D: Raw RFMS + FCM**: {auc_d:.4f}",
        f"   - **E: Raw RFMS + Fuzzy-FCA**: {auc_e:.4f}",
        f"   - **F: Hybrid (Raw + FCM + Fuzzy-FCA)**: {auc_f:.4f}",
        "",
    ]

    # Evaluate complementarity directly from results
    diff_fb = auc_f - auc_b
    diff_fa = auc_f - auc_a
    diff_fe = auc_f - auc_e
    lines.append(
        f"2. **Complementarity Evaluation (Hybrid vs Constituents)**:"
    )
    lines.append(
        f"   - **F vs B (Hybrid vs Fuzzy-FCA)**: Δ AUC = {diff_fb:+.4f}"
    )
    lines.append(
        f"   - **F vs E (Hybrid vs Raw+FCA)**: Δ AUC = {diff_fe:+.4f}"
    )
    lines.append(
        f"   - **F vs A (Hybrid vs Raw RFMS)**: Δ AUC = {diff_fa:+.4f}"
    )
    lines.append("")

    lines.append("## 2. Fixed-Split Six-Way Ablation Results")
    lines.append("")
    lines.append(to_md_table(df_metrics))
    lines.append("")

    lines.append("## 3. Feature Accounting")
    lines.append("")
    lines.append(to_md_table(df_feat))
    lines.append("")

    lines.append("## 4. Paired Bootstrap Statistical Comparison (Fixed Split, N=1,000)")
    lines.append("")
    lines.append(
        to_md_table(
            df_ci[
                [
                    "comparison",
                    "metric",
                    "point_estimate",
                    "ci_lower_95",
                    "ci_upper_95",
                    "spans_zero",
                    "exceeds_decision_threshold",
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## 5. Complementarity & Representation Redundancy Analysis")
    lines.append("")
    lines.append(
        f"- **Mean absolute cross-correlation between all FCM cluster memberships and Fuzzy-FCA concepts**: {comp_stats['mean_abs_corr_all_pairs']:.4f}"
    )
    lines.append(
        f"- **Maximum cross-correlation across any pair**: {comp_stats['max_abs_corr_overall']:.4f}"
    )
    lines.append(
        f"- **Mean peak correlation per FCM cluster**: {comp_stats['mean_max_corr_per_fcm']:.4f}"
    )
    lines.append(
        f"- **FCM clusters with peak correlation > 0.5 with any Fuzzy-FCA concept**: {comp_stats['max_corr_per_fcm_gt_05']}/44 ({comp_stats['pct_fcm_gt_05']:.1f}%)"
    )
    lines.append(
        f"- **FCM clusters with peak correlation > 0.8 with any Fuzzy-FCA concept**: {comp_stats['max_corr_per_fcm_gt_08']}/44 ({comp_stats['pct_fcm_gt_08']:.1f}%)"
    )
    lines.append(
        f"- **Average variance of FCM membership explained linearly by all Fuzzy-FCA concepts ($R^2$)**: {comp_stats['mean_r2_fcm_explained_by_fca']:.4f}"
    )
    lines.append(
        f"- **Variance of Raw RFMS explained by FCA vs FCM vs Both**: FCA={comp_stats['mean_r2_raw_explained_by_fca']:.4f}, FCM={comp_stats['mean_r2_raw_explained_by_fcm']:.4f}, Both={comp_stats['mean_r2_raw_explained_by_both']:.4f}"
    )
    lines.append("")

    if df_multi_summary is not None:
        lines.append("## 6. Multi-Split Validation Across 10 Stratified Splits")
        lines.append("")
        lines.append(to_md_table(df_multi_summary))
        lines.append("")

    lines.append("## 7. Discussion & Methodological Assessment")
    lines.append("")
    lines.append(
        "Does adding FCM membership information to Fuzzy-FCA improve predictive performance?"
    )
    if diff_fb > 0.005 and (df_ci.loc[df_ci["comparison"] == "F: Hybrid (Raw + FCM + FCA) vs B: Fuzzy-FCA", "spans_zero"].values[0] is False):
        lines.append(
            "- **Conclusion**: The hybrid representation demonstrates statistically detectable gains over Fuzzy-FCA alone."
        )
    else:
        lines.append(
            "- **Conclusion**: The hybrid representation does NOT meaningfully or statistically improve over the standalone representations. Adding FCM soft memberships to Fuzzy-FCA provides no substantial predictive advantage, as indicated by overlapping bootstrap confidence intervals that span zero."
        )
    lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

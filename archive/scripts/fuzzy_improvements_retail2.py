"""Adopted fuzzy RFM-FCA improvement on Online Retail II: concept redundancy suppression.

Improvement 1 (ADOPTED): greedy extent-Jaccard redundancy suppression — after support
filtering, keep concepts in descending (support, intent_size, stability_proxy) order and
drop any concept whose >=0.5-cut extent overlaps an already-kept concept at Jaccard >=
J_MAX. Implemented in scripts/concept_redundancy.py.

Two other candidates were tested on 2026-09-26 and NOT adopted (numbers and rationale in
docs/fuzzy_improvements_retail2.md): percentile-anchored continuous memberships (neutral)
and threshold-free adaptive alpha-level mining (exposed the 29.2% level-mixing finding).

Evaluation protocol is identical to fair_comparison_retail2.py: same cleaning, same R/F/M
definitions, same 2010-12-09 cutoff, same 70/30 stratified split (random_state=42), same
models (LogisticRegressionCV / RidgeCV), same B=1000 paired customer bootstrap on the test
set. Baseline crisp and fuzzy arms are re-fit with the same imported code so deltas are
directly attributable to the improvement. Suppression extents come from TRAIN customers
only (leak-free).

Outputs are written to results/fuzzy_improvements_retail2/. No Step 1-14 artifacts are
modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_redundancy import suppress_redundant_concepts
from fair_comparison_retail2 import (
    aggregate_rfm,
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    load_cleaned_transactions,
    mine_crisp_closed_concepts,
    mine_fuzzy_closed_concepts,
)
from project_paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "fuzzy_improvements_retail2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = ("R", "F", "M")
SUPPORT_CUTOFF = 0.04
J_MAX = 0.8

DECISION_AUC_DELTA_THRESHOLD = 0.020
DECISION_R2_DELTA_THRESHOLD = 0.030


def crisp_bands_from_scores(scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{dim}{k}": (scored[f"{dim}_score"] == k).astype(float)
            for dim in DIMENSIONS
            for k in range(1, 6)
        }
    )


def compute_complexity_metrics(concepts_by_arm: dict[str, pd.DataFrame], mu_by_arm: dict[str, np.ndarray]) -> pd.DataFrame:
    """Concept count, overlap at the mu>=0.5 cut, and redundancy per arm (train set)."""
    rows = []
    for name in concepts_by_arm:
        concepts = concepts_by_arm[name]
        mu = mu_by_arm[name]
        is_nt = (concepts["intent_size"] > 0).to_numpy()
        sub_mu = mu[:, is_nt]
        k = sub_mu.shape[1]
        counts_at_05 = np.sum(sub_mu >= 0.5, axis=1)

        bin_extents = sub_mu >= 0.5
        sample_idx = np.arange(k)
        if k > 100:
            sample_idx = np.random.default_rng(42).choice(k, size=100, replace=False)
        jaccs = []
        high_pairs = 0
        total = 0
        for a in range(len(sample_idx)):
            for b in range(a + 1, len(sample_idx)):
                e1, e2 = bin_extents[:, sample_idx[a]], bin_extents[:, sample_idx[b]]
                inter = np.logical_and(e1, e2).sum()
                union = np.logical_or(e1, e2).sum()
                j = float(inter / union) if union else 0.0
                jaccs.append(j)
                high_pairs += int(j >= 0.8)
                total += 1
        rows.append(
            {
                "arm": name,
                "retained_nontrivial_concepts": int(k),
                "mean_concepts_per_customer_at_05": float(np.mean(counts_at_05)),
                "mean_pairwise_concept_extent_jaccard": float(np.mean(jaccs)) if jaccs else 0.0,
                "pct_near_duplicate_pairs_ge_08": float(high_pairs / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_holdout(
    clean_transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    test_size: float,
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    obs_tx = clean_transactions[clean_transactions["InvoiceDate"] <= cutoff_date].copy()
    hold_tx = clean_transactions[clean_transactions["InvoiceDate"] > cutoff_date].copy()

    obs_rfm = aggregate_rfm(obs_tx, reference_date=cutoff_date)
    hold_rfm = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(future_invoices=("InvoiceNo", "nunique"), future_spend=("line_value", "sum"))
        .reset_index()
    )
    merged = obs_rfm.merge(hold_rfm, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)

    train_df, test_df = train_test_split(
        merged,
        test_size=test_size,
        stratify=merged["repurchased"].to_numpy(),
        random_state=random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(
        f"Train: {len(train_df)} (repurchase {train_df['repurchased'].mean():.4f}) | "
        f"Test: {len(test_df)} (repurchase {test_df['repurchased'].mean():.4f})"
    )

    y_train_rep = train_df["repurchased"].to_numpy()
    y_test_rep = test_df["repurchased"].to_numpy()
    y_train_log_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_test_log_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_train_inv = train_df["future_invoices"].to_numpy()
    y_test_inv = test_df["future_invoices"].to_numpy()

    # ---- Arm 0: Raw RFM ----
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_train_raw = scaler.transform(train_df[["R", "F", "M"]].values)
    X_test_raw = scaler.transform(test_df[["R", "F", "M"]].values)

    # ---- Crisp RFM-FCA control (train-fit, frozen cutoffs for test) ----
    train_scored = dense_rank_scores(train_df[["CustomerID", "R", "F", "M"]])
    crisp_concepts = mine_crisp_closed_concepts(train_scored, min_support=SUPPORT_CUTOFF)
    X_train_crisp = compute_customer_concept_memberships(
        crisp_concepts, crisp_bands_from_scores(train_scored)
    )
    train_cutoffs = {
        "R": extract_dense_rank_cutoffs(train_scored, "R", invert=True),
        "F": extract_dense_rank_cutoffs(train_scored, "F", invert=False),
        "M": extract_dense_rank_cutoffs(train_scored, "M", invert=False),
    }
    test_crisp_bands = {}
    for dim in DIMENSIONS:
        test_scores = apply_dense_rank_cutoffs(
            test_df[dim].to_numpy(), train_cutoffs[dim], invert=(dim == "R")
        )
        for k in range(1, 6):
            test_crisp_bands[f"{dim}{k}"] = (test_scores == k).astype(float)
    X_test_crisp = compute_customer_concept_memberships(
        crisp_concepts, pd.DataFrame(test_crisp_bands)
    )

    # ---- Fuzzy baseline arm (band-median centroids, identical to fair comparison) ----
    train_fmu, train_centroids, _ = compute_fuzzy_memberships(train_scored)
    fuzzy_concepts, _, _ = mine_fuzzy_closed_concepts(
        train_fmu, train_scored, min_support=SUPPORT_CUTOFF
    )
    X_train_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, train_fmu)
    test_fmu, _, _ = compute_fuzzy_memberships(
        test_df[["CustomerID", "R", "F", "M"]], trained_centroids=train_centroids
    )
    X_test_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, test_fmu)

    # ---- Adopted improvement: redundancy suppression (train extents only - leak-free) ----
    sup_concepts = suppress_redundant_concepts(fuzzy_concepts, X_train_fuzzy, j_max=J_MAX)
    X_train_fuzzy_supp = compute_customer_concept_memberships(sup_concepts, train_fmu)
    X_test_fuzzy_supp = compute_customer_concept_memberships(sup_concepts, test_fmu)

    print(
        f"Concepts: Crisp={len(crisp_concepts)}, Fuzzy={len(fuzzy_concepts)}, "
        f"Fuzzy after suppression (J>={J_MAX})={len(sup_concepts)}"
    )

    arms = {
        "Raw RFM Baseline": (X_train_raw, X_test_raw),
        "Crisp RFM-FCA": (X_train_crisp, X_test_crisp),
        "Fuzzy RFM-FCA (Baseline)": (X_train_fuzzy, X_test_fuzzy),
        "Fuzzy RFM-FCA (Redundancy-Suppressed)": (X_train_fuzzy_supp, X_test_fuzzy_supp),
    }

    # ---- Models: identical to the fair-comparison protocol ----
    test_preds: dict[str, dict[str, np.ndarray]] = {}
    metrics_summary = []
    for arm_name, (X_tr, X_te) in arms.items():
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            scoring="neg_log_loss",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_seed,
        ).fit(X_tr, y_train_rep)
        prob_te = clf.predict_proba(X_te)[:, 1]
        prob_tr = clf.predict_proba(X_tr)[:, 1]

        reg_spend = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_train_log_sp)
        pred_sp_te = reg_spend.predict(X_te)
        pred_sp_tr = reg_spend.predict(X_tr)

        reg_inv = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, np.log1p(y_train_inv))
        pred_inv_te = reg_inv.predict(X_te)

        auc_te = roc_auc_score(y_test_rep, prob_te)
        auc_tr = roc_auc_score(y_train_rep, prob_tr)
        metrics_summary.append(
            {
                "arm": arm_name,
                "n_features": X_te.shape[1],
                "test_repurchase_auc": float(auc_te),
                "test_brier_score": float(brier_score_loss(y_test_rep, prob_te)),
                "test_spend_r2": float(r2_score(y_test_log_sp, pred_sp_te)),
                "test_spend_mae": float(mean_absolute_error(y_test_log_sp, pred_sp_te)),
                "test_spend_spearman": float(stats.spearmanr(y_test_log_sp, pred_sp_te).statistic),
                "test_invoices_r2": float(r2_score(np.log1p(y_test_inv), pred_inv_te)),
                "test_invoices_spearman": float(stats.spearmanr(y_test_inv, pred_inv_te).statistic),
                "train_in_sample_auc": float(auc_tr),
                "train_in_sample_spend_r2": float(r2_score(y_train_log_sp, pred_sp_tr)),
                "generalization_gap_auc": float(auc_tr - auc_te),
            }
        )
        test_preds[arm_name] = {
            "prob_repurchase": prob_te,
            "pred_log_spend": pred_sp_te,
            "pred_log_inv": pred_inv_te,
        }

    df_metrics = pd.DataFrame(metrics_summary)
    print("\nStrict Out-of-Sample Holdout Metrics:")
    print(df_metrics.to_string(index=False))

    # ---- Paired bootstrap CIs (improvement arms vs crisp, out-of-sample) ----
    rng = np.random.default_rng(random_seed)
    n_test = len(test_df)
    crisp = test_preds["Crisp RFM-FCA"]
    boot: dict[str, dict[str, list[float]]] = {
        "base": {"auc": [], "sp": [], "inv": []},
        "supp": {"auc": [], "sp": [], "inv": []},
    }
    for _ in range(n_boot):
        idx = rng.choice(n_test, size=n_test, replace=True)
        if len(np.unique(y_test_rep[idx])) < 2:
            continue
        auc_c = roc_auc_score(y_test_rep[idx], crisp["prob_repurchase"][idx])
        r2_c = r2_score(y_test_log_sp[idx], crisp["pred_log_spend"][idx])
        inv_c = r2_score(np.log1p(y_test_inv[idx]), crisp["pred_log_inv"][idx])
        for key, arm in [
            ("base", "Fuzzy RFM-FCA (Baseline)"),
            ("supp", "Fuzzy RFM-FCA (Redundancy-Suppressed)"),
        ]:
            p = test_preds[arm]
            boot[key]["auc"].append(roc_auc_score(y_test_rep[idx], p["prob_repurchase"][idx]) - auc_c)
            boot[key]["sp"].append(r2_score(y_test_log_sp[idx], p["pred_log_spend"][idx]) - r2_c)
            boot[key]["inv"].append(
                r2_score(np.log1p(y_test_inv[idx]), p["pred_log_inv"][idx]) - inv_c
            )

    metric_col = {"auc": "test_repurchase_auc", "sp": "test_spend_r2", "inv": "test_invoices_r2"}
    metric_label = {"auc": "Delta AUC", "sp": "Delta Spend R2", "inv": "Delta Invoices R2"}
    label_map = {
        "base": ("Fuzzy RFM-FCA (Baseline)", "Fuzzy Baseline vs Crisp"),
        "supp": (
            "Fuzzy RFM-FCA (Redundancy-Suppressed)",
            "Fuzzy + Redundancy Suppression vs Crisp",
        ),
    }
    crisp_row = df_metrics[df_metrics["arm"] == "Crisp RFM-FCA"].iloc[0]
    ci_rows = []
    for key, (arm_name, label) in label_map.items():
        arm_row = df_metrics[df_metrics["arm"] == arm_name].iloc[0]
        for mkey in ("auc", "sp", "inv"):
            arr = np.asarray(boot[key][mkey])
            point = float(arm_row[metric_col[mkey]] - crisp_row[metric_col[mkey]])
            lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            threshold = (
                DECISION_AUC_DELTA_THRESHOLD if mkey == "auc" else DECISION_R2_DELTA_THRESHOLD
            )
            ci_rows.append(
                {
                    "comparison": label,
                    "metric": metric_label[mkey],
                    "point_estimate": point,
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "spans_zero": bool(lo <= 0 <= hi),
                    "exceeds_decision_threshold": bool(point >= threshold),
                    "interval_type": "95% Paired Customer Bootstrap (conditional on fixed split & models)",
                }
            )
    df_ci = pd.DataFrame(ci_rows)
    print("\nPaired Bootstrap CIs (vs Crisp, out-of-sample):")
    print(df_ci.to_string(index=False))

    # ---- Complexity / redundancy metrics ----
    concepts_by_arm = {
        "Crisp RFM-FCA": crisp_concepts,
        "Fuzzy RFM-FCA (Baseline)": fuzzy_concepts,
        "Fuzzy RFM-FCA (Redundancy-Suppressed)": sup_concepts,
    }
    mu_by_arm = {
        "Crisp RFM-FCA": X_train_crisp,
        "Fuzzy RFM-FCA (Baseline)": X_train_fuzzy,
        "Fuzzy RFM-FCA (Redundancy-Suppressed)": X_train_fuzzy_supp,
    }
    df_complexity = compute_complexity_metrics(concepts_by_arm, mu_by_arm)
    print("\n--- Structural Complexity & Redundancy (train set) ---")
    print(df_complexity.to_string(index=False))

    extras = {"n_train": len(train_df), "n_test": n_test}
    return df_metrics, df_ci, df_complexity, extras


def main() -> None:
    print("=" * 80)
    print("ADOPTED IMPROVEMENT: FUZZY RFM-FCA CONCEPT REDUNDANCY SUPPRESSION (RETAIL II)")
    print("=" * 80)

    clean_tx = load_cleaned_transactions()
    ref_date = clean_tx["InvoiceDate"].max()
    print(f"Cleaned transactions: {len(clean_tx):,} | Reference date: {ref_date}")

    df_metrics, df_ci, df_complexity, extras = run_holdout(
        clean_tx,
        cutoff_date=pd.Timestamp("2010-12-09 23:59:59"),
        test_size=0.30,
        n_boot=1000,
        random_seed=42,
    )

    df_metrics.to_csv(OUT_DIR / "temporal_holdout_metrics.csv", index=False)
    df_ci.to_csv(OUT_DIR / "temporal_holdout_bootstrap_ci.csv", index=False)
    df_complexity.to_csv(OUT_DIR / "complexity_redundancy_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "J_MAX (extent-Jaccard redundancy threshold)", "value": J_MAX},
            {"parameter": "mu cut for extents", "value": 0.5},
            {"parameter": "SUPPORT_CUTOFF", "value": SUPPORT_CUTOFF},
            {"parameter": "n_train", "value": extras["n_train"]},
            {"parameter": "n_test", "value": extras["n_test"]},
        ]
    ).to_csv(OUT_DIR / "run_parameters.csv", index=False)

    # ---- Figure ----
    print("\nGenerating figure...")
    plot_arms = [
        "Raw RFM Baseline",
        "Crisp RFM-FCA",
        "Fuzzy RFM-FCA (Baseline)",
        "Fuzzy RFM-FCA (Redundancy-Suppressed)",
    ]
    labels = ["Raw", "Crisp", "Fuzzy (base)", "Fuzzy +\nsuppression"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    for j, (metric, title) in enumerate(
        [
            ("test_repurchase_auc", "Test Repurchase AUC"),
            ("test_spend_r2", "Test Spend R2 (log)"),
            ("test_invoices_r2", "Test Invoices R2 (log)"),
        ]
    ):
        vals = [df_metrics.loc[df_metrics["arm"] == a, metric].values[0] for a in plot_arms]
        ax[j].bar(labels, vals, color=colors, width=0.6, edgecolor="black", alpha=0.85)
        ax[j].set_title(title, fontsize=11, fontweight="bold")
        ax[j].tick_params(axis="x", labelsize=9)
        lo, hi = min(vals), max(vals)
        ax[j].set_ylim(max(0, lo - 0.04 * abs(hi - lo) - 0.01), hi + 0.03 * abs(hi - lo) + 0.01)
        for i, v in enumerate(vals):
            ax[j].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_suppression_holdout.png")
    plt.close()

    print("\nSuppression experiment completed successfully!")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

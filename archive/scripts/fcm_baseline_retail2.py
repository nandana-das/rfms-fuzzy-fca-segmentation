"""Fuzzy C-Means baseline vs FCA structure on Online Retail II (improvement 4).

Question: does the fuzzy FCA *lattice structure* add anything over generic fuzzy
clustering? All current benchmarks compare fuzzy FCA against HARD K-means /
hierarchical, so "fuzzy vs crisp" confounds representation with algorithm. FCM at
matched k isolates the two: same fuzziness (soft memberships, same m=2-style
overlap), different structure (dense prototypes vs formal concepts).

Arms (identical temporal holdout protocol as fair_comparison_retail2.py and
fuzzy_improvements_retail2.py: same split seed, same models, same B=1000 paired
bootstrap on test):
- Crisp RFM-FCA (control, 15-attr crisp context concepts)
- Fuzzy RFM-FCA + redundancy suppression (ADOPTED improvement 1, J_MAX=0.8)
- FCM soft at matched k=30 (memberships as features)
- FCM soft at natural k (knee of WSS curve on train, k=4..12)
- FCM hardened argmax at matched k (one-hot features; isolates fuzziness itself)
- K-means one-hot at matched k=30 (same geometry as FCM, zero fuzziness)

FCM: canonical Bezdek alternate optimization on standardized raw R/F/M, m=2,
k-means++ seeded memberships, 300 iterations, tol 1e-7. Test customers are
assigned by frozen train centroids. No outcomes touch centroid fitting.

Fuzzy validity indices (FCM only): FPC, partition entropy, Xie-Beni; plus
Silhouette / Davies-Bouldin of hardened labels on standardized raw RFM, and a
natural-k scan (k=2..12) for FCM, K-means, and the FCA hardening rule.

Outputs: results/fcm_baseline_retail2/. No Step 1-14 artifacts are modified.
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    brier_score_loss,
    davies_bouldin_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_redundancy import suppress_redundant_concepts
from fair_comparison_retail2 import (
    aggregate_rfm,
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    load_cleaned_transactions,
    mine_crisp_closed_concepts,
    mine_fuzzy_closed_concepts,
)
from fcm import FuzzyCMeans
from project_paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "fcm_baseline_retail2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = ("R", "F", "M")
SUPPORT_CUTOFF = 0.04


# -------------------------------------------------------------------------
# Validity indices
# -------------------------------------------------------------------------
def fuzzy_validity_indices(mu: np.ndarray, X: np.ndarray, centroids: np.ndarray, m: float = 2.0) -> dict[str, float]:
    """FPC, partition entropy, Xie-Beni for a soft partition; silhouette/DB for its argmax.

    Valid only for genuine fuzzy partitions (membership rows sum to 1). FCA concept
    memberships do NOT satisfy that, so callers must not apply this to FCA matrices.
    """
    n = mu.shape[0]

    # Fuzzy Partition Coefficient: sum(mu^2) / n, in [1/k, 1]
    fpc = float((mu**2).sum() / n)

    # Partition entropy: -sum(mu * ln(mu)) / n, in [0, ln(k)]
    pe = float(-(mu * np.log(np.clip(mu, 1e-12, None))).sum() / n)

    # Xie-Beni
    dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    xb_den = float((um := mu**m).sum(axis=0).min() + 1e-12)
    xb = float((um * dist2).sum() / (n * xb_den))

    labels = np.argmax(mu, axis=1)
    k_eff = len(np.unique(labels))
    if 1 < k_eff < n:
        sil = float(
            silhouette_score(X, labels, sample_size=min(5000, n), random_state=42)
        )
        db = float(davies_bouldin_score(X, labels))
    else:
        sil, db = float("nan"), float("nan")
    return {
        "fpc": fpc,
        "partition_entropy": pe,
        "xie_beni": xb,
        "silhouette_hardened": sil,
        "davies_bouldin_hardened": db,
        "n_hard_clusters": int(k_eff),
    }


def _xie_beni(mu: np.ndarray, X: np.ndarray, centroids: np.ndarray, m: float = 2.0) -> float:
    um = mu**m
    dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    n_pairs = centroids.shape[0] * (centroids.shape[0] - 1) / 2
    if n_pairs == 0:
        return float("nan")
    cdist2 = ((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    min_cdist2 = float(cdist2[cdist2 > 0].min())
    return float((um * dist2).sum() / (len(X) * (min_cdist2 + 1e-12)))


def scan_natural_k(X: np.ndarray, max_k: int = 12) -> pd.DataFrame:
    """WSS knee scan for K-means; FPC/PE/Xie-Beni scan for FCM at each k."""
    rows = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        wss = float(km.inertia_)
        fcm = FuzzyCMeans(n_clusters=k, m=2.0, max_iter=300, tol=1e-7, random_state=42).fit(X)
        um = fcm.memberships_ ** 2
        rows.append(
            {
                "k": k,
                "kmeans_wss": wss,
                "fcm_fpc": float(um.sum() / len(X)),
                "fcm_partition_entropy": float(
                    -(fcm.memberships_ * np.log(np.clip(fcm.memberships_, 1e-12, None))).sum() / len(X)
                ),
                "fcm_xie_beni": _xie_beni(fcm.memberships_, X, fcm.centroids_),
            }
        )
    df = pd.DataFrame(rows)
    # Knee = max distance to the WSS line (Kneedle-style)
    w = df["kmeans_wss"].to_numpy()
    x = df["k"].to_numpy(dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    w_n = (w - w.min()) / (w.max() - w.min() + 1e-12)
    d = np.abs((w_n[-1] - w_n[0]) * x_n - (x_n[-1] - x_n[0]) * w_n + x_n[-1] * w_n[0] - w_n[-1] * x_n[0])
    df["kmeans_knee_k"] = int(x[int(np.argmax(d))])
    return df


# -------------------------------------------------------------------------
# Holdout evaluation
# -------------------------------------------------------------------------
def run_holdout(clean_tx: pd.DataFrame, n_boot: int, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cutoff_date = pd.Timestamp("2010-12-09 23:59:59")
    obs_tx = clean_tx[clean_tx["InvoiceDate"] <= cutoff_date].copy()
    hold_tx = clean_tx[clean_tx["InvoiceDate"] > cutoff_date].copy()

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
        merged, test_size=0.30, stratify=merged["repurchased"].to_numpy(), random_state=random_seed
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} (repurchase {train_df['repurchased'].mean():.4f}) | "
          f"Test: {len(test_df)} (repurchase {test_df['repurchased'].mean():.4f})")

    y_train_rep = train_df["repurchased"].to_numpy()
    y_test_rep = test_df["repurchased"].to_numpy()
    y_train_log_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_test_log_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_train_inv = train_df["future_invoices"].to_numpy()
    y_test_inv = test_df["future_invoices"].to_numpy()

    # Standardized raw RFM space (shared by FCM / K-means)
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_train_raw3 = scaler.transform(train_df[["R", "F", "M"]].values)
    X_test_raw3 = scaler.transform(test_df[["R", "F", "M"]].values)

    # ---- Crisp RFM-FCA control ----
    train_scored = dense_rank_scores(train_df[["CustomerID", "R", "F", "M"]])
    crisp_concepts = mine_crisp_closed_concepts(train_scored, min_support=SUPPORT_CUTOFF)
    X_train_crisp = compute_customer_concept_memberships(
        crisp_concepts,
        pd.DataFrame(
            {
                f"{dim}{k}": (train_scored[f"{dim}_score"] == k).astype(float)
                for dim in DIMENSIONS
                for k in range(1, 6)
            }
        ),
    )
    train_cutoffs = {
        "R": extract_dense_rank_cutoffs(train_scored, "R", invert=True),
        "F": extract_dense_rank_cutoffs(train_scored, "F", invert=False),
        "M": extract_dense_rank_cutoffs(train_scored, "M", invert=False),
    }
    test_crisp_bands = {}
    for dim in DIMENSIONS:
        test_scores = apply_dense_rank_cutoffs(test_df[dim].to_numpy(), train_cutoffs[dim], invert=(dim == "R"))
        for k in range(1, 6):
            test_crisp_bands[f"{dim}{k}"] = (test_scores == k).astype(float)
    X_test_crisp = compute_customer_concept_memberships(crisp_concepts, pd.DataFrame(test_crisp_bands))

    # ---- Fuzzy FCA + adopted suppression ----
    train_fmu, train_centroids, _ = baseline_fuzzy_memberships(train_scored)
    fuzzy_concepts, _, _ = mine_fuzzy_closed_concepts(train_fmu, train_scored, min_support=SUPPORT_CUTOFF)
    X_train_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, train_fmu)
    sup_fuzzy_concepts = suppress_redundant_concepts(fuzzy_concepts, X_train_fuzzy)
    X_train_fuzzy_supp = compute_customer_concept_memberships(sup_fuzzy_concepts, train_fmu)
    test_fmu, _, _ = baseline_fuzzy_memberships(test_df[["CustomerID", "R", "F", "M"]], trained_centroids=train_centroids)
    X_test_fuzzy_supp = compute_customer_concept_memberships(sup_fuzzy_concepts, test_fmu)

    # ---- FCM / K-means at matched k and natural k ----
    matched_k = X_train_crisp.shape[1]  # = number of crisp concept features
    print(f"Matched k for FCM/K-means: {matched_k} (crisp concept count)")

    fcm_matched = FuzzyCMeans(n_clusters=matched_k, m=2.0, max_iter=300, tol=1e-7, random_state=42).fit(X_train_raw3)
    scan = scan_natural_k(X_train_raw3, max_k=12)
    natural_k = int(scan.loc[scan["k"] == scan["kmeans_knee_k"], "k"].iloc[0])
    fcm_natural = FuzzyCMeans(n_clusters=natural_k, m=2.0, max_iter=300, tol=1e-7, random_state=42).fit(X_train_raw3)
    km_matched = KMeans(n_clusters=matched_k, n_init=10, random_state=42).fit(X_train_raw3)

    print(f"FCM natural k (WSS knee): {natural_k}")

    mu_tr_m, mu_te_m = fcm_matched.memberships_, fcm_matched.assign(X_test_raw3)
    mu_tr_n, mu_te_n = fcm_natural.memberships_, fcm_natural.assign(X_test_raw3)
    onehot_tr_m = (fcm_matched.assign(X_train_raw3).argmax(axis=1)[:, None] == np.arange(matched_k)).astype(float)
    onehot_te_m = (mu_te_m.argmax(axis=1)[:, None] == np.arange(matched_k)).astype(float)
    km_tr = (km_matched.labels_[:, None] == np.arange(matched_k)).astype(float)
    km_te = (km_matched.predict(X_test_raw3)[:, None] == np.arange(matched_k)).astype(float)

    arms = {
        "Crisp RFM-FCA": (X_train_crisp, X_test_crisp),
        "Fuzzy RFM-FCA (Suppressed, adopted)": (X_train_fuzzy_supp, X_test_fuzzy_supp),
        f"FCM soft (matched k={matched_k})": (mu_tr_m, mu_te_m),
        f"FCM soft (natural k={natural_k})": (mu_tr_n, mu_te_n),
        f"FCM hardened (matched k={matched_k})": (onehot_tr_m, onehot_te_m),
        f"K-means one-hot (matched k={matched_k})": (km_tr, km_te),
    }

    test_preds: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for arm, (X_tr, X_te) in arms.items():
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5, scoring="neg_log_loss",
            solver="lbfgs", max_iter=1000, random_state=random_seed,
        ).fit(X_tr, y_train_rep)
        prob_te = clf.predict_proba(X_te)[:, 1]
        reg_s = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_train_log_sp)
        sp_te = reg_s.predict(X_te)
        reg_i = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, np.log1p(y_train_inv))
        inv_te = reg_i.predict(X_te)
        rows.append(
            {
                "arm": arm,
                "n_features": X_te.shape[1],
                "test_repurchase_auc": float(roc_auc_score(y_test_rep, prob_te)),
                "test_brier_score": float(brier_score_loss(y_test_rep, prob_te)),
                "test_spend_r2": float(r2_score(y_test_log_sp, sp_te)),
                "test_spend_spearman": float(stats.spearmanr(y_test_log_sp, sp_te).statistic),
                "test_invoices_r2": float(r2_score(np.log1p(y_test_inv), inv_te)),
                "test_invoices_spearman": float(stats.spearmanr(y_test_inv, inv_te).statistic),
            }
        )
        test_preds[arm] = {"rep": prob_te, "sp": sp_te, "inv": inv_te}

    df_metrics = pd.DataFrame(rows)
    print("\nStrict Out-of-Sample Holdout Metrics:")
    print(df_metrics.to_string(index=False))

    # ---- Paired bootstrap vs the adopted FCA arm ----
    rng = np.random.default_rng(random_seed)
    n_test = len(test_df)
    ref = test_preds["Fuzzy RFM-FCA (Suppressed, adopted)"]
    boot: dict[str, dict[str, list[float]]] = {
        a: {"auc": [], "sp": [], "inv": []} for a in arms if a != ref_arm_name()
    }
    for _ in range(n_boot):
        idx = rng.choice(n_test, size=n_test, replace=True)
        if len(np.unique(y_test_rep[idx])) < 2:
            continue
        auc_r = roc_auc_score(y_test_rep[idx], ref["rep"][idx])
        r2_r = r2_score(y_test_log_sp[idx], ref["sp"][idx])
        inv_r = r2_score(np.log1p(y_test_inv[idx]), ref["inv"][idx])
        for a in boot:
            p = test_preds[a]
            boot[a]["auc"].append(roc_auc_score(y_test_rep[idx], p["rep"][idx]) - auc_r)
            boot[a]["sp"].append(r2_score(y_test_log_sp[idx], p["sp"][idx]) - r2_r)
            boot[a]["inv"].append(r2_score(np.log1p(y_test_inv[idx]), p["inv"][idx]) - inv_r)

    ci_rows = []
    for a, d in boot.items():
        for key, label in [("auc", "Delta AUC"), ("sp", "Delta Spend R2"), ("inv", "Delta Invoices R2")]:
            arr = np.asarray(d[key])
            ci_rows.append(
                {
                    "comparison": f"{a} vs FCA-suppressed",
                    "metric": label,
                    "ci_lower_95": float(np.percentile(arr, 2.5)),
                    "ci_upper_95": float(np.percentile(arr, 97.5)),
                    "spans_zero": bool(np.percentile(arr, 2.5) <= 0 <= np.percentile(arr, 97.5)),
                    "interval_type": "95% Paired Customer Bootstrap (conditional on fixed split & models)",
                }
            )
    df_ci = pd.DataFrame(ci_rows)
    print("\nPaired Bootstrap CIs (vs adopted FCA-suppressed arm, out-of-sample):")
    print(df_ci.to_string(index=False))

    # ---- Fuzzy validity indices ----
    val_rows = []
    for name, fcm_model in [
        (f"FCM matched k={matched_k}", fcm_matched),
        (f"FCM natural k={natural_k}", fcm_natural),
    ]:
        v = fuzzy_validity_indices(fcm_model.memberships_, X_train_raw3, fcm_model.centroids_)
        val_rows.append({"model": name, **v})

    fca_hard_labels_tr = X_train_fuzzy_supp.argmax(axis=1)
    k_eff = len(np.unique(fca_hard_labels_tr))
    if 1 < k_eff < len(fca_hard_labels_tr):
        fca_sil = float(silhouette_score(X_train_raw3, fca_hard_labels_tr, sample_size=min(5000, len(X_train_raw3)), random_state=42))
        fca_db = float(davies_bouldin_score(X_train_raw3, fca_hard_labels_tr))
    else:
        fca_sil, fca_db = float("nan"), float("nan")
    val_rows.append(
        {
            "model": "FCA-suppressed (hardened argmax)",
            # FPC / partition entropy are undefined here: FCA concept memberships are
            # per-concept compatibilities and do not sum to 1 per customer, so they do
            # not form a fuzzy partition. Reported as NaN rather than a meaningless value.
            "fpc": float("nan"),
            "partition_entropy": float("nan"),
            "xie_beni": float("nan"),  # not defined for prototype-free concepts
            "silhouette_hardened": fca_sil,
            "davies_bouldin_hardened": fca_db,
            "n_hard_clusters": int(k_eff),
        }
    )
    df_validity = pd.DataFrame(val_rows)
    print("\n--- Fuzzy Validity Indices (train set) ---")
    print(df_validity.to_string(index=False))

    extras = {
        "natural_k_scan": scan,
        "validity": df_validity,
        "matched_k": matched_k,
        "natural_k": natural_k,
        "n_train": len(train_df),
        "n_test": n_test,
    }
    return df_metrics, df_ci, extras


def ref_arm_name() -> str:
    return "Fuzzy RFM-FCA (Suppressed, adopted)"


def baseline_fuzzy_memberships(scored: pd.DataFrame, trained_centroids=None):
    """Thin wrapper reusing the project-standard band-median fuzzy memberships."""
    from fair_comparison_retail2 import compute_fuzzy_memberships

    return compute_fuzzy_memberships(scored, trained_centroids=trained_centroids)


def main() -> None:
    print("=" * 80)
    print("IMPROVEMENT 4: FUZZY C-MEANS BASELINE vs FCA STRUCTURE (ONLINE RETAIL II)")
    print("=" * 80)

    clean_tx = load_cleaned_transactions()
    print(f"Cleaned transactions: {len(clean_tx):,}")

    df_metrics, df_ci, extras = run_holdout(clean_tx, n_boot=1000, random_seed=42)

    df_metrics.to_csv(OUT_DIR / "temporal_holdout_metrics.csv", index=False)
    df_ci.to_csv(OUT_DIR / "paired_bootstrap_ci.csv", index=False)
    extras["validity"].to_csv(OUT_DIR / "fuzzy_validity_indices.csv", index=False)
    extras["natural_k_scan"].to_csv(OUT_DIR / "natural_k_scan.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "FCM fuzzifier m", "value": 2.0},
            {"parameter": "matched_k (crisp concept count)", "value": extras["matched_k"]},
            {"parameter": "FCM natural k (WSS knee)", "value": extras["natural_k"]},
            {"parameter": "n_train", "value": extras["n_train"]},
            {"parameter": "n_test", "value": extras["n_test"]},
        ]
    ).to_csv(OUT_DIR / "run_parameters.csv", index=False)

    # ---- Figure ----
    fig, ax = plt.subplots(1, 3, figsize=(19, 5), dpi=300)
    arms = df_metrics["arm"].tolist()
    short = ["Crisp FCA", "FCA-suppr (adopted)", "FCM soft\n(k=30)", "FCM soft\n(nat k)", "FCM hard\n(k=30)", "K-means\none-hot"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#17becf", "#7f7f7f"]
    for j, (metric, title) in enumerate(
        [
            ("test_repurchase_auc", "Test Repurchase AUC"),
            ("test_spend_r2", "Test Spend R2 (log)"),
            ("test_invoices_r2", "Test Invoices R2 (log)"),
        ]
    ):
        vals = df_metrics[metric].to_numpy()
        ax[j].bar(short, vals, color=colors, width=0.6, edgecolor="black", alpha=0.85)
        ax[j].set_title(title, fontsize=11, fontweight="bold")
        ax[j].tick_params(axis="x", labelsize=8)
        lo, hi = float(min(vals)), float(max(vals))
        ax[j].set_ylim(max(0, lo - 0.04 * abs(hi - lo) - 0.01), hi + 0.03 * abs(hi - lo) + 0.01)
        for i, v in enumerate(vals):
            ax[j].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_fcm_baseline_holdout.png")
    plt.close()

    print("\nFCM baseline run completed successfully!")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

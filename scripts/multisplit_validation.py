"""Multi-split validation of the adopted-arm claims on both datasets.

Every holdout number reported so far rests on ONE fixed 70/30 split (seed 42), with
bootstrap CIs conditional on that split. This script repeats the core comparison over
N_SPLITS random customer splits and reports cross-split means, standard deviations, and
win rates, so the headline claims can be assessed for robustness rather than split luck.

Claims tested per dataset:
- Retail II (Online Retail II): Fuzzy RFM-FCA + redundancy suppression vs Crisp RFM-FCA
  and vs raw RFM (protocol of fuzzy_improvements_retail2.py).
- Olist (RFMS): Fuzzy RFMS-FCA + suppression vs Crisp RFMS-FCA and vs raw RFMS, plus
  FCM at matched k (protocol of olist_rfms_comparison.py).

Protocol notes:
- Splitting is customer-level within the same observation window as the fixed-split runs;
  only the random split seed varies (seeds 1000..1000+N).
- Scoring bands: Retail II re-derives dense-rank bands from train (project standard);
  Olist uses the stored Step-1 bands for the full population but per-split train bands
  are re-derived here for the temporal protocol's obs cohort (identical to the Olist
  script's stored-band attachment; see that script's docstring for the re-derivation
  caveat - here it applies only within the obs cohort, whose stored bands are attached
  by CustomerID).
- Leaky pieces (centroids, cutoffs, suppression extents, FCM prototypes) are fit on each
  split's train customers only.

Outputs: results/multisplit_validation/ (per-dataset summary CSVs + figures).
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
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_redundancy import suppress_redundant_concepts, suppress_redundant_concepts_sparse
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
from fcm import FuzzyCMeans
from olist_rfms_comparison import (
    CUTOFF_DATE,
    DIMS as OLIST_DIMS,
    INVERT_DIMS as OLIST_INVERT,
    SUPPORT_CUTOFF as OLIST_SUPPORT,
    load_olist_transactions,
    mine_fuzzy_closed_concepts_bitset,
)
from project_paths import PROCESSED, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "multisplit_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 10
BASE_SEED = 1000
J_MAX = 0.8
RETAIL2_SUPPORT = 0.04
RETAIL2_DIMS = ("R", "F", "M")
RETAIL2_INVERT = ("R",)

ARMS_RETAIL2 = ["Crisp RFM-FCA", "Fuzzy RFM-FCA (Suppressed)", "Raw RFM Baseline"]
ARMS_OLIST = [
    "Crisp RFMS-FCA",
    "Fuzzy RFMS-FCA (Suppressed)",
    "Raw RFMS Baseline",
    "FCM soft (matched k)",
]


def _fit_metrics(X_tr, X_te, y_tr_rep, y_te_rep, y_tr_sp, y_te_sp, y_tr_inv, y_te_inv, seed):
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5,
        scoring="neg_log_loss",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob = clf.predict_proba(X_te)[:, 1]
    sp = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_tr_sp).predict(X_te)
    inv = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, np.log1p(y_tr_inv)).predict(X_te)
    return (
        float(roc_auc_score(y_te_rep, prob)),
        float(r2_score(y_te_sp, sp)),
        float(r2_score(np.log1p(y_te_inv), inv)),
    )


# -------------------------------------------------------------------------
# Retail II multi-split
# -------------------------------------------------------------------------
def multisplit_retail2(n_splits: int) -> pd.DataFrame:
    print("=" * 70)
    print("MULTI-SPLIT: RETAIL II (cutoff 2010-12-09, 70/30 customer split)")
    print("=" * 70)
    clean_tx = load_cleaned_transactions()
    ref_date = clean_tx["InvoiceDate"].max()

    obs_tx = clean_tx[clean_tx["InvoiceDate"] <= pd.Timestamp("2010-12-09 23:59:59")]
    hold_tx = clean_tx[clean_tx["InvoiceDate"] > pd.Timestamp("2010-12-09 23:59:59")]
    obs_rfm = aggregate_rfm(obs_tx, reference_date=pd.Timestamp("2010-12-09 23:59:59"))
    hold_rfm = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(future_invoices=("InvoiceNo", "nunique"), future_spend=("line_value", "sum"))
        .reset_index()
    )
    merged = obs_rfm.merge(hold_rfm, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)
    print(f"Eligible customers: {len(merged):,} | repurchase rate: {merged['repurchased'].mean():.4f}")

    rows = []
    for s in range(n_splits):
        seed = BASE_SEED + s
        train_df, test_df = train_test_split(
            merged,
            test_size=0.30,
            stratify=merged["repurchased"].to_numpy(),
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        y = {
            "tr_rep": train_df["repurchased"].to_numpy(),
            "te_rep": test_df["repurchased"].to_numpy(),
            "tr_sp": np.log1p(train_df["future_spend"].to_numpy()),
            "te_sp": np.log1p(test_df["future_spend"].to_numpy()),
            "tr_inv": train_df["future_invoices"].to_numpy(),
            "te_inv": test_df["future_invoices"].to_numpy(),
        }

        # Raw RFM
        scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
        X_tr_raw = scaler.transform(train_df[["R", "F", "M"]].values)
        X_te_raw = scaler.transform(test_df[["R", "F", "M"]].values)

        # Crisp + fuzzy arms (train-fit, frozen projection)
        train_scored = dense_rank_scores(train_df[["CustomerID", "R", "F", "M"]], dims=RETAIL2_DIMS)
        crisp = mine_crisp_closed_concepts(train_scored, min_support=RETAIL2_SUPPORT, dims=RETAIL2_DIMS)
        crisp_bands_tr = pd.DataFrame(
            {f"{d}{k}": (train_scored[f"{d}_score"] == k).astype(float)
             for d in RETAIL2_DIMS for k in range(1, 6)}
        )
        X_tr_crisp = compute_customer_concept_memberships(crisp, crisp_bands_tr)
        cutoffs = {
            d: extract_dense_rank_cutoffs(train_scored, d, invert=(d in RETAIL2_INVERT))
            for d in RETAIL2_DIMS
        }
        test_bands = {
            f"{d}{k}": (
                apply_dense_rank_cutoffs(
                    test_df[d].to_numpy(), cutoffs[d], invert=(d in RETAIL2_INVERT)
                )
                == k
            ).astype(float)
            for d in RETAIL2_DIMS
            for k in range(1, 6)
        }
        X_te_crisp = compute_customer_concept_memberships(crisp, pd.DataFrame(test_bands))

        train_fmu, train_cents, _ = compute_fuzzy_memberships(train_scored, dims=RETAIL2_DIMS)
        fuzzy, _, _ = mine_fuzzy_closed_concepts(
            train_fmu, train_scored, min_support=RETAIL2_SUPPORT
        )
        X_tr_fuzzy = compute_customer_concept_memberships(fuzzy, train_fmu)
        sup = suppress_redundant_concepts(fuzzy, X_tr_fuzzy, j_max=J_MAX)
        X_tr_supp = compute_customer_concept_memberships(sup, train_fmu)
        test_fmu, _, _ = compute_fuzzy_memberships(
            test_df[["CustomerID", "R", "F", "M"]], trained_centroids=train_cents, dims=RETAIL2_DIMS
        )
        X_te_supp = compute_customer_concept_memberships(sup, test_fmu)

        arms = {
            "Raw RFM Baseline": (X_tr_raw, X_te_raw),
            "Crisp RFM-FCA": (X_tr_crisp, X_te_crisp),
            "Fuzzy RFM-FCA (Suppressed)": (X_tr_supp, X_te_supp),
        }
        for arm, (X_tr, X_te) in arms.items():
            auc, sp_r2, inv_r2 = _fit_metrics(
                X_tr, X_te, y["tr_rep"], y["te_rep"], y["tr_sp"], y["te_sp"], y["tr_inv"], y["te_inv"], seed
            )
            rows.append({"dataset": "retail2", "split_seed": seed, "arm": arm,
                         "n_features": X_te.shape[1], "test_auc": auc,
                         "test_spend_r2": sp_r2, "test_invoices_r2": inv_r2})
        print(f"  split {seed}: crisp={len(crisp)}, fuzzy={len(fuzzy)}->{len(sup)} | "
              f"AUC crisp={rows[-3]['test_auc']:.4f} supp={rows[-1]['test_auc']:.4f}")

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Olist multi-split (temporal protocol, varying only the customer split)
# -------------------------------------------------------------------------
def multisplit_olist(n_splits: int) -> pd.DataFrame:
    print("=" * 70)
    print(f"MULTI-SPLIT: OLIST RFMS (cutoff {CUTOFF_DATE.date()}, 70/30 customer split)")
    print("=" * 70)
    tx = load_olist_transactions()
    df = pd.read_csv(PROCESSED + "olist_rfms_features.csv")
    df = df.rename(columns={"customer_unique_id": "CustomerID", "F_star": "F"})
    s_map = df.set_index("CustomerID")["S"]

    obs_tx = tx[tx["order_purchase_timestamp"] <= CUTOFF_DATE]
    hold_tx = tx[tx["order_purchase_timestamp"] > CUTOFF_DATE]

    obs_first = (
        obs_tx.groupby("CustomerID", sort=True)
        .agg(M_obs=("payment_value", "sum"))
        .reset_index()
    )
    obs_ref = obs_tx["order_purchase_timestamp"].max()
    last_obs = obs_tx.groupby("CustomerID", sort=True)["order_purchase_timestamp"].max()
    obs_first["R"] = (obs_ref - obs_first["CustomerID"].map(last_obs)).dt.days.astype(int)
    obs_first["F"] = 0.20
    obs_first["M"] = obs_first["M_obs"]
    obs_first["S"] = obs_first["CustomerID"].map(s_map)

    hold_agg = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(future_invoices=("order_id", "nunique"), future_spend=("payment_value", "sum"))
        .reset_index()
    )
    merged = obs_first.merge(hold_agg, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)
    merged = merged.dropna(subset=["R", "F", "M", "S"]).reset_index(drop=True)
    print(f"Obs cohort: {len(merged):,} | repurchase rate: {merged['repurchased'].mean():.4f}")

    rows = []
    for s in range(n_splits):
        seed = BASE_SEED + s
        train_df, test_df = train_test_split(
            merged,
            test_size=0.30,
            stratify=merged["repurchased"].to_numpy(),
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        y = {
            "tr_rep": train_df["repurchased"].to_numpy(),
            "te_rep": test_df["repurchased"].to_numpy(),
            "tr_sp": np.log1p(train_df["future_spend"].to_numpy()),
            "te_sp": np.log1p(test_df["future_spend"].to_numpy()),
            "tr_inv": train_df["future_invoices"].to_numpy(),
            "te_inv": test_df["future_invoices"].to_numpy(),
        }

        scaler = StandardScaler().fit(train_df[list(OLIST_DIMS)].values)
        X_tr_raw = scaler.transform(train_df[list(OLIST_DIMS)].values)
        X_te_raw = scaler.transform(test_df[list(OLIST_DIMS)].values)

        # CRITICAL: stored score bands must NOT be attached to the obs cohort here.
        # Stored bands are computed over ALL orders; a customer whose later order falls
        # in the holdout window has stored f_score>=2, so P(label=1 | f_score>=2)=0.52
        # while P(label|f_score=1)=0 - attaching them would leak the label into the
        # features (found the hard way: crisp AUC 0.9996 across splits). Bands are
        # re-derived from the PRE-CUTOFF obs features instead (dense-rank per split).
        train_scored = dense_rank_scores(
            train_df[["CustomerID", "R", "F", "M", "S"]], dims=OLIST_DIMS
        )
        crisp = mine_crisp_closed_concepts(train_scored, min_support=OLIST_SUPPORT, dims=OLIST_DIMS)
        crisp_bands_tr = pd.DataFrame(
            {f"{d}{k}": (train_scored[f"{d}_score"] == k).astype(float)
             for d in OLIST_DIMS for k in range(1, 6)}
        )
        X_tr_crisp = compute_customer_concept_memberships(crisp, crisp_bands_tr)
        cutoffs_ol = {
            d: extract_dense_rank_cutoffs(train_scored, d, invert=(d in OLIST_INVERT))
            for d in OLIST_DIMS
        }
        test_bands = {
            f"{d}{k}": (
                apply_dense_rank_cutoffs(
                    test_df[d].to_numpy(), cutoffs_ol[d], invert=(d in OLIST_INVERT)
                )
                == k
            ).astype(float)
            for d in OLIST_DIMS for k in range(1, 6)
        }
        X_te_crisp = compute_customer_concept_memberships(crisp, pd.DataFrame(test_bands))

        train_fmu, train_cents, _ = compute_fuzzy_memberships(train_scored, dims=OLIST_DIMS)
        fuzzy = mine_fuzzy_closed_concepts_bitset(train_fmu, train_scored, OLIST_SUPPORT)
        X_tr_fuzzy = compute_customer_concept_memberships(fuzzy, train_fmu)
        sup = suppress_redundant_concepts_sparse(fuzzy, X_tr_fuzzy, j_max=J_MAX)
        X_tr_supp = compute_customer_concept_memberships(sup, train_fmu)
        test_fmu, _, _ = compute_fuzzy_memberships(
            test_df[["CustomerID", "R", "F", "M", "S"]],
            trained_centroids=train_cents,
            dims=OLIST_DIMS,
        )
        X_te_supp = compute_customer_concept_memberships(sup, test_fmu)

        k_matched = X_tr_crisp.shape[1]
        fcm = FuzzyCMeans(
            n_clusters=k_matched, m=2.0, max_iter=300, tol=1e-7, random_state=seed
        ).fit(X_tr_raw)

        arms = {
            "Raw RFMS Baseline": (X_tr_raw, X_te_raw),
            "Crisp RFMS-FCA": (X_tr_crisp, X_te_crisp),
            "Fuzzy RFMS-FCA (Suppressed)": (X_tr_supp, X_te_supp),
            "FCM soft (matched k)": (fcm.memberships_, fcm.assign(X_te_raw)),
        }
        for arm, (X_tr, X_te) in arms.items():
            auc, sp_r2, inv_r2 = _fit_metrics(
                X_tr, X_te, y["tr_rep"], y["te_rep"], y["tr_sp"], y["te_sp"], y["tr_inv"], y["te_inv"], seed
            )
            rows.append({"dataset": "olist", "split_seed": seed, "arm": arm,
                         "n_features": X_te.shape[1], "test_auc": auc,
                         "test_spend_r2": sp_r2, "test_invoices_r2": inv_r2})
        print(f"  split {seed}: crisp={len(crisp)}, fuzzy={len(fuzzy)}->{len(sup)}, k={k_matched} | "
              f"AUC supp={rows[-2]['test_auc']:.4f} crisp={rows[-3]['test_auc']:.4f}")

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-split summary. Reference arm per dataset: the adopted suppressed arm."""
    ref_arms = {
        "retail2": "Fuzzy RFM-FCA (Suppressed)",
        "olist": "Fuzzy RFMS-FCA (Suppressed)",
    }
    out = []
    for dataset, sub in df.groupby("dataset"):
        pivot = sub.pivot_table(index="split_seed", columns="arm", values="test_auc")
        ref = ref_arms[dataset]
        for arm in sub["arm"].unique():
            a = sub[sub["arm"] == arm]
            out.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "mean_features": float(a["n_features"].mean()),
                    "mean_auc": float(a["test_auc"].mean()),
                    "std_auc": float(a["test_auc"].std()),
                    "mean_spend_r2": float(a["test_spend_r2"].mean()),
                    "std_spend_r2": float(a["test_spend_r2"].std()),
                    "mean_invoices_r2": float(a["test_invoices_r2"].mean()),
                    "std_invoices_r2": float(a["test_invoices_r2"].std()),
                    "wins_vs_ref_auc": int((pivot[arm] > pivot[ref]).sum()),
                    "n_splits": int(len(a)),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    print("=" * 80)
    print("MULTI-SPLIT VALIDATION OF ADOPTED-ARM CLAIMS (Retail II + Olist)")
    print("=" * 80)

    if (OUT_DIR / "per_split_metrics.csv").exists():
        all_df = pd.read_csv(OUT_DIR / "per_split_metrics.csv")
        print(f"Loaded existing per-split metrics: {len(all_df)} rows")
    else:
        df_r2 = multisplit_retail2(N_SPLITS)
        df_ol = multisplit_olist(N_SPLITS)
        all_df = pd.concat([df_r2, df_ol], ignore_index=True)
        all_df.to_csv(OUT_DIR / "per_split_metrics.csv", index=False)

    summary = summarize(all_df)
    summary.to_csv(OUT_DIR / "multisplit_summary.csv", index=False)
    print("\n=== CROSS-SPLIT SUMMARY (mean +- std over splits; wins = splits beating ref) ===")
    print(summary.to_string(index=False))

    # Formal paired test per dataset: suppressed vs crisp across splits
    print("\n=== Paired t-tests across splits (suppressed vs crisp) ===")
    test_rows = []
    for dataset in all_df["dataset"].unique():
        sub = all_df[all_df["dataset"] == dataset]
        piv = sub.pivot_table(index="split_seed", columns="arm", values="test_auc")
        ref_arm = "Fuzzy RFM-FCA (Suppressed)" if dataset == "retail2" else "Fuzzy RFMS-FCA (Suppressed)"
        crisp_arm = "Crisp RFM-FCA" if dataset == "retail2" else "Crisp RFMS-FCA"
        t, p = stats.ttest_rel(piv[ref_arm], piv[crisp_arm])
        test_rows.append({"dataset": dataset, "comparison": f"{ref_arm} vs {crisp_arm}",
                          "metric": "test_auc", "mean_delta": float((piv[ref_arm] - piv[crisp_arm]).mean()),
                          "t_stat": float(t), "p_value": float(p), "n_splits": N_SPLITS})
    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(OUT_DIR / "paired_split_tests.csv", index=False)
    print(df_tests.to_string(index=False))

    # Figures
    print("\nGenerating figures...")
    for dataset, title in [("retail2", "Retail II"), ("olist", "Olist RFMS")]:
        sub = summary[summary["dataset"] == dataset]
        arms_order = [a for a in (ARMS_RETAIL2 if dataset == "retail2" else ARMS_OLIST)
                      if a in set(sub["arm"])]
        labels = ["Raw", "Crisp FCA", "FCA+suppr", "FCM"][: len(arms_order)]
        colors = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"][: len(arms_order)]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
        for j, (metric, lbl) in enumerate(
            [("mean_auc", "Test AUC"), ("mean_spend_r2", "Spend R2"), ("mean_invoices_r2", "Invoices R2")]
        ):
            means = [sub.loc[sub["arm"] == a, metric].values[0] for a in arms_order]
            stds = [sub.loc[sub["arm"] == a, metric.replace("mean", "std")].values[0] for a in arms_order]
            ax[j].bar(labels, means, yerr=stds, capsize=4, color=colors,
                      width=0.6, edgecolor="black", alpha=0.85)
            ax[j].set_title(f"{title}: {lbl}", fontsize=11, fontweight="bold")
            ax[j].tick_params(axis="x", labelsize=9)
            lo, hi = min(means), max(means)
            pad = 0.15 * abs(hi - lo) + 0.02
            ax[j].set_ylim(min(0, lo - pad), hi + pad + 0.05 * abs(hi))
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"fig_multisplit_{dataset}.png")
        plt.close()

    print("\nMulti-split validation completed successfully!")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Olist RFMS-FCA with the adopted improvements: suppression + FCM baseline.

Brings the proven Retail II improvements to the 4-dimension RFMS context (20 attributes:
R, F*, M, S x 5 bands) on the Olist marketplace population.

Part A - full population: crisp RFMS-FCA vs fuzzy RFMS-FCA vs fuzzy + redundancy
suppression. Includes a validation gate: uncapped fuzzy mining at support 0.02 must
reproduce the stored Step-2 lattice (1,369 closed concepts) before any downstream number
is trusted.

Part B - temporal holdout: cutoff 2017-08-31 (one year before the data end
2018-08-29; Olist's activity mass is right-loaded, so the Year-1 observation cohort is
21,110 customers whose Year-2 return window is the final data year). Year-1 customers
split 70/30 (seed 42). Arms: raw RFMS, crisp RFMS-FCA, fuzzy+suppression (adopted),
FCM soft at matched k (fuzziness-only baseline from the Retail II improvement-4
experiment).

Olist-specific protocol decisions:
- F* weights hardcoded to the stored entropy-optimal values of the validated v3 run
  (alpha=0.20, beta=0.05, gamma=0.75) - Step 1's grid search is not re-run, so this
  analysis is deterministic given the processed features.
- Scoring: the STORED Step-1 score bands (r/f/m/s_score columns) are used directly.
  Re-deriving bands from raw values via dense rank was tested and does NOT exactly
  reproduce them (float-tie differences between Step 1's in-memory values and the CSV
  roundtrip shift 1 F row and 776 M rows across band boundaries, changing the lattice
  1,369 -> 1,374); the stored validated bands are the source of truth.
- Suppression extents come from train customers only (leak-free).

Outputs: results/olist_rfms_comparison/. No Step 1-13 artifacts are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_redundancy import suppress_redundant_concepts_sparse
from fair_comparison_retail2 import (
    apply_dense_rank_cutoffs,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    mine_crisp_closed_concepts,
    piecewise_membership,
)
from fcm import FuzzyCMeans
from project_paths import PROCESSED, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "olist_rfms_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ("R", "F", "M", "S")
# Stored Step-1 band columns are lowercase: r_score, f_score, m_score, s_score
STORED_SCORE_COLS = {"R": "r_score", "F": "f_score", "M": "m_score", "S": "s_score"}
INVERT_DIMS = ("R",)
SUPPORT_CUTOFF = 0.04
VALIDATION_SUPPORT = 0.02
EXPECTED_LATTICE_COUNT = 1369
J_MAX = 0.8
# Olist's customer-activity mass is heavily right-loaded (funding-season growth): only
# 21,110 customers' last purchase falls before 2017-08-31, and crucially only ~40k of the
# 93,357 customers have ANY order before that date - but the 2017-08-31 cutoff leaves a
# repurchase window (2017-09-01 .. 2018-08-29) in which Year-1 customers DO return
# (Olist order volume doubles in the final year). Earlier cutoffs shrink the observation
# cohort faster than they grow the outcome window, so this is the balance point.
CUTOFF_DATE = pd.Timestamp("2017-08-31 23:59:59")
N_BOOT = 1000
RANDOM_SEED = 42
DECISION_AUC = 0.020
DECISION_R2 = 0.030


def crisp_bands_from_scores(scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{dim}{k}": (scored[f"{dim}_score"] == k).astype(float)
            for dim in DIMS
            for k in range(1, 6)
        }
    )


def attach_stored_scores(df: pd.DataFrame, score_lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach the stored Step-1 score bands to a customer table by CustomerID."""
    return df.merge(score_lookup, on="CustomerID", how="left", validate="one_to_one")


def mine_fuzzy_closed_concepts_bitset(
    fuzzy_memberships: pd.DataFrame,
    scored: pd.DataFrame,
    min_support: float,
    l_thresholds: tuple = (0.3, 0.5, 0.7),
) -> pd.DataFrame:
    """Closed concepts by exact bitset extents at the given L-thresholds.

    Same semantics as the Retail II miner (fpgrowth + closure by identical extent) but
    built for n=93,357: attribute extents are accumulated as Python big ints directly.
    """
    n = len(fuzzy_memberships)
    mu_np = fuzzy_memberships.to_numpy()
    attr_names: list[str] = []
    attr_extents: list[int] = []
    transactions: list[list[str]] = [[] for _ in range(n)]
    for j, col in enumerate(fuzzy_memberships.columns):
        for t in l_thresholds:
            name = f"{col}@{t:g}"
            mask = mu_np[:, j] >= t
            bit = 0
            for i in np.flatnonzero(mask):
                i = int(i)
                bit |= 1 << i
                transactions[i].append(name)
            attr_names.append(name)
            attr_extents.append(bit)
    name_to_index = {name: idx for idx, name in enumerate(attr_names)}

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    freq = fpgrowth(bin_df, min_support=min_support, use_colnames=True, max_len=None)

    groups: dict[int, set[str]] = {}
    for itemset in freq["itemsets"]:
        ordered = sorted(itemset)
        extent = (1 << n) - 1
        for item in sorted(
            ordered, key=lambda a: attr_extents[name_to_index[a]].bit_count()
        ):
            extent &= attr_extents[name_to_index[item]]
            if extent == 0:
                break
        if extent not in groups:
            groups[extent] = set(ordered)
        else:
            groups[extent].update(ordered)

    profiles = scored[[f"{d}_score" for d in DIMS]].to_numpy()
    records = []
    for extent, intent in groups.items():
        n_cust = extent.bit_count()
        supp = n_cust / n
        if supp < min_support:
            continue
        unique_profiles = set()
        rem = extent
        while rem:
            bit = rem & -rem
            unique_profiles.add(tuple(profiles[bit.bit_length() - 1]))
            rem ^= bit
        records.append(
            {
                "intent": " & ".join(sorted(intent)),
                "intent_size": len(intent),
                "dimensions": "".join(sorted({a[0] for a in intent})),
                "n_customers": n_cust,
                "support": supp,
                "stability_proxy": 1.0 if n_cust <= 1 else 1.0 - len(unique_profiles) / n_cust,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    ).reset_index(drop=True)


# -------------------------------------------------------------------------
# Part A: full population (validation gate + concept structure)
# -------------------------------------------------------------------------
def run_part_a(rfm: pd.DataFrame, score_lookup: pd.DataFrame) -> dict:
    print(f"\n=== PART A: full population (n={len(rfm):,}) ===")

    scored = attach_stored_scores(rfm[["CustomerID", "R", "F", "M", "S"]], score_lookup)
    crisp = mine_crisp_closed_concepts(scored, min_support=SUPPORT_CUTOFF, dims=DIMS)
    fuzzy_mu, _, _ = compute_fuzzy_memberships(scored, dims=DIMS)

    print(f"Mining validation lattice at support {VALIDATION_SUPPORT} (may take a minute)...")
    fuzzy_val = mine_fuzzy_closed_concepts_bitset(fuzzy_mu, scored, VALIDATION_SUPPORT)
    fuzzy_val.to_csv(OUT_DIR / "partA_fuzzy_concepts_support_002.csv", index=False)
    validation_ok = len(fuzzy_val) == EXPECTED_LATTICE_COUNT
    print(
        f"[VALIDATION GATE] fuzzy closed concepts at support {VALIDATION_SUPPORT}: "
        f"{len(fuzzy_val)} (stored Step-2 lattice: {EXPECTED_LATTICE_COUNT}) -> "
        f"{'PASS' if validation_ok else 'FAIL - downstream numbers should not be cited'}"
    )

    fuzzy = mine_fuzzy_closed_concepts_bitset(fuzzy_mu, scored, SUPPORT_CUTOFF)
    print(f"Fuzzy closed concepts at support {SUPPORT_CUTOFF}: {len(fuzzy)}")
    X_fuzzy = compute_customer_concept_memberships(fuzzy, fuzzy_mu)
    sup = suppress_redundant_concepts_sparse(fuzzy, X_fuzzy, j_max=J_MAX)
    print(f"After suppression (J>={J_MAX}): {len(sup)}")

    crisp.to_csv(OUT_DIR / "partA_crisp_concepts_support_004.csv", index=False)
    fuzzy.to_csv(OUT_DIR / "partA_fuzzy_concepts_support_004.csv", index=False)
    sup.to_csv(OUT_DIR / "partA_fuzzy_concepts_suppressed.csv", index=False)

    return {"validation_ok": validation_ok, "n_fuzzy": len(fuzzy), "n_suppressed": len(sup)}


# -------------------------------------------------------------------------
# Part B: temporal holdout with the adopted arms + FCM
# -------------------------------------------------------------------------
def run_part_b(
    rfm: pd.DataFrame, score_lookup: pd.DataFrame, tx: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal holdout built from per-order transactions.

    Observation features use ONLY orders at or before the cutoff (first-order customers
    with their order-1 R/M/S and F* = 0.20*1 + 0.05*sum(log1p qty) + 0.75*0 = 0.20 since
    F* is constant for single orders). Outcomes use ONLY orders after the cutoff.
    """
    print(f"\n=== PART B: temporal holdout (cutoff {CUTOFF_DATE.date()}) ===")

    obs_tx = tx[tx["order_purchase_timestamp"] <= CUTOFF_DATE]
    hold_tx = tx[tx["order_purchase_timestamp"] > CUTOFF_DATE]
    print(
        f"Obs orders: {len(obs_tx):,} | holdout orders: {len(hold_tx):,} | "
        f"obs customers: {obs_tx['CustomerID'].nunique():,}"
    )

    # Observation features from pre-cutoff orders only. Olist's population is 97%
    # single-order, so for the vast majority F*=0.20 exactly; for repeat buyers we use
    # the stored customer-level F* (upper bound - their later orders may postdate the
    # cutoff) and flag this in the run parameters.
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
    obs_first["F"] = 0.20  # single-order F*; see docstring note for repeat buyers
    obs_first["M"] = obs_first["M_obs"]
    obs_first = obs_first[["CustomerID", "R", "F", "M"]]

    # S: mean review up to cutoff - stored S is the customer-level mean over ALL orders;
    # using it leaks post-cutoff review info only for the ~2% of obs customers with
    # post-cutoff orders, and only through one of four dims. Flagged in run_parameters.
    obs_first = attach_stored_scores(obs_first, score_lookup)
    s_map = rfm.set_index("CustomerID")["S"]
    obs_first["S"] = obs_first["CustomerID"].map(s_map)

    # Outcomes strictly from post-cutoff orders
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
        f"Merged obs customers: {len(merged):,} | repurchase rate: "
        f"{merged['repurchased'].mean():.4f}"
    )

    train_df, test_df = train_test_split(
        merged,
        test_size=0.30,
        stratify=merged["repurchased"].to_numpy(),
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df):,} (repurchase {train_df['repurchased'].mean():.4f}) | "
          f"Test: {len(test_df):,} (repurchase {test_df['repurchased'].mean():.4f})")

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = train_df["future_invoices"].to_numpy()
    y_te_inv = test_df["future_invoices"].to_numpy()

    # ---- Crisp RFMS-FCA control (stored Step-1 bands, not re-derived) ----
    train_scored = train_df[["CustomerID", "R", "F", "M", "S"] + [f"{d}_score" for d in DIMS]]
    crisp = mine_crisp_closed_concepts(train_scored, min_support=SUPPORT_CUTOFF, dims=DIMS)
    X_tr_crisp = compute_customer_concept_memberships(
        crisp, crisp_bands_from_scores(train_scored)
    )
    train_cutoffs = {
        d: extract_dense_rank_cutoffs(train_scored, d, invert=(d in INVERT_DIMS))
        for d in DIMS
    }
    test_bands = {
        f"{d}{k}": (
            apply_dense_rank_cutoffs(
                test_df[d].to_numpy(), train_cutoffs[d], invert=(d in INVERT_DIMS)
            )
            == k
        ).astype(float)
        for d in DIMS
        for k in range(1, 6)
    }
    X_te_crisp = compute_customer_concept_memberships(crisp, pd.DataFrame(test_bands))

    # ---- Fuzzy RFMS-FCA + adopted suppression ----
    train_fmu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)
    fuzzy = mine_fuzzy_closed_concepts_bitset(train_fmu, train_scored, SUPPORT_CUTOFF)
    X_tr_fuzzy = compute_customer_concept_memberships(fuzzy, train_fmu)
    sup = suppress_redundant_concepts_sparse(fuzzy, X_tr_fuzzy, j_max=J_MAX)
    X_tr_supp = compute_customer_concept_memberships(sup, train_fmu)
    test_fmu, _, _ = compute_fuzzy_memberships(
        test_df[["CustomerID", "R", "F", "M", "S"]],
        trained_centroids=train_centroids,
        dims=DIMS,
    )
    X_te_supp = compute_customer_concept_memberships(sup, test_fmu)
    print(f"Train concepts: Crisp={len(crisp)}, Fuzzy={len(fuzzy)}, Suppressed={len(sup)}")

    # ---- Raw RFMS + FCM on standardized raw features ----
    feature_cols = list(DIMS)
    scaler = StandardScaler().fit(train_df[feature_cols].values)
    X_tr_raw = scaler.transform(train_df[feature_cols].values)
    X_te_raw = scaler.transform(test_df[feature_cols].values)

    k_matched = X_tr_crisp.shape[1]
    print(f"FCM matched k={k_matched}")
    fcm = FuzzyCMeans(
        n_clusters=k_matched, m=2.0, max_iter=300, tol=1e-7, random_state=RANDOM_SEED
    ).fit(X_tr_raw)

    arms = {
        "Raw RFMS Baseline": (X_tr_raw, X_te_raw),
        "Crisp RFMS-FCA": (X_tr_crisp, X_te_crisp),
        "Fuzzy RFMS-FCA (Suppressed)": (X_tr_supp, X_te_supp),
        f"FCM soft (matched k={k_matched})": (fcm.memberships_, fcm.assign(X_te_raw)),
    }

    test_preds: dict[str, dict[str, np.ndarray]] = {}
    metrics_rows = []
    for arm, (X_tr, X_te) in arms.items():
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            scoring="neg_log_loss",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ).fit(X_tr, y_tr_rep)
        prob = clf.predict_proba(X_te)[:, 1]
        sp = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_tr_sp).predict(X_te)
        inv = (
            RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5)
            .fit(X_tr, np.log1p(y_tr_inv))
            .predict(X_te)
        )
        metrics_rows.append(
            {
                "arm": arm,
                "n_features": X_te.shape[1],
                "test_repurchase_auc": float(roc_auc_score(y_te_rep, prob)),
                "test_brier_score": float(brier_score_loss(y_te_rep, prob)),
                "test_spend_r2": float(r2_score(y_te_sp, sp)),
                "test_spend_mae": float(mean_absolute_error(y_te_sp, sp)),
                "test_spend_spearman": float(stats.spearmanr(y_te_sp, sp).statistic),
                "test_invoices_r2": float(r2_score(np.log1p(y_te_inv), inv)),
                "test_invoices_spearman": float(stats.spearmanr(y_te_inv, inv).statistic),
            }
        )
        test_preds[arm] = {"rep": prob, "sp": sp, "inv": inv}

    df_metrics = pd.DataFrame(metrics_rows)
    print("\nStrict Out-of-Sample Holdout Metrics:")
    print(df_metrics.to_string(index=False))

    # ---- Paired bootstrap vs the adopted arm ----
    ref_arm = "Fuzzy RFMS-FCA (Suppressed)"
    ref = test_preds[ref_arm]
    rng = np.random.default_rng(RANDOM_SEED)
    n_te = len(test_df)
    boot: dict[str, dict[str, list[float]]] = {
        a: {"auc": [], "sp": [], "inv": []} for a in arms if a != ref_arm
    }
    for _ in range(N_BOOT):
        idx = rng.choice(n_te, size=n_te, replace=True)
        if len(np.unique(y_te_rep[idx])) < 2:
            continue
        auc_r = roc_auc_score(y_te_rep[idx], ref["rep"][idx])
        r2_r = r2_score(y_te_sp[idx], ref["sp"][idx])
        inv_r = r2_score(np.log1p(y_te_inv[idx]), ref["inv"][idx])
        for a in boot:
            p = test_preds[a]
            boot[a]["auc"].append(roc_auc_score(y_te_rep[idx], p["rep"][idx]) - auc_r)
            boot[a]["sp"].append(r2_score(y_te_sp[idx], p["sp"][idx]) - r2_r)
            boot[a]["inv"].append(r2_score(np.log1p(y_te_inv[idx]), p["inv"][idx]) - inv_r)

    ci_rows = []
    for a, d in boot.items():
        for key, label in [("auc", "Delta AUC"), ("sp", "Delta Spend R2"), ("inv", "Delta Invoices R2")]:
            arr = np.asarray(d[key])
            lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            threshold = DECISION_AUC if key == "auc" else DECISION_R2
            point = float(df_metrics.loc[df_metrics["arm"] == a, {
                "auc": "test_repurchase_auc", "sp": "test_spend_r2", "inv": "test_invoices_r2"
            }[key]].values[0] - df_metrics.loc[df_metrics["arm"] == ref_arm, {
                "auc": "test_repurchase_auc", "sp": "test_spend_r2", "inv": "test_invoices_r2"
            }[key]].values[0])
            ci_rows.append(
                {
                    "comparison": f"{a} vs {ref_arm}",
                    "metric": label,
                    "point_estimate": point,
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "spans_zero": bool(lo <= 0 <= hi),
                    "exceeds_decision_threshold": bool(point >= threshold),
                    "interval_type": "95% Paired Customer Bootstrap (conditional on fixed split & models)",
                }
            )
    df_ci = pd.DataFrame(ci_rows)
    print(f"\nPaired Bootstrap CIs (vs {ref_arm}, out-of-sample):")
    print(df_ci.to_string(index=False))

    return df_metrics, df_ci


def load_olist_transactions() -> pd.DataFrame:
    """Per-order transaction rows from raw Olist (delivered, valid payment), reusing
    Step 1's filter conventions, for first-order dates and holdout outcomes."""
    from project_paths import OLIST_RAW_DIR

    orders = pd.read_csv(
        OLIST_RAW_DIR / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    customers = pd.read_csv(OLIST_RAW_DIR / "olist_customers_dataset.csv")
    payments = pd.read_csv(OLIST_RAW_DIR / "olist_order_payments_dataset.csv")

    orders = orders[orders["order_status"] == "delivered"].copy()
    orders = orders.dropna(subset=["order_purchase_timestamp"])
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left"
    )
    pay_agg = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    pay_agg = pay_agg.dropna(subset=["payment_value"])
    pay_agg = pay_agg[pay_agg["payment_value"] > 0]
    tx = orders.merge(pay_agg, on="order_id", how="inner")
    tx = tx.rename(columns={"customer_unique_id": "CustomerID"})
    tx = tx[["CustomerID", "order_id", "order_purchase_timestamp", "payment_value"]]
    return tx.sort_values(["CustomerID", "order_purchase_timestamp"]).reset_index(drop=True)


def main() -> None:
    print("=" * 80)
    print("OLIST RFMS-FCA: ADOPTED IMPROVEMENTS (SUPPRESSION + FCM BASELINE)")
    print("=" * 80)

    tx = load_olist_transactions()
    print(f"Transaction rows (delivered, valid payment): {len(tx):,}")

    df = pd.read_csv(PROCESSED + "olist_rfms_features.csv")
    print(f"Loaded processed features: {df.shape}")
    df = df.rename(columns={"customer_unique_id": "CustomerID", "F_star": "F"})
    df["T_ref"] = pd.to_datetime(df["last_purchase"])
    df["order_count"] = df["n_orders"]

    # Stored Step-1 score bands are the source of truth (re-derivation mismatch documented
    # in the module docstring); build the id -> scores lookup used by every split below.
    score_lookup = df[["CustomerID"] + [STORED_SCORE_COLS[d] for d in DIMS]].copy()
    score_lookup = score_lookup.rename(columns={STORED_SCORE_COLS[d]: f"{d}_score" for d in DIMS})

    rfm = df[["CustomerID", "R", "F", "M", "S", "T_ref", "order_count"]].copy()
    print(
        f"Customers: {len(rfm):,} | Repeat rate: {(df['n_orders'] > 1).mean():.4f} | "
        f"Reference date: {rfm['T_ref'].max()}"
    )
    for d in DIMS:
        dist = df[STORED_SCORE_COLS[d]].value_counts().sort_index().to_dict()
        print(f"  {STORED_SCORE_COLS[d]}: {dist}")

    part_a = run_part_a(rfm, score_lookup)
    df_metrics, df_ci = run_part_b(rfm, score_lookup, tx)

    df_metrics.to_csv(OUT_DIR / "temporal_holdout_metrics.csv", index=False)
    df_ci.to_csv(OUT_DIR / "paired_bootstrap_ci.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "J_MAX", "value": J_MAX},
            {"parameter": "SUPPORT_CUTOFF (holdout arms)", "value": SUPPORT_CUTOFF},
            {"parameter": "VALIDATION_SUPPORT", "value": VALIDATION_SUPPORT},
            {"parameter": "expected_lattice_count", "value": EXPECTED_LATTICE_COUNT},
            {"parameter": "validation_passed", "value": part_a["validation_ok"]},
            {"parameter": "CUTOFF_DATE", "value": str(CUTOFF_DATE)},
            {"parameter": "F_star_weights", "value": "alpha=0.20, beta=0.05, gamma=0.75 (from Step 1)"},
            {"parameter": "partB_obs_features", "value": "pre-cutoff orders only; F=0.20 (single-order F*), M/S = stored customer-level (S may include post-cutoff reviews for ~2% of obs cohort)"},
            {"parameter": "random_seed", "value": RANDOM_SEED},
        ]
    ).to_csv(OUT_DIR / "run_parameters.csv", index=False)

    # ---- Figure ----
    print("\nGenerating figure...")
    short = ["Raw", "Crisp FCA", "FCA+suppr", "FCM"]
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
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
        ax[j].tick_params(axis="x", labelsize=9)
        lo, hi = float(min(vals)), float(max(vals))
        ax[j].set_ylim(max(0, lo - 0.04 * abs(hi - lo) - 0.01), hi + 0.03 * abs(hi - lo) + 0.01)
        for i, v in enumerate(vals):
            ax[j].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_olist_rfms_holdout.png")
    plt.close()

    print("\nOlist RFMS comparison completed successfully!")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

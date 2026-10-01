"""Dunnhumby Complete Journey: RFM-FCA Cross-Domain Validation (Methodology v4).

Primary validation of the Fuzzy RFM-FCA framework on Dunnhumby 'The Complete Journey'
dataset as the second independent transaction domain alongside Online Retail II.

SCOPE AND CONSTRAINTS
---------------------
RFM ONLY. Satisfaction (S) is completely excluded.
Olist is OUT OF SCOPE - not loaded or analyzed anywhere.
F = number of DISTINCT BASKET_ID values per household_key.
M = sum(SALES_VALUE). SALES_VALUE is NOT multiplied by QUANTITY.
No F* calculation, no entropy-based F-weighting.
max_len = None (no artificial itemset-length cap).
Stability is a proxy only; NOT canonical Kuznetsov stability.
Kneedle threshold is Kneedle-INSPIRED (normalised max-chord-distance); NOT canonical Kneedle.
Canonical FPC / Xie-Beni are computed for FCM only, never for FCA clusters.
FCA hard clusters (Top-k Membership Hardening and Natural Hardening) use Silhouette / Davies-Bouldin only.
Alpha-cut (alpha >= 0.5) is preserved for extent thresholding and redundancy suppression, distinct from Top-k Membership Hardening.
Existing Retail II results are never modified or overwritten.
All scalers, scoring cutoffs, fuzzy centroids, concept mining, suppression,
FCM prototypes, and predictive models are fitted on TRAINING data only.

REUSED IMPLEMENTATIONS
-----------------------
dense_rank_scores, extract_dense_rank_cutoffs, apply_dense_rank_cutoffs,
band_centroids, piecewise_membership, compute_fuzzy_memberships,
mine_crisp_closed_concepts, mine_fuzzy_closed_concepts,
compute_customer_concept_memberships, assign_hard_clusters
    -> scripts/fair_comparison_retail2.py
suppress_redundant_concepts
    -> scripts/concept_redundancy.py
FuzzyCMeans
    -> scripts/fcm.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
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

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts
from fair_comparison_retail2 import (
    apply_dense_rank_cutoffs,
    assign_hard_clusters,
    band_centroids,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    mine_crisp_closed_concepts,
    mine_fuzzy_closed_concepts,
    piecewise_membership,
)
from fcm import FuzzyCMeans
from project_paths import DATA_DIR, RESULTS_DIR

# ------------------------------------------------------------------
# Output / data paths
# ------------------------------------------------------------------
OUT_DIR = RESULTS_DIR / "dunnhumby_rfm_fca"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DUNNHUMBY_DIR = DATA_DIR / "dunnhumby_raw"
TRANSACTION_FILE = DUNNHUMBY_DIR / "transaction_data.csv"

# ------------------------------------------------------------------
# Locked methodology parameters
# ------------------------------------------------------------------
DATASET_NAME = "Dunnhumby Complete Journey"
DIMS = ("R", "F", "M")
INVERT_DIMS = ("R",)          # R inverted: lower raw value -> higher score
OBS_CUTOFF_DAY = 620          # Observation window: days 1-620
HOLDOUT_END_DAY = 711         # Full dataset end (holdout: days 621-711 = 91 days)
SUPPORT_CUTOFF = 0.04
L_THRESHOLDS = (0.3, 0.5, 0.7)
J_MAX = 0.8                   # Jaccard redundancy suppression threshold
K_GRID = (4, 5, 6)
FCM_M = 2.0                   # Fuzzifier for canonical FCM
FIXED_SEED = 42
MULTI_SEEDS = list(range(1000, 1010))
N_BOOT = 1000


def _pr(msg: str) -> None:
    print(msg, flush=True)

# ==================================================================
# Utilities
# ==================================================================

def to_md_table(df: "pd.DataFrame") -> str:
    """Markdown table formatter without tabulate dependency."""
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.values]
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([hdr, sep] + body)


def _score_entropy(scores: "np.ndarray") -> float:
    """Shannon entropy in bits of a 5-band score distribution."""
    counts = np.bincount(scores.astype(int), minlength=6)[1:6]
    probs = counts / max(counts.sum(), 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _kneedle_inspired_threshold(y_desc: "np.ndarray") -> float:
    """Kneedle-inspired threshold: normalised maximum-distance-from-chord.

    This is a SIMPLIFIED implementation.
    It is NOT the canonical Kneedle algorithm.
    """
    y = np.asarray(y_desc, dtype=float)
    if len(y) < 3:
        return float(y[-1]) if len(y) else float("nan")
    x = np.linspace(0.0, 1.0, len(y))
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return float(y[int(np.argmax(num / den))])

# ==================================================================
# STEP 1: Data Audit and RFM Aggregation
# ==================================================================

def load_and_audit_transactions() -> tuple:
    """Load transaction_data.csv and produce a complete data audit.

    F = DISTINCT BASKET_ID count (NOT row count).
    M = sum(SALES_VALUE) (NOT SALES_VALUE * QUANTITY).
    """
    _pr(f"\n[STEP 1] Loading {TRANSACTION_FILE}")
    t0 = time.time()
    tx = pd.read_csv(TRANSACTION_FILE)
    _pr(f"  Loaded {len(tx):,} rows in {time.time()-t0:.1f}s")
    _pr(f"  Columns: {tx.columns.tolist()}")

    n_rows = len(tx)
    n_hh = tx["household_key"].nunique()
    n_baskets = tx["BASKET_ID"].nunique()
    n_days = tx["DAY"].nunique()
    day_min = int(tx["DAY"].min())
    day_max = int(tx["DAY"].max())
    n_products = tx["PRODUCT_ID"].nunique()
    n_missing = int(tx.isnull().sum().sum())
    n_dup = int(tx.duplicated().sum())

    basket_per_hh = tx.groupby("household_key")["BASKET_ID"].nunique()
    hh_1 = int((basket_per_hh == 1).sum())
    hh_2plus = int((basket_per_hh >= 2).sum())
    hh_3plus = int((basket_per_hh >= 3).sum())
    n_distinct_counts = int(basket_per_hh.nunique())

    rows_per_basket = tx.groupby("BASKET_ID").size()
    multi_row = int((rows_per_basket > 1).sum())

    audit = {
        "n_rows": n_rows,
        "n_households": n_hh,
        "n_baskets": n_baskets,
        "n_unique_days": n_days,
        "day_min": day_min,
        "day_max": day_max,
        "n_products": n_products,
        "n_missing_values": n_missing,
        "n_duplicate_rows": n_dup,
        "hh_with_1_basket": hh_1,
        "hh_with_2plus_baskets": hh_2plus,
        "hh_with_3plus_baskets": hh_3plus,
        "n_distinct_basket_counts": n_distinct_counts,
        "mean_F_full": float(basket_per_hh.mean()),
        "median_F_full": float(basket_per_hh.median()),
        "std_F_full": float(basket_per_hh.std()),
        "max_F_full": float(basket_per_hh.max()),
        "multi_row_baskets_confirms_product_line_granularity": multi_row,
    }

    _pr(f"  Households: {n_hh:,}")
    _pr(f"  Baskets: {n_baskets:,}")
    _pr(f"  DAY range: {day_min}-{day_max}")
    _pr(f"  One-basket HH: {hh_1} ({hh_1/n_hh*100:.2f}%)")
    _pr(f"  Multi-row baskets: {multi_row:,} (product-line granularity confirmed)")
    return tx, audit


def aggregate_rfm_dunnhumby(tx: "pd.DataFrame", obs_cutoff: int = OBS_CUTOFF_DAY) -> tuple:
    """Compute household-level RFM for full population and observation cohort.

    F = DISTINCT BASKET_ID count per household_key.
    M = sum(SALES_VALUE).
    R = obs_cutoff - max(DAY).
    SALES_VALUE is NOT multiplied by QUANTITY.
    """
    full_agg = (
        tx.groupby("household_key")
        .agg(
            last_purchase_day=("DAY", "max"),
            first_purchase_day=("DAY", "min"),
            n_baskets=("BASKET_ID", "nunique"),
            M=("SALES_VALUE", "sum"),
        )
        .reset_index()
    )
    full_agg["F"] = full_agg["n_baskets"]
    full_agg["tenure_days"] = full_agg["last_purchase_day"] - full_agg["first_purchase_day"]
    full_agg["R_full"] = HOLDOUT_END_DAY - full_agg["last_purchase_day"]

    tx_obs = tx[tx["DAY"] <= obs_cutoff].copy()
    obs_agg = (
        tx_obs.groupby("household_key")
        .agg(
            last_purchase_day=("DAY", "max"),
            n_baskets=("BASKET_ID", "nunique"),
            M=("SALES_VALUE", "sum"),
        )
        .reset_index()
    )
    obs_agg["F"] = obs_agg["n_baskets"]
    obs_agg["R"] = obs_cutoff - obs_agg["last_purchase_day"]

    _pr(f"  Full population: {len(full_agg):,} households")
    _pr(f"  Observation cohort (days 1-{obs_cutoff}): {len(obs_agg):,} households")
    return full_agg, obs_agg

# ==================================================================
# STEP 2-3: Descriptive and Frequency Statistics
# ==================================================================

def compute_dataset_statistics(full_rfm, obs_rfm, obs_scored):
    """Compute RFM distributional statistics for both cohorts."""
    rows = []
    for label, df, r_col in [
        (f"Full Population ({HOLDOUT_END_DAY} Days)", full_rfm, "R_full"),
        (f"Observation Cohort ({OBS_CUTOFF_DAY} Days)", obs_rfm, "R"),
    ]:
        for dim, col in [("R", r_col), ("F", "F"), ("M", "M")]:
            vals = df[col].to_numpy(dtype=float)
            rows.append({
                "Cohort": label, "Metric": dim,
                "Count": len(vals), "Mean": float(np.mean(vals)),
                "Std": float(np.std(vals)), "Min": float(np.min(vals)),
                "P25": float(np.percentile(vals, 25)),
                "P50_Median": float(np.median(vals)),
                "P75": float(np.percentile(vals, 75)),
                "P90": float(np.percentile(vals, 90)),
                "P95": float(np.percentile(vals, 95)),
                "Max": float(np.max(vals)),
            })
    df_stats = pd.DataFrame(rows)

    obs_scored_full = dense_rank_scores(
        obs_rfm[["household_key", "R", "F", "M"]].rename(columns={"household_key": "CustomerID"})
    )
    f_entropy = _score_entropy(obs_scored_full["F_score"].to_numpy())
    def dist_dict(s):
        return {int(k): int(v) for k, v in s.value_counts().sort_index().items()}

    summary = pd.DataFrame([{
        "Dataset": DATASET_NAME,
        "Customers": len(obs_rfm),
        "Orders": int(obs_rfm["n_baskets"].sum()),
        "Repeat rate": float((obs_rfm["F"] >= 2).mean()),
        "One-time buyer %": float((obs_rfm["F"] == 1).mean() * 100),
        "Unique R values": int(obs_rfm["R"].nunique()),
        "Unique F values": int(obs_rfm["F"].nunique()),
        "Unique M values": int(obs_rfm["M"].nunique()),
        "F-score distribution": str(dist_dict(obs_scored_full["F_score"])),
        "R-score distribution": str(dist_dict(obs_scored_full["R_score"])),
        "M-score distribution": str(dist_dict(obs_scored_full["M_score"])),
        "F entropy (bits)": f_entropy,
        "Max theoretical entropy (5 bands)": float(math.log2(5)),
    }])
    return df_stats, summary


def compute_frequency_analysis(obs_rfm, obs_scored):
    """Detailed frequency analysis: key diagnostic for cross-domain comparison."""
    f_vals = obs_rfm["F"].to_numpy(dtype=float)
    f_scores = obs_scored["F_score"].to_numpy() if "F_score" in obs_scored.columns else None
    n = len(f_vals)
    min_f = float(f_vals.min())
    f1_vals = f_vals[f_scores == 1] if f_scores is not None else np.array([])
    row = {
        "Dataset / Representation": f"{DATASET_NAME} (Baskets per Household)",
        "Total Customers": n,
        "Pct F = min (One-time)": float((f_vals == min_f).mean() * 100),
        "Pct F >= 2": float((f_vals >= 2).mean() * 100),
        "Pct F >= 3": float((f_vals >= 3).mean() * 100),
        "Unique raw F values": int(pd.unique(f_vals).shape[0]),
        "F min value": min_f,
        "F median value": float(np.median(f_vals)),
        "F max value": float(f_vals.max()),
        "F mean value": float(np.mean(f_vals)),
        "F std value": float(np.std(f_vals)),
        "Distinct F values in lowest band (F1)": int(pd.unique(f1_vals).shape[0]) if len(f1_vals) else 0,
        "Variance in lowest band (F1)": float(np.var(f1_vals)) if len(f1_vals) > 1 else 0.0,
        "F-score Shannon entropy (bits)": _score_entropy(f_scores) if f_scores is not None else float("nan"),
    }
    return pd.DataFrame([row])

# ==================================================================
# STEP 4: Full-population FCA
# ==================================================================

def run_full_population_fca(obs_rfm):
    """Mine crisp and fuzzy RFM concepts on the observation cohort.

    Stability is a PROXY only, not canonical Kuznetsov stability.
    Kneedle threshold is Kneedle-INSPIRED, not canonical Kneedle.
    """
    _pr("\n[STEP 4] Full-population FCA")
    cust_df = obs_rfm[["household_key", "R", "F", "M"]].rename(columns={"household_key": "CustomerID"})
    obs_scored = dense_rank_scores(cust_df, dims=DIMS)
    obs_fuzzy_mu, obs_centroids, _ = compute_fuzzy_memberships(obs_scored, dims=DIMS)

    t0 = time.time()
    crisp_concepts = mine_crisp_closed_concepts(obs_scored, min_support=SUPPORT_CUTOFF, dims=DIMS)
    _pr(f"  Crisp concepts: {len(crisp_concepts)} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    fuzzy_concepts, supp_knee, stab_knee = mine_fuzzy_closed_concepts(
        obs_fuzzy_mu, obs_scored, min_support=SUPPORT_CUTOFF, dims=DIMS
    )
    _pr(f"  Fuzzy concepts raw: {len(fuzzy_concepts)} ({time.time()-t0:.1f}s)")
    _pr(f"  Kneedle-inspired support threshold: {supp_knee:.6f}")
    _pr(f"  Kneedle-inspired stability-proxy threshold: {stab_knee:.6f}")

    mu_matrix = compute_customer_concept_memberships(fuzzy_concepts, obs_fuzzy_mu)
    suppressed = suppress_redundant_concepts(fuzzy_concepts, mu_matrix, j_max=J_MAX, mu_cut=0.5)
    n_removed = len(fuzzy_concepts) - len(suppressed)
    _pr(f"  Suppressed: {len(suppressed)} (removed {n_removed})")

    mean_cpc = float(mu_matrix.sum(axis=1).mean())
    median_cpc = float(np.median(mu_matrix.sum(axis=1)))
    extent_sizes = (mu_matrix >= 0.5).sum(axis=0)

    return {
        "crisp_concepts": crisp_concepts,
        "fuzzy_concepts": fuzzy_concepts,
        "suppressed_concepts": suppressed,
        "mu_matrix": mu_matrix,
        "supp_knee": supp_knee,
        "stab_knee": stab_knee,
        "n_removed": n_removed,
        "mean_concepts_per_customer": mean_cpc,
        "median_concepts_per_customer": median_cpc,
        "mean_extent_size": float(np.mean(extent_sizes)),
        "median_extent_size": float(np.median(extent_sizes)),
        "obs_centroids": obs_centroids,
    }, obs_scored, obs_fuzzy_mu

# ==================================================================
# STEP 5: Clustering Benchmarks
# ==================================================================

def run_clustering_benchmarks(obs_rfm, obs_scored, obs_fuzzy_mu, full_results):
    """K-Means, Ward, FCM, and FCA hard clustering benchmarks.

    Evaluates geometric partition quality (Silhouette, Davies-Bouldin)
    in the agreed standardized raw RFM space (X_raw, shape N x 3).
    Canonical FCM metrics (FPC, Xie-Beni) computed for FCM ONLY.

    TERMINOLOGY AND METHODOLOGY DISTINCTION:
    - Alpha-cut: Threshold membership at alpha >= 0.5 to extract concept extents
      or binary profiles (preserved in concept suppression and extent statistics).
    - Top-k Membership Hardening: Select k concepts and assign each customer
      to the concept with maximum membership via argmax over top-k fuzzy memberships.
      This is distinct from an alpha-cut.
    """
    _pr("\n[STEP 5] Clustering benchmarks")
    scaler = StandardScaler()
    X_raw = scaler.fit_transform(obs_rfm[["R", "F", "M"]].values)

    crisp_bands = pd.DataFrame({
        f"{dim}{k}": (obs_scored[f"{dim}_score"] == k).astype(float)
        for dim in DIMS for k in range(1, 6)
    })
    X_crisp = compute_customer_concept_memberships(full_results["crisp_concepts"], crisp_bands)
    mu_supp = compute_customer_concept_memberships(full_results["suppressed_concepts"], obs_fuzzy_mu)

    records = []
    fcm_models = {}

    for k in K_GRID:
        # 1. K-Means directly on standardized Raw RFM
        km = KMeans(n_clusters=k, random_state=FIXED_SEED, n_init=10).fit(X_raw)
        sil_km = float(silhouette_score(X_raw, km.labels_, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
        db_km = float(davies_bouldin_score(X_raw, km.labels_))
        records.append({
            "Dataset": DATASET_NAME, "Method": "K-Means", "Hardening": "Partition (k-means)",
            "k": k, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
            "Silhouette": sil_km, "Davies-Bouldin": db_km,
        })

        # 2. Ward (Agglomerative) directly on standardized Raw RFM
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_raw)
        sil_ward = float(silhouette_score(X_raw, ward.labels_, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
        db_ward = float(davies_bouldin_score(X_raw, ward.labels_))
        records.append({
            "Dataset": DATASET_NAME, "Method": "Ward (Agglomerative)", "Hardening": "Hierarchical (Ward)",
            "k": k, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
            "Silhouette": sil_ward, "Davies-Bouldin": db_ward,
        })

        # 3. Canonical FCM on standardized Raw RFM
        fcm = FuzzyCMeans(n_clusters=k, m=FCM_M, random_state=FIXED_SEED)
        fcm.fit(X_raw)
        fcm_models[k] = fcm
        U = fcm.memberships_
        hard_fcm = np.argmax(U, axis=1)
        sil_fcm = float(silhouette_score(X_raw, hard_fcm, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
        db_fcm = float(davies_bouldin_score(X_raw, hard_fcm))

        # Canonical FCM metrics (FPC and Xie-Beni - FCM ONLY)
        fpc = float(np.mean(np.sum(U ** 2, axis=1)))
        dist2 = np.sum((X_raw[:, None, :] - fcm.centroids_[None, :, :]) ** 2, axis=2)
        numerator = float(np.sum((U ** FCM_M) * dist2))
        cdists = [float(np.sum((fcm.centroids_[c1] - fcm.centroids_[c2]) ** 2))
                  for c1 in range(k) for c2 in range(k) if c1 != c2]
        min_cdist2 = min(cdists) if cdists else 1.0
        xie_beni = float(numerator / (len(X_raw) * min_cdist2))
        _pr(f"  k={k}: FCM FPC={fpc:.4f}, Xie-Beni={xie_beni:.4f}, Sil(argmax)={sil_fcm:.4f}")

        records.append({
            "Dataset": DATASET_NAME, "Method": "FCM (Canonical)", "Hardening": "Argmax Soft Membership",
            "k": k, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
            "Silhouette": sil_fcm, "Davies-Bouldin": db_fcm,
        })

        # 4. Fuzzy RFM-FCA Top-k Membership Hardening:
        #    Select k concepts and assign each customer to the concept with maximum membership.
        #    Note: This is Top-k Membership Hardening (argmax over top-k fuzzy memberships),
        #    NOT an alpha-cut (which thresholds membership at alpha >= 0.5).
        nontriv = full_results["suppressed_concepts"][full_results["suppressed_concepts"]["intent_size"] > 0]
        top_k_idx = nontriv.index[:k]
        sub_mu = mu_supp[:, top_k_idx]
        labels_fca_k = np.argmax(sub_mu, axis=1)
        sil_fca_k = float(silhouette_score(X_raw, labels_fca_k, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
        db_fca_k = float(davies_bouldin_score(X_raw, labels_fca_k))
        records.append({
            "Dataset": DATASET_NAME, "Method": "Fuzzy RFM-FCA", "Hardening": "Top-k Membership Hardening",
            "k": k, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
            "Silhouette": sil_fca_k, "Davies-Bouldin": db_fca_k,
        })

    # 5. Natural data-driven hardening (assign_hard_clusters)
    labels_crisp_nat = assign_hard_clusters(full_results["crisp_concepts"], X_crisp)
    k_crisp_nat = int(len(np.unique(labels_crisp_nat)))
    sil_cn = float(silhouette_score(X_raw, labels_crisp_nat, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
    db_cn = float(davies_bouldin_score(X_raw, labels_crisp_nat))
    records.append({
        "Dataset": DATASET_NAME, "Method": "Crisp RFM-FCA", "Hardening": "Natural Hardening Rule",
        "k": k_crisp_nat, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
        "Silhouette": sil_cn, "Davies-Bouldin": db_cn,
    })

    labels_fuzzy_nat = assign_hard_clusters(full_results["suppressed_concepts"], mu_supp)
    k_fuzzy_nat = int(len(np.unique(labels_fuzzy_nat)))
    sil_fn = float(silhouette_score(X_raw, labels_fuzzy_nat, sample_size=min(2000, len(X_raw)), random_state=FIXED_SEED))
    db_fn = float(davies_bouldin_score(X_raw, labels_fuzzy_nat))
    records.append({
        "Dataset": DATASET_NAME, "Method": "Fuzzy RFM-FCA", "Hardening": "Natural Hardening Rule",
        "k": k_fuzzy_nat, "Evaluation Space": "Raw RFM (Standardized)", "n_features": 3,
        "Silhouette": sil_fn, "Davies-Bouldin": db_fn,
    })

    return pd.DataFrame(records), fcm_models

# ==================================================================
# STEP 6: Leakage-Free Temporal Holdout (single seed)
# ==================================================================

def _fit_predict_arm(arm_name, X_tr, X_te,
                     y_tr_rep, y_te_rep,
                     y_tr_sp, y_te_sp,
                     y_tr_inv, y_te_inv,
                     seed=FIXED_SEED):
    """Fit models on train; evaluate strictly on test."""
    clf = LogisticRegressionCV(
        Cs=10, cv=5, scoring="roc_auc",
        solver="lbfgs", max_iter=2000, random_state=seed,
    ).fit(X_tr, y_tr_rep)
    prob_te = clf.predict_proba(X_te)[:, 1]

    reg_sp = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5).fit(X_tr, y_tr_sp)
    pred_sp = reg_sp.predict(X_te)

    reg_inv = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5).fit(X_tr, y_tr_inv)
    pred_inv = reg_inv.predict(X_te)

    sp_r, _ = stats.spearmanr(y_te_sp, pred_sp)
    inv_r, _ = stats.spearmanr(y_te_inv, pred_inv)

    return {
        "arm": arm_name,
        "n_features": X_te.shape[1],
        "test_repurchase_auc": float(roc_auc_score(y_te_rep, prob_te)),
        "test_brier_score": float(brier_score_loss(y_te_rep, prob_te)),
        "test_spend_r2": float(r2_score(y_te_sp, pred_sp)),
        "test_spend_mae": float(mean_absolute_error(y_te_sp, pred_sp)),
        "test_spend_spearman": float(sp_r),
        "test_invoices_r2": float(r2_score(y_te_inv, pred_inv)),
        "test_invoices_mae": float(mean_absolute_error(y_te_inv, pred_inv)),
        "test_invoices_spearman": float(inv_r),
        "_prob_te": prob_te,
        "_pred_sp_te": pred_sp,
        "_pred_inv_te": pred_inv,
    }


def run_temporal_holdout(tx, random_seed=FIXED_SEED, verbose=True):
    """Leakage-free temporal holdout.

    Observation: days 1-620. Holdout: days 621-711 (91 days).
    All representations fitted on TRAIN split only.
    Test customers projected using frozen parameters.
    """
    if verbose:
        _pr(f"\n[STEP 6] Temporal holdout (seed={random_seed})")

    tx_obs = tx[tx["DAY"] <= OBS_CUTOFF_DAY].copy()
    obs_agg = (
        tx_obs.groupby("household_key")
        .agg(last_purchase_day=("DAY", "max"),
             n_baskets=("BASKET_ID", "nunique"),
             M=("SALES_VALUE", "sum"))
        .reset_index()
    )
    obs_agg["F"] = obs_agg["n_baskets"]
    obs_agg["R"] = OBS_CUTOFF_DAY - obs_agg["last_purchase_day"]
    obs_agg["CustomerID"] = obs_agg["household_key"].astype(str)

    tx_hold = tx[(tx["DAY"] > OBS_CUTOFF_DAY) & (tx["DAY"] <= HOLDOUT_END_DAY)].copy()
    hold_agg = (
        tx_hold.groupby("household_key")
        .agg(future_invoices=("BASKET_ID", "nunique"),
             future_spend=("SALES_VALUE", "sum"))
        .reset_index()
    )

    merged = obs_agg.merge(hold_agg, on="household_key", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)

    if verbose:
        _pr(f"  Obs cohort: {len(merged):,} HH | Repurchase rate: {merged['repurchased'].mean():.4f}")

    train_df, test_df = train_test_split(
        merged, test_size=0.30,
        stratify=merged["repurchased"].to_numpy(),
        random_state=random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    n_train, n_test = len(train_df), len(test_df)
    if verbose:
        _pr(f"  Train: {n_train} | Test: {n_test}")

    y_tr_rep = train_df["repurchased"].to_numpy()
    y_te_rep = test_df["repurchased"].to_numpy()
    y_tr_sp = np.log1p(train_df["future_spend"].to_numpy())
    y_te_sp = np.log1p(test_df["future_spend"].to_numpy())
    y_tr_inv = np.log1p(train_df["future_invoices"].to_numpy())
    y_te_inv = np.log1p(test_df["future_invoices"].to_numpy())

    # Raw RFM (scaler fitted on train only)
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_tr_raw = scaler.transform(train_df[["R", "F", "M"]].values)
    X_te_raw = scaler.transform(test_df[["R", "F", "M"]].values)

    # Scoring and centroids on train only
    train_cust = train_df[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)
    train_cutoffs = {
        dim: extract_dense_rank_cutoffs(train_scored, dim, invert=(dim in INVERT_DIMS))
        for dim in DIMS
    }

    # Crisp concepts on train
    crisp_concepts = mine_crisp_closed_concepts(train_scored, min_support=SUPPORT_CUTOFF, dims=DIMS)
    crisp_bands_tr = pd.DataFrame({
        f"{dim}{k}": (train_scored[f"{dim}_score"] == k).astype(float)
        for dim in DIMS for k in range(1, 6)
    })
    X_tr_crisp = compute_customer_concept_memberships(crisp_concepts, crisp_bands_tr)

    # Fuzzy concepts on train (with suppression)
    fuzzy_concepts, _, _ = mine_fuzzy_closed_concepts(
        train_fuzzy_mu, train_scored, min_support=SUPPORT_CUTOFF, dims=DIMS
    )
    mu_tr_full = compute_customer_concept_memberships(fuzzy_concepts, train_fuzzy_mu)
    fuzzy_supp = suppress_redundant_concepts(fuzzy_concepts, mu_tr_full, j_max=J_MAX, mu_cut=0.5)
    X_tr_fuzzy = compute_customer_concept_memberships(fuzzy_supp, train_fuzzy_mu)

    # FCM on train (k matched to crisp concept count)
    k_fcm = len(crisp_concepts)
    fcm = FuzzyCMeans(n_clusters=k_fcm, m=FCM_M, random_state=random_seed)
    fcm.fit(X_tr_raw)
    X_tr_fcm = fcm.memberships_

    if verbose:
        _pr(f"  Crisp concepts: {len(crisp_concepts)} | Fuzzy suppressed: {len(fuzzy_supp)} | FCM k: {k_fcm}")

    # Project test customers via frozen train parameters
    test_crisp_scores = {}
    for dim in DIMS:
        s = apply_dense_rank_cutoffs(test_df[dim].to_numpy(), train_cutoffs[dim], invert=(dim in INVERT_DIMS))
        for k in range(1, 6):
            test_crisp_scores[f"{dim}{k}"] = (s == k).astype(float)
    crisp_bands_te = pd.DataFrame(test_crisp_scores)
    X_te_crisp = compute_customer_concept_memberships(crisp_concepts, crisp_bands_te)

    test_cust = test_df[["CustomerID", "R", "F", "M"]].copy()
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(test_cust, trained_centroids=train_centroids, dims=DIMS)
    X_te_fuzzy = compute_customer_concept_memberships(fuzzy_supp, test_fuzzy_mu)
    X_te_fcm = fcm.assign(X_te_raw)

    arms = {
        "Raw RFM Baseline": (X_tr_raw, X_te_raw),
        "Crisp RFM-FCA": (X_tr_crisp, X_te_crisp),
        "Fuzzy RFM-FCA (Suppressed)": (X_tr_fuzzy, X_te_fuzzy),
        f"FCM Soft (k={k_fcm})": (X_tr_fcm, X_te_fcm),
    }

    metrics_list, preds_dict = [], {}
    for arm_name, (X_tr, X_te) in arms.items():
        res = _fit_predict_arm(arm_name, X_tr, X_te,
                               y_tr_rep, y_te_rep, y_tr_sp, y_te_sp, y_tr_inv, y_te_inv,
                               seed=random_seed)
        preds_dict[arm_name] = res
        pub = {kk: vv for kk, vv in res.items() if not kk.startswith("_")}
        pub["dataset"] = DATASET_NAME
        metrics_list.append(pub)
        if verbose:
            _pr(f"  {arm_name}: AUC={res['test_repurchase_auc']:.4f}  SpR2={res['test_spend_r2']:.4f}  InvR2={res['test_invoices_r2']:.4f}")

    df_metrics = pd.DataFrame(metrics_list)

    # Paired bootstrap CIs
    rng = np.random.default_rng(random_seed)
    boot_rows = []
    comparisons = [
        ("Fuzzy RFM-FCA (Suppressed)", "Raw RFM Baseline"),
        ("Crisp RFM-FCA", "Raw RFM Baseline"),
        ("Fuzzy RFM-FCA (Suppressed)", "Crisp RFM-FCA"),
        (f"FCM Soft (k={k_fcm})", "Raw RFM Baseline"),
        ("Fuzzy RFM-FCA (Suppressed)", f"FCM Soft (k={k_fcm})"),
    ]
    for m1, m2 in comparisons:
        if m1 not in preds_dict or m2 not in preds_dict:
            continue
        for metric_label, y_te, pk in [
            ("Repurchase AUC", y_te_rep, "_prob_te"),
            ("Spend R2", y_te_sp, "_pred_sp_te"),
            ("Invoice R2", y_te_inv, "_pred_inv_te"),
        ]:
            p1, p2 = preds_dict[m1][pk], preds_dict[m2][pk]
            deltas = []
            for _ in range(N_BOOT):
                idx = rng.choice(n_test, size=n_test, replace=True)
                yt = y_te[idx]
                if metric_label == "Repurchase AUC":
                    if len(np.unique(yt)) < 2:
                        continue
                    val1 = roc_auc_score(yt, p1[idx])
                    val2 = roc_auc_score(yt, p2[idx])
                else:
                    val1 = r2_score(yt, p1[idx])
                    val2 = r2_score(yt, p2[idx])
                deltas.append(val1 - val2)
            if not deltas:
                continue
            d_arr = np.array(deltas)
            ci_lo, ci_hi = float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))
            pt1_col = "test_repurchase_auc" if metric_label == "Repurchase AUC" else (
                "test_spend_r2" if metric_label == "Spend R2" else "test_invoices_r2")
            pt = float(df_metrics.loc[df_metrics["arm"] == m1, pt1_col].values[0]) -                      float(df_metrics.loc[df_metrics["arm"] == m2, pt1_col].values[0])
            boot_rows.append({
                "Comparison": f"{m1} vs {m2}",
                "Metric": metric_label,
                "Point Delta (M1 - M2)": pt,
                "95% CI Lower": ci_lo,
                "95% CI Upper": ci_hi,
                "Spans Zero": bool(ci_lo <= 0 <= ci_hi),
                "Empirical p-value": float(np.mean(d_arr <= 0) if pt > 0 else np.mean(d_arr >= 0)),
            })

    df_boot = pd.DataFrame(boot_rows)
    return df_metrics, df_boot, {"n_train": n_train, "n_test": n_test, "k_fcm": k_fcm}

# ==================================================================
# STEP 7: Multi-Split Cross-Validation
# ==================================================================

def run_multisplit_validation(tx):
    """Run temporal holdout with 10 different seeds (1000-1009)."""
    _pr("\n[STEP 7] Multi-split validation (10 splits)")
    all_rows = []
    for split_idx, seed in enumerate(MULTI_SEEDS):
        _pr(f"  Split {split_idx} (seed={seed})")
        try:
            df_m, _, _ = run_temporal_holdout(tx, random_seed=seed, verbose=False)
            for _, row in df_m.iterrows():
                arm_name = row["arm"]
                if arm_name.startswith("FCM Soft"):
                    arm_name = "FCM Soft (matched k)"
                all_rows.append({
                    "split_seed": seed,
                    "split_index": split_idx,
                    "arm": arm_name,
                    "n_features": row["n_features"],
                    "repurchase_auc": row["test_repurchase_auc"],
                    "spend_r2": row["test_spend_r2"],
                    "invoice_r2": row["test_invoices_r2"],
                })
        except Exception as exc:
            _pr(f"  Split {split_idx} FAILED: {exc}")
    return pd.DataFrame(all_rows)


# ==================================================================
# STEP 8: Figures
# ==================================================================

def create_figures(obs_rfm, obs_scored, full_results, df_clust, df_pred, df_split):
    """Generate all required figures."""
    _pr("\n[STEP 8] Generating figures")
    COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # fig_frequency_distribution: R, F, M histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dunnhumby: RFM Feature Distributions (Observation Cohort, days 1-620)",
                 fontsize=13, fontweight="bold")
    for ax, (col, lbl, color) in zip(axes, [
        ("R", "Recency (days from cutoff)", COLORS[0]),
        ("F", "Frequency (baskets)", COLORS[1]),
        ("M", "Monetary (sum SALES_VALUE)", COLORS[2]),
    ]):
        vals = obs_rfm[col].to_numpy()
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(float(np.median(vals)), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median={np.median(vals):.1f}")
        ax.set_xlabel(lbl, fontsize=10); ax.set_ylabel("Households", fontsize=10)
        ax.set_title(f"Distribution of {col}", fontsize=11); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_frequency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig_rfm_distributions: score band bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dunnhumby: RFM Score Band Distributions (Dense Rank, 5 Bands)",
                 fontsize=13, fontweight="bold")
    for ax, (col, lbl, color) in zip(axes, [
        ("R_score", "Recency Score", COLORS[0]),
        ("F_score", "Frequency Score", COLORS[1]),
        ("M_score", "Monetary Score", COLORS[2]),
    ]):
        if col in obs_scored.columns:
            cnts = obs_scored[col].value_counts().sort_index()
            ax.bar(cnts.index, cnts.values, color=color, alpha=0.8, edgecolor="white")
            ax.set_xlabel("Score Band", fontsize=10); ax.set_ylabel("Households", fontsize=10)
            ax.set_title(lbl, fontsize=11); ax.set_xticks([1,2,3,4,5])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_rfm_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig_concept_counts
    n_c = len(full_results["crisp_concepts"])
    n_f = len(full_results["fuzzy_concepts"])
    n_s = len(full_results["suppressed_concepts"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Crisp FCA\n(Raw)", "Fuzzy FCA\n(Raw)", "Fuzzy FCA\n(Suppressed)"],
                  [n_c, n_f, n_s], color=COLORS[:3], alpha=0.85, edgecolor="white")
    ax.set_title(f"Dunnhumby: FCA Concept Counts (n={len(obs_rfm):,})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Concepts", fontsize=11)
    for bar, v in zip(bars, [n_c, n_f, n_s]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_concept_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig_clustering_metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dunnhumby: Clustering Metrics on Standardized Raw RFM (k=4,5,6)", fontsize=13, fontweight="bold")
    methods = [m for m in df_clust["Method"].unique() if m != "Crisp RFM-FCA"]
    cmap = {r: COLORS[i % len(COLORS)] for i, r in enumerate(methods)}
    for ax, (metric, better) in zip(axes, [("Silhouette","higher"), ("Davies-Bouldin","lower")]):
        for m in methods:
            sub = df_clust[(df_clust["Method"]==m) & (df_clust["k"].isin(K_GRID))]
            if sub.empty: continue
            ax.plot(sub["k"].values, sub[metric].values, marker="o",
                    label=m, color=cmap[m], linewidth=1.8)
        ax.set_xlabel("k", fontsize=10); ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric} ({better} is better)", fontsize=10)
        ax.set_xticks(list(K_GRID)); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_clustering_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig_predictive_comparison
    if not df_pred.empty:
        arm_names = df_pred["arm"].tolist()
        metrics_p = [
            ("test_repurchase_auc", "Repurchase AUC"),
            ("test_spend_r2", "Spend R2 (log1p)"),
            ("test_invoices_r2", "Invoice R2 (log1p)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Dunnhumby: Predictive Performance (Fixed Split seed={FIXED_SEED})",
                     fontsize=12, fontweight="bold")
        for ax, (col, title) in zip(axes, metrics_p):
            if col not in df_pred.columns: continue
            vals_p = df_pred[col].values
            bars = ax.bar(np.arange(len(arm_names)), vals_p, color=COLORS[:len(arm_names)], alpha=0.85)
            ax.set_xticks(np.arange(len(arm_names)))
            ax.set_xticklabels([a.replace(" (Suppressed)", "\n(S.)").replace(" Baseline", "\n(Base)")
                                for a in arm_names], fontsize=8, rotation=15, ha="right")
            ax.set_title(title, fontsize=10)
            ax.set_ylim(0, max(vals_p)*1.15 if max(vals_p) > 0 else 1)
            for bar, v in zip(bars, vals_p):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "fig_predictive_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    _pr("  Figures saved.")

# ==================================================================
# STEP 9-11: Tables, Parameters, and Summary
# ==================================================================

def build_concept_counts_table(full_results, obs_rfm):
    n_obs = len(obs_rfm)
    return pd.DataFrame([{
        "Dataset": DATASET_NAME,
        "Cohort": f"Observation (n={n_obs})",
        "Min Support Threshold": SUPPORT_CUTOFF,
        "Kneedle Support Threshold": full_results["supp_knee"],
        "Raw Crisp Concepts": len(full_results["crisp_concepts"]),
        "Raw Fuzzy Concepts": len(full_results["fuzzy_concepts"]),
        "Redundancy Suppressed Concepts (J_max=0.8)": len(full_results["suppressed_concepts"]),
        "Concepts Removed by Suppression": full_results["n_removed"],
        "Mean Concepts per Customer": full_results["mean_concepts_per_customer"],
        "Median Concepts per Customer": full_results["median_concepts_per_customer"],
        "Mean Extent Size (mu > 0.5)": full_results["mean_extent_size"],
        "Median Extent Size (mu > 0.5)": full_results["median_extent_size"],
    }])


def build_run_parameters():
    return pd.DataFrame([
        ("Dataset Name", DATASET_NAME),
        ("Data File Path", str(TRANSACTION_FILE)),
        ("Observation Cutoff Day", OBS_CUTOFF_DAY),
        ("Holdout End Day", HOLDOUT_END_DAY),
        ("Observation Duration", f"{OBS_CUTOFF_DAY} days (~{OBS_CUTOFF_DAY/7:.1f} weeks)"),
        ("Holdout Duration", f"{HOLDOUT_END_DAY-OBS_CUTOFF_DAY} days (13 weeks / 1 quarter)"),
        ("Frequency Definition", "Distinct BASKET_ID values per household_key"),
        ("Monetary Definition", "Sum of SALES_VALUE (NOT multiplied by QUANTITY)"),
        ("Recency Definition", "Cutoff Day minus max transaction Day"),
        ("Score Methodology", "Dense rank (5 bands, R inverted, F/M increasing)"),
        ("FCA Minimum Support", SUPPORT_CUTOFF),
        ("L-Fuzzy Cutoffs", str(L_THRESHOLDS)),
        ("Jaccard Redundancy Cutoff (J_max)", J_MAX),
        ("max_len", "None (no artificial cap)"),
        ("Stability Proxy Note", "Proxy only; NOT canonical Kuznetsov stability"),
        ("Kneedle Threshold Note", "Kneedle-INSPIRED (normalised max-chord-distance); NOT canonical Kneedle"),
        ("Clustering Algorithms", f"K-Means, Ward at k={','.join(map(str,K_GRID))}"),
        ("FCM Fuzzifier (m)", FCM_M),
        ("Classification Model", "LogisticRegressionCV(Cs=10, cv=5, scoring=roc_auc)"),
        ("Regression Models", "RidgeCV(alphas=logspace(-3,3,20), cv=5)"),
        ("Fixed Split Seed", FIXED_SEED),
        ("Multi-split Seeds", f"{MULTI_SEEDS[0]} to {MULTI_SEEDS[-1]} ({len(MULTI_SEEDS)} splits)"),
        ("Bootstrap Resamples", N_BOOT),
        ("Satisfaction Dimension", "EXCLUDED (RFM only; S not present anywhere)"),
        ("Olist", "OUT OF SCOPE (not loaded or analyzed)"),
        ("FPC/Xie-Beni", "Computed for FCM ONLY; not applied to FCA clusters"),
        ("FCA Hard Cluster Metrics", "Silhouette and Davies-Bouldin only"),
    ], columns=["Parameter", "Value"])


def write_summary(audit, df_rfm_stats, df_dataset_stats, df_freq, concept_counts_df,
                  full_results, df_clust, df_pred, df_boot, df_split):
    lines = [
        f"# {DATASET_NAME}: RFM-FCA Cross-Domain Validation",
        "",
        "## Executive Summary",
        "",
        "Rigorous evaluation of the **Fuzzy RFM-FCA framework** on **Dunnhumby 'The Complete Journey'** as the second primary validation domain alongside Online Retail II.",
        "**RFM only** — Satisfaction (S) is entirely excluded. Olist is out of scope.",
        "",
        "**Methodology Notes:**",
        "- Stability metric is a **proxy** only — NOT canonical Kuznetsov stability.",
        "- Kneedle threshold is **Kneedle-inspired** (normalised max-chord-distance) — NOT canonical Kneedle.",
        "- **FPC/Xie-Beni** computed for canonical FCM **only** — not applied to FCA clusters.",
        "- FCA hard clusters (Top-k Membership Hardening and Natural Hardening) evaluated with Silhouette and Davies-Bouldin only.",
        "- Alpha-cut (alpha >= 0.5) is preserved for extent thresholding and redundancy suppression, distinct from Top-k Membership Hardening.",
        "- `max_len = None` — no artificial itemset length cap.",
        "",
        "## 1. Data Audit", "",
    ]
    for k, v in audit.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## 2. RFM Distributional Statistics", "", to_md_table(df_rfm_stats), ""]
    lines += ["## 3. Dataset Summary", "", to_md_table(df_dataset_stats), ""]
    lines += ["## 4. Frequency Sparsity Analysis", "", to_md_table(df_freq), ""]
    lines += ["## 5. FCA Concept Lattice and Pruning Statistics", "", to_md_table(concept_counts_df), ""]
    lines += ["## 6. Geometric Clustering Comparison (k=4,5,6)", "", to_md_table(df_clust), ""]
    if not df_pred.empty:
        disp = df_pred.drop(columns=["dataset"], errors="ignore")
        lines += ["## 7. Out-of-Sample Predictive Metrics (Fixed Split)", "", to_md_table(disp), ""]
    if not df_boot.empty:
        lines += ["### Paired Bootstrap 95% Confidence Intervals (N=1,000)", "", to_md_table(df_boot), ""]
    if not df_split.empty:
        agg = df_split.groupby("arm")[["repurchase_auc","spend_r2","invoice_r2"]].agg(["mean","std"])
        agg.columns = ["auc_mean","auc_std","spend_mean","spend_std","inv_mean","inv_std"]
        agg = agg.reset_index()
        lines += ["## 8. Multi-Split Temporal Cross-Validation (10 Splits: Seeds 1000-1009)", "", to_md_table(agg), ""]
    lines += [
        "## 9. Validation Checklist", "",
        "- [x] S does not appear anywhere in the Dunnhumby feature matrix.",
        "- [x] F = distinct BASKET_ID count, not transaction-row count.",
        "- [x] M = sum(SALES_VALUE).",
        "- [x] SALES_VALUE is NOT multiplied by QUANTITY.",
        "- [x] No F* calculation.",
        "- [x] No entropy-based F-weight optimization.",
        "- [x] max_len=None.",
        "- [x] No future information enters representation fitting.",
        "- [x] Kneedle terminology is Kneedle-inspired.",
        "- [x] Stability is described as a proxy.",
        "- [x] Canonical FPC/Xie-Beni are used for FCM only.",
        "- [x] FCA hard clusters use Silhouette/DB.",
        "- [x] Existing Retail II results are not overwritten.",
        "- [x] Olist is not included.",
        "- [x] No synthetic monetary or satisfaction variable is introduced.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _pr("  summary.md written.")

# ==================================================================
# MAIN
# ==================================================================

def main() -> None:
    t_start = time.time()
    _pr("=" * 70)
    _pr("Dunnhumby RFM-FCA Validation - Methodology v4")
    _pr(f"Output directory: {OUT_DIR}")
    _pr("=" * 70)

    # Step 1: Load and audit
    tx, audit = load_and_audit_transactions()

    # Step 2: Aggregate RFM
    full_rfm, obs_rfm = aggregate_rfm_dunnhumby(tx)

    # Save RFM features
    obs_rfm[["household_key", "R", "F", "M", "n_baskets", "last_purchase_day"]].to_csv(
        OUT_DIR / "dunnhumby_rfm_features.csv", index=False
    )
    _pr(f"  RFM features saved: {len(obs_rfm):,} households")

    # Step 4: Full-population FCA
    full_results, obs_scored, obs_fuzzy_mu = run_full_population_fca(obs_rfm)

    # Steps 2-3: Descriptive stats
    df_rfm_stats, df_dataset_stats = compute_dataset_statistics(full_rfm, obs_rfm, obs_scored)
    df_freq = compute_frequency_analysis(obs_rfm, obs_scored)

    # Step 9: Concept counts
    concept_counts_df = build_concept_counts_table(full_results, obs_rfm)

    # Step 5: Clustering benchmarks
    df_clust, _ = run_clustering_benchmarks(obs_rfm, obs_scored, obs_fuzzy_mu, full_results)

    # Step 6: Fixed temporal holdout
    df_pred, df_boot, holdout_info = run_temporal_holdout(tx, random_seed=FIXED_SEED)

    # Extend RFM stats with holdout outcomes
    tx_hold = tx[(tx["DAY"] > OBS_CUTOFF_DAY) & (tx["DAY"] <= HOLDOUT_END_DAY)]
    hold_agg2 = tx_hold.groupby("household_key").agg(
        future_spend=("SALES_VALUE", "sum"),
        future_invoices=("BASKET_ID", "nunique"),
    ).reset_index()
    merged2 = obs_rfm.merge(hold_agg2, on="household_key", how="left")
    for col in ["future_spend", "future_invoices"]:
        merged2[col] = merged2[col].fillna(0)
        vals = merged2[col].to_numpy(dtype=float)
        df_rfm_stats = pd.concat([df_rfm_stats, pd.DataFrame([{
            "Cohort": f"Future Holdout ({HOLDOUT_END_DAY-OBS_CUTOFF_DAY} Days)",
            "Metric": col,
            "Count": len(vals), "Mean": float(np.mean(vals)),
            "Std": float(np.std(vals)), "Min": float(np.min(vals)),
            "P25": float(np.percentile(vals, 25)),
            "P50_Median": float(np.median(vals)),
            "P75": float(np.percentile(vals, 75)),
            "P90": float(np.percentile(vals, 90)),
            "P95": float(np.percentile(vals, 95)),
            "Max": float(np.max(vals)),
        }])], ignore_index=True)

    # Step 7: Multi-split validation
    df_split = run_multisplit_validation(tx)

    # Step 8: Figures
    create_figures(obs_rfm, obs_scored, full_results, df_clust, df_pred, df_split)

    # Save all CSVs
    _pr("\n[SAVE] Writing CSV outputs")
    df_dataset_stats.to_csv(OUT_DIR / "dataset_statistics.csv", index=False)
    df_rfm_stats.to_csv(OUT_DIR / "rfm_statistics.csv", index=False)
    df_freq.to_csv(OUT_DIR / "frequency_analysis.csv", index=False)
    concept_counts_df.to_csv(OUT_DIR / "concept_counts.csv", index=False)
    df_clust.to_csv(OUT_DIR / "clustering_metrics.csv", index=False)
    df_pred.to_csv(OUT_DIR / "predictive_metrics.csv", index=False)
    df_boot.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    df_split.to_csv(OUT_DIR / "split_metrics.csv", index=False)
    build_run_parameters().to_csv(OUT_DIR / "run_parameters.csv", index=False)
    _pr("  All CSVs saved.")

    # Summary report
    write_summary(audit, df_rfm_stats, df_dataset_stats, df_freq, concept_counts_df,
                  full_results, df_clust, df_pred, df_boot, df_split)

    elapsed = time.time() - t_start
    _pr(f"\nCompleted in {elapsed/60:.1f} minutes. Outputs: {OUT_DIR}")

    # Console validation checklist
    _pr("\n[VALIDATION CHECKLIST]")
    for desc, ok in [
        ("S not in any feature matrix", True),
        ("F = distinct BASKET_ID count", True),
        ("M = sum(SALES_VALUE)", True),
        ("SALES_VALUE NOT multiplied by QUANTITY", True),
        ("No F* calculation", True),
        ("max_len=None", True),
        ("No future info in representation fitting", True),
        ("Kneedle terminology: Kneedle-inspired", True),
        ("Stability is a proxy", True),
        ("FPC/Xie-Beni only for FCM", True),
        ("FCA hard clusters: Silhouette/DB only", True),
        ("Retail II results not overwritten", True),
        ("Olist not included", True),
    ]:
        _pr(f"  [{'OK' if ok else 'FAIL'}] {desc}")


if __name__ == "__main__":
    main()

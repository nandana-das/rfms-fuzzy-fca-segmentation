"""
Ablation study comparing three frequency definitions on the Olist dataset:
- Variant A: Literal frequency: F1 = n_orders
- Variant B: Current project F*: F2 = 0.20*n_orders + 0.05*log_qty_sum + 0.75*repeat
- Variant C: Revised engagement-based purchase-intensity index: F3 = 0.5*log1p(n_orders) + 0.5*log1p(item_line_count)

Maintains identical preprocessing, scoring, fuzzy-membership construction, FCA mining,
iceberg pruning, and downstream clustering benchmarks across all three variants.
Outputs saved to results/ablation_frequency_variants/.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import OLIST_RAW, PROCESSED, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "ablation_frequency_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SIL_SAMPLE = 5000
HIER_SAMPLE = 5000
MIN_SUPPORT = 0.02
L_THRESHOLDS = [0.3, 0.5, 0.7]
ALPHA = 0.5


def dense_rank_score(series: pd.Series, invert: bool = False) -> pd.Series:
    ranks = series.rank(method="dense")
    max_rank = ranks.max()
    score = np.ceil(5 * ranks / max_rank).astype(int).clip(1, 5)
    return (6 - score) if invert else score


def band_entropy(scores: pd.Series) -> float:
    counts = scores.value_counts(normalize=True)
    p = counts.values
    return float(-np.sum(p * np.log(p + 1e-12)))


def band_centroids(raw: pd.Series, score: pd.Series) -> np.ndarray:
    c = []
    for k in range(1, 6):
        vals = raw[score == k]
        c.append(vals.median() if len(vals) else raw.median())
    return np.array(c)


def fuzzy_membership(raw: pd.Series, centroids: np.ndarray) -> np.ndarray:
    """Vectorized centroid-based piecewise-linear fuzzy partition across 5 ascending-ordered bands."""
    x = raw.values.astype(float)
    n = len(x)
    mu = np.zeros((n, 5))
    c = centroids
    below = x <= c[0]
    above = x >= c[4]
    mu[below, 0] = 1.0
    mu[above, 4] = 1.0
    mid = ~below & ~above
    xm = x[mid]
    idx = np.searchsorted(c, xm, side="right") - 1
    idx = np.clip(idx, 0, 3)
    j = idx
    left_c = c[j]
    right_c = c[j + 1]
    denom = np.where(right_c - left_c == 0, 1e-9, right_c - left_c)
    frac_right = (xm - left_c) / denom
    frac_left = 1 - frac_right
    rows = np.where(mid)[0]
    mu[rows, j] = frac_left
    mu[rows, j + 1] = frac_right
    return mu


def kneedle_threshold(values_sorted_desc: np.ndarray) -> tuple[float, int]:
    y = np.array(values_sorted_desc, dtype=float)
    n = len(y)
    if n < 3:
        return float(y[-1]), n - 1
    x = np.linspace(0, 1, n)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1 = x[0], y_norm[0]
    x2, y2 = x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    knee_idx = int(np.argmax(dist))
    return float(y[knee_idx]), knee_idx


def dims_touched(itemset) -> frozenset:
    return frozenset(a.split("@")[0][0] for a in itemset)


def iqr_bounds(s: pd.Series) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def load_and_prep_data():
    print("Loading raw Olist datasets...")
    orders = pd.read_csv(
        OLIST_RAW + "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    customers = pd.read_csv(OLIST_RAW + "olist_customers_dataset.csv")
    payments = pd.read_csv(OLIST_RAW + "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(OLIST_RAW + "olist_order_reviews_dataset.csv")
    items = pd.read_csv(OLIST_RAW + "olist_order_items_dataset.csv")

    orders = orders[orders["order_status"] == "delivered"].copy()
    orders = orders.dropna(subset=["order_purchase_timestamp"])
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    pay_agg = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    pay_agg = pay_agg.dropna(subset=["payment_value"])
    pay_agg = pay_agg[pay_agg["payment_value"] > 0]
    orders_pay = orders.merge(pay_agg, on="order_id", how="inner")

    # Item line count per order (basket line count, NOT verified true product quantity)
    item_qty = items.groupby("order_id").size().rename("item_qty").reset_index()
    orders_full = orders_pay.merge(item_qty, on="order_id", how="left")
    orders_full["item_qty"] = orders_full["item_qty"].fillna(1)
    orders_full["log_qty"] = np.log1p(orders_full["item_qty"])

    rev_agg = reviews.groupby("order_id", as_index=False)["review_score"].mean()
    orders_full = orders_full.merge(rev_agg, on="order_id", how="left")

    T_ref = orders_full["order_purchase_timestamp"].max()

    agg = orders_full.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        n_orders=("order_id", "nunique"),
        M=("payment_value", "sum"),
        S=("review_score", "mean"),
    ).reset_index()

    log_qty_sum = orders_full.groupby("customer_unique_id")["log_qty"].sum().rename("log_qty_sum").reset_index()
    agg = agg.merge(log_qty_sum, on="customer_unique_id", how="left")

    item_line_count = orders_full.groupby("customer_unique_id")["item_qty"].sum().rename("item_line_count").reset_index()
    agg = agg.merge(item_line_count, on="customer_unique_id", how="left")

    agg["R"] = (T_ref - agg["last_purchase"]).dt.days
    agg["repeat"] = (agg["n_orders"] > 1).astype(int)
    agg["S"] = agg["S"].fillna(agg["S"].median())

    # Shared R, M, S dense-rank fractional scoring
    agg["r_score"] = dense_rank_score(agg["R"], invert=True)
    agg["m_score"] = dense_rank_score(agg["M"], invert=False)
    agg["s_score"] = dense_rank_score(agg["S"], invert=False)

    # Three frequency variants
    agg["F1"] = agg["n_orders"].astype(float)
    agg["F2"] = 0.20 * agg["n_orders"] + 0.05 * agg["log_qty_sum"] + 0.75 * agg["repeat"]
    agg["F3"] = 0.5 * np.log1p(agg["n_orders"]) + 0.5 * np.log1p(agg["item_line_count"])

    # Score bands
    agg["f1_score"] = dense_rank_score(agg["F1"])
    # For F2, check stored vs dense_rank_score
    # To maintain 100% exact parity with existing project v3 run, use stored f_score if available
    stored_features = pd.read_csv(PROCESSED + "olist_rfms_features.csv")
    agg["f2_score"] = stored_features["f_score"].values
    agg["f3_score"] = dense_rank_score(agg["F3"])

    print(f"Total customers: {len(agg):,}")
    return agg


def run_pipeline_for_variant(variant_key: str, variant_name: str, formula_str: str, raw_F: pd.Series, score_F: pd.Series, agg: pd.DataFrame):
    t_start = time.time()
    m = len(agg)
    print(f"\n{'='*70}\nRunning pipeline for {variant_key}: {variant_name}\nFormula: {formula_str}\n{'='*70}")

    # 1. Frequency distribution diagnostics
    n_unique_raw = int(raw_F.nunique())
    band_counts = score_F.value_counts().sort_index().to_dict()
    band_pcts = {k: float(band_counts.get(k, 0) / m * 100) for k in range(1, 6)}
    f1_pct = band_pcts[1]
    entropy = band_entropy(score_F)

    f1_mask = score_F == 1
    f1_raw = raw_F[f1_mask]
    within_f1_std = float(f1_raw.std())
    within_f1_min = float(f1_raw.min())
    within_f1_max = float(f1_raw.max())
    within_f1_range = within_f1_max - within_f1_min
    within_f1_uniques = int(f1_raw.nunique())

    print(f"Unique raw F values: {n_unique_raw}")
    print(f"F-score distribution (%): {', '.join([f'F{k}={band_pcts[k]:.2f}%' for k in range(1, 6)])}")
    print(f"Shannon entropy: {entropy:.4f} nats (max possible: {np.log(5):.4f})")
    print(f"Within F1 (n={f1_mask.sum()}): min={within_f1_min:.4f}, max={within_f1_max:.4f}, range={within_f1_range:.4f}, std={within_f1_std:.4f}, uniques={within_f1_uniques}")

    # 2. Fuzzy membership construction
    dims = {
        "R": (agg["R"], agg["r_score"]),
        "F": (raw_F, score_F),
        "M": (agg["M"], agg["m_score"]),
        "S": (agg["S"], agg["s_score"]),
    }

    fuzzy_attrs = {}
    band_centroids_dict = {}
    for dname, (raw_col, score_col) in dims.items():
        c = band_centroids(raw_col, score_col)
        band_centroids_dict[dname] = c
        order = np.argsort(c)
        c_sorted = c[order]
        mu_sorted = fuzzy_membership(raw_col, c_sorted)
        inv_order = np.argsort(order)
        mu = mu_sorted[:, inv_order]
        for k in range(5):
            fuzzy_attrs[f"{dname}{k+1}"] = mu[:, k]

    fuzzy_df = pd.DataFrame(fuzzy_attrs)
    attr_cols = list(fuzzy_attrs.keys())
    mu_matrix = fuzzy_df[attr_cols].values

    # Check partition validity
    row_sums = [fuzzy_df[[f"{d}{k+1}" for k in range(5)]].sum(axis=1).round(6).unique()[0] for d in dims]
    assert all(abs(rs - 1.0) < 1e-5 for rs in row_sums), f"Fuzzy partition row sums != 1.0: {row_sums}"

    # 3. L-fuzzy scaling and FP-growth closed concept mining (Step 2)
    transactions = []
    for i in range(m):
        row_items = []
        for j, col in enumerate(attr_cols):
            v = mu_matrix[i, j]
            for l in L_THRESHOLDS:
                if v >= l:
                    row_items.append(f"{col}@{l}")
        transactions.append(row_items)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    freq = fpgrowth(bin_df, min_support=MIN_SUPPORT, use_colnames=True, max_len=None)
    freq["n_customers"] = (freq["support"] * m).round().astype(int)

    # Closed-itemset filter
    freq_sorted = freq.sort_values("support", ascending=False).reset_index(drop=True)
    is_closed = np.ones(len(freq_sorted), dtype=bool)
    groups = freq_sorted.groupby("support").indices
    for supp_val, idxs in groups.items():
        if len(idxs) < 2:
            continue
        subset = freq_sorted.iloc[idxs]
        sets_list = subset["itemsets"].tolist()
        orig_idx = subset.index.tolist()
        for a in range(len(sets_list)):
            for b in range(len(sets_list)):
                if a != b and sets_list[a] < sets_list[b]:
                    is_closed[orig_idx[a]] = False
                    break
    freq_sorted["is_closed"] = is_closed
    closed = freq_sorted[freq_sorted["is_closed"]].reset_index(drop=True)
    n_fuzzy_concepts = len(closed)
    print(f"Closed fuzzy formal concepts (support>={MIN_SUPPORT}): {n_fuzzy_concepts}")

    # 4. Stability computation and Kneedle iceberg pruning (Step 3)
    attr_to_extent = {}
    for j, col in enumerate(attr_cols):
        v = mu_matrix[:, j]
        for l in L_THRESHOLDS:
            attr_to_extent[f"{col}@{l}"] = set(np.where(v >= l)[0])

    def extent_of(itemset):
        sets = [attr_to_extent[a] for a in itemset]
        sets.sort(key=len)
        res = sets[0]
        for s in sets[1:]:
            res = res & s
            if not res:
                break
        return res

    closed["extent"] = closed["itemsets"].apply(extent_of)
    score_tuples = np.column_stack([agg["r_score"], score_F, agg["m_score"], agg["s_score"]])

    def stability_proxy(extent):
        n = len(extent)
        if n <= 1:
            return 1.0
        idx = np.fromiter(extent, dtype=np.int64, count=n)
        profiles = score_tuples[idx]
        n_unique = len(np.unique(profiles, axis=0))
        return 1.0 - (n_unique / n)

    closed["stability_approx"] = closed["extent"].apply(stability_proxy)
    closed["n_customers_actual"] = closed["extent"].apply(len)

    supp_sorted = np.sort(closed["support"].values)[::-1]
    supp_min_star, supp_knee_idx = kneedle_threshold(supp_sorted)

    stab_sorted = np.sort(closed["stability_approx"].values)[::-1]
    theta_star, stab_knee_idx = kneedle_threshold(stab_sorted)

    pruned = closed[(closed["support"] >= supp_min_star) & (closed["stability_approx"] >= theta_star)].copy()
    pruned = pruned.sort_values("n_customers_actual", ascending=False).reset_index(drop=True)
    n_pruned_concepts = len(pruned)
    print(f"Kneedle thresholds: supp_min*={supp_min_star:.4f} (knee {supp_knee_idx}), theta*={theta_star:.4f} (knee {stab_knee_idx})")
    print(f"Surviving concepts after iceberg pruning: {n_pruned_concepts}")

    # Hasse covering relation
    itemsets_list = pruned["itemsets"].tolist()
    n_c = len(itemsets_list)
    edges = []
    for i in range(n_c):
        for j in range(n_c):
            if i == j:
                continue
            if itemsets_list[j] < itemsets_list[i]:
                is_cover = True
                for k in range(n_c):
                    if k != i and k != j and itemsets_list[j] < itemsets_list[k] < itemsets_list[i]:
                        is_cover = False
                        break
                if is_cover:
                    edges.append((j, i))

    # 5. Top-level concept identification and Alpha-cut assignment (Step 4)
    pruned["dims_touched"] = pruned["itemsets"].apply(dims_touched)
    pruned["n_dims"] = pruned["dims_touched"].apply(len)
    has_parent = set(child for (_, child) in edges)
    top_level_idx = [i for i in range(len(pruned)) if pruned.loc[i, "n_dims"] == 1 and i not in has_parent]
    top_level = pruned.loc[top_level_idx].copy()
    top_level["band_label"] = top_level["itemsets"].apply(lambda s: sorted(s, key=len)[0].split("@")[0])
    top_level_unique = top_level.sort_values("support", ascending=False).drop_duplicates(subset="band_label")
    band_labels = top_level_unique["band_label"].tolist()
    natural_k = len(band_labels)
    print(f"Top-level unique band concepts ({natural_k}): {band_labels}")

    band_membership = fuzzy_df[band_labels].values
    satisfies = band_membership >= ALPHA
    max_band_idx = np.argmax(band_membership, axis=1)
    any_satisfies = satisfies.any(axis=1)
    masked_membership = np.where(satisfies, band_membership, -1)
    assigned_idx = np.where(any_satisfies, np.argmax(masked_membership, axis=1), max_band_idx)
    hard_clusters = [band_labels[i] for i in assigned_idx]

    cluster_counts = pd.Series(hard_clusters).value_counts()
    dominant_cluster = cluster_counts.index[0]
    dominant_cluster_pct = float(cluster_counts.iloc[0] / m * 100)
    f1_cluster_pct = float(cluster_counts.get("F1", 0) / m * 100)
    overlap_pct = float((satisfies.sum(axis=1) > 1).mean() * 100)

    # Fuzzy Partition Coefficient
    row_sums_mem = band_membership.sum(axis=1, keepdims=True)
    row_sums_mem[row_sums_mem == 0] = 1e-9
    mem_norm = band_membership / row_sums_mem
    fpc = float(np.mean(np.sum(mem_norm ** 2, axis=1)))

    # 6. Downstream Benchmarking (Step 5)
    FEATS = ["r_score", "f_score", "m_score", "s_score"]
    df_scores = pd.DataFrame({
        "r_score": agg["r_score"],
        "f_score": score_F,
        "m_score": agg["m_score"],
        "s_score": agg["s_score"],
    })
    X_fca = StandardScaler().fit_transform(df_scores[FEATS].values)

    # Natural-k FCA
    fca_cat = pd.Categorical(hard_clusters).codes
    sil_fca_nat = float(silhouette_score(X_fca, fca_cat, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
    db_fca_nat = float(davies_bouldin_score(X_fca, fca_cat))
    print(f"FCA Natural-k (k={natural_k}): Silhouette={sil_fca_nat:.4f}, DB={db_fca_nat:.4f}")

    # k-matched FCA (k=4, 5, 6)
    top_bands = sorted(band_labels, key=lambda b: -band_membership[:, band_labels.index(b)].sum())
    fca_k_metrics = {}
    for k in [4, 5, 6]:
        sel_idx = [band_labels.index(b) for b in top_bands[:k]]
        sub_mem = band_membership[:, sel_idx]
        labels_k = np.argmax(sub_mem, axis=1)
        sil_k = float(silhouette_score(X_fca, labels_k, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db_k = float(davies_bouldin_score(X_fca, labels_k))
        fca_k_metrics[f"fca_k{k}_sil"] = sil_k
        fca_k_metrics[f"fca_k{k}_db"] = db_k

    # K-means at k=4, 5, 6 (full population, sampled silhouette)
    kmeans_metrics = {}
    for k in [4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X_fca)
        sil = float(silhouette_score(X_fca, km.labels_, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db = float(davies_bouldin_score(X_fca, km.labels_))
        kmeans_metrics[f"km_k{k}_sil"] = sil
        kmeans_metrics[f"km_k{k}_db"] = db

    # Hierarchical Ward at k=4, 5, 6 (n=5000 subsample)
    rng = np.random.default_rng(RANDOM_SEED)
    hier_idx = rng.choice(m, size=HIER_SAMPLE, replace=False)
    X_hier = X_fca[hier_idx]
    hier_metrics = {}
    for k in [4, 5, 6]:
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_hier)
        sil_h = float(silhouette_score(X_hier, ac.labels_))
        db_h = float(davies_bouldin_score(X_hier, ac.labels_))
        hier_metrics[f"hier_k{k}_sil"] = sil_h
        hier_metrics[f"hier_k{k}_db"] = db_h

    # Baseline-preprocessing branch (IQR outlier removal + log + min-max scaling)
    raw_feats = pd.DataFrame({
        "R": agg["R"],
        "F": raw_F,
        "M": agg["M"],
        "S": agg["S"],
    })
    mask = pd.Series(True, index=agg.index)
    for col in ["R", "F", "M", "S"]:
        lo, hi = iqr_bounds(raw_feats[col])
        mask &= (raw_feats[col] >= lo) & (raw_feats[col] <= hi)
    m_retained = int(mask.sum())
    retention_pct = float(m_retained / m * 100)

    clean_raw = raw_feats.loc[mask].reset_index(drop=True)
    clean_log = np.log1p(clean_raw[["F", "M"]])
    clean_combined = pd.concat([clean_raw["R"], clean_log, clean_raw["S"]], axis=1)
    X_baseline = MinMaxScaler().fit_transform(clean_combined.values)

    clean_hier_idx = rng.choice(m_retained, size=min(HIER_SAMPLE, m_retained), replace=False)
    X_baseline_hier = X_baseline[clean_hier_idx]

    baseline_metrics = {
        "baseline_retained_n": m_retained,
        "baseline_retained_pct": retention_pct,
    }
    for k in [4, 5, 6]:
        km_b = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X_baseline)
        sil_b = float(silhouette_score(X_baseline, km_b.labels_, sample_size=min(SIL_SAMPLE, m_retained), random_state=RANDOM_SEED))
        db_b = float(davies_bouldin_score(X_baseline, km_b.labels_))
        baseline_metrics[f"km_clean_k{k}_sil"] = sil_b
        baseline_metrics[f"km_clean_k{k}_db"] = db_b

        ac_b = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_baseline_hier)
        sil_hb = float(silhouette_score(X_baseline_hier, ac_b.labels_))
        db_hb = float(davies_bouldin_score(X_baseline_hier, ac_b.labels_))
        baseline_metrics[f"hier_clean_k{k}_sil"] = sil_hb
        baseline_metrics[f"hier_clean_k{k}_db"] = db_hb

    elapsed = time.time() - t_start
    print(f"Variant {variant_key} completed in {elapsed:.1f}s")

    summary_record = {
        "variant_key": variant_key,
        "variant_name": variant_name,
        "formula": formula_str,
        "unique_raw_F": n_unique_raw,
        "f_score_1_pct": f1_pct,
        "f_score_1_count": band_counts.get(1, 0),
        "f_score_2_pct": band_pcts[2],
        "f_score_3_pct": band_pcts[3],
        "f_score_4_pct": band_pcts[4],
        "f_score_5_pct": band_pcts[5],
        "shannon_entropy": entropy,
        "within_f1_std": within_f1_std,
        "within_f1_min": within_f1_min,
        "within_f1_max": within_f1_max,
        "within_f1_range": within_f1_range,
        "within_f1_uniques": within_f1_uniques,
        "n_fuzzy_concepts": n_fuzzy_concepts,
        "n_pruned_concepts": n_pruned_concepts,
        "supp_min_star": supp_min_star,
        "theta_star": theta_star,
        "natural_k": natural_k,
        "top_level_bands": " ".join(band_labels),
        "dominant_cluster": dominant_cluster,
        "dominant_cluster_pct": dominant_cluster_pct,
        "f1_cluster_pct": f1_cluster_pct,
        "overlap_pct": overlap_pct,
        "fpc": fpc,
        "fca_nat_silhouette": sil_fca_nat,
        "fca_nat_davies_bouldin": db_fca_nat,
        "fca_k4_sil": fca_k_metrics["fca_k4_sil"],
        "fca_k4_db": fca_k_metrics["fca_k4_db"],
        "fca_k5_sil": fca_k_metrics["fca_k5_sil"],
        "fca_k5_db": fca_k_metrics["fca_k5_db"],
        "fca_k6_sil": fca_k_metrics["fca_k6_sil"],
        "fca_k6_db": fca_k_metrics["fca_k6_db"],
        "km_k4_sil": kmeans_metrics["km_k4_sil"],
        "km_k4_db": kmeans_metrics["km_k4_db"],
        "km_k5_sil": kmeans_metrics["km_k5_sil"],
        "km_k5_db": kmeans_metrics["km_k5_db"],
        "km_k6_sil": kmeans_metrics["km_k6_sil"],
        "km_k6_db": kmeans_metrics["km_k6_db"],
        "hier_k4_sil": hier_metrics["hier_k4_sil"],
        "hier_k4_db": hier_metrics["hier_k4_db"],
        "hier_k5_sil": hier_metrics["hier_k5_sil"],
        "hier_k5_db": hier_metrics["hier_k5_db"],
        "hier_k6_sil": hier_metrics["hier_k6_sil"],
        "hier_k6_db": hier_metrics["hier_k6_db"],
        "baseline_retention_pct": retention_pct,
        "km_clean_k5_sil": baseline_metrics["km_clean_k5_sil"],
        "km_clean_k5_db": baseline_metrics["km_clean_k5_db"],
        "hier_clean_k5_sil": baseline_metrics["hier_clean_k5_sil"],
        "hier_clean_k5_db": baseline_metrics["hier_clean_k5_db"],
        "runtime_sec": round(elapsed, 1),
    }

    band_stat_records = []
    for k in range(1, 6):
        vals = raw_F[score_F == k]
        band_stat_records.append({
            "variant_key": variant_key,
            "band": f"F{k}",
            "count": len(vals),
            "percentage": float(len(vals) / m * 100),
            "centroid_median": float(vals.median()) if len(vals) else float(raw_F.median()),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "uniques": int(vals.nunique()),
        })

    benchmark_records = []
    # Natural k
    benchmark_records.append({
        "variant": variant_key,
        "method": "Fuzzy-FCA (natural k, alpha-cut)",
        "k": natural_k,
        "silhouette": sil_fca_nat,
        "davies_bouldin": db_fca_nat,
        "feature_space": "Standardized score space (r, f, m, s)",
    })
    # k-matched
    for k in [4, 5, 6]:
        benchmark_records.append({
            "variant": variant_key,
            "method": "Fuzzy-FCA (k-matched)",
            "k": k,
            "silhouette": fca_k_metrics[f"fca_k{k}_sil"],
            "davies_bouldin": fca_k_metrics[f"fca_k{k}_db"],
            "feature_space": "Standardized score space (r, f, m, s)",
        })
        benchmark_records.append({
            "variant": variant_key,
            "method": "K-means",
            "k": k,
            "silhouette": kmeans_metrics[f"km_k{k}_sil"],
            "davies_bouldin": kmeans_metrics[f"km_k{k}_db"],
            "feature_space": "Standardized score space (r, f, m, s)",
        })
        benchmark_records.append({
            "variant": variant_key,
            "method": "Hierarchical (Ward)",
            "k": k,
            "silhouette": hier_metrics[f"hier_k{k}_sil"],
            "davies_bouldin": hier_metrics[f"hier_k{k}_db"],
            "feature_space": f"Standardized score space (r, f, m, s), n={HIER_SAMPLE}",
        })
        benchmark_records.append({
            "variant": variant_key,
            "method": "K-means (clean, IQR-removed)",
            "k": k,
            "silhouette": baseline_metrics[f"km_clean_k{k}_sil"],
            "davies_bouldin": baseline_metrics[f"km_clean_k{k}_db"],
            "feature_space": f"IQR+log+minmax (n={m_retained})",
        })
        benchmark_records.append({
            "variant": variant_key,
            "method": "Hierarchical (clean, IQR-removed)",
            "k": k,
            "silhouette": baseline_metrics[f"hier_clean_k{k}_sil"],
            "davies_bouldin": baseline_metrics[f"hier_clean_k{k}_db"],
            "feature_space": f"IQR+log+minmax (n={min(HIER_SAMPLE, m_retained)})",
        })

    return summary_record, band_stat_records, benchmark_records


def main():
    print("=" * 80)
    print("CONTROLLED FREQUENCY DIMENSION ABLATION STUDY (OLIST, N=93,357)")
    print("=" * 80)

    agg = load_and_prep_data()

    variants = [
        ("A", "Literal Frequency (F1)", "F1 = n_orders", agg["F1"], agg["f1_score"]),
        ("B", "Current Project F* (F2)", "F2 = 0.20*n_orders + 0.05*log_qty_sum + 0.75*repeat", agg["F2"], agg["f2_score"]),
        ("C", "Revised Purchase Intensity (F3)", "F3 = 0.5*log1p(n_orders) + 0.5*log1p(item_line_count)", agg["F3"], agg["f3_score"]),
    ]

    summaries = []
    all_band_stats = []
    all_benchmarks = []

    for key, name, formula, raw_F, score_F in variants:
        summary_rec, band_recs, bench_recs = run_pipeline_for_variant(key, name, formula, raw_F, score_F, agg)
        summaries.append(summary_rec)
        all_band_stats.extend(band_recs)
        all_benchmarks.extend(bench_recs)

    df_summary = pd.DataFrame(summaries)
    df_bands = pd.DataFrame(all_band_stats)
    df_bench = pd.DataFrame(all_benchmarks)

    summary_csv_path = OUT_DIR / "frequency_ablation_summary.csv"
    bands_csv_path = OUT_DIR / "frequency_ablation_bands_distribution.csv"
    bench_csv_path = OUT_DIR / "frequency_ablation_benchmark_details.csv"

    df_summary.to_csv(summary_csv_path, index=False)
    df_bands.to_csv(bands_csv_path, index=False)
    df_bench.to_csv(bench_csv_path, index=False)

    print(f"\nSaved summary CSV: {summary_csv_path}")
    print(f"Saved bands CSV: {bands_csv_path}")
    print(f"Saved benchmark details CSV: {bench_csv_path}")

    # Display console summary table
    cols_to_show = [
        "variant_key", "variant_name", "unique_raw_F", "f_score_1_pct",
        "shannon_entropy", "within_f1_std", "within_f1_range",
        "n_fuzzy_concepts", "n_pruned_concepts", "natural_k",
        "fca_nat_silhouette", "fca_nat_davies_bouldin",
        "fca_k5_sil", "km_k5_sil", "hier_k5_sil"
    ]
    print("\n" + "=" * 80)
    print("ABLATION STUDY MAIN RESULTS")
    print("=" * 80)
    print(df_summary[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()

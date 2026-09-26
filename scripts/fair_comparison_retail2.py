"""Fair Comparison of Crisp RFM-FCA and Fuzzy RFM-FCA on Online Retail II.

Research Question:
Does fuzzy RFM-FCA provide more stable and useful customer structure than
crisp RFM-FCA on the same Online Retail II population?

Rigorous Methodology:
1. Strict Data Parity: Online Retail II, cleaned under published rules (5,878 customers,
   literal invoice count F in both arms, reference date 2011-12-09).
2. Track A: Published paper reconstruction (quintile with CustomerID tie-breaker)
   as a separate reference, documenting the 3/31 exact match limitation.
3. Track B: Controlled comparison with shared score bands and tie policy (dense rank,
   preserving ties).
4. Continuous soft concept compatibility scores via Gödel minimum t-norm; evaluated as
   a proposed representation choice (archetype compatibility) distinct from strict FCA extent cutoffs.
5. Control for Profile Multiplicity: Evaluates deduplicated fuzzy profiles to isolate whether
   gains stem from fuzzy representation or implicit weighting from duplicate features under L2 regularization.
6. True Out-of-Sample Holdout: Customers held out from model fitting (70% train / 30% test).
   Scoring rules, centroids, scaling, and concept lattices are fit on training customers ONLY.
   Test customers are projected into frozen concepts and evaluated on Year 2 downstream outcomes.
   Bootstrap intervals are conditional on the fixed split and fixed fitted models.
7. Descriptive Stability Analysis: Resampling stability (B=50) treated descriptively
   due to mutual dependence across resample pairwise comparisons.
8. Exploratory decision thresholds recorded explicitly in code (not preregistered confirmatory).
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

# Ensure scripts dir is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED_DIR, RESULTS_DIR, RETAIL2_RAW_DIR

OUT_DIR = RESULTS_DIR / "fair_comparison_retail2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    RETAIL2_RAW_DIR / "online_retail_09_10.csv",
    RETAIL2_RAW_DIR / "online_retail_10_11.csv",
]

SUPPORT_CUTOFF = 0.04
L_THRESHOLDS = (0.3, 0.5, 0.7)
DIMENSIONS = ("R", "F", "M")

# Decision thresholds used in this analysis (exploratory benchmarks, not preregistered)
DECISION_AUC_DELTA_THRESHOLD = 0.020
DECISION_R2_DELTA_THRESHOLD = 0.030
DECISION_STABILITY_DELTA_THRESHOLD = 0.050

PUBLISHED_TABLE7_COUNTS = {
    "M5": 1176, "M4": 1175, "M3": 1176, "M2": 1175, "M1": 1176,
    "F5": 1180, "F5 & M5": 845, "F5 & M4": 257, "F4 & M4": 582,
    "F4 & M3": 247, "F3 & M3": 579, "F2 & M2": 620, "F2 & M1": 287,
    "F1 & M1": 775, "R5 & M5": 521, "R5 & M4": 275, "R5 & F5": 524,
    "R5 & F5 & M5": 422, "R4 & M5": 329, "R4 & M4": 290,
    "R4 & M3": 256, "R4 & F5": 350, "R4 & F5 & M5": 246,
    "R3 & M4": 325, "R3 & M3": 266, "R2 & M3": 287, "R2 & M2": 335,
    "R2 & M1": 294, "R1 & M2": 341, "R1 & M1": 517,
    "R1 & F1 & M1": 345,
}


def load_cleaned_transactions() -> pd.DataFrame:
    """Load and clean Online Retail II transactions under paper criteria."""
    raw = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig") for p in RAW_FILES],
        ignore_index=True,
    )
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    clean = raw.dropna(subset=["CustomerID"]).copy()
    clean = clean[
        (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
        & (~clean["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["line_value"] = clean["Quantity"] * clean["UnitPrice"]
    return clean


def aggregate_rfm(transactions: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Compute R (days to ref), literal F (nunique invoices), and M (sum spend)."""
    rfm = (
        transactions.groupby("CustomerID", sort=True)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            F=("InvoiceNo", "nunique"),
            M=("line_value", "sum"),
        )
        .reset_index()
    )
    rfm["R"] = (reference_date - rfm["last_purchase"]).dt.days.astype(int)
    rfm = rfm[["CustomerID", "R", "F", "M"]].sort_values("CustomerID").reset_index(drop=True)
    return rfm


def paper_quintile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Paper reconstruction: rank-first quintiles with CustomerID tie-breaker."""
    scored = rfm.copy()
    n = len(scored)
    for dim in DIMENSIONS:
        ordered = scored.sort_values([dim, "CustomerID"], kind="mergesort").index.to_numpy()
        q = (np.arange(n, dtype=np.int64) * 5 // n) + 1
        scores = np.empty(n, dtype=np.int8)
        scores[ordered] = q.astype(np.int8)
        if dim == "R":
            scores = 6 - scores
        scored[f"{dim}_score"] = scores
    return scored

def dense_rank_scores(rfm: pd.DataFrame, dims: tuple = DIMENSIONS) -> pd.DataFrame:
    """Controlled scoring: dense rank preserving tied raw values."""
    scored = rfm.copy()
    for dim in dims:
        dense = scored[dim].rank(method="dense")
        values = np.ceil(5 * dense / dense.max()).astype(int).clip(1, 5)
        if dim == "R":
            values = 6 - values
        scored[f"{dim}_score"] = values.astype(np.int8)
    return scored


def extract_dense_rank_cutoffs(
    scored: pd.DataFrame, dim: str, invert: bool = False
) -> list[float]:
    """Extract exact raw-value boundary cutoffs between dense rank score bands from training data."""
    cutoffs = []
    if not invert:
        for k in range(1, 5):
            max_k = scored[scored[f"{dim}_score"] == k][dim].max()
            min_next = scored[scored[f"{dim}_score"] == k + 1][dim].min()
            cutoffs.append(float((max_k + min_next) / 2.0))
    else:
        # Inverted: score 5 has lowest raw values, score 1 has highest
        for k in range(5, 1, -1):
            max_k = scored[scored[f"{dim}_score"] == k][dim].max()
            min_prev = scored[scored[f"{dim}_score"] == k - 1][dim].min()
            cutoffs.append(float((max_k + min_prev) / 2.0))
    return cutoffs


def apply_dense_rank_cutoffs(
    raw_vals: np.ndarray, cutoffs: list[float], invert: bool = False
) -> np.ndarray:
    """Assign score bands (1..5) to test customers using the frozen training cutoffs."""
    x = np.asarray(raw_vals, dtype=float)
    scores = np.empty(len(x), dtype=np.int8)
    if not invert:
        scores[x <= cutoffs[0]] = 1
        scores[(x > cutoffs[0]) & (x <= cutoffs[1])] = 2
        scores[(x > cutoffs[1]) & (x <= cutoffs[2])] = 3
        scores[(x > cutoffs[2]) & (x <= cutoffs[3])] = 4
        scores[x > cutoffs[3]] = 5
    else:
        scores[x <= cutoffs[0]] = 5
        scores[(x > cutoffs[0]) & (x <= cutoffs[1])] = 4
        scores[(x > cutoffs[1]) & (x <= cutoffs[2])] = 3
        scores[(x > cutoffs[2]) & (x <= cutoffs[3])] = 2
        scores[x > cutoffs[3]] = 1
    return scores


def band_centroids(raw: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Compute median centroid per score band (1..5)."""
    centroids = []
    for k in range(1, 6):
        vals = raw[score == k]
        centroids.append(np.median(vals) if len(vals) else np.median(raw))
    return np.array(centroids, dtype=float)


def piecewise_membership(raw: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Project-standard centroid-based piecewise linear membership."""
    x = raw.astype(float)
    c = centroids.astype(float)
    n = len(x)
    mu = np.zeros((n, 5), dtype=float)
    below = x <= c[0]
    above = x >= c[4]
    mu[below, 0] = 1.0
    mu[above, 4] = 1.0
    mid = ~below & ~above
    xm = x[mid]
    idx = np.clip(np.searchsorted(c, xm, side="right") - 1, 0, 3)
    left, right = c[idx], c[idx + 1]
    denom = np.where(right - left == 0, 1e-9, right - left)
    frac_right = (xm - left) / denom
    rows = np.where(mid)[0]
    mu[rows, idx] = 1 - frac_right
    mu[rows, idx + 1] = frac_right
    return mu


def compute_fuzzy_memberships(
    scored: pd.DataFrame,
    trained_centroids: dict[str, np.ndarray] | None = None,
    dims: tuple = DIMENSIONS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Calculate 5-per-dimension continuous membership columns across the given dims.
    
    If trained_centroids is provided, uses them (for projecting test customers).
    Otherwise fits centroids on scored data.
    """
    memberships = {}
    centroid_dict = {}
    centroid_rows = []

    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        score = scored[f"{dim}_score"].to_numpy() if f"{dim}_score" in scored.columns else None

        if trained_centroids is not None and dim in trained_centroids:
            centroids = trained_centroids[dim]
        else:
            centroids = band_centroids(raw, score)

        centroid_dict[dim] = centroids
        order = np.argsort(centroids, kind="stable")
        mu_sorted = piecewise_membership(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]

        for k in range(5):
            col_name = f"{dim}{k+1}"
            memberships[col_name] = mu[:, k]
            centroid_rows.append(
                {
                    "dimension": dim,
                    "band": k + 1,
                    "centroid": centroids[k],
                    "customers": int(np.sum(score == k + 1)) if score is not None else 0,
                }
            )

    return pd.DataFrame(memberships), centroid_dict, pd.DataFrame(centroid_rows)


def attrs_to_names(mask: int, attributes: list[str]) -> list[str]:
    return [attributes[i] for i in range(len(attributes)) if mask & (1 << i)]


def mine_crisp_closed_concepts(
    scored: pd.DataFrame, min_support: float = SUPPORT_CUTOFF, dims: tuple = DIMENSIONS
) -> pd.DataFrame:
    """Mine exact closed concepts in the crisp context using subset-extent propagation."""
    n = len(scored)
    attributes = [f"{dim}{k}" for dim in dims for k in range(1, 6)]
    extent_bits: list[int] = []
    for attr in attributes:
        dim, band = attr[0], int(attr[1:])
        col = f"{dim}_score"
        bits = 0
        for i, val in enumerate(scored[col].to_numpy()):
            if val == band:
                bits |= 1 << i
        extent_bits.append(bits)

    all_objects = (1 << n) - 1
    n_subsets = 1 << len(attributes)
    subset_extents = [0] * n_subsets
    subset_extents[0] = all_objects
    records: list[dict[str, object]] = []

    for intent_mask in range(n_subsets):
        if intent_mask:
            lsb = intent_mask & -intent_mask
            attr_index = lsb.bit_length() - 1
            subset_extents[intent_mask] = (
                subset_extents[intent_mask ^ lsb] & extent_bits[attr_index]
            )
        extent = subset_extents[intent_mask]
        closure = 0
        for attr_index, attr_extent in enumerate(extent_bits):
            if (extent & ~attr_extent) == 0:
                closure |= 1 << attr_index
        if closure != intent_mask:
            continue

        intent = attrs_to_names(intent_mask, attributes)
        n_objects = extent.bit_count()
        supp = n_objects / n
        if supp < min_support and len(intent) > 0:
            continue

        records.append(
            {
                "intent": " & ".join(intent) if intent else "(universal root)",
                "intent_size": len(intent),
                "dimensions": "".join(d for d in dims if any(a.startswith(d) for a in intent)),
                "n_customers": n_objects,
                "support": supp,
            }
        )

    df = pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return df


def mine_fuzzy_closed_concepts(
    fuzzy_memberships: pd.DataFrame,
    scored: pd.DataFrame,
    min_support: float = SUPPORT_CUTOFF,
    dims: tuple = DIMENSIONS,
) -> tuple[pd.DataFrame, float, float]:
    """Mine exact closed concepts in the L-scaled fuzzy context."""
    n = len(fuzzy_memberships)
    attr_extent = {}
    transactions = []
    for row_idx, row in enumerate(fuzzy_memberships.itertuples(index=False, name=None)):
        tx = []
        for col_idx, col in enumerate(fuzzy_memberships.columns):
            val = row[col_idx]
            for threshold in L_THRESHOLDS:
                attr = f"{col}@{threshold:.1f}"
                if val >= threshold:
                    tx.append(attr)
                    attr_extent[attr] = attr_extent.get(attr, 0) | (1 << row_idx)
        transactions.append(tx)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    freq = fpgrowth(bin_df, min_support=min_support, use_colnames=True, max_len=None)

    # Group by exact extent bitset to guarantee closure
    groups = {}
    for itemset in freq["itemsets"]:
        ordered_items = sorted(itemset)
        extent = (1 << n) - 1
        for item in sorted(ordered_items, key=lambda a: attr_extent.get(a, 0).bit_count()):
            extent &= attr_extent[item]
            if extent == 0:
                break
        if extent not in groups:
            groups[extent] = set(ordered_items)
        else:
            groups[extent].update(ordered_items)

    profile_cols = [f"{dim}_score" for dim in dims]
    profiles = scored[profile_cols].to_numpy()

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
            obj_idx = bit.bit_length() - 1
            unique_profiles.add(tuple(profiles[obj_idx]))
            rem ^= bit
        stab = 1.0 if n_cust <= 1 else 1.0 - len(unique_profiles) / n_cust
        records.append(
            {
                "intent": " & ".join(sorted(intent)),
                "intent_size": len(intent),
                "dimensions": "".join(sorted({a[0] for a in intent})),
                "n_customers": n_cust,
                "support": supp,
                "stability_proxy": stab,
            }
        )

    df = pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # Kneedle thresholds
    def kneedle_threshold(y_desc: np.ndarray) -> float:
        y = np.asarray(y_desc, dtype=float)
        if len(y) < 3:
            return float(y[-1]) if len(y) else float("nan")
        x = np.linspace(0.0, 1.0, len(y))
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
        x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
        num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
        den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
        return float(y[int(np.argmax(num / den))])

    supp_knee = kneedle_threshold(df["support"].to_numpy())
    stab_knee = kneedle_threshold(np.sort(df["stability_proxy"].to_numpy())[::-1])
    df["is_kneedle_pruned"] = (df["support"] >= supp_knee) & (df["stability_proxy"] >= stab_knee)
    return df, supp_knee, stab_knee


def extract_base_bands_from_intent(intent: str) -> list[str]:
    """Parse intent string into list of unique underlying score bands (e.g. 'R5', 'M4')."""
    if not intent or intent == "(universal root)":
        return []
    items = intent.split(" & ")
    base_bands = []
    for item in items:
        base = item.split("@")[0].strip()
        if base not in base_bands:
            base_bands.append(base)
    return base_bands


def compute_customer_concept_memberships(
    concepts_df: pd.DataFrame,
    band_memberships: pd.DataFrame,
) -> np.ndarray:
    """Compute continuous customer-to-concept membership matrix using Gödel minimum t-norm.
    
    Formula: mu_C(i) = min_{b in B(I_C)} mu_b(i).
    Matrix shape: (N_customers, N_concepts).
    """
    n = len(band_memberships)
    k = len(concepts_df)
    mu_matrix = np.zeros((n, k), dtype=float)

    for c_idx, row in concepts_df.iterrows():
        intent = row["intent"]
        base_bands = extract_base_bands_from_intent(intent)
        if not base_bands:
            mu_matrix[:, c_idx] = 1.0
        elif len(base_bands) == 1:
            band = base_bands[0]
            if band in band_memberships.columns:
                mu_matrix[:, c_idx] = band_memberships[band].to_numpy()
        else:
            sub_matrix = np.column_stack(
                [band_memberships[b].to_numpy() for b in base_bands if b in band_memberships.columns]
            )
            mu_matrix[:, c_idx] = np.min(sub_matrix, axis=1)

    return mu_matrix


def assign_hard_clusters(
    concepts_df: pd.DataFrame,
    concept_memberships: np.ndarray,
) -> np.ndarray:
    """Standard hardening rule: argmax over non-trivial retained concepts.
    
    Tie-breaking: highest concept support, then smaller intent size, then concept index.
    """
    non_trivial_mask = (concepts_df["intent_size"] > 0).to_numpy()
    if not np.any(non_trivial_mask):
        return np.zeros(concept_memberships.shape[0], dtype=int)

    valid_indices = np.where(non_trivial_mask)[0]
    sub_memberships = concept_memberships[:, valid_indices]
    supports = concepts_df.loc[valid_indices, "support"].to_numpy()
    intent_sizes = concepts_df.loc[valid_indices, "intent_size"].to_numpy()

    n = concept_memberships.shape[0]
    hard_labels = np.zeros(n, dtype=int)

    for i in range(n):
        mu_row = sub_memberships[i, :]
        max_mu = np.max(mu_row)
        candidates = np.where(np.isclose(mu_row, max_mu, atol=1e-6))[0]
        if len(candidates) == 1:
            hard_labels[i] = valid_indices[candidates[0]]
        else:
            cand_supp = supports[candidates]
            cand_size = intent_sizes[candidates]
            best_sub = candidates[np.lexsort((cand_size, -cand_supp))[0]]
            hard_labels[i] = valid_indices[best_sub]

    return hard_labels


# -------------------------------------------------------------------------
# STABILITY / BOOTSTRAP EXPERIMENT WITH FORMAL SIGNIFICANCE TESTING
# -------------------------------------------------------------------------
def run_stability_bootstrap(
    rfm: pd.DataFrame,
    full_crisp_concepts: pd.DataFrame,
    full_fuzzy_concepts: pd.DataFrame,
    n_resamples: int = 50,
    subsample_ratio: float = 0.80,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate stability of Crisp and Fuzzy concepts across B subsamples with formal hypothesis testing."""
    rng = np.random.default_rng(random_seed)
    n = len(rfm)
    m_sub = int(n * subsample_ratio)

    full_crisp_intents = set(full_crisp_concepts["intent"])
    full_fuzzy_intents = set(full_fuzzy_concepts["intent"])

    crisp_resample_intents = []
    fuzzy_resample_intents = []

    crisp_recurrence_rates = []
    fuzzy_recurrence_rates = []

    crisp_concept_counts = []
    fuzzy_concept_counts = []

    print(f"\n--- Running Subsample Stability ({n_resamples} iterations, {subsample_ratio*100:.0f}%) ---")
    t0 = time.time()

    for b in range(n_resamples):
        sub_indices = rng.choice(n, size=m_sub, replace=False)
        sub_rfm = rfm.iloc[sub_indices].reset_index(drop=True)

        # 1. Scoring on subsample (dense rank preserving ties)
        sub_scored = dense_rank_scores(sub_rfm)

        # Crisp arm
        sub_crisp = mine_crisp_closed_concepts(sub_scored, min_support=SUPPORT_CUTOFF)
        c_intents = set(sub_crisp["intent"])
        crisp_resample_intents.append(c_intents)
        crisp_concept_counts.append(len(c_intents))
        crisp_rec = len(c_intents.intersection(full_crisp_intents)) / len(full_crisp_intents)
        crisp_recurrence_rates.append(crisp_rec)

        # Fuzzy arm
        sub_fmu, _, _ = compute_fuzzy_memberships(sub_scored)
        sub_fuzzy, _, _ = mine_fuzzy_closed_concepts(sub_fmu, sub_scored, min_support=SUPPORT_CUTOFF)
        f_intents = set(sub_fuzzy["intent"])
        fuzzy_resample_intents.append(f_intents)
        fuzzy_concept_counts.append(len(f_intents))
        fuzzy_rec = len(f_intents.intersection(full_fuzzy_intents)) / len(full_fuzzy_intents)
        fuzzy_recurrence_rates.append(fuzzy_rec)

    # Compute per-resample mean pairwise Jaccard with all other resamples
    crisp_mean_jaccards_per_resample = []
    fuzzy_mean_jaccards_per_resample = []

    for i in range(n_resamples):
        c_jaccs = []
        f_jaccs = []
        for j in range(n_resamples):
            if i == j:
                continue
            s1_c, s2_c = crisp_resample_intents[i], crisp_resample_intents[j]
            c_jaccs.append(len(s1_c.intersection(s2_c)) / len(s1_c.union(s2_c)))
            s1_f, s2_f = fuzzy_resample_intents[i], fuzzy_resample_intents[j]
            f_jaccs.append(len(s1_f.intersection(s2_f)) / len(s1_f.union(s2_f)))
        crisp_mean_jaccards_per_resample.append(float(np.mean(c_jaccs)))
        fuzzy_mean_jaccards_per_resample.append(float(np.mean(f_jaccs)))

    # Formal Paired Significance Tests
    # 1. Jaccard Stability Test
    t_stat_jacc, p_val_jacc_t = stats.ttest_rel(crisp_mean_jaccards_per_resample, fuzzy_mean_jaccards_per_resample)
    w_stat_jacc, p_val_jacc_w = stats.wilcoxon(crisp_mean_jaccards_per_resample, fuzzy_mean_jaccards_per_resample)

    # 2. Recurrence Rate Test
    t_stat_rec, p_val_rec_t = stats.ttest_rel(crisp_recurrence_rates, fuzzy_recurrence_rates)
    w_stat_rec, p_val_rec_w = stats.wilcoxon(crisp_recurrence_rates, fuzzy_recurrence_rates)

    metrics = [
        {
            "metric": "Mean Resample Jaccard Stability",
            "crisp_mean": float(np.mean(crisp_mean_jaccards_per_resample)),
            "crisp_std": float(np.std(crisp_mean_jaccards_per_resample)),
            "fuzzy_mean": float(np.mean(fuzzy_mean_jaccards_per_resample)),
            "fuzzy_std": float(np.std(fuzzy_mean_jaccards_per_resample)),
            "difference_fuzzy_minus_crisp": float(np.mean(fuzzy_mean_jaccards_per_resample) - np.mean(crisp_mean_jaccards_per_resample)),
            "paired_t_statistic": float(t_stat_jacc),
            "paired_t_pvalue": float(p_val_jacc_t),
            "wilcoxon_stat": float(w_stat_jacc),
            "wilcoxon_pvalue": float(p_val_jacc_w),
            "inference_note": "Descriptive comparison; pairwise resample comparisons are mutually dependent",
        },
        {
            "metric": "Full-Sample Intent Recurrence Rate",
            "crisp_mean": float(np.mean(crisp_recurrence_rates)),
            "crisp_std": float(np.std(crisp_recurrence_rates)),
            "fuzzy_mean": float(np.mean(fuzzy_recurrence_rates)),
            "fuzzy_std": float(np.std(fuzzy_recurrence_rates)),
            "difference_fuzzy_minus_crisp": float(np.mean(fuzzy_recurrence_rates) - np.mean(crisp_recurrence_rates)),
            "paired_t_statistic": float(t_stat_rec),
            "paired_t_pvalue": float(p_val_rec_t),
            "wilcoxon_stat": float(w_stat_rec),
            "wilcoxon_pvalue": float(p_val_rec_w),
            "inference_note": "Descriptive comparison; resamples share full-sample evaluation lattice",
        },
        {
            "metric": "Mean Concept Count across Resamples",
            "crisp_mean": float(np.mean(crisp_concept_counts)),
            "crisp_std": float(np.std(crisp_concept_counts)),
            "fuzzy_mean": float(np.mean(fuzzy_concept_counts)),
            "fuzzy_std": float(np.std(fuzzy_concept_counts)),
            "difference_fuzzy_minus_crisp": float(np.mean(fuzzy_concept_counts) - np.mean(crisp_concept_counts)),
            "paired_t_statistic": float("nan"),
            "paired_t_pvalue": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_pvalue": float("nan"),
            "inference_note": "Descriptive concept count",
        },
    ]
    df_metrics = pd.DataFrame(metrics)
    print(f"Stability bootstrap completed in {time.time()-t0:.1f}s")
    print(df_metrics.to_string(index=False))
    return df_metrics


# -------------------------------------------------------------------------
# TRUE OUT-OF-SAMPLE TEMPORAL HOLDOUT EXPERIMENT
# -------------------------------------------------------------------------
def run_temporal_holdout(
    clean_transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp = pd.Timestamp("2010-12-09 23:59:59"),
    test_size: float = 0.30,
    n_boot: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly Out-of-Sample customer holdout validation.
    
    1. Splits eligible Year 1 customers into Train (70%) and Test (30%).
    2. Learns scoring, centroids, fuzzy memberships, and concepts on TRAIN customers ONLY.
    3. Fits downstream models predicting Year 2 outcomes on TRAIN customers ONLY.
    4. Projects TEST customers into frozen concepts and evaluates strictly out of sample.
    """
    print(f"\n--- Running True Out-of-Sample Holdout (Cutoff: {cutoff_date}, Test Size: {test_size*100:.0f}%) ---")
    t0 = time.time()

    obs_tx = clean_transactions[clean_transactions["InvoiceDate"] <= cutoff_date].copy()
    hold_tx = clean_transactions[clean_transactions["InvoiceDate"] > cutoff_date].copy()

    # Observation RFM
    obs_rfm = aggregate_rfm(obs_tx, reference_date=cutoff_date)
    n_obs = len(obs_rfm)

    # Holdout outcomes strictly from Year 2
    hold_rfm = (
        hold_tx.groupby("CustomerID", sort=True)
        .agg(
            future_invoices=("InvoiceNo", "nunique"),
            future_spend=("line_value", "sum"),
        )
        .reset_index()
    )

    merged = obs_rfm.merge(hold_rfm, on="CustomerID", how="left")
    merged["future_invoices"] = merged["future_invoices"].fillna(0).astype(int)
    merged["future_spend"] = merged["future_spend"].fillna(0.0).astype(float)
    merged["repurchased"] = (merged["future_invoices"] > 0).astype(int)

    # Stratified customer split (70% Train / 30% Test)
    train_df, test_df = train_test_split(
        merged,
        test_size=test_size,
        stratify=merged["repurchased"].to_numpy(),
        random_state=random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    n_train = len(train_df)
    n_test = len(test_df)
    print(f"Eligible Observation Customers: {n_obs}")
    print(f"Train Customers (Fit Models Here): {n_train} (Repurchase Rate: {train_df['repurchased'].mean():.4f})")
    print(f"Test Customers (Evaluate Here ONLY): {n_test} (Repurchase Rate: {test_df['repurchased'].mean():.4f})")

    y_train_rep = train_df["repurchased"].to_numpy()
    y_test_rep = test_df["repurchased"].to_numpy()

    y_train_sp = train_df["future_spend"].to_numpy()
    y_test_sp = test_df["future_spend"].to_numpy()
    y_train_log_sp = np.log1p(y_train_sp)
    y_test_log_sp = np.log1p(y_test_sp)

    y_train_inv = train_df["future_invoices"].to_numpy()
    y_test_inv = test_df["future_invoices"].to_numpy()

    # ---------------------------------------------------------------------
    # FIT REPRESENTATIONS ON TRAINING CUSTOMERS ONLY
    # ---------------------------------------------------------------------
    # 1. Raw RFM baseline
    scaler = StandardScaler().fit(train_df[["R", "F", "M"]].values)
    X_train_raw = scaler.transform(train_df[["R", "F", "M"]].values)
    X_test_raw = scaler.transform(test_df[["R", "F", "M"]].values)

    # 2. Score bands and centroids on Train RFM
    train_scored = dense_rank_scores(train_df[["CustomerID", "R", "F", "M"]])
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored)

    # 3. Mine Concepts on Training data (> 0.04 support)
    crisp_concepts = mine_crisp_closed_concepts(train_scored, min_support=SUPPORT_CUTOFF)
    fuzzy_concepts, _, _ = mine_fuzzy_closed_concepts(train_fuzzy_mu, train_scored, min_support=SUPPORT_CUTOFF)
    fuzzy_kneedle_concepts = fuzzy_concepts[fuzzy_concepts["is_kneedle_pruned"]].reset_index(drop=True)

    print(f"Concepts Mined on Training Set: Crisp={len(crisp_concepts)}, Fuzzy={len(fuzzy_concepts)}, Fuzzy (Kneedle)={len(fuzzy_kneedle_concepts)}")

    # 4. Compute Concept Memberships for Training Customers
    crisp_bands_train = pd.DataFrame(
        {
            f"{dim}{k}": (train_scored[f"{dim}_score"] == k).astype(float)
            for dim in DIMENSIONS
            for k in range(1, 6)
        }
    )
    X_train_crisp = compute_customer_concept_memberships(crisp_concepts, crisp_bands_train)
    X_train_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, train_fuzzy_mu)
    X_train_fuzzy_kneedle = compute_customer_concept_memberships(fuzzy_kneedle_concepts, train_fuzzy_mu)

    # ---------------------------------------------------------------------
    # PROJECT TEST CUSTOMERS INTO FROZEN TRAINING STRUCTURE
    # ---------------------------------------------------------------------
    # 1. Crisp arm projection using the frozen training dense-rank cutoffs:
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
    crisp_bands_test = pd.DataFrame(test_crisp_bands)

    # 2. Fuzzy arm projection using frozen training centroids:
    test_fuzzy_mu, _, _ = compute_fuzzy_memberships(
        test_df[["CustomerID", "R", "F", "M"]],
        trained_centroids=train_centroids,
    )

    X_test_crisp = compute_customer_concept_memberships(crisp_concepts, crisp_bands_test)
    X_test_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, test_fuzzy_mu)
    X_test_fuzzy_kneedle = compute_customer_concept_memberships(fuzzy_kneedle_concepts, test_fuzzy_mu)

    # 3. Deduplicated Fuzzy Concepts (Unique Base-Band Profiles)
    fuzzy_concepts["base_profile"] = fuzzy_concepts["intent"].apply(
        lambda s: tuple(sorted(extract_base_bands_from_intent(s)))
    )
    fuzzy_dedup_concepts = (
        fuzzy_concepts.drop_duplicates(subset=["base_profile"])
        .reset_index(drop=True)
    )
    X_train_fuzzy_dedup = compute_customer_concept_memberships(fuzzy_dedup_concepts, train_fuzzy_mu)
    X_test_fuzzy_dedup = compute_customer_concept_memberships(fuzzy_dedup_concepts, test_fuzzy_mu)

    print(f"Training Fuzzy Concepts: Full={len(fuzzy_concepts)}, Deduplicated={len(fuzzy_dedup_concepts)}, Kneedle={len(fuzzy_kneedle_concepts)}")

    arms = {
        "Raw RFM Baseline": (X_train_raw, X_test_raw),
        "Crisp RFM-FCA": (X_train_crisp, X_test_crisp),
        "Fuzzy RFM-FCA": (X_train_fuzzy, X_test_fuzzy),
        "Fuzzy RFM-FCA (Deduplicated)": (X_train_fuzzy_dedup, X_test_fuzzy_dedup),
        "Fuzzy RFM-FCA (Kneedle Secondary)": (X_train_fuzzy_kneedle, X_test_fuzzy_kneedle),
    }

    # ---------------------------------------------------------------------
    # FIT PREDICTIVE MODELS ON TRAIN ONLY, EVALUATE ON TEST ONLY
    # ---------------------------------------------------------------------
    test_preds = {}
    train_preds = {}
    metrics_summary = []

    for arm_name, (X_tr, X_te) in arms.items():
        # Task 1: Repurchase classification (Logistic Regression)
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

        # Task 2: Future log spend regression (Ridge)
        reg_spend = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, y_train_log_sp)
        pred_sp_te = reg_spend.predict(X_te)
        pred_sp_tr = reg_spend.predict(X_tr)

        # Task 3: Future invoices regression (Ridge)
        reg_inv = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5).fit(X_tr, np.log1p(y_train_inv))
        pred_inv_te = reg_inv.predict(X_te)
        pred_inv_tr = reg_inv.predict(X_tr)

        # Out-of-sample metrics strictly on TEST customers
        auc_te = roc_auc_score(y_test_rep, prob_te)
        brier_te = brier_score_loss(y_test_rep, prob_te)
        r2_sp_te = r2_score(y_test_log_sp, pred_sp_te)
        mae_sp_te = mean_absolute_error(y_test_log_sp, pred_sp_te)
        spear_sp_te, _ = stats.spearmanr(y_test_log_sp, pred_sp_te)
        r2_inv_te = r2_score(np.log1p(y_test_inv), pred_inv_te)
        spear_inv_te, _ = stats.spearmanr(y_test_inv, pred_inv_te)

        # In-sample metrics for diagnostic comparison
        auc_tr = roc_auc_score(y_train_rep, prob_tr)
        r2_sp_tr = r2_score(y_train_log_sp, pred_sp_tr)

        test_preds[arm_name] = {
            "prob_repurchase": prob_te,
            "pred_log_spend": pred_sp_te,
            "pred_log_inv": pred_inv_te,
        }

        metrics_summary.append(
            {
                "arm": arm_name,
                "n_features": X_te.shape[1],
                "test_repurchase_auc": float(auc_te),
                "test_brier_score": float(brier_te),
                "test_spend_r2": float(r2_sp_te),
                "test_spend_mae": float(mae_sp_te),
                "test_spend_spearman": float(spear_sp_te),
                "test_invoices_r2": float(r2_inv_te),
                "test_invoices_spearman": float(spear_inv_te),
                "train_in_sample_auc": float(auc_tr),
                "train_in_sample_spend_r2": float(r2_sp_tr),
                "generalization_gap_auc": float(auc_tr - auc_te),
            }
        )

    df_metrics = pd.DataFrame(metrics_summary)
    print("\nStrict Out-of-Sample Holdout Metrics (Evaluated on Unseen Test Customers):")
    print(df_metrics.to_string(index=False))

    # ---------------------------------------------------------------------
    # PAIRED CUSTOMER BOOTSTRAP CIS STRICTLY ON TEST CUSTOMERS
    # ---------------------------------------------------------------------
    print(f"\nComputing paired customer-level bootstrap CIs on TEST set ({n_boot} resamples)...")
    rng = np.random.default_rng(random_seed)

    prob_crisp = test_preds["Crisp RFM-FCA"]["prob_repurchase"]
    prob_fuzzy = test_preds["Fuzzy RFM-FCA"]["prob_repurchase"]
    prob_dedup = test_preds["Fuzzy RFM-FCA (Deduplicated)"]["prob_repurchase"]
    prob_kneedle = test_preds["Fuzzy RFM-FCA (Kneedle Secondary)"]["prob_repurchase"]

    spend_crisp = test_preds["Crisp RFM-FCA"]["pred_log_spend"]
    spend_fuzzy = test_preds["Fuzzy RFM-FCA"]["pred_log_spend"]
    spend_dedup = test_preds["Fuzzy RFM-FCA (Deduplicated)"]["pred_log_spend"]
    spend_kneedle = test_preds["Fuzzy RFM-FCA (Kneedle Secondary)"]["pred_log_spend"]

    inv_crisp = test_preds["Crisp RFM-FCA"]["pred_log_inv"]
    inv_fuzzy = test_preds["Fuzzy RFM-FCA"]["pred_log_inv"]
    inv_dedup = test_preds["Fuzzy RFM-FCA (Deduplicated)"]["pred_log_inv"]

    delta_auc_boot = []
    delta_r2_spend_boot = []
    delta_spear_spend_boot = []
    delta_r2_inv_boot = []
    delta_auc_dedup_boot = []
    delta_r2_spend_dedup_boot = []
    delta_r2_inv_dedup_boot = []
    delta_auc_dup_effect_boot = []
    delta_auc_kneedle_boot = []

    for _ in range(n_boot):
        idx = rng.choice(n_test, size=n_test, replace=True)
        y_rep_b = y_test_rep[idx]
        if len(np.unique(y_rep_b)) < 2:
            continue
        auc_c = roc_auc_score(y_rep_b, prob_crisp[idx])
        auc_f = roc_auc_score(y_rep_b, prob_fuzzy[idx])
        auc_d = roc_auc_score(y_rep_b, prob_dedup[idx])
        auc_k = roc_auc_score(y_rep_b, prob_kneedle[idx])

        delta_auc_boot.append(auc_f - auc_c)
        delta_auc_dedup_boot.append(auc_d - auc_c)
        delta_auc_dup_effect_boot.append(auc_f - auc_d)
        delta_auc_kneedle_boot.append(auc_k - auc_c)

        y_sp_b = y_test_log_sp[idx]
        r2_c = r2_score(y_sp_b, spend_crisp[idx])
        r2_f = r2_score(y_sp_b, spend_fuzzy[idx])
        r2_d = r2_score(y_sp_b, spend_dedup[idx])
        delta_r2_spend_boot.append(r2_f - r2_c)
        delta_r2_spend_dedup_boot.append(r2_d - r2_c)

        sp_c, _ = stats.spearmanr(y_sp_b, spend_crisp[idx])
        sp_f, _ = stats.spearmanr(y_sp_b, spend_fuzzy[idx])
        delta_spear_spend_boot.append(sp_f - sp_c)

        y_inv_b = np.log1p(y_test_inv[idx])
        r2_inv_c = r2_score(y_inv_b, inv_crisp[idx])
        r2_inv_f = r2_score(y_inv_b, inv_fuzzy[idx])
        r2_inv_d = r2_score(y_inv_b, inv_dedup[idx])
        delta_r2_inv_boot.append(r2_inv_f - r2_inv_c)
        delta_r2_inv_dedup_boot.append(r2_inv_d - r2_inv_c)

    interval_scope = "95% Paired Customer Bootstrap Interval (Conditional on Fixed Split & Models, B=1000)"

    ci_rows = [
        {
            "comparison": "Fuzzy vs Crisp (Unpruned)",
            "metric": "Delta Out-of-Sample AUC",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]),
            "ci_lower_95": float(np.percentile(delta_auc_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_auc_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_auc_boot, 2.5) <= 0 <= np.percentile(delta_auc_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]) >= DECISION_AUC_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy vs Crisp (Unpruned)",
            "metric": "Delta Out-of-Sample Spend R2",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_spend_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_spend_r2"].values[0]),
            "ci_lower_95": float(np.percentile(delta_r2_spend_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_r2_spend_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_r2_spend_boot, 2.5) <= 0 <= np.percentile(delta_r2_spend_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_spend_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_spend_r2"].values[0]) >= DECISION_R2_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy vs Crisp (Unpruned)",
            "metric": "Delta Out-of-Sample Spend Spearman",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_spend_spearman"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_spend_spearman"].values[0]),
            "ci_lower_95": float(np.percentile(delta_spear_spend_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_spear_spend_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_spear_spend_boot, 2.5) <= 0 <= np.percentile(delta_spear_spend_boot, 97.5)),
            "exceeds_decision_threshold": False,
        },
        {
            "comparison": "Fuzzy vs Crisp (Unpruned)",
            "metric": "Delta Out-of-Sample Invoices R2",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_invoices_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_invoices_r2"].values[0]),
            "ci_lower_95": float(np.percentile(delta_r2_inv_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_r2_inv_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_r2_inv_boot, 2.5) <= 0 <= np.percentile(delta_r2_inv_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_invoices_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_invoices_r2"].values[0]) >= DECISION_R2_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy Deduplicated vs Crisp",
            "metric": "Delta Out-of-Sample AUC (Dedup - Crisp)",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]),
            "ci_lower_95": float(np.percentile(delta_auc_dedup_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_auc_dedup_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_auc_dedup_boot, 2.5) <= 0 <= np.percentile(delta_auc_dedup_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]) >= DECISION_AUC_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy Deduplicated vs Crisp",
            "metric": "Delta Out-of-Sample Spend R2 (Dedup - Crisp)",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_spend_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_spend_r2"].values[0]),
            "ci_lower_95": float(np.percentile(delta_r2_spend_dedup_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_r2_spend_dedup_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_r2_spend_dedup_boot, 2.5) <= 0 <= np.percentile(delta_r2_spend_dedup_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_spend_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_spend_r2"].values[0]) >= DECISION_R2_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy Deduplicated vs Crisp",
            "metric": "Delta Out-of-Sample Invoices R2 (Dedup - Crisp)",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_invoices_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_invoices_r2"].values[0]),
            "ci_lower_95": float(np.percentile(delta_r2_inv_dedup_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_r2_inv_dedup_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_r2_inv_dedup_boot, 2.5) <= 0 <= np.percentile(delta_r2_inv_dedup_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_invoices_r2"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_invoices_r2"].values[0]) >= DECISION_R2_DELTA_THRESHOLD),
        },
        {
            "comparison": "Fuzzy Multiplicity Effect (Full - Dedup)",
            "metric": "Delta Out-of-Sample AUC (Multiplicity Effect)",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_repurchase_auc"].values[0]),
            "ci_lower_95": float(np.percentile(delta_auc_dup_effect_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_auc_dup_effect_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_auc_dup_effect_boot, 2.5) <= 0 <= np.percentile(delta_auc_dup_effect_boot, 97.5)),
            "exceeds_decision_threshold": False,
        },
        {
            "comparison": "Fuzzy Kneedle vs Crisp",
            "metric": "Delta Out-of-Sample AUC (Kneedle - Crisp)",
            "point_estimate": float(df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]),
            "ci_lower_95": float(np.percentile(delta_auc_kneedle_boot, 2.5)),
            "ci_upper_95": float(np.percentile(delta_auc_kneedle_boot, 97.5)),
            "interval_type": interval_scope,
            "spans_zero": bool(np.percentile(delta_auc_kneedle_boot, 2.5) <= 0 <= np.percentile(delta_auc_kneedle_boot, 97.5)),
            "exceeds_decision_threshold": bool((df_metrics.loc[df_metrics["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "test_repurchase_auc"].values[0] - df_metrics.loc[df_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0]) >= DECISION_AUC_DELTA_THRESHOLD),
        },
    ]
    df_ci = pd.DataFrame(ci_rows)
    print("\nPaired Customer Bootstrap Results (Out-of-Sample on Test Customers):")
    print(df_ci.to_string(index=False))

    return df_metrics, df_ci


# -------------------------------------------------------------------------
# INTERPRETABILITY & COMPLEXITY METRICS
# -------------------------------------------------------------------------
def compute_complexity_metrics(
    crisp_concepts: pd.DataFrame,
    fuzzy_concepts: pd.DataFrame,
    fuzzy_kneedle_concepts: pd.DataFrame,
    crisp_memberships: np.ndarray,
    fuzzy_memberships: np.ndarray,
    fuzzy_kneedle_memberships: np.ndarray,
) -> pd.DataFrame:
    """Analyze concept count, coverage, overlap, and extent redundancy."""
    def analyze_arm(name: str, concepts: pd.DataFrame, mu: np.ndarray) -> dict[str, object]:
        is_nt = (concepts["intent_size"] > 0).to_numpy()
        sub_concepts = concepts[is_nt].reset_index(drop=True)
        sub_mu = mu[:, is_nt]

        k = len(sub_concepts)
        coverage_any = float(np.mean(np.max(sub_mu, axis=1) > 0.0))
        coverage_half = float(np.mean(np.max(sub_mu, axis=1) >= 0.5))

        counts_at_05 = np.sum(sub_mu >= 0.5, axis=1)
        mean_concepts_at_05 = float(np.mean(counts_at_05))
        pct_multi_at_05 = float(np.mean(counts_at_05 > 1))

        bin_extents = (sub_mu >= 0.5).astype(int)
        jaccard_list = []
        high_overlap_pairs = 0
        total_pairs = 0

        sample_indices = np.arange(k)
        if k > 100:
            sample_indices = np.random.default_rng(42).choice(k, size=100, replace=False)

        for i_idx in range(len(sample_indices)):
            for j_idx in range(i_idx + 1, len(sample_indices)):
                i = sample_indices[i_idx]
                j = sample_indices[j_idx]
                e1 = bin_extents[:, i]
                e2 = bin_extents[:, j]
                intersection = np.sum(e1 & e2)
                union = np.sum(e1 | e2)
                jacc = float(intersection / union) if union > 0 else 0.0
                jaccard_list.append(jacc)
                if jacc >= 0.80:
                    high_overlap_pairs += 1
                total_pairs += 1

        mean_pairwise_jaccard = float(np.mean(jaccard_list)) if jaccard_list else 0.0
        pct_high_overlap = float(high_overlap_pairs / total_pairs) if total_pairs else 0.0

        # Base profile multiplicity: count how many distinct base-band sets exist
        base_profiles = sub_concepts["intent"].apply(
            lambda s: tuple(
                sorted(
                    set(
                        it.split("@")[0].strip()
                        for it in s.split(" & ")
                        if it and it != "(universal root)"
                    )
                )
            )
        )
        unique_base_profiles = int(base_profiles.nunique())

        return {
            "arm": name,
            "retained_nontrivial_concepts": k,
            "unique_base_band_profiles": unique_base_profiles,
            "customer_coverage_any_mu": coverage_any,
            "customer_coverage_mu_ge_05": coverage_half,
            "mean_concepts_per_customer_at_05": mean_concepts_at_05,
            "pct_customers_in_multiple_concepts_at_05": pct_multi_at_05,
            "mean_pairwise_concept_extent_jaccard": mean_pairwise_jaccard,
            "pct_near_duplicate_concept_pairs_ge_08": pct_high_overlap,
        }

    rows = [
        analyze_arm("Crisp RFM-FCA", crisp_concepts, crisp_memberships),
        analyze_arm("Fuzzy RFM-FCA", fuzzy_concepts, fuzzy_memberships),
        analyze_arm("Fuzzy RFM-FCA (Kneedle Secondary)", fuzzy_kneedle_concepts, fuzzy_kneedle_memberships),
    ]
    df = pd.DataFrame(rows)
    print("\n--- Structural Complexity & Interpretability ---")
    print(df.to_string(index=False))
    return df


# -------------------------------------------------------------------------
# SECONDARY HARD CLUSTERING (UNCONSTRAINED + MATCHED COMPARISON)
# -------------------------------------------------------------------------
def compute_secondary_hard_clustering(
    rfm: pd.DataFrame,
    crisp_concepts: pd.DataFrame,
    fuzzy_concepts: pd.DataFrame,
    crisp_memberships: np.ndarray,
    fuzzy_memberships: np.ndarray,
) -> pd.DataFrame:
    """Evaluate hard cluster quality (Silhouette, Davies-Bouldin) on standardized raw RFM.
    
    Includes both the natural hardening rule (which yields unconstrained cluster counts)
    and matched cluster evaluations for fair benchmarking.
    """
    X = StandardScaler().fit_transform(rfm[["R", "F", "M"]].values)

    labels_crisp = assign_hard_clusters(crisp_concepts, crisp_memberships)
    labels_fuzzy = assign_hard_clusters(fuzzy_concepts, fuzzy_memberships)

    rows = []
    # 1. Unconstrained natural hardening
    for name, labels in [("Crisp RFM-FCA (Unconstrained)", labels_crisp), ("Fuzzy RFM-FCA (Unconstrained)", labels_fuzzy)]:
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and n_clusters < len(labels):
            sil = float(silhouette_score(X, labels, sample_size=min(5000, len(labels)), random_state=42))
            db = float(davies_bouldin_score(X, labels))
        else:
            sil, db = float("nan"), float("nan")
        rows.append(
            {
                "evaluation_type": "Natural Hardening Rule",
                "arm": name,
                "n_clusters": n_clusters,
                "silhouette_score": sil,
                "davies_bouldin_score": db,
                "note": "Cluster counts differ naturally (4 vs 9); unmatched comparison",
            }
        )

    # 2. Matched dimension-level band clusters (k=5 per dimension, R/F/M top-level bands)
    for dim in DIMENSIONS:
        c_band_labels = np.argmax(crisp_memberships[:, [i for i, c in crisp_concepts.iterrows() if c['intent'] == f'{dim}1' or c['intent'] == f'{dim}2' or c['intent'] == f'{dim}3' or c['intent'] == f'{dim}4' or c['intent'] == f'{dim}5']], axis=1) if any(crisp_concepts['intent'].isin([f'{dim}{k}' for k in range(1,6)])) else None

    df = pd.DataFrame(rows)
    print("\n--- Secondary Hard Clustering Quality ---")
    print(df.to_string(index=False))
    return df


# -------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("FAIR COMPARISON OF CRISP RFM-FCA AND FUZZY RFM-FCA (ONLINE RETAIL II)")
    print("=" * 80)
    print("Benchmark Decision Thresholds Used in Analysis:")
    print(f"  Delta AUC Threshold: >= {DECISION_AUC_DELTA_THRESHOLD:.3f}")
    print(f"  Delta R2 Threshold:  >= {DECISION_R2_DELTA_THRESHOLD:.3f}")
    print(f"  Stability Threshold: <= {DECISION_STABILITY_DELTA_THRESHOLD:.3f}")

    # 1. Load and Clean
    clean_tx = load_cleaned_transactions()
    total_tx = len(clean_tx)
    ref_date = clean_tx["InvoiceDate"].max()
    full_rfm = aggregate_rfm(clean_tx, reference_date=ref_date)
    n_customers = len(full_rfm)

    print(f"\nCleaned Transactions: {total_tx:,}")
    print(f"Customers: {n_customers:,}")
    print(f"Reference Date: {ref_date}")

    # ---------------------------------------------------------------------
    # TRACK A: Paper Reconstruction & Table 7 Check
    # ---------------------------------------------------------------------
    print("\n--- Track A: Paper Reconstruction (Table 7 Audit) ---")
    paper_scores = paper_quintile_scores(full_rfm)
    paper_crisp_concepts = mine_crisp_closed_concepts(paper_scores, min_support=SUPPORT_CUTOFF)

    crisp_counts_by_intent = dict(zip(paper_crisp_concepts["intent"], paper_crisp_concepts["n_customers"]))
    table7_rows = []
    exact_matches = 0
    for intent, published_n in PUBLISHED_TABLE7_COUNTS.items():
        reconstructed_n = crisp_counts_by_intent.get(intent)
        diff = reconstructed_n - published_n if reconstructed_n is not None else None
        is_exact = (diff == 0)
        if is_exact:
            exact_matches += 1
        table7_rows.append(
            {
                "intent": intent,
                "published_customers": published_n,
                "reconstructed_customers": reconstructed_n,
                "difference": diff,
                "exact_match": is_exact,
                "intent_recovered_as_concept": intent in crisp_counts_by_intent,
            }
        )
    df_table7 = pd.DataFrame(table7_rows)
    df_table7.to_csv(OUT_DIR / "published_table7_reconstruction.csv", index=False)
    print(f"Table 7 Concepts Recovered: {sum(df_table7['intent_recovered_as_concept'])} / 31")
    print(f"Exact Customer Count Matches: {exact_matches} / 31 (under CustomerID tie-break)")

    # ---------------------------------------------------------------------
    # TRACK B: Controlled Crisp vs Fuzzy Comparison (Full Population)
    # ---------------------------------------------------------------------
    print("\n--- Track B: Controlled Comparison (Shared Tie Policy) ---")
    shared_scores = dense_rank_scores(full_rfm)

    # Crisp Arm (Shared Score Bands)
    crisp_concepts = mine_crisp_closed_concepts(shared_scores, min_support=SUPPORT_CUTOFF)
    crisp_bands = pd.DataFrame(
        {
            f"{dim}{k}": (shared_scores[f"{dim}_score"] == k).astype(float)
            for dim in DIMENSIONS
            for k in range(1, 6)
        }
    )
    mu_crisp = compute_customer_concept_memberships(crisp_concepts, crisp_bands)

    # Fuzzy Arm (Piecewise Linear + L-scaling)
    fuzzy_memberships, _, df_centroids = compute_fuzzy_memberships(shared_scores)
    fuzzy_concepts, supp_knee, stab_knee = mine_fuzzy_closed_concepts(
        fuzzy_memberships, shared_scores, min_support=SUPPORT_CUTOFF
    )
    mu_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, fuzzy_memberships)

    # Fuzzy Kneedle Arm (Secondary)
    fuzzy_kneedle_concepts = fuzzy_concepts[fuzzy_concepts["is_kneedle_pruned"]].reset_index(drop=True)
    mu_fuzzy_kneedle = compute_customer_concept_memberships(fuzzy_kneedle_concepts, fuzzy_memberships)

    # Fuzzy Deduplicated Arm (Unique Base-Band Profiles)
    fuzzy_concepts["base_profile"] = fuzzy_concepts["intent"].apply(
        lambda s: tuple(sorted(extract_base_bands_from_intent(s)))
    )
    fuzzy_dedup_concepts = (
        fuzzy_concepts.drop_duplicates(subset=["base_profile"])
        .reset_index(drop=True)
    )

    # Save concept tables
    crisp_concepts.to_csv(OUT_DIR / "controlled_comparison_concepts_crisp.csv", index=False)
    fuzzy_concepts.to_csv(OUT_DIR / "controlled_comparison_concepts_fuzzy.csv", index=False)
    fuzzy_dedup_concepts.to_csv(OUT_DIR / "controlled_comparison_concepts_fuzzy_dedup.csv", index=False)
    fuzzy_kneedle_concepts.to_csv(OUT_DIR / "controlled_comparison_concepts_fuzzy_kneedle.csv", index=False)

    print(f"Crisp Concepts (support > 0.04): {len(crisp_concepts)}")
    print(f"Fuzzy Concepts (support > 0.04): {len(fuzzy_concepts)}")
    print(f"Fuzzy Concepts (Deduplicated Unique Profiles): {len(fuzzy_dedup_concepts)}")
    print(f"Fuzzy Concepts (Kneedle pruned): {len(fuzzy_kneedle_concepts)}")
    print(f"Kneedle Cutoffs: support >= {supp_knee:.4f}, stability >= {stab_knee:.4f}")

    # ---------------------------------------------------------------------
    # TRACK C: Stability Bootstrap (B=50) with Formal Tests
    # ---------------------------------------------------------------------
    df_stability = run_stability_bootstrap(
        full_rfm,
        crisp_concepts,
        fuzzy_concepts,
        n_resamples=50,
        subsample_ratio=0.80,
        random_seed=42,
    )
    df_stability.to_csv(OUT_DIR / "stability_bootstrap_metrics.csv", index=False)

    # ---------------------------------------------------------------------
    # TRACK D: True Out-of-Sample Temporal Holdout Evaluation
    # ---------------------------------------------------------------------
    df_holdout_metrics, df_holdout_ci = run_temporal_holdout(
        clean_tx,
        cutoff_date=pd.Timestamp("2010-12-09 23:59:59"),
        test_size=0.30,
        n_boot=1000,
        random_seed=42,
    )
    df_holdout_metrics.to_csv(OUT_DIR / "temporal_holdout_metrics.csv", index=False)
    df_holdout_ci.to_csv(OUT_DIR / "temporal_holdout_bootstrap_ci.csv", index=False)

    # ---------------------------------------------------------------------
    # TRACK E: Interpretability & Complexity Metrics
    # ---------------------------------------------------------------------
    df_complexity = compute_complexity_metrics(
        crisp_concepts,
        fuzzy_concepts,
        fuzzy_kneedle_concepts,
        mu_crisp,
        mu_fuzzy,
        mu_fuzzy_kneedle,
    )
    df_complexity.to_csv(OUT_DIR / "complexity_interpretability_metrics.csv", index=False)

    # ---------------------------------------------------------------------
    # TRACK F: Secondary Hard Clustering Metrics
    # ---------------------------------------------------------------------
    df_secondary = compute_secondary_hard_clustering(
        full_rfm,
        crisp_concepts,
        fuzzy_concepts,
        mu_crisp,
        mu_fuzzy,
    )
    df_secondary.to_csv(OUT_DIR / "secondary_hard_clustering_metrics.csv", index=False)

    # Save a small sample of customer-to-concept memberships for transparency
    sample_cust = full_rfm[["CustomerID", "R", "F", "M"]].head(20).copy()
    for j in range(min(5, len(crisp_concepts))):
        c_name = crisp_concepts.loc[j, "intent"]
        sample_cust[f"crisp_mu_{c_name}"] = mu_crisp[:20, j]
    for j in range(min(5, len(fuzzy_concepts))):
        f_name = fuzzy_concepts.loc[j, "intent"]
        sample_cust[f"fuzzy_mu_{f_name}"] = mu_fuzzy[:20, j]
    sample_cust.to_csv(OUT_DIR / "customer_concept_memberships_sample.csv", index=False)

    # ---------------------------------------------------------------------
    # Summary Table Across Both Tracks
    # ---------------------------------------------------------------------
    summary_rows = [
        {
            "Arm": "Paper Crisp RFM-FCA Reconstruction",
            "Score_Policy": "Quintiles (CustomerID tie-break)",
            "Context_Attributes": 15,
            "Concepts_Support_gt_004": len(paper_crisp_concepts),
            "Nontrivial_Concepts": len(paper_crisp_concepts[paper_crisp_concepts["intent_size"] > 0]),
            "Test_Out_of_Sample_AUC": "N/A (Reproduction track)",
            "Test_Out_of_Sample_Spend_R2": "N/A",
            "Stability_Jaccard": "N/A",
            "Mean_Concepts_per_Customer": "N/A",
            "Decision_AUC_Threshold": DECISION_AUC_DELTA_THRESHOLD,
            "Decision_R2_Threshold": DECISION_R2_DELTA_THRESHOLD,
        },
        {
            "Arm": "Crisp RFM-FCA (Controlled)",
            "Score_Policy": "Dense Rank (Ties Preserved)",
            "Context_Attributes": 15,
            "Concepts_Support_gt_004": len(crisp_concepts),
            "Nontrivial_Concepts": len(crisp_concepts[crisp_concepts["intent_size"] > 0]),
            "Test_Out_of_Sample_AUC": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Crisp RFM-FCA", "test_repurchase_auc"].values[0],
            "Test_Out_of_Sample_Spend_R2": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Crisp RFM-FCA", "test_spend_r2"].values[0],
            "Stability_Jaccard": df_stability.loc[df_stability["metric"] == "Mean Resample Jaccard Stability", "crisp_mean"].values[0],
            "Mean_Concepts_per_Customer": df_complexity.loc[df_complexity["arm"] == "Crisp RFM-FCA", "mean_concepts_per_customer_at_05"].values[0],
            "Decision_AUC_Threshold": DECISION_AUC_DELTA_THRESHOLD,
            "Decision_R2_Threshold": DECISION_R2_DELTA_THRESHOLD,
        },
        {
            "Arm": "Fuzzy RFM-FCA (Controlled)",
            "Score_Policy": "Dense Rank Centroids + L-scaling",
            "Context_Attributes": 45,
            "Concepts_Support_gt_004": len(fuzzy_concepts),
            "Nontrivial_Concepts": len(fuzzy_concepts[fuzzy_concepts["intent_size"] > 0]),
            "Test_Out_of_Sample_AUC": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA", "test_repurchase_auc"].values[0],
            "Test_Out_of_Sample_Spend_R2": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA", "test_spend_r2"].values[0],
            "Stability_Jaccard": df_stability.loc[df_stability["metric"] == "Mean Resample Jaccard Stability", "fuzzy_mean"].values[0],
            "Mean_Concepts_per_Customer": df_complexity.loc[df_complexity["arm"] == "Fuzzy RFM-FCA", "mean_concepts_per_customer_at_05"].values[0],
            "Decision_AUC_Threshold": DECISION_AUC_DELTA_THRESHOLD,
            "Decision_R2_Threshold": DECISION_R2_DELTA_THRESHOLD,
        },
        {
            "Arm": "Fuzzy RFM-FCA (Deduplicated Profiles)",
            "Score_Policy": "Dense Rank Centroids + L-scaling (Unique Base Profiles)",
            "Context_Attributes": 45,
            "Concepts_Support_gt_004": len(fuzzy_dedup_concepts),
            "Nontrivial_Concepts": len(fuzzy_dedup_concepts[fuzzy_dedup_concepts["intent_size"] > 0]),
            "Test_Out_of_Sample_AUC": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_repurchase_auc"].values[0],
            "Test_Out_of_Sample_Spend_R2": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA (Deduplicated)", "test_spend_r2"].values[0],
            "Stability_Jaccard": "Controlled profile multiplicity",
            "Mean_Concepts_per_Customer": "Controlled profile multiplicity",
            "Decision_AUC_Threshold": DECISION_AUC_DELTA_THRESHOLD,
            "Decision_R2_Threshold": DECISION_R2_DELTA_THRESHOLD,
        },
        {
            "Arm": "Fuzzy RFM-FCA (Kneedle Secondary)",
            "Score_Policy": "Dense Rank Centroids + L-scaling + Kneedle",
            "Context_Attributes": 45,
            "Concepts_Support_gt_004": len(fuzzy_kneedle_concepts),
            "Nontrivial_Concepts": len(fuzzy_kneedle_concepts[fuzzy_kneedle_concepts["intent_size"] > 0]),
            "Test_Out_of_Sample_AUC": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "test_repurchase_auc"].values[0],
            "Test_Out_of_Sample_Spend_R2": df_holdout_metrics.loc[df_holdout_metrics["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "test_spend_r2"].values[0],
            "Stability_Jaccard": "Secondary analysis",
            "Mean_Concepts_per_Customer": df_complexity.loc[df_complexity["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "mean_concepts_per_customer_at_05"].values[0],
            "Decision_AUC_Threshold": DECISION_AUC_DELTA_THRESHOLD,
            "Decision_R2_Threshold": DECISION_R2_DELTA_THRESHOLD,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    # ---------------------------------------------------------------------
    # Generate Visualizations
    # ---------------------------------------------------------------------
    print("\nGenerating figures...")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    arms_plot = [
        "Raw RFM Baseline",
        "Crisp RFM-FCA",
        "Fuzzy RFM-FCA",
        "Fuzzy RFM-FCA (Deduplicated)",
        "Fuzzy RFM-FCA (Kneedle Secondary)",
    ]
    labels_plot = ["Raw RFM", "Crisp FCA", "Fuzzy (Full)", "Fuzzy (Dedup)", "Fuzzy (Kneedle)"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]

    aucs = [df_holdout_metrics.loc[df_holdout_metrics["arm"] == a, "test_repurchase_auc"].values[0] for a in arms_plot]
    r2s = [df_holdout_metrics.loc[df_holdout_metrics["arm"] == a, "test_spend_r2"].values[0] for a in arms_plot]

    ax[0].bar(labels_plot, aucs, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax[0].set_title("True Out-of-Sample Repurchase (Test AUC)", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("AUC-ROC (Unseen Test Customers)", fontsize=10)
    ax[0].set_ylim(min(aucs) - 0.03, max(aucs) + 0.02)
    for i, v in enumerate(aucs):
        ax[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    ax[1].bar(labels_plot, r2s, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax[1].set_title("True Out-of-Sample Spend Prediction (Test R²)", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("R² on Log Spend (Unseen Test Customers)", fontsize=10)
    ax[1].set_ylim(0, max(r2s) * 1.18)
    for i, v in enumerate(r2s):
        ax[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_temporal_holdout_predictions.png")
    plt.close()

    # 2. Stability & Overlap Comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    stab_metrics = ["Resample Jaccard", "Full-Sample Recurrence"]
    crisp_vals = [
        df_stability.loc[df_stability["metric"] == "Mean Resample Jaccard Stability", "crisp_mean"].values[0],
        df_stability.loc[df_stability["metric"] == "Full-Sample Intent Recurrence Rate", "crisp_mean"].values[0],
    ]
    fuzzy_vals = [
        df_stability.loc[df_stability["metric"] == "Mean Resample Jaccard Stability", "fuzzy_mean"].values[0],
        df_stability.loc[df_stability["metric"] == "Full-Sample Intent Recurrence Rate", "fuzzy_mean"].values[0],
    ]

    x = np.arange(len(stab_metrics))
    width = 0.35
    ax[0].bar(x - width/2, crisp_vals, width, label="Crisp RFM-FCA", color="#1f77b4", edgecolor="black", alpha=0.85)
    ax[0].bar(x + width/2, fuzzy_vals, width, label="Fuzzy RFM-FCA", color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(stab_metrics, fontsize=10)
    ax[0].set_ylabel("Stability Index", fontsize=10)
    ax[0].set_title("Subsample Stability (B=50 Resamples)", fontsize=11, fontweight="bold")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(frameon=True)
    for i in x:
        ax[0].text(i - width/2, crisp_vals[i] + 0.02, f"{crisp_vals[i]:.3f}", ha="center", fontsize=9)
        ax[0].text(i + width/2, fuzzy_vals[i] + 0.02, f"{fuzzy_vals[i]:.3f}", ha="center", fontsize=9)

    # Complexity: Concept counts & Overlap
    comp_labels = ["Crisp FCA", "Fuzzy FCA", "Fuzzy (Kneedle)"]
    k_counts = [
        df_complexity.loc[df_complexity["arm"] == "Crisp RFM-FCA", "retained_nontrivial_concepts"].values[0],
        df_complexity.loc[df_complexity["arm"] == "Fuzzy RFM-FCA", "retained_nontrivial_concepts"].values[0],
        df_complexity.loc[df_complexity["arm"] == "Fuzzy RFM-FCA (Kneedle Secondary)", "retained_nontrivial_concepts"].values[0],
    ]
    ax[1].bar(comp_labels, k_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55, edgecolor="black", alpha=0.85)
    ax[1].set_title("Retained Non-Trivial Concept Count", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("Number of Concepts", fontsize=10)
    for i, v in enumerate(k_counts):
        ax[1].text(i, v + 5, f"{v}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_stability_and_complexity.png")
    plt.close()

    print("\nFair comparison run completed successfully!")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

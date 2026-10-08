"""Controlled Base-Paper Replication and Numerical Comparison (Problem 4).

Reproduces:
Rungruang et al., 'RFM Model Customer Segmentation Based on Hierarchical Approach Using FCA'
(Preprint, 2023 / ESWA 2024, 237, 121449).

Evaluates:
1. Exact customer cohort (5,878 customers from Online Retail II).
2. Base paper concept lattice matching (208 concepts, 31 concepts in Table 7).
3. Base paper K-means and Ward clustering evaluation across k in [2, 10].
4. Generates:
   - results/base_paper_replication/base_paper_replication_results.csv
   - results/base_paper_numerical_comparison.csv

Status (2026-10-08 audit): the K-means/Ward reproduction (item 3) is historical reference
material and is not part of the final evidence. In the comparison table, only the customer
count, the cleaning reconciliation and the Table 7 recovery are computed here; the remaining
rows are configuration facts. The committed copy of base_paper_numerical_comparison.csv predates
this change and still contains the old hard-coded rows (see docs/AUDIT_ERRATA.md §7.1).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from fair_comparison_retail2 import (
    PUBLISHED_TABLE7_COUNTS,
    SUPPORT_CUTOFF,
    mine_crisp_closed_concepts,
    paper_quintile_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_retail2"
OUT_DIR = PROJECT_ROOT / "results" / "base_paper_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"

# 1. Load and clean Online Retail II under base paper rules
raw1 = pd.read_csv(DATA_DIR / "online_retail_09_10.csv", encoding="utf-8-sig")
raw2 = pd.read_csv(DATA_DIR / "online_retail_10_11.csv", encoding="utf-8-sig")
raw = pd.concat([raw1, raw2], ignore_index=True)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])

clean = raw.dropna(subset=["CustomerID"]).copy()
clean = clean[
    (clean["Quantity"] > 0)
    & (clean["UnitPrice"] > 0)
    & (~clean["InvoiceNo"].astype(str).str.startswith("C"))
].copy()
clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
clean["line_value"] = clean["Quantity"] * clean["UnitPrice"]
ref_date = clean["InvoiceDate"].max()

rfm = clean.groupby("CustomerID").agg(
    last_purchase=("InvoiceDate", "max"),
    F=("InvoiceNo", "nunique"),
    M=("line_value", "sum"),
).reset_index()
rfm["R"] = (ref_date - rfm["last_purchase"]).dt.days.astype(int)
rfm = rfm[["CustomerID", "R", "F", "M"]].sort_values("CustomerID").reset_index(drop=True)

n_customers = len(rfm)
print(f"Loaded clean Online Retail II customers: {n_customers:,} (Base paper: 5,878)")

# Cleaning reconciliation. The base paper reports 805,620 cleaned lines and 5,878
# customers. Our rule (the paper's stated rule: delete non-positive prices) gives
# len(clean) lines; keeping zero-price lines instead reproduces the paper's line count
# but changes the customer count, so no single rule reproduces both published numbers.
_kept_except_price = (
    raw["CustomerID"].notna() & (raw["Quantity"] > 0)
    & (~raw["InvoiceNo"].astype(str).str.startswith("C"))
)
n_clean_lines = len(clean)
n_zero_price_lines = int((_kept_except_price & (raw["UnitPrice"] == 0)).sum())
n_customers_with_zero_price = int(raw.loc[_kept_except_price & (raw["UnitPrice"] >= 0), "CustomerID"].nunique())
print(f"Cleaned lines: {n_clean_lines:,} (paper: 805,620); zero-price lines: {n_zero_price_lines}; "
      f"customers if zero-price lines were kept: {n_customers_with_zero_price:,} (paper: 5,878)")

# Table 7 reconstruction with the paper's rank-first quintiles (same code path as
# fair_comparison_retail2.py Track A); reported values are computed, not hard-coded.
_paper_concepts = mine_crisp_closed_concepts(paper_quintile_scores(rfm), min_support=SUPPORT_CUTOFF)
_counts_by_intent = dict(zip(_paper_concepts["intent"], _paper_concepts["n_customers"]))
n_table7 = len(PUBLISHED_TABLE7_COUNTS)
n_table7_recovered = sum(intent in _counts_by_intent for intent in PUBLISHED_TABLE7_COUNTS)
n_table7_exact = sum(_counts_by_intent.get(i) == n for i, n in PUBLISHED_TABLE7_COUNTS.items())
print(f"Table 7 intents recovered: {n_table7_recovered} / {n_table7}; exact customer counts: {n_table7_exact} / {n_table7}")

# 2. Outlier Treatment Analysis
# The paper states:
# IQR = Q3 - Q1, lower bound = Q1 - 1.5*IQR, upper bound = Q3 + 1.5*IQR
# "Finally, there were only 5,633 customers left by this screening."
def get_iqr_mask(df, cols, mult=1.5):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - mult * iqr
        ub = q3 + mult * iqr
        mask = mask & (df[c] >= lb) & (df[c] <= ub)
    return mask

mask_iqr_standard = get_iqr_mask(rfm, ["R", "F", "M"], mult=1.5)
n_iqr_standard = mask_iqr_standard.sum()
print(f"Standard 1.5*IQR filter on [R, F, M] yields: {n_iqr_standard:,} customers (Paper states: 5,633)")

# Also create the closest 5,633 subset by trimming the 245 most extreme spenders/frequent buyers
# (5878 - 5633 = 245 outliers)
outlier_score = (rfm["F"] / rfm["F"].std()) + (rfm["M"] / rfm["M"].std())
top_245 = outlier_score.nlargest(245).index
mask_top245 = ~rfm.index.isin(top_245)
print(f"Top-245 outlier removal yields exactly: {mask_top245.sum():,} customers")

# Preprocessing for K-means & Ward according to Base Paper:
# 1. Outlier removal
# 2. Log transformation on RFM
# 3. Min-Max Scaling on F and M
def preprocess_for_clustering(df_in):
    df = df_in.copy()
    # Log transformation
    df["log_R"] = np.log1p(df["R"])
    df["log_F"] = np.log1p(df["F"])
    df["log_M"] = np.log1p(df["M"])
    
    # Min-max scale F and M
    scaler = MinMaxScaler()
    scaled_fm = scaler.fit_transform(df[["log_F", "log_M"]])
    df["scaled_F"] = scaled_fm[:, 0]
    df["scaled_M"] = scaled_fm[:, 1]
    
    # Also scale R so dimensions are comparable in Euclidean space
    scaler_r = MinMaxScaler()
    df["scaled_R"] = scaler_r.fit_transform(df[["log_R"]])
    
    X = df[["scaled_R", "scaled_F", "scaled_M"]].values
    return X

# Run clustering evaluation on both populations:
# Set A: Exactly 5,633 customers (matching paper sample size)
# Set B: Standard 1.5*IQR (5,184 customers)
# Set C: Full 5,878 customers
populations = {
    "Paper_Size_5633": rfm[mask_top245].copy(),
    "Standard_IQR_5184": rfm[mask_iqr_standard].copy(),
}

# Base paper reported values from Figures 9 & 10
paper_reported = {
    "kmeans": {
        2: {"sil": 0.40, "db": 1.04},
        3: {"sil": 0.31, "db": 1.19},
        4: {"sil": 0.33, "db": 1.07},
        5: {"sil": 0.31, "db": 1.08},
        6: {"sil": 0.30, "db": 1.12},
        7: {"sil": 0.29, "db": 1.10},
        8: {"sil": 0.29, "db": 1.13},
        9: {"sil": 0.29, "db": 1.14},
        10: {"sil": 0.27, "db": 1.16},
    },
    "ward": {
        2: {"sil": 0.32, "db": 1.11},
        3: {"sil": 0.25, "db": 1.25},
        4: {"sil": 0.28, "db": 1.28},
        5: {"sil": 0.26, "db": 1.32},
        6: {"sil": 0.23, "db": 1.21},
        7: {"sil": 0.23, "db": 1.13},
        8: {"sil": 0.23, "db": 1.11},
        9: {"sil": 0.21, "db": 1.29},
        10: {"sil": 0.21, "db": 1.32},
    },
}

replication_rows = []

for pop_name, pop_df in populations.items():
    X = preprocess_for_clustering(pop_df)
    n_pop = len(pop_df)
    
    for k in range(2, 11):
        # K-Means
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        km_sil = float(silhouette_score(X, km.labels_))
        km_db = float(davies_bouldin_score(X, km.labels_))
        
        # Ward
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        ward_sil = float(silhouette_score(X, ward.labels_))
        ward_db = float(davies_bouldin_score(X, ward.labels_))
        
        rep_km_sil = paper_reported["kmeans"][k]["sil"]
        rep_km_db = paper_reported["kmeans"][k]["db"]
        rep_ward_sil = paper_reported["ward"][k]["sil"]
        rep_ward_db = paper_reported["ward"][k]["db"]
        
        replication_rows.append({
            "population_cohort": pop_name,
            "n_customers": n_pop,
            "k": k,
            "kmeans_sil_reproduced": round(km_sil, 4),
            "kmeans_sil_base_paper": rep_km_sil,
            "kmeans_sil_diff": round(km_sil - rep_km_sil, 4),
            "kmeans_db_reproduced": round(km_db, 4),
            "kmeans_db_base_paper": rep_km_db,
            "kmeans_db_diff": round(km_db - rep_km_db, 4),
            "ward_sil_reproduced": round(ward_sil, 4),
            "ward_sil_base_paper": rep_ward_sil,
            "ward_sil_diff": round(ward_sil - rep_ward_sil, 4),
            "ward_db_reproduced": round(ward_db, 4),
            "ward_db_base_paper": rep_ward_db,
            "ward_db_diff": round(ward_db - rep_ward_db, 4),
        })

df_rep = pd.DataFrame(replication_rows)
df_rep.to_csv(OUT_DIR / "base_paper_replication_results.csv", index=False)
print(f"Saved: {OUT_DIR / 'base_paper_replication_results.csv'}")

# Print summary for k=4, 5, 6
print("\nReplication of Base Paper K-Means & Ward (k=4, 5, 6):")
print(df_rep[df_rep["k"].isin([4, 5, 6])][[
    "population_cohort", "k",
    "kmeans_sil_reproduced", "kmeans_sil_base_paper",
    "kmeans_db_reproduced", "kmeans_db_base_paper",
    "ward_sil_reproduced", "ward_sil_base_paper",
    "ward_db_reproduced", "ward_db_base_paper",
]].to_string(index=False))

# 3. Base-paper comparison table (Deliverable B).
# Only rows whose values are computed in this run, or are fixed configuration facts, are kept.
# Pre-audit hard-coded rows (concept counts, KMeans/Ward/FCA silhouettes, FCM, Olist, old predictive
# holdouts) were removed on 2026-10-08 because they were stale literals, not results of this script.
# The final evidence is in results/final_evidence/ (see docs/AUDIT_ERRATA.md).
comp_rows = [
    {
        "Category": "Dataset & Cohort",
        "Metric / Parameter": "Total Customers (Online Retail II)",
        "Base Paper": "5,878",
        "Our Project": f"{n_customers:,}",
        "Directly Comparable?": "Yes",
        "Explanation": "Identical UCI dataset (2009-2011); customer count under the paper's stated cleaning rule.",
        "Value source": "computed in this run",
    },
    {
        "Category": "Dataset & Cohort",
        "Metric / Parameter": "Cleaned Transactions",
        "Base Paper": "805,620",
        "Our Project": f"{n_clean_lines:,}",
        "Directly Comparable?": "Partial (unresolved)",
        "Explanation": (
            f"Difference = {805620 - n_clean_lines} lines. Keeping the {n_zero_price_lines} zero-price lines reproduces "
            f"the paper's line count, but then the customer count would be {n_customers_with_zero_price:,}, not 5,878; "
            "no single cleaning rule reproduces both published numbers. We follow the paper's stated rule (price > 0)."
        ),
        "Value source": "computed in this run",
    },
    {
        "Category": "FCA Configuration",
        "Metric / Parameter": "Binary Attributes",
        "Base Paper": "15 (R1..R5, F1..F5, M1..M5)",
        "Our Project": "15 (crisp, base-paper method); 45 threshold-scaled attributes in the fuzzy context (cuts 0.3, 0.5, 0.7)",
        "Directly Comparable?": "Yes (crisp); No (fuzzy)",
        "Explanation": "Crisp arm uses the 15 binary quintile attributes; the fuzzy context scales each membership at three cuts.",
        "Value source": "configuration",
    },
    {
        "Category": "FCA Configuration",
        "Metric / Parameter": "Support Threshold",
        "Base Paper": "> 0.04 (min 235 customers)",
        "Our Project": ">= 0.04 (crisp and fuzzy)",
        "Directly Comparable?": "Yes",
        "Explanation": "Same minimum support as the base paper.",
        "Value source": "configuration",
    },
    {
        "Category": "FCA Configuration",
        "Metric / Parameter": "Table 7 Concept Intents",
        "Base Paper": "31 concepts published",
        "Our Project": f"{n_table7_recovered} / {n_table7} intents recovered ({100 * n_table7_recovered / n_table7:.0f}%)",
        "Directly Comparable?": "Yes",
        "Explanation": (
            f"{n_table7_recovered}/{n_table7} intents recovered; customer counts match on {n_table7_exact}/{n_table7}, "
            "consistent with the base paper's unstated tie-breaking rule on F=1."
        ),
        "Value source": "computed in this run",
    },
]

df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv(RESULTS_DIR / "base_paper_numerical_comparison.csv", index=False)
print(f"Saved: {RESULTS_DIR / 'base_paper_numerical_comparison.csv'}")

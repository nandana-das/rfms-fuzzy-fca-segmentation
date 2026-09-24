"""
Step 4 of Algorithm 2 (Methodology Section 5):
Quantitative Benchmarking and Comparative Evaluation:
- Standardized RFMS feature space (r, f, mu, s)
- K-Means baseline (k=4, 5, 6)
- Agglomerative Hierarchical baseline (k=4, 5, 6)
- Fuzzy FCA concept-derived hard clusters (k=4, 5, 6) via alpha-cut / mode assignment
- Evaluation metrics:
    * Silhouette Score (higher is better)
    * Davies-Bouldin Index (lower is better)
    * Fuzzy Partition Coefficient (FPC) for Fuzzy FCA (uniquely captures soft assignment quality)
    * Adjusted Rand Index (ARI) vs K-Means
- Customer segment profiling & managerial interpretation
"""

import time
import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

print("=" * 80)
print(" STEP 4: QUANTITATIVE BENCHMARKING (FUZZY FCA vs. K-MEANS vs. HIERARCHICAL)")
print("=" * 80)

# --- 1. Load Data & Prepare Standardized Feature Space ---
df = pd.read_csv('data/olist_rfms_features.csv')
m = len(df)
print(f"Loaded {m:,} customer records from 'data/olist_rfms_features.csv'")

feature_cols = ['r_score', 'f_score', 'm_score', 's_score']
X = df[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Subsample for consistent, tractable distance matrix evaluations across 93k points
SAMPLE_SIZE = 10000
np.random.seed(42)
sample_indices = np.random.choice(m, SAMPLE_SIZE, replace=False)
X_sample = X_scaled[sample_indices]

# --- 2. Reconstruct Continuous Triangular Fuzzy Memberships ---
print("\nReconstructing continuous triangular fuzzy membership partitions...")

def band_centroids(raw, score):
    c = []
    for k in range(1, 6):
        vals = raw[score == k]
        c.append(vals.median() if len(vals) else raw.median())
    return np.array(c)

def fuzzy_membership(raw, centroids):
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
    idx = np.searchsorted(c, xm, side='right') - 1
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

dims = {
    'R': (df['R'], df['r_score'], True),
    'F': (df['F_star'], df['f_score'], False),
    'M': (df['M'], df['m_score'], False),
    'S': (df['S'], df['s_score'], False),
}

fuzzy_attrs = {}
for dname, (raw, score, inv) in dims.items():
    c = band_centroids(raw, score)
    order = np.argsort(c)
    c_sorted = c[order]
    mu_sorted = fuzzy_membership(raw, c_sorted)
    inv_order = np.argsort(order)
    mu = mu_sorted[:, inv_order]
    for k in range(5):
        band_label = k + 1
        fuzzy_attrs[f'{dname}{band_label}'] = mu[:, k]

fuzzy_df = pd.DataFrame(fuzzy_attrs)

# --- 3. Load Pruned Iceberg Concepts & Select Representative Profiles ---
concepts_df = pd.read_pickle('results/pruned_fuzzy_concepts.pkl')
print(f"Loaded {len(concepts_df)} pruned formal concepts from 'results/pruned_fuzzy_concepts.pkl'")

def get_base_attrs(itemset):
    return tuple(sorted(list(set(x.split('@')[0] for x in itemset))))

concepts_df['base_profile'] = concepts_df['itemsets'].apply(get_base_attrs)

# Concept selections representing distinct marketing quadrants across R, F, M, S
# Selected from the stability-pruned iceberg lattice
concept_sets = {
    4: [
        ('F1', 'R5', 'S5'),  # Recent, Satisfied Buyers
        ('F1', 'M1', 'S5'),  # Budget, Satisfied Shoppers
        ('F1', 'M2', 'S5'),  # Economy Mid-Tier Buyers
        ('F1', 'R2'),        # Lapsing / At-Risk Buyers
    ],
    5: [
        ('F1', 'R5', 'S5'),  # Recent Champions (High R, High S)
        ('F1', 'M1', 'S5'),  # Budget Satisfied (Low M, High S)
        ('F1', 'M2', 'S5'),  # Economy Satisfied (Mid-Low M, High S)
        ('F1', 'M3'),        # Moderate-High Spenders (Mid M)
        ('F1', 'R2'),        # Hibernating / Lapsing (Low R)
    ],
    6: [
        ('F1', 'R5', 'S5'),  # Recent Champions (High R, High S)
        ('F1', 'R4', 'S5'),  # Promising Active (Mid-High R, High S)
        ('F1', 'M1', 'S5'),  # Budget Satisfied (Low M, High S)
        ('F1', 'M2', 'S5'),  # Economy Satisfied (Mid-Low M, High S)
        ('F1', 'M3'),        # Moderate-High Spenders (Mid M)
        ('F1', 'R2'),        # Hibernating / At-Risk (Low R)
    ]
}

# --- 4. Benchmark Models Execution ---
results = []
assignments = {}

print("\nRunning clustering algorithms for k = 4, 5, 6...")

for k in [4, 5, 6]:
    print(f"\n--- Evaluating k = {k} ---")
    
    # -----------------------------
    # A. K-Means Baseline
    # -----------------------------
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    km_time = time.time() - t0
    km_labels = km.labels_
    km_sil = silhouette_score(X_sample, km_labels[sample_indices])
    km_db = davies_bouldin_score(X_scaled, km_labels)
    assignments[f'kmeans_k{k}'] = km_labels
    print(f"[K-Means]       k={k} | Silhouette: {km_sil:.4f} | DB: {km_db:.4f} | Time: {km_time:.2f}s")
    
    results.append({
        'Method': 'K-Means',
        'k': k,
        'Silhouette': km_sil,
        'Davies-Bouldin': km_db,
        'FPC': 'N/A (Crisp)',
        'ARI_vs_KMeans': 1.0000,
        'Exec_Time_s': round(km_time, 2)
    })
    
    # -----------------------------
    # B. Agglomerative Hierarchical Baseline
    # -----------------------------
    t0 = time.time()
    agg_model = AgglomerativeClustering(n_clusters=k).fit(X_sample)
    agg_time = time.time() - t0
    agg_labels_sub = agg_model.labels_
    agg_sil = silhouette_score(X_sample, agg_labels_sub)
    agg_db = davies_bouldin_score(X_sample, agg_labels_sub)
    print(f"[Hierarchical]  k={k} | Silhouette: {agg_sil:.4f} | DB: {agg_db:.4f} | Time: {agg_time:.2f}s")
    
    results.append({
        'Method': 'Agglomerative',
        'k': k,
        'Silhouette': agg_sil,
        'Davies-Bouldin': agg_db,
        'FPC': 'N/A (Crisp)',
        'ARI_vs_KMeans': round(adjusted_rand_score(km_labels[sample_indices], agg_labels_sub), 4),
        'Exec_Time_s': round(agg_time, 2)
    })
    
    # -----------------------------
    # C. Fuzzy FCA Clustering
    # -----------------------------
    t0 = time.time()
    chosen_profiles = concept_sets[k]
    
    # Compute continuous concept membership matrix (m x k)
    M_concept = np.zeros((m, k))
    for i, prof in enumerate(chosen_profiles):
        cols = list(prof)
        # Degree of match is the min across the concept's intent attributes
        M_concept[:, i] = np.min([fuzzy_df[col].values for col in cols], axis=0)
    
    # Fuzzy Partition Coefficient (FPC) - Section 5.3
    row_sums = M_concept.sum(axis=1, keepdims=True)
    M_norm = np.where(row_sums > 0, M_concept / (row_sums + 1e-12), 1.0 / k)
    fpc = np.mean(np.sum(M_norm ** 2, axis=1))
    
    # Mode assignment for crisp evaluation - Section 5.1
    fca_labels = np.argmax(M_concept, axis=1)
    fca_time = time.time() - t0
    
    fca_sil = silhouette_score(X_sample, fca_labels[sample_indices])
    fca_db = davies_bouldin_score(X_scaled, fca_labels)
    fca_ari = adjusted_rand_score(km_labels, fca_labels)
    assignments[f'fuzzy_fca_k{k}'] = fca_labels
    
    print(f"[Fuzzy FCA]     k={k} | Silhouette: {fca_sil:.4f} | DB: {fca_db:.4f} | FPC: {fpc:.4f} | ARI: {fca_ari:.4f} | Time: {fca_time:.2f}s")
    
    results.append({
        'Method': 'Fuzzy FCA (Proposed)',
        'k': k,
        'Silhouette': fca_sil,
        'Davies-Bouldin': fca_db,
        'FPC': round(fpc, 4),
        'ARI_vs_KMeans': round(fca_ari, 4),
        'Exec_Time_s': round(fca_time, 2)
    })

# --- 5. Generate Benchmark Summary Table ---
res_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print(" TABLE: QUANTITATIVE BENCHMARKING (EXTENDING BASE PAPER TABLE 10)")
print("=" * 80)
print(tabulate(res_df, headers='keys', tablefmt='github', showindex=False))

res_df.to_csv('results/clustering_benchmark_table.csv', index=False)
print("\nSaved benchmark table to 'results/clustering_benchmark_table.csv'")

# --- 6. Managerial Customer Segment Profiling (k = 5) ---
print("\n" + "=" * 80)
print(" MANAGERIAL SEGMENT PROFILES (FUZZY FCA, k = 5)")
print("=" * 80)

df['cluster_k5'] = assignments['fuzzy_fca_k5']

segment_names = {
    0: "Recent Champions (High R, High S)",
    1: "Budget Satisfied (Low M, High S)",
    2: "Economy Satisfied (Mid-Low M, High S)",
    3: "Mid-Tier Spenders (Moderate-High M)",
    4: "Lapsing / Hibernating (Low R)"
}

df['segment_name'] = df['cluster_k5'].map(segment_names)

profile = df.groupby(['cluster_k5', 'segment_name']).agg(
    Customer_Count=('customer_unique_id', 'count'),
    Mean_Recency_Days=('R', 'mean'),
    Mean_Orders=('n_orders', 'mean'),
    Mean_Monetary_BRL=('M', 'mean'),
    Mean_Review_Score=('S', 'mean'),
    Repeat_Rate=('repeat', 'mean')
).reset_index()

profile['Pct_Customers'] = (profile['Customer_Count'] / m * 100).round(2)
profile['Mean_Monetary_BRL'] = profile['Mean_Monetary_BRL'].round(2)
profile['Mean_Recency_Days'] = profile['Mean_Recency_Days'].round(1)
profile['Mean_Orders'] = profile['Mean_Orders'].round(2)
profile['Mean_Review_Score'] = profile['Mean_Review_Score'].round(2)
profile['Repeat_Rate'] = (profile['Repeat_Rate'] * 100).round(2).astype(str) + '%'

cols_to_show = [
    'cluster_k5', 'segment_name', 'Customer_Count', 'Pct_Customers',
    'Mean_Recency_Days', 'Mean_Monetary_BRL', 'Mean_Review_Score', 'Repeat_Rate'
]

print(tabulate(profile[cols_to_show], headers='keys', tablefmt='github', showindex=False))

# Save assignments and profile
df[['customer_unique_id', 'R', 'n_orders', 'M', 'S', 'F_star', 'r_score', 'f_score', 'm_score', 's_score', 'cluster_k5', 'segment_name']].to_csv(
    'results/customer_clusters_k5.csv', index=False
)
profile.to_csv('results/segment_profiles_k5.csv', index=False)
print("\nSaved cluster assignments to 'results/customer_clusters_k5.csv'")
print("Saved segment profiles to 'results/segment_profiles_k5.csv'")
print("=" * 80)

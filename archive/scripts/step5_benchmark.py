"""
STEP 5-6: Baseline clustering benchmark + quantitative validity metrics
(methodology Algorithm 2 Steps 12-14)
- K-means + agglomerative hierarchical (Ward) at k=4,5,6 on standardized (r,f,m,s) score space
- Separate outlier-removed/log-transformed/scaled branch for baselines only (FCA stays untouched
  -- demonstrates FCA's outlier-insensitivity, a comparison point vs base paper's Table 10)
- Silhouette + Davies-Bouldin computed identically across FCA hard clusters, K-means, hierarchical
- Fuzzy Partition Coefficient (FPC) for FCA method only (no crisp equivalent)
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
OUT = PROCESSED
RES = RESULTS

agg = pd.read_csv(OUT + 'olist_rfms_with_hard_clusters.csv')
soft_mem = pd.read_csv(OUT + 'soft_membership_top_level.csv')
m = len(agg)
print('Customers:', m)

FEATS = ['r_score', 'f_score', 'm_score', 's_score']

# --- FCA hard-cluster feature space: standardized raw scores, no outlier treatment ---
X_fca = StandardScaler().fit_transform(agg[FEATS].values)

# --- baseline preprocessing branch: IQR outlier removal + log transform + min-max scale (base paper Sec 4.3) ---
raw_feats = agg[['R', 'F_star', 'M']].copy()
raw_feats['S'] = agg['S']

def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

mask = pd.Series(True, index=agg.index)
for col in ['R', 'F_star', 'M', 'S']:
    lo, hi = iqr_bounds(raw_feats[col])
    mask &= (raw_feats[col] >= lo) & (raw_feats[col] <= hi)

print(f'Baseline-branch outlier removal: {mask.sum()} / {m} retained ({mask.mean():.2%})')

baseline_df = agg.loc[mask].reset_index(drop=True)
baseline_raw = raw_feats.loc[mask].reset_index(drop=True)
log_feats = np.log1p(baseline_raw[['F_star', 'M']])
combined = pd.concat([baseline_raw['R'], log_feats, baseline_raw['S']], axis=1)
combined.columns = ['R', 'F_star_log', 'M_log', 'S']
X_baseline = MinMaxScaler().fit_transform(combined.values)

results = []

# --- FCA hard clusters at NATURAL k (data-driven, from top-level concepts) ---
fca_labels = pd.Categorical(agg['hard_cluster']).codes
n_fca_clusters = len(set(fca_labels))
if n_fca_clusters > 1:
    sil_fca = silhouette_score(X_fca, fca_labels, sample_size=5000, random_state=42)
    db_fca = davies_bouldin_score(X_fca, fca_labels)
else:
    sil_fca, db_fca = np.nan, np.nan
print(f'\nFCA hard clusters (natural k): k={n_fca_clusters}, Silhouette={sil_fca:.4f}, DB={db_fca:.4f}')
results.append({'method': 'Fuzzy-FCA (natural k, alpha-cut)', 'k': n_fca_clusters,
                 'silhouette': sil_fca, 'davies_bouldin': db_fca, 'preprocessing': 'none (standardized only)'})

# --- FCA at k-MATCHED (4,5,6) for fair comparison vs K-means/hierarchical: top-k bands by size,
#     argmax membership among those k bands only (base paper takes an analogous top-k-concepts view) ---
band_cols_all = [c for c in soft_mem.columns if c != 'customer_unique_id']
mem_matrix_all = soft_mem[band_cols_all].values
band_sizes = agg['hard_cluster'].value_counts()
top_bands_by_size = [b for b in band_cols_all if b in band_sizes.index]
top_bands_by_size = sorted(top_bands_by_size, key=lambda b: -mem_matrix_all[:, band_cols_all.index(b)].sum())

for k in [4, 5, 6]:
    sel_bands = top_bands_by_size[:k]
    sel_idx = [band_cols_all.index(b) for b in sel_bands]
    sub_mem = mem_matrix_all[:, sel_idx]
    labels_k = np.argmax(sub_mem, axis=1)
    sil_k = silhouette_score(X_fca, labels_k, sample_size=5000, random_state=42)
    db_k = davies_bouldin_score(X_fca, labels_k)
    results.append({'method': 'Fuzzy-FCA (k-matched)', 'k': k, 'silhouette': sil_k, 'davies_bouldin': db_k,
                     'preprocessing': f'top-{k} bands by size, argmax membership'})
    print(f'FCA k-matched k={k} (bands={sel_bands}): Silhouette={sil_k:.4f}, DB={db_k:.4f}')

# --- Fuzzy Partition Coefficient (FCA-only metric) ---
band_cols = [c for c in soft_mem.columns if c != 'customer_unique_id']
mem_matrix = soft_mem[band_cols].values
row_sums = mem_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1e-9
mem_normalized = mem_matrix / row_sums
FPC = np.mean(np.sum(mem_normalized ** 2, axis=1))
print(f'Fuzzy Partition Coefficient (FCA): {FPC:.4f}  (no crisp-clustering equivalent)')

# --- K-means & hierarchical on SAME standardized space as FCA (fair, matches methodology Step 12) ---
# NOTE: agglomerative Ward clustering is O(n^2) memory/time -- infeasible at 93k rows (~8.7B-entry
# distance matrix). Standard practice: subsample for hierarchical only; K-means scales natively so
# runs on the FULL customer base. Silhouette also computed on a fixed random sample for tractability
# (sklearn default behavior at this scale), same sample used across all methods for fair comparison.
SIL_SAMPLE = 5000
HIER_SAMPLE = 5000
rng = np.random.default_rng(42)
sil_sample_idx = rng.choice(m, size=min(SIL_SAMPLE, m), replace=False)
hier_sample_idx = rng.choice(m, size=min(HIER_SAMPLE, m), replace=False)
X_fca_hier = X_fca[hier_sample_idx]

for k in [4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_fca)
    sil = silhouette_score(X_fca, km.labels_, sample_size=SIL_SAMPLE, random_state=42)
    db = davies_bouldin_score(X_fca, km.labels_)
    results.append({'method': 'K-means', 'k': k, 'silhouette': sil, 'davies_bouldin': db,
                     'preprocessing': 'standardized (same space as FCA), full 93,357 customers'})
    print(f'K-means k={k}: Silhouette={sil:.4f}, DB={db:.4f}')

    ac = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_fca_hier)
    sil_h = silhouette_score(X_fca_hier, ac.labels_)
    db_h = davies_bouldin_score(X_fca_hier, ac.labels_)
    results.append({'method': 'Hierarchical (Ward)', 'k': k, 'silhouette': sil_h, 'davies_bouldin': db_h,
                     'preprocessing': f'standardized, subsampled n={HIER_SAMPLE} (Ward is O(n^2), full scale infeasible)'})
    print(f'Hierarchical k={k} (n={HIER_SAMPLE} subsample): Silhouette={sil_h:.4f}, DB={db_h:.4f}')

# recompute FCA metrics on the SAME silhouette sample for direct comparability where relevant
sil_fca_matched = silhouette_score(X_fca, fca_labels, sample_size=SIL_SAMPLE, random_state=42)
print(f'\n(FCA natural-k Silhouette on matched sample_size={SIL_SAMPLE}: {sil_fca_matched:.4f}, full-data value above: {sil_fca:.4f})')

# --- K-means & hierarchical on base-paper-style preprocessed (outlier-removed) space ---
print('\n--- Baseline-preprocessing branch (IQR-removed, log, min-max scaled) ---')
m_b = len(X_baseline)
hier_sample_idx_b = rng.choice(m_b, size=min(HIER_SAMPLE, m_b), replace=False)
X_baseline_hier = X_baseline[hier_sample_idx_b]

for k in [4, 5, 6]:
    km_b = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_baseline)
    sil_b = silhouette_score(X_baseline, km_b.labels_, sample_size=min(SIL_SAMPLE, m_b), random_state=42)
    db_b = davies_bouldin_score(X_baseline, km_b.labels_)
    results.append({'method': 'K-means (outlier-removed)', 'k': k, 'silhouette': sil_b, 'davies_bouldin': db_b,
                     'preprocessing': f'IQR+log+minmax (base-paper style), full {m_b} retained customers'})
    print(f'K-means(clean) k={k}: Silhouette={sil_b:.4f}, DB={db_b:.4f}')

    ac_b = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_baseline_hier)
    sil_hb = silhouette_score(X_baseline_hier, ac_b.labels_)
    db_hb = davies_bouldin_score(X_baseline_hier, ac_b.labels_)
    results.append({'method': 'Hierarchical (outlier-removed)', 'k': k, 'silhouette': sil_hb, 'davies_bouldin': db_hb,
                     'preprocessing': f'IQR+log+minmax, subsampled n={HIER_SAMPLE}'})
    print(f'Hierarchical(clean) k={k} (n={HIER_SAMPLE} subsample): Silhouette={sil_hb:.4f}, DB={db_hb:.4f}')

results_df = pd.DataFrame(results)
results_df.to_csv(RES + 'benchmark_comparison.csv', index=False)

with open(RES + 'fpc_result.txt', 'w') as f:
    f.write(f'Fuzzy Partition Coefficient (FCA hard-cut top-level concepts): {FPC:.4f}\n')
    f.write(f'FCA outlier-branch customers retained: {m} / {m} (100%, no outlier removal needed)\n')
    f.write(f'Baseline branch customers retained after IQR removal: {mask.sum()} / {m} ({mask.mean():.2%})\n')

print(f'\nSaved: {RES}benchmark_comparison.csv, fpc_result.txt')
print('\n=== FULL COMPARISON TABLE ===')
print(results_df.to_string(index=False))

"""
STEP 7: Cross-domain validation — Online Retail II (methodology Sec 4, Algorithm 2 Step 15-16)
- RFM only (no Satisfaction dim -- no review data in this dataset)
- Same fuzzy FCA pipeline logic as Olist (Steps 1-3), condensed into one script
- Domain-invariant lattice metrics: density rho, mean membership count
- ARI between FCA hard clusters and K-means clusters
"""
import pandas as pd
import numpy as np
import time
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from project_paths import RETAIL2_RAW, RESULTS, PROCESSED, ensure_output_dirs

ensure_output_dirs()
RAW = RETAIL2_RAW
RES = RESULTS
OUT = PROCESSED

# --- load + combine both years ---
df1 = pd.read_csv(RAW + 'online_retail_09_10.csv', encoding='utf-8-sig')
df2 = pd.read_csv(RAW + 'online_retail_10_11.csv', encoding='utf-8-sig')
df = pd.concat([df1, df2], ignore_index=True)
print(f'Combined raw rows: {len(df)}')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --- cleaning (base paper Sec 4.1 steps) ---
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]  # drop cancellations
print(f'After cleaning (drop null CustomerID, qty<=0, price<=0, cancellations): {len(df)}')

df['Total'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

# --- per-customer RFM aggregation ---
T_ref = df['InvoiceDate'].max()
print(f'Reference date: {T_ref}')

agg = df.groupby('CustomerID').agg(
    last_purchase=('InvoiceDate', 'max'),
    n_orders=('InvoiceNo', 'nunique'),
    M=('Total', 'sum'),
).reset_index()
agg['R'] = (T_ref - agg['last_purchase']).dt.days
agg['repeat'] = (agg['n_orders'] > 1).astype(int)

log_qty = df.copy()
log_qty['log_qty'] = np.log1p(log_qty['Quantity'])
log_qty_sum = log_qty.groupby('CustomerID')['log_qty'].sum().rename('log_qty_sum').reset_index()
agg = agg.merge(log_qty_sum, on='CustomerID', how='left')

m = len(agg)
print(f'Unique customers: {m}')
print(f'Repeat-buyer rate: {agg["repeat"].mean():.4f}  (contrast vs Olist 0.0300 -- Online Retail II is repeat-buyer-heavy)')

# --- F* (purchase-intensity index) weight grid search: entropy-of-quintile-bands objective
#     (REVISED, consistent with Olist Step 1 fix -- variance-ratio objective was degenerate) ---
def dense_rank_score_raw(s, invert=False):
    ranks = s.rank(method='dense')
    score = np.ceil(5 * ranks / ranks.max()).astype(int).clip(1, 5)
    return 6 - score if invert else score

def band_entropy(alpha, beta, gamma, d):
    Fcand = alpha * d['n_orders'] + beta * d['log_qty_sum'] + gamma * d['repeat']
    if Fcand.nunique() <= 1:
        return -np.inf
    counts = dense_rank_score_raw(Fcand).value_counts(normalize=True)
    p = counts.values
    return -np.sum(p * np.log(p + 1e-12))

best = None
grid = np.arange(0, 1.05, 0.05)
for a in grid:
    for b in grid:
        g = round(1 - a - b, 2)
        if g < -1e-9 or g > 1 + 1e-9:
            continue
        g = max(0, g)
        ent = band_entropy(a, b, g, agg)
        if best is None or ent > best[0]:
            best = (ent, round(a, 2), round(b, 2), g)
ent, alpha, beta, gamma = best
print(f'F* weights: alpha={alpha} beta={beta} gamma={gamma} (band entropy={ent:.4f}/{np.log(5):.4f})')
agg['F_star'] = alpha * agg['n_orders'] + beta * agg['log_qty_sum'] + gamma * agg['repeat']

# --- dense-rank scoring ---
def dense_rank_score(s, invert=False):
    ranks = s.rank(method='dense')
    score = np.ceil(5 * ranks / ranks.max()).astype(int).clip(1, 5)
    return 6 - score if invert else score

agg['r_score'] = dense_rank_score(agg['R'], invert=True)
agg['f_score'] = dense_rank_score(agg['F_star'])
agg['m_score'] = dense_rank_score(agg['M'])

print('Score distributions:')
for c in ['r_score', 'f_score', 'm_score']:
    print(f'  {c}:', agg[c].value_counts().sort_index().to_dict())

agg.to_csv(OUT + 'retail2_rfm_features.csv', index=False)

# --- fuzzy membership (RFM only, 3 dims x 5 bands = 15 attrs) ---
def band_centroids(raw, score):
    return np.array([raw[score == k].median() if (score == k).any() else raw.median() for k in range(1, 6)])

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
    idx = np.clip(np.searchsorted(c, xm, side='right') - 1, 0, 3)
    left_c, right_c = c[idx], c[idx + 1]
    denom = np.where(right_c - left_c == 0, 1e-9, right_c - left_c)
    frac_right = (xm - left_c) / denom
    rows = np.where(mid)[0]
    mu[rows, idx] = 1 - frac_right
    mu[rows, idx + 1] = frac_right
    return mu

dims = {'R': (agg['R'], agg['r_score']), 'F': (agg['F_star'], agg['f_score']), 'M': (agg['M'], agg['m_score'])}
fuzzy_attrs = {}
for dname, (raw, score) in dims.items():
    c = band_centroids(raw, score)
    order = np.argsort(c)
    mu_sorted = fuzzy_membership(raw, c[order])
    mu = mu_sorted[:, np.argsort(order)]
    for k in range(5):
        fuzzy_attrs[f'{dname}{k+1}'] = mu[:, k]

fuzzy_df = pd.DataFrame(fuzzy_attrs)
attr_cols = list(fuzzy_attrs.keys())
mu_matrix = fuzzy_df[attr_cols].values
print(f'\nFuzzy membership matrix: {fuzzy_df.shape}')

# --- L-fuzzy scaling + closed itemset mining ---
L = [0.3, 0.5, 0.7]
t0 = time.time()
transactions = []
for i in range(m):
    row = []
    for j, col in enumerate(attr_cols):
        v = mu_matrix[i, j]
        for l in L:
            if v >= l:
                row.append(f'{col}@{l}')
    transactions.append(row)
print(f'Transactions built [{time.time()-t0:.1f}s]')

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)
bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
print(f'Crisp scaled context: {bin_df.shape}')

t0 = time.time()
freq = fpgrowth(bin_df, min_support=0.02, use_colnames=True, max_len=None)  # no length cap, see Step 2 note
freq['n_customers'] = (freq['support'] * m).round().astype(int)
print(f'Frequent itemsets: {len(freq)} [{time.time()-t0:.1f}s]')

freq_sorted = freq.sort_values('support', ascending=False).reset_index(drop=True)
is_closed = np.ones(len(freq_sorted), dtype=bool)
groups = freq_sorted.groupby('support').indices
for supp_val, idxs in groups.items():
    if len(idxs) < 2:
        continue
    subset = freq_sorted.iloc[idxs]
    sets_list = subset['itemsets'].tolist()
    orig_idx = subset.index.tolist()
    for a in range(len(sets_list)):
        for b in range(len(sets_list)):
            if a != b and sets_list[a] < sets_list[b]:
                is_closed[orig_idx[a]] = False
                break
freq_sorted['is_closed'] = is_closed
closed = freq_sorted[freq_sorted['is_closed']].reset_index(drop=True)
print(f'Closed itemsets (fuzzy concepts): {len(closed)}')

# --- domain-invariant lattice metrics ---
n_attrs = len(attr_cols) * len(L)  # 15 * 3 = 45 possible crisp attrs (theoretical universe for density calc)
rho = len(closed) / (2 ** len(attr_cols))  # density relative to fuzzy-attribute powerset (RFM: 2^15)
print(f'\nConcept lattice density rho = |L_f|/2^|M| = {len(closed)}/{2**len(attr_cols)} = {rho:.6f}')
print(f'  (Olist RFMS comparison: 1369/2^20 = {1369/(2**20):.6f})')

# --- extents + mean membership count ---
attr_to_extent = {}
for j, col in enumerate(attr_cols):
    v = mu_matrix[:, j]
    for l in L:
        attr_to_extent[f'{col}@{l}'] = set(np.where(v >= l)[0])

def extent_of(itemset):
    sets = [attr_to_extent[a] for a in itemset]
    sets.sort(key=len)
    result = sets[0]
    for s in sets[1:]:
        result = result & s
        if not result:
            break
    return result

t0 = time.time()
closed['extent'] = closed['itemsets'].apply(extent_of)
print(f'Extents computed [{time.time()-t0:.1f}s]')

membership_count = np.zeros(m, dtype=int)
for ext in closed['extent']:
    idx = np.fromiter(ext, dtype=np.int64, count=len(ext))
    membership_count[idx] += 1
mean_k = membership_count.mean()
print(f'Mean concept-membership count per customer: {mean_k:.2f}')
print(f'  (Olist comparison: reported separately, natural expectation similar order given fuzzy overlap)')

closed.to_pickle(RES + 'retail2_fuzzy_concepts_raw.pkl')
fuzzy_df.to_pickle(RES + 'retail2_fuzzy_membership_matrix.pkl')

# --- ARI: FCA hard clusters (top-level bands, argmax) vs K-means ---
score_tuples = agg[['r_score', 'f_score', 'm_score']].values

def stability_proxy(extent):
    n = len(extent)
    if n <= 1:
        return 1.0
    idx = np.fromiter(extent, dtype=np.int64, count=n)
    profiles = score_tuples[idx]
    return 1 - (len(np.unique(profiles, axis=0)) / n)

closed['n_customers_actual'] = closed['extent'].apply(len)
closed['stability_approx'] = closed['extent'].apply(stability_proxy)

def kneedle_threshold(values_sorted_desc):
    y = np.array(values_sorted_desc, dtype=float)
    n = len(y)
    if n < 3:
        return y[-1]
    x = np.linspace(0, 1, n)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return y[np.argmax(num / den)]

supp_min_star = kneedle_threshold(np.sort(closed['support'].values)[::-1])
theta_star = kneedle_threshold(np.sort(closed['stability_approx'].values)[::-1])
print(f'\nKneedle thresholds: supp_min*={supp_min_star:.4f}, theta*={theta_star:.4f}')

pruned = closed[(closed['support'] >= supp_min_star) & (closed['stability_approx'] >= theta_star)].copy()
print(f'Surviving concepts after pruning: {len(pruned)} (from {len(closed)})')
pruned.to_pickle(RES + 'retail2_pruned_fuzzy_concepts.pkl')
import pickle
with open(RES + 'retail2_hasse_concepts.pkl', 'wb') as f:
    pickle.dump({'pruned': pruned}, f)

def dims_touched(itemset):
    return frozenset(a.split('@')[0][0] for a in itemset)
pruned['dims_touched'] = pruned['itemsets'].apply(dims_touched)
pruned['n_dims'] = pruned['dims_touched'].apply(len)
top_level = pruned[pruned['n_dims'] == 1].copy()
top_level['band_label'] = top_level['itemsets'].apply(lambda s: sorted(s, key=len)[0].split('@')[0])
top_level_unique = top_level.sort_values('support', ascending=False).drop_duplicates(subset='band_label')
print(f'\nTop-level band concepts: {len(top_level_unique)}')
print(top_level_unique[['band_label', 'support', 'n_customers_actual']].to_string())

band_labels = top_level_unique['band_label'].tolist()
band_membership = fuzzy_df[band_labels].values
fca_hard_labels = np.argmax(band_membership, axis=1)
agg['hard_cluster'] = [band_labels[i] for i in fca_hard_labels]
agg.to_csv(OUT + 'retail2_rfm_with_hard_clusters.csv', index=False)

X = StandardScaler().fit_transform(agg[['r_score', 'f_score', 'm_score']].values)
km = KMeans(n_clusters=len(band_labels), random_state=42, n_init=10).fit(X)
ari = adjusted_rand_score(fca_hard_labels, km.labels_)
print(f'\nARI (FCA hard clusters vs K-means, k={len(band_labels)}): {ari:.4f}')

sil_fca = silhouette_score(X, fca_hard_labels, sample_size=min(5000, m), random_state=42)
db_fca = davies_bouldin_score(X, fca_hard_labels)
sil_km = silhouette_score(X, km.labels_, sample_size=min(5000, m), random_state=42)
db_km = davies_bouldin_score(X, km.labels_)
print(f'FCA hard (k={len(band_labels)}): Silhouette={sil_fca:.4f}, DB={db_fca:.4f}')
print(f'K-means (k={len(band_labels)}):  Silhouette={sil_km:.4f}, DB={db_km:.4f}')

cross_domain_results = {
    'dataset': 'Online Retail II',
    'n_customers': m,
    'repeat_buyer_rate': agg['repeat'].mean(),
    'n_raw_concepts': len(closed),
    'n_pruned_concepts': len(pruned),
    'lattice_density_rho': rho,
    'mean_membership_count': mean_k,
    'n_top_level_bands': len(band_labels),
    'ari_fca_vs_kmeans': ari,
    'silhouette_fca': sil_fca,
    'db_fca': db_fca,
    'silhouette_kmeans': sil_km,
    'db_kmeans': db_km,
}
pd.DataFrame([cross_domain_results]).to_csv(RES + 'cross_domain_comparison.csv', index=False)
print(f'\nSaved: {RES}cross_domain_comparison.csv')

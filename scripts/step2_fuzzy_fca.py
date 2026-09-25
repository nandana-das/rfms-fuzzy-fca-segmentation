"""
STEP 2: Fuzzy formal context + closed frequent itemset mining (methodology Sec 3.2-3.4)
- Centroid-based piecewise-linear fuzzy membership per dim (band centroids, ascending-order fix for inverted R)
- L-fuzzy scaling (Belohlavek reduction) at L={0.3,0.5,0.7}
- FP-growth closed itemsets = fuzzy formal concepts (tractable at 93k scale)
"""
import pandas as pd
import numpy as np
import time
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
OUT = PROCESSED
RES = RESULTS

agg = pd.read_csv(OUT + 'olist_rfms_features.csv')
m = len(agg)
print('Customers:', m)

def band_centroids(raw, score):
    c = []
    for k in range(1, 6):
        vals = raw[score == k]
        c.append(vals.median() if len(vals) else raw.median())
    return np.array(c)

def fuzzy_membership(raw, centroids):
    """Vectorized centroid-based piecewise-linear fuzzy partition across 5 ascending-ordered bands
    (middle bands are triangular; outer bands 1 and 5 saturate to 1.0 beyond their centroid --
    technically shoulder/trapezoidal at the extremes, not pure triangular)."""
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
    'R': (agg['R'], agg['r_score']),
    'F': (agg['F_star'], agg['f_score']),
    'M': (agg['M'], agg['m_score']),
    'S': (agg['S'], agg['s_score']),
}

fuzzy_attrs = {}
for dname, (raw, score) in dims.items():
    c = band_centroids(raw, score)
    order = np.argsort(c)          # handles R's inverted (descending) centroid order
    c_sorted = c[order]
    mu_sorted = fuzzy_membership(raw, c_sorted)
    inv_order = np.argsort(order)
    mu = mu_sorted[:, inv_order]   # mu[:,k] = membership in score-band k+1
    for k in range(5):
        fuzzy_attrs[f'{dname}{k+1}'] = mu[:, k]

fuzzy_df = pd.DataFrame(fuzzy_attrs)
attr_cols = list(fuzzy_attrs.keys())
mu_matrix = fuzzy_df[attr_cols].values
print('Fuzzy membership matrix:', fuzzy_df.shape)
print('Row-sum sanity check (should be 1.0 per dim):')
for d in dims:
    cols = [c for c in attr_cols if c.startswith(d)]
    print(f'  {d}:', fuzzy_df[cols].sum(axis=1).round(6).unique()[:3])

# --- L-fuzzy scaling -> crisp multi-threshold context ---
L = [0.3, 0.5, 0.7]
t0 = time.time()
transactions = []
for i in range(m):
    row_items = []
    for j, col in enumerate(attr_cols):
        v = mu_matrix[i, j]
        for l in L:
            if v >= l:
                row_items.append(f'{col}@{l}')
    transactions.append(row_items)
print(f'Transactions built [{time.time()-t0:.1f}s], avg items/customer: {np.mean([len(t) for t in transactions]):.2f}')

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)
bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
print('Crisp scaled context shape:', bin_df.shape)

# --- closed frequent itemset mining (fuzzy formal concepts) ---
t0 = time.time()
min_support = 0.02
freq = fpgrowth(bin_df, min_support=min_support, use_colnames=True, max_len=None)  # no length cap:
# an earlier max_len=6 was arbitrary and actively binding (9,618/10,283 concepts sat exactly at the
# size-6 wall); uncapped mining found true max itemset length 12 and runs in ~1s at this scale, so
# there is no performance justification for capping.
freq['n_customers'] = (freq['support'] * m).round().astype(int)
print(f'Frequent itemsets (support>={min_support}): {len(freq)}  [{time.time()-t0:.1f}s]')

# closed-itemset filter: grouped by exact support (avoids O(n^2) all-pairs)
t0 = time.time()
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
print(f'Closed itemsets (= fuzzy formal concepts): {len(closed)}  [{time.time()-t0:.1f}s]')

fuzzy_df.to_pickle(RES + 'fuzzy_membership_matrix.pkl')
closed.to_pickle(RES + 'fuzzy_concepts_raw.pkl')
print(closed[['itemsets', 'support', 'n_customers']].sort_values('n_customers', ascending=False).head(15).to_string())
print(f'\nSaved: {RES}fuzzy_concepts_raw.pkl, fuzzy_membership_matrix.pkl')

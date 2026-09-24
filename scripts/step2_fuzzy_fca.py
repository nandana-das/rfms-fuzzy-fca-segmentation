"""
Steps 6-9 Algorithm 2:
- Build triangular fuzzy membership per dim (R,F*,M,S) over 5 bands (Sec 3.2)
- L-fuzzy scaling (thresholds L={0.3,0.5,0.7}) -> crisp context (standard Belohlavek reduction)
- Closed frequent itemset mining (fpgrowth) = fuzzy formal concepts (tractable at 93k scale)
- Approx stability index (sampling) + support -> iceberg pruning (elbow method, Sec 3.5)
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

agg = pd.read_csv('data/olist_rfms_features.csv')
m = len(agg)
print('Customers:', m)

# --- Step 6: band centroids per dim from crisp score groups, then triangular membership ---
def band_centroids(raw, score):
    c = []
    for k in range(1, 6):
        vals = raw[score == k]
        c.append(vals.median() if len(vals) else raw.median())
    return np.array(c)

def fuzzy_membership(raw, centroids):
    """Linear-interpolation triangular fuzzy partition across 5 bands. Returns (n,5) matrix."""
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
    # find interval j such that c[j] <= x < c[j+1]
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
    'R': (agg['R'], agg['r_score'], True),
    'F': (agg['F_star'], agg['f_score'], False),
    'M': (agg['M'], agg['m_score'], False),
    'S': (agg['S'], agg['s_score'], False),
}

fuzzy_attrs = {}  # name -> (n,) membership array
for dname, (raw, score, inv) in dims.items():
    c = band_centroids(raw, score)          # index k=0..4 <-> score k+1, may be non-monotonic (e.g. R inverted)
    order = np.argsort(c)                    # ascending raw-value order needed by fuzzy_membership
    c_sorted = c[order]
    mu_sorted = fuzzy_membership(raw, c_sorted)   # cols in ascending-raw order
    inv_order = np.argsort(order)             # map back: score-band k -> its column in mu_sorted
    mu = mu_sorted[:, inv_order]               # mu[:,k] = membership in score-band k+1
    for k in range(5):
        band_label = k + 1
        fuzzy_attrs[f'{dname}{band_label}'] = mu[:, k]

fuzzy_df = pd.DataFrame(fuzzy_attrs)
fuzzy_df.insert(0, 'customer_unique_id', agg['customer_unique_id'].values)
print('Fuzzy membership matrix:', fuzzy_df.shape)
print(fuzzy_df.iloc[:5, 1:6])

# --- Step 7: L-fuzzy scaling -> crisp multi-threshold context (Belohlavek scaling reduction) ---
L = [0.3, 0.5, 0.7]
transactions = []
attr_cols = [c for c in fuzzy_df.columns if c != 'customer_unique_id']
mu_matrix = fuzzy_df[attr_cols].values

for i in range(m):
    row_items = []
    for j, col in enumerate(attr_cols):
        v = mu_matrix[i, j]
        for l in L:
            if v >= l:
                row_items.append(f'{col}@{l}')
    transactions.append(row_items)

avg_items = np.mean([len(t) for t in transactions])
print(f'Avg crisp items per customer (post L-scaling): {avg_items:.2f}')

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)
bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
print('Crisp scaled context shape:', bin_df.shape)

import time
t0 = time.time()
# --- Step 8: closed frequent itemsets = fuzzy formal concepts (support-pruned, tractable) ---
min_support = 0.02  # initial; refined via elbow below
freq = fpgrowth(bin_df, min_support=min_support, use_colnames=True, max_len=6)
freq['n_customers'] = (freq['support'] * m).round().astype(int)
print(f'Frequent itemsets (support>={min_support}): {len(freq)}  [{time.time()-t0:.1f}s]')
freq.to_pickle('results/freq_itemsets_raw.pkl')

# filter to CLOSED itemsets: itemset X is closed if no superset Y with SAME support exists.
# group by support value first (closed check only needed within equal-support groups) -> avoids O(n^2) over all pairs
t0 = time.time()
freq_sorted = freq.sort_values('support', ascending=False).reset_index(drop=True)
freq_sorted['size'] = freq_sorted['itemsets'].apply(len)
is_closed = np.ones(len(freq_sorted), dtype=bool)

groups = freq_sorted.groupby('support').indices  # support -> row indices with that exact support
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
print(f'Closed itemsets (= formal concepts): {len(closed)}  [{time.time()-t0:.1f}s]')

closed.to_pickle('results/fuzzy_concepts_raw.pkl')
print(closed[['itemsets', 'support', 'n_customers']].sort_values('n_customers', ascending=False).head(15))

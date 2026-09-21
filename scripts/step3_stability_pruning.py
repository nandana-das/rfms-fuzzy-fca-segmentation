"""
Step 9-10 Algorithm 2 (Sec 3.5):
- Approx stability index (subsampling, since exact is exponential -> Kuznetsov approx-stability, standard practice)
- Elbow method on stability + support distributions -> derive theta*, supp_min (not arbitrary)
- Build Hasse diagram edges among surviving concepts (subset relation, covering pairs only)
"""
import pandas as pd
import numpy as np
import pickle
import time

import os
agg = pd.read_csv('/home/claude/olist_rfms_features.csv')
m = len(agg)
CACHE = '/home/claude/closed_with_stability_cache.pkl'

if os.path.exists(CACHE):
    closed = pd.read_pickle(CACHE)
    print('Loaded cached concepts+stability:', len(closed))
else:
    closed = pd.read_pickle('/home/claude/fuzzy_concepts_raw.pkl')
    print('Loaded concepts:', len(closed))

    dims = ['R', 'F', 'M', 'S']
    L = [0.3, 0.5, 0.7]
    exec(open('/home/claude/step2_fuzzy_fca.py').read().split("# --- Step 7")[0])
    attr_cols = [c for c in fuzzy_df.columns if c != 'customer_unique_id']
    mu_matrix = fuzzy_df[attr_cols].values

    attr_to_extent = {}
    for col in attr_cols:
        v = mu_matrix[:, attr_cols.index(col)]
        for l in L:
            attr_to_extent[f'{col}@{l}'] = set(np.where(v >= l)[0])

    t0 = time.time()
    def extent_of(itemset):
        sets = [attr_to_extent[a] for a in itemset]
        sets.sort(key=len)
        result = sets[0]
        for s in sets[1:]:
            result = result & s
            if not result:
                break
        return result

    closed['extent'] = closed['itemsets'].apply(extent_of)
    print(f'Extents computed [{time.time()-t0:.1f}s]')

    score_tuples = agg[['r_score', 'f_score', 'm_score', 's_score']].values

    t0 = time.time()
    def stability_proxy(extent):
        n = len(extent)
        if n <= 1:
            return 1.0
        idx = np.fromiter(extent, dtype=np.int64, count=n)
        profiles = score_tuples[idx]
        n_unique = len(np.unique(profiles, axis=0))
        return 1 - (n_unique / n)

    closed['n_customers_actual'] = closed['extent'].apply(len)
    closed['stability_approx'] = closed['extent'].apply(stability_proxy)
    print(f'Stability computed [{time.time()-t0:.1f}s]')
    closed.to_pickle(CACHE)

# --- Kneedle elbow method (Satopaa et al. 2011): normalize curve, max distance from chord ---
def kneedle_threshold(values_sorted_desc):
    y = np.array(values_sorted_desc, dtype=float)
    n = len(y)
    if n < 3:
        return y[-1]
    x = np.linspace(0, 1, n)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # chord from (0, y_norm[0]) to (1, y_norm[-1]); distance of each point from chord
    x1, y1 = x[0], y_norm[0]
    x2, y2 = x[-1], y_norm[-1]
    # perpendicular distance from line (x1,y1)-(x2,y2)
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    knee_idx = np.argmax(dist)
    return y[knee_idx], knee_idx

supp_sorted = np.sort(closed['support'].values)[::-1]
supp_min_star, supp_knee_idx = kneedle_threshold(supp_sorted)
print(f'Kneedle-derived supp_min* = {supp_min_star:.4f} (knee at rank {supp_knee_idx}/{len(supp_sorted)})')

stab_sorted = np.sort(closed['stability_approx'].values)[::-1]
theta_star, stab_knee_idx = kneedle_threshold(stab_sorted)
print(f'Kneedle-derived theta* (stability) = {theta_star:.4f} (knee at rank {stab_knee_idx}/{len(stab_sorted)})')

pruned = closed[(closed['support'] >= supp_min_star) & (closed['stability_approx'] >= theta_star)].copy()
pruned = pruned.sort_values('n_customers_actual', ascending=False).reset_index(drop=True)
print(f'\nSurviving concepts after iceberg pruning: {len(pruned)} (from {len(closed)})')
print(pruned[['itemsets', 'support', 'n_customers_actual', 'stability_approx']].head(20))

pruned.to_pickle('/home/claude/pruned_fuzzy_concepts.pkl')

# --- Hasse diagram: covering relation among pruned concepts (subset relation on itemsets, no intermediate) ---
t0 = time.time()
itemsets_list = pruned['itemsets'].tolist()
n_c = len(itemsets_list)
edges = []
for i in range(n_c):
    for j in range(n_c):
        if i == j:
            continue
        if itemsets_list[j] < itemsets_list[i]:  # j is more general (fewer attrs) = parent of i
            # check no intermediate k with itemsets[j] < itemsets[k] < itemsets[i]
            is_cover = True
            for k in range(n_c):
                if k != i and k != j and itemsets_list[j] < itemsets_list[k] < itemsets_list[i]:
                    is_cover = False
                    break
            if is_cover:
                edges.append((j, i))  # parent -> child
print(f'Hasse edges: {len(edges)} [{time.time()-t0:.1f}s]')

with open('/home/claude/hasse_edges.pkl', 'wb') as f:
    pickle.dump({'edges': edges, 'concepts': pruned}, f)

print('\nSaved pruned_fuzzy_concepts.pkl and hasse_edges.pkl')

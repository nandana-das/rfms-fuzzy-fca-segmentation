"""
STEP 11: Additional fuzzy-validity diagnostics (Outstanding item #4, both audits)
Covers: (a) membership entropy, (b) alpha-cut sensitivity, (c) L-fuzzy threshold sensitivity,
(d) resampling stability / concept persistence.
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
fuzzy_df = pd.read_pickle(RES + 'fuzzy_membership_matrix.pkl')
attr_cols = list(fuzzy_df.columns)
mu_matrix = fuzzy_df[attr_cols].values
dims = ['R', 'F', 'M', 'S']
score_tuples = agg[['r_score', 'f_score', 'm_score', 's_score']].values

# ============================================================
# (a) Membership entropy: how "genuinely fuzzy" is each customer's assignment per dimension?
# ============================================================
print('=== (a) Membership entropy ===')
entropy_by_dim = {}
for d in dims:
    cols_idx = [j for j, c in enumerate(attr_cols) if c.startswith(d)]
    mu_d = mu_matrix[:, cols_idx]
    p = np.clip(mu_d, 1e-12, 1.0)
    H = -np.sum(mu_d * np.log(p), axis=1)  # 0 = fully crisp (one band=1), ln(5)=2.32 = maximally spread
    entropy_by_dim[d] = H
    frac_crisp = np.mean(H < 1e-6)
    print(f'  {d}: mean H={H.mean():.4f} (max possible ln(5)={np.log(5):.4f}), '
          f'{100*frac_crisp:.1f}% fully crisp (H=0, raw value exactly at/beyond an outer centroid), '
          f'{100*np.mean(H > 0.1):.1f}% meaningfully fuzzy (H>0.1)')
mean_entropy_all = np.mean([entropy_by_dim[d] for d in dims], axis=0)
print(f'  Overall (mean across R/F/M/S): mean H={mean_entropy_all.mean():.4f}, '
      f'std={mean_entropy_all.std():.4f}')
print('  Reading: near-zero mean entropy would mean the fuzzy representation is degenerating back to')
print('  crisp scoring in practice (no real fuzziness surviving); a healthy positive mean indicates')
print('  genuine partial membership is common, not an edge-case rarity.')

# ============================================================
# (b) Alpha-cut sensitivity: rerun hard-cluster overlap stats at alpha=0.4/0.5/0.6
# ============================================================
print('\n=== (b) Alpha-cut sensitivity (top-level band assignment) ===')
hasse = pd.read_pickle(RES + 'hasse_edges.pkl')
pruned = hasse['concepts']
edges = hasse['edges']

def dims_touched(itemset):
    return frozenset(a.split('@')[0][0] for a in itemset)

pruned = pruned.reset_index(drop=True).copy()
pruned['dims_touched'] = pruned['itemsets'].apply(dims_touched)
pruned['n_dims'] = pruned['dims_touched'].apply(len)
has_parent = set(child for (_, child) in edges)
top_level_idx = [i for i in range(len(pruned)) if pruned.loc[i, 'n_dims'] == 1 and i not in has_parent]
top_level = pruned.loc[top_level_idx].copy()
top_level['band_label'] = top_level['itemsets'].apply(lambda s: sorted(s, key=len)[0].split('@')[0])
top_level_unique = top_level.sort_values('support', ascending=False).drop_duplicates(subset='band_label')
print(f'  Top-level single-dim bands (fixed across alpha, from pruned lattice): {len(top_level_unique)}')

band_membership = {}
for _, row in top_level_unique.iterrows():
    label = row['band_label']
    dim_letter = label[0]
    band_idx = int(label[1:]) - 1
    col_idx = [j for j, c in enumerate(attr_cols) if c.startswith(dim_letter)][band_idx]
    band_membership[label] = mu_matrix[:, col_idx]

for alpha in [0.4, 0.5, 0.6]:
    n_bands_satisfying = np.zeros(m, dtype=int)
    for label, mu in band_membership.items():
        n_bands_satisfying += (mu >= alpha).astype(int)
    pct_multi = 100 * np.mean(n_bands_satisfying > 1)
    pct_zero = 100 * np.mean(n_bands_satisfying == 0)
    print(f'  alpha={alpha}: {pct_multi:.2f}% satisfy >1 band, {pct_zero:.2f}% satisfy 0 bands '
          f'(fallback to argmax), mean bands/customer={n_bands_satisfying.mean():.2f}')
print('  Reading: overlap percentage should move smoothly with alpha (lower alpha -> more overlap,')
print('  higher alpha -> less), with no discontinuity -- a sanity check that alpha=0.5 is not a')
print('  knife-edge choice.')

# ============================================================
# (c) L-fuzzy threshold sensitivity: rerun scaling+mining at alternate threshold sets
# ============================================================
print('\n=== (c) L-fuzzy threshold sensitivity (closed-concept count under alternate thresholds) ===')

def mine_closed(mu_matrix, attr_cols, L, min_support=0.02):
    t0 = time.time()
    m_ = mu_matrix.shape[0]
    transactions = []
    for i in range(m_):
        row_items = []
        for j, col in enumerate(attr_cols):
            v = mu_matrix[i, j]
            for l in L:
                if v >= l:
                    row_items.append(f'{col}@{l}')
        transactions.append(row_items)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    freq = fpgrowth(bin_df, min_support=min_support, use_colnames=True, max_len=None)
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
    closed = freq_sorted[is_closed].reset_index(drop=True)
    return len(freq_sorted), len(closed), time.time() - t0

for L, tag in [([0.3, 0.5, 0.7], 'baseline'), ([0.2, 0.4, 0.6], 'shifted-down'),
               ([0.25, 0.5, 0.75], 'wider-spread')]:
    n_raw, n_closed, dt = mine_closed(mu_matrix, attr_cols, L)
    print(f'  L={L} ({tag}): {n_raw} raw itemsets -> {n_closed} closed concepts  [{dt:.1f}s]')
print('  Reading: the baseline {0.3,0.5,0.7} threshold set is one defensible choice; closed-concept')
print('  count moving smoothly (not collapsing to ~0 or exploding) under nearby threshold sets')
print('  indicates the lattice size is not an artifact of that specific choice.')

# ============================================================
# (d) Resampling stability / concept persistence
# ============================================================
print('\n=== (d) Resampling stability: concept persistence under 80% subsampling ===')
baseline_pruned_itemsets = set(frozenset(s) for s in pruned['itemsets'])
n_baseline = len(baseline_pruned_itemsets)
print(f'  Baseline pruned lattice: {n_baseline} concepts')

N_BOOT = 10
rng = np.random.default_rng(42)
persistence_counts = {s: 0 for s in baseline_pruned_itemsets}
raw_concept_counts = []
for rep in range(N_BOOT):
    idx = rng.choice(m, size=int(0.8 * m), replace=False)
    sub_mu = mu_matrix[idx]
    n_raw, n_closed_boot, dt = mine_closed(sub_mu, attr_cols, [0.3, 0.5, 0.7])
    raw_concept_counts.append(n_closed_boot)
    # recompute closed itemsets as a set for overlap check
    transactions = []
    for i in range(sub_mu.shape[0]):
        row_items = []
        for j, col in enumerate(attr_cols):
            v = sub_mu[i, j]
            for l in [0.3, 0.5, 0.7]:
                if v >= l:
                    row_items.append(f'{col}@{l}')
        transactions.append(row_items)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    bin_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    freq = fpgrowth(bin_df, min_support=0.02, use_colnames=True, max_len=None)
    boot_itemsets = set(frozenset(s) for s in freq['itemsets'])
    for s in baseline_pruned_itemsets:
        if s in boot_itemsets:
            persistence_counts[s] += 1
    print(f'  rep {rep+1}/{N_BOOT}: n=80% subsample, {n_closed_boot} closed concepts found '
          f'[{dt:.1f}s], {sum(1 for s in baseline_pruned_itemsets if s in boot_itemsets)}/{n_baseline} '
          f'baseline pruned concepts still appear as frequent itemsets')

persistence_frac = np.array(list(persistence_counts.values())) / N_BOOT
print(f'\n  Concept persistence across {N_BOOT} subsamples (fraction of reps a baseline pruned concept')
print(f'  reappears as a frequent itemset, before re-pruning): mean={persistence_frac.mean():.3f}, '
      f'median={np.median(persistence_frac):.3f}')
print(f'  {100*np.mean(persistence_frac == 1.0):.1f}% of baseline concepts appear in ALL {N_BOOT} '
      f'subsamples; {100*np.mean(persistence_frac >= 0.8):.1f}% appear in >=80% of subsamples')
print(f'  Closed-concept count across subsamples: {raw_concept_counts} (baseline full-population: 1369)')

summary = pd.DataFrame({
    'itemset': ['+'.join(sorted(s)) for s in baseline_pruned_itemsets],
    'persistence_fraction': persistence_frac,
})
summary.to_csv(RES + 'concept_persistence.csv', index=False)
print(f'\nSaved: {RES}concept_persistence.csv')

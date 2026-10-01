"""
STEP 12: Retail II parity tests (Steps 9-11 equivalent for the cross-domain dataset)
Olist's dominant crisp-tie group was f_score=1 (97.00%, frequency-tied, since the marketplace is
sparse). Retail II is NOT frequency-sparse (72.39% repeat rate) -- its dominant crisp-tie group is
r_score=5 (54.79%, recency-tied: most customers in this repeat-heavy retailer ordered recently).
Running the same three tests here (structure recovery, overlap profiles, validity diagnostics)
checks whether the Olist findings are domain-specific or hold symmetrically cross-domain.
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

hc = pd.read_csv(OUT + 'retail2_rfm_with_hard_clusters.csv')
fuzzy_df = pd.read_pickle(RES + 'retail2_fuzzy_membership_matrix.pkl')
attr_cols = list(fuzzy_df.columns)
mu_matrix = fuzzy_df[attr_cols].values
m = len(hc)
dims = ['R', 'F', 'M']  # no S in Retail II
print(f'Retail II customers: {m}')

with open(RES + 'retail2_hasse_concepts.pkl', 'rb') as f:
    import pickle
    pruned = pickle.load(f)['pruned']

# ============================================================
# PART A (analog Step 9): r_score=5 structure-recovery test
# ============================================================
print('\n' + '=' * 70)
print('PART A: r_score=5 group structure-recovery test (analog of Olist f_score=1)')
print('=' * 70)

r5 = hc[hc['r_score'] == 5].copy()
n_r5 = len(r5)
print(f'r_score=5 customers: {n_r5} ({100*n_r5/m:.2f}% of {m})')

print('\n=== Hard-cluster fan-out ===')
dist = r5['hard_cluster'].value_counts()
print(dist.to_string())
non_r5_cluster = (~r5['hard_cluster'].str.startswith('R')).sum()
print(f'\nr_score=5 customers assigned to a NON-R hard_cluster (driven by F/M instead): '
      f'{non_r5_cluster} ({100*non_r5_cluster/n_r5:.2f}%)')

print('\n=== Concept-membership fan-out ===')
membership_count = np.zeros(m, dtype=int)
for ext in pruned['extent']:
    idx = np.fromiter(ext, dtype=np.int64, count=len(ext))
    membership_count[idx] += 1
hc['n_concepts'] = membership_count
r5 = hc[hc['r_score'] == 5].copy()
overall_std = hc['n_concepts'].std()
r5_std = r5['n_concepts'].std()
print(f'Overall population: mean={hc["n_concepts"].mean():.2f}, std={overall_std:.2f}')
print(f'r_score=5 subgroup:  mean={r5["n_concepts"].mean():.2f}, std={r5_std:.2f}')
print(f'Ratio within-group/overall std: {r5_std/overall_std:.3f}')

# Retail II's "residual signal" analog: R itself is continuous (raw days), unlike Olist's F* which
# was a constructed index -- test whether raw R correlates with downstream structure within the group
# (should be near-zero by construction since r_score=5 already IS the top R band, i.e. this tests
# whether being "more recent within recent" matters, analogous to F* within f_score=1)
corr_r_nconcepts = r5['R'].corr(r5['n_concepts'])
print(f'\ncorr(R, n_concepts) within r_score=5: {corr_r_nconcepts:.4f}')
print('(Analog of the Olist F* correlation test -- checks whether within-band recency variation')
print('itself drives downstream structure, or whether F/M dimensions are doing the differentiating.)')

# ============================================================
# PART B (analog Step 10): overlap-segment profiles
# ============================================================
print('\n' + '=' * 70)
print('PART B: Interpretable overlap-segment profiles')
print('=' * 70)

pruned = pruned.reset_index(drop=True).copy()
def dims_touched(itemset):
    return frozenset(a.split('@')[0][0] for a in itemset)
pruned['dims_touched'] = pruned['itemsets'].apply(dims_touched)
top_level = pruned[pruned['dims_touched'].apply(len) == 1].copy()
top_level['band_label'] = top_level['itemsets'].apply(lambda s: sorted(s, key=len)[0].split('@')[0])
top_level_unique = top_level.sort_values('support', ascending=False).drop_duplicates(subset='band_label')
band_cols = top_level_unique['band_label'].tolist()
print(f'Top-level bands: {band_cols}')

sig_bin = (fuzzy_df[band_cols] >= 0.5).astype(int)
hc2 = hc.reset_index(drop=True)
hc2['bands_on'] = sig_bin.apply(lambda row: tuple(c for c, v in zip(band_cols, row) if v == 1), axis=1)
hc2['n_dims_touched'] = sig_bin.apply(lambda row: len(set(c[0] for c, v in zip(band_cols, row) if v == 1)), axis=1)

nontrivial = hc2[hc2['n_dims_touched'] >= 2]  # only 3 dims total (R/F/M), so >=2 = nontrivial here
sig_counts = nontrivial['bands_on'].value_counts()
print(f'Nontrivial (>=2 dims, out of 3) signatures: {len(sig_counts)} distinct, '
      f'{len(nontrivial)} customers ({100*len(nontrivial)/m:.1f}%)')

TOP_K = 5
top_sigs = sig_counts.head(TOP_K)
rows_out = []
for sig, cnt in top_sigs.items():
    sub = hc2[hc2['bands_on'] == sig]
    hard_labels = sub['hard_cluster'].value_counts()
    print(f'\nSignature {sig}  (n={cnt}, {100*cnt/m:.2f}%)')
    print(f'  hard_cluster labels: {dict(hard_labels)}')
    print(f'  within-signature spread: R=[{sub["R"].min():.0f},{sub["R"].max():.0f}]d, '
          f'M=[{sub["M"].min():.2f},{sub["M"].max():.2f}], '
          f'n_orders=[{sub["n_orders"].min()},{sub["n_orders"].max()}]')
    for _, r in sub.head(3).iterrows():
        rows_out.append({'signature': '+'.join(sig), 'n_in_signature': cnt,
                          'CustomerID': r['CustomerID'], 'R': r['R'], 'M': r['M'],
                          'n_orders': r['n_orders'], 'hard_cluster': r['hard_cluster']})

# collapse test: what does the dominant hard_cluster="R5" alone hide?
r5_hard = hc2[hc2['hard_cluster'] == 'R5']
print(f'\n=== What hard_cluster="R5" alone collapses (n={len(r5_hard)}) ===')
print(f'  M: min={r5_hard["M"].min():.2f}, max={r5_hard["M"].max():.2f}, median={r5_hard["M"].median():.2f}')
print(f'  n_orders: min={r5_hard["n_orders"].min()}, max={r5_hard["n_orders"].max()}, '
      f'median={r5_hard["n_orders"].median():.0f}')
print(f'  Distinct overlap signatures within hard_cluster=R5: {r5_hard["bands_on"].nunique()}')

pd.DataFrame(rows_out).to_csv(RES + 'retail2_overlap_profile_examples.csv', index=False)

# ============================================================
# PART C (analog Step 11): fuzzy-validity diagnostics
# ============================================================
print('\n' + '=' * 70)
print('PART C: Fuzzy-validity diagnostics')
print('=' * 70)

print('\n=== (a) Membership entropy ===')
for d in dims:
    cols_idx = [j for j, c in enumerate(attr_cols) if c.startswith(d)]
    mu_d = mu_matrix[:, cols_idx]
    p = np.clip(mu_d, 1e-12, 1.0)
    H = -np.sum(mu_d * np.log(p), axis=1)
    print(f'  {d}: mean H={H.mean():.4f} (max={np.log(5):.4f}), '
        f'{100*np.mean(H < 1e-6):.1f}% fully crisp, {100*np.mean(H > 0.1):.1f}% meaningfully fuzzy')

print('\n=== (b) Alpha-cut sensitivity ===')
band_membership_mu = {label: mu_matrix[:, [j for j, c in enumerate(attr_cols) if c.startswith(label[0])][int(label[1:])-1]]
                       for label in band_cols}
for alpha in [0.4, 0.5, 0.6]:
    n_bands_satisfying = np.zeros(m, dtype=int)
    for label, mu in band_membership_mu.items():
        n_bands_satisfying += (mu >= alpha).astype(int)
    print(f'  alpha={alpha}: {100*np.mean(n_bands_satisfying > 1):.2f}% satisfy >1 band, '
          f'mean bands/customer={n_bands_satisfying.mean():.2f}')

print('\n=== (c) L-fuzzy threshold sensitivity ===')
def mine_closed(mu_matrix, attr_cols, L, min_support=0.02):
    t0 = time.time()
    transactions = []
    for i in range(mu_matrix.shape[0]):
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
    print(f'  L={L} ({tag}): {n_raw} raw -> {n_closed} closed  [{dt:.1f}s]')

print('\n=== (d) Resampling stability ===')
baseline_itemsets = set(frozenset(s) for s in pruned['itemsets'])
n_baseline = len(baseline_itemsets)
N_BOOT = 10
rng = np.random.default_rng(42)
persistence_counts = {s: 0 for s in baseline_itemsets}
raw_counts = []
for rep in range(N_BOOT):
    idx = rng.choice(m, size=int(0.8 * m), replace=False)
    sub_mu = mu_matrix[idx]
    n_raw_boot, n_closed_boot, dt = mine_closed(sub_mu, attr_cols, [0.3, 0.5, 0.7])
    raw_counts.append(n_closed_boot)
    # separate raw-frequent-itemset pass for the membership check (superset of closed; a valid,
    # if slightly generous, persistence test -- closed itemsets are always also frequent itemsets)
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
    for s in baseline_itemsets:
        if s in boot_itemsets:
            persistence_counts[s] += 1
    n_survive = sum(1 for s in baseline_itemsets if s in boot_itemsets)
    print(f'  rep {rep+1}/{N_BOOT}: {n_closed_boot} closed concepts [{dt:.1f}s], '
          f'{n_survive}/{n_baseline} baseline concepts persist')

persistence_frac = np.array(list(persistence_counts.values())) / N_BOOT
print(f'\nConcept persistence: mean={persistence_frac.mean():.3f}, '
      f'{100*np.mean(persistence_frac == 1.0):.1f}% persist in all {N_BOOT} reps')
print(f'Raw closed-concept counts across subsamples: {raw_counts} (baseline: {n_baseline})')

pd.DataFrame({'itemset': ['+'.join(sorted(s)) for s in baseline_itemsets],
              'persistence_fraction': persistence_frac}).to_csv(
    RES + 'retail2_concept_persistence.csv', index=False)
print(f'\nSaved: retail2_overlap_profile_examples.csv, retail2_concept_persistence.csv')

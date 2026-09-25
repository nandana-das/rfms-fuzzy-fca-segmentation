"""
STEP 4: Alpha-cut hard cluster assignment (methodology Algorithm 2 Step 11)
- Identify top-level single-dimension concepts (directly below top in Hasse diagram)
- alpha-cut at alpha=0.5 on raw fuzzy membership -> mode (max-membership) assignment
- Preserve full soft membership vector per customer for business interpretation
"""
import pandas as pd
import numpy as np
import pickle
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
RES = RESULTS
OUT = PROCESSED

agg = pd.read_csv(OUT + 'olist_rfms_features.csv')
m = len(agg)
fuzzy_df = pd.read_pickle(RES + 'fuzzy_membership_matrix.pkl')

with open(RES + 'hasse_edges.pkl', 'rb') as f:
    hasse = pickle.load(f)
pruned = hasse['concepts']
edges = hasse['edges']

# --- identify top-level concepts: single-attribute-DIMENSION concepts with no parent in pruned lattice ---
# (a concept's "dimension set" = which of R/F/M/S it touches, ignoring the @threshold multiplicity)
def dims_touched(itemset):
    return frozenset(a.split('@')[0][0] for a in itemset)  # first char = R/F/M/S

pruned = pruned.reset_index(drop=True)
pruned['dims_touched'] = pruned['itemsets'].apply(dims_touched)
pruned['n_dims'] = pruned['dims_touched'].apply(len)

# children indices (has a parent edge pointing to it)
has_parent = set(child for (_, child) in edges)
top_level_idx = [i for i in range(len(pruned)) if pruned.loc[i, 'n_dims'] == 1 and i not in has_parent]
top_level = pruned.loc[top_level_idx].copy()
print(f'Top-level single-dimension concepts: {len(top_level)}')

# collapse to one representative concept per (dimension, band) -- take the tightest (highest support) L-cut variant already dominant
top_level['band_label'] = top_level['itemsets'].apply(lambda s: sorted(s, key=len)[0].split('@')[0])  # e.g. 'F1'
top_level_unique = top_level.sort_values('support', ascending=False).drop_duplicates(subset='band_label')
print(f'Unique top-level band concepts: {len(top_level_unique)}')
print(top_level_unique[['band_label', 'support', 'n_customers_actual']].to_string())

band_labels = top_level_unique['band_label'].tolist()

# --- raw fuzzy membership per band (not L-thresholded) -> for alpha-cut ---
band_membership = fuzzy_df[band_labels].values  # (m, n_bands)

ALPHA = 0.5
satisfies = band_membership >= ALPHA  # (m, n_bands) boolean

# mode assignment: customer -> band with MAX membership among those satisfying alpha-cut;
# if none satisfy alpha-cut, assign to band of global max membership anyway (fallback, documented)
max_band_idx = np.argmax(band_membership, axis=1)
any_satisfies = satisfies.any(axis=1)
# where satisfies has ties/multiple True, pick max-membership among satisfying ones only
masked_membership = np.where(satisfies, band_membership, -1)
assigned_idx = np.where(any_satisfies, np.argmax(masked_membership, axis=1), max_band_idx)

agg['hard_cluster'] = [band_labels[i] for i in assigned_idx]
agg['hard_cluster_membership'] = band_membership[np.arange(m), assigned_idx]
agg['n_bands_satisfying_alpha'] = satisfies.sum(axis=1)

print('\nHard cluster assignment distribution:')
print(agg['hard_cluster'].value_counts())
print(f'\nCustomers satisfying alpha-cut in >1 band (overlapping structure): '
      f'{(agg["n_bands_satisfying_alpha"]>1).sum()} ({(agg["n_bands_satisfying_alpha"]>1).mean():.2%})')
print(f'Customers satisfying NO band at alpha=0.5 (fallback to max-membership): '
      f'{(~any_satisfies).sum()} ({(~any_satisfies).mean():.2%})')

# save full soft membership matrix (for business interpretation) + hard assignment
soft_membership_df = pd.DataFrame(band_membership, columns=band_labels)
soft_membership_df.insert(0, 'customer_unique_id', agg['customer_unique_id'].values)
soft_membership_df.to_csv(OUT + 'soft_membership_top_level.csv', index=False)

agg.to_csv(OUT + 'olist_rfms_with_hard_clusters.csv', index=False)
print(f'\nSaved: olist_rfms_with_hard_clusters.csv, soft_membership_top_level.csv')

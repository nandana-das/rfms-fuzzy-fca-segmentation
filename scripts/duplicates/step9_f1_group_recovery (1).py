"""
STEP 9: f_score=1 structure-recovery test (Outstanding item #2, both audits)
Question: within Olist's dominant f_score=1 group (n=90,553, 97.00% of customers -- all
crisp-tied on frequency), does the fuzzy/FCA machinery recover any usable differentiating
structure from the other three dimensions (R, M, S) and from F*'s continuous residual signal,
or do these customers collapse into one indistinguishable mass regardless of method?

Tests:
  1. Hard-cluster fan-out: crisp RFM alone assigns "frequency band 1" to all of them. Does the
     alpha-cut hard-cluster assignment (which also uses R/M/S) actually split them across
     multiple different top-level clusters, or do they all still land in F1?
  2. Concept-membership fan-out: how many of the 153 pruned fuzzy concepts does each f_score=1
     customer belong to, and how much does that vary within the group (vs how much it varies in
     the full population)? Low variance here would mean the lattice treats them as one blob;
     high variance means real differentiation survives.
  3. F* residual correlation: within the group, does F* (continuous, std=0.008 per Step 1
     diagnostic) correlate with concept-membership count or with R/M/S band diversity -- i.e.
     is the tiny F* spread actually load-bearing for anything downstream, or dead weight?
  4. Soft top-level membership diversity: distribution of distinct (band-subset) profiles among
     f_score=1 customers using the 12 top-level soft-membership columns.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
OUT = PROCESSED
RES = RESULTS

hc = pd.read_csv(OUT + 'olist_rfms_with_hard_clusters.csv')
soft = pd.read_csv(OUT + 'soft_membership_top_level.csv')
pruned = pd.read_pickle(RES + 'pruned_fuzzy_concepts.pkl')  # 153 concepts, has 'extent' (row-index sets)

f1 = hc[hc['f_score'] == 1].copy()
n_f1 = len(f1)
n_total = len(hc)
print(f'f_score=1 customers: {n_f1} ({100*n_f1/n_total:.2f}% of {n_total})')

# --- Test 1: hard-cluster fan-out ---
print('\n=== Test 1: hard-cluster assignment within f_score=1 group ===')
dist = f1['hard_cluster'].value_counts()
print(dist.to_string())
non_f1_cluster = (f1['hard_cluster'] != 'F1').sum()
print(f'\nf_score=1 customers assigned to a hard_cluster OTHER than F1: {non_f1_cluster} '
      f'({100*non_f1_cluster/n_f1:.2f}%)')
print('(Crisp RFM alone cannot do this -- it has already tied them all on frequency; any split '
      'here is driven by R/M/S, i.e. the fuzzy machinery is using information crisp scoring on F '
      'alone would discard.)')

# --- Test 2: concept-membership fan-out (row index -> which pruned concepts contain them) ---
print('\n=== Test 2: concept-membership count (out of 153 pruned concepts) ===')
row_idx = hc.index  # positional index used when the feature matrix was built
membership_count = np.zeros(n_total, dtype=int)
for ext in pruned['extent']:
    idx = np.fromiter(ext, dtype=np.int64, count=len(ext))
    membership_count[idx] += 1
hc['n_concepts'] = membership_count
f1 = hc[hc['f_score'] == 1].copy()  # refresh with new column

overall_std = hc['n_concepts'].std()
f1_std = f1['n_concepts'].std()
print(f'Concept-membership count -- overall population: mean={hc["n_concepts"].mean():.2f}, '
      f'std={overall_std:.2f}, range=[{hc["n_concepts"].min()},{hc["n_concepts"].max()}]')
print(f'Concept-membership count -- f_score=1 subgroup:  mean={f1["n_concepts"].mean():.2f}, '
      f'std={f1_std:.2f}, range=[{f1["n_concepts"].min()},{f1["n_concepts"].max()}]')
print(f'Distinct concept-membership-count values within f_score=1 group: {f1["n_concepts"].nunique()}')
print(f'Ratio of within-group std to overall std: {f1_std/overall_std:.3f} '
      f'(close to 1.0 means the group is about as internally differentiated as the whole population)')

# --- Test 3: F* residual correlation within the group ---
print('\n=== Test 3: does residual F* spread (within f_score=1) correlate with anything downstream? ===')
corr_fstar_nconcepts = f1['F_star'].corr(f1['n_concepts'])
corr_fstar_nbands = f1['F_star'].corr(f1['n_bands_satisfying_alpha'])
print(f'corr(F_star, n_concepts)              within f_score=1: {corr_fstar_nconcepts:.4f}')
print(f'corr(F_star, n_bands_satisfying_alpha) within f_score=1: {corr_fstar_nbands:.4f}')
print('(F_star has std=0.008 in this group per Step 1 -- a near-zero correlation here would mean '
      'that tiny residual spread is NOT actually driving downstream differentiation; any real '
      'signal is coming from R/M/S instead, not from F* itself.)')

# --- Test 4: soft top-level membership profile diversity ---
print('\n=== Test 4: soft top-level membership profile diversity within f_score=1 group ===')
band_cols = [c for c in soft.columns if c != 'customer_unique_id']
soft_f1 = soft.merge(hc[hc['f_score'] == 1][['customer_unique_id']], on='customer_unique_id')
# binarize at alpha=0.5 to get a "which bands satisfied" signature, same convention as Step 4
sig = (soft_f1[band_cols] >= 0.5).astype(int)
sig_tuples = [tuple(row) for row in sig.values]
n_unique_sigs = len(set(sig_tuples))
print(f'Distinct alpha-cut band-membership signatures among f_score=1 customers: {n_unique_sigs} '
      f'(out of {len(sig_tuples)} customers, {len(band_cols)} possible top-level bands)')
sig_counts = pd.Series(sig_tuples).value_counts()
print('Top 5 signatures by frequency:')
for s, cnt in sig_counts.head(5).items():
    bands_on = [band_cols[i] for i, v in enumerate(s) if v == 1]
    print(f'  {cnt:6d} customers ({100*cnt/n_f1:.1f}%): bands = {bands_on}')

print('\n=== Summary ===')
print(f'{non_f1_cluster}/{n_f1} ({100*non_f1_cluster/n_f1:.2f}%) f_score=1 customers get a non-F1 '
      f'hard cluster; {n_unique_sigs} distinct alpha-cut band signatures exist within the group; '
      f'within-group concept-membership std is {f1_std/overall_std:.1%} of the population-wide std; '
      f"F*'s own residual spread correlates at r={corr_fstar_nconcepts:.3f} with concept count -- "
      'read together, these establish whether the fuzzy representation is doing real differentiating '
      'work inside the group crisp scoring collapses, and whether that work is coming from F* itself '
      'or from the other dimensions.')

out_cols = ['customer_unique_id', 'F_star', 'R', 'M', 'S', 'hard_cluster', 'n_concepts', 'n_bands_satisfying_alpha']
f1[out_cols].to_csv(RES + 'f1_group_recovery_detail.csv', index=False)
print(f'\nSaved: {RES}f1_group_recovery_detail.csv')

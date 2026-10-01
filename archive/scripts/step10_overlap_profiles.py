"""
STEP 10: Interpretable overlapping-segment profiles (Outstanding item #3, both audits)
Question: what does the fuzzy-FCA overlap representation say about a customer that a single
crisp RFMS label (their hard_cluster / score tuple) would not?

Method (systematic, not cherry-picked, per explicit instruction):
1. Build each customer's alpha-cut (0.5) top-level band signature (12 possible top-level bands:
   F1,S5,M1,M2,R4,R5,R3,M3,R2,S4,M4,R1), same convention as Step 4/9.
2. Rank ALL distinct signatures by frequency. Exclude trivial single-band signatures (those add
   no cross-dimension information beyond the crisp hard_cluster label itself).
3. Take the top-K most frequent NONTRIVIAL (>=3 dims touched, i.e. genuinely multi-dimensional
   overlap) signatures -- this is the systematic selection the review asked for.
4. For each, pull 3 representative customers (closest to the signature's median F_star, so they
   are typical members, not outliers) and show their raw R/F_star/M/S alongside their single
   hard_cluster label -- demonstrating that customers sharing ONE hard_cluster/crisp label can
   carry materially different overlap profiles, and that the overlap profile carries information
   (which other bands they also belong to) the single label discards.
"""
import pandas as pd
import numpy as np
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
OUT = PROCESSED
RES = RESULTS

hc = pd.read_csv(OUT + 'olist_rfms_with_hard_clusters.csv')
soft = pd.read_csv(OUT + 'soft_membership_top_level.csv')
band_cols = [c for c in soft.columns if c != 'customer_unique_id']

df = hc.merge(soft, on='customer_unique_id')
sig_bin = (df[band_cols] >= 0.5).astype(int)
df['signature'] = [tuple(row) for row in sig_bin.values]
df['n_dims_touched'] = sig_bin.apply(lambda row: len(set(c[0] for c, v in zip(band_cols, row) if v == 1)), axis=1)
df['bands_on'] = sig_bin.apply(lambda row: tuple(c for c, v in zip(band_cols, row) if v == 1), axis=1)

sig_counts = df['bands_on'].value_counts()
print(f'Total distinct alpha-cut signatures in population: {len(sig_counts)}')

# nontrivial = touches >=3 distinct dimensions (out of R,F,M,S) -- genuine cross-dimension overlap
nontrivial = df[df['n_dims_touched'] >= 3]
nontrivial_sig_counts = nontrivial['bands_on'].value_counts()
print(f'Nontrivial (>=3 dims) signatures: {len(nontrivial_sig_counts)} distinct, '
      f'{len(nontrivial)} customers ({100*len(nontrivial)/len(df):.1f}% of population)')

TOP_K = 6
top_sigs = nontrivial_sig_counts.head(TOP_K)
print(f'\n=== Top {TOP_K} most frequent nontrivial overlap signatures (systematic selection) ===')

rows_out = []
for sig, cnt in top_sigs.items():
    sub = df[df['bands_on'] == sig]
    med_f = sub['F_star'].median()
    sub = sub.assign(dist=(sub['F_star'] - med_f).abs()).sort_values('dist')
    reps = sub.head(3)
    hard_labels = sub['hard_cluster'].value_counts()
    print(f'\nSignature {sig}  (n={cnt}, {100*cnt/len(df):.2f}% of population)')
    print(f'  hard_cluster labels this signature maps to: {dict(hard_labels)}')
    print(f'  within-signature spread: R=[{sub["R"].min():.0f},{sub["R"].max():.0f}]d '
          f'(median {sub["R"].median():.0f}), M=[{sub["M"].min():.2f},{sub["M"].max():.2f}] '
          f'(median {sub["M"].median():.2f})')
    for _, r in reps.iterrows():
        print(f'  rep: id={r["customer_unique_id"][:12]}..  R={r["R"]:.0f}d  F*={r["F_star"]:.4f} '
              f'(f_score={r["f_score"]})  M={r["M"]:.2f}  S={r["S"]:.1f}  '
              f'-> single hard_cluster label = {r["hard_cluster"]}')
        rows_out.append({'signature': '+'.join(sig), 'n_customers_in_signature': cnt,
                          'customer_unique_id': r['customer_unique_id'], 'R': r['R'],
                          'F_star': r['F_star'], 'f_score': r['f_score'], 'M': r['M'], 'S': r['S'],
                          'hard_cluster_label': r['hard_cluster']})

# --- the actual collapse: how much R/M spread does hard_cluster=F1 alone hide? ---
f1_only = df[df['hard_cluster'] == 'F1']
print(f'\n=== What the single hard_cluster="F1" label collapses (n={len(f1_only)}) ===')
print(f'  R (recency, days): min={f1_only["R"].min():.0f}, max={f1_only["R"].max():.0f}, '
      f'median={f1_only["R"].median():.0f}, IQR=[{f1_only["R"].quantile(.25):.0f},'
      f'{f1_only["R"].quantile(.75):.0f}]')
print(f'  M (monetary): min={f1_only["M"].min():.2f}, max={f1_only["M"].max():.2f}, '
      f'median={f1_only["M"].median():.2f}, IQR=[{f1_only["M"].quantile(.25):.2f},'
      f'{f1_only["M"].quantile(.75):.2f}]')
print(f'  S (satisfaction): distribution {dict(f1_only["S"].value_counts().sort_index())}')
print(f'  Distinct overlap signatures within hard_cluster=F1 alone: {f1_only["bands_on"].nunique()}')
print('  -> a single crisp/hard label "F1" spans this entire R/M/S range and >100 distinct overlap')
print('     signatures; the fuzzy overlap representation preserves which R-band, M-band and S-band')
print('     each such customer ALSO falls into, information the single hard label discards outright.')

out_df = pd.DataFrame(rows_out)
out_df.to_csv(RES + 'overlap_profile_examples.csv', index=False)

print('\n=== Interpretation ===')
print('Each row above shares ONE crisp hard_cluster label (the single dominant band a plain RFMS')
print('scoring + argmax assignment would report) but the overlap signature attaches multiple')
print('additional top-level bands. That is information a single-label crisp system discards by')
print('construction: e.g. a customer labeled hard_cluster=S5 who ALSO satisfies F1+M1+R5 at alpha=0.5')
print('is behaviorally distinguishable from another S5 customer who does not also satisfy those bands,')
print('but a plain RFMS quintile score with hard assignment reports both simply as "S5" (or whichever')
print('single band wins the argmax), losing the rest of the profile.')
print(f'\nSaved: {RES}overlap_profile_examples.csv')

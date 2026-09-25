"""
STEP 13: Publication figures.
1. Lattice size comparison (raw vs pruned, Olist vs Retail II)
2. Benchmark comparison (Silhouette / DB, FCA vs K-means vs Hierarchical)
3. Hasse diagram (Olist pruned lattice, top-level + immediate structure)
4. Kneedle knee-point plots (support and stability curves, Olist + Retail II)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from project_paths import PROCESSED, RESULTS, FIGURES, ensure_output_dirs

ensure_output_dirs()
RES = RESULTS
OUT = PROCESSED
FIG = FIGURES

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ============================================================
# Figure 1: Lattice size comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
datasets = ['Olist\n(RFMS, 20 attrs)', 'Retail II\n(RFM, 15 attrs)']
raw = [1369, 1064]
pruned = [153, 115]
x = np.arange(len(datasets))
w = 0.35
ax.bar(x - w/2, raw, w, label='Raw closed concepts (uncapped)', color='#4C72B0')
ax.bar(x + w/2, pruned, w, label='Pruned concepts (Kneedle)', color='#DD8452')
for i, (r, p) in enumerate(zip(raw, pruned)):
    ax.text(i - w/2, r + 20, str(r), ha='center', fontsize=9)
    ax.text(i + w/2, p + 20, str(p), ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel('Number of concepts')
ax.set_title('Concept lattice size: raw vs. Kneedle-pruned\n(uncapped FP-growth)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG + 'fig1_lattice_size.png')
plt.close()
print('Saved fig1_lattice_size.png')

# ============================================================
# Figure 2: Benchmark comparison (Silhouette + DB, Olist)
# ============================================================
bench = pd.read_csv(RES + 'benchmark_comparison.csv')
methods_k5 = bench[bench['k'] == 5]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
labels = methods_k5['method'].tolist()
sil = methods_k5['silhouette'].tolist()
db = methods_k5['davies_bouldin'].tolist()
colors = ['#4C72B0' if 'Fuzzy' in l else ('#DD8452' if 'K-means' in l and 'outlier' not in l
          else ('#55A868' if 'Ward' in l and 'outlier' not in l else '#999999')) for l in labels]
short_labels = [l.replace(' (k-matched)', '\n(k-matched)').replace(' (outlier-removed)', '\n(outlier-rm)')
                for l in labels]
axes[0].barh(short_labels, sil, color=colors)
axes[0].set_xlabel('Silhouette (higher better)')
axes[0].set_title('Silhouette @ k=5 (Olist)')
axes[0].axvline(0, color='black', linewidth=0.5)
axes[1].barh(short_labels, db, color=colors)
axes[1].set_xlabel('Davies-Bouldin (lower better)')
axes[1].set_title('Davies-Bouldin @ k=5 (Olist)')
plt.tight_layout()
plt.savefig(FIG + 'fig2_benchmark_k5.png')
plt.close()
print('Saved fig2_benchmark_k5.png')

# ============================================================
# Figure 3: Hasse diagram (Olist pruned lattice)
# ============================================================
with open(RES + 'hasse_edges.pkl', 'rb') as f:
    hasse = pickle.load(f)
edges = hasse['edges']
concepts = hasse['concepts'].reset_index(drop=True)

# label each node by its itemset (shortened) and support
def short_label(itemset):
    attrs = sorted(set(a.split('@')[0] for a in itemset))
    return '\n'.join(attrs) if len(attrs) <= 3 else f'{len(attrs)} attrs'

G = nx.DiGraph()
for i in range(len(concepts)):
    G.add_node(i, support=concepts.loc[i, 'support'])
for (child, parent) in edges:
    G.add_edge(child, parent)

# For legibility, restrict to the 40 highest-support concepts + their edges among that set
top_idx = concepts.sort_values('support', ascending=False).head(40).index.tolist()
top_set = set(top_idx)
G_sub = G.subgraph(top_idx).copy()

fig, ax = plt.subplots(figsize=(14, 10))
try:
    pos = nx.nx_agraph.graphviz_layout(G_sub, prog='dot')
except Exception:
    pos = nx.spring_layout(G_sub, k=1.5, seed=42, iterations=100)

node_sizes = [200 + 4000 * concepts.loc[i, 'support'] for i in G_sub.nodes()]
node_colors = [concepts.loc[i, 'support'] for i in G_sub.nodes()]
nodes = nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors,
                                cmap='viridis', ax=ax, alpha=0.85)
nx.draw_networkx_edges(G_sub, pos, arrows=True, arrowsize=8, width=0.6, alpha=0.4, ax=ax)
labels = {i: short_label(concepts.loc[i, 'itemsets']) for i in G_sub.nodes()}
nx.draw_networkx_labels(G_sub, pos, labels, font_size=6, ax=ax)
plt.colorbar(nodes, ax=ax, label='Support', shrink=0.7)
ax.set_title(f'Hasse diagram: top-40 (by support) of the 153 pruned Olist concepts\n'
             f'({G_sub.number_of_edges()} covering edges among this subset)')
ax.axis('off')
plt.tight_layout()
plt.savefig(FIG + 'fig3_hasse_diagram_top40.png')
plt.close()
print('Saved fig3_hasse_diagram_top40.png')

# ============================================================
# Figure 4: Kneedle knee-point curves (Olist + Retail II)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

datasets_info = [
    ('Olist', 'fuzzy_concepts_raw.pkl', 0.1025, 0.9876),
    ('Retail II', 'retail2_fuzzy_concepts_raw.pkl', 0.1170, 0.9315),
]
for row, (name, fname, supp_star, theta_star) in enumerate(datasets_info):
    closed = pd.read_pickle(RES + fname)
    if 'stability_approx' not in closed.columns:
        # olist raw pkl doesn't have stability precomputed; recompute quickly for the plot
        agg = pd.read_csv(OUT + ('olist_rfms_features.csv' if name == 'Olist'
                                 else 'retail2_rfm_features.csv'))
        score_cols = ['r_score', 'f_score', 'm_score', 's_score'] if name == 'Olist' else ['r_score', 'f_score', 'm_score']
        score_tuples = agg[score_cols].values
        fuzzy_df = pd.read_pickle(RES + ('fuzzy_membership_matrix.pkl' if name == 'Olist'
                                          else 'retail2_fuzzy_membership_matrix.pkl'))
        attr_cols = list(fuzzy_df.columns)
        mu_matrix = fuzzy_df[attr_cols].values
        L = [0.3, 0.5, 0.7]
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
        def stability_proxy(extent):
            n = len(extent)
            if n <= 1:
                return 1.0
            idx = np.fromiter(extent, dtype=np.int64, count=n)
            profiles = score_tuples[idx]
            return 1 - (len(np.unique(profiles, axis=0)) / n)
        closed['extent'] = closed['itemsets'].apply(extent_of)
        closed['stability_approx'] = closed['extent'].apply(stability_proxy)

    supp_sorted = np.sort(closed['support'].values)[::-1]
    stab_sorted = np.sort(closed['stability_approx'].values)[::-1]

    ax = axes[row, 0]
    ax.plot(range(len(supp_sorted)), supp_sorted, color='#4C72B0', linewidth=1.2)
    knee_idx_s = np.argmin(np.abs(supp_sorted - supp_star))
    ax.axvline(knee_idx_s, color='red', linestyle='--', linewidth=1, label=f'knee: rank {knee_idx_s}')
    ax.axhline(supp_star, color='red', linestyle=':', linewidth=0.8)
    ax.set_title(f'{name}: support curve (Kneedle knee)')
    ax.set_xlabel('Concept rank (sorted by support desc)')
    ax.set_ylabel('Support')
    ax.legend(fontsize=8)

    ax = axes[row, 1]
    ax.plot(range(len(stab_sorted)), stab_sorted, color='#DD8452', linewidth=1.2)
    knee_idx_t = np.argmin(np.abs(stab_sorted - theta_star))
    ax.axvline(knee_idx_t, color='red', linestyle='--', linewidth=1, label=f'knee: rank {knee_idx_t}')
    ax.axhline(theta_star, color='red', linestyle=':', linewidth=0.8)
    ax.set_title(f'{name}: stability curve (Kneedle knee)')
    ax.set_xlabel('Concept rank (sorted by stability desc)')
    ax.set_ylabel('Stability proxy')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG + 'fig4_kneedle_curves.png')
plt.close()
print('Saved fig4_kneedle_curves.png')

print('\nAll figures saved to', FIG)

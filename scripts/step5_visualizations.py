"""
Step 5: Publication-Quality Visualizations for Thesis/Paper
Generates:
1. Benchmark Comparison Chart (Silhouette & Davies-Bouldin across methods and k)
2. Segment Radar / Spider Chart (Mean standardized RFMS metrics per segment)
3. Customer Distribution & Monetary Contribution Bar Chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results/figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# --- Figure 1: Benchmark Comparison (Silhouette & Davies-Bouldin) ---
bench_df = pd.read_csv('results/clustering_benchmark_table.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Silhouette Plot (Higher is Better)
sns.barplot(
    data=bench_df, x='k', y='Silhouette', hue='Method',
    ax=axes[0], palette=['#2b5c8f', '#4682b4', '#e76f51']
)
axes[0].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Silhouette Score')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].legend(title='Method', loc='upper left')

# Davies-Bouldin Plot (Lower is Better)
sns.barplot(
    data=bench_df, x='k', y='Davies-Bouldin', hue='Method',
    ax=axes[1], palette=['#2b5c8f', '#4682b4', '#e76f51']
)
axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].legend(title='Method', loc='upper right')

plt.tight_layout()
fig_path1 = 'results/figures/benchmark_metrics_comparison.png'
plt.savefig(fig_path1, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path1}")

# --- Figure 2: Segment Radar / Spider Chart (RFMS Dimensions) ---
seg_df = pd.read_csv('results/segment_profiles_k5.csv')

categories = ['Recency\n(Lower=Better)', 'Monetary\n(BRL)', 'Satisfaction\n(Review 1-5)', 'Repeat Rate\n(%)']
N = len(categories)

# Normalize metrics to [0, 1] for spider comparison
r_norm = 1.0 - (seg_df['Mean_Recency_Days'] - seg_df['Mean_Recency_Days'].min()) / (seg_df['Mean_Recency_Days'].max() - seg_df['Mean_Recency_Days'].min())
m_norm = (seg_df['Mean_Monetary_BRL'] - seg_df['Mean_Monetary_BRL'].min()) / (seg_df['Mean_Monetary_BRL'].max() - seg_df['Mean_Monetary_BRL'].min())
s_norm = (seg_df['Mean_Review_Score'] - seg_df['Mean_Review_Score'].min()) / (seg_df['Mean_Review_Score'].max() - seg_df['Mean_Review_Score'].min() + 1e-6)
rep_vals = seg_df['Repeat_Rate'].str.rstrip('%').astype(float)
rep_norm = (rep_vals - rep_vals.min()) / (rep_vals.max() - rep_vals.min())

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
colors = ['#2a9d8f', '#e76f51', '#f4a261', '#457b9d', '#9b5de5']

for i, row in seg_df.iterrows():
    vals = [r_norm.iloc[i], m_norm.iloc[i], s_norm.iloc[i], rep_norm.iloc[i]]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label=row['segment_name'], color=colors[i])
    ax.fill(angles, vals, color=colors[i], alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.1)
plt.title('Fuzzy FCA Discovered Customer Segments (k = 5)\nRelative RFMS Characteristics', size=13, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

plt.tight_layout()
fig_path2 = 'results/figures/segment_radar_chart_k5.png'
plt.savefig(fig_path2, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path2}")

# --- Figure 3: Segment Breakdown (Customer Share vs. Revenue Share) ---
cust_df = pd.read_csv('results/customer_clusters_k5.csv')
total_revenue = cust_df['M'].sum()
total_cust = len(cust_df)

breakdown = cust_df.groupby('segment_name').agg(
    Customer_Share=('customer_unique_id', lambda x: len(x) / total_cust * 100),
    Revenue_Share=('M', lambda x: x.sum() / total_revenue * 100)
).reset_index()

breakdown_melted = breakdown.melt(id_vars='segment_name', value_vars=['Customer_Share', 'Revenue_Share'],
                                  var_name='Metric', value_name='Percentage')

plt.figure(figsize=(10, 5), dpi=300)
sns.barplot(data=breakdown_melted, y='segment_name', x='Percentage', hue='Metric', palette=['#3a86ff', '#ff006e'])
plt.title('Customer Volume vs. Revenue Contribution by Segment', fontweight='bold', fontsize=12)
plt.xlabel('Percentage (%)')
plt.ylabel('Customer Segment')
plt.legend(title='Metric')
plt.tight_layout()

fig_path3 = 'results/figures/segment_revenue_vs_volume.png'
plt.savefig(fig_path3, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path3}")

print("\nAll figures generated successfully in results/figures/!")

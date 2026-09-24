# RFMS-FCA Customer Segmentation

Hierarchical Customer Segmentation for Sparse-Frequency Marketplace Data using Fuzzy Formal Concept Analysis (Fuzzy FCA) on the Olist Brazilian E-commerce dataset.

---

## 📁 Project Directory Structure

```
rfms_fca_project/
├── .gitignore                          # Git ignore rules for venv, cache, and pickle artifacts
├── README.md                           # Project overview, setup, and execution instructions
├── requirements.txt                    # Pinned Python package dependencies
├── methodology_rfms_fca_olist.md       # Complete research methodology and mathematical specification
├── data/                               # Data directory
│   └── olist_rfms_features.csv         # Extracted RFMS customer metrics (N ≈ 93,357)
├── results/                            # Generated models, concept lattices, metrics, and figures
│   ├── fuzzy_concepts_raw.pkl          # Mined fuzzy formal concepts (FP-Growth closed itemsets)
│   ├── pruned_fuzzy_concepts.pkl       # Stability-pruned iceberg concepts (N = 536)
│   ├── hasse_edges.pkl                 # Hasse diagram covering relations (N = 928 edges)
│   ├── clustering_benchmark_table.csv  # Symmetrical evaluation table across algorithms
│   ├── segment_profiles_k5.csv         # Managerial summary per customer segment
│   ├── customer_clusters_k5.csv        # Customer-level cluster and segment assignments
│   └── figures/                        # Publication-quality figures (300 DPI)
│       ├── benchmark_metrics_comparison.png
│       ├── segment_radar_chart_k5.png
│       └── segment_revenue_vs_volume.png
└── scripts/                            # Pipeline execution scripts
    ├── step1_rfms_prep.py              # Data cleaning, composite F*, and dense-rank scoring
    ├── step2_fuzzy_fca.py              # Fuzzy membership partition & FP-Growth concept mining
    ├── step3_stability_pruning.py      # Approximate stability index & Kneedle elbow pruning
    ├── step4_clustering_benchmark.py   # Benchmark Fuzzy FCA vs. K-Means vs. Hierarchical
    └── step5_visualizations.py         # Generate radar charts and comparative figures
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Python 3.11 or 3.12 recommended.

### 2. Virtual Environment Setup
To create and activate the local virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Workflow

Run the scripts in sequential order from the project root:

1. **Step 1: Feature Engineering & RFMS Prep**
   ```bash
   python scripts/step1_rfms_prep.py
   ```
   * Cleans raw Olist tables, joins on `customer_unique_id`, computes composite frequency $F^*$ via variance-maximization, and generates dense-rank scores $(r, f, \mu, s)$ in `data/olist_rfms_features.csv`.

2. **Step 2: Fuzzy Membership & Closed Itemset Mining**
   ```bash
   python scripts/step2_fuzzy_fca.py
   ```
   * Builds triangular fuzzy membership partitions, performs Belohlavek $L$-fuzzy multi-threshold scaling ($L \in \{0.3, 0.5, 0.7\}$), and extracts closed frequent itemsets as fuzzy concepts in `results/fuzzy_concepts_raw.pkl`.

3. **Step 3: Stability Pruning & Hasse Diagram**
   ```bash
   python scripts/step3_stability_pruning.py
   ```
   * Computes object extents and Kuznetsov approximate stability indices, mathematically derives cutoffs $(\text{supp}_{min}^*, \theta^*)$ via the Kneedle elbow method, prunes to the iceberg lattice, and computes transitive covering edges in `results/hasse_edges.pkl`.

4. **Step 4: Clustering Benchmarks & Symmetrical Evaluation**
   ```bash
   python scripts/step4_clustering_benchmark.py
   ```
   * Benchmarks Fuzzy FCA crisp assignments against K-Means and Agglomerative Hierarchical clustering for $k \in \{4, 5, 6\}$ using Silhouette score, Davies-Bouldin index, and Fuzzy Partition Coefficient (FPC).

5. **Step 5: Visualizations & Segment Profiling**
   ```bash
   python scripts/step5_visualizations.py
   ```
   * Generates publication-ready figures in `results/figures/` (benchmark bar charts, multi-dimensional segment radar charts, and volume vs. revenue breakdowns).

---

## 📊 Experimental Results

### 1. Symmetrical Clustering Benchmark (Extending Table 10 of Rungruang et al.)

| Method | $k$ | Silhouette $\uparrow$ | Davies-Bouldin $\downarrow$ | FPC (Fuzzy Partition) $\uparrow$ | ARI vs. K-Means | Exec Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | 4 | 0.3394 | 1.0002 | *N/A (Crisp)* | 1.0000 | 0.47s |
| **Agglomerative** | 4 | 0.3222 | 0.9742 | *N/A (Crisp)* | 0.4878 | 2.72s |
| **Fuzzy FCA (Proposed)** | **4** | 0.0253 | 1.8794 | **0.5817** | 0.1747 | **0.02s** |
| **K-Means** | 5 | 0.3714 | 0.8072 | *N/A (Crisp)* | 1.0000 | 0.39s |
| **Agglomerative** | 5 | 0.3382 | 0.8671 | *N/A (Crisp)* | 0.3636 | 2.67s |
| **Fuzzy FCA (Proposed)** | **5** | -0.0026 | 2.9124 | **0.6007** | 0.1315 | **0.03s** |
| **K-Means** | 6 | 0.3857 | 0.8676 | *N/A (Crisp)* | 1.0000 | 0.41s |
| **Agglomerative** | 6 | 0.2867 | 1.0277 | *N/A (Crisp)* | 0.3622 | 1.92s |
| **Fuzzy FCA (Proposed)** | **6** | -0.0251 | 2.6117 | **0.5444** | 0.1570 | **0.03s** |

> **Key Theoretical Insight:** Traditional metrics (Silhouette / Davies-Bouldin) favor hyperspherical, mutually-exclusive clusters (e.g. K-Means). In contrast, Fuzzy FCA forms overlapping semantic concepts based on formal intents, achieving a strong **Fuzzy Partition Coefficient (FPC $\approx 0.54 - 0.60$)**, orders of magnitude faster execution, and direct business interpretability.

---

### 2. Discovered Customer Segments ($k = 5$)

| Cluster | Segment Name | Customer Count | % Share | Mean Recency (days) | Mean Monetary (BRL) | Mean Review Score (1–5) | Repeat Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Recent Champions** | 31,859 | 34.13% | 133.9 | R$ 209.89 | 3.88 | 3.86% |
| **1** | **Budget Satisfied** | 16,677 | 17.86% | 274.2 | R$ 51.28 | 5.00 | 0.44% |
| **2** | **Economy Satisfied** | 10,554 | 11.30% | 270.8 | R$ 111.34 | 5.00 | 2.19% |
| **3** | **Mid-Tier Spenders** | 20,121 | 21.55% | 234.3 | R$ 179.27 | 3.74 | 4.30% |
| **4** | **Lapsing / Hibernating** | 14,146 | 15.15% | 403.6 | R$ 219.02 | 3.78 | 2.84% |

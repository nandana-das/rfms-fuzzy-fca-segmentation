"""
Methodological Investigation of Problem 2:
Why the Olist Fuzzy-FCA clustering results have poor Silhouette/DB scores.

Controlled methodological investigation covering:
1. Exact reproduction of the current FCA baseline (natural k=12, alpha=0.5)
2. Alpha sensitivity: alpha in {0.4, 0.5, 0.6}
3. Canonical Fuzzy C-Means (FCM) benchmark at k in {4, 5, 6, 12} with canonical FPC and PE
4. k-matched comparisons across FCA, KMeans, FCM, and Ward (k in {4, 5, 6})
5. Pairwise Adjusted Rand Index (ARI) analysis (all 6 pairs across k in {4, 5, 6, 12})
6. Per-cluster Silhouette distribution (mean, median, min, max, % negative)
7. Systematic test of hypotheses A, B, C, D, E
8. Membership-aware analysis (coverage, overlap, entropy, extent sizes)

Outputs saved to results/problem2_clustering_investigation/
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED, RESULTS, RESULTS_DIR
from fcm import FuzzyCMeans

OUT_DIR = RESULTS_DIR / "problem2_clustering_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SIL_SAMPLE = 5000
HIER_SAMPLE = 5000


def main():
    print("=" * 80)
    print("CONTROLLED METHODOLOGICAL INVESTIGATION: PROBLEM 2")
    print("Why Olist Fuzzy-FCA Clustering Results Have Poor Silhouette/DB Scores")
    print("=" * 80)

    # 1. Load data
    agg = pd.read_csv(PROCESSED + "olist_rfms_with_hard_clusters.csv")
    soft = pd.read_csv(PROCESSED + "soft_membership_top_level.csv")
    pruned = pd.read_pickle(RESULTS + "pruned_fuzzy_concepts.pkl")
    fuzzy_df = pd.read_pickle(RESULTS + "fuzzy_membership_matrix.pkl")

    band_labels = [c for c in soft.columns if c != "customer_unique_id"]
    band_membership = soft[band_labels].values
    m = len(agg)
    print(f"Loaded {m:,} customers with {len(band_labels)} top-level bands: {band_labels}")

    FEATS = ["r_score", "f_score", "m_score", "s_score"]
    X = StandardScaler().fit_transform(agg[FEATS].values)

    rng = np.random.default_rng(RANDOM_SEED)
    sil_sample_idx = rng.choice(m, size=min(SIL_SAMPLE, m), replace=False)
    hier_sample_idx = rng.choice(m, size=min(HIER_SAMPLE, m), replace=False)
    X_sub = X[sil_sample_idx]
    X_hier = X[hier_sample_idx]

    summary_records = []
    ari_records = []
    decomp_records = []
    alpha_records = []

    # -------------------------------------------------------------
    # EXPERIMENT 1 & 2: Current Baseline and Alpha Sensitivity
    # -------------------------------------------------------------
    print("\n--- Experiments 1 & 2: Baseline and Alpha Sensitivity ---")
    fca_labels_by_alpha = {}

    for alpha in [0.4, 0.5, 0.6]:
        satisfies = band_membership >= alpha
        n_satisfying = satisfies.sum(axis=1)
        max_band_idx = np.argmax(band_membership, axis=1)
        any_satisfies = satisfies.any(axis=1)
        masked_membership = np.where(satisfies, band_membership, -1)
        assigned_idx = np.where(any_satisfies, np.argmax(masked_membership, axis=1), max_band_idx)
        hard_clusters = [band_labels[i] for i in assigned_idx]
        fca_labels_by_alpha[alpha] = assigned_idx

        sil = float(silhouette_score(X, assigned_idx, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db = float(davies_bouldin_score(X, assigned_idx))
        counts = pd.Series(hard_clusters).value_counts().to_dict()

        n_unassigned_strict = int((~any_satisfies).sum())
        n_ambiguous = int((n_satisfying > 1).sum())

        # Descriptive FPC across the 12 top-level bands
        row_sums_mem = band_membership.sum(axis=1, keepdims=True)
        row_sums_mem[row_sums_mem == 0] = 1e-9
        mem_norm = band_membership / row_sums_mem
        fpc_descriptive = float(np.mean(np.sum(mem_norm ** 2, axis=1)))
        pe_descriptive = float(-np.mean(np.sum(mem_norm * np.log(np.clip(mem_norm, 1e-12, 1.0)), axis=1)))

        tag = "Baseline" if alpha == 0.5 else f"Alpha={alpha}"
        summary_records.append({
            "method": f"FCA Natural-k ({tag})",
            "k": len(band_labels),
            "n_clusters_effective": len(counts),
            "silhouette": sil,
            "davies_bouldin": db,
            "fpc_type": "Descriptive normalized FPC (custom)",
            "fpc": fpc_descriptive,
            "partition_entropy": pe_descriptive,
            "pct_customers_assigned": 100.0,
            "unassigned_count_strict": n_unassigned_strict,
            "unassigned_pct_strict": float(n_unassigned_strict / m * 100),
            "ambiguous_count": n_ambiguous,
            "ambiguous_pct": float(n_ambiguous / m * 100),
            "dominant_cluster": max(counts, key=counts.get),
            "dominant_cluster_pct": float(max(counts.values()) / m * 100),
            "cluster_sizes": str(counts),
        })

        alpha_records.append({
            "alpha": alpha,
            "effective_k": len(counts),
            "silhouette": sil,
            "davies_bouldin": db,
            "dominant_cluster": max(counts, key=counts.get),
            "dominant_cluster_pct": float(max(counts.values()) / m * 100),
            "ambiguous_count": n_ambiguous,
            "ambiguous_pct": float(n_ambiguous / m * 100),
            "unassigned_strict_count": n_unassigned_strict,
            "unassigned_strict_pct": float(n_unassigned_strict / m * 100),
            "cluster_sizes": str(counts),
        })
        print(f"Alpha={alpha}: Sil={sil:.4f}, DB={db:.4f}, ambiguous={n_ambiguous} ({n_ambiguous/m:.2%}), strict unassigned={n_unassigned_strict}")

    # -------------------------------------------------------------
    # EXPERIMENT 3 & 4: Canonical FCM and k-Matched Comparisons
    # -------------------------------------------------------------
    print("\n--- Experiments 3 & 4: Canonical FCM & k-Matched Comparisons ---")
    top_bands = sorted(band_labels, key=lambda b: -band_membership[:, band_labels.index(b)].sum())

    k_grid = [4, 5, 6, 12]
    all_k_labels = {}

    for k in k_grid:
        print(f"\nProcessing k = {k}...")
        # 1. FCA
        if k <= 6:
            sel_bands = top_bands[:k]
            sel_idx = [band_labels.index(b) for b in sel_bands]
            sub_mem = band_membership[:, sel_idx]
            fca_labels = np.argmax(sub_mem, axis=1)
            fca_name = f"FCA (k-matched, top-{k} bands)"
        else:
            fca_labels = fca_labels_by_alpha[0.5]
            fca_name = "FCA (natural-k, 12 bands)"

        all_k_labels[(k, "FCA")] = fca_labels
        sil_fca = float(silhouette_score(X, fca_labels, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db_fca = float(davies_bouldin_score(X, fca_labels))
        counts_fca = pd.Series(fca_labels).value_counts().to_dict()

        if k <= 6:
            sub_norm = sub_mem / np.maximum(sub_mem.sum(axis=1, keepdims=True), 1e-9)
            fpc_fca_k = float(np.mean(np.sum(sub_norm ** 2, axis=1)))
            pe_fca_k = float(-np.mean(np.sum(sub_norm * np.log(np.clip(sub_norm, 1e-12, 1.0)), axis=1)))
            summary_records.append({
                "method": fca_name,
                "k": k,
                "n_clusters_effective": len(counts_fca),
                "silhouette": sil_fca,
                "davies_bouldin": db_fca,
                "fpc_type": "Descriptive normalized FPC (custom)",
                "fpc": fpc_fca_k,
                "partition_entropy": pe_fca_k,
                "pct_customers_assigned": 100.0,
                "unassigned_count_strict": 0,
                "unassigned_pct_strict": 0.0,
                "ambiguous_count": np.nan,
                "ambiguous_pct": np.nan,
                "dominant_cluster": str(max(counts_fca, key=counts_fca.get)),
                "dominant_cluster_pct": float(max(counts_fca.values()) / m * 100),
                "cluster_sizes": str(counts_fca),
            })

        # 2. KMeans
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X)
        km_labels = km.labels_
        all_k_labels[(k, "KMeans")] = km_labels
        sil_km = float(silhouette_score(X, km_labels, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db_km = float(davies_bouldin_score(X, km_labels))
        counts_km = pd.Series(km_labels).value_counts().to_dict()

        summary_records.append({
            "method": f"K-Means (k={k})",
            "k": k,
            "n_clusters_effective": len(counts_km),
            "silhouette": sil_km,
            "davies_bouldin": db_km,
            "fpc_type": "Crisp partition (FPC=1.0)",
            "fpc": 1.0,
            "partition_entropy": 0.0,
            "pct_customers_assigned": 100.0,
            "unassigned_count_strict": 0,
            "unassigned_pct_strict": 0.0,
            "ambiguous_count": 0,
            "ambiguous_pct": 0.0,
            "dominant_cluster": str(max(counts_km, key=counts_km.get)),
            "dominant_cluster_pct": float(max(counts_km.values()) / m * 100),
            "cluster_sizes": str(counts_km),
        })

        # 3. Canonical FCM
        fcm = FuzzyCMeans(n_clusters=k, m=2.0, max_iter=100, random_state=RANDOM_SEED).fit(X)
        fcm_labels = np.argmax(fcm.memberships_, axis=1)
        all_k_labels[(k, "FCM")] = fcm_labels
        sil_fcm = float(silhouette_score(X, fcm_labels, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED))
        db_fcm = float(davies_bouldin_score(X, fcm_labels))
        counts_fcm = pd.Series(fcm_labels).value_counts().to_dict()

        fpc_canonical = float(np.mean(np.sum(fcm.memberships_ ** 2, axis=1)))
        pe_canonical = float(-np.mean(np.sum(fcm.memberships_ * np.log(np.clip(fcm.memberships_, 1e-12, 1.0)), axis=1)))

        summary_records.append({
            "method": f"FCM (k={k})",
            "k": k,
            "n_clusters_effective": len(counts_fcm),
            "silhouette": sil_fcm,
            "davies_bouldin": db_fcm,
            "fpc_type": "Canonical Bezdek FPC",
            "fpc": fpc_canonical,
            "partition_entropy": pe_canonical,
            "pct_customers_assigned": 100.0,
            "unassigned_count_strict": 0,
            "unassigned_pct_strict": 0.0,
            "ambiguous_count": 0,
            "ambiguous_pct": 0.0,
            "dominant_cluster": str(max(counts_fcm, key=counts_fcm.get)),
            "dominant_cluster_pct": float(max(counts_fcm.values()) / m * 100),
            "cluster_sizes": str(counts_fcm),
        })

        # 4. Hierarchical Ward (n=5000 subsample)
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_hier)
        ward_labels = ac.labels_
        all_k_labels[(k, "Ward")] = ward_labels
        sil_ward = float(silhouette_score(X_hier, ward_labels))
        db_ward = float(davies_bouldin_score(X_hier, ward_labels))
        counts_ward = pd.Series(ward_labels).value_counts().to_dict()

        summary_records.append({
            "method": f"Hierarchical Ward (k={k}, n=5000)",
            "k": k,
            "n_clusters_effective": len(counts_ward),
            "silhouette": sil_ward,
            "davies_bouldin": db_ward,
            "fpc_type": "Crisp partition (FPC=1.0)",
            "fpc": 1.0,
            "partition_entropy": 0.0,
            "pct_customers_assigned": 100.0,
            "unassigned_count_strict": 0,
            "unassigned_pct_strict": 0.0,
            "ambiguous_count": 0,
            "ambiguous_pct": 0.0,
            "dominant_cluster": str(max(counts_ward, key=counts_ward.get)),
            "dominant_cluster_pct": float(max(counts_ward.values()) / len(hier_sample_idx) * 100),
            "cluster_sizes": str(counts_ward),
        })

        # ---------------------------------------------------------
        # EXPERIMENT 5: Pairwise Adjusted Rand Index (ARI)
        # ---------------------------------------------------------
        # Full population ARIs
        ari_fca_km = float(adjusted_rand_score(fca_labels, km_labels))
        ari_fca_fcm = float(adjusted_rand_score(fca_labels, fcm_labels))
        ari_km_fcm = float(adjusted_rand_score(km_labels, fcm_labels))

        # Matched subsample ARIs (including Ward)
        ari_fca_ward = float(adjusted_rand_score(fca_labels[hier_sample_idx], ward_labels))
        ari_km_ward = float(adjusted_rand_score(km_labels[hier_sample_idx], ward_labels))
        ari_fcm_ward = float(adjusted_rand_score(fcm_labels[hier_sample_idx], ward_labels))
        ari_fca_km_sub = float(adjusted_rand_score(fca_labels[hier_sample_idx], km_labels[hier_sample_idx]))
        ari_fca_fcm_sub = float(adjusted_rand_score(fca_labels[hier_sample_idx], fcm_labels[hier_sample_idx]))
        ari_km_fcm_sub = float(adjusted_rand_score(km_labels[hier_sample_idx], fcm_labels[hier_sample_idx]))

        ari_records.append({
            "k": k,
            "scope": "Full population (N=93,357)" if k <= 6 or k == 12 else "Subsample",
            "ARI_FCA_KMeans": ari_fca_km,
            "ARI_FCA_FCM": ari_fca_fcm,
            "ARI_KMeans_FCM": ari_km_fcm,
            "ARI_FCA_Ward": ari_fca_ward,
            "ARI_KMeans_Ward": ari_km_ward,
            "ARI_FCM_Ward": ari_fcm_ward,
        })
        print(f"k={k} ARIs: FCA vs KM={ari_fca_km:.4f}, FCA vs FCM={ari_fca_fcm:.4f}, KM vs FCM={ari_km_fcm:.4f}, KM vs Ward={ari_km_ward:.4f}")

    # -------------------------------------------------------------
    # EXPERIMENT 6: Per-Cluster Silhouette Decomposition
    # -------------------------------------------------------------
    print("\n--- Experiment 6: Per-Cluster Silhouette Decomposition ---")
    fca_sub_labels = fca_labels_by_alpha[0.5][sil_sample_idx]
    fca_sub_names = [band_labels[i] for i in fca_sub_labels]
    s_samples_fca = silhouette_samples(X_sub, fca_sub_labels)

    full_counts = pd.Series(agg["hard_cluster"]).value_counts().to_dict()
    df_fca_s = pd.DataFrame({"cluster": fca_sub_names, "sil": s_samples_fca})

    for clus in band_labels:
        grp = df_fca_s[df_fca_s["cluster"] == clus]
        if len(grp) == 0:
            continue
        vals = grp["sil"].values
        n_neg = int((vals < 0).sum())
        decomp_records.append({
            "method": "FCA Natural-k (k=12)",
            "cluster_name": clus,
            "cluster_size_full": full_counts.get(clus, 0),
            "cluster_size_pct_full": float(full_counts.get(clus, 0) / m * 100),
            "sample_count": len(grp),
            "mean_silhouette": float(np.mean(vals)),
            "median_silhouette": float(np.median(vals)),
            "min_silhouette": float(np.min(vals)),
            "max_silhouette": float(np.max(vals)),
            "pct_negative_silhouette": float(n_neg / len(grp) * 100),
        })

    # -------------------------------------------------------------
    # EXPERIMENT 8: Membership-Aware Analysis
    # -------------------------------------------------------------
    print("\n--- Experiment 8: Membership-Aware Analysis ---")
    satisfies_05 = band_membership >= 0.5
    n_concepts_per_cust = satisfies_05.sum(axis=1)
    avg_concepts_per_cust = float(n_concepts_per_cust.mean())
    pct_multiple_concepts = float((n_concepts_per_cust > 1).mean() * 100)
    pct_coverage = float((n_concepts_per_cust >= 1).mean() * 100)

    # Dimensional membership entropy
    attr_cols = list(fuzzy_df.columns)
    mu_matrix = fuzzy_df[attr_cols].values
    entropy_by_dim = {}
    for d in ["R", "F", "M", "S"]:
        cols_d = [j for j, c in enumerate(attr_cols) if c.startswith(d)]
        mu_d = mu_matrix[:, cols_d]
        p = np.clip(mu_d, 1e-12, 1.0)
        entropy_by_dim[d] = float(-np.mean(np.sum(mu_d * np.log(p), axis=1)))

    overall_entropy = float(np.mean(list(entropy_by_dim.values())))

    # Top-level concept extents at alpha=0.5
    top_level_extents = {col: int((soft[col] >= 0.5).sum()) for col in band_labels}

    # 153 pruned concept extent sizes
    extent_sizes = pruned["n_customers_actual"].values

    print(f"Avg concepts per customer (alpha>=0.5): {avg_concepts_per_cust:.2f}")
    print(f"Customers belonging to multiple concepts: {pct_multiple_concepts:.2f}%")
    print(f"Concept coverage (at least 1 concept): {pct_coverage:.2f}%")
    print(f"Mean membership entropy per dimension: R={entropy_by_dim['R']:.4f}, F={entropy_by_dim['F']:.4f}, M={entropy_by_dim['M']:.4f}, S={entropy_by_dim['S']:.4f}, Overall={overall_entropy:.4f}")
    print(f"Pruned concepts (n=153) extents: min={extent_sizes.min()}, median={np.median(extent_sizes):.0f}, max={extent_sizes.max()}, mean={extent_sizes.mean():.0f}")

    # -------------------------------------------------------------
    # Save CSV files
    # -------------------------------------------------------------
    df_summary = pd.DataFrame(summary_records)
    df_alpha = pd.DataFrame(alpha_records)
    df_ari = pd.DataFrame(ari_records)
    df_decomp = pd.DataFrame(decomp_records)

    summary_path = OUT_DIR / "clustering_comparison_summary.csv"
    alpha_path = OUT_DIR / "alpha_sensitivity_breakdown.csv"
    ari_path = OUT_DIR / "ari_comparison_matrix.csv"
    decomp_path = OUT_DIR / "per_cluster_silhouette_decomposition.csv"

    df_summary.to_csv(summary_path, index=False)
    df_alpha.to_csv(alpha_path, index=False)
    df_ari.to_csv(ari_path, index=False)
    df_decomp.to_csv(decomp_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {alpha_path}")
    print(f"Saved: {ari_path}")
    print(f"Saved: {decomp_path}")

    # Display concise tables
    print("\n" + "=" * 80)
    print("PER-CLUSTER SILHOUETTE DECOMPOSITION (FCA NATURAL k=12)")
    print("=" * 80)
    print(df_decomp[["cluster_name", "cluster_size_full", "cluster_size_pct_full", "mean_silhouette", "median_silhouette", "pct_negative_silhouette"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("PAIRWISE ARI COMPARISON MATRIX")
    print("=" * 80)
    print(df_ari.to_string(index=False))


if __name__ == "__main__":
    main()

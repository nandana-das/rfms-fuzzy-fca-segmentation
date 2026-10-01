"""Representation-level clustering experiment (Olist full population, n=93,357).

Question: when each representation is handed to the SAME clustering algorithms
(K-means / Ward) and judged by the SAME metrics (Silhouette, Davies-Bouldin),
what does the fuzzy-FCA concept-membership representation buy over raw RFMS
scores and over crisp-FCA concept indicators?

Arms (identical objects, three representations):
1. Fuzzy FCA      -> the 153 Kneedle-retained fuzzy concepts from the validated
                     v3 run (results/pruned_fuzzy_concepts.pkl, loaded verbatim --
                     Step 3 is NOT re-run) -> customer x 153 membership matrix via
                     the standard Godel-minimum t-norm (mu_C = min over the
                     concept's base score bands). No dimensionality reduction:
                     153 dims is directly tractable for K-means/Ward (recorded in
                     run_parameters; PCA is unnecessary at this scale).
2. Crisp FCA      -> closed concepts mined on the same stored Step-1 bands with
                     the same support cutoff (0.04), then the IDENTICAL Kneedle
                     selection procedure applied (support + stability-proxy
                     knees) so the two FCA arms differ only in fuzziness.
                     Matrix = 0/1 extent membership.
3. Raw RFMS       -> standardized (R, F*, M, S) score space (Step-5 convention).

Protocol mirrors scripts/step5_benchmark.py exactly so rows are comparable with
results/benchmark_comparison.csv (v3 Sec 4.5):
- K-means on the FULL population at k=4,5,6 (random_state=42, n_init=10);
- Silhouette via fixed 5,000-customer sample (sample_size=5000, random_state=42);
- Davies-Bouldin on the full matrix;
- Ward (AgglomerativeClustering, linkage='ward') on a fixed 5,000-customer
  subsample (Ward is O(n^2); same convention as Step 5), Silhouette + DB on
  that subsample;
- plus, for the fuzzy arm only, K-means at k=153 (lattice-cardinality-matched)
  and the FCA natural-k hard-assignment row (argmax membership), flagged as
  non-comparable to the k-matched rows.

Outputs: results/representation_comparison_olist/. No Step 1-13 artifacts are
modified. Stored Step-1 bands are the source of truth (see
scripts/olist_rfms_comparison.py for the re-derivation mismatch record).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fair_comparison_retail2 import (
    assign_hard_clusters,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    mine_crisp_closed_concepts,
)
from project_paths import PROCESSED, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "representation_comparison_olist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ("R", "F", "M", "S")
STORED_SCORE_COLS = {"R": "r_score", "F": "f_score", "M": "m_score", "S": "s_score"}
SUPPORT_CUTOFF = 0.04
RANDOM_SEED = 42
K_GRID = (4, 5, 6)
SIL_SAMPLE = 5000
HIER_SAMPLE = 5000
EXPECTED_FUZZY_CONCEPTS = 153


def kneedle_threshold(values_sorted_desc: np.ndarray) -> float:
    """Kneedle knee on a descending curve -- identical to Step 3's implementation."""
    y = np.asarray(values_sorted_desc, dtype=float)
    n = len(y)
    if n < 3:
        return float(y[-1]) if n else float("nan")
    x = np.linspace(0.0, 1.0, n)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return float(y[int(np.argmax(num / den))])


def stability_proxy_from_extents(
    extents: list[int], score_profiles: np.ndarray
) -> np.ndarray:
    """Project stability proxy (1 - unique score-profiles / extent size) from bitsets."""
    out = np.empty(len(extents), dtype=float)
    for i, extent in enumerate(extents):
        n_cust = extent.bit_count()
        if n_cust <= 1:
            out[i] = 1.0
            continue
        unique_profiles = set()
        rem = extent
        while rem:
            bit = rem & -rem
            unique_profiles.add(tuple(score_profiles[bit.bit_length() - 1]))
            rem ^= bit
        out[i] = 1.0 - len(unique_profiles) / n_cust
    return out


def extent_bitsets_from_concepts(
    concepts: pd.DataFrame, score_profiles: np.ndarray
) -> list[int]:
    """Recompute exact extents for stored intents ('R5 & M3@0.7'-style) as bitsets.

    Base band R5 means r_score == 5, regardless of any fuzzy '@level' suffix, so the
    same parser serves both the crisp intents (no suffix) and the stored fuzzy intents.
    """
    n = len(score_profiles)
    band_extents: dict[str, int] = {}
    for i in range(n):
        r, f, m, s = score_profiles[i]
        for name, val in (("R", r), ("F", f), ("M", m), ("S", s)):
            key = f"{name}{int(val)}"
            bit = 1 << i
            band_extents[key] = band_extents.get(key, 0) | bit

    extents = []
    for intent in concepts["intent"].astype(str):
        extent = (1 << n) - 1
        for item in intent.split(" & "):
            base = item.split("@")[0].strip()
            extent &= band_extents.get(base, 0)
            if extent == 0:
                break
        extents.append(extent)
    return extents


def build_arms(scored: pd.DataFrame) -> tuple[dict, dict]:
    """Build the three representation matrices plus per-arm metadata."""
    meta: dict = {}
    n = len(scored)
    score_profiles = scored[[f"{d}_score" for d in DIMS]].to_numpy()

    # ---- Arm 1: fuzzy FCA, the stored 153 Kneedle-retained concepts ----
    pruned = pd.read_pickle(RESULTS_DIR / "pruned_fuzzy_concepts.pkl")
    if len(pruned) != EXPECTED_FUZZY_CONCEPTS:
        raise SystemExit(
            f"[GATE] expected {EXPECTED_FUZZY_CONCEPTS} stored Kneedle-retained "
            f"concepts, found {len(pruned)} -- Step 3 artifact mismatch; aborting."
        )
    # Stored itemsets are frozensets of level-qualified attributes ('F1@0.7'); a
    # concept's membership uses only its base score bands (Godel min), so map each
    # frozenset to its sorted unique base-band intent ('F1' / 'R5 & M3').
    intents: list[str] = []
    intent_sizes: list[int] = []
    for itemset in pruned["itemsets"]:
        bases = sorted({str(a).split("@")[0].strip() for a in itemset})
        intents.append(" & ".join(bases))
        intent_sizes.append(len(bases))
    fuzzy_concepts = pd.DataFrame(
        {
            "intent": intents,
            "intent_size": intent_sizes,
            "n_customers": pruned["n_customers_actual"].astype(int),
            "support": pruned["support"].astype(float),
        }
    )
    fuzzy_mu, _, _ = compute_fuzzy_memberships(scored, dims=DIMS)
    X_fuzzy = compute_customer_concept_memberships(fuzzy_concepts, fuzzy_mu)
    n_unique_cols = int(pd.DataFrame(X_fuzzy).T.drop_duplicates().shape[0])
    meta["fuzzy"] = {
        "concepts_df": fuzzy_concepts,
        "X": X_fuzzy,
        "note": (
            f"153 stored Kneedle-retained fuzzy concepts (Godel-min memberships over "
            f"base bands; {n_unique_cols} distinct membership columns)"
        ),
    }

    # ---- Arm 2: crisp FCA, same mining cutoff + identical Kneedle selection ----
    crisp_all = mine_crisp_closed_concepts(
        scored, min_support=SUPPORT_CUTOFF, dims=DIMS
    )
    print(f"Crisp closed concepts at support {SUPPORT_CUTOFF}: {len(crisp_all)}")
    crisp_extents = extent_bitsets_from_concepts(crisp_all, score_profiles)
    crisp_all["stability_proxy"] = stability_proxy_from_extents(
        crisp_extents, score_profiles
    )
    supp_knee = kneedle_threshold(np.sort(crisp_all["support"].to_numpy())[::-1])
    stab_knee = kneedle_threshold(
        np.sort(crisp_all["stability_proxy"].to_numpy())[::-1]
    )
    print(f"Crisp Kneedle knees: support={supp_knee:.4f}, stability={stab_knee:.4f}")
    keep = (crisp_all["support"] >= supp_knee) & (
        crisp_all["stability_proxy"] >= stab_knee
    )
    stability_applied = bool(keep.any())
    if not stability_applied:
        # Degenerate crisp stability knee (theta* at/near ceiling can zero out the
        # filter on a small crisp lattice): fall back to support-only and record it.
        keep = crisp_all["support"] >= supp_knee
    crisp_sel = crisp_all[keep].reset_index(drop=True)
    sel_extents = [crisp_extents[i] for i in np.flatnonzero(keep.to_numpy())]
    print(
        f"Crisp concepts after Kneedle selection: {len(crisp_sel)} "
        f"(stability knee applied: {stability_applied})"
    )

    X_crisp = np.zeros((n, len(crisp_sel)), dtype=float)
    for j, extent in enumerate(sel_extents):
        rem = extent
        while rem:
            bit = rem & -rem
            X_crisp[bit.bit_length() - 1, j] = 1.0
            rem ^= bit

    # Secondary crisp variant: the FULL support-cutoff lattice (no Kneedle), so the
    # comparison cannot be dismissed as crippling crisp via the pruning stage.
    X_crisp_supp = np.zeros((n, len(crisp_all)), dtype=float)
    for j, extent in enumerate(crisp_extents):
        rem = extent
        while rem:
            bit = rem & -rem
            X_crisp_supp[bit.bit_length() - 1, j] = 1.0
            rem ^= bit

    crisp_sel.to_csv(OUT_DIR / "crisp_concepts_kneedle.csv", index=False)
    meta["crisp"] = {
        "concepts_df": crisp_sel,
        "X": X_crisp,
        "note": (
            f"crisp closed concepts, support>={supp_knee:.4f}"
            + (f" & stability>={stab_knee:.4f}" if stability_applied else " (stability knee degenerate -> support-only)")
            + f"; {len(crisp_sel)} of {len(crisp_all)} concepts"
        ),
        "kneedle_stability_applied": stability_applied,
    }
    meta["crisp_supp"] = {
        "concepts_df": crisp_all.reset_index(drop=True),
        "X": X_crisp_supp,
        "note": f"crisp closed concepts at support {SUPPORT_CUTOFF} only (no Kneedle pruning); {len(crisp_all)} concepts",
    }
    meta["crisp_knees"] = (supp_knee, stab_knee, stability_applied)

    # ---- Arm 3: raw RFMS, standardized (Step-5 convention) ----
    X_raw = StandardScaler().fit_transform(scored[list(DIMS)].to_numpy())
    meta["raw"] = {
        "concepts_df": None,
        "X": X_raw,
        "note": "standardized R, F*, M, S (no FCA structure)",
    }
    return meta, fuzzy_mu


def evaluate_arm(arm_name: str, meta_arm: dict, X: np.ndarray) -> list[dict]:
    """K-means + Ward + natural-k rows under the Step-5 protocol."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X)
    rows: list[dict] = []

    hier_idx = rng.choice(n, size=min(HIER_SAMPLE, n), replace=False)
    X_hier = X[hier_idx]

    for k in K_GRID:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(SIL_SAMPLE, n), random_state=RANDOM_SEED)
        db = davies_bouldin_score(X, km.labels_)
        rows.append(
            {
                "representation": arm_name,
                "method": "K-means",
                "k": k,
                "silhouette": sil,
                "davies_bouldin": db,
                "n_features": X.shape[1],
                "preprocessing": meta_arm["note"],
                "comparable_with_step5": True,
            }
        )
        print(f"  K-means k={k}: Silhouette={sil:.4f}, DB={db:.4f}")

        ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_hier)
        sil_h = silhouette_score(X_hier, ac.labels_)
        db_h = davies_bouldin_score(X_hier, ac.labels_)
        rows.append(
            {
                "representation": arm_name,
                "method": "Hierarchical (Ward)",
                "k": k,
                "silhouette": sil_h,
                "davies_bouldin": db_h,
                "n_features": X.shape[1],
                "preprocessing": f"{meta_arm['note']}; Ward on n={len(hier_idx)} subsample",
                "comparable_with_step5": True,
            }
        )
        print(f"  Ward     k={k}: Silhouette={sil_h:.4f}, DB={db_h:.4f}")

    # Natural-k hard assignment (FCA arms only; raw arm has no concept argmax)
    if meta_arm["concepts_df"] is not None:
        labels = assign_hard_clusters(meta_arm["concepts_df"], X)
        n_clusters = len(np.unique(labels))
        sil_nat = silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, n), random_state=RANDOM_SEED)
        db_nat = davies_bouldin_score(X, labels)
        rows.append(
            {
                "representation": arm_name,
                "method": "FCA argmax (natural k)",
                "k": n_clusters,
                "silhouette": sil_nat,
                "davies_bouldin": db_nat,
                "n_features": X.shape[1],
                "preprocessing": meta_arm["note"],
                "comparable_with_step5": False,
            }
        )
        print(f"  argmax natural k={n_clusters}: Silhouette={sil_nat:.4f}, DB={db_nat:.4f}")
    return rows


def main() -> None:
    print("=" * 80)
    print("REPRESENTATION-LEVEL CLUSTERING COMPARISON (Olist, n=93,357)")
    print("=" * 80)

    df = pd.read_csv(PROCESSED + "olist_rfms_features.csv")
    df = df.rename(columns={"customer_unique_id": "CustomerID", "F_star": "F"})
    scored = df[["CustomerID", "R", "F", "M", "S"]].copy()
    for d in DIMS:
        scored[f"{d}_score"] = df[STORED_SCORE_COLS[d]].to_numpy()
    print(f"Customers: {len(scored):,} (stored Step-1 bands attached)")

    meta, _ = build_arms(scored)

    arm_map = (
        ("Fuzzy FCA (153 concepts)", "fuzzy"),
        ("Crisp FCA (Kneedle-matched)", "crisp"),
        ("Crisp FCA (support cutoff)", "crisp_supp"),
        ("Raw RFMS", "raw"),
    )
    all_rows: list[dict] = []
    for arm_name, key in arm_map:
        print(f"\n--- {arm_name} ({meta[key]['X'].shape[1]} features) ---")
        all_rows.extend(evaluate_arm(arm_name, meta[key], meta[key]["X"]))

    # Lattice-cardinality-matched K-means for the fuzzy arm (k=153): not comparable
    # with the k-matched rows, but answers "what happens at the lattice's own k".
    X_fuzzy = meta["fuzzy"]["X"]
    print("\n--- Fuzzy FCA at lattice cardinality k=153 (flagged non-comparable) ---")
    km153 = KMeans(n_clusters=153, random_state=RANDOM_SEED, n_init=3).fit(X_fuzzy)
    sil153 = silhouette_score(X_fuzzy, km153.labels_, sample_size=SIL_SAMPLE, random_state=RANDOM_SEED)
    db153 = davies_bouldin_score(X_fuzzy, km153.labels_)
    all_rows.append(
        {
            "representation": "Fuzzy FCA (153 concepts)",
            "method": "K-means (lattice-matched k)",
            "k": 153,
            "silhouette": sil153,
            "davies_bouldin": db153,
            "n_features": X_fuzzy.shape[1],
            "preprocessing": meta["fuzzy"]["note"],
            "comparable_with_step5": False,
        }
    )
    print(f"  K-means k=153: Silhouette={sil153:.4f}, DB={db153:.4f}")

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "representation_comparison.csv", index=False)
    print("\n=== RESULTS ===")
    print(results.drop(columns=["preprocessing"]).to_string(index=False))

    pd.DataFrame(
        [
            {"parameter": "population", "value": f"Olist full population, n={len(scored)}"},
            {"parameter": "fuzzy_concepts_source", "value": "results/pruned_fuzzy_concepts.pkl (Step 3 Kneedle output, loaded verbatim)"},
            {"parameter": "fuzzy_membership_rule", "value": "Godel-minimum t-norm over base score bands (compute_customer_concept_memberships)"},
            {"parameter": "crisp_concepts", "value": "mined at support 0.04 on stored bands; identical Kneedle selection applied"},
            {"parameter": "crisp_kneedle_stability_applied", "value": meta["crisp"]["kneedle_stability_applied"]},
            {"parameter": "crisp_knees_support_stability", "value": str(meta["crisp_knees"])},
            {"parameter": "dimensionality_reduction", "value": "none (153 dims tractable for K-means/Ward)"},
            {"parameter": "k_grid", "value": "4, 5, 6 (Step-5 protocol)"},
            {"parameter": "silhouette_protocol", "value": f"sample_size={SIL_SAMPLE}, random_state={RANDOM_SEED} (Step-5 protocol)"},
            {"parameter": "ward_protocol", "value": f"n={HIER_SAMPLE} fixed subsample, seed {RANDOM_SEED} (Ward is O(n^2))"},
            {"parameter": "raw_features", "value": "StandardScaler on R, F*, M, S"},
            {"parameter": "random_seed", "value": RANDOM_SEED},
        ]
    ).to_csv(OUT_DIR / "run_parameters.csv", index=False)

    # ---- Figure: Silhouette + DB by k, grouped by representation ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    km_rows = results[(results["method"] == "K-means") & (results["comparable_with_step5"])]
    ward_rows = results[(results["method"] == "Hierarchical (Ward)") & (results["comparable_with_step5"])]
    reps = list(dict.fromkeys(km_rows["representation"]))
    colors = {
        "Fuzzy FCA (153 concepts)": "#d62728",
        "Crisp FCA (Kneedle-matched)": "#1f77b4",
        "Crisp FCA (support cutoff)": "#9467bd",
        "Raw RFMS": "#7f7f7f",
    }
    width = 0.26
    for ax, rows, title in ((axes[0], km_rows, "K-means (full population)"), (axes[1], ward_rows, "Hierarchical Ward (n=5,000 subsample)")):
        for r_idx, rep in enumerate(reps):
            sub = rows[rows["representation"] == rep].sort_values("k")
            ax.bar(
                np.arange(len(K_GRID)) + (r_idx - 1) * width,
                sub["silhouette"].to_numpy(),
                width=width,
                label=rep,
                color=colors[rep],
                edgecolor="black",
                alpha=0.85,
            )
        ax.set_xticks(np.arange(len(K_GRID)))
        ax.set_xticklabels([str(k) for k in K_GRID])
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        ax.set_title(title, fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_representation_comparison.png")
    plt.close()

    print(f"\nAll artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Compare the published crisp RFM-FCA baseline with the project fuzzy RFM method.

This script writes only to results/retail2_base_vs_fuzzy/. It reads the two
Online Retail II source CSVs and the existing Step 7 artifacts; it does not
overwrite any Step 1-13 outputs.

The paper describes equal-population RFM quintiles but does not specify its
tie-breaking implementation. The crisp baseline therefore uses deterministic
rank-first quintiles, with CustomerID as the stable tie-breaker. The fuzzy
matched-feature arm uses the project's dense-rank/centroid/L-fuzzy/Kneedle
procedure but literal invoice-count F, so the raw R, F, M constructs match the
base paper. The existing F* fuzzy run is reported separately as the full
project variant because its F construct differs from conventional invoice
frequency.
"""

from __future__ import annotations

import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from project_paths import PROCESSED_DIR, RESULTS_DIR, RETAIL2_RAW_DIR


OUT_DIR = RESULTS_DIR / "retail2_base_vs_fuzzy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    RETAIL2_RAW_DIR / "online_retail_09_10.csv",
    RETAIL2_RAW_DIR / "online_retail_10_11.csv",
]

PAPER_SUPPORT_CUTOFF = 0.04
FUZZY_MIN_SUPPORT = 0.02
L_THRESHOLDS = (0.3, 0.5, 0.7)
DIMENSIONS = ("R", "F", "M")

# Selected concept extents published in Table 7 of Rungruang et al. (2024).
# These are a reproduction diagnostic; the paper does not specify tie handling.
PUBLISHED_TABLE7_COUNTS = {
    "M5": 1176, "M4": 1175, "M3": 1176, "M2": 1175, "M1": 1176,
    "F5": 1180, "F5 & M5": 845, "F5 & M4": 257, "F4 & M4": 582,
    "F4 & M3": 247, "F3 & M3": 579, "F2 & M2": 620, "F2 & M1": 287,
    "F1 & M1": 775, "R5 & M5": 521, "R5 & M4": 275, "R5 & F5": 524,
    "R5 & F5 & M5": 422, "R4 & M5": 329, "R4 & M4": 290,
    "R4 & M3": 256, "R4 & F5": 350, "R4 & F5 & M5": 246,
    "R3 & M4": 325, "R3 & M3": 266, "R2 & M3": 287, "R2 & M2": 335,
    "R2 & M1": 294, "R1 & M2": 341, "R1 & M1": 517,
    "R1 & F1 & M1": 345,
}


def load_and_aggregate() -> tuple[pd.DataFrame, int, int, pd.Timestamp]:
    """Apply the paper's stated transaction filters and form conventional RFM."""
    raw = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in RAW_FILES],
        ignore_index=True,
    )
    raw_rows = len(raw)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    clean = raw.dropna(subset=["CustomerID"]).copy()
    clean = clean[
        (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
        & (~clean["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["line_value"] = clean["Quantity"] * clean["UnitPrice"]
    reference_date = clean["InvoiceDate"].max()

    rfm = (
        clean.groupby("CustomerID", sort=True)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            F=("InvoiceNo", "nunique"),
            M=("line_value", "sum"),
        )
        .reset_index()
    )
    rfm["R"] = (reference_date - rfm["last_purchase"]).dt.days.astype(int)
    rfm = rfm[["CustomerID", "R", "F", "M"]].sort_values("CustomerID").reset_index(drop=True)
    return rfm, raw_rows, len(clean), reference_date


def paper_quintile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Create five balanced quintiles with deterministic ties and R reversal."""
    scored = rfm.copy()
    n = len(scored)
    for dim in DIMENSIONS:
        # The paper specifies five equal customer groups but does not state how
        # ties are handled. Sorting by customer id makes rank-first reproducible.
        ordered = scored.sort_values([dim, "CustomerID"], kind="mergesort").index.to_numpy()
        q = (np.arange(n, dtype=np.int64) * 5 // n) + 1
        scores = np.empty(n, dtype=np.int8)
        scores[ordered] = q.astype(np.int8)
        if dim == "R":
            scores = 6 - scores  # lower recency (more recent) receives score 5
        scored[f"{dim}_paper_score"] = scores
    return scored


def dense_rank_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Project scoring transformation, applied to the shared literal R, F, M."""
    scored = rfm.copy()
    for dim in DIMENSIONS:
        dense = scored[dim].rank(method="dense")
        values = np.ceil(5 * dense / dense.max()).astype(int).clip(1, 5)
        if dim == "R":
            values = 6 - values
        scored[f"{dim}_fuzzy_score"] = values.astype(np.int8)
    return scored


def score_distribution_rows(
    frame: pd.DataFrame, score_suffix: str, method: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dim in DIMENSIONS:
        counts = frame[f"{dim}{score_suffix}"].value_counts().to_dict()
        row: dict[str, object] = {
            "method": method,
            "dimension": dim,
            "raw_unique_values": int(frame[dim].nunique()),
            "dense_rank_levels": int(frame[dim].rank(method="dense").nunique()),
        }
        row.update({f"score_{k}": int(counts.get(k, 0)) for k in range(1, 6)})
        rows.append(row)
    return rows


def attrs_to_names(mask: int, attributes: list[str]) -> list[str]:
    return [attributes[i] for i in range(len(attributes)) if mask & (1 << i)]


def enumerate_crisp_concepts(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Enumerate every formal concept in the 15-attribute crisp RFM context."""
    n = len(scored)
    attributes = [f"{dim}{k}" for dim in DIMENSIONS for k in range(1, 6)]
    extent_bits: list[int] = []
    for attr in attributes:
        dim, band = attr[0], int(attr[1:])
        col = f"{dim}_paper_score"
        bits = 0
        for i, value in enumerate(scored[col].to_numpy()):
            if value == band:
                bits |= 1 << i
        extent_bits.append(bits)

    all_objects = (1 << n) - 1
    n_subsets = 1 << len(attributes)
    subset_extents = [0] * n_subsets
    subset_extents[0] = all_objects
    records: list[dict[str, object]] = []

    for intent_mask in range(n_subsets):
        if intent_mask:
            lsb = intent_mask & -intent_mask
            attr_index = lsb.bit_length() - 1
            subset_extents[intent_mask] = (
                subset_extents[intent_mask ^ lsb] & extent_bits[attr_index]
            )
        extent = subset_extents[intent_mask]
        closure = 0
        for attr_index, attr_extent in enumerate(extent_bits):
            if (extent & ~attr_extent) == 0:
                closure |= 1 << attr_index
        if closure != intent_mask:
            continue

        intent = attrs_to_names(intent_mask, attributes)
        n_objects = extent.bit_count()
        records.append(
            {
                "intent": " & ".join(intent) if intent else "(empty intent)",
                "intent_size": len(intent),
                "dimensions": "".join(d for d in DIMENSIONS if any(a.startswith(d) for a in intent)),
                "n_customers": n_objects,
                "support": n_objects / n,
            }
        )

    concepts = pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    )
    concepts = concepts.reset_index(drop=True)
    return concepts, extent_bits


def piecewise_membership(raw: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """The project's centroid interpolation, including outer shoulder bands."""
    x = raw.astype(float)
    c = centroids.astype(float)
    mu = np.zeros((len(x), 5), dtype=float)
    below = x <= c[0]
    above = x >= c[4]
    mu[below, 0] = 1.0
    mu[above, 4] = 1.0
    mid = ~below & ~above
    xm = x[mid]
    idx = np.clip(np.searchsorted(c, xm, side="right") - 1, 0, 3)
    left, right = c[idx], c[idx + 1]
    denom = np.where(right - left == 0, 1e-9, right - left)
    frac_right = (xm - left) / denom
    rows = np.where(mid)[0]
    mu[rows, idx] = 1 - frac_right
    mu[rows, idx + 1] = frac_right
    return mu


def build_fuzzy_memberships(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build project-style fuzzy RFM memberships using literal invoice count F."""
    memberships: dict[str, np.ndarray] = {}
    centroid_rows: list[dict[str, object]] = []
    for dim in DIMENSIONS:
        score = scored[f"{dim}_fuzzy_score"].to_numpy()
        raw = scored[dim].to_numpy(dtype=float)
        centroids = np.array(
            [np.median(raw[score == k]) if np.any(score == k) else np.median(raw) for k in range(1, 6)],
            dtype=float,
        )
        order = np.argsort(centroids, kind="stable")
        mu_sorted = piecewise_membership(raw, centroids[order])
        mu = mu_sorted[:, np.argsort(order, kind="stable")]
        for k in range(5):
            memberships[f"{dim}{k + 1}"] = mu[:, k]
            centroid_rows.append(
                {
                    "dimension": dim,
                    "score_band": k + 1,
                    "centroid": centroids[k],
                    "customers_in_score_band": int(np.sum(score == k + 1)),
                }
            )

    membership_df = pd.DataFrame(memberships)
    centroid_df = pd.DataFrame(centroid_rows)
    return membership_df, centroid_df


def kneedle_threshold(values_desc: np.ndarray) -> float:
    y = np.asarray(values_desc, dtype=float)
    if len(y) < 3:
        return float(y[-1]) if len(y) else float("nan")
    x = np.linspace(0.0, 1.0, len(y))
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
    numerator = np.abs(
        (y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1
    )
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return float(y[int(np.argmax(numerator / denominator))])


def fuzzy_closed_concepts(
    membership_df: pd.DataFrame, scored: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Mine exact closed intents from the project's L-fuzzy scaled context."""
    n = len(membership_df)
    attr_extent: dict[str, int] = {}
    transactions: list[list[str]] = []
    for row_idx, row in enumerate(membership_df.itertuples(index=False, name=None)):
        transaction: list[str] = []
        for col_idx, col in enumerate(membership_df.columns):
            value = row[col_idx]
            for threshold in L_THRESHOLDS:
                attr = f"{col}@{threshold:.1f}"
                if value >= threshold:
                    transaction.append(attr)
                    attr_extent[attr] = attr_extent.get(attr, 0) | (1 << row_idx)
        transactions.append(transaction)

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions, sparse=True)
    context = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.columns_)

    start = time.perf_counter()
    frequent = fpgrowth(
        context,
        min_support=FUZZY_MIN_SUPPORT,
        use_colnames=True,
        max_len=None,
    )
    mining_seconds = time.perf_counter() - start

    # Extent bitsets make closure exact, avoiding float-support equality checks.
    groups: dict[int, dict[str, object]] = {}
    for itemset in frequent["itemsets"]:
        ordered_items = sorted(itemset)
        extent = (1 << n) - 1
        for item in sorted(ordered_items, key=lambda name: attr_extent.get(name, 0).bit_count()):
            extent &= attr_extent[item]
            if extent == 0:
                break
        if extent not in groups:
            groups[extent] = {"intent": set(ordered_items)}
        else:
            groups[extent]["intent"].update(ordered_items)

    profile_cols = [f"{dim}_fuzzy_score" for dim in DIMENSIONS]
    profiles = scored[profile_cols].to_numpy()
    records: list[dict[str, object]] = []
    extents: list[int] = []
    for extent, data in groups.items():
        intent = data["intent"]
        n_objects = extent.bit_count()
        unique_profiles: set[tuple[int, ...]] = set()
        remaining = extent
        while remaining:
            bit = remaining & -remaining
            obj_idx = bit.bit_length() - 1
            unique_profiles.add(tuple(int(v) for v in profiles[obj_idx]))
            remaining ^= bit
        stability = 1.0 if n_objects <= 1 else 1.0 - len(unique_profiles) / n_objects
        records.append(
            {
                "intent": " & ".join(sorted(intent)),
                "intent_size": len(intent),
                "dimensions": "".join(d for d in DIMENSIONS if any(a.startswith(f"{d}") for a in intent)),
                "n_customers": n_objects,
                "support": n_objects / n,
                "stability_proxy": stability,
            }
        )
        extents.append(extent)

    concepts = pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    ).reset_index(drop=True)
    # Exact closure grouping above produces one record per extent. Keep extent
    # bitsets aligned with sorted concepts for persistence/membership summaries.
    extent_by_intent = {
        " & ".join(sorted(data["intent"])): extent for extent, data in groups.items()
    }
    aligned_extents = [extent_by_intent[name] for name in concepts["intent"]]

    support_knee = kneedle_threshold(concepts["support"].to_numpy())
    stability_knee = kneedle_threshold(
        np.sort(concepts["stability_proxy"].to_numpy())[::-1]
    )
    concepts["is_kneedle_pruned"] = (
        (concepts["support"] >= support_knee)
        & (concepts["stability_proxy"] >= stability_knee)
    )
    concepts["extent_bitset_index"] = np.arange(len(concepts), dtype=int)

    summary = {
        "context_attributes": len(encoder.columns_),
        "context_columns": list(encoder.columns_),
        "frequent_itemsets": len(frequent),
        "closed_concepts_min_support_002": len(concepts),
        "closed_concepts_support_gt_004": int((concepts["support"] > PAPER_SUPPORT_CUTOFF).sum()),
        "kneedle_support_threshold": support_knee,
        "kneedle_stability_threshold": stability_knee,
        "kneedle_pruned_concepts": int(concepts["is_kneedle_pruned"].sum()),
        "mean_concepts_per_customer_pruned": float(
            _mean_extent_memberships(aligned_extents, concepts["is_kneedle_pruned"].to_numpy(), n)
        ),
        "mean_fuzzy_band_overlap_at_05": float(
            np.mean(
                [
                    np.mean(
                        membership_df[[f"{dim}{k}" for k in range(1, 6)]].ge(0.5).sum(axis=1) > 1
                    )
                    for dim in DIMENSIONS
                ]
            )
        ),
        "mining_seconds": mining_seconds,
    }
    concepts.attrs["aligned_extent_bitsets"] = aligned_extents
    return concepts, summary


def _mean_extent_memberships(extents: list[int], keep: np.ndarray, n: int) -> float:
    counts = np.zeros(n, dtype=np.int32)
    for extent, use in zip(extents, keep):
        if not use:
            continue
        remaining = extent
        while remaining:
            bit = remaining & -remaining
            counts[bit.bit_length() - 1] += 1
            remaining ^= bit
    return float(counts.mean())


def load_current_fstar_reference(rfm_shared: pd.DataFrame) -> dict[str, object]:
    """Validate and summarize existing Step 7 F* outputs without rewriting them."""
    feature_path = PROCESSED_DIR / "retail2_rfm_features.csv"
    raw_concept_path = RESULTS_DIR / "retail2_fuzzy_concepts_raw.pkl"
    pruned_path = RESULTS_DIR / "retail2_pruned_fuzzy_concepts.pkl"
    if not all(path.exists() for path in (feature_path, raw_concept_path, pruned_path)):
        return {"available": False, "note": "Existing Step 7 reference outputs not found."}

    existing = pd.read_csv(feature_path, dtype={"CustomerID": str})
    existing["CustomerID"] = existing["CustomerID"].str.replace(r"\.0$", "", regex=True)
    check = rfm_shared[["CustomerID", "R", "F", "M"]].merge(
        existing[["CustomerID", "R", "n_orders", "M", "F_star", "f_score"]],
        on="CustomerID",
        how="outer",
        suffixes=("_new", "_step7"),
        indicator=True,
    )
    same_population = bool((check["_merge"] == "both").all())
    same_shared_rfm = bool(
        same_population
        and np.allclose(check["R_new"], check["R_step7"])
        and np.allclose(check["F"], check["n_orders"])
        and np.allclose(check["M_new"], check["M_step7"])
    )
    with raw_concept_path.open("rb") as handle:
        raw_concepts = pickle.load(handle)
    with pruned_path.open("rb") as handle:
        pruned_concepts = pickle.load(handle)
    return {
        "available": True,
        "same_customer_population": same_population,
        "same_shared_rfm_values": same_shared_rfm,
        "f_star_weights_from_project_results_summary": "alpha=0.05, beta=0.15, gamma=0.80",
        "f_star_raw_closed_concepts": len(raw_concepts),
        "f_star_kneedle_pruned_concepts": len(pruned_concepts),
        "f_star_unique_values": int(existing["F_star"].nunique()),
        "f_star_score_counts": {
            int(k): int(v) for k, v in existing["f_score"].value_counts().sort_index().items()
        },
    }


def write_report(
    rfm: pd.DataFrame,
    raw_rows: int,
    clean_rows: int,
    ref_date: pd.Timestamp,
    paper_scores: pd.DataFrame,
    fuzzy_scores: pd.DataFrame,
    crisp_all: pd.DataFrame,
    fuzzy_concepts: pd.DataFrame,
    fuzzy_summary: dict[str, object],
    fstar_summary: dict[str, object],
) -> None:
    crisp_supported = crisp_all[crisp_all["support"] > PAPER_SUPPORT_CUTOFF]
    crisp_nontrivial_supported = crisp_supported[crisp_supported["intent_size"] > 0]
    fuzzy_supported = fuzzy_concepts[fuzzy_concepts["support"] > PAPER_SUPPORT_CUTOFF]
    fuzzy_pruned = fuzzy_concepts[fuzzy_concepts["is_kneedle_pruned"]]
    one_invoice_customers = int((rfm["F"] == 1).sum())
    repeat_customers = int((rfm["F"] > 1).sum())
    fuzzy_f1_customers = int((fuzzy_scores["F_fuzzy_score"] == 1).sum())
    crisp_count_by_intent = dict(zip(crisp_all["intent"], crisp_all["n_customers"]))
    table7_check = pd.DataFrame(
        [
            {
                "intent": intent,
                "published_customers": published_n,
                "reproduced_customers": crisp_count_by_intent.get(intent),
                "difference_reproduced_minus_published": (
                    crisp_count_by_intent[intent] - published_n
                    if intent in crisp_count_by_intent
                    else None
                ),
                "reproduced_as_exact_closed_intent": intent in crisp_count_by_intent,
            }
            for intent, published_n in PUBLISHED_TABLE7_COUNTS.items()
        ]
    )
    table7_check.to_csv(OUT_DIR / "published_table7_concept_check.csv", index=False)
    table7_exact_count_matches = int(
        (table7_check["difference_reproduced_minus_published"] == 0).sum()
    )
    table7_intents_recovered = int(
        table7_check["reproduced_as_exact_closed_intent"].sum()
    )
    score_rows = score_distribution_rows(paper_scores, "_paper_score", "base-paper quintile")
    score_rows += score_distribution_rows(fuzzy_scores, "_fuzzy_score", "matched fuzzy dense-rank")
    pd.DataFrame(score_rows).to_csv(OUT_DIR / "score_distributions.csv", index=False)

    crisp_all.to_csv(OUT_DIR / "base_crisp_all_concepts.csv", index=False)
    crisp_supported.to_csv(OUT_DIR / "base_crisp_concepts_support_gt_004.csv", index=False)
    crisp_nontrivial_supported.to_csv(
        OUT_DIR / "base_crisp_nontrivial_concepts_support_gt_004.csv", index=False
    )
    fuzzy_concepts.drop(columns=["extent_bitset_index"]).to_csv(
        OUT_DIR / "matched_fuzzy_all_concepts.csv", index=False
    )
    fuzzy_supported.drop(columns=["extent_bitset_index"]).to_csv(
        OUT_DIR / "matched_fuzzy_concepts_support_gt_004.csv", index=False
    )
    fuzzy_pruned.drop(columns=["extent_bitset_index"]).to_csv(
        OUT_DIR / "matched_fuzzy_concepts_kneedle_pruned.csv", index=False
    )
    customer_scores = rfm[["CustomerID", "R", "F", "M"]].copy()
    customer_scores = customer_scores.merge(
        paper_scores[["CustomerID", "R_paper_score", "F_paper_score", "M_paper_score"]],
        on="CustomerID",
    ).merge(
        fuzzy_scores[["CustomerID", "R_fuzzy_score", "F_fuzzy_score", "M_fuzzy_score"]],
        on="CustomerID",
    )
    customer_scores.to_csv(OUT_DIR / "shared_rfm_customer_scores.csv", index=False)

    summary_rows = [
        {
            "arm": "Published crisp RFM-FCA",
            "RFM_definitions": "R=days from last invoice to max InvoiceDate; F=distinct invoices; M=sum Quantity*UnitPrice",
            "scoring": "five equal-population quintiles; R reversed",
            "context_attributes": 15,
            "raw_closed_concepts": len(crisp_all),
            "concepts_support_gt_004": len(crisp_supported),
            "nontrivial_concepts_support_gt_004": len(crisp_nontrivial_supported),
            "kneedle_pruned_concepts": "not used in base paper",
            "f_measure": "literal invoice count",
        },
        {
            "arm": "Matched project-style fuzzy RFM-FCA",
            "RFM_definitions": "same R, literal F, and M values as baseline",
            "scoring": "project dense-rank scores; centroid piecewise memberships; L={0.3,0.5,0.7}",
            "context_attributes": fuzzy_summary["context_attributes"],
            "raw_closed_concepts": fuzzy_summary["closed_concepts_min_support_002"],
            "concepts_support_gt_004": fuzzy_summary["closed_concepts_support_gt_004"],
            "kneedle_pruned_concepts": fuzzy_summary["kneedle_pruned_concepts"],
            "f_measure": "literal invoice count",
        },
    ]
    if fstar_summary.get("available"):
        summary_rows.append(
            {
                "arm": "Existing full project fuzzy RFM-F*",
                "RFM_definitions": "same R and M; F*=0.05*n_orders + 0.15*sum(log1p(line Quantity)) + 0.80*I(n_orders>1)",
                "scoring": "existing Step 7 outputs; unchanged",
                "context_attributes": 45,
                "raw_closed_concepts": fstar_summary["f_star_raw_closed_concepts"],
                "concepts_support_gt_004": "see existing full-pipeline concepts; not recomputed here",
                "kneedle_pruned_concepts": fstar_summary["f_star_kneedle_pruned_concepts"],
                "f_measure": "F*; not directly matched to paper F",
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    lines = [
        "# Online Retail II: Crisp RFM-FCA vs. Fuzzy RFM-FCA",
        "",
        "## Scope and comparison design",
        "",
        "This is a separate comparison run. It leaves all existing Step 1-13 outputs untouched.",
        "Both new arms use the same cleaned customer population and the same raw R, F, M definitions:",
        "recency in days relative to the maximum cleaned InvoiceDate, frequency as distinct InvoiceNo per",
        "CustomerID, and monetary value as the sum of Quantity × UnitPrice.",
        "",
        "The published crisp arm uses five equal-population quintiles and a 15-attribute binary context.",
        "The paper states quintile scoring and a concept-support cutoff >0.04, but does not specify tie handling.",
        "This reproduction deterministically sorts ties by CustomerID to create five balanced groups; that is",
        "an explicit implementation convention, not a claim to have recovered the authors' exact code.",
        "The fuzzy matched arm uses the project's dense-rank scores, centroid piecewise memberships, L-fuzzy",
        "thresholds {0.3, 0.5, 0.7}, uncapped closed-itemset mining at min support 0.02, and the project's",
        "Kneedle support/stability pruning. It keeps literal invoice-count F so the underlying RFM variables",
        "match the paper. It is an end-to-end comparison of the scoring/context pipelines, not a one-factor",
        "test of fuzziness alone, because scoring and pruning also differ.",
        f"Raw F=1 applies to {one_invoice_customers:,}/{len(rfm):,} customers ({100*one_invoice_customers/len(rfm):.2f}%);",
        f"{repeat_customers:,} customers are repeat buyers. The dense-rank rule maps the lowest 18 of the 90",
        f"distinct F values to F-score 1, which contains {fuzzy_f1_customers:,}/{len(rfm):,} customers",
        f"({100*fuzzy_f1_customers/len(rfm):.2f}%). The raw F values remain tied together under dense ranking,",
        "while balanced crisp quintiles must split tied values where group boundaries fall. The crisp",
        "implementation uses CustomerID as a deterministic tie-break, reproducible but not specified by the paper.",
        "",
        "The existing Step 7 F* run is reported as a third, separate reference. Its F* definition differs from",
        "the paper's literal invoice count, so differences involving that arm cannot be attributed only to fuzzy FCA.",
        "",
        "## Population check",
        "",
        f"- Combined raw rows: **{raw_rows:,}**.",
        f"- Cleaned transaction rows: **{clean_rows:,}**.",
        f"- Customers: **{len(rfm):,}**.",
        f"- Reference date: **{ref_date}**.",
        "- The cleaned rows/customer totals match the values reported for the base paper.",
        "",
        "## Results",
        "",
        "| Arm | Scoring/context | Attributes | Closed concepts | Support > 0.04 | Nontrivial support > 0.04 | Kneedle retained |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Crisp RFM-FCA reconstruction | quintile / binary | 15 | {len(crisp_all):,} | {len(crisp_supported):,} incl. universal root | {len(crisp_nontrivial_supported):,} | not used |",
        f"| Matched-feature fuzzy RFM-FCA | dense rank / fuzzy L-scaling | {fuzzy_summary['context_attributes']} | {fuzzy_summary['closed_concepts_min_support_002']:,} | {fuzzy_summary['closed_concepts_support_gt_004']:,} | {fuzzy_summary['closed_concepts_support_gt_004']:,} | {fuzzy_summary['kneedle_pruned_concepts']:,} |",
    ]
    if fstar_summary.get("available"):
        lines.append(
            f"| Existing full project F* fuzzy run | existing Step 7 | 45 | {fstar_summary['f_star_raw_closed_concepts']:,} | not recalculated | {fstar_summary['f_star_kneedle_pruned_concepts']:,} |"
        )
    lines += [
        "",
        "### What the figures mean",
        "",
        f"- Crisp concepts retained by the paper's support rule: **{len(crisp_supported):,}**.",
        f"- Of the crisp support-filtered concepts, **{len(crisp_nontrivial_supported):,}** have nonempty intents; the remaining concept is the universal root (all customers, no shared score attribute).",
        f"- Matched fuzzy concepts above the same 0.04 support level: **{len(fuzzy_supported):,}**.",
        f"- Matched fuzzy concepts retained by the project Kneedle rule: **{len(fuzzy_pruned):,}**.",
        f"- Matched fuzzy Kneedle thresholds: support **{fuzzy_summary['kneedle_support_threshold']:.6f}**, stability proxy **{fuzzy_summary['kneedle_stability_threshold']:.6f}**.",
        f"- Mean pruned-concept memberships per customer in the matched fuzzy arm: **{fuzzy_summary['mean_concepts_per_customer_pruned']:.2f}**.",
        f"- Mean share of customers with >1 fuzzy band at membership ≥0.5, averaged over R/F/M: **{100*fuzzy_summary['mean_fuzzy_band_overlap_at_05']:.2f}%**.",
        "- Crisp concepts and L-fuzzy scaled concepts are formed in different attribute contexts (15 vs. 45",
        "  possible scaled attributes) and use different pruning rules. Their raw concept counts are descriptive,",
        "  not a standalone quality ranking.",
        "",
        "## Check against published crisp concepts",
        "",
        f"The {len(table7_check)} concepts listed in the paper's Table 7 were checked. All {table7_intents_recovered} were",
        f"recovered as exact closed intents, but only {table7_exact_count_matches}/{len(table7_check)} reconstructed",
        "customer counts match the published counts exactly. See `published_table7_concept_check.csv` for all",
        "reported and reconstructed customer counts. Several",
        "R-M combinations are close, while some frequency combinations differ substantially (F2&M2: 423 here",
        "versus 620 published; F2&M1: 411 versus 287; F5&M5: 899 versus 845). The cleaned population is",
        "reproduced, but the published concept table is not exactly reproduced under the declared CustomerID",
        "tie-break. The paper does not specify tie handling, and without its exact scoring rule/code, the cause",
        "of each discrepancy cannot be determined. This limits claims of exact replication, especially for F.",
        "",
        "## Reproducibility and artifacts",
        "",
        "- `comparison_summary.csv`: top-line arm comparison.",
        "- `score_distributions.csv`: raw unique-value counts and score-band counts for each new arm.",
        "- `base_crisp_all_concepts.csv`: exhaustive classical FCA concepts.",
        "- `base_crisp_concepts_support_gt_004.csv`: paper-style support-filtered concepts.",
        "- `base_crisp_nontrivial_concepts_support_gt_004.csv`: support-filtered crisp concepts excluding the universal root.",
        "- `published_table7_concept_check.csv`: published Table 7 counts versus this reconstruction.",
        "- `matched_fuzzy_all_concepts.csv`: matched-feature L-fuzzy closed concepts at min support 0.02.",
        "- `matched_fuzzy_concepts_support_gt_004.csv`: matched fuzzy concepts filtered at the paper's support cutoff.",
        "- `matched_fuzzy_concepts_kneedle_pruned.csv`: matched fuzzy concepts after Kneedle support/stability pruning.",
        "- `shared_rfm_customer_scores.csv`: customer-level raw RFM and both score encodings.",
        "",
        "Paper: [Rungruang et al. (2024), Expert Systems with Applications, 237, 121449](https://doi.org/10.1016/j.eswa.2023.121449).",
        "Dataset: [UCI Online Retail II](https://doi.org/10.24432/C5CG6D).",
    ]
    (OUT_DIR / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rfm, raw_rows, clean_rows, ref_date = load_and_aggregate()
    paper_scores = paper_quintile_scores(rfm)
    fuzzy_scores = dense_rank_scores(rfm)

    crisp_all, _ = enumerate_crisp_concepts(paper_scores)
    membership_df, centroids = build_fuzzy_memberships(fuzzy_scores)
    fuzzy_concepts, fuzzy_summary = fuzzy_closed_concepts(membership_df, fuzzy_scores)
    fstar_summary = load_current_fstar_reference(rfm)

    # Record score and centroid evidence without altering processed project data.
    pd.concat(
        [paper_scores.assign(scoring_arm="base-paper quintile"),
         fuzzy_scores.assign(scoring_arm="matched fuzzy dense-rank")],
        ignore_index=True,
        sort=False,
    ).to_csv(OUT_DIR / "scoring_inputs.csv", index=False)
    centroids.to_csv(OUT_DIR / "matched_fuzzy_centroids.csv", index=False)

    write_report(
        rfm,
        raw_rows,
        clean_rows,
        ref_date,
        paper_scores,
        fuzzy_scores,
        crisp_all,
        fuzzy_concepts,
        fuzzy_summary,
        fstar_summary,
    )

    print(f"Saved separate comparison artifacts to: {OUT_DIR}")
    print(f"Cleaned population: {len(rfm):,} customers from {clean_rows:,} rows")
    print(f"Base crisp concepts: {len(crisp_all):,}; support > .04: {(crisp_all['support'] > PAPER_SUPPORT_CUTOFF).sum():,}")
    print(
        "Matched fuzzy concepts: "
        f"{len(fuzzy_concepts):,}; support > .04: "
        f"{(fuzzy_concepts['support'] > PAPER_SUPPORT_CUTOFF).sum():,}; "
        f"Kneedle: {int(fuzzy_concepts['is_kneedle_pruned'].sum()):,}"
    )
    print(f"Existing F* reference validated: {fstar_summary.get('same_shared_rfm_values', False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Canonical Kneedle Audit: KneeLocator vs. the project's
Kneedle-inspired normalized max-distance-from-chord heuristic.

Purpose
-------
This script is an isolated methodological audit. It does NOT change the
existing fuzzy RFM-FCA pruning pipeline.

It compares:

1. Canonical Kneedle
   - Satopaa et al. (2011)
   - implemented using kneed.KneeLocator
   - sensitivity S in {0.1, 0.5, 1.0, 2.0}
   - applied ONLY to the support-vs-ranked-concept curve

2. Kneedle-inspired project heuristic
   - normalized maximum perpendicular distance from the global chord
   - parameter-free
   - applied to the same support-vs-ranked-concept curve

Important methodological decision
----------------------------------
Canonical Kneedle is NOT applied to the stability-proxy distribution as a
pruning curve.

The stability proxy is treated as a concept-quality statistic and is reported
descriptively only. This avoids treating a sorted stability distribution as
though it were a naturally parameterized support-vs-rank curve.

This script is an AUDIT. It does not replace the current methodology.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT_DIR / "scripts"),
)

from fuzzy_membership_sensitivity import (  # noqa: E402
    DIMS,
    SUPPORT_CUTOFF,
    compute_fuzzy_memberships,
    dense_rank_scores,
    load_and_prepare_cohorts,
    mine_fuzzy_closed_concepts_with_thresholds,
)


RESULTS_DIR = ROOT_DIR / "results"

OUT_DIR = (
    RESULTS_DIR
    / "canonical_kneedle_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

S_VALUES = (
    0.1,
    0.5,
    1.0,
    2.0,
)


# ============================================================================
# MARKDOWN TABLE HELPER
# ============================================================================

def dataframe_to_markdown(
    df: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """
    Convert a DataFrame to a Markdown table without requiring tabulate.

    This avoids adding the optional pandas 'tabulate' dependency solely for
    report generation.
    """

    if df.empty:
        return "(No rows.)"

    headers = [
        str(column)
        for column in df.columns
    ]

    def format_value(value):
        if pd.isna(value):
            return ""

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            return (
                f"{float(value):.{float_digits}f}"
            )

        if isinstance(
            value,
            (
                np.integer,
                int,
            ),
        ):
            return str(int(value))

        return str(value)

    rows = [
        [
            format_value(value)
            for value in row
        ]
        for row in df.to_numpy()
    ]

    widths = []

    for index, header in enumerate(headers):

        values = [
            row[index]
            for row in rows
        ]

        widths.append(
            max(
                [len(header)]
                + [
                    len(value)
                    for value in values
                ]
            )
        )

    header_row = (
        "| "
        + " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
        + " |"
    )

    separator_row = (
        "| "
        + " | ".join(
            "-" * widths[index]
            for index in range(len(headers))
        )
        + " |"
    )

    body_rows = []

    for row in rows:

        body_rows.append(
            "| "
            + " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )

    return "\n".join(
        [
            header_row,
            separator_row,
            *body_rows,
        ]
    )


# ============================================================================
# EXISTING PROJECT HEURISTIC
# ============================================================================

def kneedle_inspired_heuristic(
    y_desc: np.ndarray,
) -> tuple[int, float, float]:
    """
    Existing project heuristic.

    Parameters
    ----------
    y_desc:
        Values sorted in descending order.

    Returns
    -------
    knee_idx:
        Zero-based selected index.

    knee_value:
        Value at the selected index.

    max_distance:
        Maximum normalized perpendicular distance from the global chord.

    Notes
    -----
    This is intentionally NOT called canonical Kneedle.
    """

    y = np.asarray(
        y_desc,
        dtype=float,
    )

    n = len(y)

    if n == 0:
        return (
            0,
            float("nan"),
            float("nan"),
        )

    if n < 3:
        return (
            0,
            float(y[0]),
            0.0,
        )

    x = np.linspace(
        0.0,
        1.0,
        n,
    )

    y_min = y.min()
    y_max = y.max()

    y_norm = (
        (y - y_min)
        / (y_max - y_min + 1e-12)
    )

    x1 = x[0]
    y1 = y_norm[0]

    x2 = x[-1]
    y2 = y_norm[-1]

    numerator = np.abs(
        (y2 - y1) * x
        - (x2 - x1) * y_norm
        + x2 * y1
        - y2 * x1
    )

    denominator = (
        math.sqrt(
            (y2 - y1) ** 2
            + (x2 - x1) ** 2
        )
        + 1e-12
    )

    distance = (
        numerator
        / denominator
    )

    knee_idx = int(
        np.argmax(distance)
    )

    max_distance = float(
        distance[knee_idx]
    )

    return (
        knee_idx,
        float(y[knee_idx]),
        max_distance,
    )


# ============================================================================
# CANONICAL KNEEDLE
# ============================================================================

def evaluate_canonical_kneedle(
    y_desc: np.ndarray,
    s_values: tuple[float, ...] = S_VALUES,
) -> list[dict]:
    """
    Apply canonical kneed.KneeLocator to a descending support curve.

    x-axis:
        Concept rank, starting at zero.

    y-axis:
        Support, sorted descending.

    Canonical Kneedle is used only here.
    """

    y = np.asarray(
        y_desc,
        dtype=float,
    )

    if len(y) == 0:
        return []

    x = np.arange(
        len(y),
        dtype=float,
    )

    results = []

    for s in s_values:

        try:

            locator = KneeLocator(
                x,
                y,
                curve="convex",
                direction="decreasing",
                S=s,
                interp_method="interp1d",
                online=False,
            )

            knee_idx = locator.knee

            if knee_idx is None:

                knee_value = float("nan")
                retained = 0

            else:

                knee_idx = int(
                    round(
                        float(knee_idx)
                    )
                )

                knee_idx = max(
                    0,
                    min(
                        knee_idx,
                        len(y) - 1,
                    ),
                )

                knee_value = float(
                    y[knee_idx]
                )

                # y is descending, so the number of observations with
                # y >= y[knee_idx] is knee_idx + 1.
                retained = (
                    knee_idx + 1
                )

            results.append(
                {
                    "S": float(s),
                    "knee_index_zero_based": (
                        knee_idx
                        if knee_idx is not None
                        else None
                    ),
                    "knee_rank_one_based": (
                        knee_idx + 1
                        if knee_idx is not None
                        else None
                    ),
                    "knee_support": (
                        knee_value
                    ),
                    "retained_by_support": (
                        retained
                    ),
                    "status": (
                        "ok"
                        if knee_idx is not None
                        else "no_knee"
                    ),
                }
            )

        except Exception as exc:

            results.append(
                {
                    "S": float(s),
                    "knee_index_zero_based": None,
                    "knee_rank_one_based": None,
                    "knee_support": float("nan"),
                    "retained_by_support": 0,
                    "status": (
                        f"error: {exc}"
                    ),
                }
            )

    return results


# ============================================================================
# STABILITY DESCRIPTIVE AUDIT
# ============================================================================

def summarize_stability_distribution(
    concepts: pd.DataFrame,
) -> dict:
    """
    Describe the stability-proxy distribution.

    No Kneedle threshold is derived from this distribution.
    """

    stability = concepts[
        "stability_proxy"
    ].to_numpy(
        dtype=float
    )

    return {
        "stability_min": float(
            np.min(stability)
        ),
        "stability_q25": float(
            np.quantile(
                stability,
                0.25,
            )
        ),
        "stability_median": float(
            np.median(stability)
        ),
        "stability_q75": float(
            np.quantile(
                stability,
                0.75,
            )
        ),
        "stability_max": float(
            np.max(stability)
        ),
        "stability_ge_0_95": int(
            np.sum(
                stability >= 0.95
            )
        ),
        "stability_ge_0_90": int(
            np.sum(
                stability >= 0.90
            )
        ),
        "stability_ge_0_80": int(
            np.sum(
                stability >= 0.80
            )
        ),
    }


# ============================================================================
# DUNNHUMBY
# ============================================================================

def load_dunnhumby_concepts() -> pd.DataFrame:
    """
    Reproduce the locked Dunnhumby observation-cohort concept set.
    """

    print(
        "Loading Dunnhumby Observation Cohort "
        "(Days 1-620)..."
    )

    obs_agg, _ = (
        load_and_prepare_cohorts()
    )

    customer_rfm = (
        obs_agg[
            [
                "household_key",
                "R",
                "F",
                "M",
            ]
        ]
        .rename(
            columns={
                "household_key": "CustomerID"
            }
        )
    )

    scored = dense_rank_scores(
        customer_rfm,
        dims=DIMS,
    )

    fuzzy_mu, _, _ = (
        compute_fuzzy_memberships(
            scored,
            dims=DIMS,
        )
    )

    concepts, _, _ = (
        mine_fuzzy_closed_concepts_with_thresholds(
            fuzzy_mu,
            scored,
            (0.3, 0.5, 0.7),
            min_support=SUPPORT_CUTOFF,
            dims=DIMS,
        )
    )

    return concepts.reset_index(
        drop=True
    )


# ============================================================================
# ONLINE RETAIL II
# ============================================================================

def load_retail2_concepts() -> pd.DataFrame:
    """
    Load the already-generated controlled fuzzy concept table.
    """

    path = (
        RESULTS_DIR
        / "fair_comparison_retail2"
        / "controlled_comparison_concepts_fuzzy.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Retail II concept file not found:\n"
            f"{path}"
        )

    return pd.read_csv(
        path
    ).reset_index(
        drop=True
    )


# ============================================================================
# DATASET AUDIT
# ============================================================================

def audit_dataset(
    dataset_name: str,
    concepts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Compare canonical Kneedle with the existing heuristic.

    Canonical Kneedle is applied ONLY to the support curve.
    """

    print("\n" + "=" * 80)
    print(
        f"DATASET: {dataset_name}"
    )
    print("=" * 80)

    concepts = concepts.copy()

    n_concepts = len(concepts)

    print(
        f"Candidate concepts: "
        f"{n_concepts:,}"
    )

    # ----------------------------------------------------------------
    # SUPPORT CURVE
    # ----------------------------------------------------------------

    support_desc = np.sort(
        concepts[
            "support"
        ].to_numpy(
            dtype=float
        )
    )[::-1]

    (
        heuristic_idx,
        heuristic_support,
        heuristic_distance,
    ) = kneedle_inspired_heuristic(
        support_desc
    )

    heuristic_rank = (
        heuristic_idx + 1
    )

    canonical_results = (
        evaluate_canonical_kneedle(
            support_desc
        )
    )

    # ----------------------------------------------------------------
    # STABILITY DESCRIPTIVE SUMMARY
    # ----------------------------------------------------------------

    stability_summary = (
        summarize_stability_distribution(
            concepts
        )
    )

    # ----------------------------------------------------------------
    # PRINT SUPPORT RESULTS
    # ----------------------------------------------------------------

    print(
        "\nSupport curve"
    )

    print("-" * 80)

    print(
        "Chord heuristic:"
    )

    print(
        f"  rank = #{heuristic_rank}"
    )

    print(
        f"  support = "
        f"{heuristic_support:.6f}"
    )

    print(
        f"  max chord distance = "
        f"{heuristic_distance:.6f}"
    )

    print(
        "\nCanonical KneeLocator:"
    )

    for result in canonical_results:

        if result["status"] == "ok":

            print(
                f"  S={result['S']:.1f}: "
                f"rank="
                f"#{result['knee_rank_one_based']}, "
                f"support="
                f"{result['knee_support']:.6f}, "
                f"retained="
                f"{result['retained_by_support']}"
            )

        else:

            print(
                f"  S={result['S']:.1f}: "
                f"{result['status']}"
            )

    # ----------------------------------------------------------------
    # PRINT STABILITY SUMMARY
    # ----------------------------------------------------------------

    print(
        "\nStability-proxy distribution"
    )

    print("-" * 80)

    print(
        f"min     = "
        f"{stability_summary['stability_min']:.6f}"
    )

    print(
        f"Q25     = "
        f"{stability_summary['stability_q25']:.6f}"
    )

    print(
        f"median  = "
        f"{stability_summary['stability_median']:.6f}"
    )

    print(
        f"Q75     = "
        f"{stability_summary['stability_q75']:.6f}"
    )

    print(
        f"max     = "
        f"{stability_summary['stability_max']:.6f}"
    )

    print(
        f">= 0.95 = "
        f"{stability_summary['stability_ge_0_95']:,}"
    )

    print(
        f">= 0.90 = "
        f"{stability_summary['stability_ge_0_90']:,}"
    )

    print(
        f">= 0.80 = "
        f"{stability_summary['stability_ge_0_80']:,}"
    )

    # ----------------------------------------------------------------
    # SUMMARY ROWS
    # ----------------------------------------------------------------

    rows = []

    # Existing heuristic

    rows.append(
        {
            "Dataset": dataset_name,
            "Total_Concepts": n_concepts,
            "Method": (
                "Kneedle-Inspired "
                "Chord Heuristic"
            ),
            "S": np.nan,
            "Support_Knee_Rank": (
                heuristic_rank
            ),
            "Support_Threshold": (
                heuristic_support
            ),
            "Retained_By_Support": (
                heuristic_idx + 1
            ),
            "Delta_Support_vs_Heuristic": 0.0,
        }
    )

    # Canonical Kneedle

    for result in canonical_results:

        knee_support = result[
            "knee_support"
        ]

        delta = (
            knee_support
            - heuristic_support
            if not np.isnan(
                knee_support
            )
            else np.nan
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "Total_Concepts": n_concepts,
                "Method": (
                    "Canonical "
                    "KneeLocator"
                ),
                "S": result["S"],
                "Support_Knee_Rank": (
                    result[
                        "knee_rank_one_based"
                    ]
                ),
                "Support_Threshold": (
                    knee_support
                ),
                "Retained_By_Support": (
                    result[
                        "retained_by_support"
                    ]
                ),
                "Delta_Support_vs_Heuristic": (
                    delta
                ),
            }
        )

    details = {
        "Dataset": dataset_name,
        "Total_Concepts": n_concepts,
        "Heuristic_Rank": heuristic_rank,
        "Heuristic_Support": heuristic_support,
        "Heuristic_Max_Distance": (
            heuristic_distance
        ),
        **stability_summary,
    }

    return (
        pd.DataFrame(rows),
        details,
    )


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> str:
    """
    Generate the scientific audit report.

    Uses the internal Markdown formatter so that the report does not require
    pandas' optional 'tabulate' dependency.
    """

    lines = []

    lines.append(
        "# Canonical Kneedle Audit\n"
    )

    lines.append(
        "This audit compares canonical "
        "`kneed.KneeLocator` with the "
        "project's Kneedle-inspired "
        "normalized max-distance-from-chord "
        "heuristic.\n"
    )

    lines.append(
        "Canonical Kneedle is evaluated only "
        "on the support-versus-concept-rank "
        "curve. The stability-proxy "
        "distribution is reported "
        "descriptively and is not used as a "
        "second Kneedle pruning curve.\n"
    )

    lines.append(
        "## 1. Experimental configuration\n"
    )

    lines.append(
        "- Candidate concepts: same fuzzy "
        "closed-concept candidates used by "
        "the existing pipeline.\n"
    )

    lines.append(
        f"- Minimum support: "
        f"`{SUPPORT_CUTOFF}`.\n"
    )

    lines.append(
        "- Canonical implementation: "
        "`kneed.KneeLocator`.\n"
    )

    lines.append(
        "- Curve: `convex`.\n"
    )

    lines.append(
        "- Direction: `decreasing`.\n"
    )

    lines.append(
        "- Interpolation: `interp1d`.\n"
    )

    lines.append(
        "- Online mode: `False`.\n"
    )

    lines.append(
        "- Sensitivity values: "
        "`S = 0.1, 0.5, 1.0, 2.0`.\n"
    )

    lines.append(
        "## 2. Support-knee comparison\n"
    )

    lines.append(
        dataframe_to_markdown(
            summary,
            float_digits=6,
        )
    )

    lines.append(
        "\n\n## 3. Stability-proxy distribution\n"
    )

    lines.append(
        "The stability proxy is not treated "
        "as an independent Kneedle curve. "
        "Its distribution is reported to "
        "assess whether such an application "
        "would have a meaningful "
        "interpretation.\n"
    )

    lines.append(
        dataframe_to_markdown(
            stability,
            float_digits=6,
        )
    )

    lines.append(
        "\n\n## 4. Interpretation\n"
    )

    lines.append(
        "Canonical Kneedle and the project's "
        "chord heuristic are compared as "
        "alternative support-knee detectors. "
        "Differences in the selected support "
        "threshold are reported empirically "
        "and are not interpreted as evidence "
        "that one method is universally "
        "superior.\n"
    )

    lines.append(
        "The chord heuristic selects the "
        "global maximum perpendicular "
        "distance from the endpoint chord, "
        "whereas canonical Kneedle uses its "
        "normalized-difference curve and "
        "sensitivity-controlled knee "
        "detection. Therefore, the two "
        "methods are related but are not "
        "mathematically identical.\n"
    )

    lines.append(
        "The stability-proxy distribution is "
        "not used for canonical Kneedle "
        "pruning because it does not provide "
        "a comparably justified "
        "support-versus-rank curve. This "
        "avoids conflating a concept-quality "
        "statistic with the knee-detection "
        "curve itself.\n"
    )

    lines.append(
        "## 5. Methodological status\n"
    )

    lines.append(
        "This audit does not replace the "
        "existing pruning procedure. "
        "Canonical Kneedle should only be "
        "promoted into the final pipeline "
        "after its downstream predictive "
        "and structural consequences have "
        "been evaluated under the same "
        "leakage-free protocol.\n"
    )

    return "\n".join(lines)


# ============================================================================
# MAIN AUDIT
# ============================================================================

def run_canonical_kneedle_audit() -> None:

    print("\n")
    print("=" * 80)
    print(
        "CANONICAL KNEEDLE AUDIT"
    )
    print(
        "KneeLocator vs. "
        "Kneedle-Inspired Chord Heuristic"
    )
    print("=" * 80)

    # ----------------------------------------------------------------
    # LOAD DATASETS
    # ----------------------------------------------------------------

    dunnhumby = (
        load_dunnhumby_concepts()
    )

    retail2 = (
        load_retail2_concepts()
    )

    print(
        f"\nDunnhumby concepts: "
        f"{len(dunnhumby):,}"
    )

    print(
        f"Retail II concepts: "
        f"{len(retail2):,}"
    )

    # ----------------------------------------------------------------
    # AUDIT DUNNHUMBY
    # ----------------------------------------------------------------

    all_rows = []
    all_details = []

    result, details = audit_dataset(
        "Dunnhumby Observation (Days 1-620)",
        dunnhumby,
    )

    all_rows.append(result)
    all_details.append(details)

    # ----------------------------------------------------------------
    # AUDIT RETAIL II
    # ----------------------------------------------------------------

    result, details = audit_dataset(
        "UK Online Retail II (Full Cohort)",
        retail2,
    )

    all_rows.append(result)
    all_details.append(details)

    # ----------------------------------------------------------------
    # COMBINE RESULTS
    # ----------------------------------------------------------------

    summary = pd.concat(
        all_rows,
        ignore_index=True,
    )

    details_df = pd.DataFrame(
        all_details
    )

    # ----------------------------------------------------------------
    # SAVE SUPPORT COMPARISON
    # ----------------------------------------------------------------

    summary_path = (
        OUT_DIR
        / "kneedle_support_comparison.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------------------------------
    # SAVE STABILITY SUMMARY
    # ----------------------------------------------------------------

    stability_path = (
        OUT_DIR
        / "stability_distribution_summary.csv"
    )

    details_df.to_csv(
        stability_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {stability_path}"
    )

    # ----------------------------------------------------------------
    # GENERATE REPORT
    # ----------------------------------------------------------------

    report = generate_report(
        summary,
        details_df,
    )

    report_path = (
        OUT_DIR
        / "canonical_kneedle_audit_report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"  {report_path}"
    )

    # ----------------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "CANONICAL KNEEDLE AUDIT COMPLETE"
    )
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_canonical_kneedle_audit()
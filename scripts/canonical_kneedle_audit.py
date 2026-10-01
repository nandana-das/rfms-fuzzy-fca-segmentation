"""Canonical Kneedle audit for fuzzy RFM-FCA concept pruning.

This is an isolated experiment. It does not change the existing methodology/results.

The experiment:
1. mines the same fuzzy closed-concept candidates used by the current pipeline;
2. applies the canonical kneed.KneeLocator algorithm to the ranked support curve;
3. audits sensitivity to S;
4. reports the resulting support threshold and number of concepts retained;
5. optionally audits the stability-proxy distribution, but does NOT use that
   distribution as a canonical Kneedle pruning curve unless explicitly justified.

Canonical Kneedle parameters for the support curve:
- x: concept rank after sorting support descending
- y: support, descending
- curve='convex'
- direction='decreasing'
- online=False
- interp_method='interp1d'
- S is sensitivity and is audited over a small prespecified grid.

Run this on the local project environment where the Dunnhumby data are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dunnhumby_rfm_fca_validation import (  # noqa: E402
    DIMS,
    SUPPORT_CUTOFF,
    aggregate_rfm_dunnhumby,
    compute_fuzzy_memberships,
    dense_rank_scores,
    load_and_audit_transactions,
)
from fair_comparison_retail2 import mine_fuzzy_closed_concepts as _legacy_miner  # noqa: E402


S_GRID = (0.5, 1.0, 2.0, 3.0)


def canonical_support_knee(
    support_desc: np.ndarray,
    S: float = 1.0,
    online: bool = False,
) -> dict:
    """Run canonical kneed.KneeLocator on the descending support curve."""
    y = np.asarray(support_desc, dtype=float)
    x = np.arange(len(y), dtype=float)

    if len(y) < 3 or np.allclose(y, y[0]):
        return {
            "S": S,
            "knee_index": np.nan,
            "knee_support": np.nan,
            "n_retained": 0,
            "status": "no-knee-degenerate-curve",
        }

    kl = KneeLocator(
        x,
        y,
        S=S,
        curve="convex",
        direction="decreasing",
        interp_method="interp1d",
        online=online,
    )

    if kl.knee is None:
        return {
            "S": S,
            "knee_index": np.nan,
            "knee_support": np.nan,
            "n_retained": 0,
            "status": "no-knee",
        }

    # x is integer rank, but KneeLocator returns a numeric value.
    knee_index = int(round(float(kl.knee)))
    knee_index = min(max(knee_index, 0), len(y) - 1)
    threshold = float(y[knee_index])

    return {
        "S": S,
        "knee_index": knee_index,
        "knee_support": threshold,
        "n_retained": int(np.sum(y >= threshold)),
        "status": "ok",
    }


def audit_stability_distribution(stability: np.ndarray, S: float = 1.0) -> dict:
    """Audit Kneedle on sorted stability values without using it for pruning."""
    y = np.sort(np.asarray(stability, dtype=float))[::-1]
    result = canonical_support_knee(y, S=S, online=False)
    result["metric"] = "stability_proxy_audit_only"
    return result


def main() -> None:
    print("=" * 72)
    print("CANONICAL KNEEDLE AUDIT — FUZZY RFM-FCA")
    print("=" * 72)

    tx, _ = load_and_audit_transactions()
    _, obs_rfm = aggregate_rfm_dunnhumby(tx)

    scored = dense_rank_scores(
        obs_rfm[["household_key", "R", "F", "M"]].rename(
            columns={"household_key": "CustomerID"}
        ),
        dims=DIMS,
    )
    fuzzy_mu, _, _ = compute_fuzzy_memberships(scored, dims=DIMS)

    # Use the existing candidate miner only to reproduce the exact candidate
    # concept set. Its old Kneedle-inspired flags are ignored here.
    candidates, _, _ = _legacy_miner(
        fuzzy_mu,
        scored,
        min_support=SUPPORT_CUTOFF,
        dims=DIMS,
    )
    candidates = candidates[candidates["intent_size"] > 0].reset_index(drop=True)

    print(f"Candidate concepts at support >= {SUPPORT_CUTOFF}: {len(candidates):,}")

    support_desc = np.sort(candidates["support"].to_numpy(dtype=float))[::-1]

    rows = []
    for S in S_GRID:
        result = canonical_support_knee(support_desc, S=S, online=False)
        result["metric"] = "support"
        result["n_candidates"] = len(candidates)
        rows.append(result)

        stability_result = audit_stability_distribution(
            candidates["stability_proxy"].to_numpy(dtype=float),
            S=S,
        )
        stability_result["n_candidates"] = len(candidates)
        rows.append(stability_result)

    audit = pd.DataFrame(rows)

    out_dir = Path("results") / "canonical_kneedle"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "kneedle_sensitivity.csv", index=False)

    # Record the S=1 canonical candidate set for downstream validation.
    base = audit[
        (audit["metric"] == "support")
        & np.isclose(audit["S"].astype(float), 1.0)
    ]
    if not base.empty and base.iloc[0]["status"] == "ok":
        threshold = float(base.iloc[0]["knee_support"])
        selected = candidates[candidates["support"] >= threshold].copy()
        selected.to_csv(
            out_dir / "support_knee_candidates_S1.csv",
            index=False,
        )
        print(
            f"Canonical Kneedle S=1 support threshold: {threshold:.6f}; "
            f"retained: {len(selected):,}/{len(candidates):,}"
        )
    else:
        print("Canonical Kneedle S=1 did not identify a support knee.")

    print("\nAudit written to:", out_dir / "kneedle_sensitivity.csv")
    print("\n" + audit.to_string(index=False))


if __name__ == "__main__":
    main()

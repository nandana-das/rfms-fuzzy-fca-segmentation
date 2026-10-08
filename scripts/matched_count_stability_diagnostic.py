"""Matched-concept-count diagnostic for the segment-stability experiment.

Question: is the lower temporal stability of fuzzy RFM-FCA (vs crisp RFM-FCA)
explained by fuzzy simply retaining more concepts?

Matching rule (single rule, recorded in docs/AUDIT_ERRATA.md §6.1 before any
matched result was computed):
    For every fit, K = number of concepts crisp RFM-FCA retains on the same
    customers. Fuzzy RFM-FCA is fitted unchanged; its retained concepts are
    ranked by the redundancy suppressor's own quality order (support desc,
    intent size asc, stability proxy desc, original row order) and the top K
    are kept (all, if fuzzy retains <= K). Crisp is untouched.

Everything else (origins, the 30 subsamples, origin pairs, behaviourally-stable
subset, measures) is identical to scripts/segment_stability.py, whose study
functions are reused unchanged. This is a diagnostic, not a method variant.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import segment_stability as ss  # noqa: E402

EXP_DIR = ROOT_DIR / "results" / "segment_stability" / "matched_count_diagnostic"
EXP_DIR.mkdir(parents=True, exist_ok=True)

MATCHED = "fuzzy_rfm_fca_matched"
_original_fit_model = ss.fit_model


def _fuzzy_ranked_keys(df: pd.DataFrame) -> tuple[dict, list[frozenset]]:
    """Fit fuzzy RFM-FCA exactly as segment_stability does, but keep the retained
    concepts' table so they can be ranked by the suppressor's quality order."""
    bl = ss.bl
    scored, _ = bl._quintile_scored(df)
    mu, centroids, _ = ss.compute_fuzzy_memberships(scored, dims=ss.DIMS)
    raw, _, _ = ss.mine_fuzzy_closed_concepts_with_thresholds(
        mu, scored, bl.L_THRESHOLDS, min_support=bl.MIN_SUPPORT, dims=ss.DIMS
    )
    raw = raw.reset_index(drop=True)
    mu_all = bl.compute_customer_concept_memberships(raw, mu)
    kept = ss.suppress_redundant_concepts(raw, mu_all, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True)
    kept = kept.reset_index(drop=False).rename(columns={"index": "_row"})
    ranked = kept.sort_values(
        ["support", "intent_size", "stability_proxy", "_row"],
        ascending=[False, True, False, True], kind="mergesort",
    )
    return centroids, ss._keys(ranked["intent"])


def fit_model(arm: str, df: pd.DataFrame) -> dict:
    if arm != MATCHED:
        return _original_fit_model(arm, df)
    k = len(_original_fit_model("crisp_rfm_fca", df)["keys"])
    centroids, keys = _fuzzy_ranked_keys(df)
    return {"arm": arm, "centroids": centroids, "keys": keys[:k]}


def main() -> None:
    t0 = time.time()
    ss.fit_model = fit_model  # study functions look this up at call time
    ss.ARMS = ["crisp_rfm_fca", "fuzzy_rfm_fca", MATCHED]
    ss.COMPARISONS = [(MATCHED, "crisp_rfm_fca"), ("fuzzy_rfm_fca", "crisp_rfm_fca"), (MATCHED, "fuzzy_rfm_fca")]

    dh_tx = pd.read_csv(ss.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = ss.bl.dunnhumby_cohorts(dh_tx) + [
        c for c in ss.bl.retail2_cohorts(ss.load_cleaned_transactions()) if c["pooled"]
    ]
    ss._pr("Study A - refit stability (matched count)")
    df_a = ss.study_a(cohorts)
    sum_a, comp_a = ss.summarise_a(df_a)
    ss._pr("Study B - re-segmentation over time (matched count)")
    sum_b, comp_b = ss.study_b(cohorts)

    # Pre-specified verdict on temporal / all customers / core_jaccard.
    verdict_rows = []
    for dataset in comp_b["dataset"].unique():
        sel = (comp_b["dataset"] == dataset) & (comp_b["subset"] == "all") & (comp_b["measure"] == "core_jaccard")
        full = comp_b[sel & (comp_b["comparison"] == "fuzzy_rfm_fca - crisp_rfm_fca")].iloc[0]
        matched = comp_b[sel & (comp_b["comparison"] == f"{MATCHED} - crisp_rfm_fca")].iloc[0]
        if matched["delta"] >= 0 or matched["ci_low"] <= 0 <= matched["ci_high"]:
            verdict = "explained by count"
        elif abs(matched["delta"]) <= 0.5 * abs(full["delta"]):
            verdict = "partly explained"
        else:
            verdict = "not explained"
        verdict_rows.append({
            "dataset": dataset, "full_gap": full["delta"], "matched_gap": matched["delta"],
            "matched_ci_low": matched["ci_low"], "matched_ci_high": matched["ci_high"], "verdict": verdict,
        })
    verdict_df = pd.DataFrame(verdict_rows)

    df_a.to_csv(EXP_DIR / "refit_per_subsample.csv", index=False)
    sum_a.to_csv(EXP_DIR / "refit_summary.csv", index=False)
    comp_a.to_csv(EXP_DIR / "refit_comparisons.csv", index=False)
    sum_b.to_csv(EXP_DIR / "temporal_summary.csv", index=False)
    comp_b.to_csv(EXP_DIR / "temporal_comparisons.csv", index=False)
    verdict_df.to_csv(EXP_DIR / "verdict.csv", index=False)
    (EXP_DIR / "run_parameters.json").write_text(json.dumps({
        "matching_rule": "K = crisp retained-concept count on the same customers; fuzzy truncated to its top-K "
                         "by (support desc, intent_size asc, stability_proxy desc, row order)",
        "arms": ss.ARMS, "comparisons": [f"{a} - {b}" for a, b in ss.COMPARISONS],
        "primary_measure": "core_jaccard", "verdict_target": "temporal / all customers / core_jaccard",
        "reuses": "scripts/segment_stability.py study_a / study_b unchanged",
    }, indent=2), encoding="utf-8")

    md = ss.bl._md
    lines = [
        "# Matched-Concept-Count Stability Diagnostic",
        "",
        "Rule and interpretation fixed in `docs/AUDIT_ERRATA.md` §6.1 before this was run. "
        "Fuzzy RFM-FCA is truncated, per fit, to the crisp concept count using the suppressor's own quality order.",
        "",
        "## Verdict (temporal re-segmentation, all customers, core Jaccard)",
        "",
        md(verdict_df),
        "",
        "## Refit stability",
        "",
        md(sum_a),
        "",
        md(comp_a),
        "",
        "## Temporal re-segmentation",
        "",
        md(sum_b.groupby(["dataset", "arm", "subset"], sort=False)[["n_customers", *ss.MEASURES]].mean().reset_index()),
        "",
        md(comp_b),
        "",
        "Caveat: truncation keeps fuzzy's highest-support concepts while crisp keeps all of its concepts, "
        "so this design can only favour the matched fuzzy arm.",
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min.",
        "",
    ]
    (EXP_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    ss._pr(verdict_df.to_string(index=False))
    ss._pr(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs: {EXP_DIR}")


if __name__ == "__main__":
    main()

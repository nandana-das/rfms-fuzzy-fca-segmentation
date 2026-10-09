"""Part A of docs/QUICK_WINS_PLAN.md: segment stability on CDNOW.

Reuses the v1 stability protocol (scripts/segment_stability.py) and the
matched-count arm (scripts/matched_count_stability_diagnostic.py) unchanged,
on the three CDNOW origins, with the planned CDNOW tie-merging extension
(scripts/cdnow_confirmation.py). Writes only to results/qw_cdnow_stability/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cdnow_confirmation as cc  # noqa: E402
import matched_count_stability_diagnostic as mcd  # noqa: E402
import segment_stability as ss  # noqa: E402

OUT = ROOT / "results" / "qw_cdnow_stability"
OUT.mkdir(parents=True, exist_ok=True)
MATCHED = mcd.MATCHED


def main() -> None:
    t0 = time.time()
    # Planned CDNOW extension: merged quintile cutpoints + n-level fuzzy memberships.
    cc.install_extension()
    ss.compute_fuzzy_memberships = cc.compute_fuzzy_memberships_n
    # Arms and comparisons fixed in the plan (Part A).
    ss.fit_model = mcd.fit_model
    ss.ARMS = ["crisp_rfm_fca", "fuzzy_rfm_fca", MATCHED]
    ss.COMPARISONS = [("fuzzy_rfm_fca", "crisp_rfm_fca"), (MATCHED, "crisp_rfm_fca"), (MATCHED, "fuzzy_rfm_fca")]

    cohorts = cc.cdnow_cohorts(cc.load_transactions())
    ss._pr("Study A - refit stability (CDNOW)")
    df_a = ss.study_a(cohorts)
    sum_a, comp_a = ss.summarise_a(df_a)
    ss._pr("Study B - quarterly re-segmentation (CDNOW)")
    sum_b, comp_b = ss.study_b(cohorts)

    sel = (comp_b.comparison == "fuzzy_rfm_fca - crisp_rfm_fca") & (comp_b.measure == "core_jaccard") & (comp_b.subset == "all")
    temporal = comp_b[sel].iloc[0]
    refit = comp_a[(comp_a.comparison == "fuzzy_rfm_fca - crisp_rfm_fca") & (comp_a.measure == "core_jaccard")].iloc[0]
    advantage = (refit.interval_low > 0) or (temporal.p_holm < 0.05 and temporal.delta > 0)
    verdict = ("H3: dataset-specific stability advantage on CDNOW (see rows)" if advantage
               else "H3 replicates as not supported on CDNOW")

    df_a.to_csv(OUT / "refit_per_subsample.csv", index=False)
    sum_a.to_csv(OUT / "refit_summary.csv", index=False)
    comp_a.to_csv(OUT / "refit_comparisons.csv", index=False)
    sum_b.to_csv(OUT / "temporal_summary.csv", index=False)
    comp_b.to_csv(OUT / "temporal_comparisons.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({
        "plan": "docs/QUICK_WINS_PLAN.md Part A", "origins": [str(o.date()) for o in cc.ORIGINS],
        "arms": ss.ARMS, "comparisons": [f"{a} - {b}" for a, b in ss.COMPARISONS],
        "n_subsamples": ss.N_SUBSAMPLES, "subsample_frac": ss.SUBSAMPLE_FRAC, "subsample_seed": ss.SUBSAMPLE_SEED,
        "boot_n": ss.BOOT_N, "boot_seed": ss.BOOT_SEED, "stable_rank_tol": ss.STABLE_RANK_TOL,
        "extension": "CDNOW tie/empty-band merging (docs/CDNOW_CONFIRMATION_PLAN.md §4, A1)", "verdict": verdict,
    }, indent=2), encoding="utf-8")
    md = ss.bl._md
    lines = [
        "# CDNOW segment stability (limitation follow-up, Part A)",
        "",
        "Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). v1 stability protocol on the three CDNOW origins.",
        "",
        f"**Decision-rule outcome: {verdict}.**",
        "",
        "## Refit stability (80% subsamples vs full fit)",
        "",
        md(sum_a),
        "",
        md(comp_a),
        "",
        "## Quarterly re-segmentation",
        "",
        md(sum_b),
        "",
        md(comp_b),
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    ss._pr(verdict)


if __name__ == "__main__":
    main()

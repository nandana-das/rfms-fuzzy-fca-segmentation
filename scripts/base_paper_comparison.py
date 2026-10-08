"""Base paper (crisp RFM-FCA) vs fuzzy RFM-FCA, from the baseline-ladder outputs.

Rungruang et al. (2024) report no predictive metrics, so their method is
re-implemented (crisp quintile RFM-FCA, support >= 0.04) and evaluated under
exactly the same protocol as the fuzzy method. Online Retail II is the base
paper's dataset (primary comparison); Dunnhumby is the reproduction.

Reads results/baseline_ladder_rolling_origin/ (run that script first) and writes
results/base_paper_comparison/. Nothing is refit here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from baseline_ladder_rolling_origin import _md  # noqa: E402

LADDER_DIR = ROOT_DIR / "results" / "baseline_ladder_rolling_origin"
EXP_DIR = ROOT_DIR / "results" / "base_paper_comparison"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BASE, OURS, STRONG = "crisp_rfm_fca", "fuzzy_rfm_fca", "spline_log_rfm"
METRICS = {"auc": "ROC AUC (repurchase)", "spend_r2": "Spend R²", "invoice_r2": "Invoice R²"}
DATASET_ROLE = {"Online Retail II": "base-paper dataset", "Dunnhumby": "reproduction"}


def main() -> None:
    m = pd.read_csv(LADDER_DIR / "per_origin_metrics.csv")
    comp = pd.read_csv(LADDER_DIR / "paired_comparisons.csv")

    per_origin = []
    for (dataset, origin), g in m.groupby(["dataset", "origin"], sort=False):
        row = {"dataset": dataset, "origin": origin}
        for met in METRICS:
            b = g.loc[g["arm"] == BASE, met].iloc[0]
            o = g.loc[g["arm"] == OURS, met].iloc[0]
            row.update({f"{met}_base": b, f"{met}_ours": o, f"{met}_delta": o - b})
        per_origin.append(row)
    per_origin = pd.DataFrame(per_origin)
    per_origin.to_csv(EXP_DIR / "per_origin.csv", index=False)

    headline = []
    for dataset in ["Online Retail II", "Dunnhumby"]:
        pooled = m[(m["dataset"] == dataset) & m["pooled"]]
        po = per_origin[per_origin["dataset"] == dataset]
        for met, label in METRICS.items():
            means = pooled.groupby("arm")[met].mean()
            c = comp[(comp["dataset"] == dataset) & (comp["scope"] == "POOLED")
                     & (comp["comparison"] == f"{OURS} - {BASE}") & (comp["metric"] == met)].iloc[0]
            headline.append({
                "dataset": f"{dataset} ({DATASET_ROLE[dataset]})", "metric": label,
                "base paper method": means[BASE], "fuzzy RFM-FCA (ours)": means[OURS],
                "gain (points)": 100 * (means[OURS] - means[BASE]),
                "relative gain %": 100 * (means[OURS] / means[BASE] - 1),
                "95% CI (points)": f"[{100 * c['ci_low']:.1f}, {100 * c['ci_high']:.1f}]",
                "p (Holm)": c["p_holm"],
                "origins won": f"{int((po[f'{met}_delta'] > 0).sum())}/{len(po)}",
                "strong non-FCA baseline": means[STRONG],
            })
    headline = pd.DataFrame(headline)
    headline.to_csv(EXP_DIR / "headline.csv", index=False)

    primary = per_origin[per_origin["origin"].str.contains("primary")]
    lines = [
        "# Base Paper vs Fuzzy RFM-FCA",
        "",
        "Rungruang et al. (2024) report no predictive metrics, so their method (crisp quintile RFM-FCA, "
        "support >= 0.04) was re-implemented and evaluated under the same protocol as ours: rolling origins "
        "with a 91-day holdout, stratified 5-fold CV, train-only fitting, paired customer bootstrap, and Holm "
        "correction. Source: `results/baseline_ladder_rolling_origin/`.",
        "",
        "## Headline (mean over rolling origins)",
        "",
        _md(headline),
        "",
        "'Origins won' counts every origin, including the Online Retail II primary protocol. "
        "'Strong non-FCA baseline' is a spline on log-RFM, shown for transparency: fuzzy RFM-FCA "
        "does not beat it, and is significantly worse on Online Retail II Invoice R² (see `docs/RESULTS_SUMMARY.md` §B).",
        "",
        "## Online Retail II, original protocol (2010-12-09 cutoff, 365-day holdout)",
        "",
        _md(primary.drop(columns=["dataset"])),
        "",
        "## Every origin",
        "",
        _md(per_origin),
        "",
    ]
    (EXP_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()

"""Frozen evidence tables for the final paper (aggregation only; nothing is refit).

Sources:
    results/baseline_ladder_rolling_origin/          predictive evidence (quintile scoring)
    results/segment_stability/                       structural stability
    results/segment_stability/matched_count_diagnostic/  matched-concept-count diagnostic
Output:
    results/final_evidence/EVIDENCE_TABLES.md and one CSV per table.

Verdict labels (applied mechanically):
    predictive / temporal (p-values available, Holm-adjusted within dataset):
        "favourable, every origin"  Holm p < 0.05, delta > 0, positive at every origin (pair)
        "favourable"                Holm p < 0.05, delta > 0
        "no significant difference" Holm p >= 0.05
        "unfavourable"              Holm p < 0.05, delta < 0
    refit stability (percentile interval over subsamples, no p-value):
        "favourable" / "unfavourable" if the 95% interval excludes 0, else "no clear difference"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from baseline_ladder_rolling_origin import _md  # noqa: E402

RES = ROOT_DIR / "results"
LADDER = RES / "baseline_ladder_rolling_origin"
STAB = RES / "segment_stability"
MATCH = STAB / "matched_count_diagnostic"
OUT = RES / "final_evidence"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Online Retail II", "Dunnhumby"]
METRIC_LABEL = {"auc": "ROC AUC", "spend_r2": "Spend R²", "invoice_r2": "Invoice R²"}
ARM_LABEL = {
    "raw_std": "Raw RFM (standardized)", "log_rfm": "Log RFM", "spline_log_rfm": "Spline on log RFM",
    "gbm_raw_rfm": "Gradient boosting on raw RFM (untuned)", "crisp_bands": "Crisp bands (no FCA)",
    "crisp_rfm_fca": "Crisp RFM-FCA (base-paper method)", "fuzzy_bands": "Fuzzy bands (no FCA)",
    "fuzzy_rfm_fca": "Fuzzy RFM-FCA (proposed)", "fuzzy_rfm_fca_denserank": "Fuzzy RFM-FCA, dense-rank (superseded)",
    "kuznetsov_fca": "Kuznetsov-FCA (excluded, ablation)", "fuzzy_rfm_fca_matched": "Fuzzy RFM-FCA, matched count (diagnostic)",
}


def _all_positive(frac: str) -> bool:
    a, b = str(frac).split("/")
    return a == b


def verdict_p(delta: float, p_holm: float, frac: str) -> str:
    if p_holm >= 0.05:
        return "no significant difference"
    if delta < 0:
        return "unfavourable"
    return "favourable, every origin" if _all_positive(frac) else "favourable"


def verdict_interval(lo: float, hi: float) -> str:
    if lo > 0:
        return "favourable"
    if hi < 0:
        return "unfavourable"
    return "no clear difference"


def predictive_table(comp: pd.DataFrame, arm: str, ref: str) -> pd.DataFrame:
    sub = comp[(comp["scope"] == "POOLED") & (comp["comparison"] == f"{arm} - {ref}")]
    rows = []
    for dataset in DATASETS:
        for met, label in METRIC_LABEL.items():
            r = sub[(sub["dataset"] == dataset) & (sub["metric"] == met)].iloc[0]
            rows.append({
                "dataset": dataset, "metric": label, "delta": r["delta"],
                "95% CI": f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", "p (Holm)": r["p_holm"],
                "origins favourable": r["origins_positive"],
                "verdict": verdict_p(r["delta"], r["p_holm"], r["origins_positive"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    metrics = pd.read_csv(LADDER / "per_origin_metrics.csv")
    comp = pd.read_csv(LADDER / "paired_comparisons.csv")
    pooled = metrics[metrics["pooled"]]

    # Means per arm (pooled rolling origins).
    means = (pooled.groupby(["dataset", "arm"], sort=False)[list(METRIC_LABEL)].mean().reset_index())
    means["arm"] = means["arm"].map(ARM_LABEL)
    means = means.rename(columns=METRIC_LABEL)

    # Online Retail II primary protocol (reported separately, not pooled).
    prim = metrics[metrics["origin"].str.contains("primary")]
    prim = prim[prim["arm"].isin(["crisp_rfm_fca", "fuzzy_rfm_fca", "spline_log_rfm", "log_rfm", "raw_std"])]
    prim = prim[["arm", *METRIC_LABEL]].assign(arm=lambda d: d["arm"].map(ARM_LABEL)).rename(columns=METRIC_LABEL)

    table_a = predictive_table(comp, "fuzzy_rfm_fca", "crisp_rfm_fca")
    table_b = pd.concat([
        predictive_table(comp, "fuzzy_rfm_fca", ref).assign(reference=ARM_LABEL[ref])
        for ref in ["raw_std", "log_rfm", "spline_log_rfm", "gbm_raw_rfm", "fuzzy_bands"]
    ])[["reference", "dataset", "metric", "delta", "95% CI", "p (Holm)", "origins favourable", "verdict"]]
    table_kuz_pred = predictive_table(comp, "kuznetsov_fca", "fuzzy_rfm_fca")

    # Structural stability.
    rows = []
    for folder, label in [(STAB, "main"), (MATCH, "matched-count diagnostic")]:
        if not (folder / "refit_comparisons.csv").exists():
            continue
        ra = pd.read_csv(folder / "refit_comparisons.csv")
        tb = pd.read_csv(folder / "temporal_comparisons.csv")
        if label != "main":  # diagnostic: only rows involving the matched arm (others duplicate the main run)
            ra = ra[ra["comparison"].str.contains("matched")]
            tb = tb[tb["comparison"].str.contains("matched")]
        for _, r in ra[ra["measure"] == "core_jaccard"].iterrows():
            rows.append({"source": label, "dataset": r["dataset"], "comparison": r["comparison"],
                         "study": "A. refit (80% subsamples)", "delta": r["delta"],
                         "95% interval": f"[{r['interval_low']:+.4f}, {r['interval_high']:+.4f}]",
                         "p (Holm)": float("nan"), "origins favourable": r["origins_positive"],
                         "verdict": verdict_interval(r["interval_low"], r["interval_high"])})
        for _, r in tb[tb["measure"] == "core_jaccard"].iterrows():
            study = "B. quarterly re-segmentation, " + ("all customers" if r["subset"] == "all" else "unchanged-behaviour subset")
            rows.append({"source": label, "dataset": r["dataset"], "comparison": r["comparison"], "study": study,
                         "delta": r["delta"], "95% interval": f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                         "p (Holm)": r["p_holm"], "origins favourable": r["pairs_positive"],
                         "verdict": verdict_p(r["delta"], r["p_holm"], r["pairs_positive"])})
    table_c = pd.DataFrame(rows)
    table_c["comparison"] = table_c["comparison"].map(
        lambda c: " - ".join(ARM_LABEL.get(part, part) for part in c.split(" - "))
    )

    stab_levels = pd.read_csv(STAB / "refit_summary.csv")
    verdict = pd.read_csv(MATCH / "verdict.csv") if (MATCH / "verdict.csv").exists() else pd.DataFrame()

    for name, df in [("means_by_arm", means), ("retail2_primary_protocol", prim), ("A_fuzzy_vs_crisp", table_a),
                     ("B_fuzzy_vs_non_fca_baselines", table_b), ("C_structural_stability", table_c),
                     ("kuznetsov_predictive_ablation", table_kuz_pred)]:
        df.to_csv(OUT / f"{name}.csv", index=False)

    lines = [
        "# Frozen Evidence Tables",
        "",
        "Generated by `scripts/final_evidence_tables.py` from the result folders listed in its docstring. "
        "Predictive results use tie-preserving quintile scoring; rolling origins with a 91-day holdout "
        "(Dunnhumby: 4 origins; Online Retail II: 5 origins), stratified 5-fold CV within each origin, "
        "paired customer bootstrap, Holm correction within each dataset. Predictive and structural results "
        "are reported separately and are not combined into one performance claim.",
        "",
        "## Mean predictive metrics by representation (pooled rolling origins)",
        "",
        _md(means),
        "",
        "Online Retail II, original protocol (2010-12-09 cutoff, 365-day holdout; not pooled):",
        "",
        _md(prim),
        "",
        "## A. Fuzzy RFM-FCA vs crisp RFM-FCA (base-paper method)",
        "",
        _md(table_a),
        "",
        "## B. Fuzzy RFM-FCA vs non-FCA RFM baselines",
        "",
        "Fuzzy RFM-FCA does **not** establish superiority over the spline-on-log-RFM baseline.",
        "",
        _md(table_b),
        "",
        "## C. Structural stability (primary measure: core-profile Jaccard)",
        "",
        _md(table_c),
        "",
        "Absolute refit-stability levels (core Jaccard, mean over origins and subsamples):",
        "",
        _md(stab_levels[["dataset", "arm", "core_jaccard", "concept_recurrence", "n_concepts_ref"]]),
        "",
    ]
    if not verdict.empty:
        lines += ["Matched-count diagnostic verdict (rule fixed in `docs/AUDIT_ERRATA.md` §6.1):", "", _md(verdict), ""]
    lines += [
        "## Kuznetsov-FCA vs fuzzy RFM-FCA (predictive; excluded component)",
        "",
        _md(table_kuz_pred),
        "",
    ]
    (OUT / "EVIDENCE_TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

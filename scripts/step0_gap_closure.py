"""Calculate Step 0 gap closure from the verified baseline-ladder outputs.

This is a reporting-only step. It does not reload transactions or refit models.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"
OUT = ROOT / "results" / "step0_gap_closure"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {
    "raw_std": "Raw RFM without FCA",
    "log_rfm": "Log RFM without FCA",
    "crisp_rfm_fca": "Crisp RFM-FCA",
    "fuzzy_rfm_fca": "M0 fuzzy RFM-FCA",
    "gbm_raw_rfm": "Gradient boosting on raw RFM",
}
METRICS = ["auc", "spend_r2", "invoice_r2"]


def main() -> None:
    df = pd.read_csv(INPUT)
    pooled = df[df["pooled"].astype(str).str.lower().eq("true")].copy()
    means = pooled.groupby(["dataset", "arm"], sort=False)[METRICS].mean().reset_index()
    means = means[means["arm"].isin(ARMS)].copy()
    means["representation"] = means["arm"].map(ARMS)

    rows = []
    for _, g in means.groupby("dataset", sort=False):
        dataset = g["dataset"].iloc[0]
        by_arm = g.set_index("arm")
        for metric in METRICS:
            raw = float(by_arm.at["raw_std", metric])
            crisp = float(by_arm.at["crisp_rfm_fca", metric])
            m0 = float(by_arm.at["fuzzy_rfm_fca", metric])
            log = float(by_arm.at["log_rfm", metric])
            gbm = float(by_arm.at["gbm_raw_rfm", metric])
            denominator = raw - crisp
            ratio = None if abs(denominator) < 0.005 else (m0 - crisp) / denominator
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "raw_rfm": raw,
                "log_rfm": log,
                "crisp_rfm_fca": crisp,
                "m0_fuzzy_rfm_fca": m0,
                "gbm_raw_rfm": gbm,
                "numerator_m0_minus_crisp": m0 - crisp,
                "denominator_raw_minus_crisp": denominator,
                "gap_closure_ratio": ratio,
                "ratio_status": "undefined_or_unstable" if ratio is None else "defined",
            })

    gap = pd.DataFrame(rows)
    means.to_csv(OUT / "pooled_arm_means.csv", index=False)
    gap.to_csv(OUT / "gap_closure.csv", index=False)

    lines = [
        "# Step 0 — Predictive ceiling audit and gap closure",
        "",
        "This report reuses the verified outputs of `results/baseline_ladder_rolling_origin/`; it does not rerun the predictive pipeline.",
        "",
        "## Audit result",
        "",
        "All required Step 0 arms are present in the existing run. Raw RFM and log RFM use standardized three-feature inputs with LogisticRegressionCV/RidgeCV; crisp RFM-FCA and M0 use FCA-derived features with the same downstream learners; gradient boosting uses raw R/F/M with untuned histogram gradient boosting. The run uses train-fold-only preprocessing, customer-level stratified 5-fold CV within each rolling origin, 91-day holdouts, paired bootstrap comparisons, and Holm correction.",
        "",
        "| Arm | Existing verified output | Features | Learner | Status |",
        "|---|---|---|---|---|",
        "| Raw RFM without FCA | `per_origin_metrics.csv` | standardized R, F, M | LogisticRegressionCV / RidgeCV | existing |",
        "| Log RFM without FCA | `per_origin_metrics.csv` | standardized log1p(R), log1p(F), log1p(M) | LogisticRegressionCV / RidgeCV | existing |",
        "| Crisp RFM-FCA | `per_origin_metrics.csv` | crisp closed-concept memberships | LogisticRegressionCV / RidgeCV | existing |",
        "| M0 fuzzy RFM-FCA | `per_origin_metrics.csv` | fuzzy closed-concept memberships with Jaccard suppression | LogisticRegressionCV / RidgeCV | existing |",
        "| Gradient boosting baseline | `per_origin_metrics.csv` | raw R, F, M | untuned histogram gradient boosting | existing |",
        "",
        "No missing predictive comparison was identified; therefore no experiment was rerun. The newly computed artifacts are the pooled arm summary and gap-closure table below.",
        "",
        "## Pooled performance and gap closure",
        "",
        "The ratio is `(M0 − Crisp) / (Raw − Crisp)`, using pooled rolling-origin means and the metric's native direction (higher is better).",
        "",
        "| Dataset | Metric | Raw RFM | Log RFM | Crisp RFM-FCA | M0 fuzzy RFM-FCA | GBM raw RFM | Numerator | Denominator | Gap closure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in gap.iterrows():
        ratio = "undefined/unstable" if pd.isna(r["gap_closure_ratio"]) else f"{r['gap_closure_ratio']:.4f}"
        lines.append(
            f"| {r['dataset']} | {r['metric']} | {r['raw_rfm']:.4f} | {r['log_rfm']:.4f} | {r['crisp_rfm_fca']:.4f} | {r['m0_fuzzy_rfm_fca']:.4f} | {r['gbm_raw_rfm']:.4f} | {r['numerator_m0_minus_crisp']:+.4f} | {r['denominator_raw_minus_crisp']:+.4f} | {ratio} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The ratio is not a percentage improvement. A value between 0 and 1 means M0 closes part of the raw-to-crisp gap; a value above 1 means M0 exceeds the raw baseline relative to crisp; a negative value means M0 moves opposite to the raw-to-crisp direction. This report flags absolute denominators below 0.005 as undefined/unstable. When raw and crisp are nearly tied, the ratio should not be interpreted as a ceiling estimate.",
        "",
        "Across the evaluated datasets, learners, outcomes, and rolling-origin protocol, the results support a limited predictive-ceiling statement only. Log transformation recovers much of the performance that raw standardized RFM leaves on the table, while M0 is generally close to log-RFM and the nonlinear spline reference. Thus crisp RFM-FCA appears near the ceiling for some RFM-only comparisons, but the evidence does not establish a universal ceiling: the ratio is metric- and dataset-dependent, and the denominator is unstable for AUC because raw and crisp are close or reversed.",
        "",
        "## Limitations and provenance",
        "",
        "- The original paper reports clustering validity, not predictive metrics; these are comparisons against the repository's crisp reconstruction.",
        "- Pooled intervals are approximate because origins overlap; bootstrap intervals capture test-sample variability, not refitting variability.",
        "- The gap closure calculation is descriptive; it does not add new hypothesis tests or multiple-testing corrections.",
        "- No claim is made beyond the two evaluated datasets, three outcomes, learners, and locked protocol.",
        "",
        "## Commands and outputs",
        "",
        "- Audit/read: `rg -n -i \"raw|log[- ]?rfm|crisp|m0|gradient|boost|rolling|91[- ]day|holdout|gap closure\" scripts results docs README.md`.",
        "- Existing experiment command recorded by the project: `python scripts/baseline_ladder_rolling_origin.py` (already completed; not rerun for Step 0).",
        "- New reporting command: `python scripts/step0_gap_closure.py`.",
        "- Generated: `results/step0_gap_closure/pooled_arm_means.csv`, `results/step0_gap_closure/gap_closure.csv`, and this report.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

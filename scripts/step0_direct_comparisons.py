"""Step 0 direct M0 comparisons: a standalone re-analysis alongside the original gap-closure report.

Reporting only. Reads the frozen baseline ladder outputs
(results/baseline_ladder_rolling_origin/), verifies them, and reports M0 versus
crisp RFM-FCA, log-RFM, GBM, and (as the strongest additive reference) spline-log-RFM
with the ladder's existing paired-bootstrap intervals and Holm-adjusted p-values.
No model is refit. results/step0_gap_closure/ is left unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402

SRC = ROOT / "results" / "baseline_ladder_rolling_origin"
OUT = ROOT / "results" / "step0_direct_comparisons"
METRICS = ["auc", "spend_r2", "invoice_r2"]
REFS = {
    "crisp_rfm_fca": "Crisp RFM-FCA",
    "log_rfm": "Log-RFM",
    "gbm_raw_rfm": "GBM (raw RFM, untuned)",
    "spline_log_rfm": "Spline log-RFM (context)",
}
ALPHA = 0.05


def verify(metrics: pd.DataFrame, pooled: pd.DataFrame) -> dict[str, object]:
    bad_delta = 0
    for _, q in pooled.iterrows():
        a, b = q.comparison.split(" - ")
        sub = metrics[(metrics.dataset == q.dataset) & metrics.pooled]
        d = [sub[(sub.origin == o) & (sub.arm == a)][q.metric].iloc[0]
             - sub[(sub.origin == o) & (sub.arm == b)][q.metric].iloc[0] for o in dict.fromkeys(sub.origin)]
        bad_delta += not np.isclose(np.mean(d), q.delta)
    bad_holm = sum(
        not np.allclose(bl._holm(pooled[pooled.dataset == d].p_boot.tolist()), pooled[pooled.dataset == d].p_holm)
        for d in pooled.dataset.unique()
    )
    family = pooled.groupby("dataset").size().to_dict()
    if bad_delta or bad_holm:
        raise RuntimeError(f"Source verification failed: {bad_delta} delta, {bad_holm} Holm mismatches")
    return {"delta_mismatches": bad_delta, "holm_mismatches": bad_holm, "family": family}


def _verdict(r) -> str:
    if r.p_holm >= ALPHA:
        return "not significant"
    return "M0 better" if r.delta_m0_minus_ref > 0 else "M0 worse"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SRC / "per_origin_metrics.csv")
    comp = pd.read_csv(SRC / "paired_comparisons.csv")
    pooled = comp[comp.scope == "POOLED"]
    checks = verify(metrics, pooled)
    means = metrics[metrics.pooled].groupby(["dataset", "arm"], sort=False)[METRICS].mean()

    rows = []
    for dataset in pooled.dataset.unique():
        for ref, label in REFS.items():
            for metric in METRICS:
                q = pooled[(pooled.dataset == dataset) & (pooled.comparison == f"fuzzy_rfm_fca - {ref}")
                           & (pooled.metric == metric)].iloc[0]
                rows.append({
                    "dataset": dataset, "reference": ref, "reference_label": label, "metric": metric,
                    "m0": means.loc[(dataset, "fuzzy_rfm_fca"), metric], "reference_value": means.loc[(dataset, ref), metric],
                    "delta_m0_minus_ref": q.delta, "ci_low": q.ci_low, "ci_high": q.ci_high,
                    "p_holm": q.p_holm, "origins_m0_ahead": q.origins_positive,
                })
    out = pd.DataFrame(rows)
    out["verdict"] = out.apply(_verdict, axis=1)
    out.to_csv(OUT / "direct_comparisons.csv", index=False)

    p_floor = min(1.0, max(family for family in checks["family"].values()) / bl.BOOT_N)
    lines = [
        "# Step 0 — direct M0 comparisons (standalone re-analysis)", "",
        "Standalone re-analysis of the Step 0 comparisons. It does not replace the original Step 0 report: "
        "`results/step0_gap_closure/` is preserved unchanged as the historical record. This analysis does not "
        "use the gap-closure ratio `(M0 − Crisp) / (Raw − Crisp)` as a ceiling measure, because its denominator uses standardized "
        "raw RFM, a known misspecified baseline that scores below crisp RFM-FCA in 4 of 6 cells, so the "
        "denominator is small or of the wrong sign and the ratio is unstable or misleading. "
        "This report shows direct paired differences instead.", "",
        "## Source and verification", "",
        "All numbers come from the frozen baseline ladder (`results/baseline_ladder_rolling_origin/`, "
        "`scripts/baseline_ladder_rolling_origin.py`). Within each origin every arm used the same customer "
        f"folds (StratifiedKFold, seed {bl.CV_SEED}), learner seed, outcomes, 91-day holdout, and bootstrap "
        "indices, so differences are paired. No model was refit for this report.", "",
        f"- Pooled deltas recomputed from `per_origin_metrics.csv`: {checks['delta_mismatches']} mismatches.",
        f"- Holm adjustment recomputed over the ladder's family ({checks['family']} pooled tests): "
        f"{checks['holm_mismatches']} mismatches.", "",
        "## Direct comparisons (pooled 91-day origins)", "",
        f"Delta = M0 minus reference. 95% paired-bootstrap interval (B={bl.BOOT_N}); Holm p from the "
        f"ladder's 27-test family per dataset; verdict at Holm p < {ALPHA}.", "",
        "| dataset | reference | metric | M0 | reference | delta | CI low | CI high | p Holm | origins M0 ahead | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in out.iterrows():
        lines.append(f"| {r.dataset} | {r.reference_label} | {r.metric} | {r.m0:.4f} | {r.reference_value:.4f} | "
                     f"{r.delta_m0_minus_ref:+.4f} | {r.ci_low:+.4f} | {r.ci_high:+.4f} | {r.p_holm:.4f} | "
                     f"{r.origins_m0_ahead} | {r.verdict} |")
    lines += ["", "## Summary by reference", ""]
    for dataset in out.dataset.unique():
        for ref, label in REFS.items():
            s = out[(out.dataset == dataset) & (out.reference == ref)]
            parts = [f"{r.metric} {r.verdict}" for _, r in s.iterrows()]
            lines.append(f"- {dataset}, M0 vs {label}: " + "; ".join(parts) + ".")
    lines += [
        "", "## Points to keep visible", "",
        "- Pooled point estimates alone do not establish superiority; only the Holm verdicts above do. "
        "Where M0's point estimate exceeds a reference but the verdict is 'not significant', no advantage is claimed.",
        "- Rows where M0 is significantly worse than a reference are reported, not omitted (for example GBM and "
        "spline on Online Retail II Invoice R²).",
        "- The GBM is untuned (fixed parameters) on raw R/F/M; M0 > GBM therefore shows M0 beats this GBM, not tuned boosting.",
        "", "## Limitations", "",
        "- Pooled intervals average independent per-origin bootstraps. Origins overlap heavily in customers "
        "(Dunnhumby: 2,497 of 2,499 households appear in all four origins), so pooled intervals and p-values "
        "are approximate and likely too narrow. The ladder did not retain customer-level predictions, so a "
        "household-cluster bootstrap cannot be computed without refitting.",
        f"- The smallest attainable Holm p is {p_floor:.4f} (27 tests × 1/{bl.BOOT_N}); many cells sit at this floor, "
        "so their p-values only bound significance from above.",
        "- Single CV seed; bootstrap intervals reflect test-sample variability, not refitting or fold-assignment variability.",
        "- Two retail datasets, three outcomes, linear learners for all non-GBM arms.",
        "", "Command: `python scripts/step0_direct_comparisons.py`. Outputs: `direct_comparisons.csv`, this report.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

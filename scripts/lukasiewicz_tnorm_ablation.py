"""Narrow M0/product/Łukasiewicz aggregation comparison.

Concepts are the existing threshold-mined M0 concepts. This script does not
implement continuous concept discovery and writes only to its own result folder.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ablation_ladder_rolling_origin as ab  # noqa: E402
import baseline_ladder_rolling_origin as bl  # noqa: E402
from fair_comparison_retail2 import load_cleaned_transactions  # noqa: E402
from fuzzy_membership_sensitivity import TRANSACTION_FILE  # noqa: E402

OUT = ROOT / "results" / "lukasiewicz_tnorm_ablation"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["M0", "M2", "M4"]
METRICS = ["auc", "spend_r2", "invoice_r2"]


def main():
    start = time.time()
    dh = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2 = load_cleaned_transactions()
    cohorts = bl.dunnhumby_cohorts(dh) + bl.retail2_cohorts(r2)
    rng = np.random.default_rng(bl.BOOT_SEED)
    metric_rows, info_rows, boots = [], [], {}
    for c in cohorts:
        print(f"[{c['dataset']} | {c['origin']}]", flush=True)
        oof, info = ab.run_origin(c, arms=ARMS, cv_seed=bl.CV_SEED)
        info_rows.extend(info)
        full = np.arange(len(c["df"]))[None, :]
        idx = rng.integers(0, len(c["df"]), size=(bl.BOOT_N, len(c["df"])))
        for arm in ARMS:
            point = {k: float(v[0]) for k, v in ab.metric_rows(c["df"], oof[arm], full).items()}
            metric_rows.append({"dataset": c["dataset"], "origin": c["origin"], "pooled": c["pooled"], "n": len(c["df"]), "arm": arm, **point})
            boots[(c["dataset"], c["origin"], arm)] = ab.metric_rows(c["df"], oof[arm], idx)
    metrics = pd.DataFrame(metric_rows)
    pairs = [("M4", "M0"), ("M2", "M0"), ("M4", "M2")]
    rows = []
    for dataset in metrics.dataset.unique():
        sub = metrics[metrics.dataset == dataset]
        origins = list(dict.fromkeys(sub.origin))
        pooled = list(dict.fromkeys(sub.loc[sub.pooled, "origin"]))
        for arm, ref in pairs:
            for metric in METRICS:
                deltas, draws = [], []
                for origin in origins:
                    a = sub[(sub.origin == origin) & (sub.arm == arm)][metric].iloc[0]
                    b = sub[(sub.origin == origin) & (sub.arm == ref)][metric].iloc[0]
                    d = boots[(dataset, origin, arm)][metric] - boots[(dataset, origin, ref)][metric]
                    deltas.append(a - b)
                    rows.append({"dataset": dataset, "scope": origin, "comparison": f"{arm} - {ref}", "metric": metric, "delta": a - b, "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5), "p_boot": bl._boot_p(d)})
                    if origin in pooled:
                        draws.append(d)
                pooled_draws = np.mean(draws, axis=0)
                pdeltas = [deltas[origins.index(o)] for o in pooled]
                rows.append({"dataset": dataset, "scope": "POOLED", "comparison": f"{arm} - {ref}", "metric": metric, "delta": np.mean(pdeltas), "ci_low": np.percentile(pooled_draws, 2.5), "ci_high": np.percentile(pooled_draws, 97.5), "p_boot": bl._boot_p(pooled_draws), "origins_positive": f"{sum(x > 0 for x in pdeltas)}/{len(pdeltas)}"})
    comp = pd.DataFrame(rows)
    comp["p_holm"] = np.nan
    for dataset in comp.dataset.unique():
        mask = (comp.dataset == dataset) & (comp.scope == "POOLED")
        comp.loc[mask, "p_holm"] = bl._holm(comp.loc[mask, "p_boot"].tolist())
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(info_rows).to_csv(OUT / "concept_counts_per_fold.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({"arms": ARMS, "tnorms": {"M0": "min", "M2": "product", "M4": "lukasiewicz"}, "formula": "max(0, sum(mu_b) - (n_attributes - 1))", "concept_discovery": "existing threshold-scaled M0 concepts", "horizon_days": bl.HORIZON_DAYS, "n_folds": bl.N_FOLDS, "cv_seed": bl.CV_SEED, "boot_n": bl.BOOT_N, "boot_seed": bl.BOOT_SEED}, indent=2), encoding="utf-8")
    means = metrics[metrics["pooled"]].groupby(["dataset", "arm"])[METRICS].mean().reset_index()
    info_df = pd.DataFrame(info_rows)
    diag = info_df.groupby(["dataset", "arm"])[["n_candidates", "n_features", "n_distinct_columns", "n_empty_core"]].mean().reset_index()
    report = [
        "# Łukasiewicz t-norm ablation", "",
        "This narrow run reuses the existing M0 threshold-mined concepts and tests only aggregation. It is not continuous concept discovery.",
        "",
        "## Mathematical definitions and arms", "",
        "- M0/Gödel-min: `min(mu_b)` over the base memberships named by each existing concept.",
        "- M2/product: `prod(mu_b)` over the same existing concepts.",
        "- M4/Łukasiewicz: `max(0, sum(mu_b) - (n_attributes - 1))` over the same existing concepts.",
        "- Membership functions, 45-attribute threshold context, support 0.04, concept closure, Jaccard suppression, folds, learners, outcomes, and holdouts are unchanged.",
        "",
        "## Existing versus newly computed", "",
        "- M0 and product were already available in `results/fuzzy_tnorm_tail_ablation/`; this narrow run recomputed them only to retain paired customer-level prediction vectors.",
        "- M4/Łukasiewicz is the only missing arm and is newly computed here.",
        "",
        "## Pooled means", "",
        "| dataset | arm | auc | spend_r2 | invoice_r2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in means.iterrows():
        report.append(f"| {r.dataset} | {r.arm} | {r.auc:.4f} | {r.spend_r2:.4f} | {r.invoice_r2:.4f} |")
    report += ["", "## Per-origin metrics", "", "| dataset | origin | arm | auc | spend_r2 | invoice_r2 |", "|---|---|---|---:|---:|---:|"]
    for _, r in metrics.iterrows():
        report.append(f"| {r.dataset} | {r.origin} | {r.arm} | {r.auc:.4f} | {r.spend_r2:.4f} | {r.invoice_r2:.4f} |")
    report += ["", "## Concept diagnostics", "", "Means across folds and origins; candidate and retained counts are unchanged across t-norms because concepts are mined and suppressed once using M0/Gödel-min.", "", "| dataset | arm | candidates | retained features | distinct feature columns | empty cores |", "|---|---|---:|---:|---:|---:|"]
    for _, r in diag.iterrows():
        report.append(f"| {r.dataset} | {r.arm} | {r.n_candidates:.1f} | {r.n_features:.1f} | {r.n_distinct_columns:.1f} | {r.n_empty_core:.1f} |")
    report += ["", "## Pooled paired comparisons", "", "Delta = first arm minus second arm. Intervals are paired customer-bootstrap percentile intervals; Holm correction is within dataset across the nine comparison/metric tests.", "", "| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |", "|---|---|---|---:|---:|---:|---:|---|"]
    for _, r in comp[comp.scope == "POOLED"].iterrows():
        report.append(f"| {r.dataset} | {r.comparison} | {r.metric} | {r.delta:+.4f} | {r.ci_low:+.4f} | {r.ci_high:+.4f} | {r.p_holm:.4f} | {r.origins_positive} |")
    report += [
        "", "## Protocol and leakage checks", "",
        "All representations were built inside each training fold. No validation or holdout rows were used for membership fitting, concept mining, suppression, or downstream preprocessing. M0/product/Lukasiewicz used identical customer folds, seeds, outcomes, 91-day holdouts, and learner settings. The experiment uses existing threshold-mined concepts; it does not test continuous concept discovery.",
        "", "Outputs: `per_origin_metrics.csv`, `paired_comparisons.csv`, `concept_counts_per_fold.csv`, and `run_parameters.json`.",
        "Command: `python scripts\\lukasiewicz_tnorm_ablation.py`.",
        f"Runtime: {(time.time()-start)/60:.1f} minutes."]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

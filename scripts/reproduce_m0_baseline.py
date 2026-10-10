"""Reproduction and verification of frozen M0 (fuzzy_rfm_fca) baseline.

Stage 2 of the preregistered plan: before running any ablation models or full experiments,
verify that M0 and Crisp RFM-FCA reproduce their frozen per-origin metrics from
results/baseline_ladder_rolling_origin/per_origin_metrics.csv to numerical tolerance (<= 1e-12).

This script:
1. Reuses scripts/baseline_ladder_rolling_origin.py functions unchanged.
2. Evaluates the smoke cohorts:
   - Dunnhumby Day 347 (Origin 1)
   - Online Retail II 2010-09-10 (Origin 1)
3. Evaluates arms: 'fuzzy_rfm_fca' (M0) and 'crisp_rfm_fca'.
4. Compares against the frozen CSV and asserts max absolute difference <= 1e-12.
5. Writes isolated artifacts to results/fuzzy_tnorm_tail_ablation/m0_reproduction/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import baseline_ladder_rolling_origin as bl  # noqa: E402

OUTPUT_DIR = ROOT_DIR / "results" / "fuzzy_tnorm_tail_ablation" / "m0_reproduction"
FROZEN_METRICS_FILE = ROOT_DIR / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"

EVAL_ARMS = ["fuzzy_rfm_fca", "crisp_rfm_fca"]
METRICS = ["auc", "spend_r2", "invoice_r2"]
TOLERANCE = 1e-12


def _df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i, _ in enumerate(headers)) + " |"
    row_lines = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def run_reproduction_smoke() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70, flush=True)
    print("STAGE 2: M0 (fuzzy_rfm_fca) & Crisp RFM-FCA Baseline Reproduction Smoke Test", flush=True)
    print("=" * 70, flush=True)

    if not FROZEN_METRICS_FILE.exists():
        raise FileNotFoundError(f"Frozen metrics file not found: {FROZEN_METRICS_FILE}")

    frozen_df = pd.read_csv(FROZEN_METRICS_FILE)

    # Load transaction data for smoke origins
    print("\n[1/4] Loading transactions for smoke cohorts...", flush=True)
    dh_tx = pd.read_csv(bl.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2_tx = bl.load_cleaned_transactions()

    dh_cohorts = bl.dunnhumby_cohorts(dh_tx)
    r2_cohorts = bl.retail2_cohorts(r2_tx)

    # Smoke cohorts: First origin of Dunnhumby (day 347) and first origin of Online Retail II (2010-09-10)
    smoke_cohorts = [dh_cohorts[0], r2_cohorts[0]]

    # Temporarily restrict arms to the evaluated subset to run fast
    orig_arms = bl.ARMS
    bl.ARMS = EVAL_ARMS

    results_records = []
    comparison_records = []

    print("\n[2/4] Executing stratified 5-fold CV on smoke cohorts...", flush=True)
    for cohort in smoke_cohorts:
        ds_name = cohort["dataset"]
        origin_label = cohort["origin"]
        df = cohort["df"]
        n_cust = len(df)
        rep_rate = float(df["repurchased"].mean())
        print(f"\n---> Running cohort: [{ds_name} | {origin_label}] (n={n_cust:,}, repurchase={rep_rate:.4f})", flush=True)

        t_cohort = time.time()
        oof, info = bl.run_origin(cohort)
        full = np.arange(n_cust)[None, :]

        for arm in EVAL_ARMS:
            pt = {k: float(v[0]) for k, v in bl.metric_rows(df, oof[arm], full).items()}
            rec = {
                "dataset": ds_name,
                "origin": origin_label,
                "n": n_cust,
                "repurchase_rate": rep_rate,
                "arm": arm,
                **pt,
            }
            results_records.append(rec)

            # Compare against frozen reference
            match = frozen_df[
                (frozen_df["dataset"] == ds_name)
                & (frozen_df["origin"] == origin_label)
                & (frozen_df["arm"] == arm)
            ]
            if len(match) == 0:
                raise ValueError(f"No frozen match found for {ds_name} | {origin_label} | {arm}")

            frozen_row = match.iloc[0]
            for m in METRICS:
                comp_val = pt[m]
                froz_val = float(frozen_row[m])
                abs_diff = abs(comp_val - froz_val)
                passes = abs_diff <= TOLERANCE
                comparison_records.append({
                    "dataset": ds_name,
                    "origin": origin_label,
                    "arm": arm,
                    "metric": m,
                    "computed_value": comp_val,
                    "frozen_value": froz_val,
                    "abs_difference": abs_diff,
                    "tolerance": TOLERANCE,
                    "passed": passes,
                })

        print(f"     Finished in {time.time() - t_cohort:.1f}s", flush=True)

    # Restore original ARMS
    bl.ARMS = orig_arms

    results_df = pd.DataFrame(results_records)
    comp_df = pd.DataFrame(comparison_records)

    # Save to output directory
    print("\n[3/4] Writing verification artifacts to isolated output namespace...", flush=True)
    results_df.to_csv(OUTPUT_DIR / "smoke_reproduction_metrics.csv", index=False)
    comp_df.to_csv(OUTPUT_DIR / "metrics_verification.csv", index=False)

    all_passed = comp_df["passed"].all()
    max_diff = comp_df["abs_difference"].max()

    cohort_str = ", ".join(f"{c['dataset']} {c['origin']}" for c in smoke_cohorts)
    report_lines = [
        "# M0 (fuzzy_rfm_fca) & Crisp RFM-FCA Baseline Reproduction Report",
        "",
        f"- **Status:** {'ALL CHECKS PASSED' if all_passed else 'REPRODUCTION MISMATCH DETECTED'}",
        f"- **Timestamp:** {pd.Timestamp.now()}",
        f"- **Evaluated Cohorts:** {cohort_str}",
        f"- **Evaluated Arms:** {', '.join(EVAL_ARMS)}",
        f"- **Max Absolute Difference:** {max_diff:.2e} (Tolerance: {TOLERANCE:.0e})",
        "",
        "## Detailed Metric Verification Table",
        "",
        _df_to_markdown(comp_df),
        "",
        f"Total runtime: {time.time() - t0:.1f}s",
    ]
    report_text = "\n".join(report_lines)
    (OUTPUT_DIR / "REPRODUCTION_REPORT.md").write_text(report_text, encoding="utf-8")

    print("\n[4/4] Verification Summary:", flush=True)
    print("-" * 70, flush=True)
    for _, row in comp_df.iterrows():
        status = "PASSED" if row["passed"] else "FAILED"
        print(
            f"[{row['dataset'][:7]} | {row['origin']} | {row['arm']:<14} | {row['metric']:<10}] "
            f"Computed: {row['computed_value']:.8f} | Frozen: {row['frozen_value']:.8f} | "
            f"Diff: {row['abs_difference']:.2e} [{status}]",
            flush=True,
        )
    print("-" * 70, flush=True)
    print(f"Overall Reproduction Result: {'PASS' if all_passed else 'FAIL'}", flush=True)
    print(f"Max Absolute Error: {max_diff:.2e}", flush=True)
    print(f"Artifacts saved in: {OUTPUT_DIR}", flush=True)

    if not all_passed:
        failed_cases = comp_df[~comp_df["passed"]]
        raise AssertionError(f"M0 baseline reproduction failed for {len(failed_cases)} metrics!\n{failed_cases}")

    return {
        "all_passed": all_passed,
        "max_difference": max_diff,
        "results": results_df,
        "comparisons": comp_df,
    }


if __name__ == "__main__":
    run_reproduction_smoke()

"""Single-origin smoke test runner for Fuzzy RFM-FCA ablation arms M0, M1, M2, M3.

Stage 4 of the preregistered experiment:
1. Evaluates only the initial smoke-test origins:
   - Dunnhumby Day 347 (Origin 1)
   - Online Retail II 2010-09-10 (Origin 1)
2. Evaluates arms:
   - M0: Frozen baseline fuzzy RFM-FCA (reproduction verification)
   - M1: LAR-PW Frequency/Monetary memberships + Gödel-min aggregation
   - M2: Frozen M0 memberships + Algebraic Product t-norm aggregation
   - M3: LAR-PW memberships + Algebraic Product t-norm aggregation
3. Verifies that M0 reproduces frozen baseline metrics within numerical tolerance (<= 1e-12).
4. Verifies all 4 arms complete with finite metrics.
5. Records fold-level diagnostics, degenerate-fold checks, and tail uniqueness diagnostics.
6. Writes isolated outputs to results/fuzzy_tnorm_tail_ablation/smoke/.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import ablation_ladder_rolling_origin as abl  # noqa: E402
import baseline_ladder_rolling_origin as bl  # noqa: E402
import tail_memberships as tm  # noqa: E402

OUTPUT_DIR = ROOT_DIR / "results" / "fuzzy_tnorm_tail_ablation" / "smoke"
FROZEN_METRICS_FILE = ROOT_DIR / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"

EVAL_ARMS = ["M0", "M1", "M2", "M3"]
METRICS = ["auc", "spend_r2", "invoice_r2"]
TOLERANCE = 1e-12


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring external packages."""
    headers = list(df.columns)
    rows = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i, _ in enumerate(headers)) + " |"
    row_lines = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def run_ablation_smoke() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 75, flush=True)
    print("STAGE 4: Fuzzy RFM-FCA Ablation Runner & Single-Origin Smoke Test", flush=True)
    print("Arms: M0 (Baseline), M1 (LAR-PW Min), M2 (M0 Prod), M3 (LAR-PW Prod)", flush=True)
    print("=" * 75, flush=True)

    if not FROZEN_METRICS_FILE.exists():
        raise FileNotFoundError(f"Frozen baseline metrics not found at: {FROZEN_METRICS_FILE}")
    frozen_df = pd.read_csv(FROZEN_METRICS_FILE)

    # 1. Load transactions for smoke cohorts
    print("\n[1/5] Loading transactions for smoke cohorts...", flush=True)
    dh_tx = pd.read_csv(bl.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2_tx = bl.load_cleaned_transactions()

    dh_cohorts = bl.dunnhumby_cohorts(dh_tx)
    r2_cohorts = bl.retail2_cohorts(r2_tx)

    smoke_cohorts = [dh_cohorts[0], r2_cohorts[0]]

    metric_records: list[dict[str, object]] = []
    parity_records: list[dict[str, object]] = []
    all_info_records: list[dict[str, object]] = []
    tail_diag_records: list[dict[str, object]] = []

    # 2. Execute 5-fold CV on smoke cohorts
    print("\n[2/5] Running stratified 5-fold CV across M0, M1, M2, M3...", flush=True)
    for cohort in smoke_cohorts:
        ds_name = cohort["dataset"]
        origin_label = cohort["origin"]
        df = cohort["df"]
        n_cust = len(df)
        rep_rate = float(df["repurchased"].mean())
        print(f"\n---> Cohort: [{ds_name} | {origin_label}] (n={n_cust:,}, repurchase_rate={rep_rate:.4f})", flush=True)

        t_cohort = time.time()
        oof, info_rows = abl.run_origin(cohort, arms=EVAL_ARMS, cv_seed=abl.CV_SEED)
        all_info_records.extend(info_rows)
        full_idx = np.arange(n_cust)[None, :]

        for arm in EVAL_ARMS:
            point = {k: float(v[0]) for k, v in abl.metric_rows(df, oof[arm], full_idx).items()}
            # Check finiteness
            for m in METRICS:
                if not np.isfinite(point[m]):
                    raise ValueError(f"Metric {m} is not finite for {ds_name} | {origin_label} | {arm}: {point[m]}")

            rec = {
                "dataset": ds_name,
                "origin": origin_label,
                "n": n_cust,
                "repurchase_rate": rep_rate,
                "arm": arm,
                "membership_type": abl.ARM_CONFIGS[arm]["membership"],
                "tnorm": abl.ARM_CONFIGS[arm]["tnorm"],
                **point,
            }
            metric_records.append(rec)
            print(f"     [{arm}] AUC={point['auc']:.6f} | Spend R2={point['spend_r2']:.6f} | Inv R2={point['invoice_r2']:.6f}", flush=True)

            # Compare M0 against frozen baseline
            if arm == "M0":
                match = frozen_df[
                    (frozen_df["dataset"] == ds_name)
                    & (frozen_df["origin"] == origin_label)
                    & (frozen_df["arm"] == "fuzzy_rfm_fca")
                ]
                if len(match) == 0:
                    raise ValueError(f"No frozen match found for {ds_name} | {origin_label} | fuzzy_rfm_fca")
                frozen_row = match.iloc[0]

                for m in METRICS:
                    comp_val = point[m]
                    froz_val = float(frozen_row[m])
                    abs_diff = abs(comp_val - froz_val)
                    passes = abs_diff <= TOLERANCE
                    parity_records.append({
                        "dataset": ds_name,
                        "origin": origin_label,
                        "arm": "M0",
                        "metric": m,
                        "computed_value": comp_val,
                        "frozen_value": froz_val,
                        "abs_difference": abs_diff,
                        "tolerance": TOLERANCE,
                        "passed": bool(passes),
                    })

        print(f"     Cohort finished in {time.time() - t_cohort:.1f}s", flush=True)

        # 3. Descriptive tail discrimination diagnostics on the full origin
        print(f"     Computing tail discrimination diagnostics on full cohort...", flush=True)
        try:
            # Fit LAR-PW on full cohort strictly for descriptive diagnostic reporting
            mu_full_m1, _, b_full_m1, _, params_full_m1 = tm.fit_and_transform_tail_memberships(df)
            u_diag = tm.compute_continuous_tail_uniqueness(df, mu_full_m1, params_full_m1)

            # Identify tail mask
            f_tail_cut = np.expm1(params_full_m1.F.anchors[3])
            m_tail_cut = np.expm1(params_full_m1.M.anchors[3])
            tail_mask = (df["F"] >= f_tail_cut) | (df["M"] >= m_tail_cut)

            c_diag = tm.compute_context_profile_diversity(b_full_m1, tail_mask)

            tail_diag_records.append({
                "dataset": ds_name,
                "origin": origin_label,
                "n_total": n_cust,
                "n_tail": u_diag["n_tail"],
                "d_mem": u_diag["d_mem"],
                "tied_raw_pairs": u_diag["tied_raw_pairs"],
                "total_tail_pairs": u_diag["total_pairs"],
                "d_ctx": c_diag["d_ctx"],
                "h_ctx": c_diag["h_ctx"],
            })
        except Exception as exc:
            print(f"     [Warning] Tail diagnostic failed for {ds_name}: {exc}", flush=True)

    # 4. Construct DataFrames
    metrics_df = pd.DataFrame(metric_records)
    parity_df = pd.DataFrame(parity_records)
    info_df = pd.DataFrame(all_info_records)
    tail_diag_df = pd.DataFrame(tail_diag_records)

    # Assertions
    m0_all_passed = bool(parity_df["passed"].all())
    max_abs_diff = float(parity_df["abs_difference"].max())
    if not m0_all_passed:
        failed_rows = parity_df[~parity_df["passed"]]
        raise AssertionError(f"M0 baseline reproduction failed for some metrics:\n{failed_rows}")

    all_four_completed = len(metrics_df) == len(smoke_cohorts) * len(EVAL_ARMS)
    if not all_four_completed:
        raise AssertionError(f"Expected {len(smoke_cohorts) * len(EVAL_ARMS)} arm evaluations, got {len(metrics_df)}")

    # 5. Write CSV Artifacts
    print("\n[3/5] Writing CSV artifacts to results/fuzzy_tnorm_tail_ablation/smoke/...", flush=True)
    metrics_df.to_csv(OUTPUT_DIR / "ablation_smoke_metrics.csv", index=False)
    parity_df.to_csv(OUTPUT_DIR / "m0_parity_verification.csv", index=False)
    info_df.to_csv(OUTPUT_DIR / "fold_diagnostics.csv", index=False)
    tail_diag_df.to_csv(OUTPUT_DIR / "tail_diagnostics.csv", index=False)

    # 6. Generate Markdown Report
    print("\n[4/5] Generating SMOKE_TEST_REPORT.md...", flush=True)
    report_lines = [
        "# Fuzzy RFM-FCA Factorial Ablation: Single-Origin Smoke Test Report",
        "",
        f"- **Execution Timestamp:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- **Status:** ALL VERIFICATION CHECKS PASSED",
        f"- **Evaluated Cohorts:** Dunnhumby Day 347 (Origin 1), Online Retail II 2010-09-10 (Origin 1)",
        f"- **Evaluated Arms:** M0 (Baseline), M1 (LAR-PW Gödel-min), M2 (M0 Product t-norm), M3 (LAR-PW Product t-norm)",
        f"- **M0 Reproduction Max Absolute Error:** {max_abs_diff:.2e} (Tolerance: <= 1e-12)",
        "",
        "> [!IMPORTANT]",
        "> **Methodological Notice:** These results represent a single-origin technical smoke test executed strictly",
        "> to verify pipeline completion, training-only parameter isolation, and numerical reproduction of M0.",
        "> In accordance with the preregistered protocol, **smoke-test metrics must NOT be interpreted as evidence",
        "> of model superiority or confirmatory hypothesis resolution**.",
        "",
        "---",
        "",
        "## 1. M0 Parity Verification against Frozen Baseline",
        "",
        _df_to_markdown(parity_df),
        "",
        "---",
        "",
        "## 2. Factorial Ablation Smoke-Test Metrics",
        "",
        _df_to_markdown(metrics_df),
        "",
        "---",
        "",
        "## 3. Representation & Fold Diagnostics",
        "",
        "### Concept Counts and Structure across Arms (Fold Summary)",
        "",
    ]

    # Summarize concept counts per cohort and arm
    fold_summary = info_df.groupby(["dataset", "origin", "arm"]).agg(
        n_candidates_mean=("n_candidates", "mean"),
        n_features_mean=("n_features", "mean"),
        n_distinct_mean=("n_distinct_columns", "mean"),
        n_empty_core_mean=("n_empty_core", "mean"),
    ).reset_index()
    report_lines.append(_df_to_markdown(fold_summary))
    report_lines.append("")

    if not tail_diag_df.empty:
        report_lines.extend([
            "---",
            "",
            "## 4. Multi-Tier Tail Discrimination Diagnostics",
            "",
            "- **Level 1 ($D_{\\text{mem}}$):** Upper-tail continuous membership pairwise uniqueness.",
            "- **Level 2 ($D_{\\text{ctx}}$, $H_{\\text{ctx}}$):** Binarized formal context distinct profile count and entropy.",
            "",
            _df_to_markdown(tail_diag_df),
            "",
        ])

    degenerate_folds = info_df[info_df["degenerate_error"].notna()]
    report_lines.extend([
        "---",
        "",
        "## 5. Degenerate Fold Audit",
        "",
        f"- **Total Degenerate Training Folds:** {len(degenerate_folds)}",
    ])
    if len(degenerate_folds) == 0:
        report_lines.append("- **Audit Finding:** Zero degenerate folds detected. Valid 5-level partitions and positive tail scales were successfully constructed on all training folds.")
    else:
        report_lines.append(_df_to_markdown(degenerate_folds[["dataset", "origin", "fold", "arm", "degenerate_error"]]))

    report_lines.extend([
        "",
        "---",
        f"- **Total Smoke Test Wallclock Time:** {time.time() - t0:.1f}s",
    ])

    report_content = "\n".join(report_lines)
    (OUTPUT_DIR / "SMOKE_TEST_REPORT.md").write_text(report_content, encoding="utf-8")

    print("\n[5/5] Smoke test completed successfully!", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)

    return {
        "m0_passed": m0_all_passed,
        "max_abs_diff": max_abs_diff,
        "metrics": metric_records,
        "parity": parity_records,
        "total_runtime": time.time() - t0,
    }


if __name__ == "__main__":
    run_ablation_smoke()

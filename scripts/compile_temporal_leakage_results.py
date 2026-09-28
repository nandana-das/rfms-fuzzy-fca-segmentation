"""Compile temporal leakage audit results and generate report."""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "temporal_holdout"

audit_df = pd.read_csv(RESULTS_DIR / "satisfaction_leakage_audit.csv")
all_metrics = pd.read_csv(RESULTS_DIR / "temporal_holdout_comparison.csv")
all_cis = pd.read_csv(RESULTS_DIR / "temporal_holdout_comparison_ci.csv")

# Audit summary stats
n_total = len(audit_df)
n_aff_ans = audit_df["is_affected_answer_rule"].sum()
pct_aff_ans = n_aff_ans / n_total * 100

n_changed = audit_df["s_changed_ans"].sum()
pct_changed = n_changed / n_total * 100

changed_deltas = audit_df[audit_df["s_changed_ans"] == 1]["delta_s_ans"]
all_deltas = audit_df["delta_s_ans"]

rep_mask = audit_df["is_repurchaser"] == 1
n_rep_total = rep_mask.sum()
n_rep_changed = audit_df.loc[rep_mask, "s_changed_ans"].sum()

audit_stats = {
    "n_obs_customers": n_total,
    "n_affected_info": n_aff_ans,
    "pct_affected_info": pct_aff_ans,
    "n_changed_value": n_changed,
    "pct_changed_value": pct_changed,
    "n_repurchasers": n_rep_total,
    "n_repurchasers_changed": n_rep_changed,
    "pct_repurchasers_changed": n_rep_changed / n_rep_total * 100,
    "mean_abs_delta_changed": float(changed_deltas.mean()),
    "median_abs_delta_changed": float(changed_deltas.median()),
    "max_abs_delta": float(all_deltas.max()),
    "mean_abs_delta_all": float(all_deltas.mean()),
    "median_abs_delta_all": float(all_deltas.median()),
    "s_old_mean": float(audit_df["s_old"].mean()),
    "s_old_std": float(audit_df["s_old"].std()),
    "s_pre_ans_mean": float(audit_df["s_pre_ans"].mean()),
    "s_pre_ans_std": float(audit_df["s_pre_ans"].std()),
}

# 6. Compute Direct Model Deltas between Arm B (Leakage-Free) and Arm A (Original)
metrics_a = all_metrics[all_metrics["pipeline"].str.contains("Arm A")].reset_index(drop=True)
metrics_b = all_metrics[all_metrics["pipeline"].str.contains("Arm B")].reset_index(drop=True)

delta_rows = []
arm_labels_clean = ["Raw RFMS Baseline", "Crisp RFMS-FCA", "Fuzzy RFMS-FCA (Suppressed)", "FCM soft"]
for i in range(len(metrics_a)):
    row_a = metrics_a.iloc[i]
    row_b = metrics_b.iloc[i]
    arm_clean = arm_labels_clean[i]
    d_auc = row_b["test_repurchase_auc"] - row_a["test_repurchase_auc"]
    d_sp = row_b["test_spend_r2"] - row_a["test_spend_r2"]
    d_inv = row_b["test_invoices_r2"] - row_a["test_invoices_r2"]
    pct_auc = (d_auc / row_a["test_repurchase_auc"]) * 100
    pct_sp = (d_sp / abs(row_a["test_spend_r2"])) * 100 if abs(row_a["test_spend_r2"]) > 1e-6 else np.nan
    pct_inv = (d_inv / abs(row_a["test_invoices_r2"])) * 100 if abs(row_a["test_invoices_r2"]) > 1e-6 else np.nan
    
    delta_rows.append({
        "arm": arm_clean,
        "orig_auc": row_a["test_repurchase_auc"],
        "leakfree_auc": row_b["test_repurchase_auc"],
        "delta_auc": d_auc,
        "pct_delta_auc": pct_auc,
        "orig_spend_r2": row_a["test_spend_r2"],
        "leakfree_spend_r2": row_b["test_spend_r2"],
        "delta_spend_r2": d_sp,
        "orig_invoices_r2": row_a["test_invoices_r2"],
        "leakfree_invoices_r2": row_b["test_invoices_r2"],
        "delta_invoices_r2": d_inv,
    })
df_deltas = pd.DataFrame(delta_rows)
print(df_deltas.to_string(index=False))
df_deltas.to_csv(RESULTS_DIR / "model_performance_deltas.csv", index=False)

# 7. Generate Comprehensive Markdown Report
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from audit_olist_temporal_leakage import generate_report
generate_report(audit_stats, all_metrics, all_cis, df_deltas)
print("All artifacts generated successfully.")

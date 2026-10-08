"""Debug the four-way stat comparison fix using existing splits CSVs.

Validates that the module-level _compare_pair now produces non-empty rows
once _seed_order returns seed values rather than positional indices.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import four_way_method_comparison as m  # noqa: E402
import numpy as np
import pandas as pd

EXP_DIR = m.EXP_DIR
assert EXP_DIR.exists(), EXP_DIR

dh_splits = pd.read_csv(EXP_DIR / "four_way_dunnhumby_splits.csv")
r2_splits = pd.read_csv(EXP_DIR / "four_way_retail2_splits.csv")
print("dh_splits:", dh_splits.shape, "r2_splits:", r2_splits.shape)

comparisons = [
    ("FCM", "Raw RFM"),
    ("Fuzzy RFM-FCA", "Raw RFM"),
    ("Kuznetsov-FCA", "Raw RFM"),
    ("Fuzzy RFM-FCA", "FCM"),
    ("Kuznetsov-FCA", "FCM"),
    ("Kuznetsov-FCA", "Fuzzy RFM-FCA"),
]

stat_rows: list[dict] = []

for dataset_name, df in [("Dunnhumby Complete Journey", dh_splits),
                         ("Online Retail II", r2_splits)]:
    multi = df[df["split_seed"].isin(m.MULTI_SEEDS)].copy()
    if "arm" not in multi.columns:
        multi["arm"] = multi.apply(m._arm_for_row, axis=1)
    seed_order = m._seed_order(multi)
    print(f"\n=== {dataset_name} ===")
    print("seed_order:", seed_order.tolist())
    for m1, m2 in comparisons:
        if m1 == "FCM":
            sub1 = multi[multi["arm"] == "FCM k=6"]
            label1 = "FCM (k=6)"
        else:
            sub1 = multi[multi["method"] == m1]
            label1 = m1
        if m2 == "FCM":
            sub2 = multi[multi["arm"] == "FCM k=6"]
            label2 = "FCM (k=6)"
        else:
            sub2 = multi[multi["method"] == m2]
            label2 = m2
        print(f"\n-- {label1} vs {label2}: sub1={len(sub1)} sub2={len(sub2)}")
        if sub1.empty or sub2.empty:
            print("   SKIP (empty sub)")
            continue
        for metric in ["auc", "spend_r2", "invoice_r2"]:
            row = m._compare_pair(dataset_name, label1, label2, sub1, sub2, metric, seed_order)
            if row is None:
                print(f"   {metric}: None")
                continue
            stat_rows.append(row)
            print(f"   {metric}: mean_diff={row['mean_diff']:+.5f} ci=[{row['bootstrap_ci_lower']:+.5f},{row['bootstrap_ci_upper']:+.5f}] p={row['permutation_p_value']:.4f} sig={row['permutation_significant']}")

stat_df = pd.DataFrame(stat_rows)
print("\n\n=== RESULT ===")
print("stat_df rows:", len(stat_df))
if not stat_df.empty:
    print(stat_df[["dataset", "comparison", "metric", "mean_diff", "bootstrap_ci_lower", "bootstrap_ci_upper", "permutation_p_value", "permutation_significant"]].to_string(index=False))

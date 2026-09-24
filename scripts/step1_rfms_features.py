#!/usr/bin/env python3
"""Step 1: Prepare Olist data and engineer RFMS features.

Run from any directory:
    python scripts/step1_rfms_features.py

By default, reads ``data/olist_raw`` relative to the project root and writes
``outputs/step1``. Paths can be overridden with --data-dir and --output-dir.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}


def read_csv(data_dir: Path, name: str, usecols: list[str]) -> pd.DataFrame:
    path = data_dir / TABLES[name]
    missing = [column for column in usecols if column not in pd.read_csv(path, nrows=0).columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return pd.read_csv(path, usecols=usecols)


def grid_fit_frequency_weights(
    n_orders: np.ndarray, log_basket_sum: np.ndarray, repeat_flag: np.ndarray
) -> tuple[dict[str, float], float, int]:
    """Maximize Var(F*) / Var(n_orders) on the step-0.05 simplex grid."""
    denominator = float(np.var(n_orders, ddof=0))
    if denominator <= 0:
        raise ValueError("Var(n_orders) is zero; the requested variance ratio is undefined.")

    best: tuple[float, tuple[int, int, int]] | None = None
    evaluated = 0
    # Integer ticks avoid floating-point gaps on the simplex: 20 ticks = 1.0.
    for alpha_tick in range(21):
        for beta_tick in range(21 - alpha_tick):
            gamma_tick = 20 - alpha_tick - beta_tick
            weights = np.array([alpha_tick, beta_tick, gamma_tick], dtype=float) / 20.0
            f_star = (
                weights[0] * n_orders
                + weights[1] * log_basket_sum
                + weights[2] * repeat_flag
            )
            ratio = float(np.var(f_star, ddof=0) / denominator)
            evaluated += 1
            # Deterministic tie break: retain the first lexicographic grid point.
            key = (ratio, tuple(-int(value) for value in (alpha_tick, beta_tick, gamma_tick)))
            if best is None or key > (best[0], tuple(-value for value in best[1])):
                best = (ratio, (alpha_tick, beta_tick, gamma_tick))

    assert best is not None
    ticks = best[1]
    fitted = {name: tick / 20.0 for name, tick in zip(("alpha", "beta", "gamma"), ticks)}
    return fitted, best[0], evaluated


def dense_fractional_score(values: pd.Series, invert: bool = False) -> pd.Series:
    """Score values 1-5 by ceil(5 * dense_rank / maximum_dense_rank)."""
    ranks = values.rank(method="dense", ascending=True)
    max_rank = float(ranks.max())
    base = np.ceil(5.0 * ranks / max_rank).clip(1, 5).astype("int8")
    return (6 - base) if invert else base


def build_features(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    customers = read_csv(data_dir, "customers", ["customer_id", "customer_unique_id"])
    orders = read_csv(
        data_dir,
        "orders",
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    )
    # The public Olist order-items table has no quantity column. Each row is
    # one item record, so item-row count per order is the available basket-size
    # proxy (duplicate rows for the same product are separate units/records).
    items = read_csv(data_dir, "items", ["order_id", "order_item_id"])
    payments = read_csv(data_dir, "payments", ["order_id", "payment_value"])
    reviews = read_csv(data_dir, "reviews", ["order_id", "review_score"])

    diagnostics: dict[str, object] = {
        "input_rows": {
            "orders": int(len(orders)),
            "customers": int(len(customers)),
            "items": int(len(items)),
            "payments": int(len(payments)),
            "reviews": int(len(reviews)),
        }
    }

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    ref_date = orders["order_purchase_timestamp"].max()
    if pd.isna(ref_date):
        raise ValueError("No parseable order_purchase_timestamp values found.")
    diagnostics["reference_date"] = ref_date.isoformat()

    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    diagnostics["delivered_orders"] = int(len(delivered))
    diagnostics["delivered_missing_purchase_timestamp"] = int(
        delivered["order_purchase_timestamp"].isna().sum()
    )
    delivered = delivered.dropna(subset=["order_purchase_timestamp"])

    # orders.customer_id is order-scoped; customers.customer_unique_id supplies
    # the actual customer key used for every customer-level aggregation.
    customer_map = customers.drop_duplicates("customer_id").set_index("customer_id")[
        "customer_unique_id"
    ]
    delivered["customer_unique_id"] = delivered["customer_id"].map(customer_map)
    diagnostics["delivered_missing_customer_key"] = int(
        delivered["customer_unique_id"].isna().sum()
    )
    delivered = delivered.dropna(subset=["customer_unique_id"])

    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce")
    payment_null_count = int(payments["payment_value"].isna().sum())
    payment_zero_count = int(payments["payment_value"].eq(0).sum())
    payment_negative_count = int(payments["payment_value"].lt(0).sum())
    valid_payments = payments.loc[payments["payment_value"].notna() & payments["payment_value"].ne(0)]
    payment_by_order = valid_payments.groupby("order_id", sort=False)["payment_value"].sum()
    diagnostics["payment_rows_dropped_null"] = payment_null_count
    diagnostics["payment_rows_dropped_zero"] = payment_zero_count
    diagnostics["payment_rows_negative_retained"] = payment_negative_count

    delivered["payment_value"] = delivered["order_id"].map(payment_by_order)
    diagnostics["delivered_orders_without_valid_payment"] = int(delivered["payment_value"].isna().sum())
    diagnostics["delivered_orders_with_zero_net_payment"] = int(
        delivered["payment_value"].eq(0).sum()
    )
    paid = delivered.loc[delivered["payment_value"].notna() & delivered["payment_value"].ne(0)].copy()
    diagnostics["eligible_paid_delivered_orders"] = int(len(paid))

    quantity_by_order = items.groupby("order_id", sort=False)["order_item_id"].count()
    paid["order_quantity"] = paid["order_id"].map(quantity_by_order).fillna(0.0)
    diagnostics["eligible_orders_without_item_quantity"] = int(
        paid["order_id"].isin(quantity_by_order.index).eq(False).sum()
    )
    paid["log_basket_quantity"] = np.log1p(paid["order_quantity"].clip(lower=0))

    # Reviews are first reduced to one mean score per order so multiple review
    # rows do not multiply revenue, order counts, or basket quantities.
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    review_by_order = reviews.groupby("order_id", sort=False)["review_score"].mean()
    paid["order_review_score"] = paid["order_id"].map(review_by_order)

    grouped = paid.groupby("customer_unique_id", sort=False)
    features = grouped.agg(
        last_order_date=("order_purchase_timestamp", "max"),
        n_orders=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
        log_basket_sum=("log_basket_quantity", "sum"),
        satisfaction=("order_review_score", "mean"),
        orders_with_review=("order_review_score", "count"),
    )
    features["recency_days"] = (ref_date - features["last_order_date"]).dt.total_seconds() / 86400.0
    features["repeat_flag"] = features["n_orders"].gt(1).astype("int8")
    median_satisfaction = float(features["satisfaction"].median())
    if pd.isna(median_satisfaction):
        raise ValueError("No non-null review scores found; cannot median-impute satisfaction.")
    missing_satisfaction = int(features["satisfaction"].isna().sum())
    features["satisfaction"] = features["satisfaction"].fillna(median_satisfaction)

    weights, variance_ratio, grid_size = grid_fit_frequency_weights(
        features["n_orders"].to_numpy(dtype=float),
        features["log_basket_sum"].to_numpy(dtype=float),
        features["repeat_flag"].to_numpy(dtype=float),
    )
    features["f_star"] = (
        weights["alpha"] * features["n_orders"]
        + weights["beta"] * features["log_basket_sum"]
        + weights["gamma"] * features["repeat_flag"]
    )

    features["r_score"] = dense_fractional_score(features["recency_days"], invert=True)
    features["f_score"] = dense_fractional_score(features["f_star"])
    features["m_score"] = dense_fractional_score(features["monetary"])
    features["s_score"] = dense_fractional_score(features["satisfaction"])

    # Quantile-bin populations are diagnostic only: the implemented scoring is
    # dense-rank fractional scoring, not quantile binning.
    fstar_qcut = pd.qcut(features["f_star"], q=5, duplicates="drop")
    count_qcut = pd.qcut(features["n_orders"], q=5, duplicates="drop")
    features.index.name = "customer_unique_id"
    features = features.reset_index()

    diagnostics.update(
        {
            "customer_counts": {
                "after_customer_key_mapping_before_payment_filter": int(
                    delivered["customer_unique_id"].nunique()
                ),
                "after_delivered_and_payment_filters": int(features["customer_unique_id"].nunique()),
                "distinct_order_scoped_customer_id_values_after_filters": int(paid["customer_id"].nunique()),
            },
            "final_customer_count": int(len(features)),
            "median_satisfaction_imputation": median_satisfaction,
            "customers_with_imputed_satisfaction": missing_satisfaction,
            "weights": weights,
            "weight_fit": {
                "objective": "population Var(F*) / population Var(n_orders)",
                "variance_ratio": variance_ratio,
                "simplex_step": 0.05,
                "grid_points_evaluated": grid_size,
            },
            "weight_fit_diagnostic_quintile_bins": {
                "requested_bins": 5,
                "f_star_effective_bins_after_duplicate_edge_collapse": int(len(fstar_qcut.cat.categories)),
                "f_star_populations": [int(value) for value in fstar_qcut.value_counts(sort=False)],
                "n_orders_effective_bins_after_duplicate_edge_collapse": int(len(count_qcut.cat.categories)),
                "n_orders_populations": [int(value) for value in count_qcut.value_counts(sort=False)],
                "note": "Diagnostic only; weights are selected solely by the specified variance-ratio objective. Duplicate quantile edges are collapsed, exposing where ordinary quintile scoring fails.",
            },
            "dense_rank_diagnostics": {
                "distinct_values": {
                    "recency_days": int(features["recency_days"].nunique()),
                    "f_star": int(features["f_star"].nunique()),
                    "monetary": int(features["monetary"].nunique()),
                    "satisfaction": int(features["satisfaction"].nunique()),
                },
                "score_populations": {
                    column: {str(int(key)): int(value) for key, value in features[column].value_counts().sort_index().items()}
                    for column in ("r_score", "f_score", "m_score", "s_score")
                },
                "f_score_1_share": float(features["f_score"].eq(1).mean()),
            },
        }
    )

    return features, diagnostics


def write_diagnostics(path: Path, diagnostics: dict[str, object]) -> None:
    lines = ["# Step 1 diagnostics", "", f"Reference date: {diagnostics['reference_date']}", ""]
    lines.extend(["## Row and customer counts", ""])
    for key, value in diagnostics["input_rows"].items():
        lines.append(f"- Input {key}: {value:,}")
    for key in (
        "delivered_orders",
        "delivered_missing_purchase_timestamp",
        "delivered_missing_customer_key",
        "payment_rows_dropped_null",
        "payment_rows_dropped_zero",
        "payment_rows_negative_retained",
        "delivered_orders_without_valid_payment",
        "delivered_orders_with_zero_net_payment",
        "eligible_paid_delivered_orders",
        "eligible_orders_without_item_quantity",
    ):
        lines.append(f"- {key.replace('_', ' ')}: {diagnostics[key]:,}")
    lines.extend(["", "Customer counts:", ""])
    for key, value in diagnostics["customer_counts"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value:,}")
    lines.extend(
        [
            "",
            "## F* weight fit",
            "",
            f"- Weights (alpha, beta, gamma): {diagnostics['weights']}",
            f"- Variance ratio Var(F*) / Var(n_orders): {diagnostics['weight_fit']['variance_ratio']:.6f}",
            f"- Simplex points evaluated: {diagnostics['weight_fit']['grid_points_evaluated']}",
            f"- Diagnostic F* qcut bins (requested 5, effective {diagnostics['weight_fit_diagnostic_quintile_bins']['f_star_effective_bins_after_duplicate_edge_collapse']}): {diagnostics['weight_fit_diagnostic_quintile_bins']['f_star_populations']}",
            f"- Diagnostic order-count qcut bins (requested 5, effective {diagnostics['weight_fit_diagnostic_quintile_bins']['n_orders_effective_bins_after_duplicate_edge_collapse']}): {diagnostics['weight_fit_diagnostic_quintile_bins']['n_orders_populations']}",
            "- Quintile bins are diagnostic only; final scores use dense ranks. Duplicate quantile edges demonstrate collapse in ordinary qcut scoring.",
            "",
            "## RFMS scoring",
            "",
            f"- Final customers: {diagnostics['final_customer_count']:,}",
            f"- Median satisfaction used for imputation: {diagnostics['median_satisfaction_imputation']:.4f}",
            f"- Customers imputed: {diagnostics['customers_with_imputed_satisfaction']:,}",
            f"- Distinct raw values: {diagnostics['dense_rank_diagnostics']['distinct_values']}",
            f"- Score populations (1–5): {diagnostics['dense_rank_diagnostics']['score_populations']}",
            f"- F-score 1 share: {diagnostics['dense_rank_diagnostics']['f_score_1_share']:.2%}",
            "",
            "## Method notes",
            "",
            "- Recency reference date is the maximum parseable purchase timestamp across the orders table.",
            "- `orders.customer_id` is mapped to `customers.customer_unique_id`; customer-level rows are grouped only by `customer_unique_id`.",
            "- Payment records with null or zero `payment_value` are dropped before order-level payment aggregation.",
            "- F* weight fitting maximizes the specified population-variance ratio on the 0.05 simplex grid; quintile balance is reported as a diagnostic and does not alter the selected weights.",
            "- The Olist order-items table has no quantity column; basket quantity is proxied by counting item rows per order.",
            "- Satisfaction is averaged across available per-order review means per customer; customers with no available review are median-imputed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "olist_raw")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "step1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, diagnostics = build_features(args.data_dir)
    csv_path = args.output_dir / "rfms_features.csv"
    pickle_path = args.output_dir / "rfms_features.pkl"
    json_path = args.output_dir / "step1_diagnostics.json"
    md_path = args.output_dir / "step1_diagnostics.md"
    features.to_csv(csv_path, index=False)
    features.to_pickle(pickle_path)
    json_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_diagnostics(md_path, diagnostics)

    print(f"Wrote {csv_path}")
    print(f"Wrote {pickle_path}")
    print(f"Wrote {md_path}")
    print(f"Reference date: {diagnostics['reference_date']}")
    print(f"Eligible delivered orders: {diagnostics['eligible_paid_delivered_orders']:,}")
    print(f"Final customers: {diagnostics['final_customer_count']:,}")
    print(f"F* weights: {diagnostics['weights']}")
    print(f"Variance ratio: {diagnostics['weight_fit']['variance_ratio']:.6f}")
    print(f"Score populations: {diagnostics['dense_rank_diagnostics']['score_populations']}")
    print(f"F-score 1 share: {diagnostics['dense_rank_diagnostics']['f_score_1_share']:.2%}")


if __name__ == "__main__":
    main()

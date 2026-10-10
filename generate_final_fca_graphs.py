"""Generate publication-oriented FCA figures from authoritative per-origin CSVs.

No experiments or metrics are rerun here. The script validates the committed
per-origin sources, plots retained origins in chronological order, and records
source and derived difference provenance in graph_data_audit.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "figures" / "final_fca_comparison"
V1 = ROOT / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"
V2 = ROOT / "results" / "v2_hybrid" / "per_origin_metrics.csv"
CDNOW = ROOT / "results" / "cdnow_confirmation" / "per_origin_metrics.csv"
METRICS = {"auc": "ROC AUC", "spend_r2": "Spend R²", "invoice_r2": "Invoice R²"}
METHODS = ("crisp_rfm_fca", "fuzzy_rfm_fca", "hybrid_fuzzy_fca")
V1_METHODS = METHODS[:2]
LABELS = {"crisp_rfm_fca": "Crisp RFM-FCA", "fuzzy_rfm_fca": "Baseline Fuzzy RFM-FCA (M0)", "hybrid_fuzzy_fca": "Hybrid Fuzzy RFM-FCA"}
COLORS = {"crisp_rfm_fca": "#E69F00", "fuzzy_rfm_fca": "#009E73", "hybrid_fuzzy_fca": "#D55E00"}
STYLES = {"crisp_rfm_fca": {"marker": "o", "linestyle": "-"}, "fuzzy_rfm_fca": {"marker": "s", "linestyle": "--"}, "hybrid_fuzzy_fca": {"marker": "^", "linestyle": ":"}}
DATASETS = ("Dunnhumby", "Online Retail II")
ROLLING_ORIGINS = {"Dunnhumby": ("day 347", "day 438", "day 529", "day 620"), "Online Retail II": ("2010-09-10", "2010-12-10", "2011-03-11", "2011-06-10", "2011-09-09")}
CDNOW_ORIGINS = ("1997-09-30", "1997-12-31", "1998-03-31")
PAIR_LABELS = {"fuzzy_minus_crisp": "Baseline Fuzzy M0 − Crisp", "hybrid_minus_m0": "Hybrid − Baseline Fuzzy M0", "hybrid_minus_crisp": "Hybrid − Crisp"}


def read_source(path: Path) -> pd.DataFrame:
    required = {"dataset", "origin", "n", "auc", "spend_r2", "invoice_r2", "arm"}
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if path != CDNOW and "pooled" not in frame.columns:
        raise ValueError(f"{path}: pooled flag is required")
    for col in ("auc", "spend_r2", "invoice_r2"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError(f"{path}: non-finite values in {col}")
    return frame


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    keys = ["dataset", "origin", "arm"]
    duplicate = frame[frame.duplicated(keys, keep=False)]
    if not duplicate.empty:
        raise ValueError(f"{label}: duplicate observations: {duplicate[keys].to_dict('records')}")


def validate_v1(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.arm.isin(V1_METHODS)].copy()
    rolling, primary = selected[selected.pooled == True].copy(), selected[selected.pooled == False]  # noqa: E712
    if len(rolling) != 18 or len(primary) != 2 or set(selected.arm) != set(V1_METHODS):
        raise ValueError(f"v1 selection mismatch: pooled={len(rolling)}, primary={len(primary)}, methods={sorted(selected.arm.unique())}")
    for dataset, origins in ROLLING_ORIGINS.items():
        for origin in origins:
            rows = rolling[(rolling.dataset == dataset) & (rolling.origin == origin)]
            if len(rows) != 2 or set(rows.arm) != set(V1_METHODS):
                raise ValueError(f"v1 missing method/origin rows: {dataset} {origin}")
    assert_unique(rolling, "v1 pooled")
    return rolling


def validate_v2(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.arm.isin(METHODS)].copy()
    rolling, primary = selected[selected.pooled == True].copy(), selected[selected.pooled == False]  # noqa: E712
    if len(rolling) != 27 or len(primary) != 3 or set(selected.arm) != set(METHODS):
        raise ValueError(f"v2 selection mismatch: pooled={len(rolling)}, primary={len(primary)}, methods={sorted(selected.arm.unique())}")
    for dataset, origins in ROLLING_ORIGINS.items():
        for origin in origins:
            rows = rolling[(rolling.dataset == dataset) & (rolling.origin == origin)]
            if len(rows) != 3 or set(rows.arm) != set(METHODS):
                raise ValueError(f"v2 missing method/origin rows: {dataset} {origin}")
    if set(primary.origin) != {"2010-12-09 (primary, 365d)"}:
        raise ValueError("v2 primary holdout origin is not isolated")
    assert_unique(rolling, "v2 pooled")
    return rolling


def validate_cdnow(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.arm.isin(METHODS)].copy()
    if len(selected) != 9 or set(selected.arm) != set(METHODS) or set(selected.origin) != set(CDNOW_ORIGINS):
        raise ValueError("CDNOW must contain 3 methods × 3 expected origins")
    for origin in CDNOW_ORIGINS:
        rows = selected[selected.origin == origin]
        if len(rows) != 3 or set(rows.arm) != set(METHODS):
            raise ValueError(f"CDNOW missing method/origin rows: {origin}")
    assert_unique(selected, "CDNOW")
    return selected


def source_file(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def add_source_audit(audit: list[dict], frame: pd.DataFrame, path: Path, metric: str, study: str) -> None:
    for _, row in frame.iterrows():
        audit.append({"record_type": "source_value", "study": study, "source_file": source_file(path), "dataset": row.dataset, "origin": row.origin, "metric": metric, "method_id": row.arm, "method_label": LABELS[row.arm], "comparison_id": "", "comparison_label": "", "source_method_a": row.arm, "source_method_b": "", "source_value_a": float(row[metric]), "source_value_b": np.nan, "derived_value": float(row[metric]), "pooled": row.get("pooled", True)})


def add_difference_audit(audit: list[dict], study: str, metric: str, dataset: str, origin: str, comparison: str, method_a: str, method_b: str, value_a: float, value_b: float, path: Path) -> None:
    audit.append({"record_type": "derived_difference", "study": study, "source_file": source_file(path), "dataset": dataset, "origin": origin, "metric": metric, "method_id": "", "method_label": "", "comparison_id": comparison, "comparison_label": PAIR_LABELS[comparison], "source_method_a": method_a, "source_method_b": method_b, "source_value_a": value_a, "source_value_b": value_b, "derived_value": value_a - value_b, "pooled": True})


def style_axis(ax: plt.Axes, metric: str, title: str, difference: bool = False, difference_values: np.ndarray | None = None) -> None:
    ax.set_ylabel(f"Δ {METRICS[metric]}" if difference else METRICS[metric], fontsize=12)
    ax.set_xlabel("Chronological evaluation origin", fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    ymax = 1.0 if metric == "auc" else 0.7
    if difference:
        bound = max(0.02, float(np.nanmax(np.abs(difference_values))) * 1.35)
        ax.set_ylim(-bound, bound)
    else:
        ax.set_ylim(-0.02 if metric == "auc" else -0.035, ymax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_lines(frame: pd.DataFrame, metric: str, methods: tuple[str, ...], origins: tuple[str, ...], title: str, path: Path, audit: list[dict], study: str, source: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    subset = frame[frame.dataset == dataset]
    x = np.arange(len(origins))
    for method in methods:
        values = subset[subset.arm == method].set_index("origin").reindex(origins)[metric]
        ax.plot(x, values.to_numpy(), color=COLORS[method], linewidth=2.1, marker=STYLES[method]["marker"], linestyle=STYLES[method]["linestyle"], markersize=6, label=LABELS[method])
    ax.set_xticks(x, origins)
    plt.setp(ax.get_xticklabels(), rotation=35 if len(origins) > 4 else 0, ha="right")
    style_axis(ax, metric, title)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    add_source_audit(audit, subset[subset.arm.isin(methods)], source, metric, study)


def difference_frame(frame: pd.DataFrame, metric: str, dataset: str, comparisons: tuple[str, ...]) -> pd.DataFrame:
    subset = frame[frame.dataset == dataset].set_index(["origin", "arm"])
    pairs = {"fuzzy_minus_crisp": ("fuzzy_rfm_fca", "crisp_rfm_fca"), "hybrid_minus_m0": ("hybrid_fuzzy_fca", "fuzzy_rfm_fca"), "hybrid_minus_crisp": ("hybrid_fuzzy_fca", "crisp_rfm_fca")}
    rows = []
    for origin in subset.index.get_level_values("origin").unique():
        for comparison in comparisons:
            method_a, method_b = pairs[comparison]
            if (origin, method_a) not in subset.index or (origin, method_b) not in subset.index:
                raise ValueError(f"missing matched rows for {dataset}, {origin}, {comparison}")
            value_a, value_b = float(subset.loc[(origin, method_a), metric]), float(subset.loc[(origin, method_b), metric])
            rows.append({"origin": origin, "comparison": comparison, "method_a": method_a, "method_b": method_b, "value_a": value_a, "value_b": value_b, "difference": value_a - value_b})
    return pd.DataFrame(rows)


def plot_differences(frame: pd.DataFrame, metric: str, methods: tuple[str, ...], origins: tuple[str, ...], dataset: str, title: str, path: Path, audit: list[dict], study: str, source: Path) -> None:
    if study == "cdnow_confirmation":
        comparisons = ("hybrid_minus_m0",)
    elif methods == V1_METHODS:
        comparisons = ("fuzzy_minus_crisp",)
    else:
        comparisons = ("hybrid_minus_m0", "hybrid_minus_crisp", "fuzzy_minus_crisp")
    diffs = difference_frame(frame, metric, dataset, comparisons)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x = np.arange(len(origins))
    styles = {"fuzzy_minus_crisp": ("o", "-", "#0072B2"), "hybrid_minus_m0": ("^", "--", "#D55E00"), "hybrid_minus_crisp": ("s", ":", "#6A3D9A")}
    for comparison in comparisons:
        part = diffs[diffs.comparison == comparison].set_index("origin").reindex(origins)
        marker, linestyle, color = styles[comparison]
        ax.plot(x, part.difference.to_numpy(), marker=marker, linestyle=linestyle, color=color, linewidth=2.1, markersize=6, label=PAIR_LABELS[comparison])
        for row in part.itertuples():
            add_difference_audit(audit, study, metric, dataset, row.Index, comparison, row.method_a, row.method_b, row.value_a, row.value_b, source)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x, origins)
    plt.setp(ax.get_xticklabels(), rotation=35 if len(origins) > 4 else 0, ha="right")
    style_axis(ax, metric, title, difference=True, difference_values=diffs.difference.to_numpy())
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frames = {"v1": read_source(V1), "v2": read_source(V2), "cdnow": read_source(CDNOW)}
    v1, v2, cdnow = validate_v1(frames["v1"]), validate_v2(frames["v2"]), validate_cdnow(frames["cdnow"])
    OUT.mkdir(parents=True, exist_ok=True)
    audit: list[dict] = []
    print(f"Validated v1={len(v1)} pooled rows, v2={len(v2)} pooled rows, CDNOW={len(cdnow)} rows")
    for metric in METRICS:
        for dataset in DATASETS:
            slug = "dunnhumby" if dataset == "Dunnhumby" else "online_retail"
            origins = ROLLING_ORIGINS[dataset]
            plot_lines(v1, metric, V1_METHODS, origins, f"Frozen v1 primary: {dataset} · {METRICS[metric]}", OUT / f"v1_{slug}_{metric}_origins.png", audit, "v1_frozen_primary", V1, dataset)
            plot_lines(v2, metric, METHODS, origins, f"Exploratory/post hoc v2: {dataset} · {METRICS[metric]}", OUT / f"v2_{slug}_{metric}_origins.png", audit, "v2_exploratory_post_hoc", V2, dataset)
            plot_differences(v1, metric, V1_METHODS, origins, dataset, f"Frozen v1 primary differences: {dataset} · {METRICS[metric]}", OUT / f"v1_{slug}_{metric}_differences.png", audit, "v1_frozen_primary", V1)
            plot_differences(v2, metric, METHODS, origins, dataset, f"Exploratory/post hoc v2 differences: {dataset} · {METRICS[metric]}", OUT / f"v2_{slug}_{metric}_differences.png", audit, "v2_exploratory_post_hoc", V2)
        plot_lines(cdnow, metric, METHODS, CDNOW_ORIGINS, f"CDNOW confirmation · {METRICS[metric]}", OUT / f"cdnow_{metric}_origins.png", audit, "cdnow_confirmation", CDNOW, "CDNOW")
        plot_differences(cdnow, metric, METHODS, CDNOW_ORIGINS, "CDNOW", f"CDNOW confirmation differences · {METRICS[metric]}", OUT / f"cdnow_{metric}_hybrid_minus_m0.png", audit, "cdnow_confirmation", CDNOW)
    pd.DataFrame(audit).to_csv(OUT / "graph_data_audit.csv", index=False)
    print(f"Generated {len(list(OUT.glob('*.png')))} PNG figures and {len(audit)} audit rows under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

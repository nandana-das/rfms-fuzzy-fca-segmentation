"""Generate reproducible figures from saved RF(M)-FCA experiment outputs.

The script reads only structured result files.  It does not run experiments,
fit models, or modify source result folders.

Run from the repository root:
    python analysis/plot_experimental_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
UNIFIED_FIGURE_DIR = FIGURE_DIR / "unified_method_comparison"
METRICS = ("auc", "spend_r2", "invoice_r2")
METRIC_LABELS = {"auc": "AUC", "spend_r2": "Spend R²", "invoice_r2": "Invoice R²"}
REPRESENTATIONS = {
    "raw_std": "Raw RFM",
    "crisp_rfm_fca": "Crisp FCA",
    "fuzzy_rfm_fca": "Baseline Fuzzy FCA",
}
DATASET_ORDER = ("Dunnhumby", "Online Retail II", "CDNOW")
UNIFIED_DATASET_ORDER = ("Online Retail II", "Dunnhumby", "CDNOW")
UNIFIED_METHODS = {
    "raw_std": "Raw RFM",
    "crisp_rfm_fca": "Crisp RFM-FCA",
    "fuzzy_rfm_fca": "Baseline Fuzzy RFM-FCA",
    "hybrid_fuzzy_fca": "Hybrid Fuzzy RFM-FCA",
}
METHOD_STYLES = {
    "raw_std": {"color": "#4c78a8", "marker": "o"},
    "crisp_rfm_fca": {"color": "#f58518", "marker": "s"},
    "fuzzy_rfm_fca": {"color": "#54a24b", "marker": "^"},
    "hybrid_fuzzy_fca": {"color": "#e45756", "marker": "D"},
}

BASELINE_ROLLING = ROOT / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"
TAIL_METRICS = ROOT / "results" / "rfm_tail_information" / "per_origin_metrics.csv"
TAIL_COMPARISONS = ROOT / "results" / "rfm_tail_information" / "paired_comparisons.csv"
CDNOW_METRICS = ROOT / "results" / "cdnow_confirmation" / "per_origin_metrics.csv"
CDNOW_COMPARISONS = ROOT / "results" / "cdnow_confirmation" / "comparisons.csv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path}")


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    return frame


def validate_metrics(frame: pd.DataFrame, path: Path, key_columns: list[str]) -> None:
    duplicated = frame.duplicated(key_columns, keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, key_columns].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate result rows in {path}: {rows[:8]}")
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Missing or non-numeric {metric} values in {path}")
        frame[metric] = values


def load_baseline() -> pd.DataFrame:
    required = ["dataset", "origin", "arm", "auc", "spend_r2", "invoice_r2"]
    frame = read_csv(BASELINE_ROLLING, required + ["pooled"])
    frame = frame.loc[frame["pooled"].astype(bool)].copy()
    frame = frame.loc[frame["arm"].isin(REPRESENTATIONS)].copy()
    validate_metrics(frame, BASELINE_ROLLING, ["dataset", "origin", "arm"])
    expected = {(dataset, arm) for dataset in ("Dunnhumby", "Online Retail II") for arm in REPRESENTATIONS}
    observed = set(zip(frame["dataset"], frame["arm"]))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing baseline combinations: {missing}")
    return frame


def load_cdnow() -> pd.DataFrame:
    required = ["dataset", "origin", "arm", "auc", "spend_r2", "invoice_r2"]
    frame = read_csv(CDNOW_METRICS, required)
    frame = frame.loc[frame["arm"].isin((*REPRESENTATIONS, "hybrid_fuzzy_fca"))].copy()
    validate_metrics(frame, CDNOW_METRICS, ["dataset", "origin", "arm"])
    return frame


def load_tail() -> pd.DataFrame:
    required = ["dataset", "origin", "pooled", "arm", "auc", "spend_r2", "invoice_r2"]
    frame = read_csv(TAIL_METRICS, required)
    frame = frame.loc[frame["pooled"].astype(bool)].copy()
    frame = frame.loc[frame["arm"].isin(("fuzzy_rfm_fca", "tail_augmented_fuzzy_rfm_fca"))].copy()
    validate_metrics(frame, TAIL_METRICS, ["dataset", "origin", "arm"])
    return frame


def add_mean_source(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    grouped = frame.groupby(["dataset", "arm"], as_index=False)[list(METRICS)].mean()
    grouped["source_dataset"] = dataset
    return grouped


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output: Path, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        target = output.with_suffix(suffix)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {target}")
        fig.savefig(target, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def figure_one(baseline: pd.DataFrame, cdnow: pd.DataFrame, output: Path, overwrite: bool) -> None:
    pooled = pd.concat([baseline, cdnow], ignore_index=True)
    means = pooled.groupby(["dataset", "arm"], as_index=False)[list(METRICS)].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True)
    colors = {"raw_std": "#4c78a8", "crisp_rfm_fca": "#f58518", "fuzzy_rfm_fca": "#54a24b"}
    x = np.arange(len(DATASET_ORDER))
    for ax, metric in zip(axes, METRICS):
        for arm, label in REPRESENTATIONS.items():
            values = [means.loc[(means.dataset == dataset) & (means.arm == arm), metric].iloc[0] for dataset in DATASET_ORDER]
            ax.plot(x, values, marker="o", linewidth=2, label=label, color=colors[arm])
        style_axes(ax, METRIC_LABELS[metric])
        ax.set_xticks(x, DATASET_ORDER, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Baseline model comparison", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, overwrite)


def hybrid_delta_frame(tail: pd.DataFrame, cdnow: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in (("Dunnhumby", tail.loc[tail.dataset == "Dunnhumby"]),
                           ("Online Retail II", tail.loc[tail.dataset == "Online Retail II"])):
        pivot = frame.pivot(index=["dataset", "origin"], columns="arm", values=list(METRICS))
        for metric in METRICS:
            for (ds, origin), row in pivot.iterrows():
                rows.append({"dataset": ds, "origin": origin, "metric": metric,
                             "delta": row[(metric, "tail_augmented_fuzzy_rfm_fca")] - row[(metric, "fuzzy_rfm_fca")]})
    frame = cdnow.loc[cdnow.arm.isin(("fuzzy_rfm_fca", "hybrid_fuzzy_fca"))]
    pivot = frame.pivot(index=["dataset", "origin"], columns="arm", values=list(METRICS))
    for metric in METRICS:
        for (ds, origin), row in pivot.iterrows():
            rows.append({"dataset": ds, "origin": origin, "metric": metric,
                         "delta": row[(metric, "hybrid_fuzzy_fca")] - row[(metric, "fuzzy_rfm_fca")]})
    result = pd.DataFrame(rows)
    if result.isna().any().any():
        raise ValueError("Hybrid delta calculation encountered missing matched-arm values")
    return result


def figure_two(deltas: pd.DataFrame, output: Path, overwrite: bool) -> None:
    means = deltas.groupby(["dataset", "metric"], as_index=False)["delta"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True)
    x = np.arange(len(DATASET_ORDER))
    colors = {"auc": "#4c78a8", "spend_r2": "#f58518", "invoice_r2": "#54a24b"}
    for ax, metric in zip(axes, METRICS):
        values = [means.loc[(means.dataset == dataset) & (means.metric == metric), "delta"].iloc[0] for dataset in DATASET_ORDER]
        ax.plot(x, values, marker="o", linewidth=2, color=colors[metric], label=f"Δ {METRIC_LABELS[metric]}")
        ax.axhline(0, color="#555555", linewidth=0.9)
        style_axes(ax, f"Δ {METRIC_LABELS[metric]}")
        ax.set_xticks(x, DATASET_ORDER, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hybrid improvement over baseline fuzzy FCA", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, overwrite)


def figure_three(cdnow: pd.DataFrame, output: Path, overwrite: bool) -> None:
    frame = cdnow.loc[cdnow.arm.isin(("fuzzy_rfm_fca", "hybrid_fuzzy_fca"))].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin"])
    frame = frame.sort_values("origin_date")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True)
    colors = {"fuzzy_rfm_fca": "#4c78a8", "hybrid_fuzzy_fca": "#e45756"}
    labels = {"fuzzy_rfm_fca": "Baseline Fuzzy FCA", "hybrid_fuzzy_fca": "Hybrid Fuzzy FCA"}
    for ax, metric in zip(axes, METRICS):
        for arm in ("fuzzy_rfm_fca", "hybrid_fuzzy_fca"):
            subset = frame.loc[frame.arm == arm]
            ax.plot(subset.origin_date, subset[metric], marker="o", linewidth=2, label=labels[arm], color=colors[arm])
        style_axes(ax, METRIC_LABELS[metric])
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Test origin")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("CDNOW temporal evaluation", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, overwrite)


def figure_four(cdnow_comparisons: pd.DataFrame, output: Path, overwrite: bool) -> None:
    tail = read_csv(TAIL_COMPARISONS, ["dataset", "comparison", "metric", "delta", "ci_low", "ci_high", "p_holm"])
    tail = tail.loc[tail.comparison == "tail_augmented_fuzzy_rfm_fca - fuzzy_rfm_fca"].copy()
    cd = cdnow_comparisons.loc[cdnow_comparisons.id == "C4"].copy()
    cd["dataset"] = "CDNOW"
    cd = cd.rename(columns={"p_holm": "p_holm"})
    combined = pd.concat([tail[["dataset", "metric", "delta", "ci_low", "ci_high", "p_holm"]],
                          cd[["dataset", "metric", "delta", "ci_low", "ci_high", "p_holm"]]], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True)
    x = np.arange(len(DATASET_ORDER))
    colors = {"auc": "#4c78a8", "spend_r2": "#f58518", "invoice_r2": "#54a24b"}
    for ax, metric in zip(axes, METRICS):
        subset = combined.loc[combined.metric == metric].set_index("dataset").reindex(DATASET_ORDER)
        ax.errorbar(x, subset.delta, yerr=[subset.delta - subset.ci_low, subset.ci_high - subset.delta],
                    fmt="o-", capsize=4, linewidth=1.8, color=colors[metric])
        ax.axhline(0, color="#555555", linewidth=0.9)
        style_axes(ax, f"Δ {METRIC_LABELS[metric]}")
        ax.set_xticks(x, DATASET_ORDER, rotation=20, ha="right")
    axes[0].set_title("95% bootstrap intervals", fontsize=10)
    fig.suptitle("Hybrid improvement with saved uncertainty estimates", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, overwrite)


def unified_mean_frame(baseline: pd.DataFrame, tail: pd.DataFrame, cdnow: pd.DataFrame) -> pd.DataFrame:
    """Build one comparable pooled-mean table for the four requested methods."""
    rows: list[pd.DataFrame] = []

    base = baseline.loc[baseline.dataset.isin(("Dunnhumby", "Online Retail II"))].copy()
    rows.append(base[["dataset", "origin", "arm", *METRICS]])

    cd = cdnow.loc[cdnow.arm.isin(("raw_std", "crisp_rfm_fca", "fuzzy_rfm_fca"))].copy()
    rows.append(cd[["dataset", "origin", "arm", *METRICS]])

    # The frozen baseline file is authoritative for Raw/Crisp/Fuzzy. The tail
    # file is used here only for its additional hybrid arm; its fuzzy rows are
    # retained solely for the explicit control-matching validation below.
    hybrid_tail = tail.loc[tail.arm == "tail_augmented_fuzzy_rfm_fca"].copy()
    hybrid_tail["arm"] = hybrid_tail["arm"].replace({"tail_augmented_fuzzy_rfm_fca": "hybrid_fuzzy_fca"})
    rows.append(hybrid_tail[["dataset", "origin", "arm", *METRICS]])

    hybrid_cd = cdnow.loc[cdnow.arm == "hybrid_fuzzy_fca"].copy()
    rows.append(hybrid_cd[["dataset", "origin", "arm", *METRICS]])

    source = pd.concat(rows, ignore_index=True)
    validate_metrics(source, Path("unified source rows"), ["dataset", "origin", "arm"])
    means = source.groupby(["dataset", "arm"], as_index=False)[list(METRICS)].mean()

    expected = {(dataset, arm) for dataset in UNIFIED_DATASET_ORDER for arm in UNIFIED_METHODS}
    observed = set(zip(means.dataset, means.arm))
    missing = sorted(expected - observed)
    if missing:
        print("Unavailable method-dataset combinations:")
        for dataset, arm in missing:
            print(f"  - {dataset}: {UNIFIED_METHODS[arm]}")
    else:
        print("Unavailable method-dataset combinations: none")
    means["dataset"] = pd.Categorical(means["dataset"], categories=UNIFIED_DATASET_ORDER, ordered=True)
    return means.sort_values(["dataset", "arm"]).reset_index(drop=True)


def validate_hybrid_matching(baseline: pd.DataFrame, tail: pd.DataFrame, cdnow: pd.DataFrame) -> None:
    """Reject comparisons if baseline and hybrid source cohorts do not match."""
    for dataset in ("Dunnhumby", "Online Retail II"):
        base_origins = set(baseline.loc[baseline.dataset == dataset, "origin"])
        tail_origins = set(tail.loc[tail.dataset == dataset, "origin"])
        if base_origins != tail_origins:
            raise ValueError(f"Mismatched rolling origins for {dataset}: baseline={base_origins}, hybrid={tail_origins}")
        base_fuzzy = baseline.loc[(baseline.dataset == dataset) & (baseline.arm == "fuzzy_rfm_fca")].set_index("origin")
        tail_fuzzy = tail.loc[(tail.dataset == dataset) & (tail.arm == "fuzzy_rfm_fca")].set_index("origin")
        for metric in METRICS:
            if not np.allclose(base_fuzzy[metric], tail_fuzzy[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Saved fuzzy controls do not match for {dataset}, {metric}")
    cd_origins = set(cdnow.loc[cdnow.arm == "fuzzy_rfm_fca", "origin"])
    hybrid_origins = set(cdnow.loc[cdnow.arm == "hybrid_fuzzy_fca", "origin"])
    if cd_origins != hybrid_origins:
        raise ValueError(f"Mismatched CDNOW origins: fuzzy={cd_origins}, hybrid={hybrid_origins}")


def unified_figure(means: pd.DataFrame, metric: str, output: Path, overwrite: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(UNIFIED_DATASET_ORDER))
    for arm, label in UNIFIED_METHODS.items():
        subset = means.loc[means.arm == arm].set_index("dataset").reindex(UNIFIED_DATASET_ORDER)
        style = METHOD_STYLES[arm]
        ax.plot(x, subset[metric].to_numpy(dtype=float), linewidth=2, markersize=7, label=label, **style)
    style_axes(ax, METRIC_LABELS[metric])
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_xticks(x, UNIFIED_DATASET_ORDER, rotation=18, ha="right", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"{METRIC_LABELS[metric]} comparison across RFM-FCA methods", fontsize=10.5, pad=8)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.93),
        borderaxespad=0,
        fontsize=8,
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.018,
        "Hybrid Fuzzy RFM-FCA combines fuzzy FCA concept features with continuous log-RFM features; "
        "it is not a pure FCA-only representation. Plotted differences show direction and magnitude "
        "and do not by themselves establish statistical significance.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.16, 1.0, 0.96])
    save_figure(fig, output, overwrite)


def write_text(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing audit file: {path}")
    path.write_text(content, encoding="utf-8")


def build_verification_table(baseline: pd.DataFrame, tail: pd.DataFrame, cdnow: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable row for every plotted aggregate value."""
    records: list[dict[str, object]] = []
    sources = {
        "baseline": "results/baseline_ladder_rolling_origin/per_origin_metrics.csv",
        "tail": "results/rfm_tail_information/per_origin_metrics.csv",
        "cdnow": "results/cdnow_confirmation/per_origin_metrics.csv",
    }

    for dataset in ("Online Retail II", "Dunnhumby"):
        base = baseline.loc[baseline.dataset == dataset]
        tail_hybrid = tail.loc[(tail.dataset == dataset) & (tail.arm == "tail_augmented_fuzzy_rfm_fca")]
        for arm in ("raw_std", "crisp_rfm_fca", "fuzzy_rfm_fca"):
            row = base.loc[base.arm == arm, list(METRICS)].mean()
            for metric in METRICS:
                records.append({
                    "dataset": dataset,
                    "method": UNIFIED_METHODS[arm],
                    "metric": metric,
                    "value": float(row[metric]),
                    "source_file": sources["baseline"],
                    "evaluation_split": "pooled=True 91-day rolling origins",
                    "verification_status": "verified",
                })
        row = tail_hybrid[list(METRICS)].mean()
        for metric in METRICS:
            records.append({
                "dataset": dataset,
                "method": UNIFIED_METHODS["hybrid_fuzzy_fca"],
                "metric": metric,
                "value": float(row[metric]),
                "source_file": sources["tail"],
                "evaluation_split": "pooled=True 91-day rolling origins",
                "verification_status": "verified",
            })

    for arm in ("raw_std", "crisp_rfm_fca", "fuzzy_rfm_fca", "hybrid_fuzzy_fca"):
        row = cdnow.loc[cdnow.arm == arm, list(METRICS)].mean()
        for metric in METRICS:
            records.append({
                "dataset": "CDNOW",
                "method": UNIFIED_METHODS[arm],
                "metric": metric,
                "value": float(row[metric]),
                "source_file": sources["cdnow"],
                "evaluation_split": "mean over three chronological 91-day origins",
                "verification_status": "verified",
            })

    table = pd.DataFrame(records)
    expected = len(UNIFIED_DATASET_ORDER) * len(UNIFIED_METHODS) * len(METRICS)
    if len(table) != expected:
        raise ValueError(f"Verification table has {len(table)} rows; expected {expected}")
    if table["value"].isna().any():
        raise ValueError("Verification table contains missing plotted values")
    return table


def audit_delta_results(table: pd.DataFrame, tail: pd.DataFrame, cdnow_comparisons: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compare plotted aggregate deltas with the saved paired-comparison files."""
    expected_sources = {
        "Online Retail II": TAIL_COMPARISONS,
        "Dunnhumby": TAIL_COMPARISONS,
        "CDNOW": CDNOW_COMPARISONS,
    }
    computed: list[dict[str, object]] = []
    issues: list[str] = []
    for dataset in UNIFIED_DATASET_ORDER:
        for metric in METRICS:
            subset = table.loc[(table.dataset == dataset) & (table.metric == metric)].set_index("method")
            computed_delta = float(subset.loc[UNIFIED_METHODS["hybrid_fuzzy_fca"], "value"] - subset.loc[UNIFIED_METHODS["fuzzy_rfm_fca"], "value"])
            if dataset == "CDNOW":
                saved = cdnow_comparisons.loc[(cdnow_comparisons.id == "C4") & (cdnow_comparisons.metric == metric)]
            else:
                saved_all = read_csv(TAIL_COMPARISONS, ["dataset", "comparison", "metric", "delta", "ci_low", "ci_high", "p_holm"])
                saved = saved_all.loc[(saved_all.dataset == dataset) & (saved_all.metric == metric) & (saved_all.comparison == "tail_augmented_fuzzy_rfm_fca - fuzzy_rfm_fca")]
            if len(saved) != 1:
                status = "missing saved paired comparison"
                issues.append(f"{dataset} {metric}: {status}")
                saved_delta = np.nan
                ci = ""
                p_holm = np.nan
            else:
                saved_delta = float(saved.iloc[0]["delta"])
                ci = f"[{float(saved.iloc[0]['ci_low']):.12g}, {float(saved.iloc[0]['ci_high']):.12g}]"
                p_holm = float(saved.iloc[0]["p_holm"])
                status = "confirmed" if np.isclose(computed_delta, saved_delta, atol=1e-12, rtol=0) else "mismatch"
                if status == "mismatch":
                    issues.append(f"{dataset} {metric}: computed {computed_delta:.12g} != saved {saved_delta:.12g}")
            computed.append({
                "dataset": dataset,
                "metric": metric,
                "computed_hybrid_minus_fuzzy": computed_delta,
                "saved_delta": saved_delta,
                "saved_95ci": ci,
                "saved_p_holm": p_holm,
                "source_file": str(expected_sources[dataset].relative_to(ROOT)).replace("\\", "/"),
                "verification_status": status,
            })
    return pd.DataFrame(computed), issues


def write_audit_artifacts(table: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "verification_table.csv"
    if table_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite audit table: {table_path}")
    table.to_csv(table_path, index=False)
    delta_path = output_dir / "hybrid_delta_audit.csv"
    if delta_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite delta audit: {delta_path}")
    deltas.to_csv(delta_path, index=False)
    verified = int((table.verification_status == "verified").sum())
    lines = [
        "# Four-method RFM-FCA figure audit",
        "",
        "The three figures were generated from saved structured experiment outputs; no models or experiments were rerun.",
        "",
        "## Sources",
        "",
        "- Raw, Crisp, and Baseline Fuzzy RFM-FCA: `results/baseline_ladder_rolling_origin/per_origin_metrics.csv` for pooled 91-day rolling origins.",
        "- Hybrid Fuzzy RFM-FCA for Dunnhumby and Online Retail II: `results/rfm_tail_information/per_origin_metrics.csv`, using the saved tail-augmented arm and pooled 91-day origins.",
        "- All four CDNOW methods: `results/cdnow_confirmation/per_origin_metrics.csv`, averaged over the three chronological 91-day origins.",
        "",
        "## Verification",
        "",
        f"- Plotted points verified: {verified}/{len(table)}.",
        f"- Missing plotted values: {int(table.value.isna().sum())}.",
        f"- Hybrid delta checks confirmed: {int((deltas.verification_status == 'confirmed').sum())}/{len(deltas)}.",
        f"- Hybrid delta mismatches or missing comparisons: {int((deltas.verification_status != 'confirmed').sum())}.",
        "- CDNOW Raw RFM and Crisp RFM-FCA are present in the CDNOW confirmation per-origin result file and were not inferred from another experiment.",
        "- No visual separation is treated as statistical significance.",
        "- The hybrid is a fuzzy FCA plus continuous log-RFM representation, not a pure FCA-only method.",
        "",
        "## Reported aggregate deltas",
        "",
        "See `hybrid_delta_audit.csv` for computed values, saved paired-comparison values, confidence intervals, and Holm-adjusted p-values.",
        "",
        "## Regeneration",
        "",
        "```powershell",
        "python analysis/plot_experimental_results.py",
        "```",
        "",
        "Primary figures are in this directory as PNG (300 DPI) and vector PDF files.",
    ]
    write_text(output_dir / "AUDIT_REPORT.md", "\n".join(lines) + "\n", overwrite)


def print_summary(baseline: pd.DataFrame, tail: pd.DataFrame, cdnow: pd.DataFrame, deltas: pd.DataFrame) -> None:
    print("Source files used:")
    for path in (BASELINE_ROLLING, TAIL_METRICS, TAIL_COMPARISONS, CDNOW_METRICS, CDNOW_COMPARISONS):
        print(f"  - {path.relative_to(ROOT)}")
    print("Rows used:")
    print(f"  - Figure 1: {len(baseline)} pooled rolling rows plus {len(cdnow.loc[cdnow.arm.isin(REPRESENTATIONS)])} CDNOW baseline rows")
    print(f"  - Figure 2: {len(deltas)} matched per-origin metric deltas")
    print(f"  - Figure 3: {len(cdnow.loc[cdnow.arm.isin(('fuzzy_rfm_fca', 'hybrid_fuzzy_fca'))])} CDNOW temporal rows")
    print("  - Figure 4: pooled saved paired comparisons with confidence intervals")
    print("No values were interpolated or filled.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail if a derived PNG/PDF figure already exists",
    )
    args = parser.parse_args()

    baseline = load_baseline()
    tail = load_tail()
    cdnow = load_cdnow()
    cdnow_comparisons = read_csv(CDNOW_COMPARISONS, ["id", "metric", "delta", "ci_low", "ci_high", "p_holm"])
    deltas = hybrid_delta_frame(tail, cdnow)
    validate_hybrid_matching(baseline, tail, cdnow)
    unified = unified_mean_frame(baseline, tail, cdnow)
    verification = build_verification_table(baseline, tail, cdnow)
    delta_audit, delta_issues = audit_delta_results(verification, tail, cdnow_comparisons)
    if delta_issues:
        print("Audit issues:")
        for issue in delta_issues:
            print(f"  - {issue}")

    overwrite = not args.no_overwrite
    unified_dir = args.output_dir / "unified_method_comparison"
    write_audit_artifacts(verification, delta_audit, unified_dir, overwrite)
    unified_figure(unified, "auc", unified_dir / "auc_four_method_comparison", overwrite)
    unified_figure(unified, "spend_r2", unified_dir / "spend_r2_four_method_comparison", overwrite)
    unified_figure(unified, "invoice_r2", unified_dir / "invoice_r2_four_method_comparison", overwrite)
    print_summary(baseline, tail, cdnow, deltas)
    print(f"Unified figures written to: {unified_dir}")


if __name__ == "__main__":
    main()

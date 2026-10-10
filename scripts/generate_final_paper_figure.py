"""Generate the compact frozen-v1 predictive figure from existing CSV values."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"
OUTPUT = ROOT / "outputs" / "figures" / "final_paper" / "predictive_performance_v1.png"
METHODS = (
    ("crisp_rfm_fca", "Crisp RFM-FCA", "#E69F00", "o", "-"),
    ("fuzzy_rfm_fca", "Baseline Fuzzy RFM-FCA (M0)", "#009E73", "s", "--"),
)
METRICS = (("auc", "ROC AUC"), ("spend_r2", "Spend R²"), ("invoice_r2", "Invoice R²"))


def main() -> None:
    frame = pd.read_csv(SOURCE)
    selected = frame[frame["arm"].isin([m[0] for m in METHODS]) & (frame["pooled"] == True)].copy()  # noqa: E712
    if len(selected) != 18 or selected.duplicated(["dataset", "origin", "arm"]).any():
        raise ValueError("unexpected v1 source coverage or duplicate observations")
    if set(selected["dataset"]) != {"Dunnhumby", "Online Retail II"}:
        raise ValueError("unexpected dataset coverage")
    if selected[[m[0] for m in METRICS]].isna().any().any():
        raise ValueError("missing metric value in selected source rows")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.22, hspace=0.40)
    for row, dataset in enumerate(("Dunnhumby", "Online Retail II")):
        subset = selected[selected["dataset"] == dataset]
        origins = list(dict.fromkeys(subset.sort_values("origin")["origin"]))
        for col, (metric, ylabel) in enumerate(METRICS):
            axis = axes[row, col]
            for identifier, label, color, marker, linestyle in METHODS:
                series = subset[subset["arm"] == identifier].set_index("origin").loc[origins, metric]
                axis.plot(range(len(origins)), series.to_numpy(), label=label, color=color,
                          marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6)
            axis.set_title(f"{dataset}: {ylabel}", fontsize=13)
            axis.set_xticks(range(len(origins)), origins, rotation=35, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_ylim(bottom=min(0, axis.get_ylim()[0]))
            if row == 0 and col == 0:
                axis.legend(frameon=False, fontsize=10, loc="best")

    fig.suptitle("Frozen v1 primary comparison: origin-level predictive performance", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.045, "Pooled 91-day rolling origins; Online Retail II 365-day primary holdout excluded.",
             ha="center", fontsize=10)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Generate publication-quality figures and audit artifacts for Fuzzy RFM-FCA research paper.

Experiments covered:
- Experiment 1: Crisp RFM-FCA vs Baseline Fuzzy RFM-FCA (M0)
- Experiment 2: Baseline Fuzzy RFM-FCA (M0) vs Tail-Sensitive Fuzzy RFM-FCA (M1)
- Experiment 3 (Figure 4): Segmentation Stability (Subsampling and Quarterly Re-segmentation)

Produces:
- Figure 1 (PNG, PDF, SVG): Crisp vs M0
- Figure 2 (PNG, PDF, SVG): M0 vs M1
- Figure 3 (PNG, PDF, SVG): Per-origin M1 - M0 differences
- Figure 4 (PNG, PDF, SVG): Segmentation stability
- plotted_data.csv: Exact values plotted in all figures
- audit_report.md: Formal audit documentation
- figure_captions.md: Formal academic captions
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Paths
ROOT = Path("d:/Nandana/MTECH/Semester 3/Projects/MP/rfms_fca_project")
OUT_DIR = ROOT / "outputs" / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Authoritative Sources
SRC_V1 = ROOT / "results" / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv"
SRC_V2 = ROOT / "results" / "fuzzy_tnorm_tail_ablation" / "multi_origin_metrics.csv"
SRC_REFIT = ROOT / "results" / "segment_stability" / "refit_summary.csv"
SRC_REFIT_COMP = ROOT / "results" / "segment_stability" / "refit_comparisons.csv"
SRC_TEMP = ROOT / "results" / "segment_stability" / "temporal_summary.csv"
SRC_TEMP_COMP = ROOT / "results" / "segment_stability" / "temporal_comparisons.csv"

# Okabe-Ito Colorblind-Safe Palette
CLR_CRISP = "#E69F00"     # Amber / Orange
CLR_M0 = "#009E73"        # Teal / Bluish Green
CLR_M1 = "#D55E00"        # Vermilion / Red-Orange
CLR_BLUE = "#0072B2"      # Blue for secondary comparisons
CLR_NEUTRAL = "#333333"
CLR_GRID = "#EAEAEA"

# Publication Matplotlib Configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 13,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": CLR_GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})

# ==============================================================================
# DATA LOADING & AUDIT VALIDATION
# ==============================================================================

v1_df = pd.read_csv(SRC_V1)
v2_df = pd.read_csv(SRC_V2)

# Verify v1 pooled rows (K=4 Dunnhumby, K=5 Online Retail II)
v1_pooled = v1_df[v1_df["pooled"] == True].copy()
v1_exp1 = v1_pooled[v1_pooled["arm"].isin(["crisp_rfm_fca", "fuzzy_rfm_fca"])].copy()

# Verify v2 exp2 rows (K=4 Dunnhumby, K=5 Online Retail II)
v2_exp2 = v2_df[v2_df["arm"].isin(["M0", "M1"])].copy()

# Audit record collector
plotted_records: list[dict] = []

print("Auditing source files...")
assert len(v1_exp1) == 18, f"Expected 18 rows in v1 exp1, found {len(v1_exp1)}"
assert len(v2_exp2) == 18, f"Expected 18 rows in v2 exp2, found {len(v2_exp2)}"

# Parity assertion between v1 fuzzy_rfm_fca and v2 M0
m0_parity_v1 = v1_exp1[v1_exp1["arm"] == "fuzzy_rfm_fca"].sort_values(["dataset", "origin"]).reset_index(drop=True)
m0_parity_v2 = v2_exp2[v2_exp2["arm"] == "M0"].sort_values(["dataset", "origin"]).reset_index(drop=True)
for m_col in ["auc", "spend_r2", "invoice_r2"]:
    max_d = np.max(np.abs(m0_parity_v1[m_col] - m0_parity_v2[m_col]))
    assert max_d < 1e-12, f"Parity check failed on {m_col}: max_diff={max_d}"

print("All parity checks passed (M0 identical across v1 and v2).")


# ==============================================================================
# FIGURE 1: Crisp versus Baseline Fuzzy RFM-FCA (M0)
# ==============================================================================

def generate_figure1():
    print("Generating Figure 1: Crisp vs Baseline Fuzzy RFM-FCA (M0)...")
    metrics = [
        ("auc", "Repurchase ROC AUC", (0.60, 1.02), 0.05),
        ("spend_r2", "Future Spend R²", (0.00, 0.70), 0.10),
        ("invoice_r2", "Future Invoice-count R²", (0.00, 0.76), 0.10)
    ]
    datasets = ["Dunnhumby", "Online Retail II"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), constrained_layout=True)
    fig.suptitle("Experiment 1: Comparison of Crisp and Baseline Fuzzy RFM-FCA (M0)", fontsize=13, fontweight="bold", y=1.03)

    for ax_idx, (m_key, m_title, y_lim, y_step) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_title(f"({chr(97 + ax_idx)}) {m_title}", fontsize=11, fontweight="semibold", pad=10)
        ax.set_ylim(y_lim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        x_positions = [0, 1]  # Dunnhumby, Online Retail II
        bar_width = 0.32

        for d_idx, ds in enumerate(datasets):
            ds_sub = v1_exp1[v1_exp1["dataset"] == ds]
            crisp_vals = ds_sub[ds_sub["arm"] == "crisp_rfm_fca"].sort_values("origin")[m_key].to_numpy()
            fuzzy_vals = ds_sub[ds_sub["arm"] == "fuzzy_rfm_fca"].sort_values("origin")[m_key].to_numpy()
            origins = ds_sub[ds_sub["arm"] == "crisp_rfm_fca"].sort_values("origin")["origin"].to_numpy()

            crisp_mean = float(np.mean(crisp_vals))
            fuzzy_mean = float(np.mean(fuzzy_vals))
            diff_mean = fuzzy_mean - crisp_mean
            pos_count = int(np.sum(fuzzy_vals > crisp_vals))
            total_origins = len(crisp_vals)

            # Record plotted data
            for orig, cv, fv in zip(origins, crisp_vals, fuzzy_vals):
                plotted_records.append({
                    "figure": "Figure 1",
                    "panel": f"({chr(97 + ax_idx)}) {m_title}",
                    "dataset": ds,
                    "origin": orig,
                    "metric": m_title,
                    "metric_key": m_key,
                    "model_a": "Crisp RFM-FCA",
                    "model_b": "Baseline Fuzzy RFM-FCA (M0)",
                    "value_a": cv,
                    "value_b": fv,
                    "difference_b_minus_a": fv - cv,
                    "mean_a": crisp_mean,
                    "mean_b": fuzzy_mean,
                    "mean_difference": diff_mean,
                    "origins_positive": f"{pos_count}/{total_origins}"
                })

            pos_crisp = x_positions[d_idx] - bar_width / 2
            pos_fuzzy = x_positions[d_idx] + bar_width / 2

            # Plot bars
            ax.bar(pos_crisp, crisp_mean, width=bar_width * 0.92, color=CLR_CRISP, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3, label="Crisp RFM-FCA" if (ax_idx == 0 and d_idx == 0) else "")
            ax.bar(pos_fuzzy, fuzzy_mean, width=bar_width * 0.92, color=CLR_M0, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3, label="Baseline Fuzzy RFM-FCA (M0)" if (ax_idx == 0 and d_idx == 0) else "")

            # Overlay individual origins as paired connected points
            for cv, fv in zip(crisp_vals, fuzzy_vals):
                ax.plot([pos_crisp, pos_fuzzy], [cv, fv], color="#555555", linestyle=":", linewidth=0.9, alpha=0.75, zorder=4)
                ax.scatter(pos_crisp, cv, color="#ffffff", edgecolor="#222222", s=32, zorder=5, linewidth=1.0)
                ax.scatter(pos_fuzzy, fv, color="#ffffff", edgecolor="#222222", s=32, zorder=5, linewidth=1.0)

            # Find maximum vertical extent for this group
            max_y_val = max(np.max(crisp_vals), np.max(fuzzy_vals), crisp_mean, fuzzy_mean)

            # Labels for bar means positioned cleanly above max points to avoid collision
            label_y = max_y_val + (y_lim[1] - y_lim[0]) * 0.035
            ax.text(pos_crisp, label_y, f"{crisp_mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")
            ax.text(pos_fuzzy, label_y, f"{fuzzy_mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")

            # Bracket showing delta
            bracket_y = label_y + (y_lim[1] - y_lim[0]) * 0.045
            ax.plot([pos_crisp, pos_crisp, pos_fuzzy, pos_fuzzy],
                    [bracket_y - (y_lim[1] - y_lim[0]) * 0.012, bracket_y, bracket_y, bracket_y - (y_lim[1] - y_lim[0]) * 0.012],
                    color="#333333", linewidth=0.8, zorder=6)
            diff_sign = "+" if diff_mean > 0 else ""
            ax.text((pos_crisp + pos_fuzzy) / 2, bracket_y + (y_lim[1] - y_lim[0]) * 0.012,
                    f"Δ = {diff_sign}{diff_mean:.3f} ({pos_count}/{total_origins})",
                    ha="center", va="bottom", fontsize=7.8, color="#222222", fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"Dunnhumby\n(K=4 origins)", f"Online Retail II\n(K=5 origins)"], fontsize=9.5, fontweight="semibold")
        ax.set_ylabel(m_title, fontsize=10)

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, edgecolor="#cccccc", fontsize=9.5)

    # Save vector & high-res PNG
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"figure1_crisp_vs_fuzzy.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 saved.")


# ==============================================================================
# FIGURE 2: M0 versus Tail-Sensitive M1
# ==============================================================================

def generate_figure2():
    print("Generating Figure 2: Baseline Fuzzy M0 vs Tail-Sensitive M1...")
    # Exact matching axes and scales as Figure 1 for direct comparability
    metrics = [
        ("auc", "Repurchase ROC AUC", (0.60, 1.02), 0.05),
        ("spend_r2", "Future Spend R²", (0.00, 0.70), 0.10),
        ("invoice_r2", "Future Invoice-count R²", (0.00, 0.76), 0.10)
    ]
    datasets = ["Dunnhumby", "Online Retail II"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), constrained_layout=True)
    fig.suptitle("Experiment 2: Evaluation of Tail-Sensitive Membership Functions (M0 vs M1)", fontsize=13, fontweight="bold", y=1.03)

    for ax_idx, (m_key, m_title, y_lim, y_step) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_title(f"({chr(97 + ax_idx)}) {m_title}", fontsize=11, fontweight="semibold", pad=10)
        ax.set_ylim(y_lim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        x_positions = [0, 1]  # Dunnhumby, Online Retail II
        bar_width = 0.32

        for d_idx, ds in enumerate(datasets):
            ds_sub = v2_exp2[v2_exp2["dataset"] == ds]
            m0_vals = ds_sub[ds_sub["arm"] == "M0"].sort_values("origin")[m_key].to_numpy()
            m1_vals = ds_sub[ds_sub["arm"] == "M1"].sort_values("origin")[m_key].to_numpy()
            origins = ds_sub[ds_sub["arm"] == "M0"].sort_values("origin")["origin"].to_numpy()

            m0_mean = float(np.mean(m0_vals))
            m1_mean = float(np.mean(m1_vals))
            diff_mean = m1_mean - m0_mean
            pos_count = int(np.sum(m1_vals > m0_vals))
            total_origins = len(m0_vals)

            # Record plotted data
            for orig, m0v, m1v in zip(origins, m0_vals, m1_vals):
                plotted_records.append({
                    "figure": "Figure 2",
                    "panel": f"({chr(97 + ax_idx)}) {m_title}",
                    "dataset": ds,
                    "origin": orig,
                    "metric": m_title,
                    "metric_key": m_key,
                    "model_a": "Baseline Fuzzy RFM-FCA (M0)",
                    "model_b": "Tail-Sensitive Fuzzy RFM-FCA (M1)",
                    "value_a": m0v,
                    "value_b": m1v,
                    "difference_b_minus_a": m1v - m0v,
                    "mean_a": m0_mean,
                    "mean_b": m1_mean,
                    "mean_difference": diff_mean,
                    "origins_positive": f"{pos_count}/{total_origins}"
                })

            pos_m0 = x_positions[d_idx] - bar_width / 2
            pos_m1 = x_positions[d_idx] + bar_width / 2

            # Plot bars
            ax.bar(pos_m0, m0_mean, width=bar_width * 0.92, color=CLR_M0, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3, label="Baseline Fuzzy RFM-FCA (M0)" if (ax_idx == 0 and d_idx == 0) else "")
            ax.bar(pos_m1, m1_mean, width=bar_width * 0.92, color=CLR_M1, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3, label="Tail-Sensitive Fuzzy RFM-FCA (M1)" if (ax_idx == 0 and d_idx == 0) else "")

            # Overlay individual origins as paired connected points
            for m0v, m1v in zip(m0_vals, m1_vals):
                ax.plot([pos_m0, pos_m1], [m0v, m1v], color="#555555", linestyle=":", linewidth=0.9, alpha=0.75, zorder=4)
                ax.scatter(pos_m0, m0v, color="#ffffff", edgecolor="#222222", s=32, zorder=5, linewidth=1.0)
                ax.scatter(pos_m1, m1v, color="#ffffff", edgecolor="#222222", s=32, zorder=5, linewidth=1.0)

            # Find maximum vertical extent for this group
            max_y_val = max(np.max(m0_vals), np.max(m1_vals), m0_mean, m1_mean)

            # Labels for bar means positioned cleanly above max points to avoid collision
            label_y = max_y_val + (y_lim[1] - y_lim[0]) * 0.035
            ax.text(pos_m0, label_y, f"{m0_mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")
            ax.text(pos_m1, label_y, f"{m1_mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")

            # Bracket showing delta
            bracket_y = label_y + (y_lim[1] - y_lim[0]) * 0.045
            ax.plot([pos_m0, pos_m0, pos_m1, pos_m1],
                    [bracket_y - (y_lim[1] - y_lim[0]) * 0.012, bracket_y, bracket_y, bracket_y - (y_lim[1] - y_lim[0]) * 0.012],
                    color="#333333", linewidth=0.8, zorder=6)
            diff_sign = "+" if diff_mean > 0 else ""
            ax.text((pos_m0 + pos_m1) / 2, bracket_y + (y_lim[1] - y_lim[0]) * 0.012,
                    f"Δ = {diff_sign}{diff_mean:.3f} ({pos_count}/{total_origins})",
                    ha="center", va="bottom", fontsize=7.8, color="#222222", fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"Dunnhumby\n(K=4 origins)", f"Online Retail II\n(K=5 origins)"], fontsize=9.5, fontweight="semibold")
        ax.set_ylabel(m_title, fontsize=10)

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, edgecolor="#cccccc", fontsize=9.5)

    # Save vector & high-res PNG
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"figure2_m0_vs_m1.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 2 saved.")


# ==============================================================================
# FIGURE 3: Per-origin M1 − M0 differences
# ==============================================================================

def generate_figure3():
    print("Generating Figure 3: Per-origin M1 - M0 differences...")
    # Symmetrical / balanced y-limits to make zero line prominent and prevent label clipping
    metrics = [
        ("auc", "Δ Repurchase ROC AUC", (-0.005, 0.024), 0.005),
        ("spend_r2", "Δ Future Spend R²", (-0.002, 0.007), 0.002),
        ("invoice_r2", "Δ Future Invoice-count R²", (-0.006, 0.040), 0.010)
    ]
    datasets = ["Dunnhumby", "Online Retail II"]

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0), constrained_layout=True)
    fig.suptitle("Experiment 2: Paired Per-Origin Metric Differences (M1 − M0)", fontsize=13, fontweight="bold", y=1.02)

    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    for r_idx, ds in enumerate(datasets):
        ds_sub = v2_exp2[v2_exp2["dataset"] == ds]
        m0_sub = ds_sub[ds_sub["arm"] == "M0"].set_index("origin")
        m1_sub = ds_sub[ds_sub["arm"] == "M1"].set_index("origin")

        # Chronological origins
        if ds == "Dunnhumby":
            origins = ["day 347", "day 438", "day 529", "day 620"]
            disp_origins = ["Day 347", "Day 438", "Day 529", "Day 620"]
        else:
            origins = ["2010-09-10", "2010-12-10", "2011-03-11", "2011-06-10", "2011-09-09"]
            disp_origins = origins

        metric_cols = [m[0] for m in metrics]
        diff_df = m1_sub.loc[origins, metric_cols].astype(float) - m0_sub.loc[origins, metric_cols].astype(float)

        for c_idx, (m_key, m_label, y_lim, y_step) in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            panel_letter = panel_labels[r_idx][c_idx]
            ax.set_title(f"{panel_letter} {ds} · {m_label}", fontsize=10.5, fontweight="semibold", pad=8)
            ax.set_ylim(y_lim)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
            ax.grid(axis="y", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

            # Prominent Zero reference line
            ax.axhline(0, color="#111111", linestyle="-", linewidth=1.1, zorder=2)

            diff_vals = diff_df[m_key].to_numpy()
            x = np.arange(len(origins))
            mean_d = float(np.mean(diff_vals))
            pos_count = int(np.sum(diff_vals > 0))
            total_k = len(diff_vals)

            # Record plotted data
            for orig, dv in zip(origins, diff_vals):
                plotted_records.append({
                    "figure": "Figure 3",
                    "panel": f"{panel_letter} {ds} · {m_label}",
                    "dataset": ds,
                    "origin": orig,
                    "metric": m_label,
                    "metric_key": m_key,
                    "model_a": "M0",
                    "model_b": "M1",
                    "value_a": float(m0_sub.loc[orig, m_key]),
                    "value_b": float(m1_sub.loc[orig, m_key]),
                    "difference_b_minus_a": dv,
                    "mean_a": float(m0_sub.loc[origins, m_key].mean()),
                    "mean_b": float(m1_sub.loc[origins, m_key].mean()),
                    "mean_difference": mean_d,
                    "origins_positive": f"{pos_count}/{total_k}"
                })

            # Plot stems and markers
            for xi, val in zip(x, diff_vals):
                clr = CLR_M1 if val >= 0 else CLR_BLUE
                ax.plot([xi, xi], [0, val], color=clr, linewidth=1.8, zorder=3)
                ax.scatter(xi, val, color=clr, edgecolor="#111111", s=55, zorder=4, linewidth=0.9)

                # Format annotation
                v_str = f"{val:+.4f}" if abs(val) < 0.01 else f"{val:+.3f}"
                va = "bottom" if val >= 0 else "top"
                y_offset = (y_lim[1] - y_lim[0]) * 0.045 * (1 if val >= 0 else -1)
                ax.text(xi, val + y_offset, v_str, ha="center", va=va, fontsize=8.0, fontweight="bold", color="#111111")

            # Connect trajectory with faint line
            ax.plot(x, diff_vals, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)

            # Subtle horizontal dashed line for mean difference
            ax.axhline(mean_d, color="#555555", linestyle=":", linewidth=1.1, alpha=0.8, zorder=2)

            # Place clean mean badge in top-left or top-right corner to avoid colliding with stems
            mean_sign = "+" if mean_d > 0 else ""
            badge_text = f"Mean Δ = {mean_sign}{mean_d:.4f}\nPos: {pos_count}/{total_k} origins"
            # Place in corner opposite to large points
            loc_x = 0.96 if (r_idx == 1 and c_idx == 0) else (0.04 if ds == "Dunnhumby" and c_idx == 1 else 0.96)
            ha = "right" if loc_x > 0.5 else "left"
            ax.text(loc_x, 0.94, badge_text, transform=ax.transAxes, ha=ha, va="top",
                    fontsize=8.0, fontweight="semibold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.92, zorder=5))

            ax.set_xticks(x)
            ax.set_xticklabels(disp_origins, rotation=25 if ds == "Online Retail II" else 0,
                               ha="right" if ds == "Online Retail II" else "center", fontsize=8.5)
            ax.set_ylabel(m_label, fontsize=9.5)
            ax.set_xlim(-0.5, len(origins) - 0.5)

    # Save vector & high-res PNG
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"figure3_per_origin_m1_m0_differences.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 saved.")


# ==============================================================================
# ==============================================================================
# FIGURE 4: Segmentation Stability Evaluation (Conditional)
# ==============================================================================

def generate_figure4():
    print("Generating Figure 4: Segmentation stability (3-panel layout)...")
    refit_sum = pd.read_csv(SRC_REFIT)
    temp_sum = pd.read_csv(SRC_TEMP)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    fig.suptitle("Figure 4: Segmentation Stability Evaluation (Crisp vs Baseline Fuzzy RFM-FCA)", fontsize=13, fontweight="bold", y=1.03)

    sub_measures = [
        ("core_jaccard", "Core Jaccard\n(μ ≥ 0.5)"),
        ("fuzzy_jaccard", "Fuzzy\nJaccard"),
        ("concept_recurrence", "Concept\nRecurrence")
    ]
    bar_w = 0.32
    x_sub = np.arange(len(sub_measures))

    # Helper for subsampling panel
    def plot_subsampling_panel(ax, ds_name, panel_letter):
        ax.set_title(f"({panel_letter}) Subsampling: {ds_name} (80% Bootstrap)", fontsize=11, fontweight="semibold", pad=10)
        ax.set_ylim(0.88, 1.025)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        for m_idx, (m_col, m_name) in enumerate(sub_measures):
            c_val = float(refit_sum[(refit_sum["dataset"] == ds_name) & (refit_sum["arm"] == "crisp_rfm_fca")][m_col].iloc[0])
            f_val = float(refit_sum[(refit_sum["dataset"] == ds_name) & (refit_sum["arm"] == "fuzzy_rfm_fca")][m_col].iloc[0])
            diff_val = f_val - c_val

            pos_c = x_sub[m_idx] - bar_w / 2
            pos_f = x_sub[m_idx] + bar_w / 2

            ax.bar(pos_c, c_val, width=bar_w * 0.90, color=CLR_CRISP, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3,
                   label="Crisp RFM-FCA" if (panel_letter == "a" and m_idx == 0) else "")
            ax.bar(pos_f, f_val, width=bar_w * 0.90, color=CLR_M0, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3,
                   label="Baseline Fuzzy RFM-FCA (M0)" if (panel_letter == "a" and m_idx == 0) else "")

            # Value labels above bars
            ax.text(pos_c, c_val + 0.003, f"{c_val:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#222222")
            ax.text(pos_f, f_val + 0.003, f"{f_val:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#222222")

            # Bracket showing delta
            max_y = max(c_val, f_val)
            bracket_y = max_y + 0.016
            ax.plot([pos_c, pos_c, pos_f, pos_f], [bracket_y - 0.003, bracket_y, bracket_y, bracket_y - 0.003],
                    color="#333333", linewidth=0.8, zorder=6)
            diff_sign = "+" if diff_val > 0 else ""
            ax.text(x_sub[m_idx], bracket_y + 0.003, f"Δ = {diff_sign}{diff_val:.3f}",
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#222222")

            # Record audit data
            plotted_records.append({
                "figure": "Figure 4", "panel": f"({panel_letter}) Subsampling: {ds_name}", "dataset": ds_name, "origin": "80% Subsample Pooled",
                "metric": m_name.replace("\n", " "), "metric_key": m_col, "model_a": "Crisp RFM-FCA", "model_b": "Baseline Fuzzy RFM-FCA (M0)",
                "value_a": c_val, "value_b": f_val, "difference_b_minus_a": diff_val,
                "mean_a": c_val, "mean_b": f_val, "mean_difference": diff_val, "origins_positive": "N/A"
            })

        ax.set_xticks(x_sub)
        ax.set_xticklabels([m[1] for m in sub_measures], fontsize=9.0, fontweight="semibold")
        ax.set_ylabel("Stability Index", fontsize=10)

    # Panel A: Dunnhumby subsampling
    plot_subsampling_panel(ax1, "Dunnhumby", "a")

    # Panel B: Online Retail II subsampling
    plot_subsampling_panel(ax2, "Online Retail II", "b")

    # --------------------------------------------------------------------------
    # PANEL C: Quarterly Re-segmentation Stability (Study B: Consecutive Quarters)
    # --------------------------------------------------------------------------
    ax3.set_title("(c) Quarterly Re-segmentation Stability", fontsize=11, fontweight="semibold", pad=10)
    ax3.set_ylim(0.35, 1.05)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax3.grid(axis="y", zorder=0)
    ax3.spines[["top", "right"]].set_visible(False)

    subsets_eval = [
        ("Dunnhumby", "all", "Dunnhumby\nAll"),
        ("Dunnhumby", "stable", "Dunnhumby\nStable"),
        ("Online Retail II", "all", "Retail II\nAll"),
        ("Online Retail II", "stable", "Retail II\nStable")
    ]
    x_pos_temp = np.arange(len(subsets_eval))
    bar_w_temp = 0.34

    for s_idx, (ds, s_type, s_label) in enumerate(subsets_eval):
        center_x = x_pos_temp[s_idx]

        sub_t = temp_sum[(temp_sum["dataset"] == ds) & (temp_sum["subset"] == s_type)]
        c_val = float(sub_t[sub_t["arm"] == "crisp_rfm_fca"]["core_jaccard"].mean())
        f_val = float(sub_t[sub_t["arm"] == "fuzzy_rfm_fca"]["core_jaccard"].mean())
        diff_val = f_val - c_val

        pos_c = center_x - bar_w_temp / 2
        pos_f = center_x + bar_w_temp / 2

        ax3.bar(pos_c, c_val, width=bar_w_temp * 0.90, color=CLR_CRISP, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3)
        ax3.bar(pos_f, f_val, width=bar_w_temp * 0.90, color=CLR_M0, alpha=0.90, edgecolor="#333333", linewidth=0.8, zorder=3)

        # Value labels above bars
        ax3.text(pos_c, c_val + 0.012, f"{c_val:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#222222")
        ax3.text(pos_f, f_val + 0.012, f"{f_val:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#222222")

        # Bracket showing delta
        max_y = max(c_val, f_val)
        bracket_y = max_y + 0.045
        ax3.plot([pos_c, pos_c, pos_f, pos_f], [bracket_y - 0.012, bracket_y, bracket_y, bracket_y - 0.012],
                 color="#333333", linewidth=0.8, zorder=6)
        diff_sign = "+" if diff_val > 0 else ""
        ax3.text(center_x, bracket_y + 0.012, f"Δ = {diff_sign}{diff_val:.3f}",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#222222")

        # Record audit data
        plotted_records.append({
            "figure": "Figure 4", "panel": "(c) Quarterly Re-segmentation Stability", "dataset": ds, "origin": f"Consecutive Quarters ({s_type})",
            "metric": f"Core Jaccard ({s_type})", "metric_key": "core_jaccard", "model_a": "Crisp RFM-FCA", "model_b": "Baseline Fuzzy RFM-FCA (M0)",
            "value_a": c_val, "value_b": f_val, "difference_b_minus_a": diff_val,
            "mean_a": c_val, "mean_b": f_val, "mean_difference": diff_val, "origins_positive": "N/A"
        })

    ax3.set_xticks(x_pos_temp)
    ax3.set_xticklabels([s[2] for s in subsets_eval], fontsize=8.8, fontweight="semibold")
    ax3.set_ylabel("Temporal Core Jaccard (μ ≥ 0.5)", fontsize=10)

    # Common legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, edgecolor="#cccccc", fontsize=9.5)

    # Save vector & high-res PNG
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"figure4_segmentation_stability.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 saved.")



# Run all generators
generate_figure1()
generate_figure2()
generate_figure3()
generate_figure4()

# Save plotted data CSV
df_plotted = pd.DataFrame(plotted_records)
df_plotted.to_csv(OUT_DIR / "plotted_data.csv", index=False)
print(f"Plotted data CSV saved ({len(df_plotted)} records) to {OUT_DIR / 'plotted_data.csv'}")

# Also copy code to the output deliverables directory for complete reproducibility
code_content = Path(__file__).read_text(encoding="utf-8")
(OUT_DIR / "generate_figures.py").write_text(code_content, encoding="utf-8")
print("Reproducible plotting script saved to deliverables directory.")

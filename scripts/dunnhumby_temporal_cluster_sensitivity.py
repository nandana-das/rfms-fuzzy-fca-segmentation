"""Post hoc sensitivity check: Dunnhumby RFM+T - M0 under a household-cluster bootstrap.

The temporal ablation's pooled intervals average independent per-origin bootstraps, although
2,497 of 2,499 households appear in all four Dunnhumby origins. This re-estimates the same three
pooled cells (fuzzy_rfm_t - fuzzy_rfm) with one household resample applied jointly to all origins.

Inputs: validated customer-level predictions from
results/dunnhumby_matched_complexity_paired_inference/ (their per-origin metrics matched the
cached temporal-ablation metrics exactly). No model is refit. This is a sensitivity check of an
existing comparison, not a new test family; the original Holm p-values are reported alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402

PRED = ROOT / "results" / "dunnhumby_matched_complexity_paired_inference" / "customer_predictions.csv.gz"
TEMPORAL = ROOT / "results" / "temporal_rfm_fca_ablation" / "paired_comparisons.csv"
OUT = ROOT / "results" / "dunnhumby_temporal_cluster_sensitivity"
ARM, REF = "fuzzy_rfm_t", "fuzzy_rfm"
COLS = {"auc": ("repurchased", "pred_prob"), "spend_r2": ("y_log_spend", "pred_log_spend"),
        "invoice_r2": ("y_log_invoices", "pred_log_invoices")}


def _metric(metric, y, p, w):
    return roc_auc_score(y, p, sample_weight=w) if metric == "auc" else r2_score(y, p, sample_weight=w)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(PRED, dtype={"CustomerID": str})
    preds = preds[preds.arm.isin([ARM, REF])]
    origins = list(dict.fromkeys(preds.origin))
    households = np.array(sorted(preds.CustomerID.unique()))
    pos = {h: i for i, h in enumerate(households)}
    rng = np.random.default_rng(bl.BOOT_SEED)
    weights = np.stack([np.bincount(d, minlength=len(households))
                        for d in rng.integers(0, len(households), (bl.BOOT_N, len(households)))]).astype(float)
    orig = pd.read_csv(TEMPORAL)
    orig = orig[(orig.dataset == "Dunnhumby") & (orig.scope == "POOLED")
                & (orig.comparison == f"{ARM} - {REF}")].set_index("metric")
    rows = []
    for metric, (ycol, pcol) in COLS.items():
        draws, point = np.zeros(bl.BOOT_N), []
        for origin in origins:
            a = preds[(preds.origin == origin) & (preds.arm == ARM)].sort_values("CustomerID")
            b = preds[(preds.origin == origin) & (preds.arm == REF)].sort_values("CustomerID")
            if not (a.CustomerID.to_numpy() == b.CustomerID.to_numpy()).all():
                raise RuntimeError(f"Customer misalignment in {origin}")
            y, pa, pb = a[ycol].to_numpy(), a[pcol].to_numpy(), b[pcol].to_numpy()
            w_all = weights[:, [pos[h] for h in a.CustomerID]]
            point.append(_metric(metric, y, pa, None) - _metric(metric, y, pb, None))
            for k in range(bl.BOOT_N):
                w = w_all[k]
                keep = w > 0
                draws[k] += (_metric(metric, y[keep], pa[keep], w[keep])
                             - _metric(metric, y[keep], pb[keep], w[keep])) / len(origins)
        o = orig.loc[metric]
        if not np.isclose(np.mean(point), o.delta):
            raise RuntimeError(f"Point delta {np.mean(point)} does not match temporal ablation {o.delta}")
        rows.append({
            "metric": metric, "delta": float(np.mean(point)),
            "original_ci_low": o.ci_low, "original_ci_high": o.ci_high, "original_p_holm": o.p_holm,
            "cluster_ci_low": float(np.percentile(draws, 2.5)), "cluster_ci_high": float(np.percentile(draws, 97.5)),
            "cluster_p_boot": bl._boot_p(draws),
            "cluster_p_bonferroni9_bound": min(1.0, 9 * bl._boot_p(draws)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cluster_vs_independent.csv", index=False)
    lines = [
        "# Dunnhumby RFM+T − M0: household-cluster sensitivity check", "",
        "Post hoc sensitivity check of an existing comparison (not a new test family). One household "
        f"resample (B={bl.BOOT_N}, seed {bl.BOOT_SEED}, {len(households)} households) is applied jointly to "
        "all four origins; statistic = mean over origins of the per-origin difference. Predictions are the "
        "validated regenerated predictions; no model was refit.", "",
        "| metric | delta | original CI | original Holm p | cluster CI | cluster p (raw) | ×9 bound |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for _, r in out.iterrows():
        lines.append(f"| {r.metric} | {r.delta:+.4f} | [{r.original_ci_low:+.4f}, {r.original_ci_high:+.4f}] | "
                     f"{r.original_p_holm:.4f} | [{r.cluster_ci_low:+.4f}, {r.cluster_ci_high:+.4f}] | "
                     f"{r.cluster_p_boot:.4f} | {r.cluster_p_bonferroni9_bound:.4f} |")
    lines += [
        "", "The ×9 bound multiplies the raw cluster p by the size of the original Holm family (a conservative "
        "Bonferroni bound). Intervals reflect test-sample variability only; refitting and fold-assignment "
        "variability are not captured.", "",
        "Command: `python scripts/dunnhumby_temporal_cluster_sensitivity.py`", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

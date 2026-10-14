"""Dunnhumby-only matched-complexity diagnostic for the temporal ablation.

The existing RFM+T mining, membership construction, and Jaccard suppression are
left unchanged.  After suppression, the RFM+T matrix is truncated in its
existing deterministic concept order to the number of M0 features from the
same training fold.  The rule uses no outcomes or test-fold information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from temporal_rfm_fca_ablation import (  # noqa: E402
    _temporal_features,
    temporal_cohorts,
)
from fair_comparison_retail2 import load_cleaned_transactions  # noqa: E402
from fuzzy_membership_sensitivity import TRANSACTION_FILE  # noqa: E402

OUT = ROOT / "results" / "dunnhumby_matched_complexity"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ("auc", "spend_r2", "invoice_r2")
ARMS = ("fuzzy_rfm", "fuzzy_rfm_t", "fuzzy_rfm_t_matched")


def _fit_predict(xtr, xte, tr, seed):
    return bl.fit_predict("fuzzy_rfm_fca", xtr, xte, tr, seed)


def run_origin(cohort: dict):
    df = cohort["df"].reset_index(drop=True)
    oof = {a: {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")} for a in ARMS}
    info_rows = []
    skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
        x_m0_tr, x_m0_te, m0_info = bl.build_features("fuzzy_rfm_fca", tr, te)
        x_t_tr, x_t_te, t_info = _temporal_features("fuzzy_rfm_t", tr, te)
        k = min(x_m0_tr.shape[1], x_t_tr.shape[1])
        # Existing post-suppression order is deterministic and training-only.
        x_match_tr, x_match_te = x_t_tr[:, :k], x_t_te[:, :k]
        for arm, xtr, xte in (
            ("fuzzy_rfm", x_m0_tr, x_m0_te),
            ("fuzzy_rfm_t", x_t_tr, x_t_te),
            ("fuzzy_rfm_t_matched", x_match_tr, x_match_te),
        ):
            p, s, v = _fit_predict(xtr, xte, tr, bl.CV_SEED + fold)
            oof[arm]["prob"][i_te], oof[arm]["sp"][i_te], oof[arm]["inv"][i_te] = p, s, v
        info_rows.append({
            "dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold,
            "m0_features": int(x_m0_tr.shape[1]),
            "rfm_t_features_unpruned": int(x_t_tr.shape[1]),
            "rfm_t_features_matched": int(k),
            "t_imputation_median": t_info["t_imputation_median"],
            "t_missing_train": t_info["t_missing_train"],
            "t_missing_test": t_info["t_missing_test"],
        })
    return oof, info_rows


def main():
    cached = pd.read_csv(ROOT / "results" / "temporal_rfm_fca_ablation" / "per_origin_metrics.csv")
    dh_tx = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = [c for c in temporal_cohorts(dh_tx, load_cleaned_transactions()) if c["dataset"] == "Dunnhumby"]
    metric_rows, info_rows = [], []
    for cohort in cohorts:
        print(f"[{cohort['dataset']} | {cohort['origin']}] n={len(cohort['df'])}", flush=True)
        oof, info = run_origin(cohort)
        info_rows.extend(info)
        full = np.arange(len(cohort["df"]))[None, :]
        for arm in ARMS:
            point = {k: float(v[0]) for k, v in bl.metric_rows(cohort["df"], oof[arm], full).items()}
            metric_rows.append({
                "dataset": cohort["dataset"], "origin": cohort["origin"],
                "pooled": cohort["pooled"], "n": len(cohort["df"]), "arm": arm, **point,
            })
    metrics = pd.DataFrame(metric_rows)
    refs = cached[(cached.dataset == "Dunnhumby") & cached.pooled & cached.arm.isin(["fuzzy_rfm", "fuzzy_rfm_t"])].copy()
    matched = metrics[metrics.arm == "fuzzy_rfm_t_matched"].merge(
        refs[refs.arm == "fuzzy_rfm"], on=["dataset", "origin", "pooled", "n"], suffixes=("", "_m0")
    ).merge(
        refs[refs.arm == "fuzzy_rfm_t"], on=["dataset", "origin", "pooled", "n"], suffixes=("", "_t")
    )
    comparisons = []
    for _, r in matched.iterrows():
        for metric in METRICS:
            comparisons.append({
                "dataset": r["dataset"], "origin": r["origin"], "metric": metric,
                "matched_minus_m0": r[metric] - r[f"{metric}_m0"],
                "matched_minus_rfm_t": r[metric] - r[f"{metric}_t"],
                "paired_ci": "unavailable: cached M0/RFM+T customer predictions not retained",
                "p_holm": "unavailable: no paired customer predictions",
            })
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    pd.DataFrame(info_rows).to_csv(OUT / "feature_counts_per_fold.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "comparisons_to_cached.csv", index=False)
    params = {
        "dataset": "Dunnhumby", "origins": [c["origin"] for c in cohorts],
        "rule": "Run standard RFM+T mining and suppression, then retain first K post-suppression concepts, K = same-fold M0 feature count",
        "selection_is_outcome_independent": True,
        "selection_fit_scope": "training fold only",
        "horizon_days": bl.HORIZON_DAYS, "n_folds": bl.N_FOLDS, "cv_seed": bl.CV_SEED,
        "paired_inference": "not available because cached M0/RFM+T customer predictions are absent",
    }
    (OUT / "run_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    lines = [
        "# Dunnhumby RFM+T matched-complexity diagnostic", "",
        "The existing M0 and unpruned RFM+T results are retained as cached references. The matched arm uses the existing RFM+T concepts after standard training-only mining and Jaccard suppression, then keeps exactly the same number of concepts as M0 in the same fold, preserving post-suppression order.", "",
        "This is an outcome-independent complexity diagnostic, not evidence that feature count alone causes the temporal effect. Paired confidence intervals and Holm tests against cached M0/RFM+T are unavailable because the source artifacts retain metrics but not customer-level predictions.", "",
        "Outputs: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

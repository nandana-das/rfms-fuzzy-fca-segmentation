"""Post hoc paired inference for the Dunnhumby matched-complexity diagnostics.

Phase 1 (regenerate + validate): rebuild customer-level out-of-fold predictions
for four Dunnhumby arms exactly as the source scripts built them, save them, and
compare every per-origin metric with the cached CSVs (tolerance 1e-15). Any
failure stops the script before inference.

  fuzzy_rfm                      M0                      (dunnhumby_matched_complexity.py)
  fuzzy_rfm_t                    unpruned RFM+T          (dunnhumby_matched_complexity.py)
  fuzzy_rfm_t_matched            first-K matched         (dunnhumby_matched_complexity.py)
  fuzzy_rfm_t_stability_matched  stability-ranked K      (dunnhumby_stability_matched_complexity.py;
                                 the saved corrected per-fold selections replace the 50 resamples)

Phase 2 (inference, post hoc): household-cluster bootstrap. Each draw resamples
household IDs once and applies the same draw to all four origins, so a sampled
household keeps all of its origin observations. Statistic = mean over origins of
the per-origin metric difference. Holm over 4 comparisons x 3 metrics = 12 tests.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from dunnhumby_stability_matched_complexity import _build_t_context  # noqa: E402
from fair_comparison_retail2 import load_cleaned_transactions  # noqa: E402
from fuzzy_membership_sensitivity import TRANSACTION_FILE  # noqa: E402
from temporal_rfm_fca_ablation import _temporal_features, temporal_cohorts  # noqa: E402

OUT = ROOT / "results" / "dunnhumby_matched_complexity_paired_inference"
FIRST_K_DIR = ROOT / "results" / "dunnhumby_matched_complexity"
STAB_DIR = ROOT / "results" / "dunnhumby_stability_matched_complexity"
ARMS = ["fuzzy_rfm", "fuzzy_rfm_t", "fuzzy_rfm_t_matched", "fuzzy_rfm_t_stability_matched"]
PAIRS = [
    ("fuzzy_rfm_t_matched", "fuzzy_rfm"),
    ("fuzzy_rfm_t_stability_matched", "fuzzy_rfm"),
    ("fuzzy_rfm_t_matched", "fuzzy_rfm_t"),
    ("fuzzy_rfm_t_stability_matched", "fuzzy_rfm_t"),
]
METRICS = ["auc", "spend_r2", "invoice_r2"]
TOL = 1e-15


def regenerate(cohorts: list[dict], selections: pd.DataFrame):
    pred_rows, metric_rows = [], []
    for cohort in cohorts:
        print(f"[{cohort['origin']}] n={len(cohort['df'])}", flush=True)
        df = cohort["df"].reset_index(drop=True)
        oof = {a: {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")} for a in ARMS}
        fold_of = np.full(len(df), -1)
        skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
        for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
            fold_of[i_te] = fold
            tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
            x_m0_tr, x_m0_te, _ = bl.build_features("fuzzy_rfm_fca", tr, te)
            x_t_tr, x_t_te, _ = _temporal_features("fuzzy_rfm_t", tr, te)
            k = min(x_m0_tr.shape[1], x_t_tr.shape[1])
            context = _build_t_context(tr, te)
            sel = selections[(selections.origin == cohort["origin"]) & (selections.fold == fold) & selections.selected]
            sel = sel.sort_values("rank")["intent"].tolist()
            kept = context["kept"]["intent"].astype(str).tolist()
            missing = [i for i in sel if i not in kept]
            if missing or len(sel) != x_m0_tr.shape[1]:
                raise RuntimeError(f"Saved selection does not fit {cohort['origin']} fold {fold}: "
                                   f"{len(missing)} unknown intents, {len(sel)} selected vs K={x_m0_tr.shape[1]}")
            idx = [kept.index(i) for i in sel]
            feats = {
                "fuzzy_rfm": (x_m0_tr, x_m0_te),
                "fuzzy_rfm_t": (x_t_tr, x_t_te),
                "fuzzy_rfm_t_matched": (x_t_tr[:, :k], x_t_te[:, :k]),
                "fuzzy_rfm_t_stability_matched": (context["x_tr"][:, idx], context["x_te"][:, idx]),
            }
            for arm, (xtr, xte) in feats.items():
                p, s, v = bl.fit_predict("fuzzy_rfm_fca", xtr, xte, tr, bl.CV_SEED + fold)
                oof[arm]["prob"][i_te], oof[arm]["sp"][i_te], oof[arm]["inv"][i_te] = p, s, v
        full = np.arange(len(df))[None, :]
        for arm in ARMS:
            point = {m: float(v[0]) for m, v in bl.metric_rows(df, oof[arm], full).items()}
            metric_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "arm": arm, **point})
            pred_rows.append(pd.DataFrame({
                "dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold_of,
                "CustomerID": df["CustomerID"].astype(str), "arm": arm,
                "repurchased": df["repurchased"].to_numpy(),
                "y_log_spend": np.log1p(df["future_spend"].to_numpy()),
                "y_log_invoices": np.log1p(df["future_invoices"].to_numpy()),
                "pred_prob": oof[arm]["prob"], "pred_log_spend": oof[arm]["sp"], "pred_log_invoices": oof[arm]["inv"],
            }))
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(metric_rows)


def validate(metrics: pd.DataFrame) -> pd.DataFrame:
    fk = pd.read_csv(FIRST_K_DIR / "per_origin_metrics.csv")
    st = pd.read_csv(STAB_DIR / "per_origin_metrics.csv")
    cached = pd.concat([fk, st], ignore_index=True)[["origin", "arm", *METRICS]]
    m = metrics.merge(cached, on=["origin", "arm"], how="outer", suffixes=("", "_cached"), indicator=True)
    rows = []
    for _, r in m.iterrows():
        for metric in METRICS:
            diff = abs(r[metric] - r[f"{metric}_cached"]) if r["_merge"] == "both" else np.nan
            rows.append({"origin": r.origin, "arm": r.arm, "metric": metric, "regenerated": r[metric],
                         "cached": r[f"{metric}_cached"], "abs_diff": diff,
                         "pass": bool(r["_merge"] == "both" and diff <= TOL)})
    return pd.DataFrame(rows)


def _metric(y, p, w, metric):
    return roc_auc_score(y, p, sample_weight=w) if metric == "auc" else r2_score(y, p, sample_weight=w)


def cluster_bootstrap(preds: pd.DataFrame) -> pd.DataFrame:
    origins = list(dict.fromkeys(preds.origin))
    households = np.array(sorted(preds.CustomerID.unique()))
    rng = np.random.default_rng(bl.BOOT_SEED)
    draws = rng.integers(0, len(households), size=(bl.BOOT_N, len(households)))
    # weight of each household in each draw = multiplicity of its index
    weights = np.stack([np.bincount(d, minlength=len(households)) for d in draws]).astype(float)
    hh_pos = {h: i for i, h in enumerate(households)}
    cols = {"auc": ("repurchased", "pred_prob"), "spend_r2": ("y_log_spend", "pred_log_spend"),
            "invoice_r2": ("y_log_invoices", "pred_log_invoices")}
    # stat[(arm, metric)] -> (B, n_origins)
    stat = {(a, m): np.empty((bl.BOOT_N, len(origins))) for a in ARMS for m in METRICS}
    point = {(a, m): np.empty(len(origins)) for a in ARMS for m in METRICS}
    for j, origin in enumerate(origins):
        print(f"bootstrap {origin}", flush=True)
        sub = preds[preds.origin == origin]
        for arm in ARMS:
            s = sub[sub.arm == arm].sort_values("CustomerID")
            w_all = weights[:, [hh_pos[h] for h in s.CustomerID]]
            for metric in METRICS:
                y, p = s[cols[metric][0]].to_numpy(), s[cols[metric][1]].to_numpy()
                point[(arm, metric)][j] = _metric(y, p, None, metric)
                for b in range(bl.BOOT_N):
                    keep = w_all[b] > 0
                    stat[(arm, metric)][b, j] = _metric(y[keep], p[keep], w_all[b][keep], metric)
    rows = []
    for arm, ref in PAIRS:
        for metric in METRICS:
            d_origin = point[(arm, metric)] - point[(ref, metric)]
            d_boot = (stat[(arm, metric)] - stat[(ref, metric)]).mean(axis=1)
            rows.append({"comparison": f"{arm} - {ref}", "metric": metric, "delta": float(d_origin.mean()),
                         "ci_low": float(np.percentile(d_boot, 2.5)), "ci_high": float(np.percentile(d_boot, 97.5)),
                         "p_boot": bl._boot_p(d_boot), "origins_positive": f"{int((d_origin > 0).sum())}/{len(origins)}",
                         **{f"delta_{o.replace(' ', '_')}": float(d) for o, d in zip(origins, d_origin)}})
    out = pd.DataFrame(rows)
    out["p_holm"] = bl._holm(out["p_boot"].tolist())
    out["status"] = "post hoc"
    return out


def write_report(validation: pd.DataFrame, comp: pd.DataFrame | None, n_households: int, runtime: float) -> None:
    v_pass = bool(validation["pass"].all())
    lines = [
        "# Dunnhumby matched-complexity: post hoc paired inference", "",
        "Customer-level predictions for M0, unpruned RFM+T, first-K matched, and stability-ranked matched "
        "RFM+T were regenerated with the source scripts' code paths, folds, seeds, and learners. The "
        "stability-matched arm reused the saved corrected per-fold selections "
        "(`results/dunnhumby_stability_matched_complexity/concept_stability_per_fold.csv`) instead of "
        "repeating the 50 selection resamples.", "",
        "## Validation", "",
        f"Every regenerated per-origin metric was compared with the cached CSVs at tolerance {TOL:g}: "
        f"{int(validation['pass'].sum())}/{len(validation)} passed; max abs difference "
        f"{validation['abs_diff'].max():.3g}. " + ("Inference proceeded." if v_pass else "**Validation failed; no inference was computed.**"),
        "",
    ]
    if comp is not None:
        lines += [
            "## Paired comparisons (post hoc)", "",
            f"Household-cluster bootstrap, B={bl.BOOT_N}, seed {bl.BOOT_SEED}: each draw resamples the "
            f"{n_households} household IDs once and applies the same draw to all four origins. Delta = mean "
            "over origins of (first arm - second arm). Percentile 95% intervals; Holm over the 12 tests.", "",
            "| comparison | metric | delta | CI low | CI high | p Holm | positive origins |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for _, r in comp.iterrows():
            lines.append(f"| {r.comparison} | {r.metric} | {r.delta:+.6f} | {r.ci_low:+.6f} | {r.ci_high:+.6f} | "
                         f"{r.p_holm:.4f} | {r.origins_positive} |")
        lines += [
            "", "## Limitations", "",
            "- Post hoc: neither matched-complexity diagnostic prespecified tests; the family above was fixed before inference was computed.",
            "- The household-cluster bootstrap handles dependence from the same households appearing in all four origins. It reflects test-sample variability only.",
            "- It does not capture model-refitting variability, alternative fold assignments (single CV seed), or uncertainty from choosing the matched-complexity rules after seeing earlier results.",
            "- Matching feature count also changes concept composition, so these comparisons cannot isolate a pure representation-size effect.",
        ]
    lines += ["", "Command: `python scripts/dunnhumby_matched_complexity_paired_inference.py`",
              f"Runtime: {runtime / 60:.1f} minutes.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(STAB_DIR / "concept_stability_per_fold.csv")
    dh = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = [c for c in temporal_cohorts(dh, load_cleaned_transactions()) if c["dataset"] == "Dunnhumby"]
    preds, metrics = regenerate(cohorts, selections)
    preds.to_csv(OUT / "customer_predictions.csv.gz", index=False)
    metrics.to_csv(OUT / "regenerated_per_origin_metrics.csv", index=False)
    validation = validate(metrics)
    validation.to_csv(OUT / "validation.csv", index=False)
    params = {"arms": ARMS, "pairs": [f"{a} - {b}" for a, b in PAIRS], "holm_family": "12 tests (4 pairs x 3 metrics)",
              "validation_tolerance": TOL, "bootstrap": "household-cluster, joint across origins",
              "boot_n": bl.BOOT_N, "boot_seed": bl.BOOT_SEED, "cv_seed": bl.CV_SEED, "status": "post hoc"}
    (OUT / "run_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    if not validation["pass"].all():
        print(validation[~validation["pass"]].to_string(), flush=True)
        write_report(validation, None, 0, time.time() - start)
        sys.exit("Validation failed; inference not computed.")
    comp = cluster_bootstrap(preds)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)
    write_report(validation, comp, preds.CustomerID.nunique(), time.time() - start)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Three-arm fuzzy RFM-FCA temporal ablation.

Adds fold-local T imputation, repeat-history, and origin-safe L while leaving
the committed M0 builder and all locked evaluation settings unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_customer_concept_memberships,
    load_cleaned_transactions,
    quantile_scores,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    _band_centroids_generic,
    _piecewise_membership_generic,
    mine_fuzzy_closed_concepts_with_thresholds,
)

OUT = ROOT / "results" / "temporal_rfm_fca_ablation"
OUT.mkdir(parents=True, exist_ok=True)
RFM_DIMS = ("R", "F", "M")
T_DIMS = ("R", "F", "M", "T", "repeat_history")
TL_DIMS = T_DIMS + ("L",)
ARMS = ["fuzzy_rfm", "fuzzy_rfm_t", "fuzzy_rfm_t_l"]
METRICS = ["auc", "spend_r2", "invoice_r2"]
BOOT_N = bl.BOOT_N


def _pooled_point_deltas(sub: pd.DataFrame, arm: str, ref: str, metric: str,
                         pooled_origins: list[str]) -> list[float]:
    """Return point deltas in the explicit pooled-origin order."""
    return [
        float(sub[(sub["origin"] == origin) & (sub["arm"] == arm)][metric].iloc[0])
        - float(sub[(sub["origin"] == origin) & (sub["arm"] == ref)][metric].iloc[0])
        for origin in pooled_origins
    ]


def _cutoff(cohort: dict) -> int | pd.Timestamp:
    if cohort["dataset"] == "Dunnhumby":
        return int(cohort["origin"].split()[-1])
    date = cohort["origin"].split()[0]
    return pd.Timestamp(date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _temporal_values(tx: pd.DataFrame, cohort: dict, key: str, date_col: str) -> pd.DataFrame:
    work = tx.copy()
    work[key] = work[key].astype(str)
    cutoff = _cutoff(cohort)
    obs = work[work[date_col] <= cutoff].copy()
    grouped = obs.groupby(key, sort=True)[date_col]
    records = []
    for customer, dates in grouped:
        vals = np.sort(pd.unique(dates))
        if np.issubdtype(vals.dtype, np.datetime64):
            nums = vals.astype("datetime64[D]").astype("int64")
            first = pd.Timestamp(vals[0])
            l_value = float((pd.Timestamp(cutoff) - first).days)
        else:
            nums = vals.astype(float)
            l_value = float(float(cutoff) - float(vals[0]))
        gaps = np.diff(nums).astype(float)
        records.append({
            "CustomerID": str(customer),
            "T": float(gaps.mean()) if len(gaps) else np.nan,
            "repeat_history": int(len(vals) >= 2),
            "L": l_value,
        })
    values = pd.DataFrame(records)
    ids = cohort["df"]["CustomerID"].astype(str)
    values = values.set_index("CustomerID") if not values.empty else pd.DataFrame()
    out = cohort["df"].copy()
    out["CustomerID"] = out["CustomerID"].astype(str)
    for col in ("T", "repeat_history", "L"):
        out[col] = ids.map(values[col] if col in values else pd.Series(dtype=float)).to_numpy()
    if out[["repeat_history", "L"]].isna().any().any():
        raise ValueError(f"Missing origin-safe temporal values for {cohort['dataset']} {cohort['origin']}")
    return out


def temporal_cohorts(tx_dh: pd.DataFrame, tx_r2: pd.DataFrame) -> list[dict]:
    dh = bl.dunnhumby_cohorts(tx_dh)
    dh = [{**c, "df": _temporal_values(tx_dh, c, "household_key", "DAY")} for c in dh]
    r2 = bl.retail2_cohorts(tx_r2)
    r2 = [{**c, "df": _temporal_values(tx_r2, c, "CustomerID", "InvoiceDate")} for c in r2]
    return dh + r2


def _merged_cutpoints(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    if len(np.unique(values)) <= 1:
        return np.array([], dtype=float)
    cut = np.unique(np.quantile(values, np.arange(1, 5) / 5))
    while len(cut):
        bands = np.searchsorted(cut, values, side="left")
        counts = np.bincount(bands, minlength=len(cut) + 1)
        empty = np.flatnonzero(counts == 0)
        if not len(empty):
            break
        empty_band = int(empty[0])
        cut = np.delete(cut, empty_band if empty_band < len(cut) else empty_band - 1)
    return cut


def _score_temporal(tr: pd.DataFrame, dims: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    scored = tr[["CustomerID", *dims]].copy()
    cuts = {}
    for dim in dims:
        vals = tr[dim].to_numpy(dtype=float)
        if dim in RFM_DIMS:
            cut = bl.fit_quantile_cutoffs(tr, dims=(dim,), n_levels=5)[dim]
        else:
            cut = _merged_cutpoints(vals)
        cuts[dim] = cut
        scored = quantile_scores(scored, {dim: cut}, dims=(dim,))
    # quantile_scores already inverts R (higher score = more recent). Only T
    # needs an additional inversion because shorter intervals are favourable.
    for dim in ("T",) if "T" in dims else ():
        n_levels = len(cuts[dim]) + 1
        scored[f"{dim}_score"] = (n_levels + 1 - scored[f"{dim}_score"]).astype(np.int8)
    return scored, cuts


def _fuzzy_memberships(scored: pd.DataFrame, dims: tuple[str, ...], centroids=None):
    memberships, fitted = {}, {}
    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        if centroids is None:
            n_levels = int(scored[f"{dim}_score"].max())
            c = _band_centroids_generic(raw, scored[f"{dim}_score"].to_numpy(), n_levels)
        else:
            c = np.asarray(centroids[dim], dtype=float)
        order = np.argsort(c, kind="stable")
        mu = _piecewise_membership_generic(raw, c[order])[:, np.argsort(order, kind="stable")]
        for k in range(len(c)):
            memberships[f"{dim}{k + 1}"] = mu[:, k]
        fitted[dim] = c
    return pd.DataFrame(memberships), fitted


def _temporal_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame):
    dims = RFM_DIMS if arm == "fuzzy_rfm" else T_DIMS if arm == "fuzzy_rfm_t" else TL_DIMS
    tr = tr.copy()
    te = te.copy()
    missing_train = int(tr["T"].isna().sum())
    missing_test = int(te["T"].isna().sum())
    t_median = float(tr["T"].median())
    if not np.isfinite(t_median):
        raise ValueError("Training fold has no observed repeat intervals for T imputation")
    tr["T"] = tr["T"].fillna(t_median)
    te["T"] = te["T"].fillna(t_median)
    scored_tr, cuts = _score_temporal(tr, dims)
    scored_te = te[["CustomerID", *dims]].copy()
    scored_te = quantile_scores(scored_te, cuts, dims=dims)
    # R is already inverted by quantile_scores; only T needs inversion here.
    for dim in ("T",) if "T" in dims else ():
        n_levels = len(cuts[dim]) + 1
        scored_te[f"{dim}_score"] = (n_levels + 1 - scored_te[f"{dim}_score"]).astype(np.int8)
    mu_tr, centroids = _fuzzy_memberships(scored_tr, dims)
    mu_te, _ = _fuzzy_memberships(scored_te, dims, centroids=centroids)
    raw, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        mu_tr, scored_tr, bl.L_THRESHOLDS, min_support=bl.MIN_SUPPORT, dims=dims
    )
    raw = raw.reset_index(drop=True)
    mu_all = compute_customer_concept_memberships(raw, mu_tr)
    kept = suppress_redundant_concepts(
        raw, mu_all, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True
    ).reset_index(drop=True)
    x_tr = compute_customer_concept_memberships(kept, mu_tr)
    x_te = compute_customer_concept_memberships(kept, mu_te)
    return x_tr, x_te, {
        "n_candidates": len(raw), "n_features": x_tr.shape[1],
        "n_temporal_levels": {d: int(scored_tr[f"{d}_score"].max()) for d in dims if d not in RFM_DIMS},
        "t_imputation_median": t_median,
        "t_missing_train": missing_train,
        "t_missing_test": missing_test,
    }


def _fit_predict(arm: str, xtr, xte, tr: pd.DataFrame, seed: int):
    y_rep = tr["repurchased"].to_numpy()
    y_sp = np.log1p(tr["future_spend"].to_numpy())
    y_inv = np.log1p(tr["future_invoices"].to_numpy())
    clf = LogisticRegressionCV(Cs=bl.LOGREG_CS, cv=5, scoring="roc_auc", solver="lbfgs",
                               max_iter=bl.LOGREG_MAX_ITER, random_state=seed).fit(xtr, y_rep)
    r1 = RidgeCV(alphas=bl.RIDGE_ALPHAS, cv=5).fit(xtr, y_sp)
    r2 = RidgeCV(alphas=bl.RIDGE_ALPHAS, cv=5).fit(xtr, y_inv)
    return clf.predict_proba(xte)[:, 1], r1.predict(xte), r2.predict(xte)


def run_origin(cohort: dict):
    df = cohort["df"].reset_index(drop=True)
    oof = {a: {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")} for a in ARMS}
    info_rows = []
    skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
        for arm in ARMS:
            if arm == "fuzzy_rfm":
                xtr, xte, info = bl.build_features("fuzzy_rfm_fca", tr, te)
            else:
                xtr, xte, info = _temporal_features(arm, tr, te)
            p, s, v = _fit_predict(arm, xtr, xte, tr, bl.CV_SEED + fold)
            oof[arm]["prob"][i_te], oof[arm]["sp"][i_te], oof[arm]["inv"][i_te] = p, s, v
            info_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold, "arm": arm, **info})
    return oof, info_rows


def _write_report(metrics: pd.DataFrame, comparisons: pd.DataFrame, info: pd.DataFrame, runtime: float):
    means = metrics[metrics["pooled"]].groupby(["dataset", "arm"])[METRICS].mean().reset_index()
    lines = [
        "# Temporal RFM-FCA ablation",
        "",
        "Three-arm rolling-origin ablation using only origin-safe temporal features. Existing Step 0 outputs were not modified.",
        "",
        "## Definitions and leakage rules",
        "",
        "- R = days since last observed purchase; F = unique invoices/baskets; M = observed monetary value.",
        "- T = mean interval between consecutive distinct observed purchase dates. T is missing when fewer than two distinct dates exist; missing T is imputed with the training-fold median only.",
        "- `repeat_history` = 1 when at least two distinct observed purchase dates exist, otherwise 0. The T arm therefore includes both interval and repeat-history information.",
        "- L = days from first observed purchase date to the prediction origin.",
        "- Future outcomes use only the subsequent 91-day holdout. Cutpoints, centroids, concept mining, redundancy suppression, imputation, and learners are fit inside each customer training fold.",
        "",
        "## Arms",
        "",
        "| Arm | Fuzzy attributes |",
        "|---|---|",
        "| Fuzzy RFM | R, F, M; existing M0 builder |",
        "| Fuzzy RFM+T | R, F, M, T, repeat_history |",
        "| Fuzzy RFM+T+L | R, F, M, T, repeat_history, L |",
        "",
        "## Pooled means",
        "",
        "| dataset | arm | auc | spend_r2 | invoice_r2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in means.iterrows():
        lines.append(f"| {r.dataset} | {r.arm} | {r.auc:.4f} | {r.spend_r2:.4f} | {r.invoice_r2:.4f} |")
    lines += ["", "## Per-origin metrics", "", "The complete per-origin table is also written to `per_origin_metrics.csv`.", "", "| dataset | origin | arm | auc | spend_r2 | invoice_r2 |", "|---|---|---|---:|---:|---:|"]
    for _, r in metrics.iterrows():
        lines.append(f"| {r.dataset} | {r.origin} | {r.arm} | {r.auc:.4f} | {r.spend_r2:.4f} | {r.invoice_r2:.4f} |")
    lines += ["", "## Paired comparisons", "", "Delta = later arm minus earlier arm; intervals are paired customer-bootstrap percentile intervals and p-values are Holm-adjusted within dataset across the nine pooled comparison tests (three arm comparisons × three metrics).", "", "| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |", "|---|---|---|---:|---:|---:|---:|---|"]
    for _, r in comparisons[comparisons.scope == "POOLED"].iterrows():
        lines.append(f"| {r.dataset} | {r.comparison} | {r.metric} | {r.delta:+.4f} | {r.ci_low:+.4f} | {r.ci_high:+.4f} | {r.p_holm:.4f} | {r.origins_positive} |")
    pooled = comparisons[comparisons.scope == "POOLED"]
    def _row(dataset: str, comparison: str, metric: str):
        return pooled[(pooled.dataset == dataset) & (pooled.comparison == comparison) & (pooled.metric == metric)].iloc[0]
    dh_t_spend = _row("Dunnhumby", "fuzzy_rfm_t - fuzzy_rfm", "spend_r2")
    r2_t_auc = _row("Online Retail II", "fuzzy_rfm_t - fuzzy_rfm", "auc")
    r2_t_spend = _row("Online Retail II", "fuzzy_rfm_t - fuzzy_rfm", "spend_r2")
    r2_t_invoice = _row("Online Retail II", "fuzzy_rfm_t - fuzzy_rfm", "invoice_r2")
    r2_l_invoice = _row("Online Retail II", "fuzzy_rfm_t_l - fuzzy_rfm_t", "invoice_r2")
    conclusion = (
        "Under this protocol, T has opposite dataset-specific effects: "
        f"Dunnhumby Spend R² changes by {dh_t_spend.delta:+.4f} (Holm p={dh_t_spend.p_holm:.4f}), "
        f"while Online Retail II changes are {r2_t_auc.delta:+.4f} for AUC (p={r2_t_auc.p_holm:.4f}), "
        f"{r2_t_spend.delta:+.4f} for Spend R² (p={r2_t_spend.p_holm:.4f}), and "
        f"{r2_t_invoice.delta:+.4f} for Invoice R² (p={r2_t_invoice.p_holm:.4f}). "
        f"Adding L after T changes Online Retail II Invoice R² by {r2_l_invoice.delta:+.4f} "
        f"(Holm p={r2_l_invoice.p_holm:.4f}). Therefore T and L do not provide a uniform predictive improvement, "
        "and the result is exploratory and limited to these datasets and learners."
    )
    lines += [
        "", "## Conclusion", "", conclusion,
        "", "## Outputs and command", "", "- Command: `python scripts\\temporal_rfm_fca_ablation.py`", "- Outputs: `results/temporal_rfm_fca_ablation/`", "- Existing Step 0 outputs and manuscript files were not changed.", f"- Runtime: {runtime / 60:.1f} minutes.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.time()
    print("Loading transactions...", flush=True)
    dh = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2 = load_cleaned_transactions()
    cohorts = temporal_cohorts(dh, r2)
    rng = np.random.default_rng(bl.BOOT_SEED)
    metric_rows, info_rows, boots = [], [], {}
    for cohort in cohorts:
        print(f"[{cohort['dataset']} | {cohort['origin']}] n={len(cohort['df'])}", flush=True)
        oof, info = run_origin(cohort)
        info_rows.extend(info)
        full = np.arange(len(cohort["df"]))[None, :]
        boot_idx = rng.integers(0, len(cohort["df"]), size=(BOOT_N, len(cohort["df"])))
        for arm in ARMS:
            point = {k: float(v[0]) for k, v in bl.metric_rows(cohort["df"], oof[arm], full).items()}
            metric_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "pooled": cohort["pooled"], "n": len(cohort["df"]), "arm": arm, **point})
            boots[(cohort["dataset"], cohort["origin"], arm)] = bl.metric_rows(cohort["df"], oof[arm], boot_idx)
    metrics = pd.DataFrame(metric_rows)
    comparisons = []
    pairs = [("fuzzy_rfm_t", "fuzzy_rfm"), ("fuzzy_rfm_t_l", "fuzzy_rfm_t"), ("fuzzy_rfm_t_l", "fuzzy_rfm")]
    for dataset in metrics.dataset.unique():
        sub = metrics[metrics.dataset == dataset]
        origins = list(dict.fromkeys(sub.origin))
        pooled_origins = list(dict.fromkeys(sub.loc[sub.pooled, "origin"]))
        for arm, ref in pairs:
            for metric in METRICS:
                draws_by_origin, deltas = {}, []
                for origin in origins:
                    a = sub[(sub.origin == origin) & (sub.arm == arm)][metric].iloc[0]
                    b = sub[(sub.origin == origin) & (sub.arm == ref)][metric].iloc[0]
                    d = boots[(dataset, origin, arm)][metric] - boots[(dataset, origin, ref)][metric]
                    draws_by_origin[origin] = d
                    deltas.append(a - b)
                    comparisons.append({"dataset": dataset, "scope": origin, "comparison": f"{arm} - {ref}", "metric": metric, "delta": a - b, "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5), "p_boot": bl._boot_p(d)})
                pooled_draws = np.mean([draws_by_origin[o] for o in pooled_origins], axis=0)
                pooled_deltas = _pooled_point_deltas(sub, arm, ref, metric, pooled_origins)
                comparisons.append({"dataset": dataset, "scope": "POOLED", "comparison": f"{arm} - {ref}", "metric": metric, "delta": np.mean(pooled_deltas), "ci_low": np.percentile(pooled_draws, 2.5), "ci_high": np.percentile(pooled_draws, 97.5), "p_boot": bl._boot_p(pooled_draws), "origins_positive": f"{sum(d > 0 for d in pooled_deltas)}/{len(pooled_deltas)}"})
    comp = pd.DataFrame(comparisons)
    comp["p_holm"] = np.nan
    for dataset in comp.dataset.unique():
        mask = (comp.dataset == dataset) & (comp.scope == "POOLED")
        comp.loc[mask, "p_holm"] = bl._holm(comp.loc[mask, "p_boot"].tolist())
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(info_rows).to_csv(OUT / "concept_counts_per_fold.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({"arms": ARMS, "dims": {"m0": RFM_DIMS, "t": T_DIMS, "t_l": TL_DIMS}, "horizon_days": bl.HORIZON_DAYS, "n_folds": bl.N_FOLDS, "cv_seed": bl.CV_SEED, "boot_n": BOOT_N, "boot_seed": bl.BOOT_SEED, "holm_family": "nine pooled tests per dataset: three comparisons by three metrics", "definitions": {"T": "mean gaps between consecutive distinct observed purchase dates", "L": "origin minus first observed purchase date", "repeat_history": "at least two distinct observed purchase dates"}}, indent=2), encoding="utf-8")
    _write_report(metrics, comp, pd.DataFrame(info_rows), time.time() - start)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

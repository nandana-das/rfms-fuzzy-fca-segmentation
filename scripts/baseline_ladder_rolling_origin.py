"""Baseline ladder with rolling-origin temporal evaluation (audit follow-up).

Question answered: how much of the predictive gain attributed to fuzzy RFM-FCA
remains once Raw RFM is given a correctly specified (nonlinear) representation?

Ladder (same downstream models for every linear arm):
    raw_std          standardized raw R/F/M (the baseline used in earlier studies)
    log_rfm          standardized log1p(R/F/M)
    spline_log_rfm   cubic B-spline basis (5 quantile knots) on log1p(R/F/M)
    gbm_raw_rfm      untuned histogram gradient boosting on raw R/F/M (reference)
    crisp_bands      15 one-hot quintile bands (no FCA)
    crisp_rfm_fca    crisp closed concepts, support >= 0.04 (base-paper analogue)
    fuzzy_bands      15 fuzzy band memberships (no FCA)
    fuzzy_rfm_fca    fuzzy concepts + Jaccard suppression (empty-core fix applied)
    fuzzy_rfm_fca_denserank  same, but with the superseded dense-rank scoring
                     (kept only to quantify the effect of the scoring change)
    kuznetsov_fca    fuzzy concepts filtered by 1 - float(stability) <= 5.4e-20
                     (the previously reported rule, pre-specified here) + Jaccard

Scoring: tie-preserving quintiles. Cutpoints at the 20/40/60/80th percentiles
are fit on the training folds; band k holds values in (c_{k-1}, c_k], so identical
raw values always share a band (bands are unequal only at mass points). Band
occupancy is written to band_occupancy.csv and the report.

Protocol:
    * Several temporal origins per dataset, each with a 91-day holdout; RFM is
      computed from all transactions up to the origin. Online Retail II also keeps
      its original primary protocol (2010-12-09 cutoff, 365-day holdout), reported
      separately and excluded from the pooled estimate.
    * Within each origin: stratified 5-fold CV over customers. Every
      representation (scaling, splines, bands, centroids, concept mining,
      stability, suppression) is fit on the training folds only. Metrics are
      computed on the pooled out-of-fold predictions (every customer scored once).
    * Comparisons: paired customer-level bootstrap (B = 2000) of the metric
      difference within each origin; pooled estimate = mean over origins with
      independent per-origin resampling. Holm correction over the pre-specified
      comparison family within each dataset.
    * The bootstrap captures test-sample variability only (not refitting
      variability); origins overlap in customers and history, so the pooled
      interval should be read as approximate. Per-origin sign consistency is
      reported alongside.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    aggregate_rfm,
    compute_customer_concept_memberships,
    compute_fuzzy_memberships,
    dense_rank_scores,
    fit_quantile_cutoffs,
    load_cleaned_transactions,
    quantile_scores,
    mine_crisp_closed_concepts,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    mine_fuzzy_closed_concepts_with_thresholds,
)
from kuznetsov_pruning_leakage_free import compute_split_stability  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EXP_DIR = ROOT_DIR / "results" / "baseline_ladder_rolling_origin"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Locked parameters (pre-specified; nothing below is tuned on outcomes)
# ---------------------------------------------------------------------------

DIMS = ("R", "F", "M")
HORIZON_DAYS = 91
N_FOLDS = 5
CV_SEED = 0
BOOT_N = 2000
BOOT_SEED = 12345

MIN_SUPPORT = 0.04
L_THRESHOLDS = (0.3, 0.5, 0.7)
J_MAX = 0.80
MU_CUT = 0.50
KUZ_LOSS_THRESHOLD = 5.4e-20  # previously reported rule; equals float(stability) == 1.0

LOGREG_CS = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)
GBM_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)

DH_ORIGINS = [347, 438, 529, 620]  # day cutoffs; holdout = next 91 days
R2_END = pd.Timestamp("2011-12-09 23:59:59")
R2_ORIGINS = [R2_END - pd.Timedelta(days=HORIZON_DAYS * k) for k in (5, 4, 3, 2, 1)]
R2_PRIMARY_CUTOFF = pd.Timestamp("2010-12-09 23:59:59")

ARMS = [
    "raw_std", "log_rfm", "spline_log_rfm", "gbm_raw_rfm",
    "crisp_bands", "crisp_rfm_fca", "fuzzy_bands",
    "fuzzy_rfm_fca", "fuzzy_rfm_fca_denserank", "kuznetsov_fca",
]
METRICS = ["auc", "spend_r2", "invoice_r2"]

# (arm, reference) pairs; difference = arm - reference
COMPARISONS = [
    ("fuzzy_rfm_fca", "raw_std"),
    ("fuzzy_rfm_fca", "log_rfm"),
    ("fuzzy_rfm_fca", "spline_log_rfm"),
    ("fuzzy_rfm_fca", "gbm_raw_rfm"),
    ("fuzzy_rfm_fca", "crisp_rfm_fca"),
    ("fuzzy_rfm_fca", "fuzzy_bands"),
    ("fuzzy_rfm_fca", "fuzzy_rfm_fca_denserank"),
    ("kuznetsov_fca", "fuzzy_rfm_fca"),
    ("kuznetsov_fca", "spline_log_rfm"),
]


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# Cohorts
# ===========================================================================

def dunnhumby_cohorts(tx: pd.DataFrame) -> list[dict]:
    out = []
    for cutoff in DH_ORIGINS:
        obs = tx[tx["DAY"] <= cutoff]
        hold = tx[(tx["DAY"] > cutoff) & (tx["DAY"] <= cutoff + HORIZON_DAYS)]
        agg = obs.groupby("household_key").agg(
            last_day=("DAY", "max"), F=("BASKET_ID", "nunique"), M=("SALES_VALUE", "sum")
        ).reset_index()
        agg["R"] = cutoff - agg["last_day"]
        fut = hold.groupby("household_key").agg(
            future_invoices=("BASKET_ID", "nunique"), future_spend=("SALES_VALUE", "sum")
        ).reset_index()
        df = agg.merge(fut, on="household_key", how="left")
        df["CustomerID"] = df["household_key"].astype(str)
        out.append(_finish("Dunnhumby", f"day {cutoff}", df, pooled=True))
    return out


def retail2_cohorts(clean: pd.DataFrame) -> list[dict]:
    out = []
    specs = [(c, c + pd.Timedelta(days=HORIZON_DAYS), True) for c in R2_ORIGINS]
    specs.append((R2_PRIMARY_CUTOFF, R2_END, False))
    for cutoff, hold_end, pooled in specs:
        obs = clean[clean["InvoiceDate"] <= cutoff]
        hold = clean[(clean["InvoiceDate"] > cutoff) & (clean["InvoiceDate"] <= hold_end)]
        df = aggregate_rfm(obs, reference_date=cutoff)
        fut = hold.groupby("CustomerID").agg(
            future_invoices=("InvoiceNo", "nunique"), future_spend=("line_value", "sum")
        ).reset_index()
        df = df.merge(fut, on="CustomerID", how="left")
        label = f"{cutoff.date()}" + ("" if pooled else " (primary, 365d)")
        out.append(_finish("Online Retail II", label, df, pooled=pooled))
    return out


def _finish(dataset: str, label: str, df: pd.DataFrame, pooled: bool) -> dict:
    df["future_invoices"] = df["future_invoices"].fillna(0).astype(int)
    df["future_spend"] = df["future_spend"].fillna(0.0).clip(lower=0.0)
    df["repurchased"] = (df["future_invoices"] > 0).astype(int)
    df = df[["CustomerID", "R", "F", "M", "future_invoices", "future_spend", "repurchased"]]
    return {"dataset": dataset, "origin": label, "pooled": pooled, "df": df.reset_index(drop=True)}


# ===========================================================================
# Representations (all fit on training folds only)
# ===========================================================================

def _log(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[list(DIMS)].to_numpy(dtype=float))


def _quintile_scored(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cut = fit_quantile_cutoffs(tr, dims=DIMS, n_levels=5)
    return quantile_scores(tr[["CustomerID", *DIMS]].copy(), cut, dims=DIMS), cut


def _fuzzy_context(tr: pd.DataFrame, te: pd.DataFrame, scoring: str = "quintile"):
    if scoring == "quintile":
        scored, _ = _quintile_scored(tr)
    else:
        scored = dense_rank_scores(tr[["CustomerID", *DIMS]].copy(), dims=DIMS)
    mu_tr, centroids, _ = compute_fuzzy_memberships(scored, dims=DIMS)
    mu_te, _, _ = compute_fuzzy_memberships(
        te[["CustomerID", *DIMS]].copy(), trained_centroids=centroids, dims=DIMS
    )
    return scored, mu_tr, mu_te


def _crisp_bands(tr: pd.DataFrame, te: pd.DataFrame):
    scored, cut = _quintile_scored(tr)
    te_scored = quantile_scores(te[["CustomerID", *DIMS]].copy(), cut, dims=DIMS)
    b_tr, b_te = {}, {}
    for dim in DIMS:
        s_te = te_scored[f"{dim}_score"].to_numpy()
        for k in range(1, 6):
            b_tr[f"{dim}{k}"] = (scored[f"{dim}_score"].to_numpy() == k).astype(float)
            b_te[f"{dim}{k}"] = (s_te == k).astype(float)
    return scored, pd.DataFrame(b_tr), pd.DataFrame(b_te)


def _concept_info(concepts: pd.DataFrame, X_tr: np.ndarray, n_candidates: int) -> dict:
    distinct = int(np.unique(np.round(X_tr, 12), axis=1).shape[1]) if X_tr.shape[1] else 0
    empty_core = int(((X_tr >= MU_CUT).sum(axis=0) == 0).sum()) if X_tr.shape[1] else 0
    return {"n_candidates": n_candidates, "n_features": int(X_tr.shape[1]),
            "n_distinct_columns": distinct, "n_empty_core": empty_core}


def build_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    if arm == "raw_std":
        sc = StandardScaler().fit(tr[list(DIMS)].to_numpy(dtype=float))
        return sc.transform(tr[list(DIMS)].to_numpy(dtype=float)), sc.transform(te[list(DIMS)].to_numpy(dtype=float)), {}
    if arm == "log_rfm":
        sc = StandardScaler().fit(_log(tr))
        return sc.transform(_log(tr)), sc.transform(_log(te)), {}
    if arm == "spline_log_rfm":
        sp = SplineTransformer(n_knots=5, degree=3, knots="quantile").fit(_log(tr))
        return sp.transform(_log(tr)), sp.transform(_log(te)), {}
    if arm == "crisp_bands":
        _, b_tr, b_te = _crisp_bands(tr, te)
        return b_tr.to_numpy(), b_te.to_numpy(), {}
    if arm == "crisp_rfm_fca":
        scored, b_tr, b_te = _crisp_bands(tr, te)
        concepts = mine_crisp_closed_concepts(scored, min_support=MIN_SUPPORT, dims=DIMS)
        concepts = concepts[concepts["intent_size"] > 0].reset_index(drop=True)
        X_tr = compute_customer_concept_memberships(concepts, b_tr)
        X_te = compute_customer_concept_memberships(concepts, b_te)
        return X_tr, X_te, _concept_info(concepts, X_tr, len(concepts))
    if arm == "fuzzy_bands":
        _, mu_tr, mu_te = _fuzzy_context(tr, te)
        return mu_tr.to_numpy(), mu_te.to_numpy(), {}
    if arm in ("fuzzy_rfm_fca", "fuzzy_rfm_fca_denserank", "kuznetsov_fca"):
        scored, mu_tr, mu_te = _fuzzy_context(
            tr, te, scoring="dense_rank" if arm == "fuzzy_rfm_fca_denserank" else "quintile"
        )
        raw, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
            mu_tr, scored, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
        )
        raw = raw.reset_index(drop=True)
        mu_all = compute_customer_concept_memberships(raw, mu_tr)
        info = {}
        if arm == "kuznetsov_fca":
            stab, _ = compute_split_stability(raw, mu_tr, L_THRESHOLDS)
            stab = stab.set_index("concept_index")
            keep = [
                i for i in range(len(raw))
                if str(stab.at[i, "status"]).startswith("exact")
                and (1.0 - float(stab.at[i, "stab_float"])) <= KUZ_LOSS_THRESHOLD
            ]
            info["n_after_stability"] = len(keep)
            pool, mu_pool = raw.iloc[keep].reset_index(drop=True), mu_all[:, keep]
        else:
            pool, mu_pool = raw, mu_all
        kept = suppress_redundant_concepts(
            pool, mu_pool, j_max=J_MAX, mu_cut=MU_CUT, drop_empty_core=True
        ).reset_index(drop=True)
        X_tr = compute_customer_concept_memberships(kept, mu_tr)
        X_te = compute_customer_concept_memberships(kept, mu_te)
        info.update(_concept_info(kept, X_tr, len(raw)))
        return X_tr, X_te, info
    raise ValueError(arm)


def band_occupancy(cohort: dict) -> list[dict]:
    """Descriptive band occupancy on the full cohort for both scoring schemes.

    Evaluation never uses these full-cohort bands; inside CV the cutpoints are
    refit on each training fold.
    """
    df = cohort["df"]
    schemes = {
        "quintile": _quintile_scored(df)[0],
        "dense_rank": dense_rank_scores(df[["CustomerID", *DIMS]].copy(), dims=DIMS),
    }
    rows = []
    for scheme, scored in schemes.items():
        for dim in DIMS:
            for k in range(1, 6):
                vals = scored.loc[scored[f"{dim}_score"] == k, dim]
                rows.append({
                    "dataset": cohort["dataset"], "origin": cohort["origin"], "scheme": scheme,
                    "dimension": dim, "band": k, "n": int(len(vals)),
                    "pct": float(len(vals) / len(scored)),
                    "raw_min": float(vals.min()) if len(vals) else np.nan,
                    "raw_max": float(vals.max()) if len(vals) else np.nan,
                    "median_centroid": float(vals.median()) if len(vals) else np.nan,
                })
    return rows


# ===========================================================================
# Models
# ===========================================================================

def fit_predict(arm: str, X_tr, X_te, tr: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_rep = tr["repurchased"].to_numpy()
    y_sp = np.log1p(tr["future_spend"].to_numpy())
    y_inv = np.log1p(tr["future_invoices"].to_numpy())
    if arm == "gbm_raw_rfm":
        Xa = tr[list(DIMS)].to_numpy(dtype=float)
        clf = HistGradientBoostingClassifier(random_state=seed, **GBM_PARAMS).fit(Xa, y_rep)
        r1 = HistGradientBoostingRegressor(random_state=seed, **GBM_PARAMS).fit(Xa, y_sp)
        r2 = HistGradientBoostingRegressor(random_state=seed, **GBM_PARAMS).fit(Xa, y_inv)
        return clf.predict_proba(X_te)[:, 1], r1.predict(X_te), r2.predict(X_te)
    clf = LogisticRegressionCV(Cs=LOGREG_CS, cv=5, scoring="roc_auc", solver="lbfgs",
                               max_iter=LOGREG_MAX_ITER, random_state=seed).fit(X_tr, y_rep)
    r1 = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_sp)
    r2 = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_inv)
    return clf.predict_proba(X_te)[:, 1], r1.predict(X_te), r2.predict(X_te)


def run_origin(cohort: dict) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    df = cohort["df"]
    n = len(df)
    oof = {a: {m: np.full(n, np.nan) for m in ("prob", "sp", "inv")} for a in ARMS}
    info_rows = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
        for arm in ARMS:
            if arm == "gbm_raw_rfm":
                X_tr, X_te, info = None, te[list(DIMS)].to_numpy(dtype=float), {}
            else:
                X_tr, X_te, info = build_features(arm, tr, te)
            p, s, v = fit_predict(arm, X_tr, X_te, tr, seed=CV_SEED + fold)
            oof[arm]["prob"][i_te], oof[arm]["sp"][i_te], oof[arm]["inv"][i_te] = p, s, v
            if info:
                info_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"],
                                  "fold": fold, "arm": arm, **info})
    return oof, info_rows


# ===========================================================================
# Metrics and paired bootstrap
# ===========================================================================

def _auc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise ROC AUC for (B, n) label/score matrices via the rank-sum identity."""
    r = rankdata(p, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    return ((r * y).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _r2_rows(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    ss_res = ((y - f) ** 2).sum(axis=1)
    ss_tot = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - ss_res / ss_tot


def metric_rows(df: pd.DataFrame, pred: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    y_rep = df["repurchased"].to_numpy()[idx]
    y_sp = np.log1p(df["future_spend"].to_numpy())[idx]
    y_inv = np.log1p(df["future_invoices"].to_numpy())[idx]
    return {
        "auc": _auc_rows(y_rep, pred["prob"][idx]),
        "spend_r2": _r2_rows(y_sp, pred["sp"][idx]),
        "invoice_r2": _r2_rows(y_inv, pred["inv"][idx]),
    }


def _boot_p(draws: np.ndarray) -> float:
    p = 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))
    return float(min(1.0, max(p, 1.0 / len(draws))))


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t0 = time.time()
    _pr("Loading Dunnhumby transactions...")
    dh_tx = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    _pr("Loading Online Retail II transactions...")
    r2_tx = load_cleaned_transactions()
    cohorts = dunnhumby_cohorts(dh_tx) + retail2_cohorts(r2_tx)

    occupancy_df = pd.DataFrame([r for c in cohorts for r in band_occupancy(c)])
    occupancy_df.to_csv(EXP_DIR / "band_occupancy.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED)
    metric_records, info_records, boot_store = [], [], {}
    for c in cohorts:
        df = c["df"]
        _pr(f"[{c['dataset']} | {c['origin']}] n={len(df):,} repurchase={df['repurchased'].mean():.4f} "
            f"negatives={int((df['repurchased'] == 0).sum())}")
        ts = time.time()
        oof, info = run_origin(c)
        info_records.extend(info)
        full = np.arange(len(df))[None, :]
        boot_idx = rng.integers(0, len(df), size=(BOOT_N, len(df)))
        for arm in ARMS:
            point = {k: float(v[0]) for k, v in metric_rows(df, oof[arm], full).items()}
            metric_records.append({"dataset": c["dataset"], "origin": c["origin"], "pooled": c["pooled"],
                                   "n": len(df), "repurchase_rate": float(df["repurchased"].mean()),
                                   "arm": arm, **point})
            boot_store[(c["dataset"], c["origin"], arm)] = metric_rows(df, oof[arm], boot_idx)
        _pr(f"    done in {time.time() - ts:.0f}s")

    metrics_df = pd.DataFrame(metric_records)
    info_df = pd.DataFrame(info_records)

    # Paired comparisons: per origin, and pooled over the 91-day origins.
    comp_records = []
    for dataset in metrics_df["dataset"].unique():
        sub = metrics_df[metrics_df["dataset"] == dataset]
        origins = list(dict.fromkeys(sub["origin"]))
        pooled_origins = list(dict.fromkeys(sub.loc[sub["pooled"], "origin"]))
        for arm, ref in COMPARISONS:
            for metric in METRICS:
                per_origin = {}
                for o in origins:
                    a = sub[(sub["origin"] == o) & (sub["arm"] == arm)][metric].iloc[0]
                    b = sub[(sub["origin"] == o) & (sub["arm"] == ref)][metric].iloc[0]
                    draws = boot_store[(dataset, o, arm)][metric] - boot_store[(dataset, o, ref)][metric]
                    per_origin[o] = draws
                    comp_records.append({
                        "dataset": dataset, "scope": o, "comparison": f"{arm} - {ref}", "metric": metric,
                        "delta": a - b, "ci_low": float(np.percentile(draws, 2.5)),
                        "ci_high": float(np.percentile(draws, 97.5)), "p_boot": _boot_p(draws),
                    })
                pooled_draws = np.mean([per_origin[o] for o in pooled_origins], axis=0)
                deltas = [
                    sub[(sub["origin"] == o) & (sub["arm"] == arm)][metric].iloc[0]
                    - sub[(sub["origin"] == o) & (sub["arm"] == ref)][metric].iloc[0]
                    for o in pooled_origins
                ]
                comp_records.append({
                    "dataset": dataset, "scope": "POOLED", "comparison": f"{arm} - {ref}", "metric": metric,
                    "delta": float(np.mean(deltas)), "ci_low": float(np.percentile(pooled_draws, 2.5)),
                    "ci_high": float(np.percentile(pooled_draws, 97.5)), "p_boot": _boot_p(pooled_draws),
                    "origins_positive": f"{sum(d > 0 for d in deltas)}/{len(deltas)}",
                })
    comp_df = pd.DataFrame(comp_records)
    comp_df["p_holm"] = np.nan
    for dataset in comp_df["dataset"].unique():
        mask = (comp_df["dataset"] == dataset) & (comp_df["scope"] == "POOLED")
        comp_df.loc[mask, "p_holm"] = _holm(comp_df.loc[mask, "p_boot"].tolist())

    metrics_df.to_csv(EXP_DIR / "per_origin_metrics.csv", index=False)
    comp_df.to_csv(EXP_DIR / "paired_comparisons.csv", index=False)
    info_df.to_csv(EXP_DIR / "concept_counts_per_fold.csv", index=False)
    params = {
        "dims": DIMS, "horizon_days": HORIZON_DAYS, "n_folds": N_FOLDS, "cv_seed": CV_SEED,
        "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "min_support": MIN_SUPPORT,
        "l_thresholds": L_THRESHOLDS, "j_max": J_MAX, "mu_cut": MU_CUT,
        "kuznetsov_loss_threshold": KUZ_LOSS_THRESHOLD,
        "logreg": f"LogisticRegressionCV(Cs={LOGREG_CS}, cv=5, roc_auc, lbfgs, max_iter={LOGREG_MAX_ITER})",
        "ridge": "RidgeCV(alphas=logspace(-3,3,20), cv=5)", "gbm": GBM_PARAMS,
        "dunnhumby_origins_day": DH_ORIGINS,
        "retail2_origins": [str(o) for o in R2_ORIGINS],
        "retail2_primary": {"cutoff": str(R2_PRIMARY_CUTOFF), "holdout_end": str(R2_END)},
        "comparisons": [f"{a} - {b}" for a, b in COMPARISONS],
    }
    (EXP_DIR / "run_parameters.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    write_report(metrics_df, comp_df, info_df, occupancy_df, time.time() - t0)
    _pr(f"Total runtime {(time.time() - t0) / 60:.1f} min. Outputs: {EXP_DIR}")


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(metrics_df: pd.DataFrame, comp_df: pd.DataFrame, info_df: pd.DataFrame,
                 occupancy_df: pd.DataFrame, runtime: float) -> None:
    lines = [
        "# Baseline Ladder with Rolling-Origin Evaluation",
        "",
        "Generated by `scripts/baseline_ladder_rolling_origin.py`. See the module docstring for the full protocol.",
        "",
        "## Protocol",
        "",
        f"- Dunnhumby origins (day cutoffs): {DH_ORIGINS}; Online Retail II origins: "
        f"{[str(o.date()) for o in R2_ORIGINS]}; holdout = next {HORIZON_DAYS} days.",
        "- Online Retail II primary protocol (2010-12-09, 365-day holdout) is reported per origin but excluded from POOLED.",
        f"- Stratified {N_FOLDS}-fold CV over customers within each origin; all representations fit on training folds.",
        f"- Paired customer-level bootstrap (B={BOOT_N}); Holm correction over the pooled comparison family per dataset.",
        "- Bootstrap intervals reflect test-sample variability only; origins overlap, so pooled intervals are approximate.",
        "- Scoring: tie-preserving quintiles fit on training folds (identical raw values share a band).",
        "",
        "## Band occupancy (% of customers, full cohort, descriptive)",
        "",
        "Rows = dimension and scheme; columns = band 1..5 (R is inverted: band 5 = most recent).",
        "",
    ]
    for (dataset, origin), g in occupancy_df.groupby(["dataset", "origin"], sort=False):
        t = g.pivot_table(index=["dimension", "scheme"], columns="band", values="pct", sort=False) * 100
        t.columns = [f"band {b} %" for b in t.columns]
        lines += [f"### {dataset} — {origin}", "", _md(t.reset_index(), "{:.1f}"), ""]
    q = occupancy_df[occupancy_df["scheme"] == "quintile"]
    rng_tbl = q.groupby(["dataset", "dimension"])["pct"].agg(["min", "max"]).mul(100).reset_index()
    rng_tbl.columns = ["dataset", "dimension", "smallest band %", "largest band %"]
    lines += ["### Quintile scheme — range of band occupancy over all origins", "", _md(rng_tbl, "{:.1f}"), ""]
    lines += [
        "## Mean metrics over pooled origins",
        "",
    ]
    pooled = metrics_df[metrics_df["pooled"]]
    for dataset in pooled["dataset"].unique():
        t = pooled[pooled["dataset"] == dataset].groupby("arm", sort=False)[METRICS].mean().reset_index()
        lines += [f"### {dataset}", "", _md(t), ""]
    lines += ["## Per-origin metrics", ""]
    for dataset in metrics_df["dataset"].unique():
        t = metrics_df[metrics_df["dataset"] == dataset].pivot_table(
            index="arm", columns="origin", values="spend_r2", sort=False).reset_index()
        lines += [f"### {dataset} — Spend R² by origin", "", _md(t), ""]
        t = metrics_df[metrics_df["dataset"] == dataset].pivot_table(
            index="arm", columns="origin", values="invoice_r2", sort=False).reset_index()
        lines += [f"### {dataset} — Invoice R² by origin", "", _md(t), ""]
        t = metrics_df[metrics_df["dataset"] == dataset].pivot_table(
            index="arm", columns="origin", values="auc", sort=False).reset_index()
        lines += [f"### {dataset} — ROC AUC by origin", "", _md(t), ""]
        rates = metrics_df[metrics_df["dataset"] == dataset].drop_duplicates("origin")[
            ["origin", "n", "repurchase_rate"]]
        lines += [f"### {dataset} — cohorts", "", _md(rates), ""]
    lines += ["## Pooled paired comparisons (delta = arm - reference)", ""]
    pc = comp_df[comp_df["scope"] == "POOLED"][
        ["dataset", "comparison", "metric", "delta", "ci_low", "ci_high", "p_boot", "p_holm", "origins_positive"]]
    lines += [_md(pc), ""]
    if not info_df.empty:
        lines += ["## Concept counts (mean over folds and origins)", ""]
        t = info_df.groupby(["dataset", "arm"]).mean(numeric_only=True).drop(columns="fold").reset_index()
        lines += [_md(t, "{:.1f}"), ""]
    lines += [f"Runtime: {runtime / 60:.1f} min.", ""]
    (EXP_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

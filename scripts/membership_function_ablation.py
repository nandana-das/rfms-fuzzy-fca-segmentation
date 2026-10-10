"""Pre-specified membership-function ablation for the audited RFM-FCA pipeline.

The script deliberately owns its representation code and output directory.  It
does not change or import the rolling-origin runner.  M0 is implemented from
the audited equations so that the focused regression test can compare its
features and predictions with the frozen runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from project_paths import DATA_DIR, RESULTS_DIR, RETAIL2_RAW_DIR  # noqa: E402


OUT_DIR = RESULTS_DIR / "membership_function_ablation"
DIMS = ("R", "F", "M")
METHODS = ("M0", "M1", "M2", "M3")
METHOD_NAMES = {
    "M0": "Frozen piecewise-linear shoulders",
    "M1": "Normalized Gaussian",
    "M2": "Normalized generalized bell",
    "M3": "Log-coordinate Gaussian",
}
L_THRESHOLDS = (0.3, 0.5, 0.7)
MIN_SUPPORT = 0.04
J_MAX = 0.80
MU_CUT = 0.50
N_FOLDS = 5
CV_SEED = 0
BOOT_N = 2000
BOOT_SEED = 12345
LOGREG_CS = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)
DH_ORIGINS = [347, 438, 529, 620]
HORIZON_DAYS = 91
R2_END = pd.Timestamp("2011-12-09 23:59:59")
R2_ORIGINS = [R2_END - pd.Timedelta(days=HORIZON_DAYS * k) for k in (5, 4, 3, 2, 1)]
R2_PRIMARY_CUTOFF = pd.Timestamp("2010-12-09 23:59:59")
RAW_FILES = [RETAIL2_RAW_DIR / "online_retail_09_10.csv", RETAIL2_RAW_DIR / "online_retail_10_11.csv"]
CONCEPT_COLUMNS = ["intent", "intent_size", "dimensions", "n_customers", "support", "stability_proxy"]


def fit_quantile_cutoffs(rfm: pd.DataFrame) -> dict[str, np.ndarray]:
    """Copy the frozen tie-preserving training quantile rule exactly.

    Quantiles are fitted on training customers only.  Repeated raw values are
    preserved by the cutoff rule and `searchsorted(..., side="left")` in
    `quantile_scores`, so equal values always receive the same band.  A
    repeated *cutpoint* is rejected exactly as in the frozen baseline because
    it would make an empty band rather than silently changing the method.
    """
    cutoffs = {}
    for dim in DIMS:
        cut = np.quantile(rfm[dim].to_numpy(dtype=float), np.arange(1, 5) / 5)
        if np.any(np.diff(cut) <= 0):
            raise ValueError(f"Quantile cutpoints for {dim} are not strictly increasing: {cut}")
        cutoffs[dim] = cut
    return cutoffs


def quantile_scores(rfm: pd.DataFrame, cutoffs: dict[str, np.ndarray]) -> pd.DataFrame:
    scored = rfm.copy()
    for dim in DIMS:
        values = 1 + np.searchsorted(cutoffs[dim], scored[dim].to_numpy(dtype=float), side="left")
        if dim == "R":
            values = 6 - values
        scored[f"{dim}_score"] = values.astype(np.int8)
    return scored


def _checked_values(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("RFM values must be finite")
    return out


def _log_coordinate(values: np.ndarray) -> np.ndarray:
    # RFM is nonnegative by construction.  Clipping is a defensive, deterministic
    # projection rule for synthetic or malformed inputs; it is not used to fit
    # any parameter on the test set.
    return np.log1p(np.maximum(_checked_values(values), 0.0))


def band_centroids(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = _checked_values(values)
    fallback = float(np.median(values))
    return np.array([
        float(np.median(values[scores == k])) if np.any(scores == k) else fallback
        for k in range(1, 6)
    ])


def _piecewise_membership(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = _checked_values(values)
    c = _checked_values(centers)
    mu = np.zeros((len(x), 5), dtype=float)
    order = np.argsort(c, kind="stable")
    cs = c[order]
    if cs[0] == cs[-1]:
        # A fully coincident set has no identifiable shoulder direction.  Use
        # the middle level deterministically rather than creating two active
        # outer shoulders whose memberships would sum to two.
        mu[:, 2] = 1.0
        return mu[:, np.argsort(order, kind="stable")]
    below = x <= cs[0]
    above = x >= cs[-1]
    mu[below, 0] = 1.0
    mu[above, 4] = 1.0
    mid = ~below & ~above
    if np.any(mid):
        xm = x[mid]
        idx = np.clip(np.searchsorted(cs, xm, side="right") - 1, 0, 3)
        left, right = cs[idx], cs[idx + 1]
        denom = np.where(right - left == 0, 1e-9, right - left)
        frac_right = (xm - left) / denom
        rows = np.where(mid)[0]
        mu[rows, idx] = 1.0 - frac_right
        mu[rows, idx + 1] = frac_right
    # The piecewise function was evaluated in ascending raw-coordinate order;
    # return columns in the project's score/band order.
    return mu[:, np.argsort(order, kind="stable")]


def adjacent_widths(centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Return deterministic half-spacing widths and the numerical floor."""
    c = _checked_values(centers)
    scale = max(1.0, float(np.max(np.abs(c))))
    floor = 1e-12 * scale
    gaps = np.abs(np.diff(c))
    positive = gaps[gaps > 0]
    replacement = float(np.median(positive)) if len(positive) else floor
    effective = np.where(gaps > 0, gaps, replacement)
    widths = np.empty(5, dtype=float)
    widths[0] = effective[0] / 2.0
    widths[-1] = effective[-1] / 2.0
    widths[1:-1] = (effective[:-1] + effective[1:]) / 4.0
    widths = np.maximum(widths, floor)
    return widths, floor


def _normalized_membership(values: np.ndarray, centers: np.ndarray, widths: np.ndarray, kind: str) -> np.ndarray:
    x = _checked_values(values)[:, None]
    c = _checked_values(centers)[None, :]
    w = np.maximum(_checked_values(widths), np.finfo(float).tiny)[None, :]
    scaled = (x - c) / w
    if kind == "gaussian":
        log_weights = -0.5 * scaled * scaled
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
    elif kind == "bell":
        weights = 1.0 / (1.0 + np.abs(scaled) ** 4)  # fixed b=2, exponent 2b=4
    else:
        raise ValueError(kind)
    totals = weights.sum(axis=1, keepdims=True)
    if not np.isfinite(weights).all() or np.any(totals <= 0):
        raise FloatingPointError("non-finite or zero normalized membership weights")
    return weights / totals


def fit_memberships(method: str, train_scored: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if method not in METHODS:
        raise ValueError(method)
    train_cols, test_cols, params = {}, {}, {"method": method, "dimensions": {}}
    for dim in DIMS:
        train_raw = train_scored[dim].to_numpy(dtype=float)
        test_raw = test[dim].to_numpy(dtype=float)
        train_coord = _log_coordinate(train_raw) if method == "M3" else _checked_values(train_raw)
        test_coord = _log_coordinate(test_raw) if method == "M3" else _checked_values(test_raw)
        scores = train_scored[f"{dim}_score"].to_numpy()
        centers = band_centroids(train_coord, scores)
        widths, floor = adjacent_widths(centers)
        if method == "M0":
            mu_tr = _piecewise_membership(train_coord, centers)
            mu_te = _piecewise_membership(test_coord, centers)
        else:
            kind = "bell" if method == "M2" else "gaussian"
            mu_tr = _normalized_membership(train_coord, centers, widths, kind)
            mu_te = _normalized_membership(test_coord, centers, widths, kind)
        for k in range(5):
            train_cols[f"{dim}{k + 1}"] = mu_tr[:, k]
            test_cols[f"{dim}{k + 1}"] = mu_te[:, k]
        params["dimensions"][dim] = {
            "centers": centers.tolist(),
            "widths": widths.tolist(),
            "width_floor": floor,
            "coordinate": "log1p_clipped_nonnegative" if method == "M3" else "raw",
            "bell_shape_b": 2.0 if method == "M2" else None,
        }
    return pd.DataFrame(train_cols), pd.DataFrame(test_cols), params


def _extract_bands(intent: str) -> list[str]:
    if not intent or intent == "(universal root)":
        return []
    bands = []
    for item in intent.split(" & "):
        band = item.split("@")[0].strip()
        if band not in bands:
            bands.append(band)
    return bands


def concept_memberships(concepts: pd.DataFrame, bands: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(bands), len(concepts)), dtype=float)
    for j, row in concepts.iterrows():
        names = [b for b in _extract_bands(row["intent"]) if b in bands.columns]
        matrix[:, j] = 1.0 if not names else np.min(bands[names].to_numpy(dtype=float), axis=1)
    return matrix


def mine_concepts(memberships: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Audited threshold scaling, FP-growth, exact extent grouping and closure."""
    n = len(memberships)
    if n == 0:
        return pd.DataFrame(columns=CONCEPT_COLUMNS)
    attr_extent: dict[str, int] = {}
    transactions = []
    for row_idx, row in enumerate(memberships.itertuples(index=False, name=None)):
        tx = []
        for col_idx, col in enumerate(memberships.columns):
            for threshold in L_THRESHOLDS:
                attr = f"{col}@{threshold:.1f}"
                if row[col_idx] >= threshold:
                    tx.append(attr)
                    attr_extent[attr] = attr_extent.get(attr, 0) | (1 << row_idx)
        transactions.append(tx)
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions, sparse=True)
    binary = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.columns_)
    if binary.shape[1] == 0:
        return pd.DataFrame(columns=CONCEPT_COLUMNS)
    frequent = fpgrowth(binary, min_support=MIN_SUPPORT, use_colnames=True, max_len=None)
    groups: dict[int, set[str]] = {}
    for itemset in frequent["itemsets"]:
        extent = (1 << n) - 1
        for item in sorted(itemset, key=lambda a: attr_extent.get(a, 0).bit_count()):
            extent &= attr_extent[item]
            if not extent:
                break
        groups.setdefault(extent, set()).update(sorted(itemset))
    profiles = scored[[f"{d}_score" for d in DIMS]].to_numpy()
    records = []
    for extent, intent in groups.items():
        n_customers = extent.bit_count()
        support = n_customers / n
        if support < MIN_SUPPORT:
            continue
        unique_profiles = set()
        remaining = extent
        while remaining:
            bit = remaining & -remaining
            unique_profiles.add(tuple(profiles[bit.bit_length() - 1]))
            remaining ^= bit
        records.append({
            "intent": " & ".join(sorted(intent)),
            "intent_size": len(intent),
            "dimensions": "".join(sorted({a[0] for a in intent})),
            "n_customers": n_customers,
            "support": support,
            "stability_proxy": 1.0 if n_customers <= 1 else 1.0 - len(unique_profiles) / n_customers,
        })
    if not records:
        return pd.DataFrame(columns=CONCEPT_COLUMNS)
    return pd.DataFrame(records).sort_values(
        ["support", "intent_size", "intent"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _concept_info(concepts: pd.DataFrame, x_train: np.ndarray, n_candidates: int, mu_train: pd.DataFrame) -> dict:
    distinct = int(np.unique(np.round(x_train, 12), axis=1).shape[1]) if x_train.shape[1] else 0
    empty = int(((x_train >= MU_CUT).sum(axis=0) == 0).sum()) if x_train.shape[1] else 0
    entropy = []
    overlap = []
    for dim in DIMS:
        a = mu_train[[f"{dim}{k}" for k in range(1, 6)]].to_numpy(dtype=float)
        entropy.extend((-np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300)), 0), axis=1) / math.log(5)).tolist())
        overlap.extend((1.0 / np.sum(a * a, axis=1)).tolist())
    return {
        "n_candidates": n_candidates,
        "n_features": int(x_train.shape[1]),
        "n_distinct_columns": distinct,
        "n_empty_core": empty,
        "mean_membership_entropy": float(np.mean(entropy)),
        "mean_effective_overlap": float(np.mean(overlap)),
        "mean_core_concepts_per_customer": float(np.mean((x_train >= MU_CUT).sum(axis=1))) if x_train.shape[1] else 0.0,
        "concept_signatures": sorted({" & ".join(_extract_bands(i)) for i in concepts["intent"]}) if len(concepts) else [],
    }


def _score_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = fit_quantile_cutoffs(train)
    base_cols = ["CustomerID", *DIMS]
    return quantile_scores(train[base_cols].copy(), cutoffs), quantile_scores(test[base_cols].copy(), cutoffs)


def build_features(method: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_scored, _ = _score_train_test(train, test)
    mu_train, mu_test, params = fit_memberships(method, train_scored, test)
    raw = mine_concepts(mu_train, train_scored)
    mu_raw = concept_memberships(raw, mu_train)
    kept = suppress_redundant_concepts(
        raw, mu_raw, j_max=J_MAX, mu_cut=MU_CUT, drop_empty_core=True
    ).reset_index(drop=True)
    x_train = concept_memberships(kept, mu_train)
    x_test = concept_memberships(kept, mu_test)
    info = _concept_info(kept, x_train, len(raw), mu_train)
    info.update({
        "n_raw_concepts": int(len(raw)),
        "n_retained_concepts": int(len(kept)),
        "parameters": params,
    })
    return x_train, x_test, info


def _log_rfm(df: pd.DataFrame) -> np.ndarray:
    return _log_coordinate(df[list(DIMS)].to_numpy(dtype=float))


def fit_predict(x_train: np.ndarray, x_test: np.ndarray, train: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_rep = train["repurchased"].to_numpy()
    y_sp = np.log1p(train["future_spend"].to_numpy())
    y_inv = np.log1p(train["future_invoices"].to_numpy())
    clf = LogisticRegressionCV(
        Cs=LOGREG_CS, cv=5, scoring="roc_auc", solver="lbfgs", max_iter=LOGREG_MAX_ITER, random_state=seed
    ).fit(x_train, y_rep)
    reg_sp = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(x_train, y_sp)
    reg_inv = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(x_train, y_inv)
    return clf.predict_proba(x_test)[:, 1], reg_sp.predict(x_test), reg_inv.predict(x_test)


def _finish(dataset: str, origin: str, frame: pd.DataFrame, pooled: bool) -> dict:
    frame = frame.copy()
    frame["future_invoices"] = frame["future_invoices"].fillna(0).astype(int)
    frame["future_spend"] = frame["future_spend"].fillna(0.0).clip(lower=0.0)
    frame["repurchased"] = (frame["future_invoices"] > 0).astype(int)
    return {"dataset": dataset, "origin": origin, "pooled": pooled, "df": frame.reset_index(drop=True)}


def dunnhumby_cohorts(tx: pd.DataFrame) -> list[dict]:
    out = []
    for cutoff in DH_ORIGINS:
        obs = tx[tx["DAY"] <= cutoff]
        hold = tx[(tx["DAY"] > cutoff) & (tx["DAY"] <= cutoff + HORIZON_DAYS)]
        agg = obs.groupby("household_key").agg(last_day=("DAY", "max"), F=("BASKET_ID", "nunique"), M=("SALES_VALUE", "sum")).reset_index()
        agg["R"] = cutoff - agg["last_day"]
        fut = hold.groupby("household_key").agg(future_invoices=("BASKET_ID", "nunique"), future_spend=("SALES_VALUE", "sum")).reset_index()
        df = agg.merge(fut, on="household_key", how="left")
        df["CustomerID"] = df["household_key"].astype(str)
        out.append(_finish("Dunnhumby", f"day {cutoff}", df, True))
    return out


def load_retail2() -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in RAW_FILES], ignore_index=True)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    clean = raw.dropna(subset=["CustomerID"]).copy()
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0) & (~clean["InvoiceNo"].astype(str).str.startswith("C"))].copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["line_value"] = clean["Quantity"] * clean["UnitPrice"]
    return clean


def retail2_cohorts(clean: pd.DataFrame) -> list[dict]:
    out = []
    specs = [(c, c + pd.Timedelta(days=HORIZON_DAYS), True) for c in R2_ORIGINS]
    specs.append((R2_PRIMARY_CUTOFF, R2_END, False))
    for cutoff, hold_end, pooled in specs:
        obs = clean[clean["InvoiceDate"] <= cutoff]
        hold = clean[(clean["InvoiceDate"] > cutoff) & (clean["InvoiceDate"] <= hold_end)]
        df = obs.groupby("CustomerID").agg(last_purchase=("InvoiceDate", "max"), F=("InvoiceNo", "nunique"), M=("line_value", "sum")).reset_index()
        df["R"] = (cutoff - df["last_purchase"]).dt.days.astype(int)
        df = df[["CustomerID", "R", "F", "M"]]
        fut = hold.groupby("CustomerID").agg(future_invoices=("InvoiceNo", "nunique"), future_spend=("line_value", "sum")).reset_index()
        df = df.merge(fut, on="CustomerID", how="left")
        label = f"{cutoff.date()}" + ("" if pooled else " (primary, 365d)")
        out.append(_finish("Online Retail II", label, df, pooled))
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    ranks = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float(((ranks * y).sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _metric_rows(frame: pd.DataFrame, pred: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    return {
        "auc": _auc(frame["repurchased"].to_numpy()[idx], pred["prob"][idx]),
        "spend_r2": _r2(np.log1p(frame["future_spend"].to_numpy())[idx], pred["sp"][idx]),
        "invoice_r2": _r2(np.log1p(frame["future_invoices"].to_numpy())[idx], pred["inv"][idx]),
    }


def _bootstrap_metric(frame: pd.DataFrame, pred: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate one method on a supplied customer bootstrap matrix."""
    y_rep = frame["repurchased"].to_numpy()[indices]
    y_sp = np.log1p(frame["future_spend"].to_numpy())[indices]
    y_inv = np.log1p(frame["future_invoices"].to_numpy())[indices]
    p, s, v = pred["prob"][indices], pred["sp"][indices], pred["inv"][indices]
    ranks = rankdata(p, axis=1)
    n1 = y_rep.sum(axis=1)
    n0 = len(frame) - n1
    return {
        "auc": ((ranks * y_rep).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0),
        "spend_r2": 1.0 - ((y_sp - s) ** 2).sum(axis=1) / ((y_sp - y_sp.mean(axis=1, keepdims=True)) ** 2).sum(axis=1),
        "invoice_r2": 1.0 - ((y_inv - v) ** 2).sum(axis=1) / ((y_inv - y_inv.mean(axis=1, keepdims=True)) ** 2).sum(axis=1),
    }


def _boot_p(draws: np.ndarray) -> float:
    p = 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0))
    return float(min(1.0, max(p, 1.0 / len(draws))))


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        out[index] = min(1.0, running)
    return out.tolist()


def run_origin(cohort: dict, boot_n: int = BOOT_N) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    frame = cohort["df"]
    n = len(frame)
    oof = {m: {k: np.full(n, np.nan) for k in ("prob", "sp", "inv")} for m in METHODS}
    split_rows, info_rows, stability_sets = [], [], {m: [] for m in METHODS}
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (i_train, i_test) in enumerate(folds.split(frame, frame["repurchased"])):
        train = frame.iloc[i_train].reset_index(drop=True)
        test = frame.iloc[i_test].reset_index(drop=True)
        split_rows.extend({"dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold, "method": m, "test_customer_ids": "|".join(test["CustomerID"].astype(str))} for m in METHODS)
        for method in METHODS:
            x_train, x_test, info = build_features(method, train, test)
            prob, spend, invoice = fit_predict(x_train, x_test, train, CV_SEED + fold)
            oof[method]["prob"][i_test] = prob
            oof[method]["sp"][i_test] = spend
            oof[method]["inv"][i_test] = invoice
            stability_sets[method].append(set(info.pop("concept_signatures", [])))
            info_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold, "method": method, **info})
    metrics, bootstrap = [], {}
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_indices = rng.integers(0, n, size=(boot_n, n))
    for method in METHODS:
        point = _metric_rows(frame, oof[method], np.arange(n))
        metrics.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "pooled": cohort["pooled"], "n": n, "repurchase_rate": float(frame["repurchased"].mean()), "method": method, **point})
        bootstrap[method] = _bootstrap_metric(frame, oof[method], bootstrap_indices)
    stability = []
    for method, sets in stability_sets.items():
        pairwise = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                pairwise.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
        stability.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "method": method, "concept_set_jaccard": float(np.mean(pairwise)) if pairwise else np.nan})
    paired_differences = {
        method: {
            metric: bootstrap[method][metric] - bootstrap["M0"][metric]
            for metric in ("auc", "spend_r2", "invoice_r2")
        }
        for method in METHODS[1:]
    }
    return pd.DataFrame(metrics), pd.DataFrame(info_rows), pd.DataFrame(split_rows), {
        "oof": oof,
        "bootstrap": bootstrap,
        "bootstrap_indices": bootstrap_indices,
        "paired_differences": paired_differences,
        "stability": pd.DataFrame(stability),
    }


def _prepare_output() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter."""
    columns = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.fillna("").itertuples(index=False, name=None)]
    widths = [max(len(col), *(len(row[i]) for row in rows)) if rows else len(col) for i, col in enumerate(columns)]
    header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def aggregate_comparisons(
    metrics_df: pd.DataFrame,
    paired_differences: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    """Keep every origin row, but pool only cohorts explicitly marked pooled."""
    comparisons = []
    for dataset in metrics_df["dataset"].unique():
        ds = metrics_df[metrics_df["dataset"] == dataset]
        all_origins = list(ds["origin"].drop_duplicates())
        pooled_origins = list(ds.loc[ds["pooled"].astype(bool), "origin"].drop_duplicates())
        if not pooled_origins:
            raise ValueError(f"No pooled origins available for {dataset}")
        for method in METHODS[1:]:
            for metric in ("auc", "spend_r2", "invoice_r2"):
                per_origin = []
                draws_by_origin = []
                for origin in all_origins:
                    point = ds[ds["origin"] == origin].set_index("method")[metric]
                    draws = paired_differences[(dataset, origin)][method][metric]
                    delta = float(point[method] - point["M0"])
                    per_origin.append(delta)
                    draws_by_origin.append(draws)
                    comparisons.append({
                        "dataset": dataset, "scope": origin, "comparison": f"{method} - M0",
                        "metric": metric, "delta": delta,
                        "ci_low": float(np.percentile(draws, 2.5)),
                        "ci_high": float(np.percentile(draws, 97.5)),
                        "p_boot": _boot_p(draws),
                    })
                pooled_positions = [all_origins.index(origin) for origin in pooled_origins]
                pooled_deltas = [per_origin[i] for i in pooled_positions]
                pooled_draws = np.mean([draws_by_origin[i] for i in pooled_positions], axis=0)
                comparisons.append({
                    "dataset": dataset, "scope": "POOLED", "comparison": f"{method} - M0",
                    "metric": metric, "delta": float(np.mean(pooled_deltas)),
                    "ci_low": float(np.percentile(pooled_draws, 2.5)),
                    "ci_high": float(np.percentile(pooled_draws, 97.5)),
                    "p_boot": _boot_p(pooled_draws),
                    "origins_positive": f"{sum(d > 0 for d in pooled_deltas)}/{len(pooled_deltas)}",
                })
    comp_df = pd.DataFrame(comparisons)
    comp_df["p_holm"] = np.nan
    for dataset in comp_df["dataset"].unique():
        mask = (comp_df["dataset"] == dataset) & (comp_df["scope"] == "POOLED")
        comp_df.loc[mask, "p_holm"] = _holm(comp_df.loc[mask, "p_boot"].tolist())
    return comp_df


def run_experiment(smoke: bool = False) -> dict[str, object]:
    _prepare_output()
    t0 = time.time()
    dh_path = DATA_DIR / "dunnhumby_raw" / "transaction_data.csv"
    dh_tx = pd.read_csv(dh_path, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = dunnhumby_cohorts(dh_tx) + retail2_cohorts(load_retail2())
    selected = cohorts[:1] if smoke else cohorts
    boot_n = 200 if smoke else BOOT_N
    metrics, info, splits, stability = [], [], [], []
    boot_store = {}
    for cohort in selected:
        print(f"[{cohort['dataset']} | {cohort['origin']}] n={len(cohort['df']):,}", flush=True)
        m, i, s, aux = run_origin(cohort, boot_n=boot_n)
        metrics.append(m)
        info.append(i.drop(columns=["parameters"], errors="ignore"))
        splits.append(s)
        stability.append(aux["stability"])
        boot_store[(cohort["dataset"], cohort["origin"])] = aux["paired_differences"]
    metrics_df = pd.concat(metrics, ignore_index=True)
    info_df = pd.concat(info, ignore_index=True)
    splits_df = pd.concat(splits, ignore_index=True)
    stability_df = pd.concat(stability, ignore_index=True)
    comp_df = aggregate_comparisons(metrics_df, boot_store)
    metrics_df.to_csv(OUT_DIR / "per_origin_metrics.csv", index=False)
    info_df.to_csv(OUT_DIR / "concept_counts_per_fold.csv", index=False)
    splits_df.to_csv(OUT_DIR / "customer_splits.csv", index=False)
    stability_df.to_csv(OUT_DIR / "concept_stability.csv", index=False)
    comp_df.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    parameters = {"methods": METHOD_NAMES, "dims": DIMS, "thresholds": L_THRESHOLDS, "min_support": MIN_SUPPORT, "j_max": J_MAX, "mu_cut": MU_CUT, "n_folds": N_FOLDS, "cv_seed": CV_SEED, "bootstrap_n": boot_n, "smoke": smoke, "selected_cohorts": [f"{c['dataset']} | {c['origin']}" for c in selected], "width_rule": "half adjacent spacing; interior half mean spacing; zero gaps use median positive gap; floor=1e-12*max(1,max|center|)", "bell_shape_b": 2.0, "outputs_are_isolated": True}
    (OUT_DIR / "run_parameters.json").write_text(json.dumps(parameters, indent=2, default=str), encoding="utf-8")
    report = ["# Membership-function ablation smoke results" if smoke else "# Membership-function ablation results", "", f"Methods: {', '.join(f'{m} ({METHOD_NAMES[m]})' for m in METHODS)}.", "", "All methods use identical customer folds and the frozen thresholds/support/suppression/model settings. Deltas are method minus M0; no method was selected from held-out outcomes.", "", "## Metrics", "", _markdown_table(metrics_df), "", "## Paired comparisons", "", _markdown_table(comp_df[comp_df["scope"] == "POOLED"]), "", "## Concept-set stability diagnostic", "", _markdown_table(stability_df), "", f"Runtime: {(time.time() - t0) / 60:.2f} minutes."]
    (OUT_DIR / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return {"metrics": metrics_df, "comparisons": comp_df, "info": info_df, "stability": stability_df, "smoke": smoke}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run the first Dunnhumby rolling origin only with 200 bootstrap draws")
    parser.add_argument("--output-subdir", default="", help="optional subdirectory under results/membership_function_ablation")
    args = parser.parse_args()
    global OUT_DIR
    if args.output_subdir:
        candidate = (OUT_DIR / args.output_subdir).resolve()
        root = OUT_DIR.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("--output-subdir must remain under results/membership_function_ablation")
        OUT_DIR = candidate
    run_experiment(smoke=args.smoke)


if __name__ == "__main__":
    main()

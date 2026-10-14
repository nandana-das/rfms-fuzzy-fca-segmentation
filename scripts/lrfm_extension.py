"""Exploratory LRFM extension.

This driver composes the frozen rolling-origin pipeline, the v2 hybrid builder,
the CDNOW tie-merging rule, and the robustness inference helpers.  It does not
modify any of those modules or their committed outputs.  All LRFM outputs are
written only after both pre-analysis guards pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
import cdnow_confirmation as cc  # noqa: E402
import qw_inference_robustness as qw  # noqa: E402
import v2_hybrid_ladder as v2  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_customer_concept_memberships,
    dense_rank_scores,
    fit_quantile_cutoffs,
    load_cleaned_transactions,
    mine_crisp_closed_concepts,
    quantile_scores,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    _band_centroids_generic,
    _piecewise_membership_generic,
    mine_fuzzy_closed_concepts_with_thresholds,
)
from concept_redundancy import suppress_redundant_concepts  # noqa: E402

_V1_BUILD_FEATURES = bl.build_features
_V2_BUILD_FEATURES = v2.build_features

OUT = ROOT / "results" / "lrfm_extension"
REPORT = ROOT / "docs" / "LRFM_EXTENSION_RESULTS.md"
RFM_DIMS = ("R", "F", "M")
LRFM_DIMS = ("R", "F", "M", "L")
ARMS = [
    "spline_log_rfm", "fuzzy_rfm_fca", "hybrid_fuzzy_fca",
    "log_lrfm", "spline_log_lrfm", "crisp_lrfm_fca",
    "fuzzy_lrfm_fca", "hybrid_lrfm",
]
COMPARISONS = [
    ("E1", "fuzzy_lrfm_fca", "crisp_lrfm_fca"),
    ("E2", "fuzzy_lrfm_fca", "spline_log_lrfm"),
    ("E3", "hybrid_lrfm", "spline_log_lrfm"),
    ("E4", "fuzzy_lrfm_fca", "fuzzy_rfm_fca"),
    ("E5", "spline_log_lrfm", "spline_log_rfm"),
]
METRICS = ["auc", "spend_r2", "invoice_r2"]
SEEDS = [0, 1, 2, 3, 4]
BOOT_N = 10_000
CHECKPOINT_VERSION = 1
CHECKPOINT_COLUMNS = [
    "dataset", "origin", "seed", "customer_id", "customer_index", "arm",
    "repurchased", "future_spend", "future_invoices", "prob", "sp", "inv",
]
IMPLEMENTATION_FILES = (
    "scripts/lrfm_extension.py",
    "scripts/baseline_ladder_rolling_origin.py",
    "scripts/v2_hybrid_ladder.py",
    "scripts/cdnow_confirmation.py",
    "scripts/qw_inference_robustness.py",
    "scripts/fair_comparison_retail2.py",
    "scripts/fuzzy_membership_sensitivity.py",
    "scripts/concept_redundancy.py",
)


def _log(msg: str) -> None:
    print(f"[lrfm] {msg}", flush=True)


def _with_length(base: list[dict], tx: pd.DataFrame, key: str, date: str) -> list[dict]:
    tx_work = tx.copy()
    tx_work[key] = tx_work[key].astype(str)
    out = []
    for cohort in base:
        cutoff_text = cohort["origin"].split(" ")[0]
        if cohort["dataset"] == "Dunnhumby":
            cutoff = int(cohort["origin"].split()[-1])
        elif cohort["dataset"] == "Online Retail II":
            # retail2_cohorts labels the baseline's 23:59:59 cutoff by date only.
            cutoff = pd.Timestamp(cutoff_text) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        else:
            cutoff = pd.Timestamp(cutoff_text)
        obs = tx_work[tx_work[date] <= cutoff]
        lengths = obs.groupby(key)[date].agg(first="min", last="max")
        delta = lengths["last"] - lengths["first"]
        lengths["L"] = delta.dt.days if hasattr(delta, "dt") else delta.astype(float)
        df = cohort["df"].copy()
        ids = df["CustomerID"].astype(str)
        lengths.index = lengths.index.astype(str)
        df["L"] = ids.astype(str).map(lengths["L"]).astype(float).to_numpy()
        if df["L"].isna().any():
            missing = ids[df["L"].isna()].drop_duplicates()
            sample = missing.head(5).tolist()
            has_qualifying = {
                customer_id: bool(((tx_work[key] == customer_id) & (tx_work[date] <= cutoff)).any())
                for customer_id in sample
            }
            _log(
                f"missing L diagnostic dataset={cohort['dataset']} origin={cohort['origin']} "
                f"count={len(missing)} sample={sample} has_qualifying={has_qualifying}"
            )
            raise ValueError(f"missing cutoff-safe L values for {cohort['dataset']} {cohort['origin']}")
        out.append({**cohort, "df": df})
    return out


def dunnhumby_cohorts(tx: pd.DataFrame) -> list[dict]:
    return _with_length(bl.dunnhumby_cohorts(tx), tx, "household_key", "DAY")


def retail2_cohorts(tx: pd.DataFrame) -> list[dict]:
    return _with_length(bl.retail2_cohorts(tx), tx, "CustomerID", "InvoiceDate")


def cdnow_cohorts(tx: pd.DataFrame) -> list[dict]:
    return _with_length(cc.cdnow_cohorts(tx), tx, "cust", "date")


def merged_cutpoints(x: np.ndarray) -> np.ndarray:
    """Return tie/empty-band merged quintile cutpoints."""
    return cc.merged_cutpoints(np.asarray(x, dtype=float))


def merged_scores(tr: pd.DataFrame, dims: tuple[str, ...] = LRFM_DIMS):
    cuts = {d: merged_cutpoints(tr[d].to_numpy(dtype=float)) for d in dims}
    return quantile_scores(tr[["CustomerID", *dims]].copy(), cuts, dims=dims), cuts


def _fuzzy_context(tr: pd.DataFrame, te: pd.DataFrame):
    scored, _ = merged_scores(tr)
    mu_tr, centroids, _ = compute_fuzzy_memberships_n(scored)
    mu_te, _, _ = compute_fuzzy_memberships_n(
        te[["CustomerID", *LRFM_DIMS]].copy(), trained_centroids=centroids
    )
    return scored, mu_tr, mu_te


def compute_fuzzy_memberships_n(scored: pd.DataFrame, trained_centroids=None):
    memberships, centroids = {}, {}
    for dim in LRFM_DIMS:
        raw = scored[dim].to_numpy(dtype=float)
        if trained_centroids is None:
            scores = scored[f"{dim}_score"].to_numpy()
            n_levels = int(scores.max())
            if set(np.unique(scores)) != set(range(1, n_levels + 1)):
                raise ValueError(f"{dim}: non-contiguous merged bands")
            c = _band_centroids_generic(raw, scores, n_levels)
        else:
            c = np.asarray(trained_centroids[dim], dtype=float)
        order = np.argsort(c, kind="stable")
        mu = _piecewise_membership_generic(raw, c[order])[:, np.argsort(order, kind="stable")]
        for k in range(len(c)):
            memberships[f"{dim}{k + 1}"] = mu[:, k]
        centroids[dim] = c
    return pd.DataFrame(memberships), centroids, pd.DataFrame()


def _concept_info(concepts: pd.DataFrame, X: np.ndarray, candidates: int) -> dict:
    distinct = int(np.unique(np.round(X, 12), axis=1).shape[1]) if X.shape[1] else 0
    empty = int(((X >= bl.MU_CUT).sum(axis=0) == 0).sum()) if X.shape[1] else 0
    return {"n_candidates": candidates, "n_features": int(X.shape[1]),
            "n_distinct_columns": distinct, "n_empty_core": empty}


def _crisp_bands(tr: pd.DataFrame, te: pd.DataFrame):
    scored, cuts = merged_scores(tr)
    te_scored = quantile_scores(te[["CustomerID", *LRFM_DIMS]].copy(), cuts, dims=LRFM_DIMS)
    b_tr, b_te = {}, {}
    for dim in LRFM_DIMS:
        levels = int(scored[f"{dim}_score"].max())
        for k in range(1, levels + 1):
            b_tr[f"{dim}{k}"] = (scored[f"{dim}_score"] == k).astype(float)
            b_te[f"{dim}{k}"] = (te_scored[f"{dim}_score"] == k).astype(float)
    return scored, pd.DataFrame(b_tr), pd.DataFrame(b_te)


def _lrfm_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame):
    if arm in ("log_lrfm", "spline_log_lrfm"):
        sc_data = np.log1p(tr[list(LRFM_DIMS)].to_numpy(float))
        te_data = np.log1p(te[list(LRFM_DIMS)].to_numpy(float))
        transformer = (SplineTransformer(n_knots=5, degree=3, knots="quantile")
                       if arm == "spline_log_lrfm" else StandardScaler())
        transformer.fit(sc_data)
        return transformer.transform(sc_data), transformer.transform(te_data), {}
    if arm == "crisp_lrfm_fca":
        scored, b_tr, b_te = _crisp_bands(tr, te)
        try:
            concepts = mine_crisp_closed_concepts(scored, min_support=bl.MIN_SUPPORT, dims=LRFM_DIMS)
        except Exception as exc:
            raise RuntimeError(f"crisp LRFM concept mining failed: {exc}") from exc
        concepts = concepts[concepts["intent_size"] > 0].reset_index(drop=True)
        xtr = compute_customer_concept_memberships(concepts, b_tr)
        xte = compute_customer_concept_memberships(concepts, b_te)
        return xtr, xte, _concept_info(concepts, xtr, len(concepts))
    if arm == "fuzzy_lrfm_fca":
        scored, mu_tr, mu_te = _fuzzy_context(tr, te)
        try:
            raw, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
                mu_tr, scored, bl.L_THRESHOLDS, min_support=bl.MIN_SUPPORT, dims=LRFM_DIMS
            )
        except Exception as exc:
            raise RuntimeError(f"fuzzy LRFM concept mining failed: {exc}") from exc
        raw = raw.reset_index(drop=True)
        mu_all = compute_customer_concept_memberships(raw, mu_tr)
        kept = suppress_redundant_concepts(
            raw, mu_all, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True
        ).reset_index(drop=True)
        xtr = compute_customer_concept_memberships(kept, mu_tr)
        xte = compute_customer_concept_memberships(kept, mu_te)
        return xtr, xte, _concept_info(kept, xtr, len(raw))
    if arm == "hybrid_lrfm":
        xtr, xte, info = _lrfm_features("fuzzy_lrfm_fca", tr, te)
        sc = StandardScaler().fit(np.log1p(tr[list(LRFM_DIMS)].to_numpy(float)))
        return (np.hstack([xtr, sc.transform(np.log1p(tr[list(LRFM_DIMS)].to_numpy(float)))]),
                np.hstack([xte, sc.transform(np.log1p(te[list(LRFM_DIMS)].to_numpy(float)))]),
                {**info, "n_features": int(xtr.shape[1] + 4)})
    raise ValueError(arm)


@contextmanager
def _pipeline_state(dims, arms, builder):
    old = bl.DIMS, bl.ARMS, bl.build_features
    bl.DIMS, bl.ARMS, bl.build_features = dims, arms, builder
    try:
        yield
    finally:
        bl.DIMS, bl.ARMS, bl.build_features = old


@contextmanager
def _cdnow_extension_state():
    old_helpers = bl._quintile_scored, bl.compute_fuzzy_memberships
    active_arms, active_builder = bl.ARMS, bl.build_features
    cc.install_extension()
    bl.ARMS, bl.build_features = active_arms, active_builder
    try:
        yield
    finally:
        bl._quintile_scored, bl.compute_fuzzy_memberships = old_helpers
        bl.ARMS, bl.build_features = active_arms, active_builder


def _dataset_extension_state(dataset: str):
    return _cdnow_extension_state() if dataset == "CDNOW" else nullcontext()


def _validate_oof_arms(oof: dict, dataset: str, origin: str) -> None:
    missing = [arm for arm in ARMS if arm not in oof]
    if missing:
        raise RuntimeError(
            f"run_origin returned incomplete LRFM arms for {dataset} {origin}: {missing}"
        )


def _checkpoint_dir() -> Path:
    return OUT / "checkpoints"


def _checkpoint_key(dataset: str, origin: str, seed: int) -> str:
    raw = f"{dataset}\0{origin}\0{seed}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _checkpoint_paths(dataset: str, origin: str, seed: int) -> tuple[Path, Path]:
    key = _checkpoint_key(dataset, origin, seed)
    directory = _checkpoint_dir()
    return directory / f"run-{key}.csv", directory / f"run-{key}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cohort_fingerprint(df: pd.DataFrame) -> str:
    columns = ["CustomerID", "R", "F", "M", "L", "repurchased", "future_spend", "future_invoices"]
    hashed = pd.util.hash_pandas_object(df[columns].reset_index(drop=True), index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _checkpoint_metadata(cohort: dict, seed: int) -> dict:
    df = cohort["df"]
    return {
        "schema_version": CHECKPOINT_VERSION,
        "dataset": cohort["dataset"],
        "origin": cohort["origin"],
        "seed": seed,
        "arms": list(ARMS),
        "dims": list(LRFM_DIMS),
        "metrics": list(METRICS),
        "customer_count": len(df),
        "cohort_fingerprint": _cohort_fingerprint(df),
        "customer_ids": df["CustomerID"].astype(str).tolist(),
        "implementation_fingerprint": _implementation_fingerprint(),
    }


def _implementation_fingerprint() -> str:
    protocol = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "dims": LRFM_DIMS,
        "arms": ARMS,
        "comparisons": COMPARISONS,
        "metrics": METRICS,
        "seeds": SEEDS,
        "bootstrap_draws": BOOT_N,
        "bootstrap_seed": qw.BOOT_SEED,
        "bootstrap_chunk": qw.CHUNK,
        "folds": bl.N_FOLDS,
        "holdout_days": bl.HORIZON_DAYS,
        "support": bl.MIN_SUPPORT,
        "thresholds": bl.L_THRESHOLDS,
        "jaccard_max": bl.J_MAX,
        "membership_cutoff": bl.MU_CUT,
        "logistic_cs": bl.LOGREG_CS,
        "logistic_max_iter": bl.LOGREG_MAX_ITER,
        "ridge_alphas": bl.RIDGE_ALPHAS.tolist(),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(protocol, sort_keys=True, default=str).encode("utf-8"))
    for relative in IMPLEMENTATION_FILES:
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_frame(cohort: dict, seed: int, oof: dict) -> pd.DataFrame:
    df = cohort["df"].reset_index(drop=True)
    rows = []
    for arm in ARMS:
        pred = oof[arm]
        for index, customer_id in enumerate(df["CustomerID"].astype(str)):
            rows.append({
                "dataset": cohort["dataset"], "origin": cohort["origin"], "seed": seed,
                "customer_id": customer_id, "customer_index": index, "arm": arm,
                "repurchased": int(df.at[index, "repurchased"]),
                "future_spend": float(df.at[index, "future_spend"]),
                "future_invoices": int(df.at[index, "future_invoices"]),
                "prob": float(pred["prob"][index]), "sp": float(pred["sp"][index]),
                "inv": float(pred["inv"][index]),
            })
    return pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)


def _write_checkpoint(cohort: dict, seed: int, oof: dict, info: list[dict]) -> None:
    frame = _checkpoint_frame(cohort, seed, oof)
    if len(frame) != len(cohort["df"]) * len(ARMS):
        raise RuntimeError("checkpoint row count does not match cohort and arm count")
    csv_path, metadata_path = _checkpoint_paths(cohort["dataset"], cohort["origin"], seed)
    metadata = _checkpoint_metadata(cohort, seed)
    metadata["info_rows"] = info
    _atomic_write_csv(csv_path, frame)
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, default=str))


def _validate_checkpoint_frame(frame: pd.DataFrame, cohort: dict, seed: int, metadata: dict) -> None:
    expected = _checkpoint_metadata(cohort, seed)
    for key in ("schema_version", "dataset", "origin", "seed", "arms", "dims", "metrics",
                "customer_count", "cohort_fingerprint", "customer_ids",
                "implementation_fingerprint"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"metadata mismatch in checkpoint field {key}")
    if list(frame.columns) != CHECKPOINT_COLUMNS:
        raise ValueError("checkpoint schema columns do not match")
    if len(frame) != len(cohort["df"]) * len(ARMS):
        raise ValueError("checkpoint row count does not match expected rows")
    if frame.duplicated(["customer_index", "arm"]).any():
        raise ValueError("checkpoint contains duplicate customer/arm rows")
    if set(frame["arm"]) != set(ARMS):
        raise ValueError("checkpoint arm set does not match the planned arms")
    expected_ids = cohort["df"]["CustomerID"].astype(str).tolist()
    by_index = frame.sort_values(["customer_index", "arm"])
    if by_index["customer_index"].tolist() != sorted(
        index for index in range(len(cohort["df"])) for _ in ARMS
    ):
        raise ValueError("checkpoint customer indices are incomplete")
    if by_index.drop_duplicates("customer_index")["customer_id"].tolist() != expected_ids:
        raise ValueError("checkpoint customer IDs do not match the cohort")
    if not np.isfinite(frame[["prob", "sp", "inv"]].to_numpy(dtype=float)).all():
        raise ValueError("checkpoint contains non-finite predictions")


def _load_checkpoint(cohort: dict, seed: int) -> tuple[dict, list[dict]]:
    csv_path, metadata_path = _checkpoint_paths(cohort["dataset"], cohort["origin"], seed)
    if not csv_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("checkpoint CSV or metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    _validate_checkpoint_frame(frame, cohort, seed, metadata)
    n = len(cohort["df"])
    oof = {}
    for arm in ARMS:
        rows = frame[frame["arm"] == arm].sort_values("customer_index")
        oof[arm] = {
            "prob": rows["prob"].to_numpy(dtype=float),
            "sp": rows["sp"].to_numpy(dtype=float),
            "inv": rows["inv"].to_numpy(dtype=float),
        }
        if any(len(values) != n for values in oof[arm].values()):
            raise ValueError(f"checkpoint prediction length mismatch for arm {arm}")
    return oof, metadata.get("info_rows", [])


def _write_checkpoint_manifest(
    cohorts: list[dict],
    completed: set[tuple[str, str, int]],
    status: str = "running",
) -> None:
    entries = [
        {
            "dataset": c["dataset"], "origin": c["origin"], "seed": seed,
            "checkpoint": _checkpoint_paths(c["dataset"], c["origin"], seed)[0].name,
            "complete": (c["dataset"], c["origin"], seed) in completed,
        }
        for c in cohorts for seed in SEEDS
    ]
    _atomic_write_text(_checkpoint_dir() / "manifest.json", json.dumps({
        "schema_version": CHECKPOINT_VERSION,
        "status": status,
        "implementation_fingerprint": _implementation_fingerprint(),
        "arms": list(ARMS),
        "seeds": SEEDS,
        "entries": entries,
    }, indent=2))


def _metrics_for_oof(cohort: dict, oof: dict, seed: int) -> list[dict]:
    full = np.arange(len(cohort["df"]))[None, :]
    rows = []
    for arm in ARMS:
        point = {k: float(v[0]) for k, v in bl.metric_rows(cohort["df"], oof[arm], full).items()}
        rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"],
                     "seed": seed, "arm": arm, **point})
    return rows


def _load_or_fit(cohort: dict, seed: int, resume: bool) -> tuple[dict, list[dict]]:
    if resume:
        try:
            oof, info = _load_checkpoint(cohort, seed)
            _validate_oof_arms(oof, cohort["dataset"], cohort["origin"])
            _log(f"resume: loaded seed={seed} {cohort['dataset']} {cohort['origin']}")
            return oof, info
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            _log(
                f"resume: recomputing seed={seed} {cohort['dataset']} {cohort['origin']} "
                f"({exc})"
            )
    with _dataset_extension_state(cohort["dataset"]):
        oof, info = bl.run_origin(cohort)
    _validate_oof_arms(oof, cohort["dataset"], cohort["origin"])
    _write_checkpoint(cohort, seed, oof, info)
    return oof, info


def build_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame):
    if arm in ("spline_log_rfm", "fuzzy_rfm_fca", "hybrid_fuzzy_fca"):
        # These reference arms deliberately use the committed RFM/v2 builders.
        builder = _V2_BUILD_FEATURES if arm == "hybrid_fuzzy_fca" else _V1_BUILD_FEATURES
        with _pipeline_state(RFM_DIMS, (arm,), builder):
            return builder(arm, tr, te)
    return _lrfm_features(arm, tr, te)


def verify_cutpoints(cohorts: list[dict]) -> None:
    for cohort in cohorts:
        if cohort["dataset"] not in ("Online Retail II", "Dunnhumby"):
            continue
        df = cohort["df"]
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=seed)
            for fold, (i_tr, _) in enumerate(skf.split(df, df["repurchased"])):
                tr = df.iloc[i_tr]
                for dim in RFM_DIMS:
                    ordinary = fit_quantile_cutoffs(tr, dims=(dim,), n_levels=5)[dim]
                    merged = merged_cutpoints(tr[dim].to_numpy(float))
                    if not np.array_equal(ordinary, merged):
                        raise RuntimeError(
                            f"cutpoint guard failed: {cohort['dataset']} {cohort['origin']} "
                            f"seed {seed} fold {fold} dimension {dim}"
                        )


def _committed_metrics(dataset: str) -> pd.DataFrame:
    r = ROOT / "results"
    if dataset == "CDNOW":
        m = pd.read_csv(r / "cdnow_confirmation" / "per_origin_metrics.csv")
        return m[m.arm.isin(["spline_log_rfm", "fuzzy_rfm_fca", "hybrid_fuzzy_fca"])]
    name = dataset
    a = pd.read_csv(r / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv")
    v = pd.read_csv(r / "v2_hybrid" / "per_origin_metrics.csv")
    return pd.concat([a[a.arm.isin(["spline_log_rfm", "fuzzy_rfm_fca"])],
                      v[v.arm == "hybrid_fuzzy_fca"]]).query("dataset == @name and pooled")


def verify_reproduction(cohorts: list[dict], tolerance: float = 1e-9) -> float:
    worst = 0.0
    for cohort in cohorts:
        if not cohort.get("pooled", True):
            continue
        dataset = cohort["dataset"]
        extension = _dataset_extension_state(dataset)
        if dataset != "CDNOW":
            _log(f"checking seed-0 reproduction for {dataset} {cohort['origin']}")

        def guard_builder(arm, tr, te):
            builder = _V2_BUILD_FEATURES if arm == "hybrid_fuzzy_fca" else _V1_BUILD_FEATURES
            return builder(arm, tr, te)

        with extension:
            with _pipeline_state(
                RFM_DIMS, ["spline_log_rfm", "fuzzy_rfm_fca", "hybrid_fuzzy_fca"], guard_builder
            ):
                bl.CV_SEED = 0
                oof, _ = bl.run_origin(cohort)
        reference = _committed_metrics(dataset)
        full = np.arange(len(cohort["df"]))[None, :]
        for arm in oof:
            got = {k: v[0] for k, v in bl.metric_rows(cohort["df"], oof[arm], full).items()}
            row = reference[(reference.origin == cohort["origin"]) & (reference.arm == arm)]
            if len(row) != 1:
                raise RuntimeError(f"missing committed guard row for {dataset} {cohort['origin']} {arm}")
            worst = max(worst, *(abs(got[m] - float(row.iloc[0][m])) for m in METRICS))
    if worst > tolerance:
        raise RuntimeError(f"seed-0 reproduction guard failed: max absolute difference {worst:.12g}")
    _log(f"seed-0 reproduction guard passed (max abs difference {worst:.3g})")
    return worst


def run_guards(cohorts: list[dict]) -> float:
    _log("running pre-analysis guards")
    verify_cutpoints(cohorts)
    _log("merged-cutpoint guard passed")
    return verify_reproduction(cohorts)


def _descriptives(cohort: dict) -> tuple[list[dict], list[dict]]:
    df = cohort["df"]
    l = df["L"]
    dist = [{"dataset": cohort["dataset"], "origin": cohort["origin"], "zero_share": float((l == 0).mean()),
             "q25": float(l.quantile(.25)), "median": float(l.median()),
             "q75": float(l.quantile(.75)), "max": float(l.max())}]
    scored, cuts = merged_scores(df)
    bands = [{"dataset": cohort["dataset"], "origin": cohort["origin"], "dimension": "L",
              "band": int(k), "n": int((scored["L_score"] == k).sum()),
              "levels": int(scored["L_score"].max())} for k in range(1, int(scored["L_score"].max()) + 1)]
    for dim in RFM_DIMS:
        dist[0][f"spearman_L_{dim}"] = float(spearmanr(df["L"], df[dim]).statistic)
    return dist, bands


def run_experiment(cohorts: list[dict], resume: bool = False) -> None:
    reproduction = run_guards(cohorts)
    OUT.mkdir(parents=True, exist_ok=True)
    _checkpoint_dir().mkdir(parents=True, exist_ok=True)
    _log("guards passed; writing exploratory LRFM results")
    all_metrics, all_info, all_dist, all_bands, predictions = [], [], [], [], {}
    completed = set()
    _write_checkpoint_manifest(cohorts, completed)
    for seed in SEEDS:
        bl.CV_SEED = seed
        with _pipeline_state(LRFM_DIMS, ARMS, build_features):
            for cohort in cohorts:
                _log(f"seed={seed} {cohort['dataset']} {cohort['origin']}")
                oof, info = _load_or_fit(cohort, seed, resume)
                completed.add((cohort["dataset"], cohort["origin"], seed))
                _write_checkpoint_manifest(cohorts, completed)
                predictions[(seed, cohort["dataset"], cohort["origin"])] = (cohort["df"], oof)
                all_info.extend([{**x, "seed": seed} for x in info])
                if seed == 0:
                    d, b = _descriptives(cohort)
                    all_dist.extend(d); all_bands.extend(b)
                all_metrics.extend(_metrics_for_oof(cohort, oof, seed))
    _write_inference(cohorts, predictions, pd.DataFrame(all_metrics), reproduction)
    pd.DataFrame(all_info).to_csv(OUT / "concept_counts_per_fold.csv", index=False)
    pd.DataFrame(all_dist).to_csv(OUT / "l_distribution.csv", index=False)
    pd.DataFrame(all_bands).to_csv(OUT / "l_band_occupancy.csv", index=False)
    _write_checkpoint_manifest(cohorts, completed, status="complete")


def _dataset_bootstrap_population(cohorts: list[dict], dataset: str) -> tuple[list[dict], pd.Index, dict]:
    """Return pooled origins and customer positions for one dataset only."""
    dataset_cohorts = [
        c for c in cohorts
        if c["dataset"] == dataset and c.get("pooled", True)
    ]
    union = pd.Index(sorted(set().union(*[
        set(c["df"]["CustomerID"]) for c in dataset_cohorts
    ])))
    positions = {
        (dataset, c["origin"]): union.get_indexer(c["df"]["CustomerID"])
        for c in dataset_cohorts
    }
    return dataset_cohorts, union, positions


def _mean_origin_deltas(origin_deltas: list[np.ndarray]) -> np.ndarray:
    if not origin_deltas:
        raise ValueError("cannot average bootstrap deltas without pooled origins")
    return np.mean(np.stack(origin_deltas, axis=0), axis=0)


def _combine_bootstrap_draws(draws_by_seed: list[list[np.ndarray]], expected: int) -> np.ndarray:
    """Flatten per-chunk, per-seed paired draws and enforce their count."""
    mix = np.concatenate([
        np.concatenate([np.asarray(chunk, dtype=float).reshape(-1) for chunk in seed_draws])
        for seed_draws in draws_by_seed
    ])
    if len(mix) != expected:
        raise RuntimeError(f"bootstrap draw count mismatch: got {len(mix)}, expected {expected}")
    return mix


def _write_inference(cohorts, predictions, metrics, reproduction):
    rng = np.random.default_rng(qw.BOOT_SEED)
    rows = []
    for dataset in sorted(metrics.dataset.unique()):
        cs, union, positions = _dataset_bootstrap_population(cohorts, dataset)
        draws = {(s, cid, m): [] for s in SEEDS for cid, _, _ in COMPARISONS for m in METRICS}
        points = {(s, cid, m): [] for s in SEEDS for cid, _, _ in COMPARISONS for m in METRICS}
        for seed in SEEDS:
            for cid, a, b in COMPARISONS:
                for m in METRICS:
                    for c in cs:
                        df, oof = predictions[(seed, c["dataset"], c["origin"])]
                        metric_a = qw.point_metrics(df, oof[a], np.ones((1, len(df))))[m][0]
                        metric_b = qw.point_metrics(df, oof[b], np.ones((1, len(df))))[m][0]
                        points[(seed, cid, m)].append(metric_a - metric_b)
        for start in range(0, BOOT_N, qw.CHUNK):
            n = min(qw.CHUNK, BOOT_N - start)
            idx = rng.integers(0, len(union), size=(n, len(union)))
            counts = np.zeros((n, len(union)))
            np.add.at(counts, (np.repeat(np.arange(n), len(union)), idx.ravel()), 1.0)
            for seed in SEEDS:
                for cid, a, b in COMPARISONS:
                    origin_deltas = {m: [] for m in METRICS}
                    for c in cs:
                        df, oof = predictions[(seed, c["dataset"], c["origin"])]
                        W = counts[:, positions[(dataset, c["origin"])]]
                        pa = qw.point_metrics(df, oof[a], W)
                        pb = qw.point_metrics(df, oof[b], W)
                        for m in METRICS:
                            origin_deltas[m].append(pa[m] - pb[m])
                    for m in METRICS:
                        draws[(seed, cid, m)].append(_mean_origin_deltas(origin_deltas[m]))
        for cid, a, b in COMPARISONS:
            for m in METRICS:
                expected_draws = len(SEEDS) * BOOT_N
                mix = _combine_bootstrap_draws(
                    [draws[(s, cid, m)] for s in SEEDS],
                    expected_draws,
                )
                p = min(1.0, max(2 * min(np.mean(mix <= 0), np.mean(mix >= 0)), 1 / len(mix)))
                per_seed = [np.mean(points[(s, cid, m)]) for s in SEEDS]
                rows.append({"dataset": dataset, "id": cid, "comparison": f"{a} - {b}", "metric": m,
                             "delta_mean_over_seeds": float(np.mean(per_seed)),
                             "delta_seed0": float(per_seed[0]), "delta_min_seed": float(min(per_seed)),
                             "delta_max_seed": float(max(per_seed)), "ci_low": float(np.percentile(mix, 2.5)),
                             "ci_high": float(np.percentile(mix, 97.5)), "p_boot": p,
                             "seed0_reproduction_max_abs_diff": reproduction})
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby("dataset")["p_boot"].transform(lambda x: qw.holm(x.to_numpy()))
    result.to_csv(OUT / "paired_comparisons.csv", index=False)
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({
        "exploratory": True, "dims": LRFM_DIMS, "seeds": SEEDS, "bootstrap_draws_per_seed": BOOT_N,
        "comparisons": [f"{i}: {a} - {b}" for i, a, b in COMPARISONS],
        "holm_tests_per_dataset": 15, "holdout_days": 91,
    }, indent=2), encoding="utf-8")
    REPORT.write_text(
        "# Exploratory LRFM extension results\n\n"
        "This analysis is exploratory and cannot confirm prior findings. "
        "Non-significance is not evidence of equivalence.\n\n"
        "## Locked protocol\n\n"
        f"- CV fold seeds: {SEEDS}; customer-clustered paired bootstrap: {BOOT_N:,} draws per seed.\n"
        "- The reported mixture interval and two-sided raw p-value use all 50,000 draws; "
        "Holm correction is within each dataset over 15 tests.\n"
        "- Estimates are dataset-specific and are not pooled across datasets. "
        "The table reports the mean over seeds, seed range, 95% percentile interval, "
        "raw p-value, and Holm-adjusted p-value.\n\n"
        "## Inference results\n\n" + bl._md(result) + "\n\n"
        "## Limitations\n\n"
        "- L is censored by the beginning of each observation window for Online Retail II "
        "and Dunnhumby; this was not corrected.\n"
        "- Origins overlap in customers and history, and the bootstrap reflects test-sample "
        "variability rather than refitting variability.\n"
        "- The same datasets were already analysed, so these results cannot confirm the "
        "prior evidence. A nonsignificant result must not be interpreted as equivalence.\n"
        "- No parameters were tuned; concept-mining failures stop the run rather than "
        "changing the precommitted settings.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="run guards only")
    parser.add_argument("--resume", action="store_true",
                        help="reuse validated per-cohort OOF checkpoints and recompute missing runs")
    args = parser.parse_args()
    t0 = time.time()
    dh = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2 = load_cleaned_transactions()
    cd = cc.load_transactions()
    cohorts = dunnhumby_cohorts(dh) + retail2_cohorts(r2) + cdnow_cohorts(cd)
    if args.skip_run:
        run_guards(cohorts)
    else:
        run_experiment(cohorts, resume=args.resume)
    _log(f"completed in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()

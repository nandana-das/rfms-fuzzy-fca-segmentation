"""Exploratory ablation of continuous RFM tail information.

This driver imports the frozen baseline pipeline and adds only the
tail-augmented feature concatenation. It never writes to the baseline output
directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402

OUT = ROOT / "results" / "rfm_tail_information"
ARMS = ("fuzzy_rfm_fca", "tail_augmented_fuzzy_rfm_fca", "spline_log_rfm")
METRICS = ("auc", "spend_r2", "invoice_r2")
COMPARISONS = (
    ("tail_augmented_fuzzy_rfm_fca", "fuzzy_rfm_fca"),
    ("tail_augmented_fuzzy_rfm_fca", "spline_log_rfm"),
    ("fuzzy_rfm_fca", "spline_log_rfm"),
)
BOOT_N = bl.BOOT_N
BOOT_SEED = bl.BOOT_SEED


def _tail_log(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[list(bl.DIMS)].to_numpy(dtype=float))


def build_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame):
    if arm != "tail_augmented_fuzzy_rfm_fca":
        return bl.build_features(arm, tr, te)
    f_tr, f_te, info = bl.build_features("fuzzy_rfm_fca", tr, te)
    scaler = StandardScaler().fit(_tail_log(tr))
    x_tr = np.column_stack((f_tr, scaler.transform(_tail_log(tr))))
    x_te = np.column_stack((f_te, scaler.transform(_tail_log(te))))
    return x_tr, x_te, {**info, "tail_features": 3}


def run_origin(cohort: dict) -> tuple[dict, list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    df = cohort["df"]
    oof = {a: {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")} for a in ARMS}
    info_rows = []
    splits = []
    skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        splits.append((i_tr.copy(), i_te.copy()))
        tr = df.iloc[i_tr].reset_index(drop=True)
        te = df.iloc[i_te].reset_index(drop=True)
        for arm in ARMS:
            x_tr, x_te, info = build_features(arm, tr, te)
            p, s, v = bl.fit_predict(arm="fuzzy_rfm_fca" if arm.startswith("tail_") else arm,
                                     X_tr=x_tr, X_te=x_te, tr=tr, seed=bl.CV_SEED + fold)
            oof[arm]["prob"][i_te] = p
            oof[arm]["sp"][i_te] = s
            oof[arm]["inv"][i_te] = v
            if info:
                info_rows.append({
                    "dataset": cohort["dataset"], "origin": cohort["origin"],
                    "fold": fold, "arm": arm, **info,
                })
    return oof, info_rows, splits


def _assert_complete(oof: dict) -> None:
    for arm in ARMS:
        for metric in ("prob", "sp", "inv"):
            if not np.isfinite(oof[arm][metric]).all():
                raise RuntimeError(f"incomplete out-of-fold predictions for {arm}/{metric}")


def verify_frozen_reproduction(cohort: dict) -> None:
    reference, _ = bl.run_origin(cohort)
    candidate, _, _ = run_origin(cohort)
    for metric in ("prob", "sp", "inv"):
        diff = np.max(np.abs(reference["fuzzy_rfm_fca"][metric] - candidate["fuzzy_rfm_fca"][metric]))
        if diff > 1e-12:
            raise RuntimeError(f"frozen fuzzy reproduction failed: {metric} max diff {diff}")


def _boot_p(draws: np.ndarray) -> float:
    return float(min(1.0, max(2 * min(np.mean(draws <= 0), np.mean(draws >= 0)), 1 / len(draws))))


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    out = np.empty(len(values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(values) - rank) * values[i])
        out[i] = min(1.0, running)
    return out.tolist()


def _safe_output() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to mix with existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)


def _load_cohorts() -> list[dict]:
    dh = pd.read_csv(bl.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    r2 = bl.load_cleaned_transactions()
    return bl.dunnhumby_cohorts(dh) + bl.retail2_cohorts(r2)


def run_experiment(smoke: bool = False) -> None:
    cohorts = _load_cohorts()
    selected = cohorts[:1] if smoke else cohorts
    for cohort in selected:
        verify_frozen_reproduction(cohort)
    _safe_output()
    started = time.time()
    rng = np.random.default_rng(BOOT_SEED)
    metric_rows = []
    info_rows = []
    boot = {}
    for cohort in selected:
        _pr(f"[{cohort['dataset']} | {cohort['origin']}]")
        oof, info, splits = run_origin(cohort)
        _assert_complete(oof)
        info_rows.extend(info)
        full = np.arange(len(cohort["df"]))[None, :]
        boot_idx = rng.integers(0, len(cohort["df"]), size=(BOOT_N, len(cohort["df"])))
        for arm in ARMS:
            vals = bl.metric_rows(cohort["df"], oof[arm], full)
            metric_rows.append({
                "dataset": cohort["dataset"], "origin": cohort["origin"],
                "pooled": cohort["pooled"], "n": len(cohort["df"]),
                "arm": arm, **{m: float(v[0]) for m, v in vals.items()},
            })
            boot[(cohort["dataset"], cohort["origin"], arm)] = bl.metric_rows(
                cohort["df"], oof[arm], boot_idx
            )
    metrics = pd.DataFrame(metric_rows)
    comparisons = []
    for dataset in metrics["dataset"].unique():
        ds = metrics[metrics["dataset"] == dataset]
        pooled = list(ds.loc[ds["pooled"], "origin"].drop_duplicates())
        for arm, ref in COMPARISONS:
            for metric in METRICS:
                pooled_draws = []
                deltas = []
                for origin in pooled:
                    a = ds[(ds.origin == origin) & (ds.arm == arm)][metric].iloc[0]
                    b = ds[(ds.origin == origin) & (ds.arm == ref)][metric].iloc[0]
                    deltas.append(a - b)
                    pooled_draws.append(
                        boot[(dataset, origin, arm)][metric]
                        - boot[(dataset, origin, ref)][metric]
                    )
                draws = np.mean(pooled_draws, axis=0)
                comparisons.append({
                    "dataset": dataset, "comparison": f"{arm} - {ref}",
                    "metric": metric, "delta": float(np.mean(deltas)),
                    "ci_low": float(np.percentile(draws, 2.5)),
                    "ci_high": float(np.percentile(draws, 97.5)),
                    "p_boot": _boot_p(draws),
                    "origins_positive": f"{sum(d > 0 for d in deltas)}/{len(deltas)}",
                })
    comp = pd.DataFrame(comparisons)
    comp["p_holm"] = np.nan
    for dataset in comp.dataset.unique():
        mask = comp.dataset == dataset
        comp.loc[mask, "p_holm"] = _holm(comp.loc[mask, "p_boot"].tolist())
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(info_rows).to_csv(OUT / "concept_counts_per_fold.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({
        "exploratory": True, "arms": ARMS, "comparisons": COMPARISONS,
        "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "n_folds": bl.N_FOLDS,
        "cv_seed": bl.CV_SEED, "source": "baseline_ladder_rolling_origin.py",
        "smoke": smoke, "cohorts": [f"{c['dataset']} | {c['origin']}" for c in selected],
    }, indent=2, default=str), encoding="utf-8")
    _pr(f"Completed in {(time.time() - started) / 60:.1f} minutes: {OUT}")


def _pr(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_experiment(smoke=args.smoke)


if __name__ == "__main__":
    main()

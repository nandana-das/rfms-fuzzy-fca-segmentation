"""Non-FCA baselines for the temporal (T) ablation.

Question: is the Online Retail II gain from adding T an FCA effect, or does T
carry the same information into a plain log/spline RFM model?

Arms (all fit inside each customer training fold; same cohorts, folds,
learners, outcomes, and bootstrap as scripts/temporal_rfm_fca_ablation.py):
  log_rfm            standardized log1p(R, F, M)                 (existing baseline)
  spline_log_rfm     cubic spline on log1p(R, F, M)              (existing baseline)
  log_rfm_t          standardized log1p(R, F, M, T) + repeat_history
  spline_log_rfm_t   cubic spline on log1p(R, F, M, T) + repeat_history
  fuzzy_rfm          M0 fuzzy RFM-FCA                            (recomputed for pairing)
  fuzzy_rfm_t        fuzzy RFM+T FCA                             (recomputed for pairing)

T is imputed with the training-fold median, exactly as in the FCA T arm.
L is out of scope.

Comparisons fixed before the run (Holm within dataset over 4 x 3 = 12 pooled tests):
  fuzzy_rfm_t - log_rfm_t
  fuzzy_rfm_t - spline_log_rfm_t
  log_rfm_t - log_rfm
  spline_log_rfm_t - spline_log_rfm
Exploratory, post hoc follow-up to the temporal ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from fair_comparison_retail2 import load_cleaned_transactions  # noqa: E402
from fuzzy_membership_sensitivity import TRANSACTION_FILE  # noqa: E402
from temporal_rfm_fca_ablation import (  # noqa: E402
    _fit_predict,
    _pooled_point_deltas,
    _temporal_features,
    temporal_cohorts,
)

OUT = ROOT / "results" / "temporal_nonfca_baseline"
ARMS = ["log_rfm", "spline_log_rfm", "log_rfm_t", "spline_log_rfm_t", "fuzzy_rfm", "fuzzy_rfm_t"]
PAIRS = [
    ("fuzzy_rfm_t", "log_rfm_t"),
    ("fuzzy_rfm_t", "spline_log_rfm_t"),
    ("log_rfm_t", "log_rfm"),
    ("spline_log_rfm_t", "spline_log_rfm"),
]
METRICS = ["auc", "spend_r2", "invoice_r2"]
LOG_DIMS = ("R", "F", "M", "T")


def _impute_t(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = float(tr["T"].median())
    if not np.isfinite(med):
        raise ValueError("Training fold has no observed repeat intervals for T imputation")
    return tr.assign(T=tr["T"].fillna(med)), te.assign(T=te["T"].fillna(med))


def _log_t(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[list(LOG_DIMS)].to_numpy(dtype=float))


def nonfca_t_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """log_rfm_t / spline_log_rfm_t features; every transform is fit on tr only."""
    tr, te = _impute_t(tr, te)
    rep_tr = tr[["repeat_history"]].to_numpy(dtype=float)
    rep_te = te[["repeat_history"]].to_numpy(dtype=float)
    if arm == "log_rfm_t":
        x_tr, x_te = np.hstack([_log_t(tr), rep_tr]), np.hstack([_log_t(te), rep_te])
        sc = StandardScaler().fit(x_tr)
        return sc.transform(x_tr), sc.transform(x_te)
    if arm == "spline_log_rfm_t":
        sp = SplineTransformer(n_knots=5, degree=3, knots="quantile").fit(_log_t(tr))
        return np.hstack([sp.transform(_log_t(tr)), rep_tr]), np.hstack([sp.transform(_log_t(te)), rep_te])
    raise ValueError(arm)


def build(arm: str, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if arm in ("log_rfm", "spline_log_rfm"):
        x_tr, x_te, _ = bl.build_features(arm, tr, te)
    elif arm == "fuzzy_rfm":
        x_tr, x_te, _ = bl.build_features("fuzzy_rfm_fca", tr, te)
    elif arm == "fuzzy_rfm_t":
        x_tr, x_te, _ = _temporal_features("fuzzy_rfm_t", tr, te)
    else:
        x_tr, x_te = nonfca_t_features(arm, tr, te)
    return x_tr, x_te


def run_origin(cohort: dict, cv_seed: int = bl.CV_SEED, arms: list[str] = ARMS):
    df = cohort["df"].reset_index(drop=True)
    oof = {a: {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")} for a in arms}
    n_feat = []
    skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=cv_seed)
    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
        for arm in arms:
            x_tr, x_te = build(arm, tr, te)
            p, s, v = _fit_predict(arm, x_tr, x_te, tr, cv_seed + fold)
            oof[arm]["prob"][i_te], oof[arm]["sp"][i_te], oof[arm]["inv"][i_te] = p, s, v
            n_feat.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold,
                           "arm": arm, "n_features": int(x_tr.shape[1])})
    return oof, n_feat


def compare(metrics: pd.DataFrame, boots: dict, pairs: list[tuple[str, str]] = PAIRS) -> pd.DataFrame:
    rows = []
    for dataset in metrics.dataset.unique():
        sub = metrics[metrics.dataset == dataset]
        origins = list(dict.fromkeys(sub.origin))
        pooled = list(dict.fromkeys(sub.loc[sub.pooled, "origin"]))
        for arm, ref in pairs:
            for metric in METRICS:
                draws = {}
                for origin in origins:
                    a = sub[(sub.origin == origin) & (sub.arm == arm)][metric].iloc[0]
                    b = sub[(sub.origin == origin) & (sub.arm == ref)][metric].iloc[0]
                    d = boots[(dataset, origin, arm)][metric] - boots[(dataset, origin, ref)][metric]
                    draws[origin] = d
                    rows.append({"dataset": dataset, "scope": origin, "comparison": f"{arm} - {ref}",
                                 "metric": metric, "delta": a - b, "ci_low": np.percentile(d, 2.5),
                                 "ci_high": np.percentile(d, 97.5), "p_boot": bl._boot_p(d)})
                pooled_draws = np.mean([draws[o] for o in pooled], axis=0)
                deltas = _pooled_point_deltas(sub, arm, ref, metric, pooled)
                rows.append({"dataset": dataset, "scope": "POOLED", "comparison": f"{arm} - {ref}",
                             "metric": metric, "delta": float(np.mean(deltas)),
                             "ci_low": np.percentile(pooled_draws, 2.5),
                             "ci_high": np.percentile(pooled_draws, 97.5),
                             "p_boot": bl._boot_p(pooled_draws),
                             "origins_positive": f"{sum(d > 0 for d in deltas)}/{len(deltas)}"})
    comp = pd.DataFrame(rows)
    comp["p_holm"] = np.nan
    for dataset in comp.dataset.unique():
        mask = (comp.dataset == dataset) & (comp.scope == "POOLED")
        comp.loc[mask, "p_holm"] = bl._holm(comp.loc[mask, "p_boot"].tolist())
    return comp


def write_report(metrics: pd.DataFrame, comp: pd.DataFrame, feats: pd.DataFrame, runtime: float,
                 out: Path = OUT, cv_seed: int = bl.CV_SEED, command: str = "python scripts/temporal_nonfca_baseline.py") -> None:
    means = metrics[metrics.pooled].groupby(["dataset", "arm"], sort=False)[METRICS].mean().reset_index()
    nf = feats.groupby(["dataset", "arm"], sort=False)["n_features"].mean().reset_index()
    lines = [
        "# Non-FCA baselines for the temporal ablation", "",
        "Exploratory, post hoc follow-up. Tests whether the gain from adding T to fuzzy RFM-FCA "
        "also appears when T is added to plain log-RFM and spline-RFM models without FCA.", "",
        "All arms use the temporal-ablation cohorts, 91-day holdouts, customer folds "
        f"(seed {cv_seed}), learners (LogisticRegressionCV / RidgeCV), and paired customer "
        f"bootstrap (B={bl.BOOT_N}). T is imputed with the training-fold median and every transform "
        "is fit on the training fold only. L is out of scope.", "",
        "Pooled intervals average per-origin bootstrap draws; origins overlap in customers, so "
        "pooled intervals are approximate. Single CV seed.", "",
        "## Pooled means", "",
        "| dataset | arm | features | auc | spend_r2 | invoice_r2 |", "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in means.iterrows():
        k = nf[(nf.dataset == r.dataset) & (nf.arm == r.arm)].n_features.iloc[0]
        lines.append(f"| {r.dataset} | {r.arm} | {k:.1f} | {r.auc:.4f} | {r.spend_r2:.4f} | {r.invoice_r2:.4f} |")
    lines += ["", "## Pooled paired comparisons", "",
              f"Delta = first arm minus second arm. Holm within dataset over the {len(comp[comp.scope == 'POOLED']) // comp.dataset.nunique()} pooled tests.", "",
              "| dataset | comparison | metric | delta | CI low | CI high | p Holm | positive origins |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for _, r in comp[comp.scope == "POOLED"].iterrows():
        lines.append(f"| {r.dataset} | {r.comparison} | {r.metric} | {r.delta:+.4f} | {r.ci_low:+.4f} | "
                     f"{r.ci_high:+.4f} | {r.p_holm:.4f} | {r.origins_positive} |")
    lines += ["", "Per-origin metrics and comparisons: `per_origin_metrics.csv`, `paired_comparisons.csv`.",
              "", f"Command: `{command}`", f"Runtime: {runtime / 60:.1f} minutes.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-seed", type=int, default=bl.CV_SEED)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--pairs", nargs="+", default=[f"{a}:{b}" for a, b in PAIRS], help="arm:reference")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    pairs = [tuple(p.split(":")) for p in args.pairs]
    unknown = {a for p in pairs for a in p} - set(args.arms) | set(args.arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown or missing arms: {sorted(unknown)}")
    start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dh = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = temporal_cohorts(dh, load_cleaned_transactions())
    rng = np.random.default_rng(bl.BOOT_SEED)
    metric_rows, feat_rows, boots = [], [], {}
    for c in cohorts:
        print(f"[{c['dataset']} | {c['origin']}] n={len(c['df'])}", flush=True)
        oof, nf = run_origin(c, args.cv_seed, args.arms)
        feat_rows.extend(nf)
        full = np.arange(len(c["df"]))[None, :]
        idx = rng.integers(0, len(c["df"]), size=(bl.BOOT_N, len(c["df"])))
        for arm in args.arms:
            point = {k: float(v[0]) for k, v in bl.metric_rows(c["df"], oof[arm], full).items()}
            metric_rows.append({"dataset": c["dataset"], "origin": c["origin"], "pooled": c["pooled"],
                                "n": len(c["df"]), "arm": arm, **point})
            boots[(c["dataset"], c["origin"], arm)] = bl.metric_rows(c["df"], oof[arm], idx)
    metrics = pd.DataFrame(metric_rows)
    comp = compare(metrics, boots, pairs)
    feats = pd.DataFrame(feat_rows)
    metrics.to_csv(out / "per_origin_metrics.csv", index=False)
    comp.to_csv(out / "paired_comparisons.csv", index=False)
    feats.to_csv(out / "feature_counts_per_fold.csv", index=False)
    (out / "run_parameters.json").write_text(json.dumps({
        "arms": args.arms, "pairs": [f"{a} - {b}" for a, b in pairs],
        "holm_family": f"{len(pairs) * len(METRICS)} pooled tests per dataset",
        "t_imputation": "training-fold median", "spline": "SplineTransformer(n_knots=5, degree=3, knots='quantile') on log1p",
        "horizon_days": bl.HORIZON_DAYS, "n_folds": bl.N_FOLDS, "cv_seed": args.cv_seed,
        "learner_seed": "cv_seed + fold", "boot_n": bl.BOOT_N, "boot_seed": bl.BOOT_SEED,
    }, indent=2), encoding="utf-8")
    command = "python scripts/temporal_nonfca_baseline.py" + (" " + " ".join(sys.argv[1:]) if argv is None and len(sys.argv) > 1 else "")
    write_report(metrics, comp, feats, time.time() - start, out, args.cv_seed, command)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()

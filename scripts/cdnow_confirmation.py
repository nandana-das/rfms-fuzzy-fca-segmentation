"""CDNOW out-of-sample confirmation of frozen v1 (H1, H2) and exploratory v2.

Plan (committed before this script): docs/CDNOW_CONFIRMATION_PLAN.md.
Data: data/cdnow_raw/CDNOW_master.txt (git-ignored; source and SHA-256 in the plan).

Reuses the frozen ladder (scripts/baseline_ladder_rolling_origin.py) and the v2
hybrid builder (scripts/v2_hybrid_ladder.py) unchanged. The only addition is the
planned tie-merging extension, applied inside this process only:
    * quintile cutpoints that coincide are merged (fewer levels for that dimension);
    * fuzzy memberships use the generalized n-level centroid/piecewise-linear
      functions, identical to the frozen 5-level ones when there are no ties.
Frozen v1/v2 scripts and results are not affected.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
import v2_hybrid_ladder as v2  # noqa: E402
from fair_comparison_retail2 import quantile_scores  # noqa: E402
from fuzzy_membership_sensitivity import _band_centroids_generic, _piecewise_membership_generic  # noqa: E402

DATA_FILE = ROOT_DIR / "data" / "cdnow_raw" / "CDNOW_master.txt"
OUT = ROOT_DIR / "results" / "cdnow_confirmation"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "CDNOW"
ORIGINS = [pd.Timestamp("1997-09-30"), pd.Timestamp("1997-12-31"), pd.Timestamp("1998-03-31")]
HORIZON = pd.Timedelta(days=91)
ARMS = ["raw_std", "log_rfm", "spline_log_rfm", "crisp_rfm_fca", "fuzzy_rfm_fca",
        "hybrid_crisp_fca", "hybrid_fuzzy_fca"]
COMPARISONS = [  # plan §5, C1..C7
    ("C1", "fuzzy_rfm_fca", "crisp_rfm_fca"),
    ("C2", "fuzzy_rfm_fca", "spline_log_rfm"),
    ("C3", "hybrid_fuzzy_fca", "spline_log_rfm"),
    ("C4", "hybrid_fuzzy_fca", "fuzzy_rfm_fca"),
    ("C5", "hybrid_fuzzy_fca", "hybrid_crisp_fca"),
    ("C6", "hybrid_fuzzy_fca", "log_rfm"),
    ("C7", "hybrid_fuzzy_fca", "crisp_rfm_fca"),
]
METRICS = ["auc", "spend_r2", "invoice_r2"]


# ---------------------------------------------------------------------------
# Data (plan §2)
# ---------------------------------------------------------------------------

def load_transactions() -> pd.DataFrame:
    tx = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=["cust", "date", "cds", "dollars"],
                     dtype={"cust": str, "date": str})
    tx["date"] = pd.to_datetime(tx["date"], format="%Y%m%d")
    return tx[tx["dollars"] > 0].reset_index(drop=True)


def cdnow_cohorts(tx: pd.DataFrame) -> list[dict]:
    out = []
    for cutoff in ORIGINS:
        obs = tx[tx["date"] <= cutoff]
        hold = tx[(tx["date"] > cutoff) & (tx["date"] <= cutoff + HORIZON)]
        agg = obs.groupby("cust").agg(last=("date", "max"), F=("date", "nunique"), M=("dollars", "sum")).reset_index()
        agg["R"] = (cutoff - agg["last"]).dt.days
        fut = hold.groupby("cust").agg(future_invoices=("date", "nunique"), future_spend=("dollars", "sum")).reset_index()
        df = agg.merge(fut, on="cust", how="left")
        df["CustomerID"] = df["cust"]
        out.append(bl._finish(DATASET, str(cutoff.date()), df, pooled=True))
    return out


# ---------------------------------------------------------------------------
# Planned extension (plan §4): merge coinciding quintile cutpoints
# ---------------------------------------------------------------------------

def merged_cutpoints(x: np.ndarray) -> np.ndarray:
    """Quintile cutpoints with ties merged and empty bands merged into the next band.

    Linear-interpolated quantiles of a discrete variable can fall strictly between
    two observed values (e.g. 1.6 for counts), leaving a band with no training
    customers. Such a band is merged by deleting its upper cutpoint (or, for the top
    band, its lower one). This changes nothing when every band is populated
    (addendum A1 in docs/CDNOW_CONFIRMATION_PLAN.md).
    """
    cut = np.unique(np.quantile(x, np.arange(1, 5) / 5))
    while len(cut):
        bands = np.searchsorted(cut, x, side="left")  # 0..len(cut)
        counts = np.bincount(bands, minlength=len(cut) + 1)
        empty = np.flatnonzero(counts == 0)
        if not len(empty):
            break
        b = int(empty[0])
        cut = np.delete(cut, b if b < len(cut) else b - 1)
    return cut


def quintile_scored_merged(tr: pd.DataFrame):
    cut = {d: merged_cutpoints(tr[d].to_numpy(dtype=float)) for d in bl.DIMS}
    return quantile_scores(tr[["CustomerID", *bl.DIMS]].copy(), cut, dims=bl.DIMS), cut


def compute_fuzzy_memberships_n(scored: pd.DataFrame, trained_centroids=None, dims=bl.DIMS):
    memberships, centroid_dict = {}, {}
    for dim in dims:
        raw = scored[dim].to_numpy(dtype=float)
        if trained_centroids is not None:
            c = np.asarray(trained_centroids[dim], dtype=float)
        else:
            score = scored[f"{dim}_score"].to_numpy()
            n_levels = int(score.max())
            if set(np.unique(score)) != set(range(1, n_levels + 1)):
                raise ValueError(f"{dim}: score bands are not contiguous 1..{n_levels}")
            c = _band_centroids_generic(raw, score, n_levels)
        order = np.argsort(c, kind="stable")
        mu = _piecewise_membership_generic(raw, c[order])[:, np.argsort(order, kind="stable")]
        for k in range(len(c)):
            memberships[f"{dim}{k + 1}"] = mu[:, k]
        centroid_dict[dim] = c
    return pd.DataFrame(memberships), centroid_dict, pd.DataFrame()


def install_extension() -> None:
    bl._quintile_scored = quintile_scored_merged
    bl.compute_fuzzy_memberships = compute_fuzzy_memberships_n
    bl.ARMS = ARMS
    bl.build_features = v2.build_features  # v1 arms pass through to the frozen builder


# ---------------------------------------------------------------------------
# Run (plan §5–6)
# ---------------------------------------------------------------------------

def verdicts(comp: pd.DataFrame) -> pd.DataFrame:
    def sig(row, sign):
        return row["p_holm"] < 0.05 and np.sign(row["delta"]) == sign
    rows = []
    get = lambda cid: comp[comp["id"] == cid].set_index("metric")
    c1 = get("C1")
    pos = sum(sig(c1.loc[m], 1) for m in METRICS)
    neg = sum(sig(c1.loc[m], -1) for m in METRICS)
    h1 = "replicates" if pos == 3 else ("partially replicates" if pos >= 1 and neg == 0 else "fails to replicate")
    rows.append({"finding": "v1 H1 (fuzzy > crisp RFM-FCA)", "rule": "C1 significantly positive on all 3 metrics", "outcome": h1})
    for cid, finding in [("C2", "v1 H2 (fuzzy vs spline)"), ("C3", "v2 beats spline")]:
        c = get(cid)
        desc = "; ".join(
            f"{m}: " + ("significantly higher" if sig(c.loc[m], 1) else "significantly lower" if sig(c.loc[m], -1) else "not significantly different")
            for m in METRICS)
        rows.append({"finding": finding, "rule": f"{cid} per metric (non-significant is not equivalence)", "outcome": desc})
    c4 = get("C4")
    if any(sig(c4.loc[m], -1) for m in METRICS):
        out4 = "not confirmed (a metric is significantly negative)"
    else:
        ok = [m for m in METRICS if sig(c4.loc[m], 1)]
        out4 = f"confirmed on: {', '.join(ok)}" if ok else "not confirmed"
    rows.append({"finding": "v2 improves on v1", "rule": "C4 significantly positive, none significantly negative", "outcome": out4})
    for cid, finding in [("C6", "v2 concepts add beyond linear log RFM"), ("C5", "fuzzification helps within the hybrid")]:
        c = get(cid)
        ok = [m for m in METRICS if sig(c.loc[m], 1)]
        rows.append({"finding": finding, "rule": f"{cid} significantly positive (per metric)",
                     "outcome": f"supported on: {', '.join(ok)}" if ok else "not supported"})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    install_extension()
    tx = load_transactions()
    cohorts = cdnow_cohorts(tx)
    occupancy = pd.DataFrame([r for c in cohorts for r in bl.band_occupancy(c)])
    occupancy.to_csv(OUT / "band_occupancy.csv", index=False)

    rng = np.random.default_rng(bl.BOOT_SEED)
    metric_records, info_records, boot = [], [], {}
    for c in cohorts:
        df = c["df"]
        bl._pr(f"[{DATASET} | {c['origin']}] n={len(df):,} repurchase={df['repurchased'].mean():.4f}")
        oof, info = bl.run_origin(c)
        info_records.extend(info)
        full = np.arange(len(df))[None, :]
        idx = rng.integers(0, len(df), size=(bl.BOOT_N, len(df)))
        for arm in ARMS:
            point = {k: float(v[0]) for k, v in bl.metric_rows(df, oof[arm], full).items()}
            metric_records.append({"dataset": DATASET, "origin": c["origin"], "n": len(df),
                                   "repurchase_rate": float(df["repurchased"].mean()), "arm": arm, **point})
            boot[(c["origin"], arm)] = bl.metric_rows(df, oof[arm], idx)
    metrics = pd.DataFrame(metric_records)
    origins = [c["origin"] for c in cohorts]

    comp_rows, per_origin_rows = [], []
    for cid, arm, ref in COMPARISONS:
        for m in METRICS:
            deltas, draws = [], []
            for o in origins:
                a = metrics[(metrics.origin == o) & (metrics.arm == arm)][m].iloc[0]
                b = metrics[(metrics.origin == o) & (metrics.arm == ref)][m].iloc[0]
                d = boot[(o, arm)][m] - boot[(o, ref)][m]
                deltas.append(a - b)
                draws.append(d)
                per_origin_rows.append({"id": cid, "comparison": f"{arm} - {ref}", "metric": m, "origin": o,
                                        "delta": a - b, "ci_low": float(np.percentile(d, 2.5)),
                                        "ci_high": float(np.percentile(d, 97.5))})
            pooled = np.mean(draws, axis=0)
            comp_rows.append({"id": cid, "comparison": f"{arm} - {ref}", "metric": m, "delta": float(np.mean(deltas)),
                              "ci_low": float(np.percentile(pooled, 2.5)), "ci_high": float(np.percentile(pooled, 97.5)),
                              "p_boot": bl._boot_p(pooled), "origins_positive": f"{sum(x > 0 for x in deltas)}/{len(deltas)}"})
    comp = pd.DataFrame(comp_rows)
    comp["p_holm"] = bl._holm(comp["p_boot"].tolist())
    verdict = verdicts(comp)

    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    comp.to_csv(OUT / "comparisons.csv", index=False)
    pd.DataFrame(per_origin_rows).to_csv(OUT / "per_origin_comparisons.csv", index=False)
    pd.DataFrame(info_records).to_csv(OUT / "concept_counts_per_fold.csv", index=False)
    verdict.to_csv(OUT / "verdicts.csv", index=False)
    (OUT / "run_parameters.json").write_text(json.dumps({
        "plan": "docs/CDNOW_CONFIRMATION_PLAN.md", "data_sha256_zip": "94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49",
        "origins": [str(o.date()) for o in ORIGINS], "horizon_days": 91, "arms": ARMS,
        "comparisons": [f"{i}: {a} - {b}" for i, a, b in COMPARISONS], "holm_family": len(comp),
        "frozen": {"min_support": bl.MIN_SUPPORT, "l_thresholds": bl.L_THRESHOLDS, "j_max": bl.J_MAX,
                   "mu_cut": bl.MU_CUT, "n_folds": bl.N_FOLDS, "cv_seed": bl.CV_SEED, "boot_n": bl.BOOT_N,
                   "boot_seed": bl.BOOT_SEED},
        "extension": "merge coinciding quintile cutpoints; generalized n-level fuzzy memberships",
    }, indent=2), encoding="utf-8")

    means = metrics.groupby("arm", sort=False)[METRICS].mean().reset_index()
    lines = [
        "# CDNOW confirmation — results",
        "",
        "Plan: `docs/CDNOW_CONFIRMATION_PLAN.md` (committed before this script). One run; no tuning.",
        "",
        "## Decision-rule outcomes",
        "",
        bl._md(verdict),
        "",
        "## Mean metrics over the 3 origins",
        "",
        bl._md(means),
        "",
        f"## Comparisons (Holm over {len(comp)} tests; origins_positive is descriptive)",
        "",
        bl._md(comp),
        "",
        "## Cohorts",
        "",
        bl._md(metrics.drop_duplicates("origin")[["origin", "n", "repurchase_rate"]]),
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    bl._pr(verdict.to_string(index=False))


if __name__ == "__main__":
    main()

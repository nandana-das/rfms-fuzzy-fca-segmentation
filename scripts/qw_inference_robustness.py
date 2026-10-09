"""Part B of docs/QUICK_WINS_PLAN.md: inference robustness for the key comparisons.

Per dataset (run separately):  python scripts/qw_inference_robustness.py --dataset {dunnhumby,retail2,cdnow}
Then aggregate:                python scripts/qw_inference_robustness.py --aggregate

Predictions come from the frozen pipeline (baseline_ladder_rolling_origin.run_origin
with the v2 hybrid builder; CDNOW with its planned extension). Only the CV fold seed
changes (0-4); seed 0 must reproduce the committed per-origin metrics exactly.
Inference: customer-clustered bootstrap across origins (same customer draw applied
to every origin, as resample weights), B = 10,000, paired across arms and seeds;
mixture over the 5 seeds; Holm over 12 tests per dataset.
Writes only to results/qw_inference_robustness/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "qw_inference_robustness"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
B = 10_000
CHUNK = 250
BOOT_SEED = 20261009
ARMS = ["crisp_rfm_fca", "fuzzy_rfm_fca", "spline_log_rfm", "hybrid_fuzzy_fca"]
KEY = [("K1", "fuzzy_rfm_fca", "crisp_rfm_fca"), ("K2", "fuzzy_rfm_fca", "spline_log_rfm"),
       ("K3", "hybrid_fuzzy_fca", "spline_log_rfm"), ("K4", "hybrid_fuzzy_fca", "fuzzy_rfm_fca")]
METRICS = ["auc", "spend_r2", "invoice_r2"]


# ---------------------------------------------------------------------------
# Weighted metrics (resample weights = bootstrap counts)
# ---------------------------------------------------------------------------

def weighted_auc(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rows of W are weights. Ties count 1/2 (equivalent to average-rank AUC)."""
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted, Ws = s[order], y[order], W[:, order]
    starts = np.flatnonzero(np.r_[True, np.diff(s_sorted) != 0])
    wp = np.add.reduceat(Ws * y_sorted, starts, axis=1)
    wn = np.add.reduceat(Ws * (1 - y_sorted), starts, axis=1)
    neg_before = np.cumsum(wn, axis=1) - wn
    num = (wp * (neg_before + 0.5 * wn)).sum(axis=1)
    return num / (wp.sum(axis=1) * wn.sum(axis=1))


def weighted_r2(y: np.ndarray, f: np.ndarray, W: np.ndarray) -> np.ndarray:
    wsum = W.sum(axis=1)
    ss_res = W @ (y - f) ** 2
    mean = (W @ y) / wsum
    ss_tot = W @ (y ** 2) - wsum * mean ** 2
    return 1.0 - ss_res / ss_tot


def unit_check() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); s = np.round(rng.normal(size=400), 1); f = rng.normal(size=400); t = f + rng.normal(size=400)
    w = rng.integers(0, 4, 400)
    rep = np.repeat(np.arange(400), w)
    assert abs(weighted_auc(y, s, w[None, :].astype(float))[0] - roc_auc_score(y[rep], s[rep])) < 1e-12
    assert abs(weighted_r2(t, f, w[None, :].astype(float))[0] - r2_score(t[rep], f[rep])) < 1e-12


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def cohorts_for(dataset: str):
    import baseline_ladder_rolling_origin as bl
    import v2_hybrid_ladder as v2
    if dataset == "cdnow":
        import cdnow_confirmation as cc
        cc.install_extension()
        cohorts = cc.cdnow_cohorts(cc.load_transactions())
    elif dataset == "dunnhumby":
        tx = pd.read_csv(bl.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
        cohorts = bl.dunnhumby_cohorts(tx)
    else:
        cohorts = [c for c in bl.retail2_cohorts(bl.load_cleaned_transactions()) if c["pooled"]]
    bl.ARMS = ARMS
    bl.build_features = v2.build_features
    return bl, cohorts


def committed_point_metrics(dataset: str) -> pd.DataFrame:
    R = ROOT / "results"
    if dataset == "cdnow":
        m = pd.read_csv(R / "cdnow_confirmation" / "per_origin_metrics.csv")
        return m[m.arm.isin(ARMS)]
    name = "Dunnhumby" if dataset == "dunnhumby" else "Online Retail II"
    lad = pd.read_csv(R / "baseline_ladder_rolling_origin" / "per_origin_metrics.csv")
    v2m = pd.read_csv(R / "v2_hybrid" / "per_origin_metrics.csv")
    m = pd.concat([lad[lad.arm.isin(ARMS[:3])], v2m[v2m.arm == "hybrid_fuzzy_fca"]])
    return m[(m.dataset == name) & m.pooled]


def run_dataset(dataset: str) -> None:
    t0 = time.time()
    unit_check()
    bl, cohorts = cohorts_for(dataset)
    preds = {}  # (seed, origin) -> (df, oof)
    for seed in SEEDS:
        bl.CV_SEED = seed
        for c in cohorts:
            oof, _ = bl.run_origin(c)
            preds[(seed, c["origin"])] = (c["df"], oof)
        print(f"[{dataset}] seed {seed} predictions done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # seed 0 must reproduce committed per-origin point metrics
    ref = committed_point_metrics(dataset)
    worst = 0.0
    for c in cohorts:
        df, oof = preds[(0, c["origin"])]
        ones = np.ones((1, len(df)))
        for arm in ARMS:
            got = point_metrics(df, oof[arm], ones)
            row = ref[(ref.origin == c["origin"]) & (ref.arm == arm)].iloc[0]
            worst = max(worst, max(abs(got[m][0] - row[m]) for m in METRICS))
    print(f"[{dataset}] seed-0 reproduction max |diff| = {worst:.3g}", flush=True)
    if worst > 1e-9:
        raise SystemExit("Seed 0 does not reproduce the committed metrics; stopping before inference.")

    # point estimates per seed
    point = {}
    for seed in SEEDS:
        for cid, a, b in KEY:
            for m in METRICS:
                ds = []
                for c in cohorts:
                    df, oof = preds[(seed, c["origin"])]
                    ones = np.ones((1, len(df)))
                    ds.append(point_metrics(df, oof[a], ones)[m][0] - point_metrics(df, oof[b], ones)[m][0])
                point[(seed, cid, m)] = float(np.mean(ds))

    # customer-clustered bootstrap: one draw over the union of customers, applied to every origin
    union = pd.Index(sorted(set().union(*[set(c["df"]["CustomerID"]) for c in cohorts])))
    pos = {c["origin"]: union.get_indexer(c["df"]["CustomerID"]) for c in cohorts}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {(seed, cid, m): np.empty(B) for seed in SEEDS for cid, _, _ in KEY for m in METRICS}
    for start in range(0, B, CHUNK):
        n = min(CHUNK, B - start)
        idx = rng.integers(0, len(union), size=(n, len(union)))
        counts = np.zeros((n, len(union)))
        np.add.at(counts, (np.repeat(np.arange(n), len(union)), idx.ravel()), 1.0)
        per_origin = {}
        for seed in SEEDS:
            for c in cohorts:
                df, oof = preds[(seed, c["origin"])]
                W = counts[:, pos[c["origin"]]]
                per_origin[(seed, c["origin"])] = {arm: point_metrics(df, oof[arm], W) for arm in ARMS}
        for seed in SEEDS:
            for cid, a, b in KEY:
                for m in METRICS:
                    d = np.mean([per_origin[(seed, c["origin"])][a][m] - per_origin[(seed, c["origin"])][b][m]
                                 for c in cohorts], axis=0)
                    draws[(seed, cid, m)][start:start + n] = d
        if (start // CHUNK) % 8 == 0:
            print(f"[{dataset}] bootstrap {start + n}/{B} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    rows = []
    for cid, a, b in KEY:
        for m in METRICS:
            per_seed = [point[(s, cid, m)] for s in SEEDS]
            mix = np.concatenate([draws[(s, cid, m)] for s in SEEDS])
            p = min(1.0, max(2 * min((mix <= 0).mean(), (mix >= 0).mean()), 1 / len(mix)))
            rows.append({"dataset": dataset, "id": cid, "comparison": f"{a} - {b}", "metric": m,
                         "delta_mean_over_seeds": float(np.mean(per_seed)), "delta_seed0": per_seed[0],
                         "delta_min_seed": float(np.min(per_seed)), "delta_max_seed": float(np.max(per_seed)),
                         "ci_low": float(np.percentile(mix, 2.5)), "ci_high": float(np.percentile(mix, 97.5)),
                         "p_boot": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_boot"].to_numpy())
    res.to_csv(OUT / f"robust_{dataset}.csv", index=False)
    (OUT / f"run_{dataset}.json").write_text(json.dumps({
        "plan": "docs/QUICK_WINS_PLAN.md Part B", "seeds": SEEDS, "B": B, "boot_seed": BOOT_SEED,
        "origins": [c["origin"] for c in cohorts], "union_customers": len(union),
        "seed0_reproduction_max_abs_diff": worst, "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2), encoding="utf-8")
    print(f"[{dataset}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def point_metrics(df: pd.DataFrame, pred: dict, W: np.ndarray) -> dict:
    y = df["repurchased"].to_numpy().astype(float)
    sp = np.log1p(df["future_spend"].to_numpy())
    inv = np.log1p(df["future_invoices"].to_numpy())
    return {"auc": weighted_auc(y, pred["prob"], W), "spend_r2": weighted_r2(sp, pred["sp"], W),
            "invoice_r2": weighted_r2(inv, pred["inv"], W)}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="mergesort")
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# Aggregation against committed results
# ---------------------------------------------------------------------------

def committed_comparisons() -> pd.DataFrame:
    R = ROOT / "results"
    lad = pd.read_csv(R / "baseline_ladder_rolling_origin" / "paired_comparisons.csv"); lad = lad[lad.scope == "POOLED"]
    v2 = pd.read_csv(R / "v2_hybrid" / "paired_comparisons.csv"); v2 = v2[v2.scope == "POOLED"]
    cd = pd.read_csv(R / "cdnow_confirmation" / "comparisons.csv")
    rows = []
    for ds, name in [("dunnhumby", "Dunnhumby"), ("retail2", "Online Retail II")]:
        for cid, a, b in KEY:
            src = lad if cid in ("K1", "K2") else v2
            for m in METRICS:
                r = src[(src.dataset == name) & (src.comparison == f"{a} - {b}") & (src.metric == m)].iloc[0]
                rows.append({"dataset": ds, "id": cid, "metric": m, "committed_delta": r.delta, "committed_p_holm": r.p_holm})
    for cid, a, b in KEY:
        for m in METRICS:
            r = cd[(cd.comparison == f"{a} - {b}") & (cd.metric == m)].iloc[0]
            rows.append({"dataset": "cdnow", "id": cid, "metric": m, "committed_delta": r.delta, "committed_p_holm": r.p_holm})
    return pd.DataFrame(rows)


def aggregate() -> None:
    import baseline_ladder_rolling_origin as bl
    res = pd.concat([pd.read_csv(OUT / f"robust_{d}.csv") for d in ["retail2", "dunnhumby", "cdnow"]])
    out = res.merge(committed_comparisons(), on=["dataset", "id", "metric"])

    def verdict(r):
        was, now = r.committed_p_holm < 0.05, r.p_holm < 0.05
        same_sign = np.sign(r.committed_delta) == np.sign(r.delta_mean_over_seeds)
        if was and now and same_sign:
            return "robust (significant, same sign)"
        if not was and not now:
            return "robust (not significant)"
        if was and not now:
            return "weakened (no longer significant)"
        return "strengthened (now significant)" if same_sign or not was else "sign change"
    out["verdict"] = out.apply(verdict, axis=1)
    out.to_csv(OUT / "robustness_vs_committed.csv", index=False)
    show = out[["dataset", "id", "comparison", "metric", "committed_delta", "committed_p_holm", "delta_mean_over_seeds",
                "delta_min_seed", "delta_max_seed", "ci_low", "ci_high", "p_holm", "verdict"]]
    lines = ["# Inference robustness (limitation follow-up, Part B)", "",
             "Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). 5 CV fold seeds; customer-clustered bootstrap "
             "across origins; B = 10,000 per seed; mixture interval and p over 50,000 draws (floor 2e-5 before Holm); "
             "Holm over 12 tests per dataset. Committed results are not replaced.", "",
             "Verdict counts: " + "; ".join(f"{k}: {v}" for k, v in out["verdict"].value_counts().items()), "",
             bl._md(show), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dunnhumby", "retail2", "cdnow"])
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    elif a.dataset:
        run_dataset(a.dataset)
    else:
        ap.error("give --dataset or --aggregate")

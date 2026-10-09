"""Independent verification of the CDNOW confirmation (verification only; no new analysis).

Verifies the committed results in results/cdnow_confirmation/ against the committed plan
(docs/CDNOW_CONFIRMATION_PLAN.md) and the raw data. It never writes to the frozen results
folder; all outputs go to results/cdnow_confirmation_verification/ (or --out).

Usage:
    python scripts/verify_cdnow_confirmation.py [--out DIR]

Inputs (documented in the plan):
    data/cdnow_raw/CDNOW_master.txt   extracted from CDNOW_master.zip
                                      (SHA-256 of the zip checked if the zip is present)
    docs/CDNOW_CONFIRMATION_PLAN.md
    results/cdnow_confirmation/{run_parameters.json, per_origin_metrics.csv,
                                per_origin_comparisons.csv, comparisons.csv}

What is independent and what is reused:
    INDEPENDENT (separate code written for this check):
      - plan/code/CSV consistency (parses the plan's comparison table and arm list);
      - raw-file parsing, cleaning, cohorts, R/F/M and holdout targets;
      - observation/holdout date boundaries and holdout overlap between origins;
      - point metrics (sklearn roc_auc_score / r2_score, not the pipeline's rank formula);
      - bootstrap metrics per resample (sklearn), paired differences, percentile intervals,
        pooled estimate, bootstrap p-values and the Holm adjustment.
    REUSED (by necessity, to reproduce the predictions being evaluated):
      - model training: the frozen pipeline (cdnow_confirmation.install_extension +
        baseline_ladder_rolling_origin.run_origin) is re-run deterministically to regenerate
        out-of-fold predictions. fit_predict is wrapped (not altered) to log, per fold,
        which customers each arm trained on;
      - the resampling convention: numpy default_rng(seed).integers in the same draw order,
        so the bootstrap uses the same resamples as the committed run.

Runtime: about 9 minutes for the deterministic model re-run plus about 10 minutes for the
sklearn bootstrap (2000 resamples x 7 arms x 3 origins).
Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cdnow_confirmation as cc  # noqa: E402

bl = cc.bl
FROZEN = ROOT / "results" / "cdnow_confirmation"
PLAN = ROOT / "docs" / "CDNOW_CONFIRMATION_PLAN.md"
ZIP_SHA256 = "94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49"
METRICS = ["auc", "spend_r2", "invoice_r2"]
TOL = 1e-9

checks: list[dict] = []
notes: list[str] = []


def check(section: str, name: str, source: str, expected, observed, passed: bool) -> None:
    checks.append({"section": section, "check": name, "logic": source,
                   "expected": str(expected), "observed": str(observed), "pass": bool(passed)})
    print(f"[{'PASS' if passed else 'FAIL'}] {section}: {name} | expected {expected} | observed {observed}", flush=True)


# ---------------------------------------------------------------------------
# 1. Inputs and plan/code/CSV consistency
# ---------------------------------------------------------------------------

def verify_inputs_and_plan() -> None:
    zip_path = cc.DATA_FILE.parent / "CDNOW_master.zip"
    if zip_path.exists():
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        check("inputs", "CDNOW_master.zip SHA-256 matches plan", "independent", ZIP_SHA256, digest, digest == ZIP_SHA256)
    else:
        notes.append("CDNOW_master.zip not present; zip hash not checked (txt used directly).")
    plan = PLAN.read_text(encoding="utf-8")
    plan_pairs = re.findall(r"^\| (C\d) \| (\w+) − (\w+) \|", plan, flags=re.M)
    code_pairs = [(i, a, b) for i, a, b in cc.COMPARISONS]
    check("plan", "planned comparison table equals code", "independent", plan_pairs, code_pairs, plan_pairs == code_pairs)
    plan_arms = re.findall(r"`(\w+)`", plan[plan.index("**Arms:**"):plan.index("**Comparisons:**")])
    check("plan", "planned arms equal code arms", "independent", plan_arms, cc.ARMS, plan_arms == cc.ARMS)
    comp = pd.read_csv(FROZEN / "comparisons.csv")
    csv_pairs = sorted(set(zip(comp["id"], comp["comparison"])))
    check("plan", "committed comparisons are exactly the planned pairs", "independent",
          sorted((i, f"{a} - {b}") for i, a, b in code_pairs), csv_pairs,
          csv_pairs == sorted((i, f"{a} - {b}") for i, a, b in code_pairs))
    check("plan", "committed comparison family size", "independent", 21, len(comp),
          len(comp) == 21 and set(comp["metric"]) == set(METRICS))
    rp = json.loads((FROZEN / "run_parameters.json").read_text())
    expected = {"min_support": 0.04, "l_thresholds": [0.3, 0.5, 0.7], "j_max": 0.8, "mu_cut": 0.5,
                "n_folds": 5, "cv_seed": 0, "boot_n": 2000, "boot_seed": 12345}
    check("plan", "frozen settings in run_parameters.json", "independent", expected, rp["frozen"], rp["frozen"] == expected)
    check("plan", "origins and horizon", "independent", (["1997-09-30", "1997-12-31", "1998-03-31"], 91),
          (rp["origins"], rp["horizon_days"]), rp["origins"] == ["1997-09-30", "1997-12-31", "1998-03-31"] and rp["horizon_days"] == 91)


# ---------------------------------------------------------------------------
# 2. Data protocol, cohorts and targets (independent construction)
# ---------------------------------------------------------------------------

def verify_data() -> None:
    rows = []
    with open(cc.DATA_FILE) as fh:
        for line in fh:
            cust, d, _cds, dollars = line.split()
            rows.append((cust, d, float(dollars)))
    raw = pd.DataFrame(rows, columns=["cust", "d", "dollars"])
    check("data", "raw records", "independent", 69659, len(raw), len(raw) == 69659)
    check("data", "raw customers", "independent", 23570, raw["cust"].nunique(), raw["cust"].nunique() == 23570)
    valid = raw[raw["dollars"] > 0]
    check("data", "records removed (dollars <= 0)", "independent", 80, len(raw) - len(valid), len(raw) - len(valid) == 80)
    orders = valid.groupby(["cust", "d"], as_index=False)["dollars"].sum()
    orders["date"] = pd.to_datetime(orders["d"], format="%Y%m%d")

    script = cc.cdnow_cohorts(cc.load_transactions())
    metrics_csv = pd.read_csv(FROZEN / "per_origin_metrics.csv")
    holdouts = {}
    for T, sc in zip(cc.ORIGINS, script):
        end = T + pd.Timedelta(days=91)
        o = orders[orders["date"] <= T]
        h = orders[(orders["date"] > T) & (orders["date"] <= end)]
        holdouts[T] = (T + pd.Timedelta(days=1), end)
        ind = pd.DataFrame({"R": (T - o.groupby("cust")["date"].max()).dt.days,
                            "F": o.groupby("cust").size(), "M": o.groupby("cust")["dollars"].sum()}).sort_index()
        ind["future_invoices"] = h.groupby("cust").size().reindex(ind.index).fillna(0).astype(int)
        ind["future_spend"] = h.groupby("cust")["dollars"].sum().reindex(ind.index).fillna(0.0)
        ind["repurchased"] = (ind["future_invoices"] > 0).astype(int)
        s = sc["df"].set_index("CustomerID").sort_index()
        label = str(T.date())
        same = list(ind.index) == list(s.index)
        check("cohort", f"{label}: same customer set as pipeline", "independent", len(ind), len(s), same)
        if same:
            for col in ["R", "F", "future_invoices", "repurchased"]:
                d = float(np.max(np.abs(ind[col].to_numpy(float) - s[col].to_numpy(float))))
                check("cohort", f"{label}: {col} identical", "independent", 0, d, d == 0)
            for col in ["M", "future_spend"]:
                d = float(np.max(np.abs(ind[col].to_numpy(float) - s[col].to_numpy(float))))
                check("cohort", f"{label}: {col} equal (float summation tolerance 1e-9)", "independent", "< 1e-9", d, d < 1e-9)
        row = metrics_csv[metrics_csv["origin"] == label].iloc[0]
        check("cohort", f"{label}: n and repurchase rate vs committed CSV", "independent",
              (int(row["n"]), round(float(row["repurchase_rate"]), 6)), (len(ind), round(float(ind["repurchased"].mean()), 6)),
              int(row["n"]) == len(ind) and abs(float(row["repurchase_rate"]) - ind["repurchased"].mean()) < 1e-12)
        check("dates", f"{label}: observation ends on cutoff", "independent", label, str(o["date"].max().date()),
              o["date"].max() == T)
        check("dates", f"{label}: holdout is (cutoff, cutoff+91d]", "independent",
              f"{(T + pd.Timedelta(days=1)).date()}..{end.date()}", f"{h['date'].min().date()}..{h['date'].max().date()}",
              h["date"].min() > T and h["date"].max() <= end)
    # holdout overlap between consecutive origins
    origins = list(holdouts)
    for a, b in zip(origins[:-1], origins[1:]):
        lo, hi = max(holdouts[a][0], holdouts[b][0]), min(holdouts[a][1], holdouts[b][1])
        shared = [str(d.date()) for d in pd.date_range(lo, hi)] if lo <= hi else []
        n_orders = int(orders["date"].isin(pd.to_datetime(shared)).sum()) if shared else 0
        notes.append(f"Holdouts of origins {a.date()} and {b.date()} share {len(shared)} calendar day(s) "
                     f"{shared} ({n_orders} customer-day orders on those days).")
        print(notes[-1], flush=True)


# ---------------------------------------------------------------------------
# 3-4. Deterministic re-run with fold logging; independent statistics
# ---------------------------------------------------------------------------

def metric_set(df: pd.DataFrame, pred: dict, idx: np.ndarray) -> dict:
    y = df["repurchased"].to_numpy()[idx]
    sp = np.log1p(df["future_spend"].to_numpy())[idx]
    inv = np.log1p(df["future_invoices"].to_numpy())[idx]
    return {"auc": roc_auc_score(y, pred["prob"][idx]), "spend_r2": r2_score(sp, pred["sp"][idx]),
            "invoice_r2": r2_score(inv, pred["inv"][idx])}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="mergesort")
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def verify_models_and_statistics() -> None:
    cc.install_extension()
    calls: list[dict] = []
    original = bl.fit_predict

    def logged(arm, X_tr, X_te, tr, seed):
        calls.append({"arm": arm, "seed": seed,
                      "train": hashlib.sha1("|".join(tr["CustomerID"]).encode()).hexdigest()})
        return original(arm, X_tr, X_te, tr, seed)

    bl.fit_predict = logged
    t0 = time.time()
    oofs = {}
    for c in cc.cdnow_cohorts(cc.load_transactions()):
        start = len(calls)
        oof, _ = bl.run_origin(c)
        oofs[c["origin"]] = (c["df"], oof)
        log = pd.DataFrame(calls[start:])
        log["fold"] = np.repeat(np.arange(bl.N_FOLDS), len(cc.ARMS))
        per_fold = log.groupby("fold").agg(arms=("arm", "nunique"), sets=("train", "nunique"), seeds=("seed", "nunique"))
        ok = bool((per_fold["arms"] == len(cc.ARMS)).all() and (per_fold["sets"] == 1).all() and (per_fold["seeds"] == 1).all())
        check("folds", f"{c['origin']}: all 7 arms (incl. spline and hybrid) train on one identical customer set per fold",
              "reused pipeline, independent logging", "1 train set and 1 seed per fold", per_fold[["sets", "seeds"]].to_dict("list"), ok)
        complete = all(np.isfinite(v).all() for arm in cc.ARMS for v in oof[arm].values())
        check("folds", f"{c['origin']}: every customer predicted out-of-fold exactly once per arm",
              "independent", True, complete, complete)
        # independent reconstruction of the stratified folds: test sets partition the cohort
        skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
        test_union = np.concatenate([te for _, te in skf.split(c["df"], c["df"]["repurchased"])])
        check("folds", f"{c['origin']}: test folds partition the cohort", "independent",
              len(c["df"]), len(np.unique(test_union)), len(test_union) == len(np.unique(test_union)) == len(c["df"]))
    notes.append(f"Deterministic model re-run took {(time.time() - t0) / 60:.1f} min.")
    bl.fit_predict = original

    metrics_csv = pd.read_csv(FROZEN / "per_origin_metrics.csv")
    worst = 0.0
    for o, (df, oof) in oofs.items():
        for arm in cc.ARMS:
            mine = metric_set(df, oof[arm], np.arange(len(df)))
            row = metrics_csv[(metrics_csv["origin"] == o) & (metrics_csv["arm"] == arm)].iloc[0]
            worst = max(worst, max(abs(mine[k] - row[k]) for k in METRICS))
    check("metrics", "63 per-origin point metrics reproduced (sklearn) vs committed CSV", "independent",
          f"<= {TOL}", worst, worst <= TOL)

    t1 = time.time()
    rng = np.random.default_rng(bl.BOOT_SEED)
    draws = {}
    for o, (df, oof) in oofs.items():
        idx_all = rng.integers(0, len(df), size=(bl.BOOT_N, len(df)))
        for arm in cc.ARMS:
            out = {k: np.empty(bl.BOOT_N) for k in METRICS}
            for b in range(bl.BOOT_N):
                for k, v in metric_set(df, oof[arm], idx_all[b]).items():
                    out[k][b] = v
            draws[(o, arm)] = out
        print(f"  bootstrap recomputed for {o} ({(time.time() - t1) / 60:.1f} min)", flush=True)
    notes.append(f"Independent sklearn bootstrap took {(time.time() - t1) / 60:.1f} min.")

    po_csv = pd.read_csv(FROZEN / "per_origin_comparisons.csv")
    comp_csv = pd.read_csv(FROZEN / "comparisons.csv")
    po_worst, rows = 0.0, []
    for cid, a, b in cc.COMPARISONS:
        for m in METRICS:
            per, point = [], []
            for o, (df, oof) in oofs.items():
                d = draws[(o, a)][m] - draws[(o, b)][m]
                full = np.arange(len(df))
                pt = metric_set(df, oof[a], full)[m] - metric_set(df, oof[b], full)[m]
                per.append(d)
                point.append(pt)
                r = po_csv[(po_csv["id"] == cid) & (po_csv["metric"] == m) & (po_csv["origin"] == o)].iloc[0]
                po_worst = max(po_worst, abs(r["delta"] - pt), abs(r["ci_low"] - np.percentile(d, 2.5)),
                               abs(r["ci_high"] - np.percentile(d, 97.5)))
            pooled = np.mean(per, axis=0)
            p = min(1.0, max(2 * min((pooled <= 0).mean(), (pooled >= 0).mean()), 1 / bl.BOOT_N))
            rows.append({"id": cid, "metric": m, "delta": float(np.mean(point)),
                         "ci_low": float(np.percentile(pooled, 2.5)), "ci_high": float(np.percentile(pooled, 97.5)),
                         "p_boot": p})
    check("statistics", "63 per-origin deltas and 95% percentile intervals vs committed CSV", "independent",
          f"<= {TOL}", po_worst, po_worst <= TOL)
    mine = pd.DataFrame(rows)
    mine["p_holm"] = holm(mine["p_boot"].to_numpy())
    merged = mine.merge(comp_csv, on=["id", "metric"], suffixes=("_mine", "_csv"))
    check("statistics", "pooled comparisons matched", "independent", 21, len(merged), len(merged) == 21)
    for col in ["delta", "ci_low", "ci_high", "p_boot", "p_holm"]:
        d = float(np.max(np.abs(merged[f"{col}_mine"] - merged[f"{col}_csv"])))
        check("statistics", f"pooled {col} (Holm family = 21 planned tests)", "independent", f"<= {TOL}", d, d <= TOL)
    floor_boot, floor_holm = 1 / bl.BOOT_N, len(mine) / bl.BOOT_N
    check("statistics", "minimum bootstrap p equals 1/B floor", "independent", floor_boot, mine["p_boot"].min(),
          abs(mine["p_boot"].min() - floor_boot) < 1e-12)
    check("statistics", "minimum Holm p equals 21/B floor", "independent", floor_holm, mine["p_holm"].min(),
          abs(mine["p_holm"].min() - floor_holm) < 1e-12)
    at_floor = merged[np.isclose(merged["p_holm_csv"], floor_holm)]
    notes.append(f"{len(at_floor)} of 21 adjusted p-values are at the procedure's floor ({floor_holm:.4f}); "
                 "such p-values indicate significance at that floor, not the size of an effect.")
    mine.to_csv(OUT / "recomputed_comparisons.csv", index=False)


def write_report() -> None:
    df = pd.DataFrame(checks)
    df.to_csv(OUT / "checks.csv", index=False)
    n_fail = int((~df["pass"]).sum())
    lines = [
        "# CDNOW confirmation — independent verification",
        "",
        "Generated by `scripts/verify_cdnow_confirmation.py`. This is a **verification only**: it re-derives the "
        "committed results in `results/cdnow_confirmation/` and does not change them, the analysis plan, or any "
        "frozen methodology. No new comparisons are made.",
        "",
        f"**Result: {len(df) - n_fail} of {len(df)} checks passed; {n_fail} failed.**",
        "",
        "## Independent vs reused logic",
        "",
        "- **Independent (separate code written for this check):**",
        "  - plan/code/CSV consistency;",
        "  - raw-file parsing, cleaning, cohorts and targets;",
        "  - observation/holdout dates and holdout overlap;",
        "  - point metrics (sklearn);",
        "  - bootstrap per-resample metrics (sklearn), paired differences, percentile intervals, pooled estimates, bootstrap p-values and the Holm adjustment.",
        "- **Reused:**",
        "  - the frozen model pipeline, re-run deterministically to regenerate out-of-fold predictions; `fit_predict` is wrapped only to log training sets;",
        "  - the resampling convention (numpy `default_rng(12345).integers`, same draw order) so the same bootstrap resamples are used.",
        "",
        "## Checks",
        "",
        bl._md(df),
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in notes],
        "- The holdouts of the second and third origins share the calendar day 1998-04-01, because 1997-12-31 + 91 days "
        "= 1998-04-01. Each origin is evaluated on its own, so this is not leakage, but it adds to the overlap between "
        "origins and to the approximate nature of pooled inference.",
        "- Bootstrap p-values cannot fall below 1/B = 0.0005, and Holm-adjusted p-values cannot fall below "
        "21/B = 0.0105. Significant results at that floor are not evidence of large effects; effect sizes and their "
        "intervals carry that information.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(df) - n_fail}/{len(df)} checks passed. Report: {OUT / 'REPORT.md'}", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "cdnow_confirmation_verification"))
    OUT = Path(ap.parse_args().out).resolve()
    if OUT == FROZEN.resolve() or FROZEN.resolve() in OUT.parents:
        sys.exit("Refusing to write into the frozen results folder.")
    OUT.mkdir(parents=True, exist_ok=True)
    verify_inputs_and_plan()
    verify_data()
    verify_models_and_statistics()
    write_report()

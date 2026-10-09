"""Part C of docs/QUICK_WINS_PLAN.md: outcome-free interpretability proxies.

Crisp vs fuzzy RFM-FCA (v2 segments = fuzzy segments), fitted on the full cohort at
every pooled origin of Online Retail II, Dunnhumby and CDNOW (CDNOW with its planned
tie-merging extension). No outcomes are used. Writes only to results/qw_interpretability/.

Measures (fixed in the plan): K (retained concepts), coverage (share of customers with
at least one core concept, membership >= 0.5), C80 (concepts needed to cover 80% of
customers in their cores, greedy maximum coverage), mean intent length (distinct bands),
core load (mean core concepts per customer), overlap (mean pairwise core-extent Jaccard).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import segment_stability as ss  # noqa: E402

OUT = ROOT / "results" / "qw_interpretability"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["crisp_rfm_fca", "fuzzy_rfm_fca"]


def greedy_c80(core: np.ndarray, target: float = 0.8) -> float:
    n = core.shape[0]
    covered = np.zeros(n, dtype=bool)
    remaining = list(range(core.shape[1]))
    k = 0
    while covered.mean() < target:
        gains = [int((core[:, j] & ~covered).sum()) for j in remaining]
        if not remaining or max(gains) == 0:
            return float("nan")
        j = remaining.pop(int(np.argmax(gains)))
        covered |= core[:, j]
        k += 1
    return float(k)


def measures(df: pd.DataFrame, arm: str) -> dict:
    model = ss.fit_model(arm, df)
    proj = ss.project(model, df)
    keys = model["keys"]
    core = np.column_stack([proj[k] >= 0.5 for k in keys])
    inter = core.T.astype(np.int64) @ core.astype(np.int64)
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(len(keys), k=1)
    jac = np.where(union[iu] > 0, inter[iu] / np.where(union[iu] > 0, union[iu], 1), 0.0)
    return {"K": len(keys), "coverage": float(core.any(axis=1).mean()), "C80": greedy_c80(core),
            "intent_length": float(np.mean([len(k) for k in keys])), "core_load": float(core.sum(axis=1).mean()),
            "overlap": float(jac.mean()) if len(jac) else 0.0}


def main() -> None:
    t0 = time.time()
    bl = ss.bl
    rows = []
    tx = pd.read_csv(bl.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    groups = [("Dunnhumby", bl.dunnhumby_cohorts(tx)),
              ("Online Retail II", [c for c in bl.retail2_cohorts(bl.load_cleaned_transactions()) if c["pooled"]])]
    for name, cohorts in groups:
        for c in cohorts:
            for arm in ARMS:
                rows.append({"dataset": name, "origin": c["origin"], "arm": arm, **measures(c["df"], arm)})
        print(f"{name} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # CDNOW last, with its planned extension (inert on the datasets above)
    import cdnow_confirmation as cc
    cc.install_extension()
    ss.compute_fuzzy_memberships = cc.compute_fuzzy_memberships_n
    for c in cc.cdnow_cohorts(cc.load_transactions()):
        for arm in ARMS:
            rows.append({"dataset": "CDNOW", "origin": c["origin"], "arm": arm, **measures(c["df"], arm)})
    print(f"CDNOW done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_origin_proxies.csv", index=False)
    cols = ["K", "coverage", "C80", "intent_length", "core_load", "overlap"]
    summ = per.groupby(["dataset", "arm"], sort=False)[cols].agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    show = per.groupby(["dataset", "arm"], sort=False)[cols].mean().round(3).reset_index()
    lines = ["# Interpretability proxies (limitation follow-up, Part C)", "",
             "Plan: `docs/QUICK_WINS_PLAN.md` (committed before this script). Outcome-free structural proxies; not "
             "measures of human interpretability. Means over origins (ranges in summary.csv). v2 hybrid segments equal "
             "fuzzy segments.", "", ss.bl._md(show), "", ss.bl._md(per.round(3)), "",
             f"Runtime: {(time.time() - t0) / 60:.1f} min.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

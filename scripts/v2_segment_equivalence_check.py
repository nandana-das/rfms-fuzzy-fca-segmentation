"""v2 Part A: implementation-equivalence check of v2 segments vs v1 segments.

Plan: docs/V2_STABILITY_ANALYSIS_PLAN.md (committed before this script).
Exploratory, post hoc. This is a verification, not a new stability experiment:
log-RFM features enter v2 only downstream, so v2 segments should equal v1
segments. The check replays every fit of the v1 stability protocol
(scripts/segment_stability.py) and compares the two code paths:

    v1 path: segment_stability.fit_model + project
    v2 path: v2_hybrid_ladder.build_features (hybrid arm), concept columns only

Per fit and arm (fuzzy: v1 fuzzy_rfm_fca vs hybrid_fuzzy_fca; crisp:
v1 crisp_rfm_fca vs hybrid_crisp_fca) it checks:
    1. the same number of concepts;
    2. the identical multiset of concept-membership columns over all evaluated
       customers (max absolute difference after column matching);
    3. identical core profiles (membership >= 0.5).
Fits replayed: full cohort + the same 30 subsamples (seed 2026, 80%) at each of
the 9 pooled origins, and the 7 consecutive-origin temporal projections.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import segment_stability as ss  # noqa: E402
import v2_hybrid_ladder as v2  # noqa: E402

OUT = ROOT_DIR / "results" / "v2_stability"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = {"fuzzy": ("fuzzy_rfm_fca", "hybrid_fuzzy_fca"), "crisp": ("crisp_rfm_fca", "hybrid_crisp_fca")}
N_LOG = 3  # log1p(R, F, M) columns appended by the hybrid


def _sorted_columns(M: np.ndarray) -> np.ndarray:
    M = np.round(M, 12)
    order = np.lexsort(M[::-1]) if M.shape[1] else np.array([], dtype=int)
    return M[:, order]


def compare(arm_key: str, train: pd.DataFrame, evaluate: pd.DataFrame) -> dict:
    v1_arm, v2_arm = PAIRS[arm_key]
    m = ss.fit_model(v1_arm, train)
    p = ss.project(m, evaluate)
    A = np.column_stack([p[k] for k in m["keys"]]) if m["keys"] else np.zeros((len(evaluate), 0))
    _, X_te, _ = v2.build_features(v2_arm, train, evaluate)
    B = X_te[:, :-N_LOG]
    row = {"k_v1": A.shape[1], "k_v2": B.shape[1]}
    if A.shape[1] != B.shape[1]:
        return {**row, "max_abs_diff": np.nan, "columns_identical": False, "core_profiles_identical": False}
    As, Bs = _sorted_columns(A), _sorted_columns(B)
    diff = float(np.max(np.abs(As - Bs))) if A.size else 0.0
    core_same = bool(np.array_equal(_sorted_columns((A >= 0.5).astype(float)), _sorted_columns((B >= 0.5).astype(float))))
    return {**row, "max_abs_diff": diff, "columns_identical": diff <= 1e-9, "core_profiles_identical": core_same}


def main() -> None:
    t0 = time.time()
    dh_tx = pd.read_csv(ss.TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = ss.bl.dunnhumby_cohorts(dh_tx) + [
        c for c in ss.bl.retail2_cohorts(ss.load_cleaned_transactions()) if c["pooled"]
    ]
    rows = []
    # Study A replay: identical RNG and draw order to segment_stability.study_a.
    rng = np.random.default_rng(ss.SUBSAMPLE_SEED)
    for c in cohorts:
        df = c["df"]
        n = len(df)
        subsamples = [rng.choice(n, size=int(round(ss.SUBSAMPLE_FRAC * n)), replace=False) for _ in range(ss.N_SUBSAMPLES)]
        fits = [("full", df)] + [(f"subsample {s}", df.iloc[idx].reset_index(drop=True)) for s, idx in enumerate(subsamples)]
        for fit_name, train in fits:
            for arm_key in PAIRS:
                rows.append({"study": "A refit", "dataset": c["dataset"], "origin": c["origin"], "fit": fit_name,
                             "arm": arm_key, **compare(arm_key, train, df)})
        ss._pr(f"  [A] {c['dataset']} | {c['origin']} checked ({time.time() - t0:.0f}s)")
    # Study B replay: model at t1 projects RFM_t1, model at t2 projects RFM_t2 (common customers).
    by_ds: dict[str, list] = {}
    for c in cohorts:
        by_ds.setdefault(c["dataset"], []).append(c)
    for ds, cs in by_ds.items():
        for c1, c2 in zip(cs[:-1], cs[1:]):
            d1, d2 = c1["df"].set_index("CustomerID"), c2["df"].set_index("CustomerID")
            common = d1.index.intersection(d2.index)
            x1, x2 = d1.loc[common].reset_index(), d2.loc[common].reset_index()
            for side, train, ev in (("t1", c1["df"], x1), ("t2", c2["df"], x2)):
                for arm_key in PAIRS:
                    rows.append({"study": "B temporal", "dataset": ds, "origin": f"{c1['origin']} -> {c2['origin']}",
                                 "fit": side, "arm": arm_key, **compare(arm_key, train, ev)})
            ss._pr(f"  [B] {ds} | {c1['origin']} -> {c2['origin']} checked")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "segment_equivalence_check.csv", index=False)
    summary = res.groupby(["study", "dataset", "arm"]).agg(
        fits=("fit", "size"),
        concept_count_match=("k_v1", lambda s: int((s == res.loc[s.index, "k_v2"]).sum())),
        columns_identical=("columns_identical", "sum"),
        core_profiles_identical=("core_profiles_identical", "sum"),
        max_abs_diff=("max_abs_diff", "max"),
        k_min=("k_v1", "min"), k_max=("k_v1", "max"),
    ).reset_index()
    summary.to_csv(OUT / "segment_equivalence_summary.csv", index=False)
    all_ok = bool(res["columns_identical"].all() and res["core_profiles_identical"].all() and (res["k_v1"] == res["k_v2"]).all())
    lines = [
        "# v2 Part A — segment equivalence check",
        "",
        "> Exploratory, post hoc. Verification, not a new stability experiment. Plan: `docs/V2_STABILITY_ANALYSIS_PLAN.md`.",
        "",
        f"**Result: {'ALL FITS IDENTICAL' if all_ok else 'DISCREPANCY FOUND'}** "
        f"({len(res)} fit comparisons; {int(res['columns_identical'].sum())} with identical concept-membership columns; "
        f"{int(res['core_profiles_identical'].sum())} with identical core profiles; max |difference| = {res['max_abs_diff'].max():.3g}).",
        "",
        ss.bl._md(summary),
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    ss._pr(lines[4])


if __name__ == "__main__":
    main()

"""Apply the decision criteria in docs/TEMPORAL_SECOND_SEED_PLAN.md.

Compares seed-0 (results/temporal_nonfca_baseline/) and seed-1 (results/temporal_second_seed/)
pooled deltas for fuzzy_rfm_t - spline_log_rfm_t. Reporting only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED0 = ROOT / "results" / "temporal_nonfca_baseline"
SEED1 = ROOT / "results" / "temporal_second_seed"
COMPARISON = "fuzzy_rfm_t - spline_log_rfm_t"
METRICS = ["auc", "spend_r2", "invoice_r2"]
THRESHOLD = 0.002
MAJORITY = {"Online Retail II": 3, "Dunnhumby": 3}


def _pooled(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path / "paired_comparisons.csv")
    return c[(c.scope == "POOLED") & (c.comparison == COMPARISON)].set_index(["dataset", "metric"])


def classify(d0: float, d1: float, n_same_sign: int, majority: int) -> str:
    if abs(d0) < THRESHOLD:
        return "null cell: consistent" if abs(d1) < THRESHOLD else "null cell: flagged"
    if np.sign(d1) != np.sign(d0):
        # The plan does not define an opposite-sign delta below the threshold; label it explicitly.
        return "reversed" if abs(d1) >= THRESHOLD else "attenuated (opposite sign, below threshold; not covered by plan)"
    if abs(d1) >= 0.5 * abs(d0) and n_same_sign >= majority:
        return "replicates"
    return "attenuated"


def main() -> None:
    p0, p1 = _pooled(SEED0), _pooled(SEED1)
    per1 = pd.read_csv(SEED1 / "paired_comparisons.csv")
    per1 = per1[(per1.scope != "POOLED") & (per1.comparison == COMPARISON)]
    m1 = pd.read_csv(SEED1 / "per_origin_metrics.csv")
    rows = []
    for dataset in ["Online Retail II", "Dunnhumby"]:
        pooled_origins = m1[(m1.dataset == dataset) & m1.pooled].origin.unique()
        for metric in METRICS:
            d0, d1 = p0.loc[(dataset, metric), "delta"], p1.loc[(dataset, metric), "delta"]
            o = per1[(per1.dataset == dataset) & (per1.metric == metric) & per1.scope.isin(pooled_origins)]
            same = int((np.sign(o.delta) == np.sign(d0)).sum())
            rows.append({
                "dataset": dataset, "metric": metric, "delta_seed0": d0, "delta_seed1": d1,
                "mean_two_seeds": (d0 + d1) / 2, "seed_difference": d1 - d0,
                "origins_same_sign_as_seed0": f"{same}/{len(o)}",
                "p_holm_seed0": p0.loc[(dataset, metric), "p_holm"], "p_holm_seed1": p1.loc[(dataset, metric), "p_holm"],
                "verdict": classify(d0, d1, same, MAJORITY[dataset]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SEED1 / "seed_replication.csv", index=False)
    v = out.set_index(["dataset", "metric"]).verdict
    retail = v[("Online Retail II", "auc")] == "replicates" and v[("Online Retail II", "spend_r2")] == "replicates"
    dunn = int((out[out.dataset == "Dunnhumby"].verdict == "replicates").sum()) >= 2
    lines = [
        "# Second CV seed: FCA+T vs spline+T", "",
        "Criteria from `docs/TEMPORAL_SECOND_SEED_PLAN.md` (locked before the run; hashes in `PLAN_LOCK.txt`). "
        f"Delta = {COMPARISON}. Practical threshold {THRESHOLD}.", "",
        "| dataset | metric | Δ seed 0 | Δ seed 1 | mean | seed 1 − seed 0 | origins same sign (seed 1) | Holm p seed 0 | Holm p seed 1 | verdict |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for _, r in out.iterrows():
        lines.append(f"| {r.dataset} | {r.metric} | {r.delta_seed0:+.4f} | {r.delta_seed1:+.4f} | {r.mean_two_seeds:+.4f} | "
                     f"{r.seed_difference:+.4f} | {r.origins_same_sign_as_seed0} | {r.p_holm_seed0:.4f} | {r.p_holm_seed1:.4f} | {r.verdict} |")
    lines += [
        "", "## Claim-level decisions", "",
        f"- Retail II FCA-specific advantage (requires AUC and Spend R² to replicate): **{'retained' if retail else 'not retained'}**.",
        f"- Dunnhumby spline advantage (requires ≥ 2 of 3 cells to replicate): **{'retained' if dunn else 'not retained'}**.",
        "", "## Scope of this result", "",
        "- 'Replicates' means the seed-0 direction and practical magnitude held under a second fold assignment "
        "(CV seed 1) on the same customers, origins, and outcomes. It shows stability across these two fold "
        "assignments only.",
        "- It is not an independent replication: no new data, customers, time periods, or datasets were used, "
        "and it supports no general claim about other customer populations.",
        "- Two seeds do not estimate fold-assignment variance.",
        "- Holm p-values are descriptive (3 tests per dataset per seed) and were not the decision criterion.",
        "- FCA+T uses ~160–170 features versus 29 for the additive spline+T; the Retail II advantage may reflect "
        "interactions rather than FCA specifically.", "",
        "Command: `python scripts/temporal_second_seed_evaluate.py`", "",
    ]
    (SEED1 / "SEED_REPLICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.round(4).to_string(index=False))
    print("Retail II retained:", retail, "| Dunnhumby retained:", dunn)


if __name__ == "__main__":
    main()

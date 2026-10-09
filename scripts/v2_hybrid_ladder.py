"""Version 2 (post hoc): hybrid fuzzy RFM-FCA = concept memberships + log1p(R, F, M).

Analysis plan (arms, comparisons, claim rules): docs/V2_HYBRID_ANALYSIS_PLAN.md
(its precedence over the run is not established by Git history; see that file). This study is post hoc (motivated by a v1
result, evaluated on the same origins) and never replaces the frozen v1 evidence.

Implementation: reuses scripts/baseline_ladder_rolling_origin.py unchanged
(cohorts, folds, models, bootstrap, Holm, report). Only the arm list, the
comparison list, the output folder and two new feature builders are swapped in,
so the v1 arms reproduce their frozen values exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402

V2_DIR = ROOT_DIR / "results" / "v2_hybrid"
V2_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [
    "raw_std", "log_rfm", "spline_log_rfm",
    "crisp_rfm_fca", "fuzzy_rfm_fca",
    "hybrid_crisp_fca", "hybrid_fuzzy_fca",
]
COMPARISONS = [
    ("hybrid_fuzzy_fca", "spline_log_rfm"),   # P1
    ("hybrid_fuzzy_fca", "fuzzy_rfm_fca"),    # P2
    ("hybrid_fuzzy_fca", "hybrid_crisp_fca"), # P3
    ("hybrid_fuzzy_fca", "log_rfm"),          # P4
    ("hybrid_fuzzy_fca", "crisp_rfm_fca"),    # P5
]
HYBRID_BASE = {"hybrid_fuzzy_fca": "fuzzy_rfm_fca", "hybrid_crisp_fca": "crisp_rfm_fca"}

_v1_build_features = bl.build_features


def build_features(arm: str, tr: pd.DataFrame, te: pd.DataFrame):
    if arm not in HYBRID_BASE:
        return _v1_build_features(arm, tr, te)
    c_tr, c_te, info = _v1_build_features(HYBRID_BASE[arm], tr, te)
    sc = StandardScaler().fit(bl._log(tr))
    return (np.hstack([c_tr, sc.transform(bl._log(tr))]),
            np.hstack([c_te, sc.transform(bl._log(te))]),
            {**info, "n_features": int(c_tr.shape[1] + 3)})


def main() -> None:
    # Swap in the v2 configuration; run_origin / main look these up at call time.
    bl.ARMS = ARMS
    bl.COMPARISONS = COMPARISONS
    bl.EXP_DIR = V2_DIR
    bl.build_features = build_features
    bl.main()
    report = V2_DIR / "REPORT.md"
    text = report.read_text(encoding="utf-8")
    header = (
        "> **Version 2, chosen after seeing v1 (post hoc).** Plan and claim rules: "
        "`docs/V2_HYBRID_ANALYSIS_PLAN.md`. Evaluated on the same origins as v1, so gains are optimistic. "
        "This does not replace the frozen v1 evidence.\n\n"
    )
    first_nl = text.index("\n") + 1
    report.write_text(text[:first_nl] + "\n" + header + text[first_nl:].lstrip("\n"), encoding="utf-8")


if __name__ == "__main__":
    main()

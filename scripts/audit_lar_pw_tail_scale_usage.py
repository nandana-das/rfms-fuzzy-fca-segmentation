"""Lightweight audit of historical/v2 tail-scale fallback reachability.

Loads the existing nine cohorts and reproduces only the training-fold
membership fitting boundary. It does not fit downstream predictive models and
never writes to the historical result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_ablation_multi_origin as runner  # noqa: E402
import tail_memberships as old  # noqa: E402
import tail_memberships_v2 as v2  # noqa: E402


def scale_inputs(values: np.ndarray) -> tuple[float, float, float, float]:
    x = old.check_rfm_values(values)
    cut = np.quantile(x, np.arange(1, 5) / 5.0)
    scores = 1 + np.searchsorted(cut, x, side="left")
    z = np.log1p(x)
    z5 = z[scores == 5]
    a4 = float(np.median(z[scores == 4]))
    iqr = float(np.percentile(z5, 75) - np.percentile(z5, 25))
    mad_a4 = float(np.median(np.abs(z5 - a4)))
    return iqr, a4, float(np.max(z5)), mad_a4


def run(output_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    cohorts = runner.load_cohorts()
    for cohort in cohorts:
        df = cohort["df"]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for fold, (i_tr, _i_te) in enumerate(skf.split(df, df["repurchased"])):
            tr = df.iloc[i_tr].reset_index(drop=True)
            for dim in ("F", "M"):
                values = tr[dim].to_numpy()
                iqr, a4, max_z5, mad_a4 = scale_inputs(values)
                old_params = old.fit_lar_pw_parameters(values, dim)
                v2_params = v2.fit_lar_pw_parameters_v2(values, dim)
                fallback = iqr <= 0 and max_z5 > a4
                rows.append({
                    "dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold,
                    "dimension": dim, "n_train": len(tr), "iqr_log_band5": iqr,
                    "a4_log_anchor": a4, "max_log_band5": max_z5,
                    "historical_fallback_reached": fallback,
                    "historical_s_tail": old_params.s_tail,
                    "v2_mad_a4": mad_a4, "v2_s_tail": v2_params.s_tail,
                    "scale_difference": float(v2_params.s_tail - old_params.s_tail),
                })
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "tail_scale_usage.csv", index=False)
    manifest = {
        "record_type": "tail_scale_reachability_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "datasets": sorted(out["dataset"].unique().tolist()),
        "origins": int(out[["dataset", "origin"]].drop_duplicates().shape[0]),
        "folds": int(out[["dataset", "origin", "fold"]].drop_duplicates().shape[0]),
        "dimensions": ["F", "M"], "downstream_model_training": False,
        "historical_fallback_rows": int(out["historical_fallback_reached"].sum()),
        "historical_s_tail_equals_v2_rows": int(np.isclose(out["historical_s_tail"], out["v2_s_tail"]).sum()),
        "source_hashes": {
            "scripts/tail_memberships.py": hashlib.sha256((ROOT / "scripts/tail_memberships.py").read_bytes()).hexdigest(),
            "scripts/tail_memberships_v2.py": hashlib.sha256((ROOT / "scripts/tail_memberships_v2.py").read_bytes()).hexdigest(),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lar_pw_v2_tail_scale_audit")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

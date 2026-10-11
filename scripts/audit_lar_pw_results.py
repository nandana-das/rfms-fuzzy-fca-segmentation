"""Audit retained M0/M1-LAR-PW rolling-origin results.

This module deliberately performs no model fitting.  It validates the retained
per-origin table, constructs the paired M1 minus M0 comparison, and writes
descriptive summaries.  Historical ``M1`` labels are normalized only in the
audit output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


METRICS = ("invoice_r2", "spend_r2", "auc")
EXPECTED_ORIGINS = {
    "Dunnhumby": ("day 347", "day 438", "day 529", "day 620"),
    "Online Retail II": ("2010-09-10", "2010-12-10", "2011-03-11", "2011-06-10", "2011-09-09"),
}
REQUIRED = {"dataset", "origin", "arm", *METRICS}


def _float(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    missing = REQUIRED.difference(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def audit_rows(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    rows = list(rows)
    key_counts: Counter[tuple[str, str, str]] = Counter(
        (r["dataset"], r["origin"], r["arm"]) for r in rows
    )
    duplicate_keys = [list(k) + [n] for k, n in key_counts.items() if n != 1]
    datasets = set(EXPECTED_ORIGINS)
    unexpected = sorted({(r["dataset"], r["origin"]) for r in rows if r["dataset"] not in datasets or r["origin"] not in EXPECTED_ORIGINS.get(r["dataset"], ())})
    missing_pairs = []
    for dataset, origins in EXPECTED_ORIGINS.items():
        for origin in origins:
            all_arms = {r["arm"] for r in rows if r["dataset"] == dataset and r["origin"] == origin}
            arms = all_arms & {"M0", "M1"}
            if arms != {"M0", "M1"}:
                missing_pairs.append([dataset, origin, sorted(arms)])
    invalid_values = []
    for r in rows:
        for metric in METRICS:
            try:
                _float(r[metric], f"{r['dataset']}/{r['origin']}/{r['arm']}/{metric}")
            except ValueError as exc:
                invalid_values.append(str(exc))
    return {
        "n_rows": len(rows),
        "arms": sorted({r["arm"] for r in rows}),
        "duplicate_keys": duplicate_keys,
        "unexpected_origins": unexpected,
        "missing_or_mismatched_pairs": missing_pairs,
        "invalid_values": invalid_values,
        "valid": not (duplicate_keys or unexpected or missing_pairs or invalid_values),
    }


def paired_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["arm"] not in {"M0", "M1"}:
            continue
        grouped[(row["dataset"], row["origin"])][row["arm"]] = row
    result = []
    for (dataset, origin), arms in sorted(grouped.items()):
        if set(arms) != {"M0", "M1"}:
            continue
        out: dict[str, object] = {"dataset": dataset, "origin": origin}
        for metric in METRICS:
            m0 = _float(arms["M0"][metric], metric)
            m1 = _float(arms["M1"][metric], metric)
            out[f"m0_{metric}"] = m0
            out[f"m1_{metric}"] = m1
            out[f"delta_{metric}"] = m1 - m0
            out[f"relative_pct_{metric}"] = 100.0 * (m1 / m0 - 1.0) if m0 else None
        result.append(out)
    return result


def summarize(pairs: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    pairs = list(pairs)
    result = []
    datasets = list(EXPECTED_ORIGINS) + ["All origins"]
    for dataset in datasets:
        selected = pairs if dataset == "All origins" else [p for p in pairs if p["dataset"] == dataset]
        for metric in METRICS:
            deltas = [float(p[f"delta_{metric}"]) for p in selected]
            m0 = [float(p[f"m0_{metric}"]) for p in selected]
            m1 = [float(p[f"m1_{metric}"]) for p in selected]
            if not deltas:
                continue
            ordered = sorted(deltas)
            median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2 - 1] + ordered[len(ordered)//2]) / 2
            result.append({
                "dataset": dataset,
                "metric": metric,
                "n_origins": len(deltas),
                "m0_mean": sum(m0) / len(m0),
                "m1_mean": sum(m1) / len(m1),
                "mean_delta": sum(deltas) / len(deltas),
                "median_delta": median,
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "positive": sum(d > 0 for d in deltas),
                "zero": sum(d == 0 for d in deltas),
                "negative": sum(d < 0 for d in deltas),
                "mean_origin_relative_pct": sum(float(p[f"relative_pct_{metric}"]) for p in selected) / len(selected),
                "uncertainty": "not available from retained aggregate metrics",
            })
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, output_dir: Path) -> dict[str, object]:
    rows = read_rows(source)
    audit = audit_rows(rows)
    if not audit["valid"]:
        raise ValueError(json.dumps(audit, indent=2))
    pairs = paired_rows(rows)
    summaries = summarize(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_origin_comparison.csv", pairs)
    _write_csv(output_dir / "descriptive_summary.csv", summaries)
    manifest = {
        "record_type": "retained_results_audit",
        "source": str(source).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "historical_label_normalization": {"M1": "M1-LAR-PW"},
        "n_paired_origins": len(pairs),
        "audit": audit,
        "uncertainty_note": "No customer-level prediction vectors or paired bootstrap draws are retained; summaries are descriptive only.",
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/fuzzy_tnorm_tail_ablation/multi_origin_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lar_pw_primary_evaluation_v1"))
    args = parser.parse_args()
    manifest = run(args.source, args.output_dir)
    print(json.dumps({"valid": manifest["audit"]["valid"], "n_paired_origins": manifest["n_paired_origins"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

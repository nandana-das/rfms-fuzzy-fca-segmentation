"""Strict row lookup for generated evidence documents.

Used by scripts/build_paper_evidence_map.py. Every lookup must match exactly one row;
missing or duplicated comparisons, non-finite deltas, non-finite p-values where one is
expected, and claimed directions that disagree with the row all raise EvidenceLookupError
with the full lookup key, so a generated document can never silently cite the wrong row.
"""

from __future__ import annotations

import math

import pandas as pd

ALPHA = 0.05


class EvidenceLookupError(ValueError):
    pass


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check_direction(key: str, delta: float, p: float, expect: str | None) -> None:
    if expect is None:
        return
    if expect == "ns":
        ok = p >= ALPHA
    elif expect in ("higher", "lower"):
        ok = p < ALPHA and (delta > 0 if expect == "higher" else delta < 0)
    else:
        raise EvidenceLookupError(f"{key}: unknown expected direction {expect!r}")
    if not ok:
        raise EvidenceLookupError(f"{key}: claim says {expect!r} but delta={delta:+.4f}, Holm p={p:.4f}")


def qrow(df: pd.DataFrame, comparison: str, metric: str, dataset: str | None = None,
         scope: str | None = "POOLED", expect_p: bool = True, expect: str | None = None,
         source: str = "") -> pd.Series:
    """Return the single row for (dataset, scope, comparison, metric).

    `scope` filters on a `scope` column when the frame has one (default: pooled rows).
    `expect` ('higher', 'lower', 'ns') asserts the direction the claim text states,
    at Holm p < 0.05.
    """
    key = f"{source or 'source'}[dataset={dataset!r}, scope={scope!r}, comparison={comparison!r}, metric={metric!r}]"
    for col in ("comparison", "metric", "delta"):
        if col not in df.columns:
            raise EvidenceLookupError(f"{key}: column {col!r} missing")
    q = df[(df["comparison"] == comparison) & (df["metric"] == metric)]
    if dataset is not None:
        if "dataset" not in df.columns:
            raise EvidenceLookupError(f"{key}: dataset requested but source has no 'dataset' column")
        q = q[q["dataset"] == dataset]
    if scope is not None and "scope" in df.columns:
        q = q[q["scope"] == scope]
    if len(q) == 0:
        raise EvidenceLookupError(f"{key}: comparison missing")
    if len(q) > 1:
        raise EvidenceLookupError(f"{key}: {len(q)} rows match (duplicated or under-specified)")
    r = q.iloc[0]
    if not _finite(r["delta"]):
        raise EvidenceLookupError(f"{key}: non-finite delta {r['delta']!r}")
    if expect_p:
        if "p_holm" not in df.columns or not _finite(r["p_holm"]):
            raise EvidenceLookupError(f"{key}: expected a finite Holm p, got {r.get('p_holm')!r}")
        _check_direction(key, float(r["delta"]), float(r["p_holm"]), expect)
    elif expect is not None:
        raise EvidenceLookupError(f"{key}: a direction check needs expect_p=True")
    return r


def seedrow(df: pd.DataFrame, dataset: str, metric: str, source: str = "") -> pd.Series:
    """Return the single second-seed replication row, with finite deltas, p-values and a verdict."""
    key = f"{source or 'source'}[dataset={dataset!r}, metric={metric!r}]"
    q = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
    if len(q) != 1:
        raise EvidenceLookupError(f"{key}: {'comparison missing' if len(q) == 0 else f'{len(q)} rows match'}")
    r = q.iloc[0]
    for col in ("delta_seed0", "delta_seed1", "p_holm_seed0", "p_holm_seed1"):
        if not _finite(r[col]):
            raise EvidenceLookupError(f"{key}: non-finite {col} {r[col]!r}")
    if not str(r["verdict"]).strip() or str(r["verdict"]) == "nan":
        raise EvidenceLookupError(f"{key}: empty verdict")
    return r


def qfmt(r: pd.Series) -> str:
    return f"Δ {float(r['delta']):+.4f} (Holm p {float(r['p_holm']):.4f})"

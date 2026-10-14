"""Regression tests for temporal comparison indexing and Holm-family metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from temporal_rfm_fca_ablation import _pooled_point_deltas  # noqa: E402


def test_pooled_deltas_follow_explicit_pooled_origin_order():
    rows = [
        {"origin": "o1", "arm": "t", "spend_r2": 3.0},
        {"origin": "o1", "arm": "m0", "spend_r2": 2.0},
        {"origin": "primary", "arm": "t", "spend_r2": 100.0},
        {"origin": "primary", "arm": "m0", "spend_r2": 0.0},
        {"origin": "o2", "arm": "t", "spend_r2": 8.0},
        {"origin": "o2", "arm": "m0", "spend_r2": 5.0},
    ]
    sub = pd.DataFrame(rows)
    assert _pooled_point_deltas(sub, "t", "m0", "spend_r2", ["o1", "o2"]) == [1.0, 3.0]


def test_holm_accepts_the_temporal_nine_test_family():
    adjusted = bl._holm([0.01] * 9)
    assert len(adjusted) == 9
    assert all(0.0 <= p <= 1.0 for p in adjusted)

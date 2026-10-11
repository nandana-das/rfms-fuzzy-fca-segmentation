"""Tests for the retained-results audit, without model execution."""

from __future__ import annotations

import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_lar_pw_results as audit  # noqa: E402


def _row(dataset="Dunnhumby", origin="day 347", arm="M0", auc="0.5", spend="0.2", invoice="0.3"):
    return {"dataset": dataset, "origin": origin, "arm": arm, "auc": auc, "spend_r2": spend, "invoice_r2": invoice}


def test_pairing_and_metric_direction():
    rows = [_row(arm="M0", auc="0.50"), _row(arm="M1", auc="0.55")]
    pairs = audit.paired_rows(rows)
    assert math.isclose(pairs[0]["delta_auc"], 0.05)


def test_duplicate_arm_is_rejected():
    rows = [_row(arm="M0"), _row(arm="M0"), _row(arm="M1")]
    checked = audit.audit_rows(rows)
    assert not checked["valid"]
    assert checked["duplicate_keys"]


def test_missing_pair_is_rejected():
    checked = audit.audit_rows([_row(arm="M0")])
    assert not checked["valid"]
    assert checked["missing_or_mismatched_pairs"]


def test_unrelated_arms_do_not_enter_pairing():
    rows = [_row(arm="M0"), _row(arm="M1"), _row(arm="M2")]
    pairs = audit.paired_rows(rows)
    assert len(pairs) == 1
    checked = audit.audit_rows(rows)
    assert checked["arms"] == ["M0", "M1", "M2"]


def test_unexpected_origin_is_rejected():
    checked = audit.audit_rows([_row(origin="not-an-origin", arm="M0")])
    assert not checked["valid"]
    assert checked["unexpected_origins"]


def run_all_tests() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)}/{len(tests)} audit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all_tests())

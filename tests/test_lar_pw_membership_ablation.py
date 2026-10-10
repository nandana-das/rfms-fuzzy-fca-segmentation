"""Compatibility checks for the existing LAR-PW ablation artifacts.

These tests validate naming and identity only; they do not rerun the experiment
or duplicate the membership implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import lar_pw_membership_identity as identity  # noqa: E402


def test_historical_m1_maps_to_canonical_lar_pw_name():
    assert identity.canonical_method_name("M1") == "M1-LAR-PW"
    assert identity.canonical_method_name("M1-LAR-PW") == "M1-LAR-PW"
    assert identity.HISTORICAL_MEMBERSHIP_TYPE == "lar_pw"


def test_normalized_manifest_preserves_historical_mapping():
    manifest_path = ROOT_DIR / "results" / "lar_pw_membership_ablation" / "run_manifest_normalized.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonical_method_name"] == "M1-LAR-PW"
    assert manifest["historical_internal_identifier"] == "M1"
    assert manifest["historical_to_canonical_mapping"]["M1"] == "M1-LAR-PW"
    assert manifest["verification"]["new_evaluation_performed"] is False

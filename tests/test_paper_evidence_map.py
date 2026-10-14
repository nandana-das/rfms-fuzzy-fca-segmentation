from pathlib import Path

from scripts import validate_paper_evidence_map as validator


def _map_text() -> str:
    return (validator.ROOT / "docs" / "PAPER_EVIDENCE_MAP.md").read_text(encoding="utf-8")


def test_temporal_followup_entries_are_valid_and_traceable(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    problems, counts = validator.validate(_map_text())

    assert problems == []
    assert counts["extra"][2] == 5


def test_missing_followup_source_is_reported(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    text = _map_text().replace(
        "`results/temporal_second_seed/SEED_REPLICATION.md`",
        "`results/temporal_second_seed/MISSING_REPLICATION.md`",
        1,
    )

    problems, _ = validator.validate(text)

    assert ("path missing", "results/temporal_second_seed/MISSING_REPLICATION.md") in problems


def test_invalid_followup_identifier_is_reported(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    text = _map_text().replace("| E5-LUKASIEWICZ |", "| E5-UNSUPPORTED |", 1)

    problems, _ = validator.validate(text)

    assert ("follow-up IDs", ["E5-LUKASIEWICZ", "E5-UNSUPPORTED"]) in problems


# --- strict lookup used by the builder -------------------------------------------------

import math

import pandas as pd
import pytest

from scripts.evidence_lookup import EvidenceLookupError, qfmt, qrow, seedrow


def _frame():
    return pd.DataFrame([
        {"dataset": "D", "scope": "o1", "comparison": "A - B", "metric": "auc", "delta": 0.5, "p_holm": float("nan")},
        {"dataset": "D", "scope": "POOLED", "comparison": "A - B", "metric": "auc", "delta": -0.0035, "p_holm": 0.042},
        {"dataset": "D", "scope": "POOLED", "comparison": "A - B", "metric": "spend_r2", "delta": 0.001, "p_holm": 0.56},
    ])


def test_qrow_selects_pooled_row_not_first_origin():
    r = qrow(_frame(), "A - B", "auc", "D", expect="lower")
    assert r.scope == "POOLED" and qfmt(r) == "Δ -0.0035 (Holm p 0.0420)"


def test_qrow_rejects_missing_comparison():
    with pytest.raises(EvidenceLookupError, match="comparison missing"):
        qrow(_frame(), "A - C", "auc", "D")


def test_qrow_rejects_duplicated_comparison():
    f = pd.concat([_frame(), _frame().iloc[[1]]], ignore_index=True)
    with pytest.raises(EvidenceLookupError, match="2 rows match"):
        qrow(f, "A - B", "auc", "D")


def test_qrow_rejects_non_finite_p_where_expected():
    f = _frame()
    f.loc[1, "p_holm"] = float("nan")
    with pytest.raises(EvidenceLookupError, match="finite Holm p"):
        qrow(f, "A - B", "auc", "D")
    assert math.isnan(qrow(f, "A - B", "auc", "D", expect_p=False).p_holm)


def test_qrow_rejects_claimed_direction_that_disagrees():
    with pytest.raises(EvidenceLookupError, match="claim says 'higher'"):
        qrow(_frame(), "A - B", "auc", "D", expect="higher")
    with pytest.raises(EvidenceLookupError, match="claim says 'lower'"):
        qrow(_frame(), "A - B", "spend_r2", "D", expect="lower")


def test_seedrow_rejects_missing_and_non_finite():
    sd = pd.DataFrame([{"dataset": "D", "metric": "auc", "delta_seed0": 0.1, "delta_seed1": float("nan"),
                        "p_holm_seed0": 0.01, "p_holm_seed1": 0.01, "verdict": "replicates"}])
    with pytest.raises(EvidenceLookupError, match="comparison missing"):
        seedrow(sd, "D", "spend_r2")
    with pytest.raises(EvidenceLookupError, match="non-finite delta_seed1"):
        seedrow(sd, "D", "auc")


# --- validator: numerical rendering and sources -----------------------------------------

def test_validator_catches_per_origin_row_rendered_as_pooled(monkeypatch):
    """The original defect: E5 showed a per-origin row (Δ +0.0005, Holm p nan) as the pooled result."""
    monkeypatch.chdir(validator.ROOT)
    text = _map_text().replace("Dunnhumby AUC (Δ -0.0035 (Holm p 0.0420))", "Dunnhumby AUC (Δ +0.0005 (Holm p nan))", 1)
    problems, _ = validator.validate(text)
    assert ("follow-up non-finite value", "E5-LUKASIEWICZ") in problems
    assert any(p[0] == "follow-up value" and p[1] == "E5-LUKASIEWICZ" for p in problems)


def test_validator_catches_wrong_rendered_number(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    text = _map_text().replace("Δ +0.0049 (Holm p 0.0220)", "Δ +0.0094 (Holm p 0.0220)", 1)
    problems, _ = validator.validate(text)
    assert any(p[0] == "follow-up value" and p[1] == "E4-MATCHED-COMPLEXITY" for p in problems)


def test_validator_catches_wrong_second_seed_value(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    text = _map_text().replace("+0.0090 → +0.0098, replicates", "+0.0090 → +0.0198, replicates", 1)
    problems, _ = validator.validate(text)
    assert any(p[0] == "second-seed value" for p in problems)


def test_validator_checks_tests_paths_and_untracked_sources(monkeypatch):
    monkeypatch.chdir(validator.ROOT)
    ignored = "results/dunnhumby_matched_complexity_paired_inference/customer_predictions.csv.gz"
    text = (_map_text()
            .replace("`tests/test_lukasiewicz_tnorm.py`", "`tests/test_missing_tnorm.py`", 1)
            .replace("`results/dunnhumby_matched_complexity_paired_inference/DATA_NOTE.md`", f"`{ignored}`", 1))
    problems, _ = validator.validate(text)
    assert ("path missing", "tests/test_missing_tnorm.py") in problems
    # git-ignored file: untracked if present locally, missing on a clean checkout
    assert any(p in (("follow-up source untracked", "E4-MATCHED-COMPLEXITY", ignored),
                     ("follow-up source missing", "E4-MATCHED-COMPLEXITY", ignored)) for p in problems)

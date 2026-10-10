import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rfm_tail_information as tail  # noqa: E402


def test_frozen_feature_builder_is_delegated_unchanged(monkeypatch):
    expected = (np.array([[1.0]]), np.array([[2.0]]), {"frozen": True})

    def fake_builder(arm, tr, te):
        assert arm == "fuzzy_rfm_fca"
        return expected

    monkeypatch.setattr(tail.bl, "build_features", fake_builder)
    got = tail.build_features(
        "fuzzy_rfm_fca",
        pd.DataFrame({"R": [1], "F": [2], "M": [3]}),
        pd.DataFrame({"R": [4], "F": [5], "M": [6]}),
    )
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert got[2] == expected[2]


def test_tail_scaler_is_fit_on_training_data_only(monkeypatch):
    def fake_builder(arm, tr, te):
        return np.zeros((len(tr), 1)), np.zeros((len(te), 1)), {}

    monkeypatch.setattr(tail.bl, "build_features", fake_builder)
    tr = pd.DataFrame({"R": [0.0, 10.0], "F": [1.0, 3.0], "M": [2.0, 6.0]})
    te = pd.DataFrame({"R": [1000.0], "F": [1000.0], "M": [1000.0]})
    x_tr, x_te, _ = tail.build_features("tail_augmented_fuzzy_rfm_fca", tr, te)
    expected = (np.log1p(tr[["R", "F", "M"]]) - np.log1p(tr[["R", "F", "M"]]).mean()) / np.log1p(
        tr[["R", "F", "M"]]
    ).std(ddof=0)
    assert np.allclose(x_tr[:, 1:], expected.to_numpy())
    assert np.all(np.abs(x_te[:, 1:]) > 1.0)


def test_all_arms_share_identical_customer_splits(monkeypatch):
    ids_seen = {arm: [] for arm in tail.ARMS}
    fit_calls = []

    def fake_builder(arm, tr, te):
        ids_seen[arm].append(tuple(te["CustomerID"]))
        return np.ones((len(tr), 1)), np.ones((len(te), 1)), {}

    def fake_fit_predict(arm, X_tr, X_te, tr, seed):
        fit_calls.append((arm, tuple(tr["CustomerID"]), len(X_te)))
        return np.zeros(len(X_te)), np.zeros(len(X_te)), np.zeros(len(X_te))

    monkeypatch.setattr(tail, "build_features", fake_builder)
    monkeypatch.setattr(tail.bl, "fit_predict", fake_fit_predict)
    df = pd.DataFrame({
        "CustomerID": [str(i) for i in range(20)],
        "repurchased": [0, 1] * 10,
        "future_spend": 0.0,
        "future_invoices": 0,
        "R": 1.0,
        "F": 1.0,
        "M": 1.0,
    })
    tail.run_origin({"dataset": "test", "origin": "o", "df": df})
    assert all(ids_seen[arm] == ids_seen[tail.ARMS[0]] for arm in tail.ARMS)
    assert len(fit_calls) == tail.bl.N_FOLDS * len(tail.ARMS)


def test_frozen_prediction_reproduction_guard(monkeypatch):
    values = {
        arm: {metric: np.array([1.0, 2.0]) for metric in ("prob", "sp", "inv")}
        for arm in tail.ARMS
    }
    monkeypatch.setattr(tail.bl, "run_origin", lambda cohort: (values, []))
    monkeypatch.setattr(tail, "run_origin", lambda cohort: (values, [], []))
    tail.verify_frozen_reproduction({"df": pd.DataFrame()})


def test_output_isolation_rejects_existing_directory(tmp_path, monkeypatch):
    out = tmp_path / "rfm_tail_information"
    out.mkdir()
    (out / "stale.csv").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(tail, "OUT", out)
    with pytest.raises(FileExistsError):
        tail._safe_output()

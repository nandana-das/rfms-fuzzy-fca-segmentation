import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import membership_function_ablation as abl  # noqa: E402


def _fixture(n_train=100, n_test=24):
    def frame(n, offset):
        i = np.arange(n, dtype=float) + offset
        return pd.DataFrame({
            "CustomerID": [f"c{int(x)}" for x in i],
            "R": i + 1.0,
            "F": (i % 97) + 1.0,
            "M": (i + 1.0) ** 1.7,
        })

    tr = frame(n_train, 0)
    te = frame(n_test, 1000)
    tr["repurchased"] = (np.arange(n_train) % 3 != 0).astype(int)
    tr["future_spend"] = np.where(tr["repurchased"], tr["M"] / 10.0, 0.0)
    tr["future_invoices"] = tr["repurchased"]
    return tr, te


def test_all_methods_are_finite_bounded_and_normalized_where_required():
    tr, te = _fixture()
    scored, _ = abl._score_train_test(tr, te)
    for method in abl.METHODS:
        got_tr, got_te, _ = abl.fit_memberships(method, scored, te)
        for values in (got_tr.to_numpy(), got_te.to_numpy()):
            assert np.isfinite(values).all()
            assert ((values >= 0.0) & (values <= 1.0)).all()
        if method != "M0":
            for values in (got_tr.to_numpy(), got_te.to_numpy()):
                assert np.allclose(values.reshape(-1, 5).sum(axis=1), 1.0, atol=1e-12)


def test_coincident_centroids_are_deterministic_and_finite():
    tr = pd.DataFrame({
        "CustomerID": ["a", "b", "c", "d", "e"],
        "R": [1.0] * 5, "F": [2.0] * 5, "M": [3.0] * 5,
        "R_score": [1] * 5, "F_score": [1] * 5, "M_score": [1] * 5,
    })
    te = tr[["CustomerID", "R", "F", "M"]].copy()
    for method in abl.METHODS:
        got, _, params = abl.fit_memberships(method, tr, te)
        assert np.isfinite(got.to_numpy()).all()
        assert ((got.to_numpy() >= 0) & (got.to_numpy() <= 1)).all()
        assert params["dimensions"]["R"]["width_floor"] > 0


def test_extreme_and_negative_values_are_deterministic_for_log_arm():
    tr = pd.DataFrame({
        "CustomerID": [f"c{i}" for i in range(10)],
        "R": np.arange(10, dtype=float), "F": np.arange(10, dtype=float) + 1,
        "M": np.arange(10, dtype=float) ** 3 + 1,
        "R_score": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "F_score": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "M_score": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    })
    te = tr[["CustomerID", "R", "F", "M"]].copy()
    te.loc[0, ["R", "F", "M"]] = [-100.0, 1e300, 1e300]
    first = abl.fit_memberships("M3", tr, te)[1].to_numpy()
    second = abl.fit_memberships("M3", tr, te)[1].to_numpy()
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


def test_log_arm_parameters_are_train_only():
    tr, te = _fixture()
    scored, _ = abl._score_train_test(tr, te)
    _, te_a, params_a = abl.fit_memberships("M3", scored, te)
    te_changed = te.copy()
    te_changed[["R", "F", "M"]] = 1e6
    _, te_b, params_b = abl.fit_memberships("M3", scored, te_changed)
    assert params_a == params_b
    assert not np.allclose(te_a.to_numpy(), te_b.to_numpy())


def test_quantile_cutoffs_and_scores_match_frozen_tie_preserving_rule():
    import baseline_ladder_rolling_origin as frozen  # noqa: WPS433

    values = np.array([0] * 8 + [1] * 4 + [2] * 4 + [3] * 2 + [4] * 2, dtype=float)
    rfm = pd.DataFrame({
        "CustomerID": [f"c{i}" for i in range(len(values))],
        "R": values,
        "F": np.array([1] * 5 + [2] * 5 + [3] * 4 + [4] * 3 + [5] * 3, dtype=float),
        "M": np.array([10] * 8 + [20] * 4 + [30] * 4 + [40] * 2 + [50] * 2, dtype=float),
    })
    ours_cutoffs = abl.fit_quantile_cutoffs(rfm)
    frozen_cutoffs = frozen.fit_quantile_cutoffs(rfm)
    for dim in abl.DIMS:
        assert np.array_equal(ours_cutoffs[dim], frozen_cutoffs[dim])
    ours_scores = abl.quantile_scores(rfm, ours_cutoffs)
    frozen_scores = frozen.quantile_scores(rfm, frozen_cutoffs)
    for dim in abl.DIMS:
        assert np.array_equal(ours_scores[f"{dim}_score"], frozen_scores[f"{dim}_score"])
        # Equal raw values must never be split across bands.
        assert ours_scores.groupby(dim)[f"{dim}_score"].nunique().max() == 1


def test_repeated_cutpoint_rejection_matches_frozen_rule():
    import baseline_ladder_rolling_origin as frozen  # noqa: WPS433

    rfm = pd.DataFrame({
        "CustomerID": [f"c{i}" for i in range(20)],
        "R": [0.0] * 20,
        "F": np.arange(20, dtype=float),
        "M": np.arange(20, dtype=float) + 10,
    })
    with pytest.raises(ValueError):
        abl.fit_quantile_cutoffs(rfm)
    with pytest.raises(ValueError):
        frozen.fit_quantile_cutoffs(rfm)


def test_m0_reproduces_frozen_feature_matrix_and_predictions():
    import baseline_ladder_rolling_origin as frozen  # noqa: WPS433

    tr, te = _fixture()
    ours_xtr, ours_xte, ours_info = abl.build_features("M0", tr, te)
    frozen_xtr, frozen_xte, frozen_info = frozen.build_features("fuzzy_rfm_fca", tr, te)
    assert np.array_equal(ours_xtr, frozen_xtr)
    assert np.array_equal(ours_xte, frozen_xte)
    assert ours_info["n_retained_concepts"] == frozen_info["n_features"]
    ours_pred = abl.fit_predict(ours_xtr, ours_xte, tr, seed=0)
    frozen_pred = frozen.fit_predict("fuzzy_rfm_fca", frozen_xtr, frozen_xte, tr, seed=0)
    for a, b in zip(ours_pred, frozen_pred):
        assert np.allclose(a, b, atol=1e-12)


def test_all_methods_use_identical_customer_splits(monkeypatch):
    tr, _ = _fixture(n_train=100, n_test=0)
    seen = {m: [] for m in abl.METHODS}

    def fake_build(method, train, test):
        seen[method].append(tuple(test["CustomerID"]))
        return np.ones((len(train), 1)), np.ones((len(test), 1)), {"concept_signatures": []}

    def fake_predict(x_train, x_test, train, seed):
        return np.zeros(len(x_test)), np.zeros(len(x_test)), np.zeros(len(x_test))

    monkeypatch.setattr(abl, "build_features", fake_build)
    monkeypatch.setattr(abl, "fit_predict", fake_predict)
    cohort = {"dataset": "test", "origin": "o", "pooled": True, "df": tr}
    _, _, splits, _ = abl.run_origin(cohort, boot_n=20)
    assert all(seen[m] == seen[abl.METHODS[0]] for m in abl.METHODS)
    assert splits.groupby("fold")["test_customer_ids"].nunique().eq(1).all()


def test_all_methods_use_identical_bootstrap_customer_indices(monkeypatch):
    tr, _ = _fixture(n_train=100, n_test=0)
    captured = []

    def fake_build(method, train, test):
        return np.ones((len(train), 1)), np.ones((len(test), 1)), {"concept_signatures": []}

    def fake_predict(x_train, x_test, train, seed):
        return np.linspace(0.1, 0.9, len(x_test)), np.arange(len(x_test), dtype=float), np.arange(len(x_test), dtype=float)

    def fake_bootstrap(frame, pred, indices):
        captured.append(indices.copy())
        base = np.arange(len(indices), dtype=float)
        return {"auc": base, "spend_r2": base + 1.0, "invoice_r2": base + 2.0}

    monkeypatch.setattr(abl, "build_features", fake_build)
    monkeypatch.setattr(abl, "fit_predict", fake_predict)
    monkeypatch.setattr(abl, "_bootstrap_metric", fake_bootstrap)
    cohort = {"dataset": "test", "origin": "o", "pooled": True, "df": tr}
    _, _, _, aux = abl.run_origin(cohort, boot_n=11)
    assert len(captured) == len(abl.METHODS)
    assert all(np.array_equal(captured[0], item) for item in captured[1:])
    for method in abl.METHODS[1:]:
        assert np.array_equal(
            aux["paired_differences"][method]["auc"],
            aux["bootstrap"][method]["auc"] - aux["bootstrap"]["M0"]["auc"],
        )


def test_primary_365_day_cohort_is_excluded_from_pooled_inference():
    rows = []
    paired = {}
    for origin, pooled, delta in (("rolling", True, 1.0), ("primary_365d", False, 100.0)):
        paired[("Online Retail II", origin)] = {}
        for method in abl.METHODS[1:]:
            paired[("Online Retail II", origin)][method] = {
                metric: np.full(20, delta, dtype=float)
                for metric in ("auc", "spend_r2", "invoice_r2")
            }
        for method in abl.METHODS:
            rows.append({
                "dataset": "Online Retail II", "origin": origin, "pooled": pooled,
                "method": method, "auc": delta if method != "M0" else 0.0,
                "spend_r2": delta if method != "M0" else 0.0,
                "invoice_r2": delta if method != "M0" else 0.0,
            })
    comparisons = abl.aggregate_comparisons(pd.DataFrame(rows), paired)
    pooled = comparisons[comparisons["scope"] == "POOLED"]
    primary = comparisons[comparisons["scope"] == "primary_365d"]
    assert len(pooled) == 9
    assert len(primary) == 9
    assert np.allclose(pooled["delta"], 1.0)
    assert np.allclose(primary["delta"], 100.0)


def test_output_isolation_rejects_nonempty_directory(monkeypatch):
    # The repository already contains preserved experiment artifacts; using
    # that directory keeps this test read-only and avoids sandbox temp-dir
    # restrictions.
    out = abl.ROOT_DIR / "results"
    monkeypatch.setattr(abl, "OUT_DIR", out)
    with pytest.raises(FileExistsError):
        abl._prepare_output()

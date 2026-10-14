import numpy as np
import pandas as pd
import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lrfm_extension as ext  # noqa: E402


def test_length_uses_only_transactions_at_origin_and_single_day_is_zero():
    tx = pd.DataFrame(
        {
            "CustomerID": ["a", "a", "a", "b"],
            "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-04", "2020-02-01", "2020-01-03"]),
        }
    )
    base = [{
        "dataset": "Online Retail II",
        "origin": "2020-01-10",
        "pooled": True,
        "df": pd.DataFrame({"CustomerID": ["a", "b"]}),
    }]
    got = ext._with_length(base, tx, "CustomerID", "InvoiceDate")[0]["df"]
    assert got.set_index("CustomerID")["L"].to_dict() == {"a": 3.0, "b": 0.0}


def test_retail2_same_day_cutoff_and_id_normalization():
    tx = pd.DataFrame(
        {
            "CustomerID": [123, 123, 456],
            "InvoiceDate": pd.to_datetime(
                ["2010-09-10 08:00:00", "2010-09-10 23:00:00", "2010-09-11 08:00:00"]
            ),
        }
    )
    base = [{
        "dataset": "Online Retail II",
        "origin": "2010-09-10",
        "pooled": True,
        "df": pd.DataFrame({"CustomerID": ["123"]}),
    }]
    got = ext._with_length(base, tx, "CustomerID", "InvoiceDate")[0]["df"]
    assert got["L"].tolist() == [0.0]


def test_numeric_day_length_is_computed_without_datetime_conversion():
    tx = pd.DataFrame({"household_key": ["a", "a", "b"], "DAY": [10, 17, 20]})
    base = [{
        "dataset": "Dunnhumby",
        "origin": "day 20",
        "pooled": True,
        "df": pd.DataFrame({"CustomerID": ["a", "b"]}),
    }]
    got = ext._with_length(base, tx, "household_key", "DAY")[0]["df"]
    assert got.set_index("CustomerID")["L"].to_dict() == {"a": 7.0, "b": 0.0}


def test_merged_cutpoints_remove_empty_bands():
    values = np.array([0] * 8 + [10] * 2, dtype=float)
    cuts = ext.merged_cutpoints(values)
    scores = ext.quantile_scores(
        pd.DataFrame({"CustomerID": range(len(values)), "X": values}),
        {"X": cuts},
        dims=("X",),
    )["X_score"]
    assert len(cuts) < 4
    assert set(scores) == set(range(1, int(scores.max()) + 1))


def test_one_level_membership_is_all_ones():
    scored = pd.DataFrame({"R": [0.0, 0.0], "R_score": [1, 1]})
    mu, centroids, _ = ext.compute_fuzzy_memberships_n(
        pd.DataFrame(
            {
                "R": [0.0, 0.0],
                "F": [1.0, 1.0],
                "M": [2.0, 2.0],
                "L": [0.0, 0.0],
                "R_score": [1, 1],
                "F_score": [1, 1],
                "M_score": [1, 1],
                "L_score": [1, 1],
            }
        )
    )
    assert all(np.allclose(mu.filter(regex=f"^{dim}").to_numpy(), 1.0) for dim in ext.LRFM_DIMS)
    assert all(len(centroids[dim]) == 1 for dim in ext.LRFM_DIMS)


def test_guard_failure_prevents_result_files(monkeypatch, tmp_path):
    out = tmp_path / "lrfm_extension"
    monkeypatch.setattr(ext, "OUT", out)

    def fail(_):
        raise RuntimeError("guard failed")

    monkeypatch.setattr(ext, "run_guards", fail)
    with pytest.raises(RuntimeError, match="guard failed"):
        ext.run_experiment([])
    assert not out.exists()


def test_bootstrap_population_is_dataset_specific_and_origin_aligned():
    def cohort(dataset, origin, customers, pooled=True):
        return {
            "dataset": dataset,
            "origin": origin,
            "pooled": pooled,
            "df": pd.DataFrame({"CustomerID": customers}),
        }

    cohorts = [
        cohort("A", "a1", ["a", "b"]),
        cohort("A", "a2", ["b", "c"]),
        cohort("A", "primary", ["outside"], pooled=False),
        cohort("B", "b1", ["x", "y"]),
    ]

    a_cs, a_union, a_positions = ext._dataset_bootstrap_population(cohorts, "A")
    b_cs, b_union, b_positions = ext._dataset_bootstrap_population(cohorts, "B")

    assert list(a_union) == ["a", "b", "c"]
    assert list(b_union) == ["x", "y"]
    assert [c["origin"] for c in a_cs] == ["a1", "a2"]
    assert [c["origin"] for c in b_cs] == ["b1"]

    a_counts = np.array([10.0, 20.0, 30.0])
    b_counts = np.array([40.0, 50.0])
    assert np.array_equal(a_counts[a_positions[("A", "a1")]], [10.0, 20.0])
    assert np.array_equal(a_counts[a_positions[("A", "a2")]], [20.0, 30.0])
    assert np.array_equal(b_counts[b_positions[("B", "b1")]], [40.0, 50.0])


def test_bootstrap_origin_deltas_are_paired_and_averaged_per_draw():
    got = ext._mean_origin_deltas([
        np.array([1.0, 3.0, 5.0]),
        np.array([3.0, 5.0, 7.0]),
    ])
    assert np.array_equal(got, [2.0, 4.0, 6.0])


def test_cdnow_extension_state_is_restored(monkeypatch):
    original = (
        ext.bl._quintile_scored,
        ext.bl.compute_fuzzy_memberships,
        ext.bl.ARMS,
        ext.bl.build_features,
    )
    monkeypatch.setattr(ext.cc, "install_extension", lambda: (
        setattr(ext.bl, "_quintile_scored", object()),
        setattr(ext.bl, "compute_fuzzy_memberships", object()),
        setattr(ext.bl, "ARMS", ["temporary"]),
        setattr(ext.bl, "build_features", object()),
    ))
    with ext._cdnow_extension_state():
        assert ext.bl.ARMS is original[2]
        assert ext.bl.build_features is original[3]
    assert (
        ext.bl._quintile_scored,
        ext.bl.compute_fuzzy_memberships,
        ext.bl.ARMS,
        ext.bl.build_features,
    ) == original


def test_cdnow_extension_preserves_active_lrfm_dispatch(monkeypatch):
    active_builder = object()
    active_arms = list(ext.ARMS)
    original = ext.bl.ARMS, ext.bl.build_features
    ext.bl.ARMS = active_arms
    ext.bl.build_features = active_builder
    try:
        monkeypatch.setattr(ext.cc, "install_extension", lambda: (
            setattr(ext.bl, "_quintile_scored", object()),
            setattr(ext.bl, "compute_fuzzy_memberships", object()),
            setattr(ext.bl, "ARMS", ["cdnow-only"]),
            setattr(ext.bl, "build_features", object()),
        ))
        with ext._dataset_extension_state("CDNOW"):
            assert ext.bl.ARMS == active_arms
            assert ext.bl.build_features is active_builder
    finally:
        ext.bl.ARMS, ext.bl.build_features = original


def test_cdnow_experiment_dispatch_installs_merged_cutpoints(monkeypatch):
    observed = []

    def fake_builder(arm, tr, te):
        observed.append(ext.bl._quintile_scored is ext.cc.quintile_scored_merged)
        return np.empty((len(tr), 1)), np.empty((len(te), 1)), {}

    monkeypatch.setattr(ext, "_V1_BUILD_FEATURES", fake_builder)
    monkeypatch.setattr(ext, "_V2_BUILD_FEATURES", fake_builder)
    with ext._dataset_extension_state("CDNOW"):
        ext.build_features(
            "fuzzy_rfm_fca",
            pd.DataFrame({"CustomerID": ["a"], "R": [1], "F": [1], "M": [1]}),
            pd.DataFrame({"CustomerID": ["b"], "R": [1], "F": [1], "M": [1]}),
        )
    assert observed == [True]


def test_oof_validation_requires_every_planned_arm():
    complete = {arm: {} for arm in ext.ARMS}
    ext._validate_oof_arms(complete, "CDNOW", "1997-09-30")
    with pytest.raises(RuntimeError, match="log_lrfm"):
        ext._validate_oof_arms(
            {arm: {} for arm in ext.ARMS if arm != "log_lrfm"},
            "CDNOW",
            "1997-09-30",
        )


def test_bootstrap_draws_flatten_to_50000_for_every_planned_key():
    for dataset in ("Dunnhumby", "Online Retail II", "CDNOW"):
        for comparison in ext.COMPARISONS:
            for metric in ext.METRICS:
                per_seed = [
                    [np.full(250, seed + chunk, dtype=float) for chunk in range(40)]
                    for seed in ext.SEEDS
                ]
                got = ext._combine_bootstrap_draws(per_seed, len(ext.SEEDS) * ext.BOOT_N)
                assert got.shape == (50_000,)


def test_bootstrap_draw_count_assertion_rejects_incomplete_chunks():
    incomplete = [[np.zeros(250) for _ in range(39)] for _ in ext.SEEDS]
    with pytest.raises(RuntimeError, match="draw count mismatch"):
        ext._combine_bootstrap_draws(incomplete, len(ext.SEEDS) * ext.BOOT_N)


def _checkpoint_fixture():
    df = pd.DataFrame({
        "CustomerID": ["a", "b", "c", "d", "e"],
        "R": [1, 2, 3, 4, 5], "F": [1, 2, 3, 4, 5],
        "M": [10, 20, 30, 40, 50], "L": [0, 1, 2, 3, 4],
        "repurchased": [0, 1, 0, 1, 0],
        "future_spend": [0., 2., 0., 4., 0.],
        "future_invoices": [0, 1, 0, 2, 0],
    })
    cohort = {"dataset": "CDNOW", "origin": "test-origin", "pooled": True, "df": df}
    oof = {
        arm: {
            "prob": np.linspace(0.1, 0.9, len(df)),
            "sp": np.linspace(0.2, 1.0, len(df)),
            "inv": np.linspace(0.3, 1.1, len(df)),
        }
        for arm in ext.ARMS
    }
    return cohort, oof


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "OUT", tmp_path / "results")
    cohort, oof = _checkpoint_fixture()
    ext._write_checkpoint(cohort, 0, oof, [])
    loaded, info = ext._load_checkpoint(cohort, 0)
    assert info == []
    for arm in ext.ARMS:
        for metric in ("prob", "sp", "inv"):
            np.testing.assert_allclose(loaded[arm][metric], oof[arm][metric], rtol=0, atol=1e-15)


def test_incomplete_checkpoint_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "OUT", tmp_path / "results")
    cohort, oof = _checkpoint_fixture()
    ext._write_checkpoint(cohort, 0, oof, [])
    csv_path, _ = ext._checkpoint_paths(cohort["dataset"], cohort["origin"], 0)
    frame = pd.read_csv(csv_path).iloc[:-1]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="row count"):
        ext._load_checkpoint(cohort, 0)


def test_stale_checkpoint_metadata_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "OUT", tmp_path / "results")
    cohort, oof = _checkpoint_fixture()
    ext._write_checkpoint(cohort, 0, oof, [])
    _, metadata_path = ext._checkpoint_paths(cohort["dataset"], cohort["origin"], 0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arms"] = ["stale-arm"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        ext._load_checkpoint(cohort, 0)


def test_checkpoint_rejects_changed_implementation_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "OUT", tmp_path / "results")
    cohort, oof = _checkpoint_fixture()
    ext._write_checkpoint(cohort, 0, oof, [])
    monkeypatch.setattr(ext, "_implementation_fingerprint", lambda: "changed-implementation")
    with pytest.raises(ValueError, match="implementation_fingerprint"):
        ext._load_checkpoint(cohort, 0)


def test_resume_uses_checkpoints_without_model_fitting(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "OUT", tmp_path / "results")
    cohort, oof = _checkpoint_fixture()
    for seed in ext.SEEDS:
        ext._write_checkpoint(cohort, seed, oof, [])
    monkeypatch.setattr(ext, "run_guards", lambda cohorts: 0.0)
    monkeypatch.setattr(ext.bl, "run_origin", lambda _: pytest.fail("model fitting was repeated"))
    captured = {}
    monkeypatch.setattr(
        ext, "_write_inference",
        lambda cohorts, predictions, metrics, reproduction: captured.update({
            "predictions": predictions, "metrics": metrics, "reproduction": reproduction
        }),
    )
    ext.run_experiment([cohort], resume=True)
    assert len(captured["predictions"]) == len(ext.SEEDS)
    assert len(captured["metrics"]) == len(ext.SEEDS) * len(ext.ARMS)
    assert captured["reproduction"] == 0.0
    manifest = json.loads((tmp_path / "results" / "checkpoints" / "manifest.json").read_text())
    assert manifest["status"] == "complete"

"""Regression tests for temporal R/F/M orientation and M0 parity."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
import temporal_rfm_fca_ablation as temporal  # noqa: E402


def _rfm_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": ["a", "b", "c", "d", "e"],
            "R": [1.0, 2.0, 3.0, 4.0, 5.0],
            "F": [1.0, 2.0, 3.0, 4.0, 5.0],
            "M": [10.0, 20.0, 30.0, 40.0, 50.0],
            "T": [1.0, 2.0, 3.0, 4.0, 5.0],
            "repeat_history": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_temporal_rfm_scores_match_m0_orientation():
    df = _rfm_frame()
    m0_scored, _, _ = bl._fuzzy_context(df, df)
    temporal_scored, _ = temporal._score_temporal(df, temporal.T_DIMS)
    for dim in temporal.RFM_DIMS:
        np.testing.assert_array_equal(
            m0_scored[f"{dim}_score"].to_numpy(),
            temporal_scored[f"{dim}_score"].to_numpy(),
        )


def test_older_recency_gets_lower_favourable_band():
    df = _rfm_frame()
    scored, _ = temporal._score_temporal(df, temporal.T_DIMS)
    assert scored.loc[0, "R_score"] == 5
    assert scored.loc[len(df) - 1, "R_score"] == 1
    assert scored.loc[0, "R_score"] > scored.loc[len(df) - 1, "R_score"]


def test_temporal_rfm_memberships_match_m0_memberships():
    df = _rfm_frame()
    _, m0_mu, _ = bl._fuzzy_context(df, df)
    scored, _ = temporal._score_temporal(df, temporal.T_DIMS)
    temporal_mu, _ = temporal._fuzzy_memberships(scored, temporal.RFM_DIMS)
    for col in m0_mu.columns:
        np.testing.assert_allclose(m0_mu[col].to_numpy(), temporal_mu[col].to_numpy())


def test_temporal_features_rfm_only_arm_matches_m0_builder():
    # Regression: the RFM-only arm used to raise KeyError: 'T'.
    rng = np.random.default_rng(7)
    n = 300
    t = rng.gamma(2.0, 10.0, n)
    t[rng.random(n) < 0.3] = np.nan
    df = pd.DataFrame({
        "CustomerID": [str(i) for i in range(n)],
        "R": rng.gamma(2.0, 30.0, n),
        "F": rng.gamma(2.0, 5.0, n),
        "M": rng.gamma(2.0, 100.0, n),
        "T": t,
        "repeat_history": (~np.isnan(t)).astype(int),
        "L": rng.gamma(2.0, 100.0, n),
    })
    tr, te = df.iloc[:240].reset_index(drop=True), df.iloc[240:].reset_index(drop=True)
    x_tr, x_te, _ = temporal._temporal_features("fuzzy_rfm", tr, te)
    m0_tr, m0_te, _ = bl.build_features("fuzzy_rfm_fca", tr, te)
    np.testing.assert_allclose(x_tr, m0_tr)
    np.testing.assert_allclose(x_te, m0_te)

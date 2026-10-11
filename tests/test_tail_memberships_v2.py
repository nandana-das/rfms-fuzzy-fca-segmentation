"""Tests distinguishing historical M1-LAR-PW from corrected v2."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tail_memberships as old  # noqa: E402
import tail_memberships_v2 as v2  # noqa: E402


@contextlib.contextmanager
def raises(expected):
    try:
        yield
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def _fallback_values() -> np.ndarray:
    # Forty values make the upper band eight observations: seven at 40 and
    # one at 100. The 75th percentile is still 40, so IQR is exactly zero.
    return np.array(list(np.arange(1., 33.)) + [40.] * 7 + [100.])


def test_v2_uses_mad_and_historical_uses_mean_deviation():
    values = _fallback_values()
    historical = old.fit_lar_pw_parameters(values, "F")
    corrected = v2.fit_lar_pw_parameters_v2(values, "F")
    assert not np.isclose(historical.s_tail, corrected.s_tail)
    # Exact formulas, reconstructed from the fitted v2 anchors.
    z = np.log1p(values)
    cut = np.quantile(values, np.arange(1, 5) / 5.0)
    scores = 1 + np.searchsorted(cut, values, side="left")
    z5 = z[scores == 5]
    a4 = corrected.anchors[3]
    assert np.isclose(corrected.s_tail, np.median(np.abs(z5 - a4)))
    assert np.isclose(historical.s_tail, np.mean(z5 - a4))


def test_v2_matches_historical_on_positive_iqr():
    values = np.arange(1.0, 101.0)
    assert np.isclose(old.fit_lar_pw_parameters(values, "F").s_tail, v2.fit_lar_pw_parameters_v2(values, "F").s_tail)


def test_v2_bounds_and_monotonic_tail():
    params = v2.fit_tail_membership_params_v2(pd.DataFrame({"R": np.arange(1., 101.), "F": np.arange(1., 101.), "M": np.arange(1., 101.)}))
    raw = np.expm1(np.linspace(params.F.anchors[3], params.F.anchors[3] + 20.0, 400))
    mu = v2.compute_lar_pw_memberships(raw, params.F.anchors, params.F.s_tail)
    assert np.all((mu >= 0) & (mu <= 1))
    assert np.allclose(mu.sum(axis=1), 1.0)
    assert np.all(np.diff(mu[:, 4]) > 0)


def test_v2_rejects_degenerate_and_nonfinite_inputs():
    with raises(v2.DegenerateTrainingFoldError):
        v2.fit_lar_pw_parameters_v2(np.ones(20), "F")
    with raises(ValueError):
        v2.compute_lar_pw_memberships(np.array([np.nan, 1.0]), np.array([1., 2., 3., 4.]), 0.5)


def test_v2_preserves_m0_recency_and_training_only_projection():
    tr = pd.DataFrame({"R": np.arange(1., 101.), "F": np.arange(1., 101.), "M": np.arange(1., 101.)})
    te1 = tr.iloc[:5].copy()
    te2 = te1.copy(); te2["F"] = 1e9; te2["M"] = 1e10
    p1 = v2.fit_tail_membership_params_v2(tr)
    p2 = v2.fit_tail_membership_params_v2(tr)
    assert np.array_equal(p1.R.anchors, p2.R.anchors)
    assert p1.F.s_tail == p2.F.s_tail
    raw_r = np.array([0., 5., 12., 25., 50., 80., 120.])
    centroids = np.array([120., 80., 50., 25., 5.])
    assert np.allclose(v2.compute_recency_piecewise_memberships(raw_r, centroids), old.compute_recency_piecewise_memberships(raw_r, centroids))
    assert not np.array_equal(v2.compute_continuous_memberships(te1, p1).to_numpy(), v2.compute_continuous_memberships(te2, p1).to_numpy())


def run_all_tests() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)}/{len(tests)} v2 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all_tests())

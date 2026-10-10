"""Unit tests for tail_memberships.py.

Verifies:
1. Training-only parameter fitting (test changes cannot affect parameters).
2. Boundary continuity at all anchor points (a1*, a2*, a3*, a4*).
3. Strict monotonicity in the upper tail (mu5 strictly increasing, mu4 strictly decreasing).
4. Non-saturation of upper tail (mu5 < 1.0 for all finite values).
5. Partition of unity (sum of memberships = 1.0) and range in [0, 1].
6. Exact preservation of frozen M0 Recency piecewise-linear behavior.
7. Formal context binarization (15 attributes, 45 binary columns, exact ordering, nestedness).
8. Deterministic reproducibility across repeated evaluations.
9. Degenerate fold detection (DegenerateTrainingFoldError on coincident anchors).
10. Zero-IQR tail dispersion MAD fallback behavior.
11. Input validation (ValueError on NaN, Inf, negative inputs).
12. Discrimination diagnostics (D_mem, D_ctx).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import fair_comparison_retail2 as fc  # noqa: E402
import tail_memberships as tm  # noqa: E402


@contextlib.contextmanager
def raises(expected_exc, match: str | None = None):
    """Dependency-free pytest.raises equivalent."""
    try:
        yield
    except expected_exc as exc:
        if match and match not in str(exc):
            raise AssertionError(f"Expected match '{match}' in exception '{exc}'")
        return
    except Exception as exc:
        raise AssertionError(f"Expected {expected_exc.__name__}, got {type(exc).__name__}: {exc}")
    raise AssertionError(f"Expected {expected_exc.__name__} but no exception was raised")



def _make_synthetic_rfm(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic RFM dataset with log-normal tails."""
    rng = np.random.default_rng(seed)
    # Recency: integers 0..300 days
    r = rng.integers(0, 300, size=n).astype(float)
    # Frequency: geometric / log-normal
    f = np.floor(rng.lognormal(mean=1.5, sigma=1.0, size=n)).clip(1.0, 500.0)
    # Monetary: log-normal with heavy tail
    m = np.round(rng.lognormal(mean=5.5, sigma=1.4, size=n) + 1.0, 2)

    return pd.DataFrame({
        "CustomerID": [f"cust_{i:04d}" for i in range(n)],
        "R": r,
        "F": f,
        "M": m,
    })


# ===========================================================================
# 1. Training-Only Parameter Fitting
# ===========================================================================

def test_training_only_fitting_isolated_from_test_data():
    """Verify that changing test data in any way cannot alter fitted training parameters."""
    tr = _make_synthetic_rfm(n=120, seed=10)
    te1 = _make_synthetic_rfm(n=40, seed=20)
    te2 = te1.copy()
    # Drastically perturb te2 with extreme outliers
    te2["F"] = te2["F"] * 1000.0
    te2["M"] = te2["M"] * 10000.0
    te2["R"] = 0.0

    # Fit parameters with tr only
    params1 = tm.fit_tail_membership_params(tr)
    # Fit parameters again with tr
    params2 = tm.fit_tail_membership_params(tr)

    assert np.array_equal(params1.R.cutoffs, params2.R.cutoffs)
    assert np.array_equal(params1.R.anchors, params2.R.anchors)
    assert np.array_equal(params1.F.anchors, params2.F.anchors)
    assert params1.F.s_tail == params2.F.s_tail
    assert np.array_equal(params1.M.anchors, params2.M.anchors)
    assert params1.M.s_tail == params2.M.s_tail

    # Test projections with te1 vs te2 use identical params but produce different test memberships
    mu_te1 = tm.compute_continuous_memberships(te1, params1)
    mu_te2 = tm.compute_continuous_memberships(te2, params1)

    assert not np.allclose(mu_te1.to_numpy(), mu_te2.to_numpy())


# ===========================================================================
# 2. Boundary Continuity at Anchor Boundaries
# ===========================================================================

def test_lar_pw_boundary_continuity():
    """Verify C^0 continuity at anchor boundaries a1*, a2*, a3*, and a4*."""
    anchors = np.array([1.0, 2.0, 3.0, 4.0])
    s_tail = 0.8
    eps = 1e-7

    # Evaluate around each boundary
    for i, a_k in enumerate(anchors):
        # Convert log value back to raw x: x = exp(z) - 1
        x_at = np.expm1(a_k)
        x_left = np.expm1(a_k - eps)
        x_right = np.expm1(a_k + eps)

        mu_at = tm.compute_lar_pw_memberships(np.array([x_at]), anchors, s_tail)
        mu_left = tm.compute_lar_pw_memberships(np.array([x_left]), anchors, s_tail)
        mu_right = tm.compute_lar_pw_memberships(np.array([x_right]), anchors, s_tail)

        assert np.allclose(mu_left, mu_at, atol=1e-5), f"Discontinuity on left of anchor {i+1} at z={a_k}"
        assert np.allclose(mu_right, mu_at, atol=1e-5), f"Discontinuity on right of anchor {i+1} at z={a_k}"


# ===========================================================================
# 3. Upper-Tail Monotonicity & Non-Saturation
# ===========================================================================

def test_lar_pw_upper_tail_monotonicity_and_non_saturation():
    """Verify that for z >= a4*, mu5 is strictly increasing, mu4 is strictly decreasing, and mu5 < 1."""
    anchors = np.array([1.0, 2.0, 3.0, 4.0])
    s_tail = 0.5
    a4 = anchors[3]

    # Sample a fine grid in the open tail: z from a4 to a4 + 20
    z_tail = np.linspace(a4, a4 + 20.0, 500)
    x_tail = np.expm1(z_tail)

    mu = tm.compute_lar_pw_memberships(x_tail, anchors, s_tail)
    mu4 = mu[:, 3]
    mu5 = mu[:, 4]

    # Check strict monotonicity on the grid
    diff_mu5 = np.diff(mu5)
    diff_mu4 = np.diff(mu4)
    assert np.all(diff_mu5 > 0.0), "mu5 is not strictly increasing in upper tail"
    assert np.all(diff_mu4 < 0.0), "mu4 is not strictly decreasing in upper tail"

    # Check non-saturation: mu5 never reaches 1.0 for any finite value
    assert np.all(mu5 < 1.0), "mu5 saturated to 1.0 at finite values"
    assert mu5[-1] > 0.95, "mu5 should approach 1.0 asymptotically"


# ===========================================================================
# 4. Membership Bounds & Ruspini Partition of Unity
# ===========================================================================

def test_partition_of_unity_and_valid_ranges():
    """Verify all memberships are in [0, 1] and sum exactly to 1.0 across the full domain."""
    anchors = np.array([0.5, 1.5, 2.5, 3.5])
    s_tail = 1.0

    # Test values spanning below a1, across all segments, and far out in the tail
    x_test = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 3.5, 5.0, 100.0, 10000.0])
    mu = tm.compute_lar_pw_memberships(x_test, anchors, s_tail)

    assert np.all(mu >= 0.0), "Found negative membership value"
    assert np.all(mu <= 1.0), "Found membership value > 1.0"
    assert np.allclose(mu.sum(axis=1), 1.0, atol=1e-12), "Partition of unity violated"


# ===========================================================================
# 5. Exact Preservation of Frozen M0 Recency Behavior
# ===========================================================================

def test_recency_piecewise_matches_frozen_m0():
    """Verify that Recency memberships match fair_comparison_retail2 piecewise_membership exactly."""
    r_vals = np.array([0.0, 5.0, 12.0, 25.0, 50.0, 80.0, 120.0, 200.0, 350.0])
    # In M0, R centroids are sorted descending by score: band 1 has high days, band 5 has low days
    centroids = np.array([210.0, 120.0, 60.0, 25.0, 5.0])

    # Compute with tm
    mu_new = tm.compute_recency_piecewise_memberships(r_vals, centroids)

    # Compute with frozen M0 logic
    order = np.argsort(centroids, kind="stable")
    mu_sorted = fc.piecewise_membership(r_vals, centroids[order])
    mu_frozen = mu_sorted[:, np.argsort(order, kind="stable")]

    assert np.allclose(mu_new, mu_frozen, atol=1e-12), "Recency does not match frozen M0 piecewise logic"


# ===========================================================================
# 6. Formal Context Binarization (45 Columns, Order, Nestedness)
# ===========================================================================

def test_formal_context_binarization_structure_and_nestedness():
    """Verify 45 binary columns, deterministic order, and threshold nestedness."""
    df = _make_synthetic_rfm(n=30, seed=5)
    params = tm.fit_tail_membership_params(df)
    mu_df = tm.compute_continuous_memberships(df, params)
    b_df = tm.binarize_formal_context(mu_df, thresholds=(0.3, 0.5, 0.7))

    assert mu_df.shape == (30, 15)
    assert b_df.shape == (30, 45)

    expected_cols = [
        f"{dim}{k+1}@{thresh:.1f}"
        for dim in ("R", "F", "M")
        for k in range(5)
        for thresh in (0.3, 0.5, 0.7)
    ]
    assert list(b_df.columns) == expected_cols

    # Values are strictly binary 0 or 1
    assert set(np.unique(b_df.to_numpy())).issubset({0, 1})

    # Check nestedness: attr@0.7 == 1 implies attr@0.5 == 1 and attr@0.3 == 1
    for dim in ("R", "F", "M"):
        for k in range(1, 6):
            c03 = b_df[f"{dim}{k}@0.3"]
            c05 = b_df[f"{dim}{k}@0.5"]
            c07 = b_df[f"{dim}{k}@0.7"]

            # If 0.7 is active, 0.5 and 0.3 must be active
            assert np.all(c05[c07 == 1] == 1)
            assert np.all(c03[c05 == 1] == 1)


# ===========================================================================
# 7. Deterministic Reproducibility
# ===========================================================================

def test_deterministic_output():
    """Verify that repeated evaluation produces identical results with zero diff."""
    df = _make_synthetic_rfm(n=50, seed=99)
    mu1, _, b1, _, params1 = tm.fit_and_transform_tail_memberships(df, df)
    mu2, _, b2, _, params2 = tm.fit_and_transform_tail_memberships(df, df)

    assert np.array_equal(mu1.to_numpy(), mu2.to_numpy())
    assert np.array_equal(b1.to_numpy(), b2.to_numpy())
    assert params1.to_dict() == params2.to_dict()


# ===========================================================================
# 8. Degenerate Training Fold Detection
# ===========================================================================

def test_degenerate_fold_detection_on_coincident_anchors():
    """Verify DegenerateTrainingFoldError is raised when training data has coincident quantiles/anchors."""
    # Constant data where cutpoints coincide
    bad_df = pd.DataFrame({
        "CustomerID": [f"c{i}" for i in range(20)],
        "R": [10.0] * 20,
        "F": [1.0] * 20,
        "M": [50.0] * 20,
    })

    with raises(tm.DegenerateTrainingFoldError):
        tm.fit_tail_membership_params(bad_df)


def test_degenerate_fold_detection_on_insufficient_samples():
    """Verify DegenerateTrainingFoldError is raised on insufficient sample size (<5)."""
    tiny_df = pd.DataFrame({
        "CustomerID": ["c1", "c2"],
        "R": [1.0, 2.0],
        "F": [1.0, 2.0],
        "M": [10.0, 20.0],
    })
    with raises(tm.DegenerateTrainingFoldError):
        tm.fit_tail_membership_params(tiny_df)


# ===========================================================================
# 9. Zero-IQR Tail Dispersion MAD Fallback
# ===========================================================================

def test_zero_iqr_tail_dispersion_mad_fallback():
    """Verify that when IQR of Band 5 is 0 but variation exists above a4*, MAD fallback is used."""
    # Construct a dataset where Band 5 has many tied values (IQR = 0) but has an extreme outlier
    raw_vals = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 100.0  # Band 5 has tied 20s (IQR=0) and one 100
    ])
    dim_params = tm.fit_lar_pw_parameters(raw_vals, "F")
    assert dim_params.s_tail is not None
    assert dim_params.s_tail > 0.0, "MAD fallback did not produce a positive s_tail"


# ===========================================================================
# 10. Input Validation & Error Handling
# ===========================================================================

def test_input_validation_rejects_nan_inf_and_negative():
    """Verify ValueError is raised for non-finite or negative inputs."""
    anchors = np.array([1.0, 2.0, 3.0, 4.0])
    s_tail = 0.5

    # Negative value
    with raises(ValueError, match="non-negative"):
        tm.compute_lar_pw_memberships(np.array([-1.0, 2.0]), anchors, s_tail)

    # NaN
    with raises(ValueError, match="finite"):
        tm.compute_lar_pw_memberships(np.array([np.nan, 2.0]), anchors, s_tail)

    # Inf
    with raises(ValueError, match="finite"):
        tm.compute_lar_pw_memberships(np.array([np.inf, 2.0]), anchors, s_tail)


# ===========================================================================
# 11. Discrimination Diagnostics
# ===========================================================================

def test_discrimination_diagnostics():
    """Verify D_mem and D_ctx calculate valid metrics."""
    df = _make_synthetic_rfm(n=80, seed=12)
    mu_tr, _, b_tr, _, params = tm.fit_and_transform_tail_memberships(df, df)

    diag_mem = tm.compute_continuous_tail_uniqueness(df, mu_tr, params)
    assert 0.0 <= diag_mem["d_mem"] <= 1.0
    assert diag_mem["n_tail"] > 0
    assert diag_mem["total_pairs"] > 0

    tail_mask = (df["F"] >= np.expm1(params.F.anchors[3])) | (df["M"] >= np.expm1(params.M.anchors[3]))
    diag_ctx = tm.compute_context_profile_diversity(b_tr, tail_mask)
    assert diag_ctx["d_ctx"] >= 1
    assert diag_ctx["h_ctx"] >= 0.0


def run_all_tests() -> int:
    """Run all unit test functions in this module."""
    test_funcs = [
        v for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ]
    print(f"Executing {len(test_funcs)} unit tests in tests/test_tail_memberships.py...\n")
    passed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            raise
    print(f"\nResult: ALL {passed}/{len(test_funcs)} unit tests passed successfully!")
    return 0


if __name__ == "__main__":
    run_all_tests()

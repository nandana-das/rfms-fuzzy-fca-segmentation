"""Tests for the non-FCA temporal baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from temporal_nonfca_baseline import nonfca_t_features  # noqa: E402


def _frame(n: int, seed: int, missing_t: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.gamma(2.0, 10.0, n)
    t[rng.random(n) < missing_t] = np.nan
    return pd.DataFrame({
        "CustomerID": [str(i) for i in range(n)],
        "R": rng.integers(0, 300, n).astype(float),
        "F": rng.integers(1, 40, n).astype(float),
        "M": rng.gamma(2.0, 100.0, n),
        "T": t,
        "repeat_history": (~np.isnan(t)).astype(int),
    })


def test_test_rows_do_not_affect_features():
    tr, te = _frame(400, 0), _frame(50, 1)
    te_shifted = te.assign(M=te["M"] * 1000, T=te["T"].fillna(0) * 1000)
    for arm in ("log_rfm_t", "spline_log_rfm_t"):
        a_tr, _ = nonfca_t_features(arm, tr, te)
        b_tr, _ = nonfca_t_features(arm, tr, te_shifted)
        np.testing.assert_array_equal(a_tr, b_tr)


def test_missing_t_imputed_with_training_median():
    tr, te = _frame(400, 0), _frame(50, 1, missing_t=1.0)
    x_tr, x_te = nonfca_t_features("log_rfm_t", tr, te)
    t_col = 3
    imputed_train = x_tr[tr["T"].isna().to_numpy(), t_col]
    assert np.allclose(x_te[:, t_col], imputed_train[0])


def test_shapes_and_finite_with_mass_point():
    tr, te = _frame(400, 0, missing_t=0.6), _frame(50, 1)
    for arm, width in (("log_rfm_t", 5), ("spline_log_rfm_t", 4 * 7 + 1)):
        x_tr, x_te = nonfca_t_features(arm, tr, te)
        assert x_tr.shape == (400, width) and x_te.shape == (50, width)
        assert np.isfinite(x_tr).all() and np.isfinite(x_te).all()

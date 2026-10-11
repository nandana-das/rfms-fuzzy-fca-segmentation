"""Versioned corrected LAR-PW membership implementation.

Historical M1-LAR-PW remains in :mod:`tail_memberships` and is intentionally
unchanged.  This module changes only the zero-IQR tail-scale fallback: it uses
the median absolute deviation from the fitted upper interior anchor ``a4``.
The fallback is a protocol clarification because the historical manifest does
not record which branch was used in each fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tail_memberships as historical

DimensionAnchors = historical.DimensionAnchors
TailMembershipParams = historical.TailMembershipParams
DegenerateTrainingFoldError = historical.DegenerateTrainingFoldError
DIMS = historical.DIMS
L_THRESHOLDS = historical.L_THRESHOLDS
STATUS_DEGENERATE = historical.STATUS_DEGENERATE


def fit_lar_pw_parameters_v2(raw_values: np.ndarray, dim_name: str) -> DimensionAnchors:
    """Fit LAR-PW parameters with the specified MAD fallback.

    The primary scale is the IQR of training band 5 in ``z = log1p(x)``.
    When that IQR is non-positive, the fallback is
    ``median(abs(z5 - a4))``.  Since valid band-5 observations are above the
    band-4 median in the intended construction, this is a robust scale of the
    upper-tail excess relative to the transition anchor.  A non-positive
    resulting scale is rejected rather than silently regularized.
    """
    x = historical.check_rfm_values(raw_values, dim_name)
    if len(x) < 5:
        raise DegenerateTrainingFoldError(f"Insufficient training observations for {dim_name}: n={len(x)}")

    cut = np.quantile(x, np.arange(1, 5) / 5.0)
    if np.any(np.diff(cut) <= 0):
        raise DegenerateTrainingFoldError(
            f"Quantile cutpoints for {dim_name} are not strictly increasing: {cut.tolist()}"
        )

    scores = (1 + np.searchsorted(cut, x, side="left")).astype(np.int8)
    z = np.log1p(x)
    anchors = []
    for k in range(1, 5):
        vals_z = z[scores == k]
        if len(vals_z) == 0:
            raise DegenerateTrainingFoldError(f"Empty training band {k} for {dim_name}")
        anchors.append(float(np.median(vals_z)))
    anchors_arr = np.asarray(anchors, dtype=float)
    if np.any(np.diff(anchors_arr) <= 0):
        raise DegenerateTrainingFoldError(
            f"Interior anchors for {dim_name} are not strictly increasing: {anchors_arr.tolist()}"
        )

    z5 = z[scores == 5]
    if len(z5) == 0:
        raise DegenerateTrainingFoldError(f"Empty training band 5 for {dim_name}")
    q75 = float(np.percentile(z5, 75.0))
    q25 = float(np.percentile(z5, 25.0))
    iqr = q75 - q25
    a4 = anchors_arr[3]
    if iqr > 0.0:
        s_tail = iqr
    else:
        s_tail = float(np.median(np.abs(z5 - a4))) if float(np.max(z5)) > a4 else 0.0

    if not np.isfinite(s_tail) or s_tail <= 0.0:
        raise DegenerateTrainingFoldError(
            f"Degenerate tail dispersion scale for {dim_name}: s_tail={s_tail} <= 0"
        )
    return DimensionAnchors(dimension=dim_name, cutoffs=cut, anchors=anchors_arr, s_tail=s_tail)


def fit_tail_membership_params_v2(tr_df: pd.DataFrame) -> TailMembershipParams:
    """Fit corrected M1-LAR-PW parameters strictly on a training frame."""
    for dim in DIMS:
        if dim not in tr_df.columns:
            raise KeyError(f"Training DataFrame missing required dimension: {dim}")
    return TailMembershipParams(
        R=historical.fit_recency_parameters(tr_df["R"].to_numpy()),
        F=fit_lar_pw_parameters_v2(tr_df["F"].to_numpy(), "F"),
        M=fit_lar_pw_parameters_v2(tr_df["M"].to_numpy(), "M"),
    )


def fit_and_transform_tail_memberships_v2(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame | None = None,
    thresholds: tuple[float, ...] = L_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, TailMembershipParams]:
    """Fit corrected parameters on train and project train/test identically."""
    params = fit_tail_membership_params_v2(tr_df)
    mu_tr = historical.compute_continuous_memberships(tr_df, params)
    b_tr = historical.binarize_formal_context(mu_tr, thresholds)
    if te_df is None:
        return mu_tr, None, b_tr, None, params
    mu_te = historical.compute_continuous_memberships(te_df, params)
    b_te = historical.binarize_formal_context(mu_te, thresholds)
    return mu_tr, mu_te, b_tr, b_te, params


# Re-export evaluation functions so a module swap preserves the mathematical
# projection and diagnostic code paths while making the fitting change explicit.
check_rfm_values = historical.check_rfm_values
compute_recency_piecewise_memberships = historical.compute_recency_piecewise_memberships
compute_lar_pw_memberships = historical.compute_lar_pw_memberships
compute_continuous_memberships = historical.compute_continuous_memberships
binarize_formal_context = historical.binarize_formal_context
compute_continuous_tail_uniqueness = historical.compute_continuous_tail_uniqueness
compute_context_profile_diversity = historical.compute_context_profile_diversity

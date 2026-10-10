"""Tail-sensitive fuzzy membership functions for RFM-FCA customer segmentation.

Stage 3 implementation of the preregistered Fuzzy RFM-FCA experiment:
- Recency (R): Frozen M0 piecewise-linear quintile memberships with outer shoulders.
- Frequency (F) and Monetary (M): Log-Coordinate Asymptotic Rational Piecewise (LAR-PW).
  - Interior transitions: piecewise-linear in log space between training medians a1* < a2* < a3* < a4*.
  - Upper tail (z >= a4*): non-saturating rational right shoulder mu5(z) = (z - a4*) / ((z - a4*) + s_tail),
    preserving continuous distinctions for all finite values.
- Training-only parameter estimation with strict degenerate-fold invalidation policy.
- Binarized formal context: 15 continuous attributes scaled at thresholds {0.3, 0.5, 0.7} -> 45 binary attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

DIMS: tuple[str, ...] = ("R", "F", "M")
L_THRESHOLDS: tuple[float, float, float] = (0.3, 0.5, 0.7)
STATUS_DEGENERATE: str = "DEGENERATE_TRAINING_FOLD"


class DegenerateTrainingFoldError(ValueError):
    """Raised when a training fold cannot construct a valid strictly increasing 5-level partition."""
    pass


@dataclass(frozen=True)
class DimensionAnchors:
    """Training-fitted anchors and parameters for one RFM dimension."""
    dimension: str
    cutoffs: np.ndarray            # 4 interior training cutpoints
    anchors: np.ndarray            # 5 centroids (R) or 4 interior log-anchors (F, M)
    s_tail: float | None = None    # tail dispersion scale for LAR-PW (F, M only)


@dataclass(frozen=True)
class TailMembershipParams:
    """Immutable collection of fitted parameters for R, F, and M."""
    R: DimensionAnchors
    F: DimensionAnchors
    M: DimensionAnchors

    def to_dict(self) -> dict[str, object]:
        return {
            "R": {
                "cutoffs": self.R.cutoffs.tolist(),
                "centroids": self.R.anchors.tolist(),
                "method": "frozen_piecewise_linear",
            },
            "F": {
                "cutoffs": self.F.cutoffs.tolist(),
                "anchors": self.F.anchors.tolist(),
                "s_tail": self.F.s_tail,
                "method": "lar_pw",
            },
            "M": {
                "cutoffs": self.M.cutoffs.tolist(),
                "anchors": self.M.anchors.tolist(),
                "s_tail": self.M.s_tail,
                "method": "lar_pw",
            },
        }


def check_rfm_values(values: np.ndarray | Sequence[float], dim_name: str = "RFM") -> np.ndarray:
    """Validate RFM input array: finite and non-negative.
    
    Unsupported inputs (NaN, infinite, negative) raise ValueError with informative messages.
    """
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{dim_name} values must be finite and contain no NaN or infinite values")
    if np.any(x < 0.0):
        raise ValueError(f"{dim_name} values must be non-negative; negative values are unsupported")
    return x


# ===========================================================================
# Recency: Frozen M0 Piecewise-Linear Memberships
# ===========================================================================

def fit_recency_parameters(r_train: np.ndarray) -> DimensionAnchors:
    """Fit Recency quintile cutpoints and band centroids strictly on training data."""
    x = check_rfm_values(r_train, "R")
    n = len(x)
    if n < 5:
        raise DegenerateTrainingFoldError(f"Insufficient training observations for R: n={n}")

    cut = np.quantile(x, np.arange(1, 5) / 5.0)
    if np.any(np.diff(cut) <= 0):
        raise DegenerateTrainingFoldError(f"Quantile cutpoints for R are not strictly increasing: {cut.tolist()}")

    # Score assignment: 1..5, inverted so Band 5 has lowest R (most recent purchases)
    scores = 6 - (1 + np.searchsorted(cut, x, side="left")).astype(np.int8)

    centroids = []
    fallback = float(np.median(x))
    for k in range(1, 6):
        vals = x[scores == k]
        if len(vals) == 0:
            raise DegenerateTrainingFoldError(f"Empty training quintile band {k} for R")
        centroids.append(float(np.median(vals)))

    centroids_arr = np.array(centroids, dtype=float)
    return DimensionAnchors(dimension="R", cutoffs=cut, anchors=centroids_arr)


def compute_recency_piecewise_memberships(
    raw_r: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute 5-level piecewise linear memberships for Recency matching frozen M0.
    
    Outer shoulders:
    - Band 5 (most recent, smallest raw days): mu5 = 1.0 for x <= centroid_5.
    - Band 1 (least recent, largest raw days): mu1 = 1.0 for x >= centroid_1.
    Interior levels: piecewise linear between adjacent centroids.
    """
    x = check_rfm_values(raw_r, "R")
    c = np.asarray(centroids, dtype=float)
    n = len(x)
    mu_sorted = np.zeros((n, 5), dtype=float)

    # Sort centroids in ascending raw-coordinate order
    order = np.argsort(c, kind="stable")
    cs = c[order]

    if cs[0] == cs[-1]:
        raise DegenerateTrainingFoldError("Recency centroids are fully coincident")

    below = x <= cs[0]
    above = x >= cs[-1]
    mu_sorted[below, 0] = 1.0
    mu_sorted[above, 4] = 1.0

    mid = ~below & ~above
    if np.any(mid):
        xm = x[mid]
        idx = np.clip(np.searchsorted(cs, xm, side="right") - 1, 0, 3)
        left, right = cs[idx], cs[idx + 1]
        denom = np.where(right - left == 0.0, 1e-9, right - left)
        frac_right = (xm - left) / denom
        rows = np.where(mid)[0]
        mu_sorted[rows, idx] = 1.0 - frac_right
        mu_sorted[rows, idx + 1] = frac_right

    # Restore to project band order (scores 1..5)
    mu = mu_sorted[:, np.argsort(order, kind="stable")]
    return mu


# ===========================================================================
# Frequency & Monetary: Log-Affine Rational Piecewise (LAR-PW)
# ===========================================================================

def fit_lar_pw_parameters(
    raw_values: np.ndarray,
    dim_name: str,
) -> DimensionAnchors:
    """Fit interior anchors a1* < a2* < a3* < a4* and tail scale s_tail strictly on training data.
    
    1. Log transform: z = log1p(x).
    2. Fit 4 interior quintile cutpoints on training customers.
    3. Calculate 4 interior anchors as medians of training bands 1..4 in log space.
    4. Calculate tail scale s_tail as IQR of training band 5 in log space.
       Fallback: if IQR <= 0 and max(z_band5) > a4*, use training Mean Absolute Deviation from a4*.
    5. Degenerate policy: if interior anchors are non-increasing or s_tail <= 0,
       raise DegenerateTrainingFoldError.
    """
    x = check_rfm_values(raw_values, dim_name)
    n = len(x)
    if n < 5:
        raise DegenerateTrainingFoldError(f"Insufficient training observations for {dim_name}: n={n}")

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

    anchors_arr = np.array(anchors, dtype=float)
    if np.any(np.diff(anchors_arr) <= 0):
        raise DegenerateTrainingFoldError(
            f"Interior anchors for {dim_name} are not strictly increasing: {anchors_arr.tolist()}"
        )

    # Tail dispersion scale: Band 5
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
        # Fallback for mass-point upper tail: training MAD from a4*
        if float(np.max(z5)) > a4:
            s_tail = float(np.mean(z5 - a4))
        else:
            s_tail = 0.0

    if s_tail <= 0.0:
        raise DegenerateTrainingFoldError(
            f"Degenerate tail dispersion scale for {dim_name}: s_tail={s_tail} <= 0"
        )

    return DimensionAnchors(
        dimension=dim_name,
        cutoffs=cut,
        anchors=anchors_arr,
        s_tail=s_tail,
    )


def compute_lar_pw_memberships(
    raw_values: np.ndarray,
    anchors: np.ndarray,
    s_tail: float,
    dim_name: str = "F",
) -> np.ndarray:
    """Evaluate closed-form LAR-PW memberships for Frequency or Monetary value.
    
    For z = log1p(x):
    - Left shoulder (z <= a1*): mu1 = 1.0, others 0.0.
    - Interior transitions (a_k* <= z <= a_{k+1}* for k in {1, 2, 3}):
        mu_k(z) = (a_{k+1}* - z) / (a_{k+1}* - a_k*)
        mu_{k+1}(z) = (z - a_k*) / (a_{k+1}* - a_k*)
    - Asymptotic open-tail (z >= a4*):
        mu5(z) = (z - a4*) / ((z - a4*) + s_tail)
        mu4(z) = s_tail / ((z - a4*) + s_tail) = 1.0 - mu5(z)
    """
    x = check_rfm_values(raw_values, dim_name)
    a = np.asarray(anchors, dtype=float)
    if len(a) != 4 or np.any(np.diff(a) <= 0):
        raise ValueError(f"LAR-PW requires exactly 4 strictly increasing interior anchors, got {a}")
    if s_tail <= 0.0:
        raise ValueError(f"LAR-PW requires positive tail scale s_tail, got {s_tail}")

    z = np.log1p(x)
    n = len(z)
    mu = np.zeros((n, 5), dtype=float)

    a1, a2, a3, a4 = a[0], a[1], a[2], a[3]

    # 1. Left shoulder: z <= a1
    left_mask = z <= a1
    mu[left_mask, 0] = 1.0

    # 2. Interior segment 1: a1 < z < a2
    seg1 = (z > a1) & (z < a2)
    if np.any(seg1):
        frac1 = (z[seg1] - a1) / (a2 - a1)
        mu[seg1, 0] = 1.0 - frac1
        mu[seg1, 1] = frac1

    # 3. Interior segment 2: a2 <= z < a3
    seg2 = (z >= a2) & (z < a3)
    if np.any(seg2):
        frac2 = (z[seg2] - a2) / (a3 - a2)
        mu[seg2, 1] = 1.0 - frac2
        mu[seg2, 2] = frac2

    # 4. Interior segment 3: a3 <= z < a4
    seg3 = (z >= a3) & (z < a4)
    if np.any(seg3):
        frac3 = (z[seg3] - a3) / (a4 - a3)
        mu[seg3, 2] = 1.0 - frac3
        mu[seg3, 3] = frac3

    # 5. Asymptotic open-tail: z >= a4
    tail_mask = z >= a4
    if np.any(tail_mask):
        dz = z[tail_mask] - a4
        denom = dz + s_tail
        mu5 = dz / denom
        mu4 = s_tail / denom
        mu[tail_mask, 3] = mu4
        mu[tail_mask, 4] = mu5

    return mu


# ===========================================================================
# Full RFM Membership & Formal Context Pipeline
# ===========================================================================

def fit_tail_membership_params(tr_df: pd.DataFrame) -> TailMembershipParams:
    """Fit all membership parameters strictly on training fold customers."""
    for dim in DIMS:
        if dim not in tr_df.columns:
            raise KeyError(f"Training DataFrame missing required dimension: {dim}")

    r_params = fit_recency_parameters(tr_df["R"].to_numpy())
    f_params = fit_lar_pw_parameters(tr_df["F"].to_numpy(), "F")
    m_params = fit_lar_pw_parameters(tr_df["M"].to_numpy(), "M")

    return TailMembershipParams(R=r_params, F=f_params, M=m_params)


def compute_continuous_memberships(
    df: pd.DataFrame,
    params: TailMembershipParams,
) -> pd.DataFrame:
    """Project customers into 15 continuous membership columns using frozen parameters."""
    for dim in DIMS:
        if dim not in df.columns:
            raise KeyError(f"Input DataFrame missing required dimension: {dim}")

    mu_r = compute_recency_piecewise_memberships(df["R"].to_numpy(), params.R.anchors)
    mu_f = compute_lar_pw_memberships(df["F"].to_numpy(), params.F.anchors, params.F.s_tail, "F")
    mu_m = compute_lar_pw_memberships(df["M"].to_numpy(), params.M.anchors, params.M.s_tail, "M")

    records: dict[str, np.ndarray] = {}
    for k in range(5):
        records[f"R{k+1}"] = mu_r[:, k]
    for k in range(5):
        records[f"F{k+1}"] = mu_f[:, k]
    for k in range(5):
        records[f"M{k+1}"] = mu_m[:, k]

    out_df = pd.DataFrame(records, index=df.index)
    return out_df


def binarize_formal_context(
    memberships_df: pd.DataFrame,
    thresholds: tuple[float, ...] = L_THRESHOLDS,
) -> pd.DataFrame:
    """Convert 15 continuous membership columns to 45 binary attributes via thresholds.
    
    Columns are deterministically ordered:
    R1@0.3, R1@0.5, R1@0.7, ..., M5@0.3, M5@0.5, M5@0.7
    """
    expected_order = [f"{dim}{k+1}" for dim in DIMS for k in range(5)]
    if list(memberships_df.columns) != expected_order:
        raise ValueError(f"Input memberships must have columns {expected_order}")

    cols: dict[str, np.ndarray] = {}
    for col in expected_order:
        vals = memberships_df[col].to_numpy(dtype=float)
        for thresh in thresholds:
            attr_name = f"{col}@{thresh:.1f}"
            cols[attr_name] = (vals >= thresh).astype(int)

    out_df = pd.DataFrame(cols, index=memberships_df.index)
    return out_df


def fit_and_transform_tail_memberships(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame | None = None,
    thresholds: tuple[float, ...] = L_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, TailMembershipParams]:
    """Fit on training fold, then project both train and test into memberships and contexts."""
    params = fit_tail_membership_params(tr_df)
    mu_tr = compute_continuous_memberships(tr_df, params)
    b_tr = binarize_formal_context(mu_tr, thresholds)

    if te_df is not None:
        mu_te = compute_continuous_memberships(te_df, params)
        b_te = binarize_formal_context(mu_te, thresholds)
    else:
        mu_te = None
        b_te = None

    return mu_tr, mu_te, b_tr, b_te, params


# ===========================================================================
# Discrimination Diagnostics
# ===========================================================================

def compute_continuous_tail_uniqueness(
    raw_df: pd.DataFrame,
    mu_df: pd.DataFrame,
    params: TailMembershipParams,
) -> dict[str, float | int]:
    """Compute empirical uniqueness diagnostic D_mem on upper-tail customers.
    
    Tail customers: F >= exp(a4_F) - 1 or M >= exp(a4_M) - 1.
    Returns D_mem, N_tail, tied_customer_pairs, and total_pairs.
    """
    f_tail_cut = np.expm1(params.F.anchors[3])
    m_tail_cut = np.expm1(params.M.anchors[3])

    is_tail = (raw_df["F"] >= f_tail_cut) | (raw_df["M"] >= m_tail_cut)
    tail_mu = mu_df[is_tail].to_numpy(dtype=float)
    tail_raw = raw_df.loc[is_tail, list(DIMS)].to_numpy(dtype=float)

    n_tail = len(tail_mu)
    if n_tail < 2:
        return {
            "n_tail": n_tail,
            "d_mem": 1.0,
            "tied_raw_pairs": 0,
            "total_pairs": 0,
        }

    total_pairs = n_tail * (n_tail - 1) // 2

    # Vector difference checks
    diff_mu_count = 0
    tied_raw_count = 0
    for i in range(n_tail):
        for j in range(i + 1, n_tail):
            if np.array_equal(tail_raw[i], tail_raw[j]):
                tied_raw_count += 1
            if not np.array_equal(tail_mu[i], tail_mu[j]):
                diff_mu_count += 1

    d_mem = diff_mu_count / total_pairs
    return {
        "n_tail": n_tail,
        "d_mem": float(d_mem),
        "tied_raw_pairs": int(tied_raw_count),
        "total_pairs": int(total_pairs),
    }


def compute_context_profile_diversity(
    b_df: pd.DataFrame,
    tail_mask: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Compute D_ctx (distinct binary profiles) and entropy H_ctx in the tail."""
    sub = b_df[tail_mask]
    n_tail = len(sub)
    if n_tail == 0:
        return {"d_ctx": 0, "h_ctx": 0.0, "n_tail": 0}

    profiles = [tuple(row) for row in sub.itertuples(index=False, name=None)]
    unique_profiles = set(profiles)
    d_ctx = len(unique_profiles)

    # Entropy
    counts = pd.Series(profiles).value_counts().to_numpy()
    probs = counts / n_tail
    h_ctx = float(-np.sum(probs * np.log2(probs)))

    return {
        "n_tail": n_tail,
        "d_ctx": int(d_ctx),
        "h_ctx": float(h_ctx),
    }

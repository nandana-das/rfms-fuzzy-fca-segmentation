"""Ablation ladder with rolling-origin temporal evaluation for Fuzzy RFM-FCA.

Stage 4 of the preregistered experiment: Factorial ablation of tail-sensitive
memberships (LAR-PW) and concept t-norm aggregation (Product vs Gödel-min).

Factorial arms:
- M0: Frozen baseline fuzzy RFM-FCA (frozen piecewise-linear memberships + Gödel-min aggregation)
- M1: LAR-PW Frequency/Monetary memberships + Gödel-min aggregation
- M2: Frozen M0 memberships + Algebraic Product t-norm aggregation (over frozen M0 concepts)
- M3: LAR-PW memberships + Algebraic Product t-norm aggregation (over M1 concepts)

Evaluation protocol:
- Stratified 5-fold CV over customers within each rolling origin.
- All transformations, anchors, quintile cutpoints, concept mining, and Jaccard suppression
  are strictly fitted on training folds only.
- Test customer predictions are generated strictly out-of-fold using frozen training parameters.
- Downstream models:
  - LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", solver="lbfgs", max_iter=2000, random_state=seed)
  - RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5) on log1p(future_spend) and log1p(future_invoices)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_fuzzy_memberships,
    extract_base_bands_from_intent,
    fit_quantile_cutoffs,
    quantile_scores,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    mine_fuzzy_closed_concepts_with_thresholds,
)
import tail_memberships as tm  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Protocol Constants (Preregistered; locked to baseline)
# ---------------------------------------------------------------------------
DIMS = ("R", "F", "M")
HORIZON_DAYS = 91
N_FOLDS = 5
CV_SEED = 0

MIN_SUPPORT = 0.04
L_THRESHOLDS = (0.3, 0.5, 0.7)
J_MAX = 0.80
MU_CUT = 0.50

LOGREG_CS = 10
LOGREG_MAX_ITER = 2000
RIDGE_ALPHAS = np.logspace(-3, 3, 20)

ARMS = ["M0", "M1", "M2", "M3"]
METRICS = ["auc", "spend_r2", "invoice_r2"]

ARM_CONFIGS = {
    "M0": {
        "membership": "m0_piecewise_linear",
        "tnorm": "godel_min",
        "description": "Frozen baseline fuzzy RFM-FCA (piecewise-linear + Gödel-min)",
    },
    "M1": {
        "membership": "lar_pw",
        "tnorm": "godel_min",
        "description": "Tail-sensitive LAR-PW (F, M) + frozen R + Gödel-min",
    },
    "M2": {
        "membership": "m0_piecewise_linear",
        "tnorm": "product",
        "description": "Frozen M0 memberships + Algebraic Product t-norm",
    },
    "M3": {
        "membership": "lar_pw",
        "tnorm": "product",
        "description": "Tail-sensitive LAR-PW + Algebraic Product t-norm",
    },
}


# ===========================================================================
# Concept Membership Computation (Gödel-min vs Algebraic Product)
# ===========================================================================

def compute_customer_concept_memberships(
    concepts_df: pd.DataFrame,
    band_memberships: pd.DataFrame,
    tnorm: str = "godel_min",
) -> np.ndarray:
    """Compute continuous customer-to-concept membership matrix.

    Formula:
    - Gödel minimum t-norm: mu_C(i) = min_{b in B(I_C)} mu_b(i)
    - Algebraic Product t-norm: mu_C(i) = prod_{b in B(I_C)} mu_b(i)

    where B(I_C) is the set of unique base score bands present in the intent I_C
    (e.g., intent 'R5@0.3 & R5@0.5 & M4@0.3' has base bands ['R5', 'M4']).
    Empty intents evaluate to 1.0 (identity element of both t-norms).
    Singleton base bands evaluate to mu_b(i) under both t-norms.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        DataFrame with an 'intent' column.
    band_memberships : pd.DataFrame
        Continuous band memberships (R1..R5, F1..F5, M1..M5).
    tnorm : str
        Either 'godel_min' or 'product'.

    Returns
    -------
    np.ndarray
        Matrix of shape (N_customers, N_concepts).
    """
    n = len(band_memberships)
    k = len(concepts_df)
    mu_matrix = np.zeros((n, k), dtype=float)

    if k == 0:
        return mu_matrix

    for c_idx, row in concepts_df.iterrows():
        intent = row["intent"]
        base_bands = extract_base_bands_from_intent(intent)
        if not base_bands:
            mu_matrix[:, c_idx] = 1.0
        elif len(base_bands) == 1:
            band = base_bands[0]
            if band in band_memberships.columns:
                mu_matrix[:, c_idx] = band_memberships[band].to_numpy()
        else:
            cols = [
                band_memberships[b].to_numpy()
                for b in base_bands
                if b in band_memberships.columns
            ]
            if cols:
                sub_matrix = np.column_stack(cols)
                if tnorm == "godel_min":
                    mu_matrix[:, c_idx] = np.min(sub_matrix, axis=1)
                elif tnorm == "product":
                    mu_matrix[:, c_idx] = np.prod(sub_matrix, axis=1)
                else:
                    raise ValueError(f"Unsupported t-norm: {tnorm}")

    return mu_matrix


# ===========================================================================
# Representations & Concept Mining per Fold
# ===========================================================================

def _concept_info(concepts: pd.DataFrame, X_tr: np.ndarray, n_candidates: int) -> dict[str, int]:
    distinct = int(np.unique(np.round(X_tr, 12), axis=1).shape[1]) if X_tr.shape[1] else 0
    empty_core = int(((X_tr >= MU_CUT).sum(axis=0) == 0).sum()) if X_tr.shape[1] else 0
    return {
        "n_candidates": n_candidates,
        "n_features": int(X_tr.shape[1]),
        "n_distinct_columns": distinct,
        "n_empty_core": empty_core,
    }


def build_fold_representations(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    arms: Sequence[str] = ARMS,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    """Build train and test feature matrices for all requested arms in a CV fold.

    Folds are fitted strictly on tr. Concepts and memberships are shared within
    factorial factor pairs:
    - M0 and M2 share M0 concept mining.
    - M1 and M3 share M1 concept mining (LAR-PW).

    Returns
    -------
    features : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping arm -> (X_tr, X_te).
    diag_rows : list[dict[str, object]]
        Per-arm diagnostic information for the fold.
    """
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diag_rows: list[dict[str, object]] = []

    need_m0_concepts = any(a in ("M0", "M2", "fuzzy_rfm_fca") for a in arms)
    need_m1_concepts = any(a in ("M1", "M3") for a in arms)

    # -----------------------------------------------------------------------
    # 1. Baseline M0 Pipeline (Piecewise-linear quintiles)
    # -----------------------------------------------------------------------
    if need_m0_concepts:
        scored_tr, _ = bl._quintile_scored(tr)
        mu_tr_m0, centroids_m0, _ = compute_fuzzy_memberships(scored_tr, dims=DIMS)
        mu_te_m0, _, _ = compute_fuzzy_memberships(
            te[["CustomerID", *DIMS]].copy(), trained_centroids=centroids_m0, dims=DIMS
        )

        raw_m0, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
            mu_tr_m0, scored_tr, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
        )
        raw_m0 = raw_m0.reset_index(drop=True)
        mu_all_m0 = compute_customer_concept_memberships(raw_m0, mu_tr_m0, tnorm="godel_min")
        kept_m0 = suppress_redundant_concepts(
            raw_m0, mu_all_m0, j_max=J_MAX, mu_cut=MU_CUT, drop_empty_core=True
        ).reset_index(drop=True)

        if "M0" in arms or "fuzzy_rfm_fca" in arms:
            X_tr_m0 = compute_customer_concept_memberships(kept_m0, mu_tr_m0, tnorm="godel_min")
            X_te_m0 = compute_customer_concept_memberships(kept_m0, mu_te_m0, tnorm="godel_min")
            arm_key = "M0" if "M0" in arms else "fuzzy_rfm_fca"
            features[arm_key] = (X_tr_m0, X_te_m0)
            diag_rows.append({
                "arm": arm_key,
                "membership_type": "m0_piecewise_linear",
                "tnorm": "godel_min",
                "membership_dims": 15,
                "context_dims": 45,
                "degenerate_error": None,
                **_concept_info(kept_m0, X_tr_m0, len(raw_m0)),
            })

        if "M2" in arms:
            X_tr_m2 = compute_customer_concept_memberships(kept_m0, mu_tr_m0, tnorm="product")
            X_te_m2 = compute_customer_concept_memberships(kept_m0, mu_te_m0, tnorm="product")
            features["M2"] = (X_tr_m2, X_te_m2)
            diag_rows.append({
                "arm": "M2",
                "membership_type": "m0_piecewise_linear",
                "tnorm": "product",
                "membership_dims": 15,
                "context_dims": 45,
                "degenerate_error": None,
                **_concept_info(kept_m0, X_tr_m2, len(raw_m0)),
            })

    # -----------------------------------------------------------------------
    # 2. LAR-PW Pipeline (Tail-sensitive memberships)
    # -----------------------------------------------------------------------
    if need_m1_concepts:
        degenerate_error_msg = None
        try:
            mu_tr_m1, mu_te_m1, b_tr_m1, b_te_m1, params_m1 = tm.fit_and_transform_tail_memberships(
                tr, te, thresholds=L_THRESHOLDS
            )
        except tm.DegenerateTrainingFoldError as exc:
            degenerate_error_msg = str(exc)
            # Re-raise to prevent silent corruption, while recording diagnostic
            raise tm.DegenerateTrainingFoldError(f"Degenerate fold in LAR-PW fitting: {exc}") from exc

        # Use training quintile scores strictly for stability proxy calculation
        scored_tr, _ = bl._quintile_scored(tr)
        raw_m1, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
            mu_tr_m1, scored_tr, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
        )
        raw_m1 = raw_m1.reset_index(drop=True)
        mu_all_m1 = compute_customer_concept_memberships(raw_m1, mu_tr_m1, tnorm="godel_min")
        kept_m1 = suppress_redundant_concepts(
            raw_m1, mu_all_m1, j_max=J_MAX, mu_cut=MU_CUT, drop_empty_core=True
        ).reset_index(drop=True)

        if "M1" in arms:
            X_tr_m1 = compute_customer_concept_memberships(kept_m1, mu_tr_m1, tnorm="godel_min")
            X_te_m1 = compute_customer_concept_memberships(kept_m1, mu_te_m1, tnorm="godel_min")
            features["M1"] = (X_tr_m1, X_te_m1)
            diag_rows.append({
                "arm": "M1",
                "membership_type": "lar_pw",
                "tnorm": "godel_min",
                "membership_dims": 15,
                "context_dims": 45,
                "degenerate_error": degenerate_error_msg,
                **_concept_info(kept_m1, X_tr_m1, len(raw_m1)),
            })

        if "M3" in arms:
            X_tr_m3 = compute_customer_concept_memberships(kept_m1, mu_tr_m1, tnorm="product")
            X_te_m3 = compute_customer_concept_memberships(kept_m1, mu_te_m1, tnorm="product")
            features["M3"] = (X_tr_m3, X_te_m3)
            diag_rows.append({
                "arm": "M3",
                "membership_type": "lar_pw",
                "tnorm": "product",
                "membership_dims": 15,
                "context_dims": 45,
                "degenerate_error": degenerate_error_msg,
                **_concept_info(kept_m1, X_tr_m3, len(raw_m1)),
            })

    return features, diag_rows


# ===========================================================================
# Downstream Models (Frozen regularized linear models)
# ===========================================================================

def fit_predict(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    tr: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit downstream models on training fold and return test predictions.

    - Repurchase: LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", solver="lbfgs")
    - Spend: RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5) on log1p(future_spend)
    - Invoices: RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5) on log1p(future_invoices)
    """
    y_rep = tr["repurchased"].to_numpy()
    y_sp = np.log1p(tr["future_spend"].to_numpy())
    y_inv = np.log1p(tr["future_invoices"].to_numpy())

    clf = LogisticRegressionCV(
        Cs=LOGREG_CS,
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    ).fit(X_tr, y_rep)

    r1 = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_sp)
    r2 = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_inv)

    prob = clf.predict_proba(X_te)[:, 1]
    sp = r1.predict(X_te)
    inv = r2.predict(X_te)

    return prob, sp, inv


# ===========================================================================
# Temporal Origin Runner
# ===========================================================================

def run_origin(
    cohort: dict[str, object],
    arms: Sequence[str] = ARMS,
    cv_seed: int = CV_SEED,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    """Execute stratified 5-fold CV for all arms on a single temporal origin.

    Returns
    -------
    oof : dict[str, dict[str, np.ndarray]]
        Out-of-fold predictions {'prob', 'sp', 'inv'} per arm.
    info_rows : list[dict[str, object]]
        Fold-level diagnostic records.
    """
    df = cohort["df"]
    n = len(df)
    oof = {a: {m: np.full(n, np.nan) for m in ("prob", "sp", "inv")} for a in arms}
    info_rows: list[dict[str, object]] = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)

    for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
        tr = df.iloc[i_tr].reset_index(drop=True)
        te = df.iloc[i_te].reset_index(drop=True)

        features, diag_rows = build_fold_representations(tr, te, arms=arms)

        for d in diag_rows:
            info_rows.append({
                "dataset": cohort["dataset"],
                "origin": cohort["origin"],
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                **d,
            })

        for arm in arms:
            X_tr, X_te = features[arm]
            p, s, v = fit_predict(X_tr, X_te, tr, seed=cv_seed + fold)
            oof[arm]["prob"][i_te] = p
            oof[arm]["sp"][i_te] = s
            oof[arm]["inv"][i_te] = v

    return oof, info_rows


# ===========================================================================
# Metrics
# ===========================================================================

def _auc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise ROC AUC for (B, n) matrices via rank-sum identity."""
    r = rankdata(p, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    return ((r * y).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _r2_rows(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Row-wise R^2 for (B, n) matrices."""
    ss_res = ((y - f) ** 2).sum(axis=1)
    ss_tot = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - ss_res / ss_tot


def metric_rows(
    df: pd.DataFrame,
    pred: dict[str, np.ndarray],
    idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute point or bootstrap metrics on pooled out-of-fold predictions."""
    y_rep = df["repurchased"].to_numpy()[idx]
    y_sp = np.log1p(df["future_spend"].to_numpy())[idx]
    y_inv = np.log1p(df["future_invoices"].to_numpy())[idx]
    return {
        "auc": _auc_rows(y_rep, pred["prob"][idx]),
        "spend_r2": _r2_rows(y_sp, pred["sp"][idx]),
        "invoice_r2": _r2_rows(y_inv, pred["inv"][idx]),
    }

"""Redundancy suppression for fuzzy formal concepts (adopted improvement 1).

Greedy extent-Jaccard redundancy suppression, evaluated on Online Retail II
(2026-09-26, scripts/fuzzy_improvements_retail2.py): compresses the 445-concept
baseline lattice to 95 concepts with no predictive loss (spend R2 0.3664 vs
0.3647, invoices R2 0.4952 vs 0.4933; test AUC 0.7875 vs 0.7915 on the same
fixed 70/30 temporal holdout, paired bootstrap CIs excluding zero for both R2
deltas vs crisp). Near-duplicate concept pairs (extent Jaccard >= 0.8) drop to
0% and mean concepts-per-customer at mu >= 0.5 falls from 41.3 to 4.1.

Leakage contract: pass a membership matrix computed from TRAINING customers
only when used inside a holdout protocol - the selection must not see test
extents. The suppressor never looks at outcomes; it is purely structural, but
the extent cut (mu >= 0.5) must still be fit-free w.r.t. the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

J_MAX_DEFAULT = 0.8


def suppress_redundant_concepts(
    concepts_df: pd.DataFrame,
    mu_matrix: np.ndarray,
    j_max: float = J_MAX_DEFAULT,
    mu_cut: float = 0.5,
) -> pd.DataFrame:
    """Greedily keep concepts in quality order; drop any whose >=mu_cut-cut extent
    overlaps an already-kept concept at Jaccard >= j_max.

    Parameters
    ----------
    concepts_df:
        Concept table with columns ``intent``, ``intent_size``, ``support`` and
        ``stability_proxy`` (project-standard schema).
    mu_matrix:
        (n_customers, n_concepts) Gödel-min membership matrix aligned row-wise with
        ``concepts_df``. MUST be computed from training customers only when the
        caller is inside a train/test protocol.
    j_max:
        Extent-Jaccard threshold above which a later (lower-quality) concept is
        considered redundant.
    mu_cut:
        Membership cut used to binarize extents.

    Returns
    -------
    pd.DataFrame
        The subset of ``concepts_df`` that survived suppression, in original row
        order, reset index. The universal root (intent_size == 0) is never kept.

    Notes
    -----
    Quality order: support desc, intent_size asc, stability_proxy desc, with the
    original row order as deterministic final tie-break. Complexity is
    O(K_kept * K) extent comparisons; at Retail II scale (445 concepts, 3,018
    customers) this runs in well under a second.
    """
    # NOTE: `mu_matrix` must come from TRAINING customers only, so the redundancy
    # selection stays leak-free under the frozen-structure holdout protocol.
    nontrivial_idx = concepts_df.index[concepts_df["intent_size"] > 0].to_numpy()
    if len(nontrivial_idx) == 0:
        return concepts_df.copy()

    order = (
        concepts_df.loc[nontrivial_idx]
        .sort_values(
            ["support", "intent_size", "stability_proxy"],
            ascending=[False, True, False],
            kind="mergesort",
        )
        .index.to_numpy()
    )

    bin_extents = mu_matrix >= mu_cut
    kept: list[int] = []
    kept_extents: list[np.ndarray] = []

    for idx in order:
        ext = bin_extents[:, idx]
        redundant = False
        for kept_ext in kept_extents:
            inter = np.logical_and(ext, kept_ext).sum()
            if inter == 0:
                continue
            union = np.logical_or(ext, kept_ext).sum()
            jacc = inter / union if union > 0 else 0.0
            if jacc >= j_max:
                redundant = True
                break
        if not redundant:
            kept.append(idx)
            kept_extents.append(ext)

    kept_set = set(kept)
    keep_mask = np.array([i in kept_set for i in range(len(concepts_df))])
    return concepts_df.loc[keep_mask].reset_index(drop=True)


def suppress_redundant_concepts_sparse(
    concepts_df: pd.DataFrame,
    mu_matrix: np.ndarray,
    j_max: float = J_MAX_DEFAULT,
    mu_cut: float = 0.5,
) -> pd.DataFrame:
    """Numpy-backed suppression for large populations (e.g. Olist n=93,357).

    Same greedy selection as ``suppress_redundant_concepts`` but with vectorized
    extent ops: incremental kept-extent matrix + max-intersection short-circuit.
    Tie-break is max support only (stability_proxy column may be absent for
    crisp-mined tables).

    Leakage contract: ``mu_matrix`` must come from TRAINING customers only.
    """
    keep_mask0 = (concepts_df["intent_size"] > 0).to_numpy()
    if not keep_mask0.any():
        return concepts_df.copy()

    supports = concepts_df["support"].to_numpy()
    order = np.flatnonzero(keep_mask0)
    order = order[np.lexsort((order, -supports[order]))]

    bin_ext = (mu_matrix >= mu_cut).T  # shape: (n_concepts, n_customers)
    kept: list[int] = []
    kept_mat = np.empty((0, mu_matrix.shape[0]), dtype=bool)
    for idx in order:
        ext = bin_ext[idx]
        if kept_mat.shape[0]:
            inter = kept_mat @ ext  # intersection sizes vs every kept concept
            cand = np.flatnonzero(inter > 0)
            redundant = False
            for row_i in cand:
                union = kept_mat[row_i].sum() + ext.sum() - inter[row_i]
                if union > 0 and inter[row_i] / union >= j_max:
                    redundant = True
                    break
            if redundant:
                continue
        kept.append(int(idx))
        kept_mat = np.vstack([kept_mat, ext])

    keep_mask = np.zeros(len(concepts_df), dtype=bool)
    keep_mask[kept] = True
    return concepts_df.loc[keep_mask].reset_index(drop=True)

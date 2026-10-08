#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-2 step 2: inspect training-only stability-loss distribution on seed-42 train.

This is the critical prerequisite before defining the kuz_loss search range.
We must see the actual exact canonical Kuznetsov stability values to set a
logarithmically-spaced loss threshold grid that covers the observed distribution.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy_membership_sensitivity import load_and_prepare_cohorts, DIMS, INVERT_DIMS, mine_fuzzy_closed_concepts_with_thresholds, _band_centroids_generic, _piecewise_membership_generic
from fair_comparison_retail2 import (
    dense_rank_scores,
    extract_dense_rank_cutoffs,
    apply_dense_rank_cutoffs,
    compute_fuzzy_memberships,
    mine_crisp_closed_concepts,
    compute_customer_concept_memberships,
    mine_fuzzy_closed_concepts,
)
from concept_redundancy import suppress_redundant_concepts
from kuznetsov_pruning_leakage_free import compute_split_stability, J_MAX, MU_CUT, MIN_SUPPORT, L_THRESHOLDS

RESULTS_DIR = ROOT_DIR.parent / "results"  # ROOT_DIR is scripts/ here (sys.path)
EXP_DIR = RESULTS_DIR / "optimized_fuzzy_fca" / "stage2"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def _pr(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    _pr("=" * 90)
    _pr("STAGE-2 STEP 2: TRAINING-ONLY KUZNETSOV STABILITY-LOSS DISTRIBUTION (SEED 42)")
    _pr("=" * 90)

    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"Cohort: {len(df):,}, repurchase rate={df['repurchased'].mean():.4f}")

    _pr("\n[1] Outer split seed=42 (70/30 stratified)...")
    train_outer, _ = train_test_split(
        df, test_size=0.30, stratify=df["repurchased"].to_numpy(), random_state=42
    )
    train_outer = train_outer.reset_index(drop=True)
    _pr(f"  train size: {len(train_outer):,}")

    train_cust = train_outer[["CustomerID", "R", "F", "M"]].copy()
    train_scored = dense_rank_scores(train_cust, dims=DIMS)
    train_fuzzy_mu, train_centroids, _ = compute_fuzzy_memberships(train_scored, dims=DIMS)

    _pr("\n[2] Mine raw fuzzy concepts at baseline min_support=0.04...")
    t0 = time.time()
    raw_concepts, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        train_fuzzy_mu, train_scored, L_THRESHOLDS, min_support=MIN_SUPPORT, dims=DIMS
    )
    raw_concepts = raw_concepts.reset_index(drop=True)
    _pr(f"  candidates={len(raw_concepts)}  (mined in {time.time()-t0:.1f}s)")

    _pr("\n[3] Compute EXACT canonical Kuznetsov stability (training context only)...")
    t0 = time.time()
    stab_df, ctx_info = compute_split_stability(raw_concepts, train_fuzzy_mu, L_THRESHOLDS)
    _pr(f"  stability computed in {time.time()-t0:.1f}s")
    _pr(f"  ctx: objects={ctx_info['n_objects']} attrs={ctx_info['n_attributes']}")
    _pr(f"  exact: {ctx_info['n_exact']}  failed: {ctx_info['n_failed']}  infeasible: {ctx_info.get('n_infeasible',0)}")

    concepts = raw_concepts.copy()
    concepts["concept_index"] = np.arange(len(concepts))
    concepts = concepts.merge(
        stab_df[
            [
                "concept_index",
                "stab_float",
                "log2_loss",
                "status",
                "extent_size",
                "n_lower_neighbors",
                "generator_count",
                "stability_fraction",
                "a_prime_equals_b",
                "b_prime_equals_a",
            ]
        ],
        on="concept_index",
        how="left",
    )

    n_exact = int((stab_df["status"].str.startswith("exact")).sum())
    n_failed = int((stab_df["status"].str.startswith("failed")).sum())
    n_with = int(concepts["stab_float"].notna().sum())
    _pr(f"\n  concepts with stability: {n_with}/{len(concepts)}")
    _pr(f"  exact: {n_exact}  failed: {n_failed}")

    # Loss = 1 - stability  (only for exact concepts)
    exact = concepts[concepts["stab_float"].notna()].copy()
    exact["loss"] = 1.0 - exact["stab_float"]
    exact["log2_loss"] = exact["log2_loss"]

    _pr("\n[4] Stability-loss distribution (1 - stability) on EXACT concepts only:")
    loss = exact["loss"].to_numpy()
    log2_loss = exact["log2_loss"].to_numpy()
    _pr(f"  n_exact_with_loss={len(loss)}")
    _pr(f"  loss: min={loss.min():.6e}  max={loss.max():.6e}  mean={loss.mean():.6e}")
    _pr(f"  log2_loss: min={log2_loss.min():.4f}  max={log2_loss.max():.4f}  mean={log2_loss.mean():.4f}")
    if len(loss) > 0:
        vz = np.sort(np.log10(np.maximum(loss, 1e-300)))
        pct = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
        _pr("  percentile(log10(loss)):")
        for p in pct:
            _pr(f"    p{int(p*100):>3d}: {np.percentile(vz, p):+.4f}  (loss={10**np.percentile(vz, p):.6e})")

    _pr("\n[5] Loss-threshold concept counts (training only, min_support=0.04, jm=0.80, mc=0.50):")
    thresholds = [0.0, 1e-300, 1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 0.062, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for thr in thresholds:
        if thr == 0.0:
            after = exact[exact["stab_float"] > 0].copy()
        else:
            after = exact[(exact["stab_float"] > 0) & ((1.0 - exact["stab_float"]) <= thr)].copy()
        mu_tr = compute_customer_concept_memberships(after.reset_index(drop=True), train_fuzzy_mu)
        suppressed = suppress_redundant_concepts(after.reset_index(drop=True), mu_tr, j_max=J_MAX, mu_cut=MU_CUT)
        rows.append({
            "kuz_loss_threshold": thr,
            "after_stability": len(after),
            "final_after_jaccard": len(suppressed),
        })
        _pr(f"  loss<={thr:<12} -> after_stab={len(after):>4}  final(J_max=0.80, mu_cut=0.50)={len(suppressed):>4}")

    loss_df = pd.DataFrame(rows)
    loss_df.to_csv(EXP_DIR / "seed42_train_stability_loss_distribution.csv", index=False)
    _pr(f"\nSaved: {EXP_DIR / 'seed42_train_stability_loss_distribution.csv'}")

    _pr("\n[6] Observation:")
    _pr(f"  loss values span many orders of magnitude (log10 from {vz.min():+.2f} to {vz.max():+.2f}).")
    _pr(f"  A logarithmic grid for kuz_loss is warranted.")
    _pr(f"  The snapshot uses: {thresholds}")
    _pr(f"  Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

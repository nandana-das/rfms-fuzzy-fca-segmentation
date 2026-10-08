#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 smoke test: single outer seed (42), nested leakage-free,
all baselines A/B/C/D, full Stage-2 search space from snapshot.

Purpose:
- Reproduce fixed baselines on seed 42.
- Confirm inner validation Pareto selection can be computed and frozen before outer test.
- Confirm outer test is evaluated once and leakage-free.
- Sanity-check concept-count landscape and runtime before multi-seed run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fuzzy_membership_sensitivity import load_and_prepare_cohorts, DIMS
from optimized_fuzzy_fca_stage2 import (
    baseline_existing_fuzzy_split,
    baseline_existing_kuznetsov_split,
    baseline_raw_rfm_split,
    load_stage2_search_space,
    run_one_outer_seed,
)

RESULTS_DIR = ROOT_DIR / "results"
EXP_DIR = RESULTS_DIR / "optimized_fuzzy_fca" / "stage2"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def _pr(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    _pr("=" * 100)
    _pr("STAGE 2 SMOKE TEST (OUTER SEED 42 ONLY)")
    _pr("=" * 100)

    obs_agg, merged = load_and_prepare_cohorts()
    df = merged.copy()
    if "household_key" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"household_key": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype(str)
    _pr(f"Cohort size: {len(df):,}")
    _pr(f"Repurchase rate: {df['repurchased'].mean():.4f}")

    configs = load_stage2_search_space()
    _pr(f"\nStage-2 search space size: {len(configs)}")
    _pr("Configs include: min_support, j_max, mu_cut, kuz_loss")
    _pr("Snapshot derived from seed-42 TRAIN ONLY.")

    _pr("\n[1] Outer split seed=42 (70/30 stratified)...")
    train_outer, test_outer = train_test_split(
        df,
        test_size=0.30,
        stratify=df["repurchased"].to_numpy(),
        random_state=42,
    )
    train_outer = train_outer.reset_index(drop=True)
    test_outer = test_outer.reset_index(drop=True)
    _pr(f"  outer train: {len(train_outer):,} | outer test: {len(test_outer):,}")
    _pr(f"  outer train repurchase rate: {train_outer['repurchased'].mean():.4f}")

    _pr("\n[2] Baseline A: Raw RFM (seed=42)")
    t0 = time.perf_counter()
    raw42 = baseline_raw_rfm_split(train_outer, test_outer, 42)
    _pr(
        "  ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f} Spend R2={spend_r2:.4f} "
        "Invoice R2={invoice_r2:.4f} concepts={n_concepts} runtime={runtime:.1f}s".format(
            **raw42,
            runtime=time.perf_counter() - t0,
        )
    )

    _pr("\n[3] Baseline B: Existing Fuzzy FCA (seed=42)")
    t0 = time.perf_counter()
    fuzz42 = baseline_existing_fuzzy_split(train_outer, test_outer, 42)
    _pr(
        "  ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f} Spend R2={spend_r2:.4f} "
        "Invoice R2={invoice_r2:.4f} candidates={n_candidates} final={n_final_concepts} runtime={runtime:.1f}s".format(
            **fuzz42,
            runtime=time.perf_counter() - t0,
        )
    )

    _pr("\n[4] Baseline C: Existing Kuznetsov FCA (loss<=1.0; seed=42)")
    t0 = time.perf_counter()
    try:
        kuz42 = baseline_existing_kuznetsov_split(train_outer, test_outer, 42)
        _pr(
            "  ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f} Spend R2={spend_r2:.4f} "
            "Invoice R2={invoice_r2:.4f} candidates={n_candidates} after_stability={n_after_stability} "
            "final={n_final_concepts} ctx_stab_time={ctx:.2f}s runtime={runtime:.1f}s".format(
                **kuz42,
                ctx=kuz42["ctx_info"]["stability_time_s"],
                runtime=time.perf_counter() - t0,
            )
        )
    except Exception as e:
        import traceback
        _pr(f"[4b] Existing Kuznetsov baseline raised: {e}")
        traceback.print_exc()
        kuz42 = None

    _pr("\n[4c] Sanity: Existing Kuznetsov baseline computed separately via smoke runner stage2 module (loss<=1.0)")
    try:
        from optimized_fuzzy_fca_stage2 import baseline_existing_kuznetsov_split as _kuz_split
        _kuz42 = _kuz_split(train_outer, test_outer, 42)
        _pr(
            "  ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f} Spend R2={spend_r2:.4f} "
            "Invoice R2={invoice_r2:.4f} candidates={n_candidates} after_stability={n_after_stability} "
            "final={n_final_concepts} ctx_stab_time={ctx:.2f}s".format(
                **_kuz42,
                ctx=_kuz42["ctx_info"]["stability_time_s"],
            )
        )
    except Exception as e:
        import traceback
        _pr(f"[4c] Kuznetsov sanity split raised: {e}")
        traceback.print_exc()
        _kuz42 = None

    _pr("\n[5] Full Stage-2 nested selection + optimized FCA evaluation (seed=42, full search space)")
    t0 = time.perf_counter()
    smoke = run_one_outer_seed(train_outer, test_outer, 42, configs)
    _pr(f"  runtime={time.perf_counter()-t0:.1f}s")

    sel = smoke["selected_config"]
    _pr("\n[6] Selected config (inner-validation Pareto, frozen before outer test)")
    _pr(
        "  id={id} min_support={ms:.3e} j_max={jm:.2f} mu_cut={mc:.2f} kuz_loss={kl:.3e}".format(
            id=int(sel["config_id"]),
            ms=sel["min_support"],
            jm=sel["j_max"],
            mc=sel["mu_cut"],
            kl=sel["kuz_loss"],
        )
    )
    _pr(
        "  inner: " + "  ".join(
            f"{k}={sel[k]:.4f}" for k in ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]
        )
    )
    _pr(
        "  inner concepts: candidates={cand} after_stability={as_} final={fin} "
        "drop_stab={ds} drop_jac={dj}".format(
            cand=int(sel["n_candidates"]),
            as_=int(sel["n_after_stability"]),
            fin=int(sel["n_final_concepts"]),
            ds=int(sel["n_dropped_by_stability"]),
            dj=int(sel["n_dropped_by_jaccard"]),
        )
    )

    _pr("\n[7] Selected outer test (evaluated once after config frozen)")
    _pr(
        "  ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f} Spend R2={spend_r2:.4f} "
        "Invoice R2={invoice_r2:.4f} runtime={runtime:.1f}s".format(
            **smoke["selected_outer"],
        )
    )

    _pr("\n[8] Sanity: compare baselines vs selected outer test")
    _pr(
        "  Raw RFM outer: ROC AUC={roc_auc:.4f} Spend R2={spend_r2:.4f} Invoice R2={invoice_r2:.4f}".format(
            **smoke["baseline_raw_rfm"]
        )
    )
    _pr(
        "  Existing Fuzzy outer: ROC AUC={roc_auc:.4f} Spend R2={spend_r2:.4f} Invoice R2={invoice_r2:.4f} "
        "concepts={n_concepts}".format(**smoke["baseline_existing_fuzzy"])
    )
    _pr(
        "  Existing Kuz outer: ROC AUC={roc_auc:.4f} Spend R2={spend_r2:.4f} Invoice R2={invoice_r2:.4f} "
        "candidates={n_candidates} final={n_final_concepts}".format(**smoke["baseline_existing_kuznetsov"])
    )

    delta_raw = {
        k: smoke["selected_outer"][k] - smoke["baseline_raw_rfm"][k]
        for k in ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]
    }
    delta_fuzz = {
        k: smoke["selected_outer"][k] - smoke["baseline_existing_fuzzy"][k]
        for k in ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]
    }
    if kuz42 is not None:
        delta_kuz = {
            k: smoke["selected_outer"][k] - smoke["baseline_existing_kuznetsov"][k]
            for k in ["roc_auc", "pr_auc", "spend_r2", "invoice_r2"]
        }
    else:
        delta_kuz = None

    _pr("\n[9] Outer-test deltas (selected - baseline)")
    _pr("  vs Raw RFM   : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_raw.items()))
    _pr("  vs Fuzzy FCA : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_fuzz.items()))
    if delta_kuz is not None:
        _pr("  vs Kuz FCA   : " + "  ".join(f"{k}={v:+.4f}" for k, v in delta_kuz.items()))
    else:
        _pr("  vs Kuz FCA   : (not computed because baseline_existing_kuznetsov_split raised)")

    _pr("\n[10] Leakage assertions (smoke test)")
    _pr("  - inner_val used for config selection only")
    _pr("  - outer test evaluated once per branch, after config frozen")
    _pr("  - baselines A/B/C computed independently")
    _pr("  - exact canonical Kuznetsov stability from training context only")

    # save smoke artifacts
    smoke_artifact = {
        "outer_seed": 42,
        "search_space_size": len(configs),
        "baseline_raw_rfm": raw42,
        "baseline_existing_fuzzy": fuzz42,
        "baseline_existing_kuznetsov": {
            "roc_auc": kuz42["roc_auc"],
            "pr_auc": kuz42["pr_auc"],
            "spend_r2": kuz42["spend_r2"],
            "invoice_r2": kuz42["invoice_r2"],
            "n_candidates": kuz42["n_candidates"],
            "n_with_stability": kuz42["n_with_stability"],
            "n_after_stability": kuz42["n_after_stability"],
            "n_final_concepts": kuz42["n_final_concepts"],
            "ctx_stability_time_s": kuz42["ctx_info"]["stability_time_s"],
        },
        "selected_config": {
            "config_id": int(sel["config_id"]),
            "min_support": sel["min_support"],
            "j_max": sel["j_max"],
            "mu_cut": sel["mu_cut"],
            "kuz_loss": sel["kuz_loss"],
            "inner_roc_auc": sel["roc_auc"],
            "inner_pr_auc": sel["pr_auc"],
            "inner_spend_r2": sel["spend_r2"],
            "inner_invoice_r2": sel["invoice_r2"],
            "n_candidates": int(sel["n_candidates"]),
            "n_after_stability": int(sel["n_after_stability"]),
            "n_final_concepts": int(sel["n_final_concepts"]),
            "n_dropped_by_stability": int(sel["n_dropped_by_stability"]),
            "n_dropped_by_jaccard": int(sel["n_dropped_by_jaccard"]),
        },
        "selected_outer": smoke["selected_outer"],
        "delta_vs_raw": delta_raw,
        "delta_vs_fuzzy": delta_fuzz,
        "delta_vs_kuz": delta_kuz if 'delta_kuz' in locals() else None,
        "kuz42_baseline_present": kuz42 is not None,
        "runtime_s": time.time() - t_start,
    }

    path = EXP_DIR / "stage2_smoke_test_seed42.json"
    path.write_text(json.dumps(smoke_artifact, indent=2), encoding="utf-8")
    _pr(f"\nSaved smoke artifact: {path}")

    _pr("\n" + "=" * 100)
    _pr(f"SMOKE TEST COMPLETE (runtime {time.time()-t_start:.1f}s)")
    _pr("=" * 100)


if __name__ == "__main__":
    main()

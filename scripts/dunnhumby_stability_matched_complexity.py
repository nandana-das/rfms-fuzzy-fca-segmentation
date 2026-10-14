"""Dunnhumby stability-ranked matched-complexity diagnostic.

The existing RFM+T representation is fitted per training fold. Conditional
suppression stability is estimated by resampling training customers while
keeping that fold's mined concept set and memberships fixed, reapplying the
existing suppressor, and matching concepts by exact canonical intent string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_customer_concept_memberships,
    load_cleaned_transactions,
    quantile_scores,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    _band_centroids_generic,
    _piecewise_membership_generic,
    mine_fuzzy_closed_concepts_with_thresholds,
)
from temporal_rfm_fca_ablation import (  # noqa: E402
    RFM_DIMS,
    T_DIMS,
    _fuzzy_memberships,
    _score_temporal,
    temporal_cohorts,
)

OUT = ROOT / "results" / "dunnhumby_stability_matched_complexity"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ("auc", "spend_r2", "invoice_r2")
N_RESAMPLES = 50
RESAMPLE_SEED = 20261013


def _build_t_context(tr: pd.DataFrame, te: pd.DataFrame):
    tr = tr.copy()
    te = te.copy()
    missing_train = int(tr["T"].isna().sum())
    missing_test = int(te["T"].isna().sum())
    t_median = float(tr["T"].median())
    if not np.isfinite(t_median):
        raise ValueError("Training fold has no observed repeat intervals for T imputation")
    tr["T"] = tr["T"].fillna(t_median)
    te["T"] = te["T"].fillna(t_median)
    scored_tr, cuts = _score_temporal(tr, T_DIMS)
    scored_te = quantile_scores(te[["CustomerID", *T_DIMS]].copy(), cuts, dims=T_DIMS)
    # R is already inverted by quantile_scores; only T needs inversion here.
    for dim in ("T",):
        n_levels = len(cuts[dim]) + 1
        scored_te[f"{dim}_score"] = (n_levels + 1 - scored_te[f"{dim}_score"]).astype(np.int8)
    mu_tr, centroids = _fuzzy_memberships(scored_tr, T_DIMS)
    mu_te, _ = _fuzzy_memberships(scored_te, T_DIMS, centroids=centroids)
    raw, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        mu_tr, scored_tr, bl.L_THRESHOLDS, min_support=bl.MIN_SUPPORT, dims=T_DIMS
    )
    raw = raw.reset_index(drop=True)
    mu_all = compute_customer_concept_memberships(raw, mu_tr)
    kept = suppress_redundant_concepts(
        raw, mu_all, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True
    ).reset_index(drop=True)
    x_tr = compute_customer_concept_memberships(kept, mu_tr)
    x_te = compute_customer_concept_memberships(kept, mu_te)
    return {
        "scored_tr": scored_tr,
        "mu_tr": mu_tr,
        "kept": kept,
        "x_tr": x_tr,
        "x_te": x_te,
        "t_imputation_median": t_median,
        "t_missing_train": missing_train,
        "t_missing_test": missing_test,
    }


def _stable_concepts(context: dict, k: int, rng: np.random.Generator):
    kept = context["kept"]
    base_intents = kept["intent"].astype(str).tolist()
    counts = dict.fromkeys(base_intents, 0)
    n = len(context["x_tr"])
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        mu_sample = context["x_tr"][idx, :]
        sampled_kept = suppress_redundant_concepts(
            kept, mu_sample, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True
        )
        present = set(sampled_kept["intent"].astype(str))
        for intent in counts:
            counts[intent] += int(intent in present)
    support = kept["support"].to_numpy(dtype=float)
    stability = np.array([counts[i] / N_RESAMPLES for i in base_intents], dtype=float)
    ranking = pd.DataFrame({
        "intent": base_intents,
        "stability": stability,
        "support": support,
        "intent_size": kept["intent_size"].to_numpy(dtype=int),
    })
    ranking = ranking.sort_values(
        ["stability", "support", "intent_size", "intent"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking["rank"] <= min(k, len(ranking))
    return ranking


def main():
    counts = pd.read_csv(ROOT / "results" / "dunnhumby_matched_complexity" / "feature_counts_per_fold.csv")
    cached = pd.read_csv(ROOT / "results" / "temporal_rfm_fca_ablation" / "per_origin_metrics.csv")
    first_k = pd.read_csv(ROOT / "results" / "dunnhumby_matched_complexity" / "per_origin_metrics.csv")
    dh_tx = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = [c for c in temporal_cohorts(dh_tx, load_cleaned_transactions()) if c["dataset"] == "Dunnhumby"]
    metric_rows, count_rows, concept_rows = [], [], []
    for cohort in cohorts:
        print(f"[{cohort['dataset']} | {cohort['origin']}] n={len(cohort['df'])}", flush=True)
        df = cohort["df"].reset_index(drop=True)
        oof = {m: np.full(len(df), np.nan) for m in ("prob", "sp", "inv")}
        skf = StratifiedKFold(n_splits=bl.N_FOLDS, shuffle=True, random_state=bl.CV_SEED)
        for fold, (i_tr, i_te) in enumerate(skf.split(df, df["repurchased"])):
            tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
            target = counts[(counts.origin == cohort["origin"]) & (counts.fold == fold)]
            if len(target) != 1:
                raise ValueError(f"Missing cached M0 count for {cohort['origin']} fold {fold}")
            k = int(target.iloc[0]["m0_features"])
            context = _build_t_context(tr, te)
            ranking = _stable_concepts(context, k, np.random.default_rng(RESAMPLE_SEED + fold))
            selected = ranking[ranking.selected]["intent"].tolist()
            kept_intents = context["kept"]["intent"].astype(str).tolist()
            selected_idx = [kept_intents.index(i) for i in selected]
            xtr, xte = context["x_tr"][:, selected_idx], context["x_te"][:, selected_idx]
            p, s, v = bl.fit_predict("fuzzy_rfm_fca", xtr, xte, tr, bl.CV_SEED + fold)
            oof["prob"][i_te], oof["sp"][i_te], oof["inv"][i_te] = p, s, v
            count_rows.append({
                "dataset": cohort["dataset"], "origin": cohort["origin"], "fold": fold,
                "m0_features": k, "rfm_t_features_unpruned": len(context["kept"]),
                "rfm_t_features_stability_matched": len(selected),
                "selected_mean_stability": float(ranking[ranking.selected]["stability"].mean()),
                "selected_min_stability": float(ranking[ranking.selected]["stability"].min()),
                "selected_max_stability": float(ranking[ranking.selected]["stability"].max()),
                "n_resamples": N_RESAMPLES,
                "t_imputation_median": context["t_imputation_median"],
                "t_missing_train": context["t_missing_train"],
                "t_missing_test": context["t_missing_test"],
            })
            ranking.insert(0, "fold", fold)
            ranking.insert(0, "origin", cohort["origin"])
            concept_rows.append(ranking)
        full = np.arange(len(df))[None, :]
        point = {k: float(v[0]) for k, v in bl.metric_rows(df, oof, full).items()}
        metric_rows.append({"dataset": cohort["dataset"], "origin": cohort["origin"], "pooled": cohort["pooled"], "n": len(df), "arm": "fuzzy_rfm_t_stability_matched", **point})
    metrics = pd.DataFrame(metric_rows)
    refs = pd.concat([
        cached[(cached.dataset == "Dunnhumby") & cached.pooled & cached.arm.isin(["fuzzy_rfm", "fuzzy_rfm_t"])],
        first_k[(first_k.dataset == "Dunnhumby") & first_k.pooled & (first_k.arm == "fuzzy_rfm_t_matched")],
    ], ignore_index=True)
    comparisons = []
    for _, r in metrics.iterrows():
        for ref_arm, label in (("fuzzy_rfm", "stability_matched_minus_m0"), ("fuzzy_rfm_t", "stability_matched_minus_unpruned"), ("fuzzy_rfm_t_matched", "stability_matched_minus_first_k")):
            ref = refs[(refs.origin == r.origin) & (refs.arm == ref_arm)].iloc[0]
            for metric in METRICS:
                comparisons.append({"dataset": r.dataset, "origin": r.origin, "metric": metric, "comparison": label, "delta": r[metric] - ref[metric], "paired_inference": "unavailable: cached customer predictions not retained"})
    metrics.to_csv(OUT / "per_origin_metrics.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "feature_counts_per_fold.csv", index=False)
    pd.concat(concept_rows, ignore_index=True).to_csv(OUT / "concept_stability_per_fold.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "comparisons_to_cached.csv", index=False)
    params = {"dataset": "Dunnhumby", "n_resamples": N_RESAMPLES, "resample_seed": RESAMPLE_SEED, "concept_identity": "exact canonical intent string", "stability": "fraction of training-only resamples in which the base fold's mined concept survives reapplication of existing suppression", "stability_scope": "conditional on the base fold's mined concept set, memberships, and support ordering; closure mining is not rerun per resample", "ranking": ["stability desc", "base-fold support desc", "intent_size asc", "intent asc"], "k": "same-fold cached M0 feature count", "paired_inference": "unavailable because cached customer-level predictions were not retained"}
    (OUT / "run_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    lines = ["# Dunnhumby stability-ranked matched-complexity diagnostic", "", "The standard RFM+T miner and suppressor were run once per training fold. Conditional suppression stability was then estimated from training-only bootstrap resamples with the fold-fitted memberships and mined concept set fixed; the existing suppressor was reapplied to each resampled extent matrix. Exact canonical intent strings identified concepts across resamples. Concepts were ranked by stability descending, support descending, intent size ascending, and intent lexicographically; K equaled the cached same-fold M0 feature count.", "", "This is exploratory: matching feature count also changes concept composition and does not isolate a pure representation-size effect. Stability is conditional on the base fold's mined concept set; it is not a full resampling stability analysis of concept discovery.", "", "Paired uncertainty intervals and Holm tests against cached arms are unavailable because customer-level cached predictions were not retained.", "", "Outputs: `per_origin_metrics.csv`, `feature_counts_per_fold.csv`, `concept_stability_per_fold.csv`, `comparisons_to_cached.csv`, `run_parameters.json`.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

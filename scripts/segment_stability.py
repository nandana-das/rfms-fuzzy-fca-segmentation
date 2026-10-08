"""Segment stability of crisp vs fuzzy RFM-FCA (structural-robustness experiment).

Question: does the fuzzy formal context give customer segment profiles that are
more stable than crisp RFM-FCA under (A) refitting on different customer samples
and (B) quarterly re-segmentation over time?

Arms (quintile scoring and locked parameters, identical to
baseline_ladder_rolling_origin.py):
    crisp_rfm_fca   crisp closed concepts, support >= 0.04
    fuzzy_rfm_fca   fuzzy concepts + Jaccard suppression (empty-core fix)
    kuznetsov_fca   fuzzy concepts filtered by 1 - float(stability) <= 5.4e-20 + Jaccard

Customer profile = the customer's membership in every concept of a fitted model.
Concepts are identified across fits by their set of base bands (e.g. {R5, M4}),
so a concept missing from one fit contributes membership 0 there.

Measures (pre-specified):
    core_jaccard   PRIMARY. Jaccard of the sets of concepts in which the customer
                   has membership >= 0.5. Gives fuzzy no partial credit for
                   graded memberships (for crisp it is the ordinary Jaccard).
    fuzzy_jaccard  secondary. sum(min) / sum(max) over membership vectors.
    concept_recurrence  secondary (study A). Share of the reference fit's concepts
                   that reappear in a refit.

Study A - refit stability: at every pooled origin, a reference model is fit on the
full cohort and S = 30 models on 80% subsamples without replacement (the same
subsamples for every arm). Each model projects the full cohort; profiles are
compared with the reference. Paired arm differences are summarised over
subsamples (2.5-97.5 percentile interval), and pooled as the mean over origins.

Study B - re-segmentation over time: for consecutive origins t1 < t2 (91 days
apart), a model is fit on each cohort (as a business re-running the segmentation
each quarter would) and every customer present at both is profiled by
model_t1(RFM_t1) and model_t2(RFM_t2). Reported for all customers and for
"behaviourally stable" customers whose within-cohort percentile rank moved by at
most 0.05 in each of R, F and M: changes for them are spurious migration. Paired
arm differences: customer-level bootstrap (B = 2000), pooled over origin pairs,
Holm-corrected within each dataset.

No outcome data is used anywhere in this experiment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import baseline_ladder_rolling_origin as bl  # noqa: E402
from concept_redundancy import suppress_redundant_concepts  # noqa: E402
from fair_comparison_retail2 import (  # noqa: E402
    compute_fuzzy_memberships,
    extract_base_bands_from_intent,
    load_cleaned_transactions,
    mine_crisp_closed_concepts,
    quantile_scores,
)
from fuzzy_membership_sensitivity import (  # noqa: E402
    TRANSACTION_FILE,
    mine_fuzzy_closed_concepts_with_thresholds,
)
from kuznetsov_pruning_leakage_free import compute_split_stability  # noqa: E402

EXP_DIR = ROOT_DIR / "results" / "segment_stability"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DIMS = bl.DIMS
ARMS = ["crisp_rfm_fca", "fuzzy_rfm_fca", "kuznetsov_fca"]
COMPARISONS = [("fuzzy_rfm_fca", "crisp_rfm_fca"), ("kuznetsov_fca", "crisp_rfm_fca")]
MEASURES = ["core_jaccard", "fuzzy_jaccard"]

N_SUBSAMPLES = 30
SUBSAMPLE_FRAC = 0.80
SUBSAMPLE_SEED = 2026
BOOT_N = 2000
BOOT_SEED = 777
STABLE_RANK_TOL = 0.05


def _pr(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# Fit / project
# ===========================================================================

def fit_model(arm: str, df: pd.DataFrame) -> dict:
    scored, cut = bl._quintile_scored(df)
    if arm == "crisp_rfm_fca":
        concepts = mine_crisp_closed_concepts(scored, min_support=bl.MIN_SUPPORT, dims=DIMS)
        concepts = concepts[concepts["intent_size"] > 0]
        return {"arm": arm, "cut": cut, "keys": _keys(concepts["intent"])}
    mu, centroids, _ = compute_fuzzy_memberships(scored, dims=DIMS)
    raw, _, _ = mine_fuzzy_closed_concepts_with_thresholds(
        mu, scored, bl.L_THRESHOLDS, min_support=bl.MIN_SUPPORT, dims=DIMS
    )
    raw = raw.reset_index(drop=True)
    mu_all = bl.compute_customer_concept_memberships(raw, mu)
    pool, mu_pool = raw, mu_all
    if arm == "kuznetsov_fca":
        stab, _ = compute_split_stability(raw, mu, bl.L_THRESHOLDS)
        stab = stab.set_index("concept_index")
        keep = [
            i for i in range(len(raw))
            if str(stab.at[i, "status"]).startswith("exact")
            and (1.0 - float(stab.at[i, "stab_float"])) <= bl.KUZ_LOSS_THRESHOLD
        ]
        pool, mu_pool = raw.iloc[keep].reset_index(drop=True), mu_all[:, keep]
    kept = suppress_redundant_concepts(pool, mu_pool, j_max=bl.J_MAX, mu_cut=bl.MU_CUT, drop_empty_core=True)
    return {"arm": arm, "centroids": centroids, "keys": _keys(kept["intent"])}


def _keys(intents) -> list[frozenset]:
    out, seen = [], set()
    for intent in intents:
        k = frozenset(extract_base_bands_from_intent(intent))
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def project(model: dict, df: pd.DataFrame) -> dict[frozenset, np.ndarray]:
    """Membership of every customer in every concept of the model."""
    cust = df[["CustomerID", *DIMS]].copy()
    if model["arm"] == "crisp_rfm_fca":
        scored = quantile_scores(cust, model["cut"], dims=DIMS)
        bands = {f"{d}{k}": (scored[f"{d}_score"].to_numpy() == k).astype(float)
                 for d in DIMS for k in range(1, 6)}
    else:
        mu, _, _ = compute_fuzzy_memberships(cust, trained_centroids=model["centroids"], dims=DIMS)
        bands = {c: mu[c].to_numpy() for c in mu.columns}
    return {k: np.min(np.column_stack([bands[b] for b in k]), axis=1) for k in model["keys"]}


def profile_similarity(pa: dict, pb: dict, n: int) -> dict[str, np.ndarray]:
    keys = list(set(pa) | set(pb))
    zero = np.zeros(n)
    A = np.column_stack([pa.get(k, zero) for k in keys])
    B = np.column_stack([pb.get(k, zero) for k in keys])
    out = {}
    for name, (a, b) in {
        "fuzzy_jaccard": (A, B),
        "core_jaccard": ((A >= bl.MU_CUT).astype(float), (B >= bl.MU_CUT).astype(float)),
    }.items():
        num, den = np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1)
        out[name] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    return out


# ===========================================================================
# Study A - refit stability
# ===========================================================================

def study_a(cohorts: list[dict]) -> pd.DataFrame:
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    rows = []
    for c in cohorts:
        df = c["df"]
        n = len(df)
        ts = time.time()
        subsamples = [rng.choice(n, size=int(round(SUBSAMPLE_FRAC * n)), replace=False) for _ in range(N_SUBSAMPLES)]
        for arm in ARMS:
            ref = fit_model(arm, df)
            p_ref = project(ref, df)
            core_ref = np.column_stack([v >= bl.MU_CUT for v in p_ref.values()]).sum(axis=1).mean()
            for s, idx in enumerate(subsamples):
                m = fit_model(arm, df.iloc[idx].reset_index(drop=True))
                sim = profile_similarity(p_ref, project(m, df), n)
                shared = len(set(ref["keys"]) & set(m["keys"]))
                rows.append({
                    "dataset": c["dataset"], "origin": c["origin"], "arm": arm, "subsample": s,
                    "core_jaccard": float(sim["core_jaccard"].mean()),
                    "fuzzy_jaccard": float(sim["fuzzy_jaccard"].mean()),
                    "concept_recurrence": shared / len(ref["keys"]),
                    "n_concepts_ref": len(ref["keys"]), "n_concepts_fit": len(m["keys"]),
                    "core_concepts_per_customer_ref": float(core_ref),
                })
        _pr(f"  [A] {c['dataset']} | {c['origin']} done in {time.time() - ts:.0f}s")
    return pd.DataFrame(rows)


def summarise_a(df_a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = df_a.groupby(["dataset", "arm"], sort=False).agg(
        core_jaccard=("core_jaccard", "mean"), fuzzy_jaccard=("fuzzy_jaccard", "mean"),
        concept_recurrence=("concept_recurrence", "mean"), n_concepts_ref=("n_concepts_ref", "mean"),
        core_concepts_per_customer=("core_concepts_per_customer_ref", "mean"),
    ).reset_index()
    comps = []
    for dataset, g in df_a.groupby("dataset", sort=False):
        origins = list(dict.fromkeys(g["origin"]))
        for arm, ref in COMPARISONS:
            for measure in MEASURES + ["concept_recurrence"]:
                per_origin = []
                for o in origins:
                    a = g[(g["origin"] == o) & (g["arm"] == arm)].sort_values("subsample")[measure].to_numpy()
                    b = g[(g["origin"] == o) & (g["arm"] == ref)].sort_values("subsample")[measure].to_numpy()
                    per_origin.append(a - b)
                d = np.mean(per_origin, axis=0)  # pooled over origins, per subsample index
                comps.append({
                    "dataset": dataset, "comparison": f"{arm} - {ref}", "measure": measure,
                    "delta": float(d.mean()), "interval_low": float(np.percentile(d, 2.5)),
                    "interval_high": float(np.percentile(d, 97.5)),
                    "origins_positive": f"{sum(x.mean() > 0 for x in per_origin)}/{len(per_origin)}",
                })
    return summary, pd.DataFrame(comps)


# ===========================================================================
# Study B - re-segmentation over time
# ===========================================================================

def _pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("CustomerID")[list(DIMS)].rank(pct=True)


def study_b(cohorts: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOT_SEED)
    summary_rows, draws = [], {}
    by_dataset: dict[str, list[dict]] = {}
    for c in cohorts:
        by_dataset.setdefault(c["dataset"], []).append(c)
    for dataset, cs in by_dataset.items():
        for c1, c2 in zip(cs[:-1], cs[1:]):
            d1 = c1["df"].set_index("CustomerID")
            d2 = c2["df"].set_index("CustomerID")
            common = d1.index.intersection(d2.index)
            r1, r2 = _pct_rank(c1["df"]).loc[common], _pct_rank(c2["df"]).loc[common]
            stable = ((r1 - r2).abs().max(axis=1) <= STABLE_RANK_TOL).to_numpy()
            x1 = d1.loc[common].reset_index()
            x2 = d2.loc[common].reset_index()
            pair = f"{c1['origin']} -> {c2['origin']}"
            boot_idx = {"all": rng.integers(0, len(common), size=(BOOT_N, len(common)))}
            st_idx = np.flatnonzero(stable)
            boot_idx["stable"] = st_idx[rng.integers(0, len(st_idx), size=(BOOT_N, len(st_idx)))]
            for arm in ARMS:
                sim = profile_similarity(
                    project(fit_model(arm, c1["df"]), x1), project(fit_model(arm, c2["df"]), x2), len(common)
                )
                for subset, mask in (("all", np.ones(len(common), bool)), ("stable", stable)):
                    row = {"dataset": dataset, "pair": pair, "arm": arm, "subset": subset,
                           "n_customers": int(mask.sum())}
                    for m in MEASURES:
                        row[m] = float(sim[m][mask].mean())
                        draws[(dataset, pair, arm, subset, m)] = sim[m][boot_idx[subset]].mean(axis=1)
                    summary_rows.append(row)
            _pr(f"  [B] {dataset} | {pair}: {len(common):,} customers, {int(stable.sum()):,} stable")
    summary = pd.DataFrame(summary_rows)

    comps = []
    for dataset, g in summary.groupby("dataset", sort=False):
        pairs = list(dict.fromkeys(g["pair"]))
        for arm, ref in COMPARISONS:
            for subset in ("all", "stable"):
                for m in MEASURES:
                    deltas = [
                        g[(g["pair"] == p) & (g["arm"] == arm) & (g["subset"] == subset)][m].iloc[0]
                        - g[(g["pair"] == p) & (g["arm"] == ref) & (g["subset"] == subset)][m].iloc[0]
                        for p in pairs
                    ]
                    pooled = np.mean([draws[(dataset, p, arm, subset, m)] - draws[(dataset, p, ref, subset, m)]
                                      for p in pairs], axis=0)
                    comps.append({
                        "dataset": dataset, "comparison": f"{arm} - {ref}", "subset": subset, "measure": m,
                        "delta": float(np.mean(deltas)), "ci_low": float(np.percentile(pooled, 2.5)),
                        "ci_high": float(np.percentile(pooled, 97.5)), "p_boot": bl._boot_p(pooled),
                        "pairs_positive": f"{sum(d > 0 for d in deltas)}/{len(deltas)}",
                    })
    comp = pd.DataFrame(comps)
    comp["p_holm"] = np.nan
    for dataset in comp["dataset"].unique():
        mask = comp["dataset"] == dataset
        comp.loc[mask, "p_holm"] = bl._holm(comp.loc[mask, "p_boot"].tolist())
    return summary, comp


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t0 = time.time()
    _pr("Loading cohorts...")
    dh_tx = pd.read_csv(TRANSACTION_FILE, usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    cohorts = bl.dunnhumby_cohorts(dh_tx) + [c for c in bl.retail2_cohorts(load_cleaned_transactions()) if c["pooled"]]

    _pr("Study A - refit stability")
    df_a = study_a(cohorts)
    sum_a, comp_a = summarise_a(df_a)
    _pr("Study B - re-segmentation over time")
    sum_b, comp_b = study_b(cohorts)

    df_a.to_csv(EXP_DIR / "refit_per_subsample.csv", index=False)
    sum_a.to_csv(EXP_DIR / "refit_summary.csv", index=False)
    comp_a.to_csv(EXP_DIR / "refit_comparisons.csv", index=False)
    sum_b.to_csv(EXP_DIR / "temporal_summary.csv", index=False)
    comp_b.to_csv(EXP_DIR / "temporal_comparisons.csv", index=False)
    (EXP_DIR / "run_parameters.json").write_text(json.dumps({
        "arms": ARMS, "comparisons": [f"{a} - {b}" for a, b in COMPARISONS],
        "primary_measure": "core_jaccard", "n_subsamples": N_SUBSAMPLES, "subsample_frac": SUBSAMPLE_FRAC,
        "subsample_seed": SUBSAMPLE_SEED, "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
        "stable_rank_tol": STABLE_RANK_TOL, "min_support": bl.MIN_SUPPORT, "l_thresholds": bl.L_THRESHOLDS,
        "j_max": bl.J_MAX, "mu_cut": bl.MU_CUT, "kuznetsov_loss_threshold": bl.KUZ_LOSS_THRESHOLD,
        "scoring": "tie-preserving quintiles", "origins": [f"{c['dataset']} | {c['origin']}" for c in cohorts],
    }, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Segment Stability: Crisp vs Fuzzy RFM-FCA",
        "",
        "Generated by `scripts/segment_stability.py`; see its docstring for the pre-specified protocol.",
        "Primary measure: **core_jaccard** (concepts with membership >= 0.5; no partial credit for graded membership).",
        "",
        "## Study A - refit stability (80% subsamples vs full-cohort fit)",
        "",
        bl._md(sum_a),
        "",
        "Paired differences, pooled over origins (interval = 2.5-97.5 percentiles over subsamples):",
        "",
        bl._md(comp_a),
        "",
        "## Study B - quarterly re-segmentation (refit at consecutive origins)",
        "",
        bl._md(sum_b.groupby(["dataset", "arm", "subset"], sort=False)[["n_customers", *MEASURES]].mean().reset_index()),
        "",
        "Paired differences, pooled over origin pairs (customer bootstrap; Holm within dataset):",
        "",
        bl._md(comp_b),
        "",
        "Per origin pair:",
        "",
        bl._md(sum_b),
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min.",
        "",
    ]
    (EXP_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _pr(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs: {EXP_DIR}")


if __name__ == "__main__":
    main()

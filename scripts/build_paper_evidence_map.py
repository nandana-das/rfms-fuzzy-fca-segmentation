"""Build docs/PAPER_EVIDENCE_MAP.md from committed result CSVs and Git metadata.

Documentation tooling only. Strictly read-only with respect to evidence: it reads
existing result CSVs and Git history, fits no models, runs no experiments, does not
regenerate any result file, and writes exactly one file: docs/PAPER_EVIDENCE_MAP.md.

Regenerate (from the repository root, with Git available):
    python scripts/build_paper_evidence_map.py

Provenance rules (applied from Git, never hand-entered):
    originating commit = the oldest commit that touched the artifact path
                         (`git log --format=%h -- <path>`); if later commits touched it,
                         the map lists them under "Modified after origin".
    originating branch = the first branch, in lineage order
                         main -> experiment/optimized-fuzzy-fca
                         -> experiment/v2-hybrid-fuzzy-fca -> experiment/cdnow-confirmation,
                         whose history contains the originating commit.
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "PAPER_EVIDENCE_MAP.md"
R = ROOT / "results"
LINEAGE = ["main", "experiment/optimized-fuzzy-fca", "experiment/v2-hybrid-fuzzy-fca", "experiment/cdnow-confirmation",
           "experiment/limitations-quick-wins"]
# `main` was fast-forwarded to the experiment chain on 2026-10-09, so it now contains every commit.
# For provenance, "main" means its tip before that consolidation. The experiment branches were
# then replaced by tags at their final commits; branch names stay as labels, resolved via REF.
MAIN_BEFORE_CONSOLIDATION = "44653e1"
REF = {"main": MAIN_BEFORE_CONSOLIDATION,
       "experiment/optimized-fuzzy-fca": "evidence/v1-frozen",
       "experiment/v2-hybrid-fuzzy-fca": "evidence/v2-hybrid",
       "experiment/cdnow-confirmation": "evidence/cdnow-confirmation",
       "experiment/limitations-quick-wins": "evidence/limitations-quick-wins"}
B = 2000  # bootstrap resamples used by every family

# ---------------- provenance (originating commit = the only commit touching each artifact) ----------------
ARTIFACTS = [
    # key, path, content, status
    ("LADDER", "results/baseline_ladder_rolling_origin/paired_comparisons.csv", "v1 predictive comparisons (pooled + per origin)", "frozen (v1 primary)"),
    ("LADDER_M", "results/baseline_ladder_rolling_origin/per_origin_metrics.csv", "v1 per-origin metrics, cohort n and repurchase rates", "frozen (v1 primary)"),
    ("OCC", "results/baseline_ladder_rolling_origin/band_occupancy.csv", "quintile band occupancy per origin", "frozen (v1 primary)"),
    ("MEANS", "results/final_evidence/means_by_arm.csv", "v1 mean metrics by representation", "frozen (v1 primary)"),
    ("STAB_A", "results/segment_stability/refit_comparisons.csv", "v1 refit stability comparisons", "frozen (v1 primary)"),
    ("STAB_B", "results/segment_stability/temporal_comparisons.csv", "v1 quarterly re-segmentation comparisons", "frozen (v1 primary)"),
    ("MATCH_A", "results/segment_stability/matched_count_diagnostic/refit_comparisons.csv", "matched-count diagnostic, refit", "frozen (v1 diagnostic; rule fixed before computing)"),
    ("MATCH_B", "results/segment_stability/matched_count_diagnostic/temporal_comparisons.csv", "matched-count diagnostic, temporal", "frozen (v1 diagnostic; rule fixed before computing)"),
    ("T7", "results/fair_comparison_retail2/published_table7_reconstruction.csv", "base-paper Table 7 intent reconstruction", "descriptive (pre-audit; values re-verified in the v1 freeze)"),
    ("V2", "results/v2_hybrid/paired_comparisons.csv", "v2 hybrid comparisons (Dunnhumby, Online Retail II)", "exploratory, post hoc"),
    ("V2_M", "results/v2_hybrid/per_origin_metrics.csv", "v2 per-origin metrics", "exploratory, post hoc"),
    ("V2_EQ", "results/v2_stability/segment_equivalence_summary.csv", "v2 segments = v1 segments check", "verification (plan 2582120 committed first)"),
    ("CD", "results/cdnow_confirmation/comparisons.csv", "CDNOW comparisons C1-C7", "confirmatory relative to plan 04ce2cd (+ addendum 48b354a)"),
    ("CD_M", "results/cdnow_confirmation/per_origin_metrics.csv", "CDNOW per-origin metrics and cohorts", "confirmatory relative to plan 04ce2cd"),
    ("CD_PO", "results/cdnow_confirmation/per_origin_comparisons.csv", "CDNOW per-origin comparisons", "confirmatory relative to plan 04ce2cd"),
    ("CD_VER", "results/cdnow_confirmation_verification/checks.csv", "independent verification of CDNOW (59 checks)", "verification"),
    ("QWA_R", "results/qw_cdnow_stability/refit_comparisons.csv", "CDNOW refit stability comparisons", "limitation follow-up (plan 67976b4)"),
    ("QWA_T", "results/qw_cdnow_stability/temporal_comparisons.csv", "CDNOW quarterly re-segmentation comparisons", "limitation follow-up (plan 67976b4)"),
    ("QWA_S", "results/qw_cdnow_stability/refit_summary.csv", "CDNOW refit stability levels and concept counts", "limitation follow-up (plan 67976b4)"),
    ("QWB", "results/qw_inference_robustness/robustness_vs_committed.csv", "inference robustness of key comparisons (5 fold seeds, clustered bootstrap)", "robustness re-analysis (plan 67976b4)"),
    ("QWC", "results/qw_interpretability/summary.csv", "outcome-free interpretability proxies", "descriptive proxies (plan 67976b4)"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance(path: str) -> dict:
    commits = git("log", "--format=%h", "--", path).split()
    if not commits:
        raise SystemExit(f"{path} is not committed; cannot establish provenance.")
    origin = commits[-1]
    branch = next((b for b in LINEAGE if subprocess.run(["git", "merge-base", "--is-ancestor", origin, REF.get(b, b)],
                                                         cwd=ROOT).returncode == 0), "unknown")
    later = commits[:-1]
    return {"commit": origin, "date": git("log", "-1", "--format=%cs", origin), "branch": branch,
            "later": ", ".join(f"`{c}`" for c in reversed(later)) if later else "no"}


P = {}
for key, path, content, status in ARTIFACTS:
    P[key] = {"key": key, "path": path, "content": content, "status": status, **provenance(path)}

LABEL = {"raw_std": "standardized raw RFM", "log_rfm": "log-RFM", "spline_log_rfm": "spline on log-RFM",
         "crisp_rfm_fca": "crisp RFM-FCA (base-paper method)", "fuzzy_rfm_fca": "fuzzy RFM-FCA (v1)",
         "kuznetsov_fca": "Kuznetsov-FCA (excluded ablation)", "hybrid_fuzzy_fca": "v2 hybrid (fuzzy concepts + log-RFM)",
         "hybrid_crisp_fca": "crisp hybrid (crisp concepts + log-RFM)", "fuzzy_rfm_fca_matched": "fuzzy RFM-FCA at matched concept count"}
MET = {"auc": "ROC AUC", "spend_r2": "Spend R²", "invoice_r2": "Invoice R²", "core_jaccard": "core-profile Jaccard"}
DS = {"Dunnhumby": "DH", "Online Retail II": "R2", "CDNOW": "CD"}
STATUS_Q = {
    "v1": "Frozen v1 primary study; rolling origins (DH 4, R2 5) overlap in customers and history; bootstrap reflects test-sample variability only.",
    "v2": "Exploratory, post hoc: designed after seeing v1 and evaluated on the same data.",
    "cd": "Confirmatory relative to plan 04ce2cd; only 3 overlapping origins; holdouts 2 and 3 share 1998-04-01; F has 4 levels via the planned tie-merging extension.",
    "qwA": "Limitation follow-up under plan 67976b4 (committed before code); CDNOW, 3 origins, 2 quarterly pairs; planned tie-merging extension.",
}


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def direction(delta, p):
    if p is None or np.isnan(p):
        return None
    if p < 0.05:
        return "significantly higher" if delta > 0 else "significantly lower"
    return "not significantly different"


rows = []


def add(rid, src, status_key, arm, ref, dataset, metric, delta, lo, hi, p, floor, pos, interval_kind="95% bootstrap CI", extra=""):
    d = direction(delta, p)
    if d is None:  # refit stability: interval only
        d = "higher (interval excludes 0)" if lo > 0 else "lower (interval excludes 0)" if hi < 0 else "no clear difference (interval spans 0)"
    quals = [STATUS_Q[status_key]]
    if p is not None and not np.isnan(p):
        if p >= 0.05:
            quals.append("Not significant: not evidence of equivalence.")
        elif abs(p - floor) < 1e-12:
            quals.append(f"Adjusted p at the procedure floor ({floor:.4f}): indicates significance at that floor, not effect size.")
    else:
        quals.append("Refit interval: 2.5–97.5 percentiles over 30 subsamples; no p-value.")
    if extra:
        quals.append(extra)
    rows.append({"ID": rid, "Claim (data-derived wording)": f"{LABEL[arm]} vs {LABEL[ref]}: {d}",
                 "Status": P[src]["status"], "Dataset": dataset, "Metric": MET[metric],
                 "Δ": fmt(delta), "Interval": f"[{fmt(lo)}, {fmt(hi)}]" + ("" if interval_kind == "95% bootstrap CI" else " (refit)"),
                 "Adj. p (Holm)": "n/a" if p is None or np.isnan(p) else f"{p:.4f}",
                 "Floor": f"{floor:.4f}" if floor else "n/a",
                 "Significant": "n/a" if p is None or np.isnan(p) else ("yes" if p < 0.05 else "no"),
                 "Origins/pairs Δ>0 (descriptive)": pos, "Source": src, "Qualification": " ".join(quals)})


# v1 predictive
lad = pd.read_csv(ROOT / P["LADDER"]["path"]); lad = lad[lad.scope == "POOLED"]
fam_v1 = lad.groupby("dataset").size().to_dict()
v1_specs = [("V1-H1", "fuzzy_rfm_fca", "crisp_rfm_fca", ""), ("V1-H2", "fuzzy_rfm_fca", "spline_log_rfm", ""),
            ("V1-LOG", "fuzzy_rfm_fca", "log_rfm", "Supporting: gain over linear log-RFM."),
            ("V1-RAW", "fuzzy_rfm_fca", "raw_std", "Supporting: reflects misspecification of untransformed RFM in a linear model, not FCA."),
            ("V1-KUZ", "kuznetsov_fca", "fuzzy_rfm_fca", "Ablation of an excluded component.")]
for pref, a, b, extra in v1_specs:
    for ds in ["Online Retail II", "Dunnhumby"]:
        for m in ["auc", "spend_r2", "invoice_r2"]:
            r = lad[(lad.dataset == ds) & (lad.comparison == f"{a} - {b}") & (lad.metric == m)].iloc[0]
            add(f"{pref}-{DS[ds]}-{m}", "LADDER", "v1", a, b, ds, m, r.delta, r.ci_low, r.ci_high, r.p_holm,
                fam_v1[ds] / B, r.origins_positive, extra=extra)

# v2 exploratory
v2 = pd.read_csv(ROOT / P["V2"]["path"]); v2 = v2[v2.scope == "POOLED"]
fam_v2 = v2.groupby("dataset").size().to_dict()
for pid, a, b in [("P1", "hybrid_fuzzy_fca", "spline_log_rfm"), ("P2", "hybrid_fuzzy_fca", "fuzzy_rfm_fca"),
                  ("P3", "hybrid_fuzzy_fca", "hybrid_crisp_fca"), ("P4", "hybrid_fuzzy_fca", "log_rfm"),
                  ("P5", "hybrid_fuzzy_fca", "crisp_rfm_fca")]:
    extra = "P4 concerns linear log-RFM only; value beyond nonlinear RFM is P1." if pid == "P4" else ""
    for ds in ["Online Retail II", "Dunnhumby"]:
        for m in ["auc", "spend_r2", "invoice_r2"]:
            r = v2[(v2.dataset == ds) & (v2.comparison == f"{a} - {b}") & (v2.metric == m)].iloc[0]
            add(f"V2-{pid}-{DS[ds]}-{m}", "V2", "v2", a, b, ds, m, r.delta, r.ci_low, r.ci_high, r.p_holm,
                fam_v2[ds] / B, r.origins_positive, extra=extra)

# CDNOW confirmatory
cd = pd.read_csv(ROOT / P["CD"]["path"])
for _, r in cd.iterrows():
    a, b = r.comparison.split(" - ")
    extra = "C6 concerns linear log-RFM only; value beyond nonlinear RFM is C3." if r.id == "C6" else ""
    add(f"CD-{r.id}-{r.metric}", "CD", "cd", a, b, "CDNOW", r.metric, r.delta, r.ci_low, r.ci_high, r.p_holm,
        len(cd) / B, r.origins_positive, extra=extra)

pred = pd.DataFrame(rows); rows.clear()

# stability (v1 measured; core_jaccard primary)
sa = pd.read_csv(ROOT / P["STAB_A"]["path"]); sb = pd.read_csv(ROOT / P["STAB_B"]["path"])
ma = pd.read_csv(ROOT / P["MATCH_A"]["path"]); mb = pd.read_csv(ROOT / P["MATCH_B"]["path"])
fam_sb = sb.groupby("dataset").size().to_dict(); fam_mb = mb.groupby("dataset").size().to_dict()
for src, df, kind in [("STAB_A", sa, "refit"), ("MATCH_A", ma[ma.comparison.str.contains("matched")], "refit")]:
    for _, r in df[df.measure == "core_jaccard"].iterrows():
        a, b = r.comparison.split(" - ")
        add(f"STAB-{'M' if src.startswith('MATCH') else ''}A-{DS[r.dataset]}-{a.split('_')[0]}-{b.split('_')[0]}", src, "v1", a, b,
            r.dataset, "core_jaccard", r.delta, r.interval_low, r.interval_high, np.nan, None, r.origins_positive,
            interval_kind="refit", extra="Matched-count diagnostic (count and/or concept size)." if src.startswith("MATCH") else "")
for src, df, fam in [("STAB_B", sb, fam_sb), ("MATCH_B", mb[mb.comparison.str.contains("matched")], fam_mb)]:
    for _, r in df[df.measure == "core_jaccard"].iterrows():
        a, b = r.comparison.split(" - ")
        sub = "all" if r.subset == "all" else "unchanged"
        add(f"STAB-{'M' if src.startswith('MATCH') else ''}B-{sub}-{DS[r.dataset]}-{a.split('_')[0]}-{b.split('_')[0]}", src, "v1", a, b,
            f"{r.dataset} ({'all customers' if sub == 'all' else 'unchanged-behaviour subset'})", "core_jaccard",
            r.delta, r.ci_low, r.ci_high, r.p_holm, fam[r.dataset] / B, r.pairs_positive,
            extra="Matched-count diagnostic: supports a concept count and/or concept size explanation, not count alone." if src.startswith("MATCH") else "")
qa = pd.read_csv(ROOT / P["QWA_R"]["path"]); qt = pd.read_csv(ROOT / P["QWA_T"]["path"])
fam_qt = qt.groupby("dataset").size().to_dict()
for _, r in qa[qa.measure == "core_jaccard"].iterrows():
    a, b = r.comparison.split(" - ")
    add(f"QWA-A-CD-{a.split('_')[0]}{'M' if 'matched' in a else ''}-{b.split('_')[0]}", "QWA_R", "qwA", a, b, "CDNOW", "core_jaccard",
        r.delta, r.interval_low, r.interval_high, np.nan, None, r.origins_positive, interval_kind="refit",
        extra="Matched-count arm (count and/or concept size)." if "matched" in r.comparison else "")
for _, r in qt[qt.measure == "core_jaccard"].iterrows():
    a, b = r.comparison.split(" - ")
    sub = "all" if r.subset == "all" else "unchanged"
    add(f"QWA-B-{sub}-CD-{a.split('_')[0]}{'M' if 'matched' in a else ''}-{b.split('_')[0]}", "QWA_T", "qwA", a, b,
        f"CDNOW ({'all customers' if sub == 'all' else 'unchanged-behaviour subset'})", "core_jaccard",
        r.delta, r.ci_low, r.ci_high, r.p_holm, fam_qt[r.dataset] / B, r.pairs_positive,
        extra="Matched-count arm: on CDNOW it does not remove the gap." if "matched" in r.comparison else "")
stab = pd.DataFrame(rows)

# inference robustness rows (plan 67976b4, Part B)
qb = pd.read_csv(ROOT / P["QWB"]["path"])
QB_DS = {"retail2": "Online Retail II", "dunnhumby": "Dunnhumby", "cdnow": "CDNOW"}
QB_N = 5 * 10_000
rob_rows = []
for _, r in qb.iterrows():
    a, b = r.comparison.split(" - ")
    fam = len(qb[qb.dataset == r.dataset])
    floor_b = fam / QB_N
    word = direction(r.delta_mean_over_seeds, r.p_holm)
    quals = ["Robustness re-analysis under plan 67976b4: 5 CV fold seeds, customer-clustered bootstrap across origins, "
             "B = 10,000 per seed (mixture of 50,000 draws), Holm over 12 tests per dataset. Does not replace the committed result."]
    if r.p_holm >= 0.05:
        quals.append("Not significant: not evidence of equivalence.")
    elif abs(r.p_holm - floor_b) < 1e-12:
        quals.append(f"Adjusted p at the procedure floor ({floor_b:.5f}): indicates significance at that floor, not effect size.")
    if r.verdict.startswith("weakened"):
        quals.append("Weakened: significant in the committed analysis, not significant here; sign unchanged.")
    rob_rows.append({"ID": f"QWB-{DS[QB_DS[r.dataset]]}-{r.id}-{r.metric}",
                     "Claim (data-derived wording)": f"{LABEL[a]} vs {LABEL[b]}: {word}",
                     "Dataset": QB_DS[r.dataset], "Metric": MET[r.metric],
                     "Committed Δ (Holm p)": f"{fmt(r.committed_delta)} ({r.committed_p_holm:.4f})",
                     "Δ (mean over seeds)": fmt(r.delta_mean_over_seeds),
                     "Range over seeds": f"{fmt(r.delta_min_seed)} to {fmt(r.delta_max_seed)}",
                     "Interval": f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]", "Adj. p (Holm)": f"{r.p_holm:.5f}",
                     "Floor": f"{floor_b:.5f}", "Significant": "yes" if r.p_holm < 0.05 else "no",
                     "Verdict vs committed": r.verdict, "Source": "QWB", "Qualification": " ".join(quals)})
rob = pd.DataFrame(rob_rows)

# interpretability proxies (plan 67976b4, Part C; descriptive)
qc = pd.read_csv(ROOT / P["QWC"]["path"])
interp = pd.DataFrame([{"Dataset": r.dataset, "Arm": "crisp RFM-FCA" if r.arm == "crisp_rfm_fca" else "fuzzy RFM-FCA (= v2 segments)",
                        "K retained (range)": f"{r.K_mean:.1f} ({r.K_min:.0f}–{r.K_max:.0f})", "Coverage": f"{r.coverage_mean:.3f}",
                        "C80": f"{r.C80_mean:.2f}", "Intent length": f"{r.intent_length_mean:.2f}", "Core load": f"{r.core_load_mean:.2f}",
                        "Overlap": f"{r.overlap_mean:.3f}"} for r in qc.itertuples()])

eq = pd.read_csv(ROOT / P["V2_EQ"]["path"])
t7 = pd.read_csv(ROOT / P["T7"]["path"])
ver = pd.read_csv(ROOT / P["CD_VER"]["path"])


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


prov = pd.DataFrame([{"Key": v["key"], "Artifact": f"`{v['path']}`", "Content": v["content"], "Originating commit": f"`{v['commit']}` ({v['date']})",
                      "Originating branch": f"`{v['branch']}`", "Status": v["status"], "Modified after origin": v["later"]} for v in P.values()])

ids = lambda prefix: ", ".join(list(pred[pred.ID.str.startswith(prefix)].ID) + list(rob[rob.ID.str.startswith(prefix)].ID) + list(stab[stab.ID.str.startswith(prefix)].ID))
doc = f"""# Paper evidence map (control document for manuscript preparation)

**Generated document; do not edit by hand.** Generated on `main`.

**How this file is produced.** Every number below is read programmatically from the committed CSVs listed in §2. Every originating commit, date and branch is read from Git history. The generator, `scripts/build_paper_evidence_map.py`, fits no models, runs no experiments and writes only this file. To change anything, edit the generator and regenerate, or cite the CSV directly in the manuscript.

Regenerate from the repository root (Git required):

```
python scripts/build_paper_evidence_map.py
```

**Reading rules.**
- **Significance is not strength.** "Significant" means Holm-adjusted p < 0.05 within the stated correction family. An adjusted p equal to the family **floor** (smallest attainable value = family size / B, B = 2000) indicates significance at that floor, not a large effect. Effect sizes (Δ) and intervals carry magnitude.
- **Not significant is not equivalence.** No equivalence margins were pre-specified.
- **Descriptive counts are not tests.** "Origins/pairs Δ>0" is descriptive only.
- **Each claim is cited to the commit that generated its evidence**, not to the latest commit containing it. The experiment branches form one linear history (v1 → v2 → CDNOW → limitation follow-ups); on 2026-10-09 `main` was fast-forwarded to it, without changing any commit.
- **Robustness rows (`QWB-*`) do not replace committed results.** They re-assess the key comparisons with 5 CV fold seeds and a customer-clustered bootstrap. Report them metric by metric next to the committed rows; never summarize them only as a count.
- **Generalizability differs by status:**
  - frozen v1: 2 datasets, 4–5 overlapping origins;
  - exploratory v2: post hoc on the same 2 datasets;
  - CDNOW: confirmatory relative to its committed plan, but 1 dataset, 3 overlapping origins and a planned tie-merging extension;
  - limitation follow-ups: planned before code (`67976b4`), on the same three datasets.

## 1. Claim-to-evidence matrix

### 1a. Headline claims (wording for the manuscript; evidence rows in 1b–1e)

| # | Claim wording | Status | Evidence rows | Required qualification |
|---|---|---|---|---|
| H1 | Fuzzifying the RFM formal context improves out-of-sample prediction over crisp RFM-FCA (Rungruang et al., 2024). | Frozen v1 (Online Retail II, Dunnhumby); replicated on CDNOW (confirmatory); robustness re-analysis | {ids('V1-H1')}; {ids('CD-C1')}; {ids('QWB-R2-K1')}; {ids('QWB-DH-K1')}; {ids('QWB-CD-K1')} | Report per dataset and metric. Under the robustness re-analysis H1 holds on Spend and Invoice R² on all three datasets and on AUC on Online Retail II and Dunnhumby; the CDNOW AUC gain is no longer significant. Effects are modest; check Δ. |
| H2 | Fuzzy RFM-FCA is not shown to be superior to a strong nonlinear RFM baseline. Against the spline baseline it is significantly lower on Invoice R² on Online Retail II, and on Spend and Invoice R² on CDNOW; on CDNOW it is significantly higher on AUC. All Dunnhumby comparisons and the remaining Online Retail II comparisons are not significant. | Frozen v1; CDNOW confirmatory | {ids('V1-H2')}; {ids('CD-C2')} | Not-significant comparisons are not equivalence. The regression shortfalls are robust; the CDNOW AUC advantage is not ({ids('QWB-CD-K2-auc')}). |
| H3 | Fuzzy RFM-FCA does not improve segment stability over crisp RFM-FCA, on any of the three datasets. | Frozen v1 (Online Retail II, Dunnhumby); limitation follow-up (CDNOW) | §4; rows `STAB-*`, `QWA-*` | On Online Retail II and Dunnhumby the matched-count diagnostic supports a count and/or concept-size explanation of the temporal gap; on CDNOW matching the concept count does not remove the gap, so its source there is not identified. |
| X1 | Adding log-RFM to the fuzzy concept memberships (v2 hybrid) improves prediction over v1 fuzzy RFM-FCA: on all three metrics on Online Retail II (exploratory), on Spend and Invoice R² on CDNOW (confirmatory), and on Invoice R² only on Dunnhumby (exploratory). | Exploratory on R2/DH; confirmatory on CDNOW | {ids('V2-P2')}; {ids('CD-C4')} | v2 was designed after v1. Non-significant metrics (Dunnhumby AUC and Spend R², CDNOW AUC) are not evidence of no difference. Under the robustness re-analysis the Online Retail II AUC improvement is no longer significant ({ids('QWB-R2-K4-auc')}). |
| X2 | The v2 hybrid's advantage over the spline baseline is dataset-dependent: significant on all three metrics on CDNOW (confirmatory, and robust under re-analysis); on Online Retail II significant on all three metrics in the exploratory analysis but only on Spend R² under the robustness re-analysis; not significant on Dunnhumby. | Confirmatory (CDNOW) / exploratory, narrowed by robustness (Online Retail II) / not significant (Dunnhumby), reported separately per dataset | {ids('V2-P1')}; {ids('CD-C3')}; {ids('QWB-R2-K3')}; {ids('QWB-DH-K3')}; {ids('QWB-CD-K3')} | Margins are small. Never pool the three datasets into one superiority claim. The Online Retail II AUC and Invoice R² advantages are not robust. |
| X3 | Within the hybrid, fuzzy concepts outperform crisp concepts and add to linear log-RFM on Spend and Invoice R² in all three datasets. AUC results differ by dataset (see rows). | Exploratory / confirmatory | {ids('V2-P3')}; {ids('V2-P4')}; {ids('CD-C5')}; {ids('CD-C6')} | "Beyond linear log-RFM" is not "beyond any nonlinear transform"; that is X2. |
| N1 | Stability-based (Kuznetsov) pruning gives no consistent predictive benefit and reduces segment stability. | Frozen v1 ablation (negative) | {ids('V1-KUZ')}; §4 | One pooled comparison is significant (Online Retail II AUC, at the floor, positive at only 1 of 5 origins); the other five are not significant. Reported as an excluded component, not a contribution. |
| I1 | Structural interpretability proxies are mixed: fuzzy RFM-FCA retains more concepts and (on Online Retail II and Dunnhumby) longer intents and higher core load, but lower overlap between concepts and no larger C80. | Descriptive proxies (limitation follow-up) | §1e | These are outcome-free structural proxies, not measures of human interpretability. No claim that fuzzy RFM-FCA is more (or less) interpretable follows. |
| D1 | All 31 published Table 7 intents of the base paper are recovered; 3 of 31 customer counts match exactly. | Descriptive | §2 key T7 | The tie-breaking rule is unspecified in the base paper. |
| D2 | Gains over untransformed raw RFM mostly reflect that baseline's misspecification. | Frozen v1 (supporting) | {ids('V1-RAW')}; {ids('V1-LOG')} | Do not present gains over raw RFM as FCA gains. |

### 1b. Predictive evidence rows (generated)

{md(pred)}

### 1c. Stability evidence rows (generated; core-profile Jaccard)

{md(stab)}

### 1d. Inference-robustness rows (generated; limitation follow-up, plan `67976b4`)

Each row re-assesses a committed key comparison (K1 fuzzy − crisp; K2 fuzzy − spline; K3 hybrid − spline; K4 hybrid − v1 fuzzy). The committed estimate and adjusted p are shown for reference; the committed result is not replaced.

{md(rob)}

### 1e. Interpretability proxies (generated; descriptive, no inference)

Means over origins; outcome-free structural proxies, not measures of human interpretability.

{md(interp)}

## 2. Artifact provenance

"Originating commit" is the oldest commit that touched the artifact; any later commits are listed under "Modified after origin". "Originating branch" is the first branch in lineage order (`main` → v1 → v2 → CDNOW → limitation follow-ups) whose history contains that commit, with `main` meaning its tip before the 2026-10-09 consolidation (`{MAIN_BEFORE_CONSOLIDATION}`); later branches contain it by linear history. The experiment branches were later replaced by tags at their final commits (`evidence/v1-frozen`, `evidence/v2-hybrid`, `evidence/cdnow-confirmation`, `evidence/limitations-quick-wins`).

{md(prov)}

Further plans and verification records:
- v1 matched-count rule: `docs/AUDIT_ERRATA.md` §6.1, in `7607e09`.
- v2 plan: `docs/V2_HYBRID_ANALYSIS_PLAN.md`, in `f0342ef`. It was committed together with its results, so Git does not establish that the plan came first.
- v2 stability plan: `2582120`, committed before its check.
- CDNOW plan: `04ce2cd`, committed before analysis; addendum A1 in `48b354a`.
- CDNOW verification: `ba747f0`, {int(ver['pass'].sum())}/{len(ver)} checks passed.
- Limitation follow-up plan: `docs/QUICK_WINS_PLAN.md`, `67976b4`, committed before any of its code; results in `b5a464d`.

## 3. Required narrative connections

1. **Shortfall → motivation.**
   - v1 fuzzy RFM-FCA is significantly below the spline baseline on Invoice R² on Online Retail II ({ids('V1-H2-R2-invoice')}; Spend R² not significant), and on Spend and Invoice R² on CDNOW ({ids('CD-C2-spend')}, {ids('CD-C2-invoice')}). On CDNOW it is significantly higher on AUC ({ids('CD-C2-auc')}).
   - This top-band saturation shortfall motivates the hybrid. Report it as a v1 limitation before introducing v2.
2. **Hybrid vs spline, reported separately per dataset; never pooled.**
   - Online Retail II, exploratory: {ids('V2-P1-R2')}. Under the robustness re-analysis only Spend R² remains significant: {ids('QWB-R2-K3')}.
   - Dunnhumby, not significant: {ids('V2-P1-DH')}; robustness: {ids('QWB-DH-K3')}.
   - CDNOW, confirmatory: {ids('CD-C3')}; robust on all three metrics: {ids('QWB-CD-K3')}.
   - Present the committed result and the robustness result side by side, metric by metric.
3. **Order of presentation.** Present v1 (frozen primary) first, then v2 (exploratory), then CDNOW (confirmation), each labelled with its status. CDNOW provides additional evidence; it does not make v2 universally superior.
4. **Stability is a separate section.** It is not supported on any of the three datasets and is inherited unchanged by v2. Never combine predictive and stability results into one "performance" claim.
5. **Interpretability is reported only as descriptive structural proxies (§1e)**, with the mixed result stated in both directions.

## 4. Stability evidence

- **v1, measured (frozen):** rows `STAB-*` in §1c.
  - Refit: no clear difference, fuzzy vs crisp.
  - Quarterly re-segmentation, all customers: fuzzy significantly lower.
  - Unchanged-behaviour subset: Dunnhumby not significant; Online Retail II fuzzy higher.
  - Matched-count rows `STAB-MA-*` / `STAB-MB-*`: the temporal all-customer gap is not significant at matched count. This supports a concept count and/or concept-size explanation.
- **v2, equivalence by construction (verification, `ccdaa9a`):**

{md(eq)}

  v2 segments are identical to v1 segments in every compared fit, so v2's segment stability is v1's. There is no v2 stability improvement, and prediction stability was not evaluated.
- **CDNOW (limitation follow-up, plan `67976b4`):** rows `QWA-*` in §1c.
  - Refit: no clear difference, fuzzy vs crisp.
  - Quarterly re-segmentation: fuzzy significantly lower, for all customers and for the unchanged-behaviour subset.
  - Matched count: the gap remains (matched − crisp also significantly lower), so the count and/or concept-size explanation from the other two datasets does not carry over to CDNOW.

## 5. Claims we must not make

- Universal or general superiority over baselines, including "the hybrid beats strong baselines" without per-dataset qualification (Dunnhumby is not significant).
- Predictive equivalence or "parity" inferred from a non-significant comparison.
- Any segment-stability advantage for fuzzy RFM-FCA or the v2 hybrid.
- Greater (or lesser) interpretability: only structural proxies were computed, and they are mixed (§1e).
- That the hybrid beats the spline on all metrics on Online Retail II: under the robustness re-analysis only Spend R² remains significant.
- A robustness summary count (e.g. "31 of 36 robust") in place of metric-specific results.
- That exploratory v2 results were pre-registered, or that the v2 analysis plan preceded its run (Git does not establish this).
- That the concept structure itself causes predictive gains beyond nonlinear RFM encodings.
- That Kuznetsov stability, Kneedle pruning or hyperparameter optimization improves the framework.
- That adjusted p-values at the floor indicate large effects.
- Any number from superseded results (dense-rank scoring, untransformed-baseline comparisons, the "10-split temporal CV", or the 98.08% repurchase figure).

## 6. Manuscript table and figure plan

| Item | Content | Values from | Status label in caption |
|---|---|---|---|
| Table 1 | Datasets and cohorts (n, repurchase rate per origin) | `results/baseline_ladder_rolling_origin/per_origin_metrics.csv`; `results/cdnow_confirmation/per_origin_metrics.csv` | frozen v1 / CDNOW confirmatory |
| Table 2 | Quintile band occupancy | `results/baseline_ladder_rolling_origin/band_occupancy.csv`; `results/cdnow_confirmation/band_occupancy.csv` | descriptive |
| Table 3 | v1 mean metrics by representation | `results/final_evidence/means_by_arm.csv` | frozen v1 |
| Table 4 | v1 comparisons (H1, H2, raw, log) with Δ, CI, Holm p, floor | `results/baseline_ladder_rolling_origin/paired_comparisons.csv` (scope POOLED) | frozen v1 |
| Table 5 | v1 segment stability, including matched count | `results/segment_stability/*_comparisons.csv`, `results/segment_stability/matched_count_diagnostic/*` | frozen v1 |
| Table 6 | v2 hybrid comparisons P1–P5, separate rows per dataset | `results/v2_hybrid/paired_comparisons.csv` (scope POOLED) | exploratory, post hoc |
| Table 7 | CDNOW comparisons C1–C7 | `results/cdnow_confirmation/comparisons.csv` | confirmatory relative to plan 04ce2cd |
| Table 8 | Base-paper Table 7 reconstruction | `results/fair_comparison_retail2/published_table7_reconstruction.csv` | descriptive |
| Figure 1 | Pipeline diagram (no data) | `docs/METHODOLOGY_FINAL.md` | — |
| Figure 2 | Forest plot of Δ ± CI: fuzzy − crisp, fuzzy − spline, hybrid − spline, one panel per dataset | the three comparison CSVs above | per-panel status labels |
| Figure 3 | Stability: core-profile Jaccard, fuzzy vs crisp, refit vs quarterly | `results/segment_stability/refit_summary.csv`, `temporal_summary.csv` | frozen v1 |
| Table 9 | Inference robustness of key comparisons (committed vs re-analysis, metric by metric) | `results/qw_inference_robustness/robustness_vs_committed.csv` | robustness re-analysis (plan 67976b4) |
| Table 10 | CDNOW segment stability | `results/qw_cdnow_stability/*_comparisons.csv` | limitation follow-up (plan 67976b4) |
| Table 11 | Interpretability proxies | `results/qw_interpretability/summary.csv` | descriptive proxies |
| Supplement | Kuznetsov ablation; v2 equivalence check; CDNOW verification report; band occupancy per origin | `results/final_evidence/kuznetsov_predictive_ablation.csv`; `results/v2_stability/`; `results/cdnow_confirmation_verification/REPORT.md` | as labelled |
"""
OUTPUT.write_text(doc, encoding="utf-8", newline="\n")
print(f"Wrote {OUTPUT.relative_to(ROOT)}: predictive rows {len(pred)}, stability rows {len(stab)}, robustness rows {len(rob)}, interpretability rows {len(interp)}, provenance {len(prov)}")

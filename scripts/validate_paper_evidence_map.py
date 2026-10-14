"""Validate docs/PAPER_EVIDENCE_MAP.md against committed result CSVs and Git metadata.

Documentation tooling only. Read-only: it reads the map, result CSVs and Git history
(`git log`, `git merge-base --is-ancestor`), runs no experiments, fits no models and
writes no files. It is independent of scripts/build_paper_evidence_map.py: it re-reads
every value from the CSVs instead of reusing the generator's code.

Usage (from anywhere; Git required):
    python scripts/validate_paper_evidence_map.py              validate the committed map
    python scripts/validate_paper_evidence_map.py --self-test  also confirm, in memory, that
                                                              planted errors are detected

Checks:
- predictive rows: displayed delta, 95% CI bounds, Holm-adjusted p, correction-family floor
  (family size / B, B = 2000), significance flag, direction wording, origin counts,
  floor and non-equivalence qualification triggers;
- stability rows: delta, interval and, for temporal rows, adjusted p, floor and significance;
- evidence-row IDs: unique, and every ID cited in the text exists;
- provenance: originating commit = oldest commit touching the artifact, originating branch =
  first branch in lineage order (main, v1, v2, CDNOW) containing it, later modifications;
- every referenced repository path exists (results/, docs/, scripts/, tests/);
- follow-up rows (1f): cited sources exist and are Git-tracked; every rendered delta, Holm p and
  second-seed value matches its pooled source row; no non-finite values;
- required wording: corrected H2, X1, N1 (including the isolated Kuznetsov result),
  dataset-specific v2-vs-spline claim, no pooled superiority claim, regeneration command.

Displayed values are compared at their displayed precision (tolerance 5e-5).
Exit status: 0 if no discrepancies, 1 otherwise.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "docs" / "PAPER_EVIDENCE_MAP.md"


def validate(t: str) -> tuple[list, dict]:
    problems, counts = [], {}

    def table(start, end):
        sec = t[t.index(start):t.index(end)]
        ls = [l for l in sec.splitlines() if l.startswith("|")]
        hdr = [c.strip() for c in ls[0].strip("|").split("|")]
        return pd.DataFrame([[c.strip() for c in l.strip("|").split("|")] for l in ls[2:]], columns=hdr)

    num = lambda s: float(s.replace("+", ""))
    pred, stab = table("### 1b.", "### 1c."), table("### 1c.", "### 1d.")
    rob, interp = table("### 1d.", "### 1e."), table("### 1e.", "### 1f.")
    follow = table("### 1f.", "## 2.")
    lad = pd.read_csv("results/baseline_ladder_rolling_origin/paired_comparisons.csv"); lad = lad[lad.scope == "POOLED"]
    v2 = pd.read_csv("results/v2_hybrid/paired_comparisons.csv"); v2 = v2[v2.scope == "POOLED"]
    cd = pd.read_csv("results/cdnow_confirmation/comparisons.csv")
    inv = {"ROC AUC": "auc", "Spend R²": "spend_r2", "Invoice R²": "invoice_r2"}
    lab_rev = {"standardized raw RFM": "raw_std", "log-RFM": "log_rfm", "spline on log-RFM": "spline_log_rfm",
               "crisp RFM-FCA (base-paper method)": "crisp_rfm_fca", "fuzzy RFM-FCA (v1)": "fuzzy_rfm_fca",
               "Kuznetsov-FCA (excluded ablation)": "kuznetsov_fca", "v2 hybrid (fuzzy concepts + log-RFM)": "hybrid_fuzzy_fca",
               "crisp hybrid (crisp concepts + log-RFM)": "hybrid_crisp_fca", "fuzzy RFM-FCA at matched concept count": "fuzzy_rfm_fca_matched"}

    # 1. predictive rows: values, CIs, p, floors, significance flags, wording, qualification triggers
    for _, r in pred.iterrows():
        a_lab, rest = r["Claim (data-derived wording)"].split(" vs ", 1); b_lab, word = rest.rsplit(": ", 1)
        a, b, m = lab_rev[a_lab], lab_rev[b_lab], inv[r["Metric"]]
        src = {"V1": lad, "V2": v2, "CD": cd}[r.ID.split("-")[0]]
        q = src[(src.comparison == f"{a} - {b}") & (src.metric == m)]
        if "dataset" in q:
            q = q[q.dataset == r.Dataset]
        if len(q) != 1:
            problems.append(("pred lookup", r.ID)); continue
        q = q.iloc[0]
        lo, hi = [num(x) for x in r.Interval.strip("[]").split(", ")]
        fam = len(src[src.dataset == r.Dataset]) if "dataset" in src else len(src)
        exp_word = ("significantly higher" if q.p_holm < 0.05 and q.delta > 0 else
                    "significantly lower" if q.p_holm < 0.05 else "not significantly different")
        tests = {"delta": abs(num(r["Δ"]) - q.delta) <= 5e-5, "ci_low": abs(lo - q.ci_low) <= 5e-5, "ci_high": abs(hi - q.ci_high) <= 5e-5,
                 "p_holm": abs(float(r["Adj. p (Holm)"]) - q.p_holm) <= 5e-5, "floor": abs(float(r.Floor) - fam / 2000) < 1e-9,
                 "sig_flag": r.Significant == ("yes" if q.p_holm < 0.05 else "no"), "wording": word == exp_word,
                 "origins": r["Origins/pairs Δ>0 (descriptive)"] == str(q.origins_positive),
                 "floor_note": ("procedure floor" in r.Qualification) == (abs(q.p_holm - fam / 2000) < 1e-12),
                 "equivalence_note": ("not evidence of equivalence" in r.Qualification) == (q.p_holm >= 0.05)}
        for k, ok in tests.items():
            counts[k] = counts.get(k, 0) + 1
            if not ok:
                problems.append(("pred", r.ID, k))

    # 2. stability rows
    sa = pd.read_csv("results/segment_stability/refit_comparisons.csv"); sb = pd.read_csv("results/segment_stability/temporal_comparisons.csv")
    ma = pd.read_csv("results/segment_stability/matched_count_diagnostic/refit_comparisons.csv")
    mb = pd.read_csv("results/segment_stability/matched_count_diagnostic/temporal_comparisons.csv")
    qa = pd.read_csv("results/qw_cdnow_stability/refit_comparisons.csv")
    qt = pd.read_csv("results/qw_cdnow_stability/temporal_comparisons.csv")
    for _, r in stab.iterrows():
        a_lab, rest = r["Claim (data-derived wording)"].split(" vs ", 1); b_lab, _ = rest.rsplit(": ", 1)
        comp = f"{lab_rev[a_lab]} - {lab_rev[b_lab]}"
        matched = "-MA-" in r.ID or "-MB-" in r.ID; refit = r.ID.startswith(("STAB-A", "STAB-MA", "QWA-A"))
        ds = r.Dataset.split(" (")[0]
        quick = r.ID.startswith("QWA")
        if refit:
            q = qa if quick else (ma if matched else sa); q = q[(q.dataset == ds) & (q.comparison == comp) & (q.measure == "core_jaccard")]
            lo_c, hi_c, p, fam = q.iloc[0].interval_low, q.iloc[0].interval_high, None, None
        else:
            sub = "all" if "all customers" in r.Dataset else "stable"; src = qt if quick else (mb if matched else sb)
            q = src[(src.dataset == ds) & (src.comparison == comp) & (src.measure == "core_jaccard") & (src.subset == sub)]
            lo_c, hi_c, p = q.iloc[0].ci_low, q.iloc[0].ci_high, q.iloc[0].p_holm
            fam = len(src[src.dataset == ds]) / 2000
        lo, hi = [num(x) for x in r.Interval.replace(" (refit)", "").strip("[]").split(", ")]
        ok = [abs(num(r["Δ"]) - q.iloc[0].delta) <= 5e-5, abs(lo - lo_c) <= 5e-5, abs(hi - hi_c) <= 5e-5]
        if p is not None:
            ok += [abs(float(r["Adj. p (Holm)"]) - p) <= 5e-5, abs(float(r.Floor) - fam) < 1e-9,
                   r.Significant == ("yes" if p < 0.05 else "no")]
        counts["stability"] = counts.get("stability", 0) + 1
        if not all(ok):
            problems.append(("stab", r.ID, ok))

    # 2b. inference-robustness rows (plan 67976b4, Part B)
    qb = pd.read_csv("results/qw_inference_robustness/robustness_vs_committed.csv")
    dsmap = {"Online Retail II": "retail2", "Dunnhumby": "dunnhumby", "CDNOW": "cdnow"}
    for _, r in rob.iterrows():
        a_lab, rest = r["Claim (data-derived wording)"].split(" vs ", 1); b_lab, word = rest.rsplit(": ", 1)
        q = qb[(qb.dataset == dsmap[r.Dataset]) & (qb.comparison == f"{lab_rev[a_lab]} - {lab_rev[b_lab]}") & (qb.metric == inv[r.Metric])]
        if len(q) != 1:
            problems.append(("rob lookup", r.ID)); continue
        q = q.iloc[0]
        fam = len(qb[qb.dataset == q.dataset]) / 50_000
        exp_word = ("significantly higher" if q.p_holm < 0.05 and q.delta_mean_over_seeds > 0 else
                    "significantly lower" if q.p_holm < 0.05 else "not significantly different")
        lo, hi = [num(x) for x in r.Interval.strip("[]").split(", ")]
        smin, smax = [num(x) for x in r["Range over seeds"].split(" to ")]
        cd_, cp_ = r["Committed Δ (Holm p)"].split(" (")
        tests = {"rob_delta": abs(num(r["Δ (mean over seeds)"]) - q.delta_mean_over_seeds) <= 5e-5,
                 "rob_ci": abs(lo - q.ci_low) <= 5e-5 and abs(hi - q.ci_high) <= 5e-5,
                 "rob_seed_range": abs(smin - q.delta_min_seed) <= 5e-5 and abs(smax - q.delta_max_seed) <= 5e-5,
                 "rob_p": abs(float(r["Adj. p (Holm)"]) - q.p_holm) <= 5e-6, "rob_floor": abs(float(r.Floor) - fam) < 1e-9,
                 "rob_sig": r.Significant == ("yes" if q.p_holm < 0.05 else "no"), "rob_word": word == exp_word,
                 "rob_verdict": r["Verdict vs committed"] == q.verdict,
                 "rob_committed": abs(num(cd_) - q.committed_delta) <= 5e-5 and abs(float(cp_.rstrip(")")) - q.committed_p_holm) <= 5e-5,
                 "rob_weakened_note": ("Weakened:" in r.Qualification) == q.verdict.startswith("weakened")}
        for k, ok in tests.items():
            counts[k] = counts.get(k, 0) + 1
            if not ok:
                problems.append(("rob", r.ID, k))
    # 2c. interpretability proxies (descriptive)
    qc = pd.read_csv("results/qw_interpretability/summary.csv")
    for _, r in interp.iterrows():
        arm = "crisp_rfm_fca" if r.Arm.startswith("crisp") else "fuzzy_rfm_fca"
        q = qc[(qc.dataset == r.Dataset) & (qc.arm == arm)].iloc[0]
        ok = [abs(float(r["K retained (range)"].split(" ")[0]) - q.K_mean) <= 0.05, abs(float(r.Coverage) - q.coverage_mean) <= 5e-4,
              abs(float(r.C80) - q.C80_mean) <= 5e-3, abs(float(r["Intent length"]) - q.intent_length_mean) <= 5e-3,
              abs(float(r["Core load"]) - q.core_load_mean) <= 5e-3, abs(float(r.Overlap) - q.overlap_mean) <= 5e-4]
        counts["interp"] = counts.get("interp", 0) + 1
        if not all(ok):
            problems.append(("interp", r.Dataset, r.Arm, ok))

    # 2d. exploratory follow-up entries: scoped claims and direct source references
    expected_followups = {"E1-TEMPORAL", "E2-NONFCA-T", "E3-SECOND-SEED", "E4-MATCHED-COMPLEXITY", "E5-LUKASIEWICZ"}
    if set(follow.ID) != expected_followups:
        problems.append(("follow-up IDs", sorted(set(follow.ID) ^ expected_followups)))
    required_sources = {
        "E1-TEMPORAL": ["results/temporal_rfm_fca_ablation/REPORT.md", "scripts/temporal_rfm_fca_ablation.py", "results/temporal_rfm_fca_ablation/CORRECTED_RUN_PROVENANCE.md"],
        "E2-NONFCA-T": ["results/temporal_nonfca_baseline/REPORT.md", "scripts/temporal_nonfca_baseline.py"],
        "E3-SECOND-SEED": ["docs/TEMPORAL_SECOND_SEED_PLAN.md", "scripts/temporal_second_seed_evaluate.py", "results/temporal_second_seed/SEED_REPLICATION.md", "results/temporal_second_seed/PLAN_LOCK.txt"],
        "E4-MATCHED-COMPLEXITY": ["scripts/dunnhumby_matched_complexity_paired_inference.py", "results/dunnhumby_matched_complexity_paired_inference/REPORT.md", "results/dunnhumby_matched_complexity_paired_inference/paired_comparisons.csv"],
        "E5-LUKASIEWICZ": ["scripts/lukasiewicz_tnorm_ablation.py", "tests/test_lukasiewicz_tnorm.py", "results/lukasiewicz_tnorm_ablation/REPORT.md", "results/lukasiewicz_tnorm_ablation/PROVENANCE.md"],
    }
    for _, r in follow.iterrows():
        missing_sources = [p for p in required_sources.get(r.ID, []) if f"`{p}`" not in r["Source references"]]
        if missing_sources:
            problems.append(("follow-up source references", r.ID, missing_sources))
        if not r["Required qualification"].strip():
            problems.append(("follow-up qualification", r.ID))

    # 2e. follow-up sources: every cited path exists and is tracked by Git
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.splitlines())
    for _, r in follow.iterrows():
        for pth in re.findall(r"`([^`]+)`", r["Source references"]):
            if not os.path.exists(pth):
                problems.append(("follow-up source missing", r.ID, pth))
            elif pth not in tracked:
                problems.append(("follow-up source untracked", r.ID, pth))

    # 2f. follow-up numbers: each rendered "Δ x (Holm p y)", in order, against the pooled source row
    fu_src = {
        "temporal": "results/temporal_rfm_fca_ablation/paired_comparisons.csv",
        "nonfca": "results/temporal_nonfca_baseline/paired_comparisons.csv",
        "matched": "results/dunnhumby_matched_complexity_paired_inference/paired_comparisons.csv",
        "luk": "results/lukasiewicz_tnorm_ablation/paired_comparisons.csv",
    }
    R2, DH = "Online Retail II", "Dunnhumby"
    fu_expected = {
        "E1-TEMPORAL": [("temporal", R2, "fuzzy_rfm_t - fuzzy_rfm", m) for m in ("auc", "spend_r2", "invoice_r2")]
                       + [("temporal", DH, "fuzzy_rfm_t - fuzzy_rfm", "spend_r2"),
                          ("temporal", R2, "fuzzy_rfm_t_l - fuzzy_rfm_t", "invoice_r2"),
                          ("temporal", DH, "fuzzy_rfm_t_l - fuzzy_rfm_t", "spend_r2"),
                          ("temporal", DH, "fuzzy_rfm_t_l - fuzzy_rfm_t", "invoice_r2")],
        "E2-NONFCA-T": [("nonfca", R2, "log_rfm_t - log_rfm", "spend_r2"), ("nonfca", R2, "log_rfm_t - log_rfm", "invoice_r2")]
                       + [("nonfca", R2, "spline_log_rfm_t - spline_log_rfm", m) for m in ("auc", "spend_r2", "invoice_r2")]
                       + [("nonfca", DH, "log_rfm_t - log_rfm", "invoice_r2"),
                          ("nonfca", DH, "spline_log_rfm_t - spline_log_rfm", "auc"),
                          ("nonfca", DH, "spline_log_rfm_t - spline_log_rfm", "spend_r2")],
        "E4-MATCHED-COMPLEXITY": [("matched", None, "fuzzy_rfm_t_matched - fuzzy_rfm", m) for m in ("auc", "spend_r2", "invoice_r2")]
                       + [("matched", None, "fuzzy_rfm_t_stability_matched - fuzzy_rfm", m) for m in ("auc", "spend_r2", "invoice_r2")]
                       + [("matched", None, "fuzzy_rfm_t_matched - fuzzy_rfm_t", "auc"),
                          ("matched", None, "fuzzy_rfm_t_stability_matched - fuzzy_rfm_t", "auc")],
        "E5-LUKASIEWICZ": [("luk", DH, "M4 - M0", "auc"), ("luk", R2, "M4 - M0", "auc"), ("luk", R2, "M2 - M0", "auc"),
                           ("luk", R2, "M4 - M2", "auc"), ("luk", R2, "M4 - M2", "spend_r2")],
    }
    fu_frames = {k: pd.read_csv(v) for k, v in fu_src.items()}
    rendered_pat = re.compile(r"Δ ([+-]?\d+\.\d+|nan) \(Holm p ([0-9.]+|nan)\)")
    for _, r in follow.iterrows():
        claim = r["Evidence claim"]
        if re.search(r"\bnan\b", claim, re.I):
            problems.append(("follow-up non-finite value", r.ID))
        if r.ID not in fu_expected:
            continue
        shown = rendered_pat.findall(claim)
        want = fu_expected[r.ID]
        if len(shown) != len(want):
            problems.append(("follow-up value count", r.ID, len(shown), len(want)))
            continue
        for (d_txt, p_txt), (src, ds, cmp, met) in zip(shown, want):
            f = fu_frames[src]
            q = f[(f.comparison == cmp) & (f.metric == met)]
            if ds is not None:
                q = q[q.dataset == ds]
            if "scope" in f:
                q = q[q.scope == "POOLED"]
            if len(q) != 1:
                problems.append(("follow-up source row", r.ID, src, ds, cmp, met, len(q)))
                continue
            counts["follow_fields"] = counts.get("follow_fields", 0) + 2
            ok_d = d_txt != "nan" and abs(float(d_txt) - q.delta.iloc[0]) <= 5e-5
            ok_p = p_txt != "nan" and abs(float(p_txt) - q.p_holm.iloc[0]) <= 5e-5
            if not (ok_d and ok_p):
                problems.append(("follow-up value", r.ID, src, ds, cmp, met, d_txt, p_txt))
    sd = pd.read_csv("results/temporal_second_seed/seed_replication.csv")
    e3 = follow[follow.ID == "E3-SECOND-SEED"]
    if len(e3):
        claim = e3["Evidence claim"].iloc[0]
        parts = {R2: claim[claim.index(R2):claim.index(DH)], DH: claim[claim.index(DH):]} if R2 in claim and DH in claim else {}
        for ds, seg in parts.items():
            got = re.findall(r"(auc|spend_r2|invoice_r2): ([+-]\d+\.\d+) → ([+-]\d+\.\d+), ([^;)]+)", seg)
            exp = sd[sd.dataset == ds]
            if len(got) != len(exp):
                problems.append(("second-seed value count", ds, len(got), len(exp)))
            for met, d0, d1, verdict in got:
                q = exp[exp.metric == met]
                counts["follow_fields"] = counts.get("follow_fields", 0) + 3
                if len(q) != 1 or abs(float(d0) - q.delta_seed0.iloc[0]) > 5e-5 or abs(float(d1) - q.delta_seed1.iloc[0]) > 5e-5                         or verdict.strip() != q.verdict.iloc[0]:
                    problems.append(("second-seed value", ds, met, d0, d1, verdict))
        if not parts:
            problems.append(("second-seed datasets missing",))

    # 3. evidence-row references
    all_ids = list(pred.ID) + list(stab.ID) + list(rob.ID) + list(follow.ID)
    if len(all_ids) != len(set(all_ids)):
        problems.append(("duplicate IDs",))
    cited = set(re.findall(r"\b(?:V1|V2|CD|QWB|QWA)-[A-Za-z0-9]+-[A-Za-z0-9_\-]+", t.split("### 1b.")[0] + t[t.index("## 3."):]))
    missing = sorted(c for c in cited if c not in all_ids)
    if missing:
        problems.append(("cited IDs missing", missing))

    # 4. provenance (commit = oldest commit touching path; branch = first in lineage containing it; later commits)
    lineage = ["main", "experiment/optimized-fuzzy-fca", "experiment/v2-hybrid-fuzzy-fca", "experiment/cdnow-confirmation",
               "experiment/limitations-quick-wins"]
    prov = table("## 2.", "Further plans")
    for _, r in prov.iterrows():
        path = r.Artifact.strip("`"); commit = r["Originating commit"].split("`")[1]
        commits = subprocess.run(["git", "log", "--format=%h", "--", path], capture_output=True, text=True).stdout.split()
        # "main" = its tip before the 2026-10-09 fast-forward consolidation (it now contains every commit);
        # the experiment branches were replaced by tags at their final commits.
        ref = {"main": "44653e1", "experiment/optimized-fuzzy-fca": "evidence/v1-frozen",
               "experiment/v2-hybrid-fuzzy-fca": "evidence/v2-hybrid", "experiment/cdnow-confirmation": "evidence/cdnow-confirmation",
               "experiment/limitations-quick-wins": "evidence/limitations-quick-wins"}
        branch = next(b for b in lineage if subprocess.run(["git", "merge-base", "--is-ancestor", commits[-1], ref.get(b, b)]).returncode == 0)
        later = "no" if len(commits) == 1 else None
        if not (os.path.exists(path) and commits[-1] == commit and r["Originating branch"].strip("`") == branch
                and r["Modified after origin"] == later):
            problems.append(("provenance", path, commits, commit, branch))

    # 5. paths
    paths = set(re.findall(r"`((?:results|docs|scripts|tests)/[^`*]+?)`", t))
    for pth in paths:
        if not os.path.exists(pth):
            problems.append(("path missing", pth))

    # 6. required wording and structure
    required = {
        "regeneration command": "python scripts/build_paper_evidence_map.py",
        "H2 corrected": "significantly lower on Invoice R² on Online Retail II, and on Spend and Invoice R² on CDNOW; on CDNOW it is significantly higher on AUC",
        "X1 corrected": "on all three metrics on Online Retail II (exploratory), on Spend and Invoice R² on CDNOW (confirmatory), and on Invoice R² only on Dunnhumby (exploratory)",
        "N1 isolated Kuznetsov result": "One pooled comparison is significant (Online Retail II AUC, at the floor, positive at only 1 of 5 origins)",
        "X2 dataset-specific and narrowed": "on Online Retail II significant on all three metrics in the exploratory analysis but only on Spend R² under the robustness re-analysis; not significant on Dunnhumby",
        "H3 three datasets": "does not improve segment stability over crisp RFM-FCA, on any of the three datasets",
        "CDNOW matched count": "on CDNOW matching the concept count does not remove the gap",
        "interpretability descriptive": "not measures of human interpretability. No claim that fuzzy RFM-FCA is more (or less) interpretable follows",
        "robustness not a count": "never summarize them only as a count",
        "X2 never pooled": "Never pool the three datasets into one superiority claim",
        "narrative separate per dataset": "Hybrid vs spline, reported separately per dataset; never pooled",
        "temporal FCA qualification": "does not by itself establish an FCA-specific temporal advantage",
        "temporal non-FCA distinction": "Temporal information gains must be separated from evidence attributable specifically to FCA",
        "second-seed limitation": "not independent replication or population-level generalization",
        "matched-complexity limitation": "does not prove feature count alone caused the differences",
        "matched-complexity non-equivalence": "does not establish causal equivalence",
        "Lukasiewicz scope": "not continuous Łukasiewicz-based concept discovery",
    }
    for name, text in required.items():
        if text not in t:
            problems.append(("required wording missing", name))
    eq = pd.read_csv("results/v2_stability/segment_equivalence_summary.csv")
    if not all(f"| {r.study} | {r.dataset} | {r.arm} | {r.fits} |" in t for r in eq.itertuples()):
        problems.append(("equivalence table",))

    counts["rows"] = (len(pred), len(stab), len(prov), len(cited), len(paths), len(required))
    counts["extra"] = (len(rob), len(interp), len(follow))
    return problems, counts


def main() -> int:
    os.chdir(ROOT)
    text = MAP.read_text(encoding="utf-8")
    problems, counts = validate(text)
    n_pred, n_stab, n_prov, n_cited, n_paths, n_req = counts["rows"]
    n_fields = sum(v for k, v in counts.items() if k not in ("stability", "rows", "extra", "interp", "follow_fields") and not k.startswith("rob_"))
    n_rob_fields = sum(v for k, v in counts.items() if k.startswith("rob_"))
    print(f"predictive rows {n_pred} ({n_fields} field checks), stability rows {n_stab}, provenance {n_prov}, "
          f"cited IDs {n_cited}, paths {n_paths}, required-wording checks {n_req}, robustness rows {counts['extra'][0]} "
          f"({n_rob_fields} field checks), interpretability rows {counts['extra'][1]}, follow-up rows {counts['extra'][2]} "
          f"({counts.get('follow_fields', 0)} field checks)")
    print("DISCREPANCIES:", problems if problems else 0)
    status = 1 if problems else 0
    if "--self-test" in sys.argv:
        row = next(l for l in text.splitlines() if l.startswith("| V1-H1-R2-auc |"))
        cells = row.split("|"); cells[6] = " +0.9999 "
        planted = text.replace(row, "|".join(cells), 1).replace(
            "significantly lower on Invoice R² on Online Retail II", "significantly lower on regression targets on Online Retail II")
        found, _ = validate(planted)
        expected = {("pred", "V1-H1-R2-auc", "delta"), ("required wording missing", "H2 corrected")}
        detected = expected <= set(found)
        print(f"self-test (in memory; no file written): planted 2 errors, detected = {detected}")
        status = status or (0 if detected else 1)
    return status


if __name__ == "__main__":
    sys.exit(main())

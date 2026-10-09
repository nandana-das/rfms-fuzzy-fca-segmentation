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
- every referenced repository path exists;
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
    pred, stab = table("### 1b.", "### 1c."), table("### 1c.", "## 2.")
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
    for _, r in stab.iterrows():
        a_lab, rest = r["Claim (data-derived wording)"].split(" vs ", 1); b_lab, _ = rest.rsplit(": ", 1)
        comp = f"{lab_rev[a_lab]} - {lab_rev[b_lab]}"
        matched = "-MA-" in r.ID or "-MB-" in r.ID; refit = r.ID.startswith(("STAB-A", "STAB-MA"))
        ds = r.Dataset.split(" (")[0]
        if refit:
            q = (ma if matched else sa); q = q[(q.dataset == ds) & (q.comparison == comp) & (q.measure == "core_jaccard")]
            lo_c, hi_c, p, fam = q.iloc[0].interval_low, q.iloc[0].interval_high, None, None
        else:
            sub = "all" if "all customers" in r.Dataset else "stable"; src = (mb if matched else sb)
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

    # 3. evidence-row references
    all_ids = list(pred.ID) + list(stab.ID)
    if len(all_ids) != len(set(all_ids)):
        problems.append(("duplicate IDs",))
    cited = set(re.findall(r"\b(?:V1|V2|CD)-[A-Z0-9]+-[A-Za-z0-9_\-]+", t.split("### 1b.")[0] + t[t.index("## 3."):]))
    missing = sorted(c for c in cited if c not in all_ids)
    if missing:
        problems.append(("cited IDs missing", missing))

    # 4. provenance (commit = oldest commit touching path; branch = first in lineage containing it; later commits)
    lineage = ["main", "experiment/optimized-fuzzy-fca", "experiment/v2-hybrid-fuzzy-fca", "experiment/cdnow-confirmation"]
    prov = table("## 2.", "Further plans")
    for _, r in prov.iterrows():
        path = r.Artifact.strip("`"); commit = r["Originating commit"].split("`")[1]
        commits = subprocess.run(["git", "log", "--format=%h", "--", path], capture_output=True, text=True).stdout.split()
        branch = next(b for b in lineage if subprocess.run(["git", "merge-base", "--is-ancestor", commits[-1], b]).returncode == 0)
        later = "no" if len(commits) == 1 else None
        if not (os.path.exists(path) and commits[-1] == commit and r["Originating branch"].strip("`") == branch
                and r["Modified after origin"] == later):
            problems.append(("provenance", path, commits, commit, branch))

    # 5. paths
    paths = set(re.findall(r"`((?:results|docs|scripts)/[^`*]+?)`", t))
    for pth in paths:
        if not os.path.exists(pth):
            problems.append(("path missing", pth))

    # 6. required wording and structure
    required = {
        "regeneration command": "python scripts/build_paper_evidence_map.py",
        "H2 corrected": "significantly lower on Invoice R² on Online Retail II, and on Spend and Invoice R² on CDNOW; on CDNOW it is significantly higher on AUC",
        "X1 corrected": "on all three metrics on Online Retail II (exploratory), on Spend and Invoice R² on CDNOW (confirmatory), and on Invoice R² only on Dunnhumby (exploratory)",
        "N1 isolated Kuznetsov result": "One pooled comparison is significant (Online Retail II AUC, at the floor, positive at only 1 of 5 origins)",
        "X2 dataset-specific": "Online Retail II (exploratory) and CDNOW (confirmatory), but not significantly on Dunnhumby",
        "X2 never pooled": "Never pool the three datasets into one superiority claim",
        "narrative separate per dataset": "Hybrid vs spline, reported separately per dataset; never pooled",
    }
    for name, text in required.items():
        if text not in t:
            problems.append(("required wording missing", name))
    eq = pd.read_csv("results/v2_stability/segment_equivalence_summary.csv")
    if not all(f"| {r.study} | {r.dataset} | {r.arm} | {r.fits} |" in t for r in eq.itertuples()):
        problems.append(("equivalence table",))

    counts["rows"] = (len(pred), len(stab), len(prov), len(cited), len(paths), len(required))
    return problems, counts


def main() -> int:
    os.chdir(ROOT)
    text = MAP.read_text(encoding="utf-8")
    problems, counts = validate(text)
    n_pred, n_stab, n_prov, n_cited, n_paths, n_req = counts["rows"]
    n_fields = sum(v for k, v in counts.items() if k not in ("stability", "rows"))
    print(f"predictive rows {n_pred} ({n_fields} field checks), stability rows {n_stab}, provenance {n_prov}, "
          f"cited IDs {n_cited}, paths {n_paths}, required-wording checks {n_req}")
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

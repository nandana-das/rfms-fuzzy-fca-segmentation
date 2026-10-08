# Final Contribution Framing (frozen, 2026-10-08)

The pre-audit framing (Methodology v4) is kept in `docs/superseded_2026-10-08/final_methodology_contribution_framing.md` and must not be used. Evidence: `docs/RESULTS_SUMMARY.md`. Method: `docs/METHODOLOGY_FINAL.md`.

## 1. Positioning

Rungruang et al. (2024, *Expert Systems with Applications* 237, 121449) built hierarchical RFM customer segmentation on a crisp binary formal context and proposed, as future work, representing RFM in a non-binary formal context. This study implements that proposal and evaluates it out of sample, against the crisp original and against strong non-FCA RFM baselines. It does not claim to introduce RFM-FCA.

## 2. Research question

To what extent does fuzzifying the RFM formal context change (a) the out-of-sample predictive adequacy and (b) the structural stability of FCA-based customer segmentation, compared with conventional crisp RFM-FCA and with strong non-FCA RFM baselines?

## 3. Hypotheses and status

| | Hypothesis | Status |
|---|---|---|
| H1 | Fuzzy RFM-FCA improves predictive performance relative to crisp RFM-FCA. | **Supported** under the rolling-origin evaluation (pooled, Holm-corrected significance on all metrics in both datasets; positive at every origin descriptively) |
| H2 | Fuzzy RFM-FCA achieves predictive performance comparable to strong nonlinear non-FCA RFM baselines. | **Inconclusive** (5/6 spline comparisons not significantly different; equivalence not established; Retail II Invoice R² significantly worse by 0.032) |
| H3 | Fuzzy RFM-FCA improves segmentation stability relative to crisp RFM-FCA. | **Not supported** (temporal gap consistent with a concept count and/or concept size explanation) |

## 4. Contributions (evidence-based)

1. **A fuzzy extension of RFM-FCA.** It uses train-fitted, tie-preserving quintile bands (occupancy reported), centroid-based piecewise-linear fuzzy memberships and a threshold-scaled concept context. Redundancy suppression with its empty-core safeguard is an implementation safeguard, not a novelty claim.
2. **A leakage-aware, multi-dataset temporal evaluation.** It uses rolling origins on the base paper's dataset (Online Retail II) and a reproduction dataset (Dunnhumby), and compares crisp FCA, fuzzy FCA and strong non-FCA RFM baselines under identical models and paired, multiplicity-corrected inference.
3. **Predictive evidence.**
   - Under the rolling-origin evaluation, fuzzification improves predictive adequacy over crisp RFM-FCA, with pooled, Holm-corrected significance on every metric in both datasets; the difference is positive at every origin, which is descriptive only.
   - It does *not* demonstrate superiority over a spline-based nonlinear RFM baseline. Five of six comparisons were not significantly different, equivalence was not established, and Retail II Invoice R² was significantly worse by 0.032. The improvement over crisp FCA cannot be attributed to the concept structure alone.
4. **Structural stability analysis.**
   - The expected stability advantage of fuzzy FCA is *not* supported: there is no significant difference on refits, and fuzzy segments are less stable over quarterly re-segmentation.
   - A pre-specified matched-count diagnostic removes the temporal gap at matched concept count. This supports a concept count and/or concept size explanation; it does not show concept count alone is causal.
   - Under every method, about half of a customer's concept profile changes from one quarter to the next.

**Ablation / negative result (not part of the proposed method or its contribution claims).** A canonical Kuznetsov stability filter was evaluated as an ablation and excluded. It gave no consistent predictive benefit, made segments markedly less stable, and its threshold acts as an absolute customer-count rule that depends on training-set size. This is reported as negative evidence.

## 5. Claims that must not be made

- Fuzzy RFM-FCA is universally or generally superior, or superior to strong nonlinear RFM models.
- FCA (the concept structure) causes the predictive improvement.
- Fuzzy RFM-FCA is more stable than crisp RFM-FCA.
- Fuzzy RFM-FCA is more interpretable. Interpretability was not measured.
- Kuznetsov stability, Kneedle pruning or hyperparameter optimization (Stage 1/2) improves the framework.
- Any result from the superseded protocol: dense-rank scoring, the untransformed raw-RFM comparison, the "10-split temporal CV", or the 98.08% repurchase figure. The correct Dunnhumby figure is 93.16% at the day-620 origin.
- The base paper's exact customer counts were reproduced. Only its 31 intents were recovered; 3 of 31 counts match.

## 6. One-sentence summary

Under a rolling-origin evaluation, fuzzifying the RFM formal context significantly improves the out-of-sample predictive adequacy of FCA-based segmentation over the crisp RFM-FCA of Rungruang et al. (2024) on both evaluated datasets, but it does not outperform a strong nonlinear RFM baseline and does not make segments more stable.

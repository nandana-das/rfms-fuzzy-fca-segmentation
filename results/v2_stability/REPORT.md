# v2 Part A — segment equivalence check

> Exploratory, post hoc. Verification, not a new stability experiment. Plan: `docs/V2_STABILITY_ANALYSIS_PLAN.md`.

**Result: ALL FITS IDENTICAL** (586 fit comparisons; 586 with identical concept-membership columns; 586 with identical core profiles; max |difference| = 0).

| study | dataset | arm | fits | concept_count_match | columns_identical | core_profiles_identical | max_abs_diff | k_min | k_max |
|---|---|---|---|---|---|---|---|---|---|
| A refit | Dunnhumby | crisp | 124 | 124 | 124 | 124 | 0.0000 | 51 | 56 |
| A refit | Dunnhumby | fuzzy | 124 | 124 | 124 | 124 | 0.0000 | 81 | 99 |
| A refit | Online Retail II | crisp | 155 | 155 | 155 | 155 | 0.0000 | 45 | 53 |
| A refit | Online Retail II | fuzzy | 155 | 155 | 155 | 155 | 0.0000 | 64 | 84 |
| B temporal | Dunnhumby | crisp | 6 | 6 | 6 | 6 | 0.0000 | 52 | 56 |
| B temporal | Dunnhumby | fuzzy | 6 | 6 | 6 | 6 | 0.0000 | 90 | 94 |
| B temporal | Online Retail II | crisp | 8 | 8 | 8 | 8 | 0.0000 | 47 | 50 |
| B temporal | Online Retail II | fuzzy | 8 | 8 | 8 | 8 | 0.0000 | 67 | 80 |

Runtime: 14.7 min.

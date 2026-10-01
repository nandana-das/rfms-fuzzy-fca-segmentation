# Canonical Kneedle Audit

This audit compares canonical `kneed.KneeLocator` with the project's Kneedle-inspired normalized max-distance-from-chord heuristic.

Canonical Kneedle is evaluated only on the support-versus-concept-rank curve. The stability-proxy distribution is reported descriptively and is not used as a second Kneedle pruning curve.

## 1. Experimental configuration

- Candidate concepts: same fuzzy closed-concept candidates used by the existing pipeline.

- Minimum support: `0.04`.

- Canonical implementation: `kneed.KneeLocator`.

- Curve: `convex`.

- Direction: `decreasing`.

- Interpolation: `interp1d`.

- Online mode: `False`.

- Sensitivity values: `S = 0.1, 0.5, 1.0, 2.0`.

## 2. Support-knee comparison

| Dataset                            | Total_Concepts | Method                           | S        | Support_Knee_Rank | Support_Threshold | Retained_By_Support | Delta_Support_vs_Heuristic |
| ---------------------------------- | -------------- | -------------------------------- | -------- | ----------------- | ----------------- | ------------------- | -------------------------- |
| Dunnhumby Observation (Days 1-620) | 502            | Kneedle-Inspired Chord Heuristic |          | 43                | 0.212885          | 43                  | 0.000000                   |
| Dunnhumby Observation (Days 1-620) | 502            | Canonical KneeLocator            | 0.100000 | 20                | 0.287715          | 20                  | 0.074830                   |
| Dunnhumby Observation (Days 1-620) | 502            | Canonical KneeLocator            | 0.500000 | 20                | 0.287715          | 20                  | 0.074830                   |
| Dunnhumby Observation (Days 1-620) | 502            | Canonical KneeLocator            | 1.000000 | 45                | 0.210484          | 45                  | -0.002401                  |
| Dunnhumby Observation (Days 1-620) | 502            | Canonical KneeLocator            | 2.000000 | 49                | 0.204882          | 49                  | -0.008003                  |
| UK Online Retail II (Full Cohort)  | 359            | Kneedle-Inspired Chord Heuristic |          | 30                | 0.218101          | 30                  | 0.000000                   |
| UK Online Retail II (Full Cohort)  | 359            | Canonical KneeLocator            | 0.100000 | 16                | 0.283770          | 16                  | 0.065669                   |
| UK Online Retail II (Full Cohort)  | 359            | Canonical KneeLocator            | 0.500000 | 16                | 0.283770          | 16                  | 0.065669                   |
| UK Online Retail II (Full Cohort)  | 359            | Canonical KneeLocator            | 1.000000 | 16                | 0.283770          | 16                  | 0.065669                   |
| UK Online Retail II (Full Cohort)  | 359            | Canonical KneeLocator            | 2.000000 | 16                | 0.283770          | 16                  | 0.065669                   |


## 3. Stability-proxy distribution

The stability proxy is not treated as an independent Kneedle curve. Its distribution is reported to assess whether such an application would have a meaningful interpretation.

| Dataset                            | Total_Concepts | Heuristic_Rank | Heuristic_Support | Heuristic_Max_Distance | stability_min | stability_q25 | stability_median | stability_q75 | stability_max | stability_ge_0_95 | stability_ge_0_90 | stability_ge_0_80 |
| ---------------------------------- | -------------- | -------------- | ----------------- | ---------------------- | ------------- | ------------- | ---------------- | ------------- | ------------- | ----------------- | ----------------- | ----------------- |
| Dunnhumby Observation (Days 1-620) | 502            | 43             | 0.212885          | 0.500897               | 0.803922      | 0.961538      | 0.976517         | 0.985055      | 0.995327      | 426               | 495               | 502               |
| UK Online Retail II (Full Cohort)  | 359            | 30             | 0.218101          | 0.509105               | 0.956693      | 0.983406      | 0.991039         | 0.994955      | 0.998435      | 359               | 359               | 359               |


## 4. Interpretation

Canonical Kneedle and the project's chord heuristic are compared as alternative support-knee detectors. Differences in the selected support threshold are reported empirically and are not interpreted as evidence that one method is universally superior.

The chord heuristic selects the global maximum perpendicular distance from the endpoint chord, whereas canonical Kneedle uses its normalized-difference curve and sensitivity-controlled knee detection. Therefore, the two methods are related but are not mathematically identical.

The stability-proxy distribution is not used for canonical Kneedle pruning because it does not provide a comparably justified support-versus-rank curve. This avoids conflating a concept-quality statistic with the knee-detection curve itself.

## 5. Methodological status

This audit does not replace the existing pruning procedure. Canonical Kneedle should only be promoted into the final pipeline after its downstream predictive and structural consequences have been evaluated under the same leakage-free protocol.

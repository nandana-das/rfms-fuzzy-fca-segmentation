# LRFM extension audit â€” frozen results v1

**Freeze recorded:** 2026-10-10
**Branch:** `experiment/lrfm-extension`
**Status:** exploratory analysis; results frozen before manuscript changes

This note versions the audit record for the completed LRFM extension. It does
not replace or modify the result write-up in
[`LRFM_EXTENSION_RESULTS.md`](./LRFM_EXTENSION_RESULTS.md), any result CSV, or
any checkpoint.

## Frozen state

- Checkpoint manifest status: `complete`
- Checkpoint schema version: `1`
- Unique `(dataset, origin, seed)` entries: `65`
- Complete entries: `65`
- Checkpoint CSV/JSON pairs present: `65/65`
- Seeds: `0, 1, 2, 3, 4`
- Implementation fingerprint:
  `c5a7c49940d247eec7f77114aa581366a50484470de8d0b2776d8b3622185d08`
- Paired-comparison rows: `45`
- Expected comparison keys: `3 datasets Ã— 5 comparisons Ã— 3 metrics`
- Duplicate comparison keys: `0`

The implementation fingerprint covers the LRFM driver, the frozen v1/v2/CDNOW
helpers used by the driver, the modelling/feature helpers, concept pruning,
the locked protocol configuration, and the inference configuration.

## Frozen file hashes

SHA-256 hashes were computed from the files present at freeze time:

| File | SHA-256 |
|---|---|
| `results/lrfm_extension/paired_comparisons.csv` | `3568d69e66dec40451396f71eefaf9603d383cb2782fc9d85a17aa840ba0c466` |
| `results/lrfm_extension/per_origin_metrics.csv` | `3287e8eedd9059cde9cdddabae94e7fe5708d1caca4b5ada36d1cbf6c29d41d2` |
| `results/lrfm_extension/concept_counts_per_fold.csv` | `34558b529614c07af78a500fb72ea6569529e66e78b21573b1d5bb4b826e6cc2` |
| `results/lrfm_extension/l_distribution.csv` | `c932a57e480b388bc74256c0f40159149dab58ae2d0c52ba8ac750f567517426` |
| `results/lrfm_extension/l_band_occupancy.csv` | `172f45980ffc85a66dc1f53f3364b78b2f8384ee1b96aeece9d682e695de3d68` |
| `results/lrfm_extension/run_parameters.json` | `818dc5b36fb02e4455d2e89e4d8520fa644c440d7a3c1bb1954fd6c07ad03a4e` |
| `results/lrfm_extension/checkpoints/manifest.json` | `5ca3f7fca051642731b6e87820f75941bbd5c3e3e728b8f5d91e70ae92e07856` |
| `docs/LRFM_EXTENSION_RESULTS.md` | `636fc0b86d10cf6d924c9a731cb5f24bca5985291a9fd39436d7bd889f93b223` |

The checkpoint directory contains the 65 manifest-listed CSV/JSON pairs. The
manifest hash above, together with the implementation fingerprint and the
manifest entry list, identifies the frozen checkpoint set.

## Protocol audit record

- Five CV seeds: `0â€“4`.
- Customer-clustered paired bootstrap: `10,000` draws per seed.
- Bootstrap population: dataset-specific customer union.
- Within each draw, the same customer weights are applied to both arms and to
  every pooled origin in that dataset.
- Origin-level arm differences are averaged within each draw.
- The five seed draw sets are combined into `50,000` draws.
- Confidence intervals: 2.5th and 97.5th percentiles.
- Two-sided raw p-value:
  `min(1, max(2 * min(mean(draws <= 0), mean(draws >= 0)), 1 / 50,000))`.
- No plus-one correction is used; zero-valued draws count in both tails.
- Holm correction is applied separately to 15 raw p-values per dataset.

## Audit limitation

The realized bootstrap draw arrays were not persisted. The completed checkpoints
persist the per-customer out-of-fold predictions and metadata needed to rerun
inference, but they do not preserve the exact random bootstrap draws used for
the frozen tables. Consequently, the exact draw-level confidence intervals and
p-values cannot be independently reconstructed from the frozen artifacts alone
without rerunning the inference stage. The existing CSV outputs, checkpoint
schema, manifest, and implementation fingerprint were preserved unchanged.

No five-hour model-fitting run was repeated to recreate these tables.

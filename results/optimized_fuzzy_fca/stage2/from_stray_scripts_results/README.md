# Files recovered from the stray scripts/results/ folder

Historical and invalidated (Stage 2; see `docs/AUDIT_ERRATA.md` §3). A path bug in three Stage 2 scripts wrote outputs to `scripts/results/`. These are the files from that folder with no identical copy in `results/optimized_fuzzy_fca/stage2/`:

- `seed42_train_stability_loss_distribution.csv`: the only copy.
- `stage2_seed_results.csv`, `stage2_summary.json`: an earlier Stage 2 run that differs from the copies one level up.

The folder's other files were byte-identical to the copies one level up and were removed with the folder (2026-10-08).

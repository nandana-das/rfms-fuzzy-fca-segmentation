# Second CV seed: FCA+T vs spline+T

Criteria from `docs/TEMPORAL_SECOND_SEED_PLAN.md` (locked before the run; hashes in `PLAN_LOCK.txt`). Delta = fuzzy_rfm_t - spline_log_rfm_t. Practical threshold 0.002.

| dataset | metric | Δ seed 0 | Δ seed 1 | mean | seed 1 − seed 0 | origins same sign (seed 1) | Holm p seed 0 | Holm p seed 1 | verdict |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Online Retail II | auc | +0.0040 | +0.0037 | +0.0039 | -0.0004 | 5/5 | 0.0060 | 0.0015 | replicates |
| Online Retail II | spend_r2 | +0.0090 | +0.0098 | +0.0094 | +0.0008 | 5/5 | 0.0060 | 0.0015 | replicates |
| Online Retail II | invoice_r2 | -0.0010 | -0.0004 | -0.0007 | +0.0006 | 2/5 | 1.0000 | 0.9140 | null cell: consistent |
| Dunnhumby | auc | -0.0079 | -0.0092 | -0.0085 | -0.0013 | 4/4 | 0.0080 | 0.0015 | replicates |
| Dunnhumby | spend_r2 | -0.0099 | -0.0078 | -0.0088 | +0.0021 | 3/4 | 0.0060 | 0.0040 | replicates |
| Dunnhumby | invoice_r2 | -0.0047 | -0.0045 | -0.0046 | +0.0002 | 3/4 | 0.1050 | 0.0160 | replicates |

## Claim-level decisions

- Retail II FCA-specific advantage (requires AUC and Spend R² to replicate): **retained**.
- Dunnhumby spline advantage (requires ≥ 2 of 3 cells to replicate): **retained**.

## Scope of this result

- 'Replicates' means the seed-0 direction and practical magnitude held under a second fold assignment (CV seed 1) on the same customers, origins, and outcomes. It shows stability across these two fold assignments only.
- It is not an independent replication: no new data, customers, time periods, or datasets were used, and it supports no general claim about other customer populations.
- Two seeds do not estimate fold-assignment variance.
- Holm p-values are descriptive (3 tests per dataset per seed) and were not the decision criterion.
- FCA+T uses ~160–170 features versus 29 for the additive spline+T; the Retail II advantage may reflect interactions rather than FCA specifically.

Command: `python scripts/temporal_second_seed_evaluate.py`

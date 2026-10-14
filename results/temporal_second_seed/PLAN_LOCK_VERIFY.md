# Verifying PLAN_LOCK.txt

`PLAN_LOCK.txt` hashes the working-copy files at lock time (2026-10-14T02:54:43+05:30).
`scripts/temporal_nonfca_baseline.py` had CRLF line endings in the working copy, while git stores
LF, so the committed blob hashes differently. The plan file was LF and matches directly.

Content is identical apart from line endings. To check against the lock:

```bash
git show 762a45d:scripts/temporal_nonfca_baseline.py | sed 's/$/\r/' | sha256sum   # d80870...
git show 95fea61:docs/TEMPORAL_SECOND_SEED_PLAN.md | sha256sum                     # a91486...
```

The script was not modified between the lock and the seed-1 run (the working-copy hash still
matched the lock after the run).

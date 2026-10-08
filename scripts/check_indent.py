import re, ast

path = "scripts/kuznetsov_pruning_experiment.py"
lines = open(path, encoding="utf-8").read().splitlines()

mi = next(i for i,l in enumerate(lines) if 'k_by_seed = k_df.groupby' in l)
print("k_by_seed at line idx", mi, "len", len(lines[mi]))
print("indent of this line:", repr(lines[mi][:len(lines[mi])-len(lines[mi].lstrip())]))
for j in range(max(0,mi-6), min(len(lines), mi+4)):
    print(j+1, repr(lines[j][:40]))

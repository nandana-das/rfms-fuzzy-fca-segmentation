#!/usr/bin/env python
from pathlib import Path
import pandas as pd, json

p = Path('results/optimized_fuzzy_fca/stage1_multiseed_seed_results.csv')
df = pd.read_csv(p)
assert len(df) == 10, len(df)
assert set(df['outer_seed']) == set(range(1000,1010))

for i,r in df.iterrows():
    sel = r['selected_outer_metrics']
    bl = r['baseline_outer_metrics']
    # tolerate python dict repr in CSV by tolerant parse; fail if structure wrong
    for label,s in [('sel',sel),('bl',bl)]:
        assert isinstance(s, str) and s.startswith('{') and s.endswith('}')
    d_sel = eval(sel, {'__builtins__':{}})
    d_bl = eval(bl, {'__builtins__':{}})
    assert set(d_sel)=={'roc_auc','pr_auc','spend_r2','invoice_r2'}
    assert set(d_bl)=={'roc_auc','pr_auc','spend_r2','invoice_r2'}

agg_path = Path('results/optimized_fuzzy_fca/stage1_multiseed_summary.json')
agg = json.loads(agg_path.read_text())
assert agg['selection_frequency']['5/5/3']['count'] == 9
assert agg['selection_frequency']['4/4/5']['count'] == 1
print('FULL VALIDATION OK')

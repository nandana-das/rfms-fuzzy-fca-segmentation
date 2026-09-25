"""
STEP 8: Kneedle threshold sensitivity analysis (mandatory per both external audits)
Tests surviving-concept-count response to 0.9x / 1.0x / 1.1x perturbation of the
Kneedle-derived supp_min* and theta* thresholds, independently and jointly, for
both Olist and Online Retail II lattices. If counts remain reasonably stable
under +-10% perturbation, the pruning choice is not knife-edge/arbitrary.
"""
import pandas as pd
import numpy as np
import time
from project_paths import PROCESSED, RESULTS, ensure_output_dirs

ensure_output_dirs()
OUT = PROCESSED
RES = RESULTS


def kneedle_threshold(values_sorted_desc):
    y = np.array(values_sorted_desc, dtype=float)
    n = len(y)
    if n < 3:
        return y[-1], n - 1
    x = np.linspace(0, 1, n)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1 = x[0], y_norm[0]
    x2, y2 = x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    knee_idx = np.argmax(dist)
    return y[knee_idx], knee_idx


def stability_proxy(extent, score_tuples):
    n = len(extent)
    if n <= 1:
        return 1.0
    idx = np.fromiter(extent, dtype=np.int64, count=n)
    profiles = score_tuples[idx]
    n_unique = len(np.unique(profiles, axis=0))
    return 1 - (n_unique / n)


def analyze(name, closed_df, supp_min_star, theta_star):
    print(f'\n=== {name} ===')
    print(f'Base thresholds: supp_min*={supp_min_star:.4f}, theta*={theta_star:.4f}')
    print(f'Total closed concepts available: {len(closed_df)}')

    factors = [0.9, 1.0, 1.1]
    rows = []
    for sf in factors:
        for tf in factors:
            s_thr = supp_min_star * sf
            t_thr = theta_star * tf
            # theta is a similarity/stability score in [0,1]; clip perturbation to valid range
            t_thr = min(t_thr, 1.0)
            n_surv = int(((closed_df['support'] >= s_thr) & (closed_df['stability_approx'] >= t_thr)).sum())
            rows.append({'supp_factor': sf, 'theta_factor': tf, 'supp_thr': s_thr,
                         'theta_thr': t_thr, 'surviving_concepts': n_surv})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    base_n = int(df[(df.supp_factor == 1.0) & (df.theta_factor == 1.0)]['surviving_concepts'].iloc[0])
    diag = df[df.supp_factor == df.theta_factor]  # joint perturbation (0.9,0.9),(1,1),(1.1,1.1)
    print(f'\nJoint perturbation (both thresholds scaled together):')
    for _, r in diag.iterrows():
        pct = 100 * (r['surviving_concepts'] - base_n) / base_n if base_n else float('nan')
        print(f"  {r['supp_factor']:.1f}x: {int(r['surviving_concepts'])} concepts ({pct:+.1f}% vs base)")

    # support-only and theta-only marginal sensitivity
    supp_only = df[df.theta_factor == 1.0]
    theta_only = df[df.supp_factor == 1.0]
    print(f'\nSupport-threshold-only sensitivity (theta fixed at theta*):')
    for _, r in supp_only.iterrows():
        pct = 100 * (r['surviving_concepts'] - base_n) / base_n if base_n else float('nan')
        print(f"  supp {r['supp_factor']:.1f}x ({r['supp_thr']:.4f}): {int(r['surviving_concepts'])} concepts ({pct:+.1f}%)")
    print(f'\nStability-threshold-only sensitivity (support fixed at supp_min*):')
    for _, r in theta_only.iterrows():
        pct = 100 * (r['surviving_concepts'] - base_n) / base_n if base_n else float('nan')
        print(f"  theta {r['theta_factor']:.1f}x ({r['theta_thr']:.4f}): {int(r['surviving_concepts'])} concepts ({pct:+.1f}%)")

    # additive perturbation for theta (multiplicative saturates at the theta<=1.0 ceiling near theta*~0.93-0.99,
    # which produces a spurious cliff to 0 that reflects the bound, not real instability)
    print(f'\nAdditive stability-threshold sensitivity (theta* +/- 0.02, +/- 0.05; support fixed at supp_min*):')
    add_rows = []
    for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
        t_thr = min(max(theta_star + delta, 0.0), 1.0)
        n_surv = int(((closed_df['support'] >= supp_min_star) & (closed_df['stability_approx'] >= t_thr)).sum())
        pct = 100 * (n_surv - base_n) / base_n if base_n else float('nan')
        print(f'  theta*{delta:+.2f} ({t_thr:.4f}): {n_surv} concepts ({pct:+.1f}%)')
        add_rows.append({'theta_delta': delta, 'theta_thr': t_thr, 'surviving_concepts': n_surv})
    pd.DataFrame(add_rows).to_csv(RES + f'kneedle_sensitivity_{name.lower().replace(" ", "_")}_theta_additive.csv', index=False)

    df.to_csv(RES + f'kneedle_sensitivity_{name.lower().replace(" ", "_")}.csv', index=False)
    return df, base_n


# ---------- OLIST ----------
t0 = time.time()
agg = pd.read_csv(OUT + 'olist_rfms_features.csv')
closed_olist = pd.read_pickle(RES + 'fuzzy_concepts_raw.pkl')  # uncapped, 1369 concepts, no stability yet
fuzzy_df = pd.read_pickle(RES + 'fuzzy_membership_matrix.pkl')
attr_cols = list(fuzzy_df.columns)
mu_matrix = fuzzy_df[attr_cols].values
L = [0.3, 0.5, 0.7]
attr_to_extent = {}
for j, col in enumerate(attr_cols):
    v = mu_matrix[:, j]
    for l in L:
        attr_to_extent[f'{col}@{l}'] = set(np.where(v >= l)[0])


def extent_of(itemset):
    sets = [attr_to_extent[a] for a in itemset]
    sets.sort(key=len)
    result = sets[0]
    for s in sets[1:]:
        result = result & s
        if not result:
            break
    return result


closed_olist['extent'] = closed_olist['itemsets'].apply(extent_of)
score_tuples_olist = agg[['r_score', 'f_score', 'm_score', 's_score']].values
closed_olist['stability_approx'] = closed_olist['extent'].apply(lambda e: stability_proxy(e, score_tuples_olist))
print(f'Olist extents+stability recomputed [{time.time()-t0:.1f}s]')

supp_sorted = np.sort(closed_olist['support'].values)[::-1]
supp_min_star_olist, _ = kneedle_threshold(supp_sorted)
stab_sorted = np.sort(closed_olist['stability_approx'].values)[::-1]
theta_star_olist, _ = kneedle_threshold(stab_sorted)

df_olist, base_olist = analyze('Olist', closed_olist, supp_min_star_olist, theta_star_olist)

# ---------- RETAIL II ----------
t0 = time.time()
closed_r2 = pd.read_pickle(RES + 'retail2_fuzzy_concepts_raw.pkl')  # has extent, but not stability_approx (that
# column was computed after this pkl was saved in step7) -- recompute here the same way step7 does
score_tuples_r2 = pd.read_csv(OUT + 'retail2_rfm_features.csv')[['r_score', 'f_score', 'm_score']].values
if 'extent' not in closed_r2.columns:
    raise RuntimeError('retail2_fuzzy_concepts_raw.pkl missing extent column')
closed_r2['stability_approx'] = closed_r2['extent'].apply(lambda e: stability_proxy(e, score_tuples_r2))
print(f'Retail II stability recomputed [{time.time()-t0:.1f}s]')

supp_sorted_r2 = np.sort(closed_r2['support'].values)[::-1]
supp_min_star_r2, _ = kneedle_threshold(supp_sorted_r2)
stab_sorted_r2 = np.sort(closed_r2['stability_approx'].values)[::-1]
theta_star_r2, _ = kneedle_threshold(stab_sorted_r2)

df_r2, base_r2 = analyze('Retail II', closed_r2, supp_min_star_r2, theta_star_r2)

print('\n\n=== SUMMARY ===')
print(f'Olist:     base={base_olist} concepts; range under joint 0.9x-1.1x perturbation: '
      f"{df_olist[df_olist.supp_factor==df_olist.theta_factor]['surviving_concepts'].min()}-"
      f"{df_olist[df_olist.supp_factor==df_olist.theta_factor]['surviving_concepts'].max()}")
print(f'Retail II: base={base_r2} concepts; range under joint 0.9x-1.1x perturbation: '
      f"{df_r2[df_r2.supp_factor==df_r2.theta_factor]['surviving_concepts'].min()}-"
      f"{df_r2[df_r2.supp_factor==df_r2.theta_factor]['surviving_concepts'].max()}")
print('\nSaved: kneedle_sensitivity_olist.csv, kneedle_sensitivity_retail_ii.csv')

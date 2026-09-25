"""
STEP 1: Data cleaning + RFMS feature engineering (methodology Sec 1-2)
- Join orders<->customers via customer_unique_id (NOT customer_id)
- Filter delivered orders, valid payments
- Aggregate R, n_j, M, S per customer
- Composite F* via variance-ratio grid search (Sec 2.3)
- Dense-rank fractional scoring 1-5 (Sec 1.2 Fix B)
"""
import pandas as pd
import numpy as np
from project_paths import OLIST_RAW, PROCESSED, ensure_output_dirs

ensure_output_dirs()
RAW = OLIST_RAW
OUT = PROCESSED

orders = pd.read_csv(RAW + 'olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date'])
customers = pd.read_csv(RAW + 'olist_customers_dataset.csv')
payments = pd.read_csv(RAW + 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(RAW + 'olist_order_reviews_dataset.csv')
items = pd.read_csv(RAW + 'olist_order_items_dataset.csv')

print(f'Raw orders: {len(orders)} | Raw customers (order-scoped id): {customers.customer_id.nunique()} '
      f'| True unique customers: {customers.customer_unique_id.nunique()}')

# --- filter delivered, non-null purchase date ---
orders = orders[orders['order_status'] == 'delivered'].copy()
orders = orders.dropna(subset=['order_purchase_timestamp'])
print(f'After delivered+nonnull filter: {len(orders)} orders')

# --- join via customer_unique_id (critical fix) ---
orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

# --- payments: sum to order level, drop null/zero ---
pay_agg = payments.groupby('order_id', as_index=False)['payment_value'].sum()
pay_agg = pay_agg.dropna(subset=['payment_value'])
pay_agg = pay_agg[pay_agg['payment_value'] > 0]
orders_pay = orders.merge(pay_agg, on='order_id', how='inner')
print(f'After valid-payment join: {len(orders_pay)} orders')

# --- item qty per order ---
item_qty = items.groupby('order_id').size().rename('item_qty').reset_index()
orders_full = orders_pay.merge(item_qty, on='order_id', how='left')
orders_full['item_qty'] = orders_full['item_qty'].fillna(1)

# --- reviews: mean score per order ---
rev_agg = reviews.groupby('order_id', as_index=False)['review_score'].mean()
orders_full = orders_full.merge(rev_agg, on='order_id', how='left')

print(f'Orders after full cleaning/join: {orders_full.shape}')
print(f'Unique customers: {orders_full.customer_unique_id.nunique()}')

# --- per-customer aggregation ---
T_ref = orders_full['order_purchase_timestamp'].max()
print(f'Reference date T_ref = {T_ref}')

orders_full['log_qty'] = np.log1p(orders_full['item_qty'])

agg = orders_full.groupby('customer_unique_id').agg(
    last_purchase=('order_purchase_timestamp', 'max'),
    n_orders=('order_id', 'nunique'),
    M=('payment_value', 'sum'),
    S=('review_score', 'mean'),
).reset_index()

log_qty_sum = orders_full.groupby('customer_unique_id')['log_qty'].sum().rename('log_qty_sum').reset_index()
agg = agg.merge(log_qty_sum, on='customer_unique_id', how='left')

agg['R'] = (T_ref - agg['last_purchase']).dt.days
agg['repeat'] = (agg['n_orders'] > 1).astype(int)
n_missing_review = agg['S'].isna().sum()
agg['S'] = agg['S'].fillna(agg['S'].median())
print(f'Customers with missing review score (median-imputed): {n_missing_review}')

m = len(agg)
print(f'\nTotal unique customers with valid orders+payment: {m}')
print(f'Repeat buyer rate: {agg["repeat"].mean():.4f}')
print(f'Single-order rate: {(agg["n_orders"]==1).mean():.4f}')

# --- composite purchase-intensity index (F*) weight fitting via grid search (Sec 2.3, REVISED) ---
# REVISED objective: the original Var(F*)/Var(n_orders) criterion is degenerate -- it is trivially
# maximized by whichever raw component has the highest variance alone, so it never produces a genuine
# blend and gives no evidence the composite actually reduces score-band degeneracy. Fixed to directly
# target the stated goal instead: maximize the Shannon entropy of the resulting 5-bin quintile
# histogram (i.e. reward weight combinations that spread customers across bands more evenly).
# NOTE: renamed from "composite frequency" to "composite purchase-intensity index" -- an audit
# correctly flagged that calling this quantity "frequency" is misleading once beta dominates, since it
# then reduces to cumulative log item-row count, not purchase-event count. n_orders remains available
# as the literal frequency variable and is reported separately for transparency.
def dense_rank_score_raw(series, invert=False):
    ranks = series.rank(method='dense')
    score = np.ceil(5 * ranks / ranks.max()).astype(int).clip(1, 5)
    return 6 - score if invert else score

def band_entropy(alpha, beta, gamma, df):
    Fcand = alpha * df['n_orders'] + beta * df['log_qty_sum'] + gamma * df['repeat']
    if Fcand.nunique() <= 1:
        return -np.inf
    scores = dense_rank_score_raw(Fcand)
    counts = scores.value_counts(normalize=True)
    p = counts.values
    return -np.sum(p * np.log(p + 1e-12))  # Shannon entropy, max = ln(5) at perfectly even bands

best = None
grid = np.arange(0, 1.05, 0.05)
for a in grid:
    for b in grid:
        g = round(1 - a - b, 2)
        if g < -1e-9 or g > 1 + 1e-9:
            continue
        g = max(0, g)
        ent = band_entropy(a, b, g, agg)
        if best is None or ent > best[0]:
            best = (ent, round(a, 2), round(b, 2), g)

ent, alpha, beta, gamma = best
max_entropy = np.log(5)
print(f'\nBest F* (purchase-intensity index) weights: alpha={alpha} beta={beta} gamma={gamma} '
      f'(band entropy={ent:.4f} / max possible={max_entropy:.4f}, {ent/max_entropy:.1%} of ideal spread)')
agg['F_star'] = alpha * agg['n_orders'] + beta * agg['log_qty_sum'] + gamma * agg['repeat']

# --- dense-rank fractional scoring (Sec 1.2 Fix B) ---
# Denominator is max(dense_rank) = number of DISTINCT values K, NOT total customer count m.
# (An earlier written methodology draft stated the denominator as m; that was a documentation error,
# not a code error -- using m would collapse nearly all scores to band 1 whenever K << m, which
# contradicts the well-spread r_score/m_score distributions actually observed. K is correct and is
# what this implementation uses throughout.)
def dense_rank_score(series, invert=False):
    ranks = series.rank(method='dense')
    max_rank = ranks.max()
    score = np.ceil(5 * ranks / max_rank).astype(int).clip(1, 5)
    if invert:
        score = 6 - score
    return score

agg['r_score'] = dense_rank_score(agg['R'], invert=True)
agg['f_score'] = dense_rank_score(agg['F_star'], invert=False)
agg['m_score'] = dense_rank_score(agg['M'], invert=False)
agg['s_score'] = dense_rank_score(agg['S'], invert=False)

print('\nScore distributions:')
for col in ['r_score', 'f_score', 'm_score', 's_score']:
    print(f'  {col}:', agg[col].value_counts().sort_index().to_dict())

print('\n*** KEY FINDING: F* redefinition alone insufficient to fix sparsity (expected) ***')
print(f'  f_score=1 share: {(agg["f_score"]==1).mean():.4f}  (near-degenerate)')

# --- diagnostic requested by review: does F* retain continuous spread WITHIN the f_score=1 group? ---
# (This does not by itself prove fuzzy membership "fixes" sparsity -- that claim is validated
# separately in Step 2 by checking whether these customers get non-trivial membership in bands 2+.
# This only checks whether the raw signal entering Step 2 has anything left to recover.)
f1_mask = agg['f_score'] == 1
f1_fstar = agg.loc[f1_mask, 'F_star']
print(f'\n  Within f_score=1 group (n={f1_mask.sum()}): F_star range [{f1_fstar.min():.3f}, {f1_fstar.max():.3f}], '
      f'std={f1_fstar.std():.3f}, unique values={f1_fstar.nunique()}')
print(f'  -> {"Non-trivial spread remains" if f1_fstar.nunique() > 1 else "NO spread -- fuzzy smoothing cannot help"} '
      f'within the degenerate band; fuzzy membership recovery is checked quantitatively in Step 2.')

agg.to_csv(OUT + 'olist_rfms_features.csv', index=False)
print(f'\nSaved: {OUT}olist_rfms_features.csv  ({agg.shape})')

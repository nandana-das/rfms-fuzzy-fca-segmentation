"""
Step 1-5 of Algorithm 2: Olist RFMS feature engineering
- Join via customer_unique_id (fix vs naive customer_id join)
- Filter delivered orders only, drop nulls
- Compute R, n_j (order count), item qty list, M, S
- Compute composite F* (weighted engagement frequency)
- Dense-rank fractional scoring (r,f,mu,s) in [1,5]
"""
import pandas as pd
import numpy as np

DATA = 'olist_data/'

orders = pd.read_csv(DATA + 'olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date'])
customers = pd.read_csv(DATA + 'olist_customers_dataset.csv')
payments = pd.read_csv(DATA + 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(DATA + 'olist_order_reviews_dataset.csv')
items = pd.read_csv(DATA + 'olist_order_items_dataset.csv')

# --- Step 2: filter delivered only, drop nulls ---
orders = orders[orders['order_status'] == 'delivered'].copy()
orders = orders.dropna(subset=['order_purchase_timestamp'])

# --- Step 1: join via customer_unique_id ---
orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

# payments: aggregate to order level first (multiple payment rows per order possible)
pay_agg = payments.groupby('order_id', as_index=False)['payment_value'].sum()
pay_agg = pay_agg.dropna(subset=['payment_value'])
pay_agg = pay_agg[pay_agg['payment_value'] > 0]

orders_pay = orders.merge(pay_agg, on='order_id', how='inner')  # drop orders w/ null/zero payment

# item quantity per order (Olist has no explicit qty col; each row = 1 unit, so qty = row count per order_id+product)
item_qty = items.groupby('order_id').size().rename('item_qty').reset_index()
orders_full = orders_pay.merge(item_qty, on='order_id', how='left')
orders_full['item_qty'] = orders_full['item_qty'].fillna(1)

# reviews: mean score per order (rare duplicates)
rev_agg = reviews.groupby('order_id', as_index=False)['review_score'].mean()
orders_full = orders_full.merge(rev_agg, on='order_id', how='left')

print('Orders after cleaning/join:', orders_full.shape)
print('Unique customers:', orders_full.customer_unique_id.nunique())

# --- Step 3: per-customer aggregation ---
T_ref = orders_full['order_purchase_timestamp'].max()

agg = orders_full.groupby('customer_unique_id').agg(
    last_purchase=('order_purchase_timestamp', 'max'),
    n_orders=('order_id', 'nunique'),
    M=('payment_value', 'sum'),
    S=('review_score', 'mean'),
    total_items=('item_qty', 'sum'),
).reset_index()

agg['R'] = (T_ref - agg['last_purchase']).dt.days
agg['repeat'] = (agg['n_orders'] > 1).astype(int)

# sum(log(1+qty)) per customer -- approximate via total_items since qty per order not preserved after groupby;
# recompute properly at order level then sum
orders_full['log_qty'] = np.log1p(orders_full['item_qty'])
log_qty_sum = orders_full.groupby('customer_unique_id')['log_qty'].sum().rename('log_qty_sum').reset_index()
agg = agg.merge(log_qty_sum, on='customer_unique_id', how='left')

agg['S'] = agg['S'].fillna(agg['S'].median())  # missing reviews -> median impute, flag in text

m = len(agg)
print('Total unique customers with valid orders+payment:', m)
print('Repeat buyer rate:', agg['repeat'].mean())

# --- Step 4: composite F* (weights via grid search, Sec 2.3) ---
def variance_ratio(alpha, beta, gamma, df):
    Fstar = alpha * df['n_orders'] + beta * df['log_qty_sum'] + gamma * df['repeat']
    return Fstar.var() / df['n_orders'].var(), Fstar

best = None
grid = np.arange(0, 1.05, 0.05)
for a in grid:
    for b in grid:
        g = 1 - a - b
        if g < 0 or g > 1:
            continue
        ratio, _ = variance_ratio(a, b, g, agg)
        if best is None or ratio > best[0]:
            best = (ratio, a, b, g)

ratio, alpha, beta, gamma = best
print(f'Best weights: alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} (variance ratio={ratio:.3f})')
agg['F_star'] = alpha * agg['n_orders'] + beta * agg['log_qty_sum'] + gamma * agg['repeat']

# --- Step 5: dense-rank fractional scoring ---
def dense_rank_score(series, invert=False):
    ranks = series.rank(method='dense')
    max_rank = ranks.max()
    score = np.ceil(5 * ranks / max_rank).astype(int)
    score = score.clip(1, 5)
    if invert:
        score = 6 - score
    return score

agg['r_score'] = dense_rank_score(agg['R'], invert=True)
agg['f_score'] = dense_rank_score(agg['F_star'], invert=False)
agg['m_score'] = dense_rank_score(agg['M'], invert=False)
agg['s_score'] = dense_rank_score(agg['S'], invert=False)

print(agg[['r_score', 'f_score', 'm_score', 's_score']].apply(pd.Series.value_counts).fillna(0))

agg.to_csv('/home/claude/olist_rfms_features.csv', index=False)
print('\nSaved: olist_rfms_features.csv')
print(agg.head(10)[['customer_unique_id','R','n_orders','M','S','F_star','r_score','f_score','m_score','s_score']])

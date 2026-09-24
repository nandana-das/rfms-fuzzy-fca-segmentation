# Step 1 diagnostics

Reference date: 2018-10-17T17:30:18

## Row and customer counts

- Input orders: 99,441
- Input customers: 99,441
- Input items: 112,650
- Input payments: 103,886
- Input reviews: 99,224
- delivered orders: 96,478
- delivered missing purchase timestamp: 0
- delivered missing customer key: 0
- payment rows dropped null: 0
- payment rows dropped zero: 9
- payment rows negative retained: 0
- delivered orders without valid payment: 1
- delivered orders with zero net payment: 0
- eligible paid delivered orders: 96,477
- eligible orders without item quantity: 0

Customer counts:

- after customer key mapping before payment filter: 93,358
- after delivered and payment filters: 93,357
- distinct order scoped customer id values after filters: 96,477

## F* weight fit

- Weights (alpha, beta, gamma): {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0}
- Variance ratio Var(F*) / Var(n_orders): 1.314703
- Simplex points evaluated: 231
- Diagnostic F* qcut bins (requested 5, effective 1): [93357]
- Diagnostic order-count qcut bins (requested 5, effective 1): [93357]
- Quintile bins are diagnostic only; final scores use dense ranks. Duplicate quantile edges demonstrate collapse in ordinary qcut scoring.

## RFMS scoring

- Final customers: 93,357
- Median satisfaction used for imputation: 5.0000
- Customers imputed: 603
- Distinct raw values: {'recency_days': 93111, 'f_star': 52, 'monetary': 28260, 'satisfaction': 34}
- Score populations (1–5): {'r_score': {'1': 18641, '2': 18674, '3': 18663, '4': 18703, '5': 18676}, 'f_score': {'1': 93161, '2': 131, '3': 32, '4': 21, '5': 12}, 'm_score': {'1': 35158, '2': 25937, '3': 16097, '4': 9363, '5': 6802}, 's_score': {'1': 8942, '2': 2921, '3': 7873, '4': 18213, '5': 55408}}
- F-score 1 share: 99.79%

## Method notes

- Recency reference date is the maximum parseable purchase timestamp across the orders table.
- `orders.customer_id` is mapped to `customers.customer_unique_id`; customer-level rows are grouped only by `customer_unique_id`.
- Payment records with null or zero `payment_value` are dropped before order-level payment aggregation.
- F* weight fitting maximizes the specified population-variance ratio on the 0.05 simplex grid; quintile balance is reported as a diagnostic and does not alter the selected weights.
- The Olist order-items table has no quantity column; basket quantity is proxied by counting item rows per order.
- Satisfaction is averaged across available per-order review means per customer; customers with no available review are median-imputed.

# Empirical Results (Methodology v4)

> **Document Status:** Authoritative Empirical Results Section  
> **Primary Domain:** Dunnhumby "The Complete Journey" ($N = 2,499$ households)  
> **Second Domain:** Online Retail II ($N = 5,878$ clean customers)  
> **Methodological Scope:** Strictly RFM Only ($R, F, M$).

---

## 1. Primary Domain: Dunnhumby "The Complete Journey"

### 1.1 Dataset & RFM Structure
The Dunnhumby "The Complete Journey" dataset comprises 2,595,732 transaction records across 2,500 unique households and 276,484 shopping baskets over 711 continuous days.
- **Repeat Rate:** **99.68%** of households made $\ge 2$ baskets (mean $F = 110.59$ baskets, median $79.0$, max $1,300$).
- **Observation Window:** Days 1–620 ($N = 2,499$ households).
- **Holdout Window:** Days 621–711 (91 days; repurchase rate = $98.08\%$, mean spend = \$482.05, mean baskets = $15.05$).

### 1.2 FCA Concept Lattice & Redundancy Suppression
- **Raw Concepts:** Uncapped FP-growth (`max_len = None`, min support = 0.04) identified **502 closed fuzzy concepts** (vs 38 crisp concepts).
- **Kneedle-Inspired Elbow Pruning:** Support threshold $= 0.2129$, Stability-proxy threshold $= 0.4285$.
- **Redundancy Suppression ($J_{\max} = 0.80$):** Compressed the representation to **123 concepts** (4.1× reduction; 379 redundant concepts removed).

### 1.3 Out-of-Sample Predictive Performance (10-Split Temporal Cross-Validation)

| Representation Arm | Features ($D$) | Repurchase AUC | Spend $R^2$ (log1p) | Invoices $R^2$ (log1p) |
|---|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | $0.8624 \pm 0.0275$ | $0.3664 \pm 0.0459$ | $0.4515 \pm 0.0334$ |
| **Crisp RFM-FCA** | 40 | $0.8541 \pm 0.0242$ | $0.4800 \pm 0.0351$ | $0.5548 \pm 0.0344$ |
| **Fuzzy RFM-FCA (Suppressed)** | **114** | **$0.8605 \pm 0.0292$** | **$0.5028 \pm 0.0376$** | **$0.6045 \pm 0.0330$** |
| **FCM Soft (matched $k$)** | 40 | $0.8379 \pm 0.0312$ | $0.4482 \pm 0.0338$ | $0.5520 \pm 0.0254$ |

*Empirical Finding:* On repeat-rich consumer purchasing, Fuzzy RFM-FCA delivers substantial, statistically significant regression gains over raw RFM ($\Delta R^2_{\text{spend}} = +0.1364$, $\Delta R^2_{\text{inv}} = +0.1530$) and canonical FCM soft memberships ($\Delta R^2_{\text{spend}} = +0.0546$, $\Delta R^2_{\text{inv}} = +0.0525$).

### 1.4 Fixed-Split Predictive Metrics (Seed = 42)

| Arm | $D$ | Repurchase AUC | Brier Score | Spend $R^2$ | Spend MAE | Spend $\rho$ | Invoice $R^2$ | Invoice MAE | Invoice $\rho$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw RFM Baseline** | 3 | 0.8694 | 0.0615 | 0.3604 | 1.0707 | 0.7680 | 0.4745 | 0.6212 | 0.7660 |
| **Crisp RFM-FCA** | 40 | 0.8611 | 0.0548 | 0.4629 | 0.9703 | 0.7477 | 0.5536 | 0.5612 | 0.7623 |
| **Fuzzy RFM-FCA (Supp.)** | **114** | **0.8629** | **0.0529** | **0.5152** | **0.8973** | **0.7924** | **0.6188** | **0.5133** | **0.8065** |
| **FCM Soft ($k=40$)** | 40 | 0.8284 | 0.0557 | 0.4445 | 0.9618 | 0.7505 | 0.5537 | 0.5564 | 0.7678 |

#### Paired Bootstrap CIs ($B=1,000$ Resamples)
- **Fuzzy RFM-FCA vs Raw RFM Baseline:**
  - Spend $R^2$: $\mathbf{\Delta = +0.1548}$, $95\%$ CI $[+0.1019, +0.2136]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.1443}$, $95\%$ CI $[+0.1028, +0.1866]$ ($p < 0.001$).
  - Repurchase AUC: $\Delta = -0.0065$, $95\%$ CI $[-0.0349, +0.0196]$ ($p = 0.330$, spans zero).
- **Fuzzy RFM-FCA vs Crisp RFM-FCA:**
  - Spend $R^2$: $\mathbf{\Delta = +0.0523}$, $95\%$ CI $[+0.0302, +0.0769]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0391, +0.0905]$ ($p < 0.001$).
- **Fuzzy RFM-FCA vs FCM Soft ($k=40$):**
  - Repurchase AUC: $\mathbf{\Delta = +0.0345}$, $95\%$ CI $[+0.0053, +0.0663]$ ($p = 0.011$).
  - Spend $R^2$: $\mathbf{\Delta = +0.0707}$, $95\%$ CI $[+0.0328, +0.1075]$ ($p < 0.001$).
  - Invoice $R^2$: $\mathbf{\Delta = +0.0652}$, $95\%$ CI $[+0.0358, +0.0948]$ ($p < 0.001$).

### 1.5 Geometric Clustering Benchmark (Standardized Raw RFM Space)

| Method | Hardening Procedure | $k$ | Evaluation Space | Silhouette | Davies–Bouldin |
|---|---|:---:|---|:---:|:---:|
| **K-Means** | Partition (k-means) | 4 | Raw RFM (Standardized) | 0.4663 | 0.7944 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 4 | Raw RFM (Standardized) | 0.3462 | 0.8421 |
| **FCM (Canonical)** | Argmax Soft Membership | 4 | Raw RFM (Standardized) | 0.4204 | 0.7998 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 4 | Raw RFM (Standardized) | -0.0851 | 2.2278 |
| **K-Means** | Partition (k-means) | 5 | Raw RFM (Standardized) | 0.4747 | 0.7300 |
| **Ward (Agglomerative)** | Hierarchical (Ward) | 5 | Raw RFM (Standardized) | 0.3571 | 0.8441 |
| **FCM (Canonical)** | Argmax Soft Membership | 5 | Raw RFM (Standardized) | 0.3227 | 0.9233 |
| **Fuzzy RFM-FCA** | Top-k Membership Hardening | 5 | Raw RFM (Standardized) | -0.2099 | 2.5513 |

---

## 2. Independent Second Domain: Online Retail II

### 2.1 Base-Paper Replication
- Replicated the 5,878 clean unique customer cohort from Rungruang et al. (2024).
- Recovered **all 31 published frequent concept intents** ($\text{support} > 0.04$).
- Replicated K-Means and Ward clustering geometries across $k=2..10$ (Silhouette ~0.33–0.38, DB ~0.99–1.07).

### 2.2 Crisp RFM-FCA versus Fuzzy RFM-FCA
- Discovered 1,064 closed concepts under uncapped FP-growth (`max_len = None`).
- Greedy redundancy suppression ($J_{\max} = 0.80$) compressed 445 candidate concepts to **95 concepts** (4.7× reduction), eliminating all near-duplicate concept pairs ($J \ge 0.80$ dropped from $3.3\%$ to $0.0\%$).
- **10-Split Temporal Performance:**
  - Raw RFM: AUC $0.7770 \pm 0.0117$, Spend $R^2 = 0.3476$, Invoice $R^2 = 0.4642$.
  - Crisp RFM-FCA: AUC $0.7768 \pm 0.0121$.
  - Fuzzy RFM-FCA (Suppressed): AUC **$0.7857 \pm 0.0121$** ($p = 2.3 \times 10^{-4}$ vs crisp), Spend $R^2 = \mathbf{0.3664}$, Invoice $R^2 = \mathbf{0.4952}$.

---

## 3. Cross-Domain Comparative Synthesis

| Dimension | Primary Domain: Dunnhumby | Second Domain: Online Retail II |
|---|---|---|
| **Setting** | Supermarket Grocery | Non-Store Giftware Retail |
| **Population ($N$)** | 2,499 households | 5,878 customers |
| **Repeat Purchase Rate** | **99.68%** | **72.39%** |
| **Median Purchases ($F$)** | 79.0 baskets | 3.0 orders |
| **Suppressed Concepts** | 123 (from 502) | 95 (from 445) |
| **Compression Ratio** | **4.1×** | **4.7×** |
| **Predictive Lift: Spend $R^2$** | **+0.1364** vs raw ($0.5028$ vs $0.3664$) | **+0.0188** vs raw ($0.3664$ vs $0.3476$) |
| **Predictive Lift: Invoice $R^2$** | **+0.1530** vs raw ($0.6045$ vs $0.4515$) | **+0.0310** vs raw ($0.4952$ vs $0.4642$) |


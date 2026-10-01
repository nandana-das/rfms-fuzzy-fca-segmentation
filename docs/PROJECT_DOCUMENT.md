# Project Document — Final RFM-FCA Research Design

## Research question
Can a centroid-based fuzzy formal context extend an established RFM-FCA customer-segmentation framework while retaining interpretable concept structure and improving downstream predictive utility across independent transaction domains?

## Contributions
1. Centroid-based fuzzy RFM representation.
2. L-fuzzy multi-level scaling.
3. Kneedle-inspired data-driven concept pruning.
4. Greedy extent-Jaccard redundancy suppression.
5. Canonical FCM as a fuzzy benchmark.
6. Leakage-controlled downstream prediction.
7. Independent validation across Dunnhumby and Online Retail II.

## Explicit exclusions
Olist, RFMS/Satisfaction, F*, marketplace-frequency-sparsity experiments, and the Olist temporal-review leakage experiment belong to earlier iterations and are not part of the final paper.

## Interpretation policy
Predictive metrics assess representation utility for future behavior. Silhouette and Davies-Bouldin are geometric diagnostics after hardening. FCA-specific evidence comes from concept structure, overlap, extent, and persistence. These evaluation perspectives must remain separate in the paper.

## Base-paper relationship
Rungruang et al. established RFM-FCA and proposed non-binary representation as future work. The present study extends that framework rather than claiming to introduce RFM-FCA itself.
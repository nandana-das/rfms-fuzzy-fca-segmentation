# Final Methodology and Contribution Framing

## Research positioning
Building on Rungruang et al.'s RFM-FCA customer-segmentation framework, this study extends the crisp/binary RFM formal context toward a centroid-based fuzzy representation and evaluates its utility through leakage-controlled prediction and independent cross-domain testing.

## Contributions
1. Fuzzy RFM representation using centroid-based piecewise-linear membership.
2. Kneedle-inspired data-driven concept pruning.
3. Extent-Jaccard redundancy suppression.
4. Canonical FCM as a fuzzy-clustering comparator, with FCA-specific structure evaluated separately.
5. Leakage-controlled prediction of future repurchase, spend, and invoice behavior.
6. Independent validation across Dunnhumby and Online Retail II.

## Claims explicitly avoided
Do not claim invention of RFM-FCA, universal superiority over K-Means/Ward/FCM, canonical Kneedle, canonical Kuznetsov stability, satisfaction-aware RFMS validation, an F* solution to marketplace sparsity, or Olist as a final validation domain.

## Hard-clustering terminology
**Alpha-cut:** threshold fuzzy membership at α = 0.5; used in the dedicated alpha-cut analysis.

**Top-k membership hardening:** select the k non-trivial concepts for the matched geometric benchmark and assign customers by maximum membership. This is the current k = 4/5/6 FCA benchmark and must not be called an alpha-cut.

## One-sentence positioning
Building on established RFM-FCA customer segmentation, this study develops and evaluates a fuzzy RFM formal context with data-driven concept pruning and redundancy suppression through FCA-specific structural diagnostics, fuzzy benchmarking, leakage-controlled prediction, and cross-domain validation.
# Membership-function ablation smoke results

Methods: M0 (Frozen piecewise-linear shoulders), M1 (Normalized Gaussian), M2 (Normalized generalized bell), M3 (Log-coordinate Gaussian).

All methods use identical customer folds and the frozen thresholds/support/suppression/model settings. Deltas are method minus M0; no method was selected from held-out outcomes.

## Metrics

| dataset   | origin  | pooled | n    | repurchase_rate    | method | auc                | spend_r2           | invoice_r2         |
| --------- | ------- | ------ | ---- | ------------------ | ------ | ------------------ | ------------------ | ------------------ |
| Dunnhumby | day 347 | True   | 2497 | 0.9219062875450541 | M0     | 0.8964623849940966 | 0.5499331432839238 | 0.6244314853034746 |
| Dunnhumby | day 347 | True   | 2497 | 0.9219062875450541 | M1     | 0.8912071999821782 | 0.550160808275808  | 0.6186731856590058 |
| Dunnhumby | day 347 | True   | 2497 | 0.9219062875450541 | M2     | 0.8911336853126601 | 0.5441106040868411 | 0.6129744441249283 |
| Dunnhumby | day 347 | True   | 2497 | 0.9219062875450541 | M3     | 0.8953329323442268 | 0.547011246004552  | 0.6225174689571449 |

## Paired comparisons

| dataset   | scope  | comparison | metric     | delta                  | ci_low                 | ci_high               | p_boot | origins_positive | p_holm |
| --------- | ------ | ---------- | ---------- | ---------------------- | ---------------------- | --------------------- | ------ | ---------------- | ------ |
| Dunnhumby | POOLED | M1 - M0    | auc        | -0.005255185011918351  | -0.013619021232747497  | 0.002672276188441301  | 0.16   | 0/1              | 0.72   |
| Dunnhumby | POOLED | M1 - M0    | spend_r2   | 0.0002276649918842022  | -0.003940725454467961  | 0.004594761469719951  | 0.9    | 1/1              | 1.0    |
| Dunnhumby | POOLED | M1 - M0    | invoice_r2 | -0.005758299644468834  | -0.01037315278465693   | -0.000753556039293767 | 0.04   | 0/1              | 0.32   |
| Dunnhumby | POOLED | M2 - M0    | auc        | -0.005328699681436455  | -0.013263108859257803  | 0.0023431789757538293 | 0.14   | 0/1              | 0.72   |
| Dunnhumby | POOLED | M2 - M0    | spend_r2   | -0.0058225391970827545 | -0.011036707489677314  | 3.200397120884441e-05 | 0.06   | 0/1              | 0.42   |
| Dunnhumby | POOLED | M2 - M0    | invoice_r2 | -0.01145704117854629   | -0.016465238673844824  | -0.006401675155948974 | 0.005  | 0/1              | 0.045  |
| Dunnhumby | POOLED | M3 - M0    | auc        | -0.0011294526498697444 | -0.005098949259474536  | 0.0031663634622352502 | 0.71   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M3 - M0    | spend_r2   | -0.0029218972793718656 | -0.006852617219671545  | 0.0005187155332560923 | 0.12   | 0/1              | 0.72   |
| Dunnhumby | POOLED | M3 - M0    | invoice_r2 | -0.0019140163463297055 | -0.0045982593723453095 | 0.0005431221236550821 | 0.12   | 0/1              | 0.72   |

## Concept-set stability diagnostic

| dataset   | origin  | method | concept_set_jaccard |
| --------- | ------- | ------ | ------------------- |
| Dunnhumby | day 347 | M0     | 0.8621580804773753  |
| Dunnhumby | day 347 | M1     | 0.9577067669172932  |
| Dunnhumby | day 347 | M2     | 0.8186685835768517  |
| Dunnhumby | day 347 | M3     | 0.9222710428439498  |

Runtime: 1.06 minutes.
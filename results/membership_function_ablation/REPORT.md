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

| dataset   | scope  | comparison | metric     | delta                  | ci_low                | ci_high              | p_boot | origins_positive | p_holm |
| --------- | ------ | ---------- | ---------- | ---------------------- | --------------------- | -------------------- | ------ | ---------------- | ------ |
| Dunnhumby | POOLED | M1 - M0    | auc        | -0.005255185011918351  | -0.02932245188061196  | 0.02392762744721288  | 0.8    | 0/1              | 1.0    |
| Dunnhumby | POOLED | M1 - M0    | spend_r2   | 0.0002276649918842022  | -0.03535655900575807  | 0.043422859629568884 | 0.89   | 1/1              | 1.0    |
| Dunnhumby | POOLED | M1 - M0    | invoice_r2 | -0.005758299644468834  | -0.04027616991186757  | 0.030881525151161642 | 0.84   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M2 - M0    | auc        | -0.005328699681436455  | -0.03222419435682508  | 0.02254340084758342  | 0.85   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M2 - M0    | spend_r2   | -0.0058225391970827545 | -0.04002225820511758  | 0.03454859631108554  | 0.95   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M2 - M0    | invoice_r2 | -0.01145704117854629   | -0.04311464995594072  | 0.02767350829711718  | 0.63   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M3 - M0    | auc        | -0.0011294526498697444 | -0.027503777647209276 | 0.026050924757395625 | 0.92   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M3 - M0    | spend_r2   | -0.0029218972793718656 | -0.04236456139353535  | 0.04243816981543712  | 0.95   | 0/1              | 1.0    |
| Dunnhumby | POOLED | M3 - M0    | invoice_r2 | -0.0019140163463297055 | -0.03892720782431526  | 0.039609431142139655 | 0.93   | 0/1              | 1.0    |

## Concept-set stability diagnostic

| dataset   | origin  | method | concept_set_jaccard |
| --------- | ------- | ------ | ------------------- |
| Dunnhumby | day 347 | M0     | 0.8621580804773753  |
| Dunnhumby | day 347 | M1     | 0.9577067669172932  |
| Dunnhumby | day 347 | M2     | 0.8186685835768517  |
| Dunnhumby | day 347 | M3     | 0.9222710428439498  |

Runtime: 1.04 minutes.
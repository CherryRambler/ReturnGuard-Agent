# ReturnGuard Agent - Held-Out Test Evaluation

Evaluated on data/test.csv (773 orders) - the first and only time this file has been used.

## Classifier metrics

- Precision: 0.479
- Recall: 0.619
- F1: 0.540
- ROC-AUC: 0.671
- PR-AUC: 0.508

## Confusion matrix

|                    | Predicted: no return | Predicted: return |
|--------------------|-----------------------|--------------------|
| Actual: no return  | 321 (TN) | 182 (FP) |
| Actual: return     | 103 (FN) | 167 (TP) |

## Cost analysis (INR)

- False-positive cost: 182 x 50 = 9,100
- False-negative cost: 103 x 250 = 25,750
- Net value (value of returns caught minus false-positive friction): 32,650

## Agent action distribution (using the tuned policy)

- allow: 424 (54.9%)
- flag_for_review: 329 (42.6%)
- restrict_cod: 20 (2.6%)

## Thresholds used

- allow_threshold: 0.35
- restrict_threshold: 0.50
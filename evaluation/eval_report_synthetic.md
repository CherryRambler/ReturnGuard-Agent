# ReturnGuard Agent - Held-Out Test Evaluation (synthetic)

Dataset: **synthetic**

Evaluated on data/synthetic/test.csv (773 orders).

## Classifier metrics

- Precision: 0.489
- Recall: 0.644
- F1: 0.556
- ROC-AUC: 0.677
- PR-AUC: 0.530

## Confusion matrix

|                    | Predicted: no return | Predicted: return |
|--------------------|-----------------------|--------------------|
| Actual: no return  | 321 (TN) | 182 (FP) |
| Actual: return     | 96 (FN) | 174 (TP) |

## Cost analysis (INR)

- False-positive cost: 182 x 50 = 9,100
- False-negative cost: 96 x 250 = 24,000
- Net value (value of returns caught minus false-positive friction): 34,400

## Agent action distribution (using the tuned policy)

- allow: 417 (53.9%)
- flag_for_review: 336 (43.5%)
- restrict_cod: 20 (2.6%)

## Thresholds used

- allow_threshold: 0.35
- restrict_threshold: 0.50
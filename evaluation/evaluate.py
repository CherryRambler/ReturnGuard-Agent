"""
ReturnGuard Agent - held-out classifier evaluation.

Day 10. Run this ONCE, after the model is fully finalized. This is the
FIRST AND ONLY TIME test.csv gets used - not for a peek, not for a
sanity check, only here. That is what makes "held-out test set" an
honest claim instead of just a label.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.features import load_split
from src.model import ReturnRiskModel
from src.policy import DecisionPolicy

COST_PER_FALSE_POSITIVE = 50   # INR - friction cost of blocking/reviewing a good order
COST_PER_FALSE_NEGATIVE = 250  # INR - avg loss from a missed return (shipping + restocking + margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the held-out test set")
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir or f"data/{dataset}"
    model_path = args.model or f"models/xgboost_model_{dataset}.joblib"
    out_path = args.out or f"evaluation/eval_report_{dataset}.md"

    model = ReturnRiskModel.load(model_path)
    test_df = load_split(f"{data_dir}/test.csv")
    y_test = test_df["was_returned"]

    scores = model.predict_proba(test_df)
    preds = scores >= model.allow_threshold

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    fp_cost = fp * COST_PER_FALSE_POSITIVE
    fn_cost = fn * COST_PER_FALSE_NEGATIVE
    net_value = tp * COST_PER_FALSE_NEGATIVE - fp * COST_PER_FALSE_POSITIVE

    policy = DecisionPolicy(allow_threshold=model.allow_threshold, restrict_threshold=model.restrict_threshold)
    actions = [
        policy.decide(risk_score=s, payment_method=pm).action
        for s, pm in zip(scores, test_df["payment_method"])
    ]
    action_counts = pd.Series(actions).value_counts()

    lines = [
        f"# ReturnGuard Agent - Held-Out Test Evaluation ({dataset})",
        "",
        f"Dataset: **{dataset}**",
        "",
        f"Evaluated on {data_dir}/test.csv ({len(test_df)} orders).",
        "",
        "## Classifier metrics",
        "",
        f"- Precision: {precision:.3f}",
        f"- Recall: {recall:.3f}",
        f"- F1: {f1:.3f}",
        f"- ROC-AUC: {roc_auc:.3f}",
        f"- PR-AUC: {pr_auc:.3f}",
        "",
        "## Confusion matrix",
        "",
        "|                    | Predicted: no return | Predicted: return |",
        "|--------------------|-----------------------|--------------------|",
        f"| Actual: no return  | {tn} (TN) | {fp} (FP) |",
        f"| Actual: return     | {fn} (FN) | {tp} (TP) |",
        "",
        "## Cost analysis (INR)",
        "",
        f"- False-positive cost: {fp} x {COST_PER_FALSE_POSITIVE} = {fp_cost:,.0f}",
        f"- False-negative cost: {fn} x {COST_PER_FALSE_NEGATIVE} = {fn_cost:,.0f}",
        f"- Net value (value of returns caught minus false-positive friction): {net_value:,.0f}",
        "",
        "## Agent action distribution (using the tuned policy)",
        "",
    ]
    for action, count in action_counts.items():
        lines.append(f"- {action}: {count} ({count / len(test_df):.1%})")

    lines += [
        "",
        "## Thresholds used",
        "",
        f"- allow_threshold: {model.allow_threshold:.2f}",
        f"- restrict_threshold: {model.restrict_threshold:.2f}",
    ]

    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
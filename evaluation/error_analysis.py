"""
ReturnGuard Agent - error analysis (Day 11).

Re-reads the held-out test set to LOOK AT mistakes - not to tune
anything further. The model and thresholds are already finalized as of
Day 10; nothing here should change them. This script exists purely to
generate the concrete examples and patterns a human writes up by hand
into "what broke" / "known limitations" for the submission.

Prints:
  1. A handful of real false-positive examples (flagged, but the order
     was never actually returned) with their SHAP reasons.
  2. A handful of real false-negative examples (allowed, but the order
     WAS actually returned) with their SHAP reasons.
  3. Simple aggregate comparisons (average feature values) between
     mistakes and correct predictions, to check for a systematic pattern
     rather than guessing from a few examples alone.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.explain import top_reasons
from src.features import load_split
from src.model import ReturnRiskModel

COMPARE_COLUMNS = [
    "discount_pct",
    "size_flag",
    "is_new_customer",
    "customer_prior_return_rate",
    "review_score_avg",
    "order_amount",
]


def print_examples(model, df: pd.DataFrame, indices, label: str, n: int) -> None:
    print(f"\n=== {label} (showing up to {n}) ===")
    for i, idx in enumerate(indices[:n]):
        row = df.loc[[idx]]
        reasons = top_reasons(model, row, n=3)
        r = df.loc[idx]
        print(f"\n  order_id={r['order_id']}  risk_score={r['risk_score']:.3f}  "
              f"category={r['category']}  payment_method={r['payment_method']}  "
              f"discount_pct={r['discount_pct']}  is_new_customer={r['is_new_customer']}")
        for reason in reasons:
            print(f"    - {reason}")


def print_comparison(df: pd.DataFrame, mistake_mask, correct_mask, label: str) -> None:
    print(f"\n=== Average feature values: {label} ===")
    mistakes = df[mistake_mask]
    correct = df[correct_mask]
    for col in COMPARE_COLUMNS:
        m_mean = mistakes[col].mean()
        c_mean = correct[col].mean()
        print(f"  {col:32s} mistakes={m_mean:.3f}   correct={c_mean:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Look at the model's mistakes on the held-out test set")
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir or f"data/{dataset}"
    model_path = args.model or f"models/xgboost_model_{dataset}.joblib"

    print(f"=== Error analysis on the '{dataset}' dataset ===")
    model = ReturnRiskModel.load(model_path)
    test_df = load_split(f"{data_dir}/test.csv").reset_index(drop=True)

    test_df["risk_score"] = model.predict_proba(test_df)
    test_df["predicted"] = test_df["risk_score"] >= model.allow_threshold

    false_positive_mask = test_df["predicted"] & ~test_df["was_returned"]
    false_negative_mask = ~test_df["predicted"] & test_df["was_returned"]
    true_negative_mask = ~test_df["predicted"] & ~test_df["was_returned"]
    true_positive_mask = test_df["predicted"] & test_df["was_returned"]

    fp_indices = test_df[false_positive_mask].sort_values("risk_score", ascending=False).index.tolist()
    fn_indices = test_df[false_negative_mask].sort_values("risk_score", ascending=True).index.tolist()

    print_examples(model, test_df, fp_indices, "FALSE POSITIVES (flagged, but never actually returned)", args.n_examples)
    print_examples(model, test_df, fn_indices, "FALSE NEGATIVES (allowed, but WAS actually returned)", args.n_examples)

    print_comparison(test_df, false_positive_mask, true_negative_mask, "false positives vs. correctly-allowed orders")
    print_comparison(test_df, false_negative_mask, true_positive_mask, "false negatives vs. correctly-flagged orders")

    print(f"\nTotals: {false_positive_mask.sum()} false positives, {false_negative_mask.sum()} false negatives, "
          f"out of {len(test_df)} test orders.")


if __name__ == "__main__":
    main()

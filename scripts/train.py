"""
ReturnGuard Agent - training entrypoint.

Day 4:  baseline models (majority-class, simple logistic regression)
Day 5:  real model - hyperparameter search + XGBoost, train/val only
Day 6:  calibration + cost/capacity-based threshold tuning

Usage:
    python -m scripts.train --data-dir data --model-out models/xgboost_model.joblib
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.features import load_split
from src.model import ReturnRiskModel
from src.policy import DecisionPolicy
from src.threshold_tuning import select_allow_threshold, select_restrict_threshold


def majority_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.Series:
    majority_class = train_df["was_returned"].mode()[0]
    return pd.Series(majority_class, index=val_df.index)


def simple_logistic_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    feature_cols = ["discount_pct", "is_new_customer", "size_flag", "customer_prior_return_rate"]
    fill_value = train_df["customer_prior_return_rate"].mean()

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feature_cols].copy()
        X["is_new_customer"] = X["is_new_customer"].astype(int)
        X["size_flag"] = X["size_flag"].astype(int)
        X["customer_prior_return_rate"] = X["customer_prior_return_rate"].fillna(fill_value)
        return X

    X_train, y_train = prep(train_df), train_df["was_returned"]
    X_val, y_val = prep(val_df), val_df["was_returned"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model.predict(X_val), y_val


def evaluate(y_true, y_pred, label: str) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"Confusion matrix:\n{cm}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ReturnGuard Agent risk model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-out", type=str, default="models/xgboost_model.joblib")
    args = parser.parse_args()

    train_df = load_split(f"{args.data_dir}/train.csv")
    val_df = load_split(f"{args.data_dir}/val.csv")

    majority_preds = majority_baseline(train_df, val_df)
    evaluate(val_df["was_returned"], majority_preds, "Majority-class baseline")

    log_preds, y_val = simple_logistic_baseline(train_df, val_df)
    evaluate(y_val, log_preds, "Logistic regression baseline")

    candidate_params = [
        dict(n_estimators=200, max_depth=4, learning_rate=0.05),
        dict(n_estimators=300, max_depth=3, learning_rate=0.05),
        dict(n_estimators=200, max_depth=5, learning_rate=0.03),
        dict(n_estimators=400, max_depth=4, learning_rate=0.03),
        dict(n_estimators=150, max_depth=6, learning_rate=0.05),
    ]

    print("\n--- Hyperparameter search (train/val only, compared by val ROC-AUC) ---")
    best_model = None
    best_auc = -1.0
    best_params = None
    for params in candidate_params:
        candidate = ReturnRiskModel().fit(train_df, **params)
        candidate_val_scores = candidate.predict_proba(val_df)
        auc = roc_auc_score(val_df["was_returned"], candidate_val_scores)
        print(f"  {params} -> val ROC-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = candidate
            best_params = params

    print(f"Best: {best_params} (val ROC-AUC={best_auc:.4f})")
    model = best_model

    val_scores_uncalibrated = model.predict_proba(val_df)
    val_preds = val_scores_uncalibrated >= 0.5
    evaluate(val_df["was_returned"], val_preds, "XGBoost model (best hyperparameters, 0.5 threshold, uncalibrated)")

    model.calibrate(val_df)
    val_scores = model.predict_proba(val_df)
    y_val = val_df["was_returned"]

    allow_row = select_allow_threshold(y_val, val_scores, min_precision=0.40, max_flag_rate=0.40)
    print(f"\n--- Day 6 (revised): threshold tuning (on val.csv, with a 40% flag-rate cap) ---")
    if allow_row.precision < 0.40:
        print(
            f"WARNING: no threshold reached the 0.40 precision floor - falling back to "
            f"pure net-value maximization (precision={allow_row.precision:.3f})."
        )
    if allow_row.flag_rate > 0.40:
        print(
            f"WARNING: no threshold satisfied both the precision floor AND the 40% flag-rate cap - "
            f"relaxed the flag-rate cap (flag_rate={allow_row.flag_rate:.1%})."
        )
    print(
        f"Net-value-optimal allow_threshold: {allow_row.threshold:.2f}  "
        f"(precision={allow_row.precision:.3f}, recall={allow_row.recall:.3f}, "
        f"flag_rate={allow_row.flag_rate:.1%}, net_value=INR {allow_row.net_value:.0f} on this validation batch)"
    )

    cod_mask = val_df["payment_method"] == "COD"
    restrict_threshold = select_restrict_threshold(
        y_true_cod=y_val[cod_mask],
        scores_cod=val_scores[cod_mask],
        min_threshold=allow_row.threshold,
        min_precision=0.65,
    )
    print(f"Precision-gated restrict_threshold (COD orders, target precision >= 0.65): {restrict_threshold:.2f}")

    model.set_thresholds(allow_threshold=allow_row.threshold, restrict_threshold=restrict_threshold)
    evaluate(y_val, val_scores >= allow_row.threshold, "XGBoost model (calibrated, tuned allow_threshold)")

    policy = DecisionPolicy(allow_threshold=model.allow_threshold, restrict_threshold=model.restrict_threshold)
    actions = [
        policy.decide(risk_score=s, payment_method=pm).action
        for s, pm in zip(val_scores, val_df["payment_method"])
    ]
    action_counts = pd.Series(actions).value_counts()
    print(f"\nAction distribution on val.csv ({len(val_df)} orders):")
    print(action_counts.to_string())

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"\nSaved calibrated model + tuned thresholds to {args.model_out}")


if __name__ == "__main__":
    main()

"""
ReturnGuard Agent - training entrypoint.

Day 4:  baseline models (majority-class, simple logistic regression)
Day 5:  real model - hyperparameter search + XGBoost, train/val only
Day 6:  calibration + cost/capacity-based threshold tuning

Usage:
    python -m scripts.train --dataset synthetic
    python -m scripts.train --dataset real

--dataset selects everything: which data/ subdir to read, which
feature-column preset to use, which guardrail values to apply during
threshold selection, and which models/xgboost_model_{dataset}.joblib to
write. Default is synthetic, so existing behaviour is unchanged.
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
from src.threshold_tuning import select_allow_threshold, select_restrict_threshold, sweep_thresholds

# Per-dataset, already-validated guardrails for allow-threshold selection.
# Real data's positive rate (~12%) is far below synthetic's (~35%), so a
# precision floor tuned for one does not transfer to the other - these
# were tested and confirmed to matter. Do NOT collapse them into one.
DATASET_GUARDRAILS = {
    "synthetic": {"min_precision": 0.40, "max_flag_rate": 0.40},
    "real": {"min_precision": 0.20, "max_flag_rate": 0.35},
}


def majority_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.Series:
    majority_class = train_df["was_returned"].mode()[0]
    return pd.Series(majority_class, index=val_df.index)


def simple_logistic_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    candidate_cols = ["discount_pct", "is_new_customer", "size_flag", "customer_prior_return_rate"]
    # Drop any column that is entirely missing in train (e.g. discount_pct
    # on the real dataset) - a baseline can't learn from an all-NaN column.
    feature_cols = [c for c in candidate_cols if train_df[c].notna().any()]
    numeric_cols = [c for c in feature_cols if c in ("discount_pct", "customer_prior_return_rate")]
    fill_values = {c: train_df[c].mean() for c in numeric_cols}

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feature_cols].copy()
        if "is_new_customer" in X:
            X["is_new_customer"] = X["is_new_customer"].astype(int)
        if "size_flag" in X:
            X["size_flag"] = X["size_flag"].astype(int)
        for c in numeric_cols:
            X[c] = X[c].fillna(fill_values[c])
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
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override the data directory (defaults to data/<dataset>)",
    )
    parser.add_argument(
        "--model-out", type=str, default=None,
        help="Override the model output path (defaults to models/xgboost_model_<dataset>.joblib)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir or f"data/{dataset}"
    model_out = args.model_out or f"models/xgboost_model_{dataset}.joblib"
    guardrails = DATASET_GUARDRAILS[dataset]
    print(f"=== Training on the '{dataset}' dataset ===")
    print(f"  data dir:  {data_dir}")
    print(f"  model out: {model_out}")
    print(f"  guardrails: {guardrails}")

    train_df = load_split(f"{data_dir}/train.csv")
    val_df = load_split(f"{data_dir}/val.csv")

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
        candidate = ReturnRiskModel(dataset=dataset).fit(train_df, **params)
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

    min_precision = guardrails["min_precision"]
    max_flag_rate = guardrails["max_flag_rate"]

    # Print the FULL threshold sweep so it is visible that the selected
    # point is not excluding the actual net-value-maximizing threshold.
    print(f"\n--- Full threshold sweep on val.csv (dataset={dataset}) ---")
    print("thresh  precision  recall  flag_rate  net_value")
    sweep = sweep_thresholds(y_val, val_scores)
    best_by_net = max(sweep, key=lambda r: r.net_value)
    for r in sweep:
        is_best = abs(r.threshold - best_by_net.threshold) < 1e-9
        marker = "  <- max net_value" if is_best else ""
        if abs((r.threshold * 100) % 5) < 1e-6 or is_best:
            print(
                f"{r.threshold:5.2f}   {r.precision:8.3f}  {r.recall:6.3f}  "
                f"{r.flag_rate:8.1%}  {r.net_value:9.0f}{marker}"
            )

    allow_row = select_allow_threshold(
        y_val, val_scores, min_precision=min_precision, max_flag_rate=max_flag_rate
    )
    print(
        f"\n--- Threshold tuning (on val.csv, guardrails: "
        f"min_precision={min_precision}, max_flag_rate={max_flag_rate:.0%}) ---"
    )
    if abs(allow_row.threshold - best_by_net.threshold) > 1e-9:
        print(
            f"NOTE: guardrails moved the selected threshold off the raw net-value max "
            f"({best_by_net.threshold:.2f}, net_value={best_by_net.net_value:.0f}, "
            f"precision={best_by_net.precision:.3f}, flag_rate={best_by_net.flag_rate:.1%})."
        )
    if allow_row.precision < min_precision:
        print(
            f"WARNING: no threshold reached the {min_precision} precision floor - falling back to "
            f"pure net-value maximization (precision={allow_row.precision:.3f})."
        )
    if allow_row.flag_rate > max_flag_rate:
        print(
            f"WARNING: no threshold satisfied both the precision floor AND the {max_flag_rate:.0%} "
            f"flag-rate cap - relaxed the flag-rate cap (flag_rate={allow_row.flag_rate:.1%})."
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

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"\nSaved calibrated model + tuned thresholds ({dataset}) to {model_out}")


if __name__ == "__main__":
    main()

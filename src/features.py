"""
ReturnGuard Agent - feature engineering pipeline.

Builds an sklearn ColumnTransformer that turns raw order columns into
the numeric matrix XGBoost needs. Fit ONLY on train.csv - val/test and
live inference calls only ever use .transform() on an already-fitted
pipeline, never .fit() or .fit_transform() again. That's what keeps the
held-out test set honest (see razorpay_buildathon_plan.md, Section 7,
"Leakage prevention").
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["category", "payment_method", "delivery_pincode_risk_tier"]
NUMERIC_COLUMNS = [
    "order_amount",
    "item_count",
    "discount_pct",
    "days_since_signup",
    "customer_prior_return_rate",
    "review_score_avg",
]
BOOLEAN_COLUMNS = ["is_new_customer", "size_flag", "has_return_history"]
LABEL_COLUMN = "was_returned"

NON_FEATURE_COLUMNS = ["order_id", "order_date", "customer_id", LABEL_COLUMN, "action_taken"]

ALL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BOOLEAN_COLUMNS


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds has_return_history: True if we have ANY prior-return data for
    this customer, False if they're unknown (new customer or missing
    data). This is a deliberately separate signal from
    customer_prior_return_rate itself - the numeric column gets imputed
    with a fallback value when missing, which quietly throws away the
    fact that it WAS missing. Making that fact its own boolean feature
    lets the model use "do we know anything about this customer" as a
    signal in its own right, not just the (possibly made-up) number."""
    df = df.copy()
    df["has_return_history"] = df["customer_prior_return_rate"].notna()
    return df


def _to_int(X):
    """Cast boolean columns to 0/1 ints - keeps dtypes predictable for
    the model regardless of how pandas happened to read the CSV."""
    return X.astype(int)


def build_feature_pipeline() -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer.

    categorical -> impute missing with most-frequent, then one-hot encode
    numeric     -> impute missing with the median, then standard-scale
    boolean     -> cast to 0/1 int, passed through unscaled
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    boolean_pipeline = Pipeline(
        steps=[("cast", FunctionTransformer(_to_int, feature_names_out="one-to-one"))]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("boolean", boolean_pipeline, BOOLEAN_COLUMNS),
        ],
        remainder="drop",
    )


def load_split(path: str) -> pd.DataFrame:
    """Thin helper so scripts/train.py, backend/, and evaluation/ all load
    train/val/test consistently (same dtypes, same date parsing)."""
    return pd.read_csv(path, parse_dates=["order_date"])


def split_features_and_label(df: pd.DataFrame):
    """Return (X, y). X is the raw column subset the pipeline expects
    (after engineered features are added); y is the boolean label, or
    None if the dataframe has no label column (e.g. a live inference
    request)."""
    df = _engineer_features(df)
    X = df[ALL_FEATURE_COLUMNS]
    y = df[LABEL_COLUMN] if LABEL_COLUMN in df.columns else None
    return X, y

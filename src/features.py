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

DATASETS = ("synthetic", "real")

# Both datasets share ONE unified column schema (approach (b) in the
# real-data adapter spec): discount_pct is always present as a column.
# For synthetic data it carries real generated values; for real (Olist)
# data it is all-NaN and gets imputed away, so it contributes no signal -
# a dead feature rather than an invented one. Keeping the schema identical
# means model.py, backend/main.py, and the DB layer need zero
# per-dataset branching. The SYNTHETIC_* / REAL_* aliases below exist so
# call sites can be explicit about which dataset they're operating on
# even though the lists are currently the same.
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

# Named per-dataset presets. Identical today (unified schema), but every
# function that selects columns goes through _columns_for(dataset) so a
# future divergence is a one-line change here, not a codebase-wide hunt.
SYNTHETIC_CATEGORICAL_COLUMNS = list(CATEGORICAL_COLUMNS)
SYNTHETIC_NUMERIC_COLUMNS = list(NUMERIC_COLUMNS)
SYNTHETIC_BOOLEAN_COLUMNS = list(BOOLEAN_COLUMNS)

REAL_CATEGORICAL_COLUMNS = list(CATEGORICAL_COLUMNS)
REAL_NUMERIC_COLUMNS = list(NUMERIC_COLUMNS)
REAL_BOOLEAN_COLUMNS = list(BOOLEAN_COLUMNS)


def _columns_for(dataset: str):
    """Return (categorical, numeric, boolean) column lists for a dataset."""
    if dataset == "synthetic":
        return SYNTHETIC_CATEGORICAL_COLUMNS, SYNTHETIC_NUMERIC_COLUMNS, SYNTHETIC_BOOLEAN_COLUMNS
    if dataset == "real":
        return REAL_CATEGORICAL_COLUMNS, REAL_NUMERIC_COLUMNS, REAL_BOOLEAN_COLUMNS
    raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")


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


def build_feature_pipeline(dataset: str = "synthetic") -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer for the given dataset.

    categorical -> impute missing with most-frequent, then one-hot encode
    numeric     -> impute missing with the median, then standard-scale
    boolean     -> cast to 0/1 int, passed through unscaled

    `dataset` selects the column preset ("synthetic" or "real"). With the
    unified schema the two presets are identical, but routing through it
    keeps the door open for a real-only schema later.
    """
    categorical_columns, numeric_columns, boolean_columns = _columns_for(dataset)
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
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
            ("boolean", boolean_pipeline, boolean_columns),
        ],
        remainder="drop",
    )


def load_split(path: str) -> pd.DataFrame:
    """Thin helper so scripts/train.py, backend/, and evaluation/ all load
    train/val/test consistently (same dtypes, same date parsing)."""
    return pd.read_csv(path, parse_dates=["order_date"])


def feature_columns_for(dataset: str = "synthetic") -> list[str]:
    """The full ordered feature-column list for a dataset (categorical +
    numeric + boolean, including engineered columns like
    has_return_history)."""
    categorical_columns, numeric_columns, boolean_columns = _columns_for(dataset)
    return categorical_columns + numeric_columns + boolean_columns


def split_features_and_label(df: pd.DataFrame, dataset: str = "synthetic"):
    """Return (X, y). X is the raw column subset the pipeline expects
    (after engineered features are added); y is the boolean label, or
    None if the dataframe has no label column (e.g. a live inference
    request).

    `dataset` selects the column preset. If the frame is missing a column
    the preset expects (e.g. real-data CSVs written without discount_pct),
    it is added as all-NaN so the imputer handles it uniformly."""
    df = _engineer_features(df)
    wanted = feature_columns_for(dataset)
    df = df.copy()
    for col in wanted:
        if col not in df.columns:
            df[col] = float("nan")
    X = df[wanted]
    y = df[LABEL_COLUMN] if LABEL_COLUMN in df.columns else None
    return X, y

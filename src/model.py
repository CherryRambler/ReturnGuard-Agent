"""
ReturnGuard Agent - model training and loading.

Wraps the fitted feature pipeline + XGBoost classifier as ONE artifact,
so backend/main.py only ever has to load one file and call
predict_proba() - it doesn't need to know anything about XGBoost or the
feature pipeline internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from src.features import build_feature_pipeline, split_features_and_label

DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)


def _wrap_prefit_for_calibration(fitted_estimator, method: str) -> CalibratedClassifierCV:
    """CalibratedClassifierCV's API for calibrating an ALREADY-fitted
    estimator changed between sklearn versions: sklearn >= 1.6 wants
    FrozenEstimator(fitted_estimator); older versions want
    cv="prefit". Support both so this doesn't silently break on
    whatever sklearn version happens to be installed."""
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6

        return CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method=method)
    except ImportError:
        return CalibratedClassifierCV(fitted_estimator, method=method, cv="prefit")


@dataclass
class ReturnRiskModel:
    """predict_proba() returns a risk_score in [0, 1] per row - this is
    the only contract src/policy.py and backend/main.py rely on.

    allow_threshold / restrict_threshold are stored on the model itself
    once Day 6's threshold tuning has run, so the whole tuned artifact -
    model, calibration, and thresholds - travels together as one file."""

    feature_pipeline: Optional[ColumnTransformer] = None
    classifier: Optional[object] = None
    raw_classifier: Optional[object] = None  # the plain XGBoost model, kept for SHAP - see explain.py
    allow_threshold: float = 0.35
    restrict_threshold: float = 0.65
    dataset: str = "synthetic"  # which column preset this model was trained on

    def fit(self, train_df: pd.DataFrame, **xgb_param_overrides) -> "ReturnRiskModel":
        """Fit the feature pipeline AND the classifier on train_df only.
        Never call this with val or test data."""
        X_train, y_train = split_features_and_label(train_df, self.dataset)

        self.feature_pipeline = build_feature_pipeline(self.dataset)
        X_train_transformed = self.feature_pipeline.fit_transform(X_train)

        # Returns are the minority class (~38% of orders) - without this,
        # XGBoost optimizes for overall accuracy and gets too conservative
        # about ever predicting "returned". scale_pos_weight fixes that,
        # but it also distorts predict_proba away from a true probability -
        # that's exactly what calibrate() below corrects for.
        n_negative = int((~y_train).sum())
        n_positive = int(y_train.sum())
        auto_scale_pos_weight = n_negative / max(n_positive, 1)

        params = {**DEFAULT_XGB_PARAMS, "scale_pos_weight": auto_scale_pos_weight, **xgb_param_overrides}
        self.classifier = XGBClassifier(**params)
        self.classifier.fit(X_train_transformed, y_train)
        # Keep an unwrapped reference NOW, before calibrate() replaces
        # self.classifier with a CalibratedClassifierCV wrapper that SHAP's
        # TreeExplainer can't read directly.
        self.raw_classifier = self.classifier
        return self

    def calibrate(self, val_df: pd.DataFrame, method: str = "sigmoid") -> "ReturnRiskModel":
        """Recalibrate predicted probabilities using a held-out validation
        set, so risk_score behaves like a real probability again. Only
        ever call this with val.csv - never train.csv, never test.csv."""
        if self.feature_pipeline is None or self.classifier is None:
            raise RuntimeError("Fit the model before calibrating it.")
        X_val, y_val = split_features_and_label(val_df, self.dataset)
        X_val_transformed = self.feature_pipeline.transform(X_val)
        calibrated = _wrap_prefit_for_calibration(self.classifier, method)
        calibrated.fit(X_val_transformed, y_val)
        self.classifier = calibrated
        return self

    def set_thresholds(self, allow_threshold: float, restrict_threshold: float) -> "ReturnRiskModel":
        self.allow_threshold = allow_threshold
        self.restrict_threshold = restrict_threshold
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk_score (probability of was_returned=True) per row."""
        if self.feature_pipeline is None or self.classifier is None:
            raise RuntimeError("Model is not fitted or loaded yet. Call fit() or load() first.")
        X, _ = split_features_and_label(df, self.dataset)
        X_transformed = self.feature_pipeline.transform(X)
        return self.classifier.predict_proba(X_transformed)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(df) >= threshold

    def save(self, path: str) -> None:
        if self.feature_pipeline is None or self.classifier is None:
            raise RuntimeError("Nothing to save - fit the model first.")
        joblib.dump(
            {
                "feature_pipeline": self.feature_pipeline,
                "classifier": self.classifier,
                "raw_classifier": self.raw_classifier,
                "allow_threshold": self.allow_threshold,
                "restrict_threshold": self.restrict_threshold,
                "dataset": self.dataset,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ReturnRiskModel":
        artifact = joblib.load(path)
        return cls(
            feature_pipeline=artifact["feature_pipeline"],
            classifier=artifact["classifier"],
            raw_classifier=artifact.get("raw_classifier"),
            allow_threshold=artifact.get("allow_threshold", 0.35),
            restrict_threshold=artifact.get("restrict_threshold", 0.65),
            dataset=artifact.get("dataset", "synthetic"),
        )
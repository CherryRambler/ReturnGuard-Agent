"""
ReturnGuard Agent - explainability (SHAP).

Turns a risk_score into a short, human-readable list of WHY - the top
features that pushed this specific order's score up or down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.features import split_features_and_label
from src.model import ReturnRiskModel


def _get_feature_names(model: ReturnRiskModel) -> list[str]:
    return list(model.feature_pipeline.get_feature_names_out())


def _phrase(feature_name: str, shap_value: float) -> str:
    direction = "pushes risk up" if shap_value > 0 else "pushes risk down"
    clean_name = feature_name.split("__", 1)[-1]
    return f"{clean_name} ({direction}, impact {abs(shap_value):.3f})"


def top_reasons(model: ReturnRiskModel, order_row: pd.DataFrame, n: int = 3) -> list[str]:
    if model.raw_classifier is None:
        raise RuntimeError(
            "This model has no raw_classifier saved - retrain with the "
            "updated src/model.py so fit() captures it before calibration."
        )

    X, _ = split_features_and_label(order_row, getattr(model, "dataset", "synthetic"))
    X_transformed = model.feature_pipeline.transform(X)

    explainer = shap.TreeExplainer(model.raw_classifier)
    shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer returns a list of per-class arrays on some shap/xgboost
    # version combinations, and a single 2D array on others. Normalize to
    # "the positive-class row" either way, instead of assuming shape.
    if isinstance(shap_values, list):
        row_shap_values = np.asarray(shap_values[-1])[0]
    else:
        row_shap_values = np.asarray(shap_values)[0]
    feature_names = _get_feature_names(model)

    ranked = sorted(
        zip(feature_names, row_shap_values),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    return [_phrase(name, value) for name, value in ranked[:n]]
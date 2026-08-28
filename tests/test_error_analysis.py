"""
Tests for evaluation/error_analysis.py.

Skips per-dataset if that model doesn't exist. Just confirms the script
runs without crashing - the actual analysis is meant to be read by a
human, not asserted on in a test.
"""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "dataset, model_path",
    [
        ("synthetic", "models/xgboost_model_synthetic.joblib"),
        ("real", "models/xgboost_model_real.joblib"),
    ],
)
def test_error_analysis_runs_without_crashing(dataset, model_path):
    if not os.path.exists(model_path):
        pytest.skip(f"Run `python -m scripts.train --dataset {dataset}` first.")

    result = subprocess.run(
        [sys.executable, "-m", "evaluation.error_analysis", "--dataset", dataset, "--n-examples", "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FALSE POSITIVES" in result.stdout
    assert "FALSE NEGATIVES" in result.stdout

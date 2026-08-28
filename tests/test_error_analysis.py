"""
Tests for evaluation/error_analysis.py.

Skips if no trained model exists yet. Just confirms the script runs
without crashing - the actual analysis is meant to be read by a human,
not asserted on in a test.
"""

import os
import subprocess
import sys

import pytest

MODEL_PATH = "models/xgboost_model.joblib"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Run scripts/train.py first.")
def test_error_analysis_runs_without_crashing():
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.error_analysis", "--n-examples", "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FALSE POSITIVES" in result.stdout
    assert "FALSE NEGATIVES" in result.stdout

"""
Tests for evaluation/evaluate.py.

Skips if no trained model exists yet. Only checks the SCRIPT runs and
produces a sensibly-shaped report - it does NOT assert specific metric
values, since those legitimately vary run to run.
"""

import os
import subprocess
import sys

import pytest

MODEL_PATH = "models/xgboost_model.joblib"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Run scripts/train.py first.")
def test_evaluate_script_runs_and_writes_report(tmp_path):
    out_path = tmp_path / "eval_report.md"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate", "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    content = out_path.read_text()
    assert "Precision" in content
    assert "Recall" in content
    assert "Confusion matrix" in content
    assert "Net value" in content
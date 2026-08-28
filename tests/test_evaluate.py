"""
Tests for evaluation/evaluate.py.

Skips per-dataset if that model doesn't exist. Only checks the SCRIPT
runs and produces a sensibly-shaped report - it does NOT assert specific
metric values, since those legitimately vary run to run.
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
def test_evaluate_script_runs_and_writes_report(dataset, model_path, tmp_path):
    if not os.path.exists(model_path):
        pytest.skip(f"Run `python -m scripts.train --dataset {dataset}` first.")

    out_path = tmp_path / f"eval_report_{dataset}.md"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate", "--dataset", dataset, "--out", str(out_path)],
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
    assert dataset in content

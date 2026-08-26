"""Unit tests for evaluation functions in src/evaluate.py."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.train import create_pipeline, FEATURE_COLUMNS
from src.evaluate import evaluate_model, save_metrics


@pytest.fixture
def synthetic_eval_data():
    """Create synthetic test features, labels, and trained pipeline."""
    np.random.seed(42)
    n_samples = 20
    data = {
        "Time": np.arange(n_samples, dtype=float),
        "Amount": np.random.uniform(10.0, 200.0, size=n_samples),
        "Class": np.array([0] * 16 + [1] * 4),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)

    df = pd.DataFrame(data)
    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    pipeline = create_pipeline(contamination=0.2, random_state=42)
    pipeline.fit(X)

    return pipeline, X, y


def test_evaluate_model_returns_expected_keys(synthetic_eval_data):
    """Test evaluate_model calculates metrics dictionary with all required keys."""
    model, X_test, y_test = synthetic_eval_data
    metrics = evaluate_model(model, X_test, y_test)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["confusion_matrix"], list)
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_save_metrics(tmp_path):
    """Test save_metrics writes valid JSON file."""
    metrics = {
        "precision": 0.85,
        "recall": 0.75,
        "f1": 0.80,
        "roc_auc": 0.92,
        "confusion_matrix": [[100, 5], [2, 10]],
    }
    output_file = tmp_path / "eval_metrics.json"
    save_metrics(metrics, output_file)

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == metrics

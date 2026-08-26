"""Unit tests for training pipeline in src/train.py."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from src.train import train_model, create_pipeline, FEATURE_COLUMNS


@pytest.fixture
def synthetic_train_df():
    """Create a synthetic 20-row training DataFrame."""
    np.random.seed(42)
    n_samples = 20
    data = {
        "Time": np.arange(n_samples, dtype=float),
        "Amount": np.random.uniform(5.0, 300.0, size=n_samples),
        "Class": [0] * 18 + [1] * 2,
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)

    return pd.DataFrame(data)


def test_train_model_creates_file_and_predicts(synthetic_train_df, tmp_path):
    """Test that train_model creates a model file on disk and can make predictions."""
    model_file = tmp_path / "models" / "model.pkl"

    pipeline = train_model(
        train_df=synthetic_train_df,
        model_path=model_file,
        contamination=0.1,
        random_state=42,
    )

    # 1. Assert model file is created
    assert model_file.exists(), "Trained model file should exist on disk"
    assert model_file.stat().st_size > 0, "Model file should not be empty"

    # 2. Assert loaded model predicts on a sample without error
    loaded_pipeline = joblib.load(model_file)
    sample = synthetic_train_df[FEATURE_COLUMNS].iloc[:3]

    predictions = loaded_pipeline.predict(sample)
    assert len(predictions) == 3
    assert set(predictions).issubset({-1, 1})

    # Test decision function / anomaly score computation
    decision_scores = loaded_pipeline.decision_function(sample)
    assert len(decision_scores) == 3
    assert isinstance(decision_scores, np.ndarray)


def test_train_model_missing_feature_columns():
    """Test train_model raises KeyError if required feature columns are missing."""
    incomplete_df = pd.DataFrame({"Time": [1.0, 2.0], "Amount": [10.0, 20.0]})
    with pytest.raises(KeyError):
        train_model(incomplete_df, model_path=None)

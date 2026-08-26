"""Unit tests for Streamlit application helper functions in app.py."""

import numpy as np
import pandas as pd
import pytest
from app import get_eval_metrics, get_sample_dataset, predict_sample, plot_confusion_matrix
from src.train import create_pipeline, FEATURE_COLUMNS


@pytest.fixture
def sample_model_and_row():
    """Create a synthetic model and single row for testing predict_sample."""
    np.random.seed(42)
    n_samples = 20
    data = {
        "Time": np.arange(n_samples, dtype=float),
        "Amount": np.random.uniform(10.0, 200.0, size=n_samples),
        "Class": np.array([0] * 18 + [1] * 2),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)

    df = pd.DataFrame(data)
    X = df[FEATURE_COLUMNS]

    pipeline = create_pipeline(contamination=0.1, random_state=42)
    pipeline.fit(X)

    return pipeline, df.iloc[[0]]


def test_predict_sample(sample_model_and_row):
    """Test predict_sample outputs boolean verdict and float anomaly score."""
    model, sample_row = sample_model_and_row
    is_fraud, anomaly_score = predict_sample(model, sample_row)

    assert isinstance(is_fraud, (bool, np.bool_))
    assert isinstance(anomaly_score, (float, np.floating))


def test_get_eval_metrics_returns_valid_dict():
    """Test get_eval_metrics returns a valid dictionary with metrics."""
    metrics = get_eval_metrics("models/eval_metrics.json")
    assert isinstance(metrics, dict)
    assert "roc_auc" in metrics
    assert "precision" in metrics


def test_get_sample_dataset_fallback():
    """Test get_sample_dataset returns DataFrame with required columns even if path doesn't exist."""
    df = get_sample_dataset("non_existent_data_path.csv")
    assert isinstance(df, pd.DataFrame)
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_plot_confusion_matrix():
    """Test plot_confusion_matrix generates valid Plotly Figure without error."""
    cm = [[56788, 76], [65, 33]]
    fig = plot_confusion_matrix(cm)
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0

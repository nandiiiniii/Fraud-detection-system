"""Unit tests for data utilities in src/utils.py."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from src.utils import load_data, split_data


@pytest.fixture
def synthetic_df():
    """Create a synthetic 20-row DataFrame mimicking credit card transactions."""
    np.random.seed(42)
    n_samples = 20
    data = {
        "Time": np.arange(n_samples, dtype=float),
        "Amount": np.random.uniform(1.0, 500.0, size=n_samples),
        "Class": np.array([0] * 16 + [1] * 4),  # 20% positive class
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_csv_file(tmp_path, synthetic_df):
    """Write synthetic dataframe to a temporary CSV file."""
    csv_file = tmp_path / "synthetic_creditcard.csv"
    synthetic_df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_load_data_valid_csv(synthetic_csv_file, synthetic_df):
    """Test load_data successfully loads a valid CSV and matches original shape."""
    df = load_data(synthetic_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == synthetic_df.shape
    assert list(df.columns) == list(synthetic_df.columns)
    assert not df.isnull().values.any()


def test_load_data_file_not_found(tmp_path):
    """Test load_data raises FileNotFoundError for non-existent file."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_data(str(non_existent))


def test_load_data_empty_file(tmp_path):
    """Test load_data raises ValueError for empty file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        load_data(str(empty_file))


def test_split_data_shapes_and_no_nans(synthetic_df):
    """Test split_data produces correct shapes without introducing NaNs."""
    train_df, test_df = split_data(synthetic_df, test_size=0.2, random_state=42)

    assert len(train_df) == 16
    assert len(test_df) == 4
    assert train_df.shape[1] == synthetic_df.shape[1]
    assert test_df.shape[1] == synthetic_df.shape[1]
    assert not train_df.isnull().values.any()
    assert not test_df.isnull().values.any()


def test_split_data_stratification(synthetic_df):
    """Test that stratified split preserves the class ratio in train and test."""
    train_df, test_df = split_data(synthetic_df, test_size=0.2, random_state=42)

    # Synthetic data has 16 zeros and 4 ones (ratio 0.20)
    train_fraud_ratio = train_df["Class"].mean()
    test_fraud_ratio = test_df["Class"].mean()

    assert train_fraud_ratio == pytest.approx(0.20, abs=0.05)
    assert test_fraud_ratio == pytest.approx(0.20, abs=0.05)
    assert train_df["Class"].sum() == 3
    assert test_df["Class"].sum() == 1


def test_split_data_missing_class_column():
    """Test split_data raises KeyError when 'Class' column is absent."""
    df_no_class = pd.DataFrame({"V1": [1.0, 2.0], "Amount": [10.0, 20.0]})
    with pytest.raises(KeyError):
        split_data(df_no_class)

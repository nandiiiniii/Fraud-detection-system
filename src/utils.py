"""Data loading and preprocessing utilities for fraud detection."""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        path: File path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Data file at {file_path} is empty.")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets with stratification on 'Class'.

    Args:
        df: Input DataFrame containing the 'Class' column.
        test_size: Proportion of the dataset to include in the test split (default 0.2).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df).

    Raises:
        KeyError: If 'Class' column is not present in the DataFrame.
        ValueError: If dataset has fewer than 2 samples per class for stratification.
    """
    if "Class" not in df.columns:
        raise KeyError("Input DataFrame must contain 'Class' column for stratified splitting.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

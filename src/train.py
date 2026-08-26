"""Isolation Forest model training pipeline for fraud anomaly detection."""

from pathlib import Path
from typing import Optional, Union, List
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import load_data, split_data


FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
DEFAULT_CONTAMINATION: float = 0.00173
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_MODEL_PATH: str = "models/model.pkl"


def create_pipeline(
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = 100,
) -> Pipeline:
    """
    Build scikit-learn Pipeline with feature scaling and Isolation Forest.

    Args:
        contamination: The amount of contamination of the data set.
        random_state: Seed for reproducible random numbers.
        n_estimators: The number of base estimators in the ensemble.

    Returns:
        Pipeline: Preprocessing and Isolation Forest pipeline.
    """
    pca_features = [f"V{i}" for i in range(1, 29)]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_time_amount", StandardScaler(), ["Time", "Amount"]),
            ("pass_pca", "passthrough", pca_features),
        ],
        remainder="drop",
    )

    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", isolation_forest),
        ]
    )


def train_model(
    train_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = DEFAULT_MODEL_PATH,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """
    Train Isolation Forest model on training DataFrame and optionally save to disk.

    Args:
        train_df: DataFrame containing features (Time, V1..V28, Amount).
        model_path: Path where the trained model should be saved (optional).
        contamination: Contamination parameter for Isolation Forest.
        random_state: Random state for reproducibility.

    Returns:
        Pipeline: Trained pipeline.
    """
    missing_cols = [col for col in FEATURE_COLUMNS if col not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Training data missing required feature columns: {missing_cols}")

    X_train = train_df[FEATURE_COLUMNS]

    pipeline = create_pipeline(
        contamination=contamination,
        random_state=random_state,
    )

    pipeline.fit(X_train)

    if model_path is not None:
        save_path = Path(model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)

    return pipeline


def run_training(
    data_path: Union[str, Path] = "data/creditcard.csv",
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """
    Full training workflow: loads data, splits, trains model, and saves to model_path.
    """
    df = load_data(data_path)
    train_df, _ = split_data(df, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)
    return train_model(train_df, model_path=model_path)


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    print(f"Starting training on {data_file}...")
    run_training(data_path=data_file)
    print(f"Model successfully trained and saved to {DEFAULT_MODEL_PATH}")

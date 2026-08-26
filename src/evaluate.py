"""Model evaluation module for fraud anomaly detection."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.train import FEATURE_COLUMNS
from src.utils import load_data, split_data


DEFAULT_METRICS_PATH: str = "models/eval_metrics.json"


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """
    Evaluate trained Isolation Forest model on test dataset.

    Args:
        model: Trained pipeline or Isolation Forest model.
        X_test: Test features (Time, V1..V28, Amount).
        y_test: Ground truth binary labels (0 = Normal, 1 = Fraud).

    Returns:
        Dict[str, Any]: Evaluation metrics including precision, recall, F1, ROC-AUC,
                        and confusion matrix.
    """
    # Isolation Forest: 1 for inlier (Normal), -1 for outlier (Fraud)
    raw_pred = model.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)

    # Anomaly scores: lower decision_function means more abnormal
    # Invert decision score so higher value indicates higher fraud likelihood
    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_test)
        anomaly_scores = -decision_scores
    elif hasattr(model, "score_samples"):
        anomaly_scores = -model.score_samples(X_test)
    else:
        anomaly_scores = y_pred

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, anomaly_scores))
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "total_test_samples": int(len(y_test)),
        "actual_fraud_count": int(np.sum(y_test == 1)),
        "predicted_fraud_count": int(np.sum(y_pred == 1)),
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path] = DEFAULT_METRICS_PATH) -> None:
    """Save metrics dictionary to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_evaluation(
    data_path: Union[str, Path] = "data/creditcard.csv",
    model_path: Union[str, Path] = "models/model.pkl",
    metrics_path: Union[str, Path] = DEFAULT_METRICS_PATH,
) -> Dict[str, Any]:
    """Load model and dataset, evaluate on test set, and save metrics."""
    df = load_data(data_path)
    _, test_df = split_data(df, test_size=0.2, random_state=42)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Class"]

    model = joblib.load(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)

    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run_evaluation()

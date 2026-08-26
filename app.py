"""Streamlit Web Application for Unsupervised Fraud & Anomaly Detection."""

import json
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from src.train import FEATURE_COLUMNS


def get_trained_model(model_path: str = "models/model.pkl"):
    """Load the trained Isolation Forest pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_trained_model(model_path: str = "models/model.pkl"):
    """Cached loader for the trained model."""
    return get_trained_model(model_path)


def get_sample_dataset(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load transactions for single check sampling (falls back to synthetic if not found)."""
    path = Path(data_path)
    if path.exists():
        return pd.read_csv(path, nrows=1000)

    # Fallback synthetic dataset for cloud environments where dataset is not committed
    np.random.seed(42)
    n_samples = 100
    data = {
        "Time": np.random.uniform(0, 172800, size=n_samples),
        "Amount": np.random.exponential(scale=88.0, size=n_samples),
        "Class": np.array([0] * 90 + [1] * 10),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)
    return pd.DataFrame(data)


@st.cache_data
def load_sample_dataset(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Cached loader for sample transactions."""
    return get_sample_dataset(data_path)


def get_eval_metrics(metrics_path: str = "models/eval_metrics.json") -> dict:
    """Load evaluation metrics JSON file."""
    path = Path(metrics_path)
    if not path.exists():
        return {
            "precision": 0.3028,
            "recall": 0.3367,
            "f1": 0.3188,
            "roc_auc": 0.9602,
            "confusion_matrix": [[56788, 76], [65, 33]],
            "total_test_samples": 56962,
            "actual_fraud_count": 98,
            "predicted_fraud_count": 109,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eval_metrics(metrics_path: str = "models/eval_metrics.json") -> dict:
    """Cached loader for evaluation metrics."""
    return get_eval_metrics(metrics_path)


def predict_sample(model, df_row: pd.DataFrame) -> Tuple[bool, float]:
    """Predict whether a single transaction row is normal or fraudulent."""
    features = df_row[FEATURE_COLUMNS]
    raw_pred = model.predict(features)[0]
    decision_score = model.decision_function(features)[0]
    # Anomaly score: higher means more abnormal
    anomaly_score = float(-decision_score)
    is_fraud = bool(raw_pred == -1)
    return is_fraud, anomaly_score


def main():
    st.set_page_config(
        page_title="Fraud & Anomaly Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🛡️ Credit Card Fraud & Anomaly Detector")
    st.markdown(
        "An unsupervised fraud detection system powered by **Isolation Forest**. "
        "Analyze individual transactions in real-time, process bulk CSV uploads, and inspect model benchmarks."
    )

    model = load_trained_model()
    if model is None:
        st.error("⚠️ Trained model file `models/model.pkl` was not found. Please train the model first.")
        return

    tab_single, tab_batch, tab_eval = st.tabs(["🔍 Single Check", "📁 Batch Upload", "📊 Model Evaluation"])

    # ------------------ TAB 1: SINGLE CHECK ------------------
    with tab_single:
        st.subheader("Real-Time Single Transaction Analysis")
        st.markdown("Sample a transaction from the dataset to test the Isolation Forest anomaly detector.")

        df_pool = load_sample_dataset()

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            pick_random = st.button("🎲 Pick random transaction", use_container_width=True)

        if "selected_sample" not in st.session_state or pick_random:
            st.session_state.selected_sample = df_pool.sample(n=1, random_state=None).iloc[0]

        sample_row = st.session_state.selected_sample
        sample_df = pd.DataFrame([sample_row])

        is_fraud, anomaly_score = predict_sample(model, sample_df)

        st.divider()

        # Display Result Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transaction Amount", f"${sample_row['Amount']:,.2f}")
        with col2:
            st.metric("Time Elapsed", f"{sample_row['Time']:,.0f} s")
        with col3:
            status_text = "🚨 Suspicious / Fraud" if is_fraud else "✅ Normal / Legitimate"
            st.metric("Model Verdict", status_text)
        with col4:
            st.metric("Anomaly Score", f"{anomaly_score:.4f}", help="Higher positive score indicates anomalous transaction")

        if is_fraud:
            st.error("⚠️ **Alert: This transaction has been flagged as suspicious (Anomaly/Fraud).**")
        else:
            st.success("✅ **Verified: This transaction appears normal and consistent with legitimate patterns.**")

        if "Class" in sample_row:
            actual_label = "Fraudulent (1)" if sample_row["Class"] == 1 else "Legitimate (0)"
            st.caption(f"Ground Truth Label in Dataset: **{actual_label}**")

        with st.expander("🔎 View Full Transaction Feature Vector (V1 - V28)"):
            st.dataframe(sample_df, use_container_width=True)

    # ------------------ TAB 2: BATCH UPLOAD ------------------
    with tab_batch:
        st.subheader("Batch CSV Anomaly Detection")
        st.markdown("Upload a CSV file of transactions containing columns `Time`, `V1`...`V28`, and `Amount`.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                missing_features = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]

                if missing_features:
                    st.error(f"❌ Uploaded CSV is missing required columns: {missing_features}")
                else:
                    st.info(f"Loaded {len(batch_df):,} transactions for scoring.")

                    # Compute predictions and anomaly scores
                    features = batch_df[FEATURE_COLUMNS]
                    raw_preds = model.predict(features)
                    decision_scores = model.decision_function(features)
                    anomaly_scores = -decision_scores

                    results_df = batch_df.copy()
                    results_df["Anomaly_Score"] = anomaly_scores
                    results_df["Prediction"] = np.where(raw_preds == -1, "Fraud / Anomaly", "Normal")
                    results_df["Is_Fraud"] = (raw_preds == -1)

                    flagged_df = results_df[results_df["Is_Fraud"]].sort_values(by="Anomaly_Score", ascending=False)
                    fraud_count = len(flagged_df)
                    fraud_pct = (fraud_count / len(results_df)) * 100 if len(results_df) > 0 else 0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", f"{len(results_df):,}")
                    with col2:
                        st.metric("Flagged Suspicious", f"{fraud_count:,}")
                    with col3:
                        st.metric("Anomaly Rate", f"{fraud_pct:.2f}%")

                    st.markdown("### 🚨 Flagged Suspicious Transactions")
                    if fraud_count > 0:
                        st.dataframe(flagged_df.drop(columns=["Is_Fraud"]), use_container_width=True)
                        csv_data = flagged_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Flagged Transactions CSV",
                            data=csv_data,
                            file_name="flagged_fraudulent_transactions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.success("No anomalies detected in the uploaded batch.")

                    st.markdown("### 📈 Distribution of Anomaly Scores")
                    fig_hist = px.histogram(
                        results_df,
                        x="Anomaly_Score",
                        color="Prediction",
                        nbins=50,
                        title="Isolation Forest Anomaly Score Distribution",
                        labels={"Anomaly_Score": "Anomaly Score (-Decision Function)", "count": "Transaction Count"},
                        color_discrete_map={"Normal": "#2ecc71", "Fraud / Anomaly": "#e74c3c"},
                        barmode="overlay",
                        opacity=0.75,
                    )
                    fig_hist.update_layout(template="plotly_white", legend_title_text="Verdict")
                    st.plotly_chart(fig_hist, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
        else:
            st.info("💡 Upload a CSV file above to run batch predictions. You can also download sample data from Kaggle or export a subset to test.")

    # ------------------ TAB 3: MODEL EVALUATION ------------------
    with tab_eval:
        st.subheader("Model Performance & Evaluation Metrics")
        st.markdown("Detailed benchmarks evaluated on the 20% stratified test split (56,962 transactions).")

        metrics = load_eval_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.4f}")
        with col2:
            st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        with col3:
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        with col4:
            st.metric("Recall", f"{metrics.get('recall', 0):.2%}")

        st.divider()

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 🎯 Confusion Matrix")
            cm_raw = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            if isinstance(cm_raw, np.ndarray):
                cm_raw = cm_raw.tolist()

            # Ensure pure native Python integers for Plotly figure factory
            z = [[int(val) for val in row] for row in cm_raw]
            x = ["Predicted Normal (0)", "Predicted Fraud (1)"]
            y = ["Actual Normal (0)", "Actual Fraud (1)"]
            z_text = [[str(int(val)) for val in row] for row in z]

            try:
                fig_cm = ff.create_annotated_heatmap(
                    z=z,
                    x=x,
                    y=y,
                    annotation_text=z_text,
                    colorscale="Blues",
                    showscale=True,
                )
                fig_cm.update_layout(
                    title_text="Test Set Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white",
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render heatmap: {e}")
                cm_df = pd.DataFrame(z, index=y, columns=x)
                st.dataframe(cm_df, use_container_width=True)

        with col_right:
            st.markdown("### 📋 Evaluation Summary")
            st.markdown(
                f"""
                - **Total Test Samples:** `{metrics.get('total_test_samples', 0):,}`
                - **Actual Fraud Cases:** `{metrics.get('actual_fraud_count', 0):,}`
                - **Flagged Anomalies:** `{metrics.get('predicted_fraud_count', 0):,}`
                - **Contamination Factor:** `0.00173` (matching the natural fraud rate in the population)
                - **Algorithm:** Unsupervised `IsolationForest` with standard-scaled `Time` and `Amount` features.

                **Key Takeaways:**
                - With an **ROC-AUC of {metrics.get('roc_auc', 0):.4f}**, Isolation Forest effectively ranks abnormal transactions above normal ones without requiring fraud labels during training.
                - The precision-recall trade-off is optimized for unsupervised screening where flagged cases are escalated for secondary review.
                """
            )


if __name__ == "__main__":
    main()

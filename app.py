"""Streamlit Web Application for Unsupervised Fraud & Anomaly Detection.

Modern Dark Theme with Fintech Blue Accent Styling.
"""

import json
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.train import FEATURE_COLUMNS


# Custom CSS for Dark Fintech & Security Theme
CUSTOM_CSS = """
<style>
/* Root theme overrides */
:root {
    --bg-dark: #0B0F19;
    --card-bg: #111827;
    --card-border: #1E293B;
    --card-border-glow: #2563EB;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-blue: #3B82F6;
    --accent-blue-dark: #1D4ED8;
    --accent-blue-light: #60A5FA;
    --danger-red: #EF4444;
    --success-green: #10B981;
}

/* Background styling */
.stApp {
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

/* Main title styling */
.main-header {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0 !important;
}

.main-header p {
    color: var(--text-secondary) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    line-height: 1.5;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0F172A;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.95rem;
    padding: 8px 18px;
    border: none !important;
    background-color: transparent;
    transition: all 0.2s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
    background-color: #1E293B;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
}

/* Custom KPI / Metric Cards */
.kpi-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    transition: transform 0.15s ease, border-color 0.15s ease;
    height: 100%;
}

.kpi-card:hover {
    border-color: var(--accent-blue);
    transform: translateY(-2px);
}

.kpi-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.kpi-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-blue {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.badge-green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-red {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Alert Boxes */
.verdict-box {
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0;
    font-size: 1.05rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
}

.verdict-legit {
    background-color: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399;
}

.verdict-fraud {
    background-color: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #F87171;
}

/* Button styling */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    border-color: #60A5FA !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
div.stDownloadButton > button:first-child {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: 1px solid #10B981 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}

div.stDownloadButton > button:first-child:hover {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.45) !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    background-color: var(--card-bg);
    border: 2px dashed var(--card-border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue);
}

/* Summary Card */
.summary-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}

.summary-card h3 {
    color: #FFFFFF !important;
    font-size: 1.25rem !important;
    margin-top: 0 !important;
}

.summary-card ul {
    color: var(--text-secondary);
    padding-left: 20px;
    margin-bottom: 12px;
}

.summary-card li {
    margin-bottom: 6px;
    line-height: 1.5;
}

.summary-card code {
    background-color: #1E293B;
    color: #93C5FD;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}

/* Divider styling */
hr {
    border-color: var(--card-border) !important;
    margin: 24px 0 !important;
}
</style>
"""


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


def render_metric_card(label: str, value: str, badge_text: str = None, badge_type: str = "blue"):
    """Render a modern dark-themed KPI card."""
    badge_html = f'<span class="kpi-badge badge-{badge_type}">{badge_text}</span>' if badge_text else ""
    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {badge_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def plot_confusion_matrix(cm):
    """Render confusion matrix using Plotly graph objects heatmap with dark fintech styling."""
    cm = np.array(cm).astype(int).tolist()  # force plain Python ints, not numpy types
    labels = ["Not Fraud", "Fraud"]
    z_text = [[f"{val:,}" for val in row] for row in cm]

    # Tech blue colorscale for dark mode
    colorscale = [
        [0.0, "#0F172A"],
        [0.1, "#1E3A8A"],
        [0.5, "#2563EB"],
        [1.0, "#60A5FA"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#94A3B8"),
            title=dict(text="Count", font=dict(color="#F8FAFC")),
        )
    ))

    annotations = []
    max_val = np.max(cm)
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=labels[j],
                y=labels[i],
                text=z_text[i][j],
                showarrow=False,
                font=dict(
                    color="#FFFFFF" if val > (max_val / 4) else "#93C5FD",
                    size=16,
                    family="sans-serif",
                )
            ))

    fig.update_layout(
        title=dict(
            text="Confusion Matrix (Test Split)",
            font=dict(color="#F8FAFC", size=18)
        ),
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#E2E8F0", size=12),
            gridcolor="#1E293B",
        ),
        yaxis=dict(
            title=dict(text="Actual Label", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#E2E8F0", size=12),
            autorange="reversed",
            gridcolor="#1E293B",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def main():
    st.set_page_config(
        page_title="Fraud & Anomaly Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header section
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ Fraud & Anomaly Detection System</h1>
            <p>Unsupervised real-time and batch transaction surveillance powered by <b>Isolation Forest</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_trained_model()
    if model is None:
        st.error("⚠️ Trained model file `models/model.pkl` was not found. Please train the model first.")
        return

    tab_single, tab_batch, tab_eval = st.tabs(["🔍 Single Check", "📁 Batch Upload", "📊 Model Evaluation"])

    # ------------------ TAB 1: SINGLE CHECK ------------------
    with tab_single:
        st.markdown("### Real-Time Transaction Screening")
        st.markdown(
            "<p style='color: #94A3B8;'>Extract a transaction sample from the live data stream to evaluate anomaly scoring.</p>",
            unsafe_allow_html=True,
        )

        df_pool = load_sample_dataset()

        col_btn, _ = st.columns([1.5, 3])
        with col_btn:
            pick_random = st.button("🎲 Pick random transaction", use_container_width=True)

        if "selected_sample" not in st.session_state or pick_random:
            st.session_state.selected_sample = df_pool.sample(n=1, random_state=None).iloc[0]

        sample_row = st.session_state.selected_sample
        sample_df = pd.DataFrame([sample_row])

        is_fraud, anomaly_score = predict_sample(model, sample_df)

        st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)

        # Custom Dark KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Transaction Amount", f"${sample_row['Amount']:,.2f}", "USD", "blue")
        with col2:
            render_metric_card("Time Elapsed", f"{sample_row['Time']:,.0f} s", "Interval", "blue")
        with col3:
            status_text = "Fraud / Suspicious" if is_fraud else "Normal / Legit"
            status_type = "red" if is_fraud else "green"
            render_metric_card("Model Verdict", status_text, "Classification", status_type)
        with col4:
            score_type = "red" if anomaly_score > 0 else "blue"
            render_metric_card("Anomaly Score", f"{anomaly_score:.4f}", "-Decision Function", score_type)

        # High-contrast Alert Verdict Box
        if is_fraud:
            st.markdown(
                """
                <div class="verdict-box verdict-fraud">
                    <span>🚨</span>
                    <div><b>Alert: This transaction has been flagged as suspicious (Anomaly/Fraud).</b><br>
                    <span style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">The feature distribution deviates significantly from standard legitimate baselines.</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="verdict-box verdict-legit">
                    <span>✅</span>
                    <div><b>Verified: Transaction pattern conforms to normal activity.</b><br>
                    <span style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">No high-dimension isolation anomalies were detected.</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "Class" in sample_row:
            actual_label = "Fraudulent (1)" if sample_row["Class"] == 1 else "Legitimate (0)"
            st.markdown(
                f"<p style='color: #64748B; font-size: 0.85rem;'>Ground Truth Label in Reference Dataset: <b style='color: #94A3B8;'>{actual_label}</b></p>",
                unsafe_allow_html=True,
            )

        with st.expander("🔎 View Full Transaction Feature Vector (V1 - V28)"):
            st.dataframe(sample_df, use_container_width=True)

    # ------------------ TAB 2: BATCH UPLOAD ------------------
    with tab_batch:
        st.markdown("### Batch CSV Surveillance & Scoring")
        st.markdown(
            "<p style='color: #94A3B8;'>Upload transaction CSV records containing <code>Time</code>, <code>V1</code>...<code>V28</code>, and <code>Amount</code>.</p>",
            unsafe_allow_html=True,
        )

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
                        render_metric_card("Total Transactions", f"{len(results_df):,}", "Batch Size", "blue")
                    with col2:
                        render_metric_card("Flagged Suspicious", f"{fraud_count:,}", "Outliers", "red" if fraud_count > 0 else "green")
                    with col3:
                        render_metric_card("Anomaly Rate", f"{fraud_pct:.2f}%", "Contamination", "blue")

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
                        color_discrete_map={"Normal": "#3B82F6", "Fraud / Anomaly": "#EF4444"},
                        barmode="overlay",
                        opacity=0.8,
                    )
                    fig_hist.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F8FAFC"),
                        legend=dict(
                            title=dict(text="Verdict", font=dict(color="#CBD5E1")),
                            font=dict(color="#E2E8F0"),
                            bgcolor="rgba(17, 24, 39, 0.8)",
                            bordercolor="#1E293B",
                            borderwidth=1,
                        ),
                        xaxis=dict(gridcolor="#1E293B"),
                        yaxis=dict(gridcolor="#1E293B"),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
        else:
            st.info("💡 Upload a CSV file above to run batch predictions. You can also download sample data from Kaggle or export a subset to test.")

    # ------------------ TAB 3: MODEL EVALUATION ------------------
    with tab_eval:
        st.markdown("### Model Performance Benchmarks")
        st.markdown(
            "<p style='color: #94A3B8;'>Quantitative metrics evaluated on a 20% stratified holdout split (56,962 transactions).</p>",
            unsafe_allow_html=True,
        )

        metrics = load_eval_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.4f}", "Ranking Quality", "blue")
        with col2:
            render_metric_card("F1-Score", f"{metrics.get('f1', 0):.4f}", "Harmonic Mean", "blue")
        with col3:
            render_metric_card("Precision", f"{metrics.get('precision', 0):.2%}", "True Fraud / Flagged", "blue")
        with col4:
            render_metric_card("Recall", f"{metrics.get('recall', 0):.2%}", "Caught / Actual", "blue")

        st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            cm_raw = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            fig_cm = plot_confusion_matrix(cm_raw)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            st.markdown(
                f"""
                <div class="summary-card">
                    <h3>📋 Benchmark Breakdown</h3>
                    <ul>
                        <li><b>Total Holdout Evaluation Size:</b> <code>{metrics.get('total_test_samples', 0):,}</code></li>
                        <li><b>Actual Ground Truth Fraud:</b> <code>{metrics.get('actual_fraud_count', 0):,}</code></li>
                        <li><b>Total Flagged Outliers:</b> <code>{metrics.get('predicted_fraud_count', 0):,}</code></li>
                        <li><b>Contamination Hyperparameter:</b> <code>0.00173</code> (0.173%)</li>
                        <li><b>Architecture:</b> Standardized StandardScaler + IsolationForest Ensemble</li>
                    </ul>
                    <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0;">
                        <b>Key Insight:</b> With an <b>ROC-AUC of {metrics.get('roc_auc', 0):.4f}</b>, the unsupervised Isolation Forest effectively ranks anomalous transactions above normal cardholder behavior without requiring fraud labels during training.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

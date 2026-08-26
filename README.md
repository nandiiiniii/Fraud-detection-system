# 🛡️ Fraud & Anomaly Detection System

An unsupervised machine learning system designed to detect fraudulent credit card transactions in real-time and batch streams using **Isolation Forest**. Includes an interactive multi-tab **Streamlit** dashboard for transaction screening, bulk CSV anomaly scoring, and model benchmark visualization.

---

## 🚀 Live Demo
- **Streamlit Community Cloud:https://fraud-detection-system-2ztufkkuu8mwwiyhlqk2fv.streamlit.app/
- **Repository:** [https://github.com/nandiiiniii/Fraud-detection-system](https://github.com/nandiiiniii/Fraud-detection-system)

---

## 📌 Problem Statement
Credit card fraud costs financial institutions billions of dollars annually. Identifying fraudulent transactions is notoriously difficult due to:
1. **Extreme Class Imbalance:** Legitimate transactions outnumber fraud by over 500 to 1 (~0.172% fraud rate).
2. **Evolving Attack Vectors:** Fraudsters continually adapt techniques, making rule-based engines and purely supervised models fragile against novel attack patterns.
3. **Latency & Cost of False Positives:** Screening millions of transactions in sub-second latency while minimizing friction for legitimate cardholders is essential.

This project addresses these challenges via **unsupervised anomaly detection** using an **Isolation Forest**, detecting novel anomalous behavior without requiring labeled fraud data during training.

---

## 🖼️ Application Overview & Features

The Streamlit web application provides three dedicated modules:

### 1. 🔍 Single Check
- Sample individual transactions from the data pool or generate transactions.
- Computes anomaly score and renders instant verdict (**Normal / Legitimate** vs **Suspicious / Fraud**).
- Displays key transaction attributes (Amount, Time elapsed, and PCA feature vectors `V1`–`V28`).

### 2. 📁 Batch Upload
- Process bulk CSV transaction batches conforming to standard feature schemas (`Time`, `V1`–`V28`, `Amount`).
- Automatically computes anomaly scores, flags anomalous records, and provides summary metrics.
- Export flagged suspicious transactions as a downloadable CSV.
- Visualizes the continuous anomaly score distribution with an interactive Plotly histogram.

### 3. 📊 Model Evaluation
- Displays quantitative performance metrics evaluated on a 20% stratified test split (56,962 transactions).
- Renders an interactive Plotly annotated heatmap of the **Confusion Matrix**.
- Explains unsupervised anomaly scoring dynamics and production screening trade-offs.

---

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Machine Learning & Preprocessing:** scikit-learn, numpy, pandas, joblib
- **Visualization & Frontend:** Streamlit, Plotly
- **Testing & Quality Assurance:** pytest
- **CI / Automation:** GitHub Actions (Automated multi-version Python test runner)

---

## 📂 Repository Structure
```
fraud-detector/
  ├── .github/
  │   └── workflows/
  │       └── test.yml          # GitHub Actions automated test workflow
  ├── data/
  │   └── .gitkeep              # Data folder placeholder (creditcard.csv ignored)
  ├── models/
  │   ├── .gitkeep              # Models folder placeholder
  │   ├── eval_metrics.json     # Saved evaluation metrics & confusion matrix
  │   └── model.pkl             # Serialized Isolation Forest pipeline
  ├── src/
  │   ├── __init__.py
  │   ├── train.py              # Training pipeline with feature scaling & Isolation Forest
  │   ├── evaluate.py           # Evaluation metrics computation (ROC-AUC, F1, Precision, Recall)
  │   └── utils.py              # Data ingestion and stratified train/test splitting
  ├── tests/
  │   ├── __init__.py
  │   ├── test_utils.py         # Unit tests for data loading & splitting
  │   ├── test_train.py         # Unit tests for training pipeline & model inference
  │   ├── test_evaluate.py      # Unit tests for evaluation & metrics export
  │   └── test_app.py           # Unit tests for Streamlit application helper functions
  ├── app.py                    # Streamlit web application
  ├── requirements.txt          # Python dependencies
  ├── .gitignore                # Git ignore configuration (ignoring datasets & caches)
  ├── LICENSE                   # MIT License
  └── README.md                 # Project documentation
```

---

## 💾 Dataset Download Instructions
This project utilizes the Kaggle **Credit Card Fraud Detection** dataset:
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **License:** Open Database License (ODbL)
- **Size:** 284,807 transactions across 30 numeric feature variables (`Time`, `V1`–`V28` PCA transformed components, and `Amount`).

### Steps to Download:
1. Download `creditcard.csv` from Kaggle.
2. Place the file inside the `data/` directory:
   ```bash
   data/creditcard.csv
   ```
*(Note: As per repository safety rules, `creditcard.csv` is excluded from git commits via `.gitignore`.)*

---

## 💻 How to Run Locally

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/nandiiiniii/Fraud-detection-system.git
cd Fraud-detection-system

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m pytest -v
```

### 3. Train Model & Run Evaluation (Optional)
```bash
# Train Isolation Forest on data/creditcard.csv
python -m src.train

# Evaluate on test split and regenerate models/eval_metrics.json
python -m src.evaluate
```

### 4. Launch Streamlit Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Approach

### Isolation Forest for Unsupervised Anomaly Detection
- **Mechanism:** Isolation Forest recursively isolates observations by randomly selecting a feature and randomly selecting a split value between the minimum and maximum values of that feature. Because anomalies are few and structurally distinct, they require fewer random partitions to isolate (shorter average tree path length).
- **Contamination Factor:** Configured to `0.00173` (0.173%), precisely matching the empirical rate of fraudulent transactions in the dataset.
- **Preprocessing Pipeline:**
  - `Time` and `Amount` are standardized using `StandardScaler` to normalize variance and prevent large dollar amounts from disproportionately dominating splits.
  - Anonymized PCA features `V1` through `V28` are preserved in the pipeline.
- **Anomaly Score Formulation:**
  - Anomaly score \( s(x, n) = - \text{decision\_function}(x) \), where higher positive scores represent stronger outlier characteristics.

---

## 📈 Evaluation Metrics

Evaluated on the **stratified 20% test holdout** (56,962 transactions; 98 actual frauds):

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9602** | Exceptional capability to rank fraudulent transactions higher than normal transactions |
| **Precision** | **30.28%** | Ratio of true fraud cases among all flagged transactions |
| **Recall** | **33.67%** | Proportion of actual fraud cases successfully caught unsupervised |
| **F1-Score** | **0.3188** | Harmonic mean of precision and recall |
| **Test Set Size** | **56,962** | 20% holdout split stratified on `Class` |
| **Actual Fraud** | **98** | Number of positive fraud cases in test set |
| **Flagged Count** | **109** | Total transactions flagged by the model |

### Confusion Matrix Breakdown
| | Predicted Normal | Predicted Fraud |
| :--- | :--- | :--- |
| **Actual Normal** | 56,788 (TN) | 76 (FP) |
| **Actual Fraud** | 65 (FN) | 33 (TP) |

---

## 🔮 What I'd Improve With More Time
1. **Hybrid Semi-Supervised Architecture:** Combine Isolation Forest embeddings or autoencoder reconstruction errors as meta-features into an XGBoost / LightGBM classifier with focal loss.
2. **Deep Learning Anomaly Detection:** Implement Variational Autoencoders (VAEs) or Deep SVDD to capture complex nonlinear transaction embeddings.
3. **Adaptive Thresholding via Cost Matrix:** Introduce financial cost optimization (weighting the financial cost of false negatives vs merchant friction of false positives) to dynamically tune operational decision thresholds.
4. **Streaming Integration:** Connect Kafka / AWS Kinesis to process transactions in sub-10ms latency micro-batches with real-time alerting.
5. **Explainability (SHAP / TreeSHAP):** Integrate SHAP force plots to explain which specific features triggered an anomaly score for a given transaction.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

# 🔍 UPI Fraud Detector

**Intelligent UPI Transaction Fraud Detection & Risk Scoring System**

UPI Fraud Detector predicts the probability that a UPI transaction is fraudulent, converts that probability into a 0–100 risk score with a Low/Medium/High category, and explains every prediction using SHAP — turning a black-box classifier into an auditable decision-support tool.

![Dashboard Screenshot](docs/screenshots/prediction.png)

---

## 📌 Business Problem

UPI transactions happen at a volume where manual fraud review is impossible. Fintech platforms need a system that can:

- Flag a transaction as fraudulent in real time
- Express that risk in a way a non-technical reviewer can act on (a score and category, not just a probability)
- Justify *why* a transaction was flagged, since an unexplained fraud flag is hard to trust or act on
- Be served both programmatically (API) and through a human-facing tool (dashboard)

This project simulates that system end-to-end, from raw data to a deployable, explainable application.

---

## 👥 Who Is This For?

- **Fraud/Risk Analysts** — review flagged transactions with a concrete, ranked explanation instead of a single opaque number
- **Engineering Teams** — see a reference implementation of a leakage-safe ML pipeline with a clean train/serve split (shared feature logic between the API and dashboard)
- **Reviewers/Recruiters** — evaluate not just model accuracy, but the process of detecting and removing data leakage in a messy, synthetic dataset — arguably the most realistic part of this project

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Interactive Dashboard | Streamlit |
| Containerization | Docker |
| Testing | Pytest, Pytest-Cov |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

---

## 📊 Dataset

- **Source:** [Kaggle — UPI Fraud Detection Dataset](https://www.kaggle.com/datasets/omsshete/upi-fraud-detection-dataset)
- **Size:** 26,393 transactions, 65 raw columns
- **Target Variable:** `is_fraud` — `0` (genuine) / `1` (fraud), 17.2% fraud rate

No real UPI transaction data is publicly available — this dataset is synthetically generated. That turned out to materially shape this project, since the synthetic generation process embedded the fraud label into many unrelated columns rather than producing a realistic, noisy signal.

### Data Quality Issues Found

| Issue | Resolution |
|---|---|
| `url_referrer`, `request_description` columns >97% missing | Dropped |
| `timestamp` column in corrupted `MM:SS.s` format, not a real datetime | Dropped — `transaction_time_of_day` (already 0–23) used instead |
| `social_media_presence`, `upi_handle_age`, `handle_contains_official_terms` had zero variance (single value across all rows) | Dropped |
| `ip_address_freq` (engineered) had zero variance — every IP appeared exactly once | Dropped |

---

## 🚨 Key Finding: Extensive Data Leakage

During EDA, several columns showed a **deterministic** relationship with the target — e.g. `unusual_device_flag = 1` corresponded to **100% fraud**, every time, across hundreds of rows. This is a classic synthetic-data artifact: the generator embedded the label into multiple unrelated columns instead of producing a realistic signal.

A systematic scan was run across every column, checking each one's fraud rate per subgroup with a minimum sample-size threshold (to rule out small-sample noise). This identified **31 leakage columns during EDA** — including every `unusual_*_flag` field, OTP/authentication-attempt counters, handle-related fields, and even `merchant_category_code = "unknown"` mapping to exactly 100% fraud.

The leakage hunt didn't stop after EDA. A Logistic Regression baseline trained on the "cleaned" 18-column set still returned a suspicious **ROC-AUC of 1.0**. Re-examining feature coefficients surfaced **3 more leakage columns missed in EDA**:

| Feature | Leakage Pattern |
|---|---|
| `merchant_id_freq` (engineered) | Frequency = 1 → 100% fraud |
| `device_id_freq` (engineered) | Frequency = 1 → 80% fraud concentration |
| `receiver_account_age` | Lowest decile → 85% fraud, every other decile → 0% |

Every removal was verified, not assumed. Before keeping `receiver_transaction_history` (the single strongest remaining feature), a diagnostic test retrained the model **without** it — ROC-AUC dropped from 0.994 to 0.959 but did not collapse, confirming the model wasn't single-feature-dependent. A later isolated perturbation test (holding all other features constant, varying only this one) also confirmed the model's behavior was internally consistent with what EDA had shown, not contradictory.

**Final feature set: 18 columns**, each individually checked for leakage rather than included by default.

This process — and the willingness to keep digging even after the dataset "looked clean" twice — is the main engineering contribution of this project, beyond the modeling itself.

---

## 🧪 Feature Engineering

| Feature | How It's Computed | Why It Matters |
|---|---|---|
| `user_id_freq` | Frequency-encoded sender ID, fit on train only | Captures sender activity level without leaking test-set information |
| `transaction_time_of_day` | Used directly (already 0–23 in source data) | EDA showed fraud concentrated in early-morning hours (peak around hour 6) |
| `receiver_transaction_history` | Used directly | Strongest legitimate predictor — gradual, non-deterministic relationship with fraud (62% → 4.8% → 0% across quartiles), unlike the deterministic leakage columns |
| All other 15 features | Genuine behavioral/transactional signals retained after leakage screening (e.g. `amount`, `input_pause_patterns`, `geographic_disparity`) | Selected only after passing the deterministic-leakage check |

**Note on leakage prevention:** the train/test split happens *before* frequency encoding and scaling. Both the frequency maps and the `StandardScaler` are fit exclusively on the training set, then applied to the test set and to live inference requests — unseen users/merchants/devices at inference time default to a frequency of 0 rather than crashing or leaking information.

---

## 🤖 Model Comparison

Three models were trained and evaluated on the final 18-feature set, with class imbalance (17.2% fraud) handled via `class_weight='balanced'` (or `scale_pos_weight` for XGBoost), and validated with 5-fold stratified cross-validation:

| Model | ROC-AUC (5-fold CV) | ROC-AUC (Test) | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9931 | 0.9943 | 0.950 | 0.797 | 0.956 | 0.869 |
| **Random Forest** | **0.9964** | **0.9967** | **0.981** | **0.987** | 0.902 | 0.943 |
| XGBoost | 0.9956 | 0.9961 | 0.978 | 0.950 | 0.921 | 0.935 |

### Why Accuracy Alone Is Not the Deciding Metric
With 17.2% fraud, a model that always predicts "genuine" would already score ~83% accuracy while being useless. **ROC-AUC and PR-AUC** (test PR-AUC: **0.9867**) were used as the primary decision criteria, with accuracy tracked only as a secondary reference.

### Final Model: Random Forest
Selected for the highest mean cross-validated ROC-AUC, the lowest variance across folds (std 0.0003), and the best precision for the fraud class in production use (fewer false alarms). Logistic Regression has marginally higher recall, but at a much lower precision (0.797 vs 0.987) — too many false positives for a production fraud queue.

> **Note on performance:** these scores are high relative to typical real-world fraud detection (commonly 0.75–0.85 ROC-AUC). This reflects the clean, synthetic nature of this dataset's behavioral signals rather than real-world fraud complexity, and is documented here rather than hidden.

---

## 🧠 Explainability (SHAP)

Every prediction — from both the API and the dashboard — is accompanied by a SHAP-based explanation showing the top contributing features and their direction of impact (e.g. *"receiver_transaction_history = 15 → decreased fraud risk"*). The `TreeExplainer` is **not** pickled as an artifact; it is recreated from the saved model at runtime, since SHAP explainers can be version-fragile to serialize.

---

## 🎯 Risk Categorization

Model probabilities are converted into three actionable tiers, configurable via `artifacts/risk_thresholds.json`:

| Risk Score | Risk Level | 
|---|---|
| 0 – 30 | 🟢 Low |
| 31 – 70 | 🟡 Medium |
| 71 – 100 | 🔴 High |

---

## 💡 Why This Project Is Different

This project uses a single Kaggle dataset, but the differentiation isn't the dataset — it's the process built around it:

- **Multi-round leakage detection**, including leakage discovered only *after* the dataset already looked clean from EDA, caught by interrogating a suspiciously perfect ROC-AUC rather than accepting it
- **Engineered features checked for leakage too** — frequency encoding isn't assumed safe just because it's "feature engineering"; `merchant_id_freq` and `device_id_freq` were both removed after failing the same leakage test applied to raw columns
- **A standalone dashboard with no train/serve duplication** — the Streamlit app imports the same prediction and explanation logic used by the FastAPI service rather than reimplementing it, and runs independently without requiring the API to be up
- **A production-shaped system**: Pydantic-validated API, pytest coverage across health/predict/feature-preparation logic, automated CI on every push, and a Docker image that can serve either the API or the dashboard

---

## 📁 Project Structure

```text
upi-fraud-detector/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   ├── features.py
│   ├── schemas.py
│   └── risk_scoring.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── explainability.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── artifacts/
│   ├── final_model.joblib
│   ├── preprocessor.joblib
│   ├── freq_maps.joblib
│   ├── feature_columns.joblib
│   └── risk_thresholds.json
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── examples/
│   └── sample_request.json
│
├── docs/
│   ├── architecture.md
│   └── screenshots/
│       ├── overview.png
│       ├── analytics.png
│       ├── prediction.png
│       └── api_docs.png
│
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_features.py
│
├── .github/workflows/ci.yml
├── .dockerignore
├── .gitignore
├── conftest.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Option 1 — Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/md-faiyazkhan/upi-fraud-detector.git
cd upi-fraud-detector
```

**2. Create a virtual environment and install dependencies**
```bash
conda create -n upi-fraud-env python=3.10
conda activate upi-fraud-env
pip install -r requirements.txt
```

**3. Download the dataset**

Download `fraud_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/omsshete/upi-fraud-detection-dataset) and place it in `data/raw/`.

**4. Run the notebooks in order**

```
notebooks/01_dataset_audit.ipynb
notebooks/02_eda.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_model_training.ipynb
```

This regenerates the processed data splits and model artifacts in `artifacts/`.

**5. Run the FastAPI service**
```bash
uvicorn app.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

**6. Run the Streamlit dashboard**
```bash
streamlit run dashboard/streamlit_app.py
```
Dashboard available at `http://localhost:8501`

---

### Option 2 — Docker

```bash
docker build -t upi-fraud-detector .
docker run -p 8000:8000 upi-fraud-detector
```

To run the Streamlit dashboard instead, inside the same image:
```bash
docker run -p 8501:8501 upi-fraud-detector streamlit run dashboard/streamlit_app.py --server.address=0.0.0.0
```

**Access**
- FastAPI Docs: `http://localhost:8000/docs`
- Streamlit Dashboard: `http://localhost:8501`

---

## 📡 API Reference

**Base URL:** `http://localhost:8000`

![API Docs Screenshot](docs/screenshots/api_docs.png)

**Health Check**
```
GET /health
```
```json
{ "status": "ok" }
```

**Prediction**
```
POST /predict
```

**Sample Request** (see also `examples/sample_request.json`):
```json
{
  "user_id": "U12345",
  "amount": 4500.0,
  "session_duration": 120,
  "receiver_transaction_history": 15,
  "transaction_amount_vs_sender_history": 1.2,
  "geographic_disparity": 0.05,
  "transaction_time_of_day": 14,
  "time_between_link_click_and_transaction": 30,
  "input_timing_consistency": 0.8,
  "keyboard_input_speed": 0.5,
  "input_pause_patterns": 0.1,
  "screen_active_time": 90,
  "geographic_location_vs_ip": 0.02,
  "background_data_usage": 0.3,
  "pin_entry_speed": 0.4,
  "request_amount_roundness": 0.1,
  "request_acceptance_rate": 0.9,
  "time_to_respond_to_request": 5
}
```

**Sample Response:**
```json
{
  "is_fraud": 0,
  "fraud_probability": 0.265,
  "risk_score": 26,
  "risk_category": "Low Risk",
  "top_reasons": [
    {"feature": "receiver_transaction_history", "shap_value": -0.361, "feature_value": 15},
    {"feature": "amount", "shap_value": 0.104, "feature_value": 4500.0},
    {"feature": "time_to_respond_to_request", "shap_value": 0.069, "feature_value": 5.0}
  ]
}
```

> `user_id` is matched against a frequency map built from the training data. Unseen user IDs default to a frequency of 0 rather than causing an error — the same logic used at training time for handling new entities.

---

## 🖥️ Dashboard

The Streamlit dashboard provides three pages over the same prediction logic used by the API (no duplicated code — it imports `app/predictor`-equivalent logic directly):

- **Overview** — total transactions, fraud count, fraud rate, and class distribution
- **Analytics** — fraud rate by transaction amount, hour of day, and receiver transaction history
- **Prediction** — a full transaction input form, live risk scoring, a SHAP-based explanation chart, and an IST timestamp on every prediction

It runs **independently of the FastAPI service** — there is no HTTP call between them — so the dashboard works even if the API isn't running.

---

## 🧪 Testing

```bash
pytest
```

**7 tests passing**, covering the health endpoint, valid and invalid prediction requests (missing fields, negative amounts), and feature-preparation logic (frequency encoding for known and unseen users). CI runs this suite automatically via GitHub Actions on every push and pull request to `main`.

---

## 🔮 Future Improvements

- **Real transaction velocity features** — the original timestamp column was corrupted (an `MM:SS.s` artifact, not a real datetime), which ruled out genuine time-window velocity features like "transactions in the last hour." A clean timestamp would unlock this.
- **Anomaly detection module** — an unsupervised layer (e.g. Isolation Forest) to flag suspicious transactions that don't carry the fraud label but deviate from normal behavior
- **Model monitoring** — track prediction and feature drift as new transaction data arrives
- **API ↔ Dashboard integration option** — allow the dashboard to optionally call the FastAPI service over HTTP for deployments where a single shared backend is preferred over duplicated artifact loading
- **Cloud deployment** — containerized deployment to a cloud provider with artifact storage outside the image

---

## ⚠️ Disclaimer

UPI Fraud Detector is a portfolio/educational project trained on a **synthetic** dataset. It is not connected to any real UPI infrastructure, does not reflect real-world fraud detection accuracy, and is **not intended for real financial or fraud-related decisions**. Several of this dataset's behavioral signals are unrealistically clean compared to real-world fraud data, which is documented above rather than hidden.

---

## 👤 Author

**Md Faiyaz Khan**
- GitHub: [@md-faiyazkhan](https://github.com/md-faiyazkhan)
- LinkedIn: [@mdfaiyazkhan](https://www.linkedin.com/in/mdfaiyazkhan)
- Email: faiyazkhan.work@gmail.com
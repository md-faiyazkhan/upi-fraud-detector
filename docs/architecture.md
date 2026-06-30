# Architecture — UPI Fraud Detector

## Overview

This project follows a layered architecture separating the model training
pipeline from the inference layer, similar to how a production ML system
would be structured.

## Pipeline Flow

```text
Raw Dataset (fraud_dataset.csv)
        │
        ▼
EDA & Leakage Detection (notebooks/01-02)
        │
        ▼
Preprocessing & Feature Engineering (notebooks/03, src/)
        │
        ▼
Model Training & Evaluation (notebooks/04, src/)
        │
        ▼
Artifacts (artifacts/) — model, scaler, freq maps, risk thresholds
        │
        ├──────────────┬──────────────┐
        ▼                              ▼
FastAPI (app/)                Streamlit Dashboard (dashboard/)
        │                              │
        ▼                              ▼
  /predict endpoint            Prediction / Analytics / Overview pages
```

## Component Responsibilities

**`src/`** — Reusable training pipeline logic: preprocessing, feature
engineering, model training, evaluation, explainability, and artifact
utilities. Used by the notebooks during model development.

**`app/`** — FastAPI inference layer. Loads trained artifacts once at
startup and exposes a `/predict` endpoint that returns fraud probability,
risk score, risk category, and a SHAP-based explanation for each
transaction.

**`dashboard/`** — Streamlit dashboard. Runs independently of the FastAPI
service, loading the same artifacts directly via `src/` and `app/` logic
to avoid duplicating preprocessing or prediction code.

**`artifacts/`** — Serialized outputs of the training pipeline:
`final_model.joblib`, `preprocessor.joblib`, `freq_maps.joblib`,
`feature_columns.joblib`, and `risk_thresholds.json`. These are the
single source of truth shared by both the API and the dashboard.

## Key Design Decisions

- **Leakage-first feature selection**: 18 final features were chosen only
  after multiple rounds of leakage detection (both during EDA and during
  model training), since the dataset is synthetically generated and
  contained several deterministic leakage columns.
- **Frequency encoding fit on train only**: user/merchant/device frequency
  maps are computed strictly from the training set and applied to test
  data and inference requests, to avoid data leakage across the split.
- **SHAP explainer regenerated at runtime**: the SHAP TreeExplainer is not
  pickled as an artifact, since SHAP explainers can be version-fragile.
  It is recreated from the saved model whenever needed.
- **Dashboard is standalone**: it does not call the FastAPI service over
  HTTP. Instead, it imports the same prediction logic directly, so it can
  run independently without requiring the API to be up.
"""
Utility functions for UPI Fraud Detector.
Handles saving and loading of model artifacts (model, scaler,
frequency maps, feature columns, risk thresholds).
"""

import json
import joblib


def save_artifact(obj, path: str):
    """Save a Python object to disk using joblib."""
    joblib.dump(obj, path)


def load_artifact(path: str):
    """Load a joblib-saved Python object from disk."""
    return joblib.load(path)


def save_risk_thresholds(thresholds: dict, path: str):
    """Save risk score thresholds as a JSON file."""
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=4)


def load_risk_thresholds(path: str) -> dict:
    """Load risk score thresholds from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_all_artifacts(
    model,
    scaler,
    freq_maps: dict,
    feature_columns: list,
    risk_thresholds: dict,
    artifacts_dir: str = "artifacts"
):
    """
    Save all training artifacts to the given directory in one call.
    """
    save_artifact(model, f"{artifacts_dir}/final_model.joblib")
    save_artifact(scaler, f"{artifacts_dir}/preprocessor.joblib")
    save_artifact(freq_maps, f"{artifacts_dir}/freq_maps.joblib")
    save_artifact(feature_columns, f"{artifacts_dir}/feature_columns.joblib")
    save_risk_thresholds(risk_thresholds, f"{artifacts_dir}/risk_thresholds.json")
    print("All artifacts saved successfully.")
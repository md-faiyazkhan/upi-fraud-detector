"""
Predictor module for UPI Fraud Detector.
Loads all trained artifacts once at startup and exposes a single
predict() function that the FastAPI endpoint calls.
"""

import shap

from app.schemas import TransactionRequest, PredictionResponse, TopReason
from app.features import build_feature_row, scale_feature_row
from app.risk_scoring import get_risk_assessment
from src.utils import load_artifact, load_risk_thresholds
from src.explainability import get_explainer, explain_single_prediction, get_top_reasons


ARTIFACTS_DIR = "artifacts"


class FraudPredictor:
    """
    Wraps the trained model, scaler, frequency maps, and SHAP explainer.
    Artifacts are loaded once when the predictor is instantiated
    (typically at FastAPI app startup), avoiding repeated disk reads.
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.model = load_artifact(f"{artifacts_dir}/final_model.joblib")
        self.scaler = load_artifact(f"{artifacts_dir}/preprocessor.joblib")
        self.freq_maps = load_artifact(f"{artifacts_dir}/freq_maps.joblib")
        self.feature_columns = load_artifact(f"{artifacts_dir}/feature_columns.joblib")
        self.risk_thresholds = load_risk_thresholds(f"{artifacts_dir}/risk_thresholds.json")
        self.explainer = get_explainer(self.model)

    def predict(self, request: TransactionRequest) -> PredictionResponse:
        """Run the full prediction pipeline for a single transaction."""
        raw_row = build_feature_row(request, self.freq_maps)
        scaled_row = scale_feature_row(raw_row, self.scaler)

        fraud_probability = float(self.model.predict_proba(scaled_row)[:, 1][0])
        is_fraud = int(fraud_probability >= 0.5)

        risk = get_risk_assessment(fraud_probability, self.risk_thresholds)

        explanation = explain_single_prediction(self.explainer, scaled_row)
        top_reasons_raw = get_top_reasons(explanation, top_n=3)

        top_reasons = [
            TopReason(feature=feat, shap_value=float(val), feature_value=float(fval))
            for feat, val, fval in top_reasons_raw
        ]

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            risk_score=risk['risk_score'],
            risk_category=risk['risk_category'],
            top_reasons=top_reasons
        )
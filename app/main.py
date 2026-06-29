"""
FastAPI entry point for UPI Fraud Detector.
Exposes health check and fraud prediction endpoints.
"""

from fastapi import FastAPI, HTTPException

from app.schemas import TransactionRequest, PredictionResponse
from app.predictor import FraudPredictor


app = FastAPI(
    title="UPI Fraud Detector API",
    description="Predicts fraud probability and risk score for UPI transactions.",
    version="1.0.0"
)

predictor: FraudPredictor | None = None


@app.on_event("startup")
def load_predictor():
    """Load model artifacts once when the API starts."""
    global predictor
    predictor = FraudPredictor()


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: TransactionRequest):
    """Predict fraud probability, risk score, and explanation for a transaction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        return predictor.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
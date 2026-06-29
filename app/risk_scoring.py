"""
Risk scoring module for UPI Fraud Detector.
Converts a fraud probability into a 0-100 risk score and risk category
based on configurable thresholds.
"""


def probability_to_risk_score(probability: float) -> int:
    """Convert a fraud probability (0-1) into a risk score (0-100)."""
    return round(probability * 100)


def risk_score_to_category(risk_score: int, thresholds: dict) -> str:
    """
    Map a risk score to a category (Low/Medium/High) using thresholds
    loaded from risk_thresholds.json.
    Expected thresholds format: {"low": 30, "medium": 70, "high": 100}
    """
    if risk_score <= thresholds['low']:
        return "Low Risk"
    elif risk_score <= thresholds['medium']:
        return "Medium Risk"
    else:
        return "High Risk"


def get_risk_assessment(probability: float, thresholds: dict) -> dict:
    """Compute risk score and category from a fraud probability."""
    risk_score = probability_to_risk_score(probability)
    risk_category = risk_score_to_category(risk_score, thresholds)
    return {
        'risk_score': risk_score,
        'risk_category': risk_category
    }
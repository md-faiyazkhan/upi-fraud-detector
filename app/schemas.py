"""
Pydantic schemas for UPI Fraud Detector API.
Defines request and response models for the prediction endpoint.
"""

from pydantic import BaseModel, Field
from typing import List


class TransactionRequest(BaseModel):
    """
    Raw transaction input from the client.
    user_id is used to look up frequency encoding internally;
    it is not passed directly as a model feature.
    """
    user_id: str = Field(..., description="Sender's user ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    session_duration: int = Field(..., ge=0)
    receiver_transaction_history: int = Field(..., ge=0)
    transaction_amount_vs_sender_history: float
    geographic_disparity: float
    transaction_time_of_day: int = Field(..., ge=0, le=23)
    time_between_link_click_and_transaction: int = Field(..., ge=0)
    input_timing_consistency: float
    keyboard_input_speed: float
    input_pause_patterns: float
    screen_active_time: int = Field(..., ge=0)
    geographic_location_vs_ip: float
    background_data_usage: float
    pin_entry_speed: float
    request_amount_roundness: float
    request_acceptance_rate: float = Field(..., ge=0, le=1)
    time_to_respond_to_request: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class TopReason(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    risk_score: int
    risk_category: str
    top_reasons: List[TopReason]
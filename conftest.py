"""
Shared pytest fixtures for UPI Fraud Detector tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI test client with lifespan events (startup) triggered."""
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client


@pytest.fixture
def sample_transaction():
    """A valid sample transaction request body for testing."""
    return {
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
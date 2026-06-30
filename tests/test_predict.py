"""
Tests for the /predict endpoint.
"""

def test_predict_returns_valid_response(client, sample_transaction):
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200

    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_category" in data
    assert "top_reasons" in data

    assert data["is_fraud"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(data["top_reasons"]) == 3


def test_predict_missing_field_returns_422(client, sample_transaction):
    bad_request = sample_transaction.copy()
    del bad_request["amount"]

    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_negative_amount_returns_422(client, sample_transaction):
    bad_request = sample_transaction.copy()
    bad_request["amount"] = -100

    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422
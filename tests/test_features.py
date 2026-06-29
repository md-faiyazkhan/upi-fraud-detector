"""
Tests for feature preparation logic (app/features.py).
"""

from app.schemas import TransactionRequest
from app.features import build_feature_row, FEATURE_ORDER


def test_build_feature_row_has_correct_columns(sample_transaction):
    request = TransactionRequest(**sample_transaction)
    freq_maps = {"user_id": {}}  # empty map, user_id_freq should default to 0

    row = build_feature_row(request, freq_maps)

    assert list(row.columns) == FEATURE_ORDER
    assert row.shape[0] == 1


def test_unseen_user_id_gets_zero_frequency(sample_transaction):
    request = TransactionRequest(**sample_transaction)
    freq_maps = {"user_id": {}}

    row = build_feature_row(request, freq_maps)

    assert row["user_id_freq"].iloc[0] == 0


def test_known_user_id_gets_correct_frequency(sample_transaction):
    request = TransactionRequest(**sample_transaction)
    freq_maps = {"user_id": {"U12345": 7}}

    row = build_feature_row(request, freq_maps)

    assert row["user_id_freq"].iloc[0] == 7
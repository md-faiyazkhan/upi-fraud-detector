"""
Inference-time feature preparation for UPI Fraud Detector.
Converts a raw TransactionRequest into a model-ready, scaled feature row.
"""

import pandas as pd

from app.schemas import TransactionRequest


FEATURE_ORDER = [
    'amount', 'session_duration', 'receiver_transaction_history',
    'transaction_amount_vs_sender_history', 'geographic_disparity',
    'transaction_time_of_day', 'time_between_link_click_and_transaction',
    'input_timing_consistency', 'keyboard_input_speed', 'input_pause_patterns',
    'screen_active_time', 'geographic_location_vs_ip', 'background_data_usage',
    'pin_entry_speed', 'request_amount_roundness', 'request_acceptance_rate',
    'time_to_respond_to_request', 'user_id_freq'
]


def build_feature_row(request: TransactionRequest, freq_maps: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from a TransactionRequest, in the exact
    column order the model expects. user_id is converted to its frequency
    encoding using the saved freq_maps; unseen users get frequency 0.
    """
    data = request.model_dump()
    user_id = data.pop('user_id')

    user_freq_map = freq_maps.get('user_id', {})
    data['user_id_freq'] = user_freq_map.get(user_id, 0)

    row = pd.DataFrame([data])
    row = row[FEATURE_ORDER]

    return row


def scale_feature_row(row: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply the fitted scaler to a single feature row."""
    scaled = scaler.transform(row)
    return pd.DataFrame(scaled, columns=row.columns, index=row.index)
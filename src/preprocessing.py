"""
Preprocessing module for UPI Fraud Detector.
Handles raw data loading, leakage/zero-variance column removal,
and train/test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# Columns identified as deterministic/near-deterministic leakage during EDA
LEAKAGE_COLS = [
    'authentication_attempts', 'merchant_category_code', 'session_source',
    'unusual_device_flag', 'unusual_ip_flag', 'unusual_location_flag',
    'dns_lookup_age', 'recent_app_installs', 'app_switching_frequency',
    'permissions_granted', 'recognized_screen_sharing_apps',
    'authentication_attempt_count', 'time_between_otp_generation_and_input',
    'pin_entry_method', 'unusual_transaction_amount_flag',
    'otp_request_frequency', 'otp_request_device_consistency',
    'transaction_velocity', 'failed_transaction_count',
    'authorization_method', 'transaction_type', 'request_description',
    'request_description_keywords', 'request_frequency',
    'time_pressure_indicators', 'requester_account_age',
    'handle_similarity_score', 'handle_typo_analysis',
    'handle_transaction_history', 'handle_registration_pattern',
    'handle_to_description_consistency', 'handle_verification_status',
    'business_name_match'
]

# Columns with zero variance (single unique value across all rows)
ZERO_VARIANCE_COLS = [
    'social_media_presence', 'upi_handle_age', 'handle_contains_official_terms',
    'relationship_to_requester'
]

# Columns with >97% missing values
HIGH_NULL_COLS = ['url_referrer', 'request_description']

# Other columns dropped for being unusable / corrupted / redundant text
OTHER_DROP_COLS = ['timestamp', 'description', 'location']

# ID columns used only for frequency encoding, dropped afterwards
ID_COLS = ['transaction_id', 'user_id', 'merchant_id', 'device_id', 'ip_address']

# Additional leakage discovered during model training phase
TRAINING_PHASE_LEAKAGE_COLS = [
    'merchant_id_freq', 'device_id_freq', 'receiver_account_age'
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw UPI transaction dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop leakage columns, zero-variance columns, high-null columns,
    and other unusable columns identified during EDA.
    """
    cols_to_drop = list(set(
        LEAKAGE_COLS + ZERO_VARIANCE_COLS + HIGH_NULL_COLS + OTHER_DROP_COLS
    ))
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(df: pd.DataFrame, target_col: str = 'is_fraud', test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataframe into train/test sets (stratified on target).
    ID columns are preserved at this stage for frequency encoding downstream.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(raw_path: str):
    """
    Full preprocessing pipeline: load raw data, drop unusable columns,
    and split into train/test sets.
    """
    df = load_raw_data(raw_path)
    df = drop_unusable_columns(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test
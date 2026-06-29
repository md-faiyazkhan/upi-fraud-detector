"""
Feature engineering module for UPI Fraud Detector.
Handles frequency encoding of ID columns and feature scaling.
Frequency maps and scaler must be fit on training data only to avoid leakage.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import ID_COLS, TRAINING_PHASE_LEAKAGE_COLS


FREQ_ENCODE_COLS = ['user_id', 'merchant_id', 'device_id', 'ip_address']


def fit_frequency_maps(X_train: pd.DataFrame, cols: list = FREQ_ENCODE_COLS) -> dict:
    """Compute frequency maps for given columns using training data only."""
    freq_maps = {}
    for col in cols:
        freq_maps[col] = X_train[col].value_counts()
    return freq_maps


def apply_frequency_encoding(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    """Apply precomputed frequency maps to a dataframe (train or test)."""
    df = df.copy()
    for col, freq_map in freq_maps.items():
        df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
    return df


def drop_id_and_leaked_freq_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw ID columns (after frequency encoding) and frequency
    features identified as leakage during model training
    (merchant_id_freq, device_id_freq), plus receiver_account_age.
    """
    cols_to_drop = [col for col in ID_COLS if col in df.columns]
    cols_to_drop += [col for col in TRAINING_PHASE_LEAKAGE_COLS if col in df.columns]
    cols_to_drop += ['ip_address_freq'] if 'ip_address_freq' in df.columns else []
    return df.drop(columns=cols_to_drop)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to a dataframe, preserving column names/index."""
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def run_feature_engineering(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Full feature engineering pipeline:
    1. Fit frequency maps on train, apply to train and test
    2. Drop raw IDs and leaked frequency/receiver_account_age columns
    3. Fit scaler on train, apply to train and test
    Returns scaled train/test sets plus fitted artifacts (freq_maps, scaler).
    """
    freq_maps = fit_frequency_maps(X_train)

    X_train_fe = apply_frequency_encoding(X_train, freq_maps)
    X_test_fe = apply_frequency_encoding(X_test, freq_maps)

    X_train_fe = drop_id_and_leaked_freq_columns(X_train_fe)
    X_test_fe = drop_id_and_leaked_freq_columns(X_test_fe)

    scaler = fit_scaler(X_train_fe)

    X_train_scaled = apply_scaler(X_train_fe, scaler)
    X_test_scaled = apply_scaler(X_test_fe, scaler)

    return X_train_scaled, X_test_scaled, freq_maps, scaler
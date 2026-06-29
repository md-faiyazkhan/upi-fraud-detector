"""
Explainability module for UPI Fraud Detector.
Generates SHAP-based global and local explanations for the trained model.
The explainer is created dynamically and is not persisted to disk,
since SHAP explainers can be version-fragile to pickle.
"""

import pandas as pd
import shap


def get_explainer(model):
    """Create a SHAP TreeExplainer for a tree-based model (e.g. Random Forest)."""
    return shap.TreeExplainer(model)


def get_shap_values(explainer, X: pd.DataFrame):
    """Compute SHAP values for a given set of samples."""
    return explainer.shap_values(X)


def get_global_importance(explainer, X: pd.DataFrame, fraud_class_index: int = 1) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature (global importance)
    for the fraud class.
    """
    shap_values = get_shap_values(explainer, X)
    mean_abs_shap = abs(shap_values[:, :, fraud_class_index]).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df


def explain_single_prediction(explainer, X_single: pd.DataFrame, fraud_class_index: int = 1) -> dict:
    """
    Generate a local explanation for a single transaction.
    Returns the base value, SHAP values, and feature values for that row.
    """
    shap_values = get_shap_values(explainer, X_single)

    return {
        'base_value': explainer.expected_value[fraud_class_index],
        'shap_values': shap_values[0, :, fraud_class_index],
        'feature_names': X_single.columns.tolist(),
        'feature_values': X_single.iloc[0].values
    }


def get_top_reasons(explanation: dict, top_n: int = 3) -> list:
    """
    Extract the top N features contributing to a fraud prediction,
    based on absolute SHAP value, for use in human-readable explanations.
    """
    pairs = list(zip(
        explanation['feature_names'],
        explanation['shap_values'],
        explanation['feature_values']
    ))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]
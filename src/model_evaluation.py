"""
Model evaluation module for UPI Fraud Detector.
Computes classification metrics, confusion matrix, and PR-AUC
for a trained model on the test set.
"""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model on the test set.
    Returns a dictionary containing classification report, ROC-AUC,
    PR-AUC, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_test, y_pred)

    return {
        'classification_report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def print_evaluation_summary(results: dict, model_name: str = "Model"):
    """Print a human-readable summary of evaluation results."""
    print(f"--- {model_name} Evaluation ---")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
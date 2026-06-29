"""
Model training module for UPI Fraud Detector.
Trains and compares Logistic Regression, Random Forest, and XGBoost,
selects the best model via cross-validation, and returns the fitted model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier


def get_candidate_models(y_train: pd.Series) -> dict:
    """Return a dictionary of candidate models with appropriate imbalance handling."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', random_state=42, max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric='logloss'
        )
    }
    return models


def compare_models_cv(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation (ROC-AUC) for all candidate models.
    Returns a dict of {model_name: (mean_score, std_score)}.
    """
    models = get_candidate_models(y_train)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = (scores.mean(), scores.std())

    return results


def select_best_model(cv_results: dict) -> str:
    """Select the model with the highest mean CV ROC-AUC score."""
    best_model_name = max(cv_results, key=lambda name: cv_results[name][0])
    return best_model_name


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = 'Random Forest'):
    """Train and return the final selected model on the full training set."""
    models = get_candidate_models(y_train)
    model = models[model_name]
    model.fit(X_train, y_train)
    return model
"""Model training and evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified split + StandardScaler fit on training data only.

    Fitting the scaler on test data would constitute data leakage: the test
    set would influence its own normalization, producing overly optimistic metrics.

    Returns
    -------
    X_train_s, X_test_s : scaled arrays
    y_train, y_test     : label arrays
    scaler              : fitted StandardScaler (save for production inference)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)  # apply training statistics, no refit
    return X_train_s, X_test_s, y_train, y_test, scaler


def tune_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict | None = None,
    n_splits: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42,
) -> tuple[LogisticRegression, float]:
    """
    GridSearchCV over LogisticRegression hyperparameters using StratifiedKFold.

    Cross-validation runs only on X_train — the test set is never touched.

    Parameters
    ----------
    param_grid : dict of parameter grids; defaults to C search over 6 values
    scoring    : sklearn metric string to optimize (default: roc_auc)

    Returns
    -------
    best_model : fitted LogisticRegression with best params
    best_score : best CV score achieved
    """
    if param_grid is None:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_


def evaluate(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return accuracy, AUC-ROC, average precision, F1, and classification report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'f1_default': f1_score(y_test, y_pred),
        'recall_default': f1_score(y_test, y_pred, average=None)[1],
        'classification_report': classification_report(
            y_test, y_pred, target_names=['Sem Default', 'Default']
        ),
    }


def compare_models(
    models: dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Fit all models on X_train and evaluate on X_test.

    Parameters
    ----------
    models : dict mapping model name → unfitted sklearn estimator

    Returns
    -------
    DataFrame with one row per model and columns for each metric.
    """
    rows = []
    for nome, modelo in models.items():
        modelo.fit(X_train, y_train)
        m = evaluate(modelo, X_test, y_test)
        rows.append({
            'Modelo': nome,
            'Acurácia': round(m['accuracy'], 4),
            'AUC-ROC': round(m['auc_roc'], 4),
            'Avg Precision': round(m['avg_precision'], 4),
            'F1 (default)': round(m['f1_default'], 4),
            'Recall (default)': round(m['recall_default'], 4),
        })
    return pd.DataFrame(rows).set_index('Modelo')


def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Returns mean and std for AUC-ROC, Average Precision, and F1 across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ['roc_auc', 'average_precision', 'f1']
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    summary = {}
    for metric in scoring:
        key = f'test_{metric}'
        summary[metric] = {
            'mean': float(results[key].mean()),
            'std': float(results[key].std()),
        }
    return summary

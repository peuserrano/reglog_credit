"""Model training and evaluation utilities."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit logistic regression with L2 regularization."""
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return accuracy, AUC-ROC, average precision, and classification report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['Sem Default', 'Default']
        ),
    }
    return metrics


def cross_validate_model(
    model: LogisticRegression,
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

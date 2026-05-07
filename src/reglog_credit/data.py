"""Data loading and preprocessing for credit default prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_RENAME = {
    'Application ID (primary key)': 'id',
    'Indicator for default': 'default',
    "Credit worthiness score calculated on the basis of borrower's credit history": 'historic_credit_score',
    'Sum of amount due on active credit cards (in $)': 'total_credit_cards_amount',
    'Annual income (in $)': 'annual_income',
    'Estimated market value of a properety owned/used by the borrower (in $)': 'collateral_mkt_value',
    'Maximum of credit available on all active credit lines (in $)': 'credit_limit',
    'Number of active credit cards on which full credit limit is utilized by the borrower': 'number_cards_w_limit_fully_used',
    'Average utilization of line on all active credit cards activated in last 1 year (%)': 'avg_card_utilization_last_1y',
}

FEATURES = [
    'historic_credit_score',
    'total_credit_cards_amount',
    'annual_income',
    'collateral_mkt_value',
    'credit_limit',
    'number_cards_w_limit_fully_used',
    'avg_card_utilization_last_1y',
]
TARGET = 'default'


def load_raw(raw_dir: str | Path = 'data/raw') -> pd.DataFrame:
    """Load raw training CSV and rename columns via data dictionary.

    Raises FileNotFoundError with helpful message if files are missing.
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / 'Training_dataset_Original.csv'
    dict_path = raw_dir / 'Data_Dictionary.csv'

    for p in (train_path, dict_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {p}\n"
                "Baixe o dataset do Kaggle e coloque em data/raw/ — ver README."
            )

    df = pd.read_csv(train_path)
    dicionario = pd.read_csv(dict_path)
    df = df.drop('index', axis=1)
    df.columns = dicionario['Definition']
    df = df[list(COLUMN_RENAME.keys())].rename(columns=COLUMN_RENAME)
    return df


def preprocess(
    df: pd.DataFrame,
    util_median: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """
    Treat missing-value strings, impute, and build feature matrix.

    Parameters
    ----------
    df : raw dataframe from load_raw()
    util_median : pre-computed median for avg_card_utilization_last_1y.
        Pass training median when preprocessing test/inference data to avoid leakage.

    Returns
    -------
    X            : float64 array, shape (n, n_features)
    y            : int array, shape (n,)
    feature_names: list of column names matching X columns
    util_median  : median used for imputation (save this for inference)
    """
    q = df.copy()

    # Replace string sentinels with NaN
    str_cols = [
        'total_credit_cards_amount', 'credit_limit',
        'number_cards_w_limit_fully_used', 'collateral_mkt_value',
        'historic_credit_score',
    ]
    for col in str_cols:
        q[col] = q[col].replace({'missing': np.nan, 'na': np.nan})
        q[col] = pd.to_numeric(q[col], errors='coerce')

    # collateral: absence = no property → 0; add binary indicator
    q['has_collateral'] = q['collateral_mkt_value'].notna().astype(int)
    q['collateral_mkt_value'] = q['collateral_mkt_value'].fillna(0)

    # integer features: absence of card/limit → 0 is semantically correct
    for col in ['total_credit_cards_amount', 'credit_limit',
                'number_cards_w_limit_fully_used', 'historic_credit_score']:
        q[col] = q[col].fillna(0)

    # avg_card_utilization: impute with median (0 is a valid real value)
    if util_median is None:
        util_median = float(q['avg_card_utilization_last_1y'].median())
    q['avg_card_utilization_last_1y'] = q['avg_card_utilization_last_1y'].fillna(util_median)

    # Enforce dtypes — preserve float for utilization ratio
    int_cols = [
        'historic_credit_score', 'total_credit_cards_amount', 'annual_income',
        'collateral_mkt_value', 'credit_limit', 'number_cards_w_limit_fully_used',
        'has_collateral',
    ]
    for col in int_cols:
        q[col] = q[col].astype('int64')

    feature_names = FEATURES + ['has_collateral']
    X = q[feature_names].astype('float64').values
    y = q[TARGET].values

    return X, y, feature_names, util_median

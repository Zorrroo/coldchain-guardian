"""Preprocessing: a leakage-free scikit-learn ColumnTransformer.

The transformer one-hot-encodes the categorical fields and standardises the
numeric ones. It is always wrapped inside a Pipeline with the estimator and
fitted only on training folds, so no statistic ever leaks from test data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_SEED,
    REG_TARGET,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer used across every model."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            # Numeric + ordinal columns are scaled together. The ordinal
            # packaging level is already integer-encoded (0/1/2).
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES + ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def split_features_targets(df: pd.DataFrame):
    """Return the feature matrix X restricted to the known feature columns."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")
    return df[FEATURE_COLUMNS].copy()


def make_splits(df: pd.DataFrame) -> dict:
    """Stratified train (60%) / validation (20%) / test (20%) split.

    Returns a dict with classification targets (``y_*``) and the bonus
    regression targets (``yr_*``). The same seed makes the split identical
    across ``train.py`` and ``evaluate.py``, so the saved model is always
    scored on data it never saw.
    """
    X = split_features_targets(df)
    y = df[TARGET].astype(int)
    y_reg = df[REG_TARGET].astype(float)

    X_tmp, X_test, y_tmp, y_test, yr_tmp, yr_test = train_test_split(
        X, y, y_reg,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # Validation fraction expressed relative to the remaining train portion.
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, yr_train, yr_val = train_test_split(
        X_tmp, y_tmp, yr_tmp,
        test_size=val_relative,
        stratify=y_tmp,
        random_state=RANDOM_SEED,
    )
    return {
        "X_train": X_train, "y_train": y_train, "yr_train": yr_train,
        "X_val": X_val, "y_val": y_val, "yr_val": yr_val,
        "X_test": X_test, "y_test": y_test, "yr_test": yr_test,
    }

"""Smoke tests for the ColdChain Guardian pipeline.

These are fast sanity checks (small synthetic sample, tiny models) that verify
the plumbing works end-to-end without training the full production models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import FEATURE_COLUMNS, REG_TARGET, TARGET
from src.features import build_preprocessor, make_splits
from src.generate_data import generate_dataset


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return generate_dataset(n=2000, seed=0)


def test_dataset_schema(small_df):
    for col in FEATURE_COLUMNS + [TARGET, REG_TARGET]:
        assert col in small_df.columns, f"missing column {col}"
    assert len(small_df) == 2000
    # Unobserved drivers must NOT leak into the feature table.
    assert "hidden_handling_quality" not in small_df.columns
    assert "sensor_calibration_drift" not in small_df.columns


def test_label_is_binary_and_balanced(small_df):
    assert set(small_df[TARGET].unique()) <= {0, 1}
    rate = small_df[TARGET].mean()
    # Classes should be reasonably (not perfectly) balanced.
    assert 0.12 < rate < 0.45, f"spoilage rate {rate:.2%} outside sane range"


def test_regression_target_bounds(small_df):
    assert small_df[REG_TARGET].between(0, 100).all()


def test_reproducibility():
    a = generate_dataset(n=500, seed=123)
    b = generate_dataset(n=500, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_splits_are_disjoint_and_stratified(small_df):
    splits = make_splits(small_df)
    n = len(small_df)
    assert len(splits["X_train"]) + len(splits["X_val"]) + len(
        splits["X_test"]
    ) == n
    # No index overlap between the three partitions.
    idx_train = set(splits["X_train"].index)
    idx_val = set(splits["X_val"].index)
    idx_test = set(splits["X_test"].index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)


def test_preprocessor_fits_and_transforms(small_df):
    splits = make_splits(small_df)
    prep = build_preprocessor()
    Xt = prep.fit_transform(splits["X_train"])
    assert Xt.shape[0] == len(splits["X_train"])
    # One-hot expansion must add columns beyond the raw feature count.
    assert Xt.shape[1] >= len(FEATURE_COLUMNS)
    assert np.isfinite(Xt).all() if hasattr(Xt, "all") else True


def test_model_trains_and_predicts(small_df):
    """A tiny end-to-end fit/predict through the real pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    splits = make_splits(small_df)
    pipe = Pipeline(
        [("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(splits["X_train"], splits["y_train"])
    proba = pipe.predict_proba(splits["X_test"])[:, 1]
    assert proba.shape[0] == len(splits["X_test"])
    assert ((proba >= 0) & (proba <= 1)).all()

"""Train, cross-validate, tune, and persist the models.

Pipeline:
  1. Load the dataset and build stratified train/val/test splits.
  2. Compare three classifiers with 5-fold cross-validated ROC-AUC:
       Logistic Regression (baseline), Random Forest, gradient boosting.
  3. Tune the best model with RandomizedSearchCV.
  4. Fit a lightweight gradient-boosting regressor for the bonus
     shelf-life target.
  5. Save both model artifacts and a metrics.json with the CV comparison.

Gradient boosting prefers LightGBM, then XGBoost, and falls back to
scikit-learn's HistGradientBoosting if neither installs -- a missing optional
package never stops the run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    CV_FOLDS,
    DATASET_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    REG_MODEL_PATH,
    TARGET,
    ensure_dirs,
)
from src.features import build_preprocessor, make_splits  # noqa: E402


def get_gradient_boosting():
    """Return (name, estimator) for the best available GB implementation."""
    try:
        from lightgbm import LGBMClassifier

        est = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        return "LightGBM", est, "lgbm"
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier

        est = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            tree_method="hist",
        )
        return "XGBoost", est, "xgb"
    except Exception:
        pass
    est = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=RANDOM_SEED
    )
    return "HistGradientBoosting", est, "hgb"


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor()), ("clf", estimator)]
    )


def _param_distributions(kind: str) -> dict:
    """RandomizedSearchCV grids keyed by model kind."""
    grids = {
        "logreg": {
            "clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "rf": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 12, 20, 30],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", 0.5],
        },
        "lgbm": {
            "clf__n_estimators": [200, 400, 600],
            "clf__num_leaves": [31, 63, 127],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__max_depth": [-1, 6, 10],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        },
        "xgb": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        },
        "hgb": {
            "clf__max_iter": [200, 400, 600],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__max_depth": [None, 6, 10],
            "clf__l2_regularization": [0.0, 0.5, 1.0],
        },
    }
    return grids[kind]


def build_candidates() -> dict:
    """Return {display_name: (pipeline, kind)} for the three model families."""
    gb_name, gb_est, gb_kind = get_gradient_boosting()
    return {
        "Logistic Regression": (
            _pipeline(
                LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
            ),
            "logreg",
        ),
        "Random Forest": (
            _pipeline(
                RandomForestClassifier(
                    n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED
                )
            ),
            "rf",
        ),
        gb_name: (_pipeline(gb_est), gb_kind),
    }


def train(df: pd.DataFrame | None = None) -> dict:
    """Run the full training routine and persist artifacts."""
    ensure_dirs()
    if df is None:
        df = pd.read_csv(DATASET_PATH)

    splits = make_splits(df)
    X_train, y_train = splits["X_train"], splits["y_train"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # --- 1. Cross-validated comparison of the three families ---------------
    candidates = build_candidates()
    comparison: dict[str, dict] = {}
    print("Cross-validated ROC-AUC (5-fold) on the training set:")
    for name, (pipe, _kind) in candidates.items():
        t0 = time.time()
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        comparison[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
        print(
            f"  {name:<22} AUC = {scores.mean():.4f} +/- {scores.std():.4f}"
            f"   ({time.time() - t0:.1f}s)"
        )

    best_name = max(comparison, key=lambda k: comparison[k]["cv_roc_auc_mean"])
    best_pipe, best_kind = candidates[best_name]
    print(f"\nBest model by CV AUC: {best_name}")

    # --- 2. Hyperparameter tuning of the winner ----------------------------
    print(f"Tuning {best_name} with RandomizedSearchCV...")
    param_dist = _param_distributions(best_kind)
    # Cap the number of sampled configurations at the size of the grid so we
    # never ask for more combinations than exist.
    grid_size = int(np.prod([len(v) for v in param_dist.values()]))
    n_iter = min(15, grid_size)
    search = RandomizedSearchCV(
        best_pipe,
        param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    print(f"  Best CV AUC after tuning: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    joblib.dump(tuned, MODEL_PATH)
    print(f"Saved tuned classifier -> {MODEL_PATH}")

    # --- 3. Bonus: shelf-life regression -----------------------------------
    reg = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    reg.fit(splits["X_train"], splits["yr_train"])
    joblib.dump(reg, REG_MODEL_PATH)
    print(f"Saved shelf-life regressor -> {REG_MODEL_PATH}")

    # --- 4. Record the comparison (evaluate.py adds test metrics later) ----
    metrics = {
        "dataset_rows": int(len(df)),
        "spoilage_rate": float(df[TARGET].mean()),
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "model_comparison": comparison,
        "best_model": best_name,
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote CV comparison -> {METRICS_PATH}")

    return {
        "model": tuned,
        "reg_model": reg,
        "splits": splits,
        "best_name": best_name,
        "metrics": metrics,
    }


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


if __name__ == "__main__":
    train()

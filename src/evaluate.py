"""Evaluate the saved models and produce the result figures.

Loads the tuned classifier and the shelf-life regressor, rebuilds the exact
same stratified test split (seed-locked), and writes:
  * a metrics table (accuracy / precision / recall / F1 / ROC-AUC),
  * confusion matrix, ROC curve and precision-recall curve,
  * permutation importance (model-agnostic; SHAP too if it imports cleanly),
  * an actual-vs-predicted plot for the bonus regression target.
All headline numbers are merged back into ``models/metrics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402

from src.config import (  # noqa: E402
    DATASET_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    REG_MODEL_PATH,
    RANDOM_SEED,
    ensure_dirs,
)
from src.features import make_splits  # noqa: E402


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion(y_true, y_pred) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["safe (0)", "spoiled (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, "confusion_matrix.png")


def plot_roc(y_true, y_proba, auc) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2a9d8f", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    return _save(fig, "roc_curve.png")


def plot_pr(y_true, y_proba) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#e76f51", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    return _save(fig, "precision_recall_curve.png")


def plot_permutation_importance(model, X_test, y_test) -> Path:
    result = permutation_importance(
        model, X_test, y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    order = result.importances_mean.argsort()
    names = np.array(FEATURE_COLUMNS)[order]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        names,
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color="#264653",
    )
    ax.set_title("Permutation importance (drop in ROC-AUC when shuffled)")
    ax.set_xlabel("Mean importance")
    fig.tight_layout()
    return _save(fig, "permutation_importance.png")


def maybe_plot_shap(model, X_test) -> Path | None:
    """Best-effort SHAP summary that adapts to the chosen model family.

    Uses TreeExplainer for tree / gradient-boosting models and LinearExplainer
    for linear models; silently skipped (permutation importance still covers
    explainability) if SHAP is unavailable or the model is unsupported.
    """
    try:
        import shap
    except Exception:
        print("SHAP not installed - skipping (permutation importance covers it).")
        return None
    try:
        clf = model.named_steps["clf"]
        prep = model.named_steps["prep"]
        sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_SEED)
        X_trans = prep.transform(sample)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feat_names = list(prep.get_feature_names_out())

        tree_like = ("RandomForest", "LGBM", "XGB", "HistGradientBoosting",
                     "GradientBoosting", "DecisionTree", "ExtraTrees")
        if any(t in type(clf).__name__ for t in tree_like):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list):  # [class0, class1]
                shap_values = shap_values[1]
            # Some tree explainers return a 3-D array (samples, feats, classes).
            if getattr(shap_values, "ndim", 2) == 3:
                shap_values = shap_values[..., 1]
        else:
            explainer = shap.LinearExplainer(clf, X_trans)
            shap_values = explainer.shap_values(X_trans)

        fig = plt.figure()
        shap.summary_plot(
            shap_values, X_trans, feature_names=feat_names, show=False,
            max_display=15,
        )
        path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"Saved SHAP summary -> {path}")
        return path
    except Exception as exc:  # pragma: no cover - SHAP is best-effort
        print(f"SHAP step skipped ({type(exc).__name__}: {exc}).")
        return None


def plot_regression(y_true, y_pred) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.25, color="#2a9d8f")
    lims = [0, 100]
    ax.plot(lims, lims, "--", color="gray", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual shelf-life remaining")
    ax.set_ylabel("Predicted shelf-life remaining")
    ax.set_title("Shelf-life regression: actual vs. predicted (test set)")
    return _save(fig, "regression_actual_vs_predicted.png")


def evaluate() -> dict:
    """Score saved models on the held-out split and write all figures."""
    ensure_dirs()
    df = pd.read_csv(DATASET_PATH)
    splits = make_splits(df)

    model = joblib.load(MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)

    X_test, y_test = splits["X_test"], splits["y_test"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    # --- Classification on validation and test -----------------------------
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_metrics = _classification_metrics(y_val, val_pred, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_metrics = _classification_metrics(y_test, test_pred, test_proba)

    print("Validation metrics:", _fmt(val_metrics))
    print("Test metrics:      ", _fmt(test_metrics))

    plot_confusion(y_test, test_pred)
    plot_roc(y_test, test_proba, test_metrics["roc_auc"])
    plot_pr(y_test, test_proba)
    plot_permutation_importance(model, X_test, y_test)
    maybe_plot_shap(model, X_test)

    # --- Regression on test -----------------------------------------------
    yr_test = splits["yr_test"]
    yr_pred = reg_model.predict(X_test)
    reg_metrics = {
        "mae": float(mean_absolute_error(yr_test, yr_pred)),
        "rmse": float(np.sqrt(mean_squared_error(yr_test, yr_pred))),
        "r2": float(r2_score(yr_test, yr_pred)),
    }
    print("Shelf-life regression:", _fmt(reg_metrics))
    plot_regression(yr_test, yr_pred)

    # --- Merge into metrics.json ------------------------------------------
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["validation_metrics"] = val_metrics
    metrics["test_metrics"] = test_metrics
    metrics["regression_metrics"] = reg_metrics
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Updated metrics -> {METRICS_PATH}")
    print(f"All figures saved under {FIGURES_DIR}")

    return metrics


def _fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


if __name__ == "__main__":
    evaluate()

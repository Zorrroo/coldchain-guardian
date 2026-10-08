"""Exploratory data analysis: saves a handful of publication-ready figures.

All plots are written to ``figures/`` with a non-interactive backend so this
runs cleanly inside a headless pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    DATASET_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET,
    ensure_dirs,
)

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    _HAS_SNS = True
except Exception:  # pragma: no cover - seaborn is optional for styling
    _HAS_SNS = False

PALETTE = {"safe": "#2a9d8f", "spoiled": "#e76f51"}


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["safe (0)", "spoiled (1)"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, counts.values, color=[PALETTE["safe"], PALETTE["spoiled"]])
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / total:.1%})", ha="center", va="bottom")
    ax.set_title("Class balance: spoiled vs. safe shipments")
    ax.set_ylabel("Number of shipments")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "class_balance.png")


def plot_feature_distributions(df: pd.DataFrame) -> Path:
    cols = [
        "transit_time_hours",
        "route_distance_km",
        "ambient_temp_max_c",
        "num_temp_excursions",
        "num_coldchain_handoffs",
        "carrier_reliability_score",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), cols):
        safe = df.loc[df[TARGET] == 0, col]
        spoiled = df.loc[df[TARGET] == 1, col]
        bins = np.histogram_bin_edges(df[col], bins=40)
        ax.hist(safe, bins=bins, alpha=0.6, label="safe", color=PALETTE["safe"])
        ax.hist(
            spoiled, bins=bins, alpha=0.6, label="spoiled", color=PALETTE["spoiled"]
        )
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Feature distributions by outcome", fontsize=14)
    fig.tight_layout()
    return _save(fig, "feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    if _HAS_SNS:
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.8},
        )
    else:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (numeric features vs. target)")
    return _save(fig, "correlation_heatmap.png")


def plot_spoilage_by_category(df: pd.DataFrame) -> Path:
    rate = df.groupby("product_category")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rate.index, rate.values, color="#264653")
    for i, v in enumerate(rate.values):
        ax.text(v, i, f" {v:.1%}", va="center")
    ax.set_title("Spoilage rate by product category")
    ax.set_xlabel("Share of shipments spoiled")
    ax.set_xlim(0, rate.max() * 1.2)
    return _save(fig, "spoilage_by_category.png")


def plot_domain_insight(df: pd.DataFrame) -> Path:
    """Spoilage rate by transport mode x packaging quality."""
    pivot = (
        df.pivot_table(
            index="transport_mode",
            columns="packaging_insulation_quality",
            values=TARGET,
            aggfunc="mean",
        )
        .reindex(columns=[0, 1, 2])
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.25
    names = {0: "basic", 1: "standard", 2: "premium"}
    colors = ["#e9c46a", "#f4a261", "#e76f51"]
    for j, level in enumerate([0, 1, 2]):
        ax.bar(
            x + (j - 1) * width,
            pivot[level].values,
            width,
            label=names[level],
            color=colors[j],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Share spoiled")
    ax.set_title("Spoilage rate by transport mode and packaging quality")
    ax.legend(title="packaging")
    return _save(fig, "spoilage_by_mode_packaging.png")


def run_eda(df: pd.DataFrame | None = None) -> list[Path]:
    """Generate and save all EDA figures; returns the list of paths."""
    ensure_dirs()
    if df is None:
        df = pd.read_csv(DATASET_PATH)
    paths = [
        plot_class_balance(df),
        plot_feature_distributions(df),
        plot_correlation_heatmap(df),
        plot_spoilage_by_category(df),
        plot_domain_insight(df),
    ]
    print(f"Saved {len(paths)} EDA figures to {FIGURES_DIR}")
    return paths


if __name__ == "__main__":
    run_eda()

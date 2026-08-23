from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np


def create_metrics_chart(metric_table: pd.DataFrame):
    row = metric_table.iloc[0]
    model_name = row.get("Model", "Model")
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [float(row[label]) * 100 for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#2563eb", "#0891b2", "#16a34a", "#9333ea"])
    ax.set_title(f"{model_name} Model Performance")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def create_confusion_matrix_chart(confusion_matrix: list[list[int]]):
    matrix = np.array(confusion_matrix)
    labels = ["No Disease", "Disease"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticks(range(2), labels)
    ax.set_yticks(range(2), labels)

    threshold = matrix.max() / 2
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def create_feature_coefficient_chart(coefficients: pd.DataFrame, top_n: int = 8):
    top_features = coefficients.head(top_n).copy()
    top_features = top_features.iloc[::-1]
    colors = [
        "#dc2626" if value > 0 else "#2563eb"
        for value in top_features["Coefficient"]
    ]
    labels = [
        "\n".join(textwrap.wrap(label, width=38))
        for label in top_features["Feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.barh(labels, top_features["Coefficient"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(
        "Top Logistic Regression Influential Features\n"
        "Red increases disease probability; blue decreases it",
        pad=14,
    )
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    min_value = float(top_features["Coefficient"].min())
    max_value = float(top_features["Coefficient"].max())
    padding = max(abs(min_value), abs(max_value)) * 0.25
    ax.set_xlim(min_value - padding, max_value + padding)

    for bar, value in zip(bars, top_features["Coefficient"]):
        x_offset = padding * 0.18 if value >= 0 else -padding * 0.18
        alignment = "left" if value >= 0 else "right"
        ax.text(
            value + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=alignment,
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def save_visualizations(
    metric_table: pd.DataFrame,
    confusion_matrix: list[list[int]],
    coefficients: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charts = {
        "metrics_chart.png": create_metrics_chart(metric_table),
        "confusion_matrix.png": create_confusion_matrix_chart(confusion_matrix),
        "feature_coefficients.png": create_feature_coefficient_chart(coefficients),
    }

    for filename, fig in charts.items():
        fig.savefig(output_path / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


def create_feature_importance_chart(importances: pd.DataFrame, top_n: int = 8):
    top_features = importances.head(top_n).copy()
    top_features = top_features.iloc[::-1]
    colors = ["#0891b2"] * len(top_features)
    labels = [
        "\n".join(textwrap.wrap(label, width=38))
        for label in top_features["Feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.barh(labels, top_features["Importance"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(
        "Top Random Forest Feature Importances\n"
        "Higher score indicates stronger predictive influence",
        pad=14,
    )
    ax.set_xlabel("Feature Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_value = float(top_features["Importance"].max())
    padding = max_value * 0.15 if max_value > 0 else 0.15
    ax.set_xlim(0, max_value + padding)

    for bar, value in zip(bars, top_features["Importance"]):
        ax.text(
            value + (padding * 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def save_rf_visualizations(
    metric_table: pd.DataFrame,
    confusion_matrix: list[list[int]],
    importances: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_metrics = create_metrics_chart(metric_table)
    fig_cm = create_confusion_matrix_chart(confusion_matrix)
    fig_imp = create_feature_importance_chart(importances)

    fig_metrics.savefig(output_path / "metrics_chart.png", dpi=160, bbox_inches="tight")
    fig_cm.savefig(output_path / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    fig_imp.savefig(output_path / "feature_importances.png", dpi=160, bbox_inches="tight")

    plt.close(fig_metrics)
    plt.close(fig_cm)
    plt.close(fig_imp)

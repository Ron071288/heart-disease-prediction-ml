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


def create_model_comparison_chart(metric_table: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    models = metric_table["Model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for index, model_name in enumerate(models):
        row = metric_table[metric_table["Model"] == model_name].iloc[0]
        values = [float(row[metric]) * 100 for metric in metrics]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=colors[index % len(colors)],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

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

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
        "model_comparison_chart.png": create_model_comparison_chart(metric_table),
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


def create_knn_tuning_chart(tuning_results: pd.DataFrame):
    """Return a matplotlib Figure showing CV accuracy vs k for each param combination.

    Parameters
    ----------
    tuning_results:
        DataFrame produced by ``tune_knn_hyperparameters()``, containing at least
        the columns ``n_neighbors``, ``weights``, ``metric``,
        ``CV Accuracy Mean``, and ``CV Accuracy Std``.
    """
    combos = (
        tuning_results[["weights", "metric"]]
        .drop_duplicates()
        .sort_values(["weights", "metric"])
        .reset_index(drop=True)
    )

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for idx, row in combos.iterrows():
        mask = (
            (tuning_results["weights"] == row["weights"])
            & (tuning_results["metric"] == row["metric"])
        )
        subset = tuning_results[mask].sort_values("n_neighbors")
        k_vals = subset["n_neighbors"].tolist()
        acc_mean = (subset["CV Accuracy Mean"] * 100).tolist()
        acc_std = (subset["CV Accuracy Std"] * 100).tolist()

        color = palette[idx % len(palette)]
        ls = line_styles[idx % len(line_styles)]
        label = f"{row['weights']}, {row['metric']}"

        ax.plot(k_vals, acc_mean, marker="o", color=color, linestyle=ls,
                linewidth=1.8, markersize=5, label=label)
        ax.fill_between(
            k_vals,
            [m - s for m, s in zip(acc_mean, acc_std)],
            [m + s for m, s in zip(acc_mean, acc_std)],
            color=color, alpha=0.08,
        )

    # Highlight the best point
    best = tuning_results.iloc[0]
    ax.axvline(
        best["n_neighbors"], color="#6b7280", linewidth=1,
        linestyle="--", alpha=0.6,
    )
    ax.annotate(
        f"Best: k={int(best['n_neighbors'])}\n{best['weights']}, {best['metric']}\n"
        f"{best['CV Accuracy Mean'] * 100:.1f}%",
        xy=(best["n_neighbors"], best["CV Accuracy Mean"] * 100),
        xytext=(best["n_neighbors"] + 1.2, best["CV Accuracy Mean"] * 100 - 3),
        fontsize=8.5,
        color="#111827",
        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1),
    )

    ax.set_title(
        "KNN Hyperparameter Tuning\n"
        "5-fold Stratified CV Accuracy vs Number of Neighbours (k)",
        pad=12,
    )
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("CV Accuracy (%)")
    ax.set_xticks(tuning_results["n_neighbors"].unique())
    ax.set_ylim(
        max(0, tuning_results["CV Accuracy Mean"].min() * 100 - 6),
        min(100, tuning_results["CV Accuracy Mean"].max() * 100 + 6),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)

    fig.tight_layout()
    return fig


def save_knn_visualizations(
    metric_table: pd.DataFrame,
    confusion_matrix: list[list[int]],
    output_dir: str | Path,
) -> None:
    """Save KNN charts (metrics and confusion matrix only — KNN has no feature importances)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_metrics = create_metrics_chart(metric_table)
    fig_cm = create_confusion_matrix_chart(confusion_matrix)

    fig_metrics.savefig(output_path / "metrics_chart.png", dpi=160, bbox_inches="tight")
    fig_cm.savefig(output_path / "confusion_matrix.png", dpi=160, bbox_inches="tight")

    plt.close(fig_metrics)
    plt.close(fig_cm)

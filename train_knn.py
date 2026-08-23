from __future__ import annotations

import argparse
from pathlib import Path

from heart_model import (
    KNN_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    metrics_to_table,
    save_artifacts,
    train_models,
)
from heart_visuals import save_knn_visualizations


def format_metric_table(table):
    display_table = table.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")
    return display_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate K-Nearest Neighbors heart disease prediction model."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional local CSV path. If omitted, the dataset is downloaded with KaggleHub.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Folder used to save trained models and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.csv)
    analysis = dataset_analysis(df)

    print("=" * 76)
    print("HEART DISEASE PREDICTION - K-NEAREST NEIGHBORS")
    print("=" * 76)
    print()

    print("1. DATASET & WORKFLOW")
    print("-" * 76)
    print(f"Raw dataset              : {analysis['raw_rows']} samples, {analysis['columns']} columns")
    print(f"Duplicate rows removed   : {analysis['duplicates']}")
    print(f"Training samples used    : {analysis['deduplicated_rows']}")
    print(
        "Training class balance   : "
        f"{analysis['deduplicated_no_heart_disease']} no disease / "
        f"{analysis['deduplicated_heart_disease']} disease"
    )
    print("Split & preprocessing    : 80/20 split, scaling, one-hot encoding")
    print("KNN setup                : n_neighbors=11, metric=minkowski")
    print()

    models, metrics, _, _ = train_models(df, selected_models=[KNN_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("2. KNN RESULT")
    print("-" * 76)
    print(format_metric_table(comparison).to_string(index=False))

    knn_metrics = metrics[KNN_MODEL_NAME]
    print()
    print(
        "Scores                   : "
        f"Accuracy {knn_metrics['accuracy'] * 100:.2f}%, "
        f"Precision {knn_metrics['precision'] * 100:.2f}%, "
        f"Recall {knn_metrics['recall'] * 100:.2f}%, "
        f"F1 {knn_metrics['f1_score'] * 100:.2f}%"
    )
    tn, fp = knn_metrics["confusion_matrix"][0]
    fn, tp = knn_metrics["confusion_matrix"][1]
    print(f"Confusion matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_knn_visualizations(
        comparison,
        knn_metrics["confusion_matrix"],
        output_dir,
    )
    print()
    print("3. OUTPUT")
    print("-" * 76)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

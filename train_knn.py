from __future__ import annotations

import argparse
from pathlib import Path

from heart_model import (
    BEST_MODEL_NAME,
    KNN_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    metrics_to_table,
    save_artifacts,
    train_models,
)
from heart_visuals import save_knn_visualizations


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

    print("=" * 72)
    print("HEART DISEASE PREDICTION - K-NEAREST NEIGHBORS")
    print("=" * 72)
    print()

    print("1. DATASET ANALYSIS")
    print("-" * 72)
    print(f"Raw dataset: {analysis['raw_rows']} samples, {analysis['columns']} columns")
    print(
        "Class balance: "
        f"{analysis['raw_no_heart_disease']} no heart disease "
        f"({analysis['raw_no_heart_disease_percent']:.2f}%), "
        f"{analysis['raw_heart_disease']} heart disease "
        f"({analysis['raw_heart_disease_percent']:.2f}%)"
    )
    print(
        f"Duplicate rows identified: {analysis['duplicates']} "
        f"(unique rows if removed: {analysis['deduplicated_rows']})"
    )
    print(
        "Unique-row class count: "
        f"{analysis['deduplicated_no_heart_disease']} no heart disease, "
        f"{analysis['deduplicated_heart_disease']} heart disease"
    )
    print("Exact duplicate rows are removed before training to keep evaluation fair.")
    print()

    print("2. PREPROCESSING & MODEL")
    print("-" * 72)
    print(
        "Duplicate removal -> 80/20 train-test split -> scaling + one-hot encoding"
    )
    print(f"Model: {KNN_MODEL_NAME} (n_neighbors=11, metric=minkowski)")
    print("StandardScaler is applied before KNN so all features share the same distance scale.")
    print()

    models, metrics, _, _ = train_models(df, selected_models=[KNN_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("3. FINAL RESULT")
    print("-" * 72)
    print(comparison.to_string(index=False))

    knn_metrics = metrics[KNN_MODEL_NAME]
    print()
    print(f"Accuracy : {knn_metrics['accuracy']:.4f}")
    print(f"Precision: {knn_metrics['precision']:.4f}")
    print(f"Recall   : {knn_metrics['recall']:.4f}")
    print(f"F1-score : {knn_metrics['f1_score']:.4f}")
    print(f"Backend prototype model: {BEST_MODEL_NAME}")
    print()
    tn, fp = knn_metrics["confusion_matrix"][0]
    fn, tp = knn_metrics["confusion_matrix"][1]
    print("Confusion matrix:")
    print(f"True Negative  (correct no heart disease): {tn}")
    print(f"False Positive (predicted disease wrongly): {fp}")
    print(f"False Negative (missed heart disease): {fn}")
    print(f"True Positive  (correct heart disease): {tp}")
    print()
    print("Classification report:")
    print(knn_metrics["classification_report"])

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_knn_visualizations(
        comparison,
        knn_metrics["confusion_matrix"],
        output_dir,
    )
    print("4. OUTPUT")
    print("-" * 72)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

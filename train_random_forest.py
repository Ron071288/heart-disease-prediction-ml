from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from heart_model import (
    RANDOM_FOREST_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    random_forest_feature_importances,
    metrics_to_table,
    save_artifacts,
    train_models,
)
from heart_visuals import save_rf_visualizations


def format_metric_table(table):
    display_table = table.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")
    return display_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest heart disease prediction model."
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
    print("HEART DISEASE PREDICTION - RANDOM FOREST")
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
    print("Random Forest setup      : 200 trees, max_depth=8, max_features=sqrt")
    print()

    models, metrics, _, _ = train_models(df, selected_models=[RANDOM_FOREST_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("2. RANDOM FOREST RESULT")
    print("-" * 76)
    print(format_metric_table(comparison).to_string(index=False))

    rf_metrics = metrics[RANDOM_FOREST_MODEL_NAME]
    print()
    print(
        "Scores                   : "
        f"Accuracy {rf_metrics['accuracy'] * 100:.2f}%, "
        f"Precision {rf_metrics['precision'] * 100:.2f}%, "
        f"Recall {rf_metrics['recall'] * 100:.2f}%, "
        f"F1 {rf_metrics['f1_score'] * 100:.2f}%"
    )
    tn, fp = rf_metrics["confusion_matrix"][0]
    fn, tp = rf_metrics["confusion_matrix"][1]
    print(f"Confusion matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    print("3. TOP RANDOM FOREST FEATURE EFFECTS")
    print("-" * 76)
    print("Sorted from strongest to weakest predictive influence.")
    importances = random_forest_feature_importances(models[RANDOM_FOREST_MODEL_NAME])
    print(f"{'No.':<4} {'Feature':<52} {'Importance':>12}")
    print("-" * 76)
    for row_number, row in enumerate(importances.head(8).itertuples(), start=1):
        feature = textwrap.shorten(row.Feature, width=52, placeholder="...")
        print(f"{row_number:<4} {feature:<52} {row.Importance:>12.4f}")
    print()

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_rf_visualizations(
        comparison,
        rf_metrics["confusion_matrix"],
        importances,
        output_dir,
    )
    print("4. OUTPUT")
    print("-" * 76)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("- feature_importances.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

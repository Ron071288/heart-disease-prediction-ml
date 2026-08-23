from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from heart_model import (
    BEST_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    random_forest_feature_importances,
    metrics_to_table,
    save_artifacts,
    train_models,
)
from heart_visuals import save_rf_visualizations


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

    print("=" * 72)
    print("HEART DISEASE PREDICTION - RANDOM FOREST")
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
    print("Training uses the Kaggle-provided dataset after validation so interaction patterns can be learned.")
    print()

    print("2. PREPROCESSING & MODEL")
    print("-" * 72)
    print(
        "Data validation -> 80/20 train-test split -> scaling + one-hot encoding"
    )
    print(f"Model: {RANDOM_FOREST_MODEL_NAME} (n_estimators=200, max_depth=8, max_features='sqrt')")
    print()

    models, metrics, _, _ = train_models(df, selected_models=[RANDOM_FOREST_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("3. FINAL RESULT")
    print("-" * 72)
    print(comparison.to_string(index=False))

    rf_metrics = metrics[RANDOM_FOREST_MODEL_NAME]
    print()
    print(f"Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall   : {rf_metrics['recall']:.4f}")
    print(f"F1-score : {rf_metrics['f1_score']:.4f}")
    print()
    tn, fp = rf_metrics["confusion_matrix"][0]
    fn, tp = rf_metrics["confusion_matrix"][1]
    print("Confusion matrix:")
    print(f"True Negative  (correct no heart disease): {tn}")
    print(f"False Positive (predicted disease wrongly): {fp}")
    print(f"False Negative (missed heart disease): {fn}")
    print(f"True Positive  (correct heart disease): {tp}")
    print()

    print("4. TOP INFLUENTIAL FEATURES")
    print("-" * 72)
    print("Sorted from strongest to weakest predictive influence.")
    print()
    importances = random_forest_feature_importances(models[RANDOM_FOREST_MODEL_NAME])
    print(f"{'No.':<4} {'Feature':<52} {'Importance':>12}")
    print("-" * 72)
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
    print("5. OUTPUT")
    print("-" * 72)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("- feature_importances.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

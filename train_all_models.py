from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from heart_model import (
    FINAL_LOGISTIC_VARIANT,
    KNN_MODEL_NAME,
    LOGISTIC_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    logistic_regression_coefficients,
    metrics_to_table,
    random_forest_feature_importances,
    save_artifacts,
    train_models,
)
from heart_visuals import save_visualizations


def format_metric_table(table):
    display_table = table.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")
    return display_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare all heart disease prediction models."
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


def print_scores(model_name: str, model_metrics: dict) -> None:
    print(f"{model_name} performance")
    print(
        "Scores                   : "
        f"Accuracy {model_metrics['accuracy'] * 100:.2f}%, "
        f"Precision {model_metrics['precision'] * 100:.2f}%, "
        f"Recall {model_metrics['recall'] * 100:.2f}%, "
        f"F1 {model_metrics['f1_score'] * 100:.2f}%"
    )
    tn, fp = model_metrics["confusion_matrix"][0]
    fn, tp = model_metrics["confusion_matrix"][1]
    print(f"Confusion matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main() -> None:
    args = parse_args()
    df = load_dataset(args.csv)
    analysis = dataset_analysis(df)

    print("=" * 76)
    print("HEART DISEASE PREDICTION - GROUP MODEL COMPARISON")
    print("=" * 76)
    print()

    print("1. DATASET & SHARED WORKFLOW")
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
    print("Fair comparison standard : same cleaned data, split, preprocessing, and metrics")
    print()

    models, metrics, _, _ = train_models(df)
    comparison = metrics_to_table(metrics)

    print("2. MODEL COMPARISON")
    print("-" * 76)
    print(format_metric_table(comparison).to_string(index=False))
    print()

    print("3. SHORT MODEL DETAILS")
    print("-" * 76)
    print_scores(LOGISTIC_MODEL_NAME, metrics[LOGISTIC_MODEL_NAME])
    print(f"Setup                    : {FINAL_LOGISTIC_VARIANT}")
    print()
    print_scores(RANDOM_FOREST_MODEL_NAME, metrics[RANDOM_FOREST_MODEL_NAME])
    print("Setup                    : 200 trees, max_depth=8, max_features=sqrt")
    print()
    print_scores(KNN_MODEL_NAME, metrics[KNN_MODEL_NAME])
    print("Setup                    : n_neighbors=11, metric=minkowski")
    print()

    print("4. TOP 8 LOGISTIC REGRESSION FEATURE EFFECTS")
    print("-" * 76)
    print("Positive = toward disease; negative = toward no disease.")
    print("Full coefficient list is saved to artifacts/logistic_regression_coefficients.csv.")
    coefficients = logistic_regression_coefficients(models[LOGISTIC_MODEL_NAME])
    print(f"{'No.':<4} {'Feature':<52} {'Direction':<25} {'Coef.':>8}")
    print("-" * 96)
    for row_number, row in enumerate(coefficients.head(8).itertuples(), start=1):
        direction = (
            "Toward disease"
            if row.Direction == "Toward heart disease"
            else "Toward no disease"
        )
        feature = textwrap.shorten(row.Feature, width=52, placeholder="...")
        print(f"{row_number:<4} {feature:<52} {direction:<25} {row.Coefficient:>8.4f}")
    print()

    print("5. RANDOM FOREST FEATURE IMPORTANCES")
    print("-" * 76)
    importances = random_forest_feature_importances(models[RANDOM_FOREST_MODEL_NAME])
    print(f"{'No.':<4} {'Feature':<52} {'Importance':>12}")
    print("-" * 76)
    for row_number, row in enumerate(importances.head(8).itertuples(), start=1):
        feature = textwrap.shorten(row.Feature, width=52, placeholder="...")
        print(f"{row_number:<4} {feature:<52} {row.Importance:>12.4f}")
    print()

    print("6. KNN FEATURE NOTE")
    print("-" * 76)
    print("KNN is distance-based, so it does not produce coefficients or feature importances.")
    print("Its performance is compared using the same accuracy, precision, recall, F1, and confusion matrix.")
    print()

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_visualizations(
        comparison,
        metrics[LOGISTIC_MODEL_NAME]["confusion_matrix"],
        coefficients,
        output_dir,
    )
    print("7. OUTPUT")
    print("-" * 76)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- model_comparison_chart.png")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("- feature_coefficients.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

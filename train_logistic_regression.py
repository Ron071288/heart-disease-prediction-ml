from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from heart_model import (
    FINAL_LOGISTIC_VARIANT,
    LOGISTIC_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    logistic_regression_coefficients,
    metrics_to_table,
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
        description="Train and evaluate Logistic Regression heart disease prediction model."
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
    print("HEART DISEASE PREDICTION - LOGISTIC REGRESSION")
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
    print("Logistic feature setup   : interaction features + L1 regularization (C=0.3)")
    print()

    models, metrics, _, _ = train_models(df, selected_models=[LOGISTIC_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("2. LOGISTIC REGRESSION RESULT")
    print("-" * 76)
    print(format_metric_table(comparison).to_string(index=False))

    logistic_metrics = metrics[LOGISTIC_MODEL_NAME]
    print()
    print("3. MODEL DETAIL")
    print("-" * 76)
    print(f"Backend model            : {FINAL_LOGISTIC_VARIANT}")
    print(
        "Scores                   : "
        f"Accuracy {logistic_metrics['accuracy'] * 100:.2f}%, "
        f"Precision {logistic_metrics['precision'] * 100:.2f}%, "
        f"Recall {logistic_metrics['recall'] * 100:.2f}%, "
        f"F1 {logistic_metrics['f1_score'] * 100:.2f}%"
    )
    tn, fp = logistic_metrics["confusion_matrix"][0]
    fn, tp = logistic_metrics["confusion_matrix"][1]
    print(f"Confusion matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    print("4. TOP LOGISTIC REGRESSION FEATURE EFFECTS")
    print("-" * 76)
    print("Positive = toward disease; negative = toward no disease.")
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

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_visualizations(
        comparison,
        logistic_metrics["confusion_matrix"],
        coefficients,
        output_dir,
    )
    print("5. OUTPUT")
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

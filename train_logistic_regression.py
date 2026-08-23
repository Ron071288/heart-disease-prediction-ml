from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from heart_model import (
    BEST_MODEL_NAME,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate heart disease prediction models."
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
    print("HEART DISEASE PREDICTION - MODEL COMPARISON")
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

    print("2. PREPROCESSING")
    print("-" * 72)
    print(
        "Duplicate removal -> 80/20 train-test split -> scaling + one-hot encoding "
        "-> interaction features"
    )
    print(f"Final model: {FINAL_LOGISTIC_VARIANT} (C=0.3, L1 regularization)")
    print("Interaction features combine two existing inputs so Logistic Regression can learn their joint effect.")
    print()

    models, metrics, _, _ = train_models(df)
    comparison = metrics_to_table(metrics)

    print("3. MODEL COMPARISON")
    print("-" * 72)
    print(comparison.to_string(index=False))

    logistic_metrics = metrics[LOGISTIC_MODEL_NAME]
    print()
    print("4. LOGISTIC REGRESSION RESULT")
    print("-" * 72)
    print(f"Accuracy : {logistic_metrics['accuracy']:.4f}")
    print(f"Precision: {logistic_metrics['precision']:.4f}")
    print(f"Recall   : {logistic_metrics['recall']:.4f}")
    print(f"F1-score : {logistic_metrics['f1_score']:.4f}")
    print(f"Backend prototype model: {BEST_MODEL_NAME}")
    print()
    tn, fp = logistic_metrics["confusion_matrix"][0]
    fn, tp = logistic_metrics["confusion_matrix"][1]
    print("Confusion matrix:")
    print(f"True Negative  (correct no heart disease): {tn}")
    print(f"False Positive (predicted disease wrongly): {fp}")
    print(f"False Negative (missed heart disease): {fn}")
    print(f"True Positive  (correct heart disease): {tp}")
    print()

    print("5. TOP INFLUENTIAL FEATURES")
    print("-" * 72)
    print("Sorted from strongest to weakest effect.")
    print("Positive direction = increases heart disease probability.")
    print("Negative direction = decreases heart disease probability.")
    print()
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
    print("6. OUTPUT")
    print("-" * 72)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- model_comparison_chart.png")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    print("- feature_coefficients.png")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

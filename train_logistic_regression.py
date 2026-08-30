"""
Ron Phua Jun Long - Logistic Regression training script.

This file is the individual demo script for the Logistic Regression part of the
heart disease prediction project. It loads the dataset, analyses the class
balance and duplicates, trains the final tuned Logistic Regression model with
interaction features, evaluates it on the 20% test set, explains the strongest
coefficients, and saves the trained model plus charts into the artifacts folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from console_format import (
    print_confusion_matrix,
    print_feature_effects,
    print_header,
    print_key_values,
    print_metric_table,
    print_output_location,
    print_score_line,
    print_section,
)
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


def parse_args() -> argparse.Namespace:
    # Keep file paths configurable without changing the code during a demo.
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

    # Step 1: Load the included local dataset by default. If --csv is provided,
    # that file is used instead.
    df = load_dataset(args.csv)

    # Step 2: Summarise the raw dataset before training so the report can show
    # sample count, class balance, and duplicate removal.
    analysis = dataset_analysis(df)

    print_header("HEART DISEASE PREDICTION - LOGISTIC REGRESSION")

    print_section(1, "Dataset and Workflow")
    print_key_values(
        [
            ("Raw dataset", f"{analysis['raw_rows']} samples, {analysis['columns']} columns"),
            ("Duplicate rows removed", str(analysis["duplicates"])),
            ("Training samples used", str(analysis["deduplicated_rows"])),
            (
                "Training class balance",
                f"{analysis['deduplicated_no_heart_disease']} no disease / "
                f"{analysis['deduplicated_heart_disease']} disease",
            ),
            ("Split", "80% training / 20% testing"),
            ("Preprocessing", "duplicate removal, scaling, one-hot encoding"),
            ("Logistic setup", "interaction features + L1 regularization (C=0.3)"),
        ]
    )

    # Step 3: Train only Ron's Logistic Regression model. The shared
    # train_models() function still applies the same 80/20 split and
    # preprocessing pipeline used in the group comparison.
    models, metrics, _, _ = train_models(df, selected_models=[LOGISTIC_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    # Step 4: Print the final test-set performance using accuracy, precision,
    # recall, F1-score, and confusion matrix.
    print_section(2, "Final Test Result")
    print_metric_table(comparison)

    logistic_metrics = metrics[LOGISTIC_MODEL_NAME]
    print_section(3, "Model Detail")
    print_key_values([("Backend model", FINAL_LOGISTIC_VARIANT)])
    print_score_line(logistic_metrics)
    print_confusion_matrix(logistic_metrics)

    # Step 5: Logistic Regression is interpretable because the learned
    # coefficients show which features push predictions toward disease or no
    # disease.
    print_section(4, "Top Logistic Regression Feature Effects")
    coefficients = logistic_regression_coefficients(models[LOGISTIC_MODEL_NAME])
    print_feature_effects(coefficients)

    # Step 6: Save reusable outputs for the Streamlit prototype and report:
    # trained model, metrics, confusion matrix chart, and coefficient chart.
    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_visualizations(
        comparison,
        logistic_metrics["confusion_matrix"],
        coefficients,
        output_dir,
    )
    print_section(5, "Output Files")
    print_output_location(
        output_dir,
        [
            "logistic_regression.joblib",
            "metrics.json",
            "model_comparison_chart.png",
            "metrics_chart.png",
            "confusion_matrix.png",
            "feature_coefficients.png",
        ],
    )


if __name__ == "__main__":
    main()

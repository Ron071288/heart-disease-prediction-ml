"""Train, evaluate, compare, and save all three heart disease prediction models.

This is the main group demonstration script. It applies one shared workflow to
Logistic Regression, Random Forest, and K-Nearest Neighbors so their test
metrics can be compared fairly. Individual training scripts remain available
for each member to explain their own model in more detail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from console_format import (
    print_confusion_matrix,
    print_feature_effects,
    print_feature_importances,
    print_header,
    print_key_values,
    print_metric_table,
    print_output_location,
    print_score_line,
    print_section,
)
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


def parse_args() -> argparse.Namespace:
    # Optional arguments let the group use another CSV or artifacts folder without
    # changing the source code.
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


def main() -> None:
    args = parse_args()

    # Step 1: Load the dataset and report its size, class balance, and exact
    # duplicate count before models are trained.
    df = load_dataset(args.csv)
    analysis = dataset_analysis(df)

    print_header("HEART DISEASE PREDICTION - GROUP MODEL COMPARISON")

    print_section(1, "Dataset and Shared Workflow")
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
            ("Comparison standard", "same cleaned data, split, preprocessing, and metrics"),
        ]
    )

    # Step 2: Train every model using the same cleaned data, stratified 80/20
    # train-test split, preprocessing pipeline, and evaluation metrics.
    models, metrics, _, _ = train_models(df)
    comparison = metrics_to_table(metrics)

    print_section(2, "Model Comparison")
    print_metric_table(comparison)

    # Step 3: Print each model's setup, four main scores, and confusion matrix.
    print_section(3, "Model Details")
    model_setups = [
        (LOGISTIC_MODEL_NAME, FINAL_LOGISTIC_VARIANT),
        (RANDOM_FOREST_MODEL_NAME, "200 trees, max_depth=8, max_features=sqrt"),
        (KNN_MODEL_NAME, "n_neighbors=11, weights=distance, metric=Manhattan"),
    ]
    for model_name, setup in model_setups:
        print(model_name)
        print("-" * len(model_name))
        print_key_values([("Setup", setup)])
        print_score_line(metrics[model_name])
        print_confusion_matrix(metrics[model_name])

    # Step 4: Logistic Regression coefficients show direction and strength of
    # individual and interaction-feature effects.
    print_section(4, "Top Logistic Regression Feature Effects")
    coefficients = logistic_regression_coefficients(models[LOGISTIC_MODEL_NAME])
    print_feature_effects(coefficients)

    # Step 5: Random Forest uses feature importances rather than coefficients.
    print_section(5, "Random Forest Feature Importances")
    importances = random_forest_feature_importances(models[RANDOM_FOREST_MODEL_NAME])
    print_feature_importances(importances)

    # Step 6: KNN predicts from distances to nearby records, so it has neither
    # coefficients nor tree-style feature-importance values.
    print_section(6, "KNN Explanation Note")
    print_key_values(
        [
            ("Feature explanation", "KNN is distance-based, so it has no coefficients."),
            (
                "Evaluation",
                "Compared using the same accuracy, precision, recall, F1-score, and confusion matrix.",
            ),
        ]
    )

    # Step 7: Save trained models, metrics, and charts for the Streamlit app,
    # report figures, and later demo sessions.
    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_visualizations(
        comparison,
        metrics[LOGISTIC_MODEL_NAME]["confusion_matrix"],
        coefficients,
        output_dir,
    )
    print_section(7, "Output Files")
    print_output_location(
        output_dir,
        [
            "model_comparison.csv",
            "metrics.json",
            "trained .joblib model files",
            "model_comparison_chart.png",
            "metrics_chart.png",
            "confusion_matrix.png",
            "feature_coefficients.png",
        ],
    )


if __name__ == "__main__":
    main()

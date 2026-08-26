from __future__ import annotations

import argparse
from pathlib import Path

from console_format import (
    print_confusion_matrix,
    print_feature_importances,
    print_header,
    print_key_values,
    print_metric_table,
    print_output_location,
    print_score_line,
    print_section,
)
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

    print_header("HEART DISEASE PREDICTION - RANDOM FOREST")

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
            ("Random Forest setup", "200 trees, max_depth=8, max_features=sqrt"),
        ]
    )

    models, metrics, _, _ = train_models(df, selected_models=[RANDOM_FOREST_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print_section(2, "Final Test Result")
    print_metric_table(comparison)

    rf_metrics = metrics[RANDOM_FOREST_MODEL_NAME]
    print_section(3, "Model Detail")
    print_score_line(rf_metrics)
    print_confusion_matrix(rf_metrics)

    print_section(4, "Top Random Forest Feature Effects")
    importances = random_forest_feature_importances(models[RANDOM_FOREST_MODEL_NAME])
    print_feature_importances(importances)

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_rf_visualizations(
        comparison,
        rf_metrics["confusion_matrix"],
        importances,
        output_dir,
    )
    print_section(5, "Output Files")
    print_output_location(
        output_dir,
        [
            "random_forest.joblib",
            "metrics.json",
            "metrics_chart.png",
            "confusion_matrix.png",
            "feature_importances.png",
        ],
    )


if __name__ == "__main__":
    main()

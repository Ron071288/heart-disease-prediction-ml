from __future__ import annotations

import argparse
from pathlib import Path

from console_format import (
    print_confusion_matrix,
    print_header,
    print_key_values,
    print_metric_table,
    print_output_location,
    print_score_line,
    print_section,
)
import heart_model as hm
from heart_model import (
    KNN_MODEL_NAME,
    MODEL_OPTIONS,
    dataset_analysis,
    load_dataset,
    metrics_to_table,
    save_artifacts,
    train_models,
    tune_knn_hyperparameters,
)
from sklearn.neighbors import KNeighborsClassifier

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
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning and use the pre-set defaults in MODEL_OPTIONS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.csv)
    analysis = dataset_analysis(df)

    print_header("HEART DISEASE PREDICTION - K-NEAREST NEIGHBORS")

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
        ]
    )

    # ------------------------------------------------------------------
    # 2. HYPERPARAMETER TUNING (training set / cross-validation only)
    # ------------------------------------------------------------------
    if not args.skip_tuning:
        print_section(2, "KNN Hyperparameter Tuning")
        print_key_values(
            [
                ("Validation method", "5-fold StratifiedKFold on training set only"),
                ("k values", "3, 5, 7, ..., 25"),
                ("Weights", "uniform, distance"),
                ("Distance metrics", "Manhattan and Euclidean"),
                ("Test-set rule", "not used during tuning"),
            ]
        )

        tuning_results = tune_knn_hyperparameters(df)

        # ---- Print top-10 combinations by CV accuracy ----
        print("Top tuning results")
        print(f"{'Rank':<5} {'k':<5} {'Weights':<10} {'Metric':<22} "
              f"{'CV Acc':>10} {'CV Std':>10} {'CV F1':>10}")
        print("-" * 78)
        for rank, row in enumerate(tuning_results.head(10).itertuples(), start=1):
            print(
                f"{rank:<5} {row.n_neighbors:<5} {row.weights:<10} {row.metric:<22} "
                f"{row._5 * 100:>9.2f}% {row._6 * 100:>9.2f}% {row._7 * 100:>9.2f}%"
            )
        print()

        # ---- Extract best combination ----
        best = tuning_results.iloc[0]
        best_k = int(best["n_neighbors"])
        best_weights = str(best["weights"])
        best_p = int(best["p"])
        best_metric_label = str(best["metric"])

        print_key_values(
            [
                (
                    "Best combination",
                    f"k={best_k}, weights={best_weights!r}, metric={best_metric_label}",
                ),
                (
                    "Best CV accuracy",
                    f"{best['CV Accuracy Mean'] * 100:.2f}% "
                    f"(+/- {best['CV Accuracy Std'] * 100:.2f}%)",
                ),
            ]
        )
        print()

        # ---- Persist full tuning grid for the Methodology report ----
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        tuning_csv = output_dir / "knn_tuning_results.csv"
        report_cols = ["n_neighbors", "weights", "metric",
                       "CV Accuracy Mean", "CV Accuracy Std",
                       "CV F1 Mean", "CV F1 Std"]
        tuning_results[report_cols].to_csv(tuning_csv, index=False, float_format="%.6f")
        print_key_values([("Full tuning grid saved", str(tuning_csv.resolve()))])

        # ---- Patch MODEL_OPTIONS so train_models() uses best params ----
        hm.MODEL_OPTIONS[KNN_MODEL_NAME] = KNeighborsClassifier(
            n_neighbors=best_k,
            weights=best_weights,
            metric="minkowski",
            p=best_p,
            n_jobs=1,
        )
        knn_setup_label = (
            f"n_neighbors={best_k}, weights={best_weights!r}, "
            f"metric={best_metric_label}"
        )
    else:
        current = MODEL_OPTIONS[KNN_MODEL_NAME]
        knn_setup_label = (
            f"n_neighbors={current.n_neighbors}, "
            f"weights={current.weights!r}, "
            f"metric={current.metric} (p={current.p}) [pre-set defaults, tuning skipped]"
        )

    print_section(3, "Selected KNN Setup")
    print_key_values([("KNN setup", knn_setup_label)])

    # ------------------------------------------------------------------
    # 3. TRAIN ON FULL TRAINING SET, EVALUATE ON HELD-OUT TEST SET
    # ------------------------------------------------------------------
    models, metrics, _, _ = train_models(df, selected_models=[KNN_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print_section(4, "Final Test Result")
    print_metric_table(comparison)

    knn_metrics = metrics[KNN_MODEL_NAME]
    print_section(5, "Model Detail")
    print_score_line(knn_metrics)
    print_confusion_matrix(knn_metrics)

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_knn_visualizations(
        comparison,
        knn_metrics["confusion_matrix"],
        output_dir,
    )
    print_section(6, "Output Files")
    output_files = [
        "knearest_neighbors.joblib",
        "metrics.json",
        "metrics_chart.png",
        "confusion_matrix.png",
    ]
    if not args.skip_tuning:
        output_files.append("knn_tuning_results.csv")
    print_output_location(output_dir, output_files)


if __name__ == "__main__":
    main()

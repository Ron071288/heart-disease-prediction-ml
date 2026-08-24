from __future__ import annotations

import argparse
from pathlib import Path

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


def format_metric_table(table):
    display_table = table.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")
    return display_table


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

    print("=" * 76)
    print("HEART DISEASE PREDICTION - K-NEAREST NEIGHBORS")
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
    print()

    # ------------------------------------------------------------------
    # 2. HYPERPARAMETER TUNING (training set / cross-validation only)
    # ------------------------------------------------------------------
    if not args.skip_tuning:
        print("2. KNN HYPERPARAMETER TUNING (5-fold StratifiedKFold on training set)")
        print("-" * 76)
        print("Searching: n_neighbors in {3,5,...,25}, weights in {uniform,distance},")
        print("           metric = minkowski with p in {1 (Manhattan), 2 (Euclidean)}")
        print("(Test set is never touched during this step.)")
        print()

        tuning_results = tune_knn_hyperparameters(df)

        # ---- Print top-10 combinations by CV accuracy ----
        print(f"{'Rank':<5} {'k':<5} {'Weights':<10} {'Metric':<22} "
              f"{'CV Acc Mean':>12} {'CV Acc Std':>11} {'CV F1 Mean':>11}")
        print("-" * 76)
        for rank, row in enumerate(tuning_results.head(10).itertuples(), start=1):
            print(
                f"{rank:<5} {row.n_neighbors:<5} {row.weights:<10} {row.metric:<22} "
                f"{row._5 * 100:>11.2f}% {row._6 * 100:>10.2f}% {row._7 * 100:>10.2f}%"
            )
        print()

        # ---- Extract best combination ----
        best = tuning_results.iloc[0]
        best_k = int(best["n_neighbors"])
        best_weights = str(best["weights"])
        best_p = int(best["p"])
        best_metric_label = str(best["metric"])

        print(
            f"Best combination         : k={best_k}, weights={best_weights!r}, "
            f"metric={best_metric_label}"
        )
        print(
            f"Best CV accuracy         : {best['CV Accuracy Mean'] * 100:.2f}% "
            f"(±{best['CV Accuracy Std'] * 100:.2f}%)"
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
        print(f"Full tuning grid saved   : {tuning_csv.resolve()}")
        print()

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

    print("KNN setup                :", knn_setup_label)
    print()

    # ------------------------------------------------------------------
    # 3. TRAIN ON FULL TRAINING SET, EVALUATE ON HELD-OUT TEST SET
    # ------------------------------------------------------------------
    models, metrics, _, _ = train_models(df, selected_models=[KNN_MODEL_NAME])
    comparison = metrics_to_table(metrics)

    print("3. KNN RESULT (held-out test set)")
    print("-" * 76)
    print(format_metric_table(comparison).to_string(index=False))

    knn_metrics = metrics[KNN_MODEL_NAME]
    print()
    print(
        "Scores                   : "
        f"Accuracy {knn_metrics['accuracy'] * 100:.2f}%, "
        f"Precision {knn_metrics['precision'] * 100:.2f}%, "
        f"Recall {knn_metrics['recall'] * 100:.2f}%, "
        f"F1 {knn_metrics['f1_score'] * 100:.2f}%"
    )
    tn, fp = knn_metrics["confusion_matrix"][0]
    fn, tp = knn_metrics["confusion_matrix"][1]
    print(f"Confusion matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    output_dir = Path(args.output)
    save_artifacts(models, metrics, output_dir)
    save_knn_visualizations(
        comparison,
        knn_metrics["confusion_matrix"],
        output_dir,
    )
    print()
    print("4. OUTPUT")
    print("-" * 76)
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("Charts saved:")
    print("- metrics_chart.png")
    print("- confusion_matrix.png")
    if not args.skip_tuning:
        print("- knn_tuning_results.csv  (methodology report data)")
    print("Run prototype: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

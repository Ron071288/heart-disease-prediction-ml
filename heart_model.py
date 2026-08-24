from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)


DATASET_HANDLE = "mfarhaannazirkhan/heart-dataset"
TARGET_COLUMN = "target"
RANDOM_STATE = 118
CODED_CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG Result",
    "thalachh": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Oldpeak",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia Result",
}


CATEGORY_LABELS = {
    "sex": {"0": "Female", "1": "Male"},
    "cp": {
        "0": "Typical angina",
        "1": "Atypical angina",
        "2": "Non-anginal pain",
        "3": "Asymptomatic",
    },
    "fbs": {
        "0": "Fasting blood sugar <= 120 mg/dl",
        "1": "Fasting blood sugar > 120 mg/dl",
    },
    "restecg": {
        "0": "Normal",
        "1": "ST-T wave abnormality",
        "2": "Left ventricular hypertrophy",
    },
    "exang": {"0": "No", "1": "Yes"},
    "slope": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"},
    "thal": {"0": "Unknown", "1": "Fixed defect", "2": "Normal", "3": "Reversible defect"},
}


LOGISTIC_MODEL_NAME = "Logistic Regression"
RANDOM_FOREST_MODEL_NAME = "Random Forest"
KNN_MODEL_NAME = "K-Nearest Neighbors"
BEST_MODEL_NAME = LOGISTIC_MODEL_NAME
FINAL_LOGISTIC_VARIANT = "Tuned Logistic Regression + Interaction Features"


MODEL_OPTIONS = {
    LOGISTIC_MODEL_NAME: LogisticRegression(
        C=0.3,
        penalty="l1",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    ),
    RANDOM_FOREST_MODEL_NAME: RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    ),
    KNN_MODEL_NAME: KNeighborsClassifier(
        n_neighbors=11,      # best from tune_knn_hyperparameters() grid search
        weights="distance",  # distance-weighted voting outperforms uniform
        metric="minkowski",
        p=1,                 # Manhattan distance (p=1) outperforms Euclidean (p=2)
        n_jobs=1,
    ),
}


LOGISTIC_VARIANTS = {
    "Baseline Logistic Regression": {
        "classifier": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "interactions": False,
    },
    "Tuned Logistic Regression": {
        "classifier": LogisticRegression(
            C=0.3,
            penalty="l1",
            solver="liblinear",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "interactions": False,
    },
    "Tuned Logistic Regression + Interaction Features": {
        "classifier": LogisticRegression(
            C=0.3,
            penalty="l1",
            solver="liblinear",
            max_iter=20000,
            random_state=RANDOM_STATE,
        ),
        "interactions": True,
    },
}


def load_dataset(local_csv: str | Path | None = None) -> pd.DataFrame:
    """Load the Kaggle heart dataset from a local CSV or KaggleHub."""
    if local_csv:
        csv_path = Path(local_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return pd.read_csv(csv_path)

    cached_csv = find_cached_kaggle_csv()
    if cached_csv:
        return pd.read_csv(cached_csv)

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install -r requirements.txt"
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file was found after downloading {DATASET_HANDLE}."
        )

    return pd.read_csv(csv_files[0])


def find_cached_kaggle_csv() -> Path | None:
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    dataset_root = cache_root / "mfarhaannazirkhan" / "heart-dataset"
    if not dataset_root.exists():
        return None

    csv_files = sorted(dataset_root.glob("**/*.csv"))
    return csv_files[0] if csv_files else None


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    """Normalize column names and validate the target."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}', but found: "
            f"{', '.join(cleaned.columns)}"
        )

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def dataset_analysis(df: pd.DataFrame) -> dict[str, Any]:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    if TARGET_COLUMN not in normalized.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'.")

    target_counts = normalized[TARGET_COLUMN].value_counts().sort_index()
    total = int(target_counts.sum())
    duplicate_count = int(normalized.duplicated().sum())
    clean_without_duplicates = clean_dataset(normalized, drop_duplicates=True)
    clean_counts = clean_without_duplicates[TARGET_COLUMN].value_counts().sort_index()

    return {
        "raw_rows": int(normalized.shape[0]),
        "columns": int(normalized.shape[1]),
        "duplicates": duplicate_count,
        "raw_no_heart_disease": int(target_counts.get(0, 0)),
        "raw_heart_disease": int(target_counts.get(1, 0)),
        "raw_no_heart_disease_percent": float(target_counts.get(0, 0) / total * 100),
        "raw_heart_disease_percent": float(target_counts.get(1, 0) / total * 100),
        "deduplicated_rows": int(clean_without_duplicates.shape[0]),
        "deduplicated_no_heart_disease": int(clean_counts.get(0, 0)),
        "deduplicated_heart_disease": int(clean_counts.get(1, 0)),
    }


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    coded_categorical = [
        column for column in CODED_CATEGORICAL_COLUMNS if column in X.columns
    ]
    numeric_features = [
        column
        for column in X.select_dtypes(include=["number", "bool"]).columns
        if column not in coded_categorical
    ]
    text_categorical = [
        column
        for column in X.columns
        if column not in numeric_features and column not in coded_categorical
    ]
    categorical_features = coded_categorical + text_categorical

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_logistic_pipeline(
    X: pd.DataFrame,
    variant_name: str = "Tuned Logistic Regression + Interaction Features",
) -> Pipeline:
    if variant_name not in LOGISTIC_VARIANTS:
        raise ValueError(f"Unknown Logistic Regression variant: {variant_name}")

    variant = LOGISTIC_VARIANTS[variant_name]
    steps = [("preprocessor", build_preprocessor(X))]
    if variant["interactions"]:
        steps.append(
            (
                "interactions",
                PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False,
                ),
            )
        )
    steps.append(("classifier", variant["classifier"]))
    return Pipeline(steps=steps)


def build_model_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    if model_name == LOGISTIC_MODEL_NAME:
        return build_logistic_pipeline(X, FINAL_LOGISTIC_VARIANT)

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", MODEL_OPTIONS[model_name]),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["No Heart Disease", "Heart Disease"],
            zero_division=0,
        ),
    }


def train_models(
    df: pd.DataFrame,
    selected_models: list[str] | None = None,
    drop_duplicates: bool = True,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]], pd.DataFrame, pd.Series]:
    cleaned = clean_dataset(df, drop_duplicates=drop_duplicates)
    X, y = split_features_target(cleaned)
    models_to_train = selected_models or list(MODEL_OPTIONS.keys())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    trained_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for model_name in models_to_train:
        pipeline = build_model_pipeline(model_name, X_train)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        trained_models[model_name] = pipeline
        metrics[model_name] = evaluate_predictions(y_test, predictions)

    return trained_models, metrics, X_test, y_test


def evaluate_logistic_experiments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = [
        ("Duplicate-removed dataset", True),
        ("Kaggle-provided dataset", False),
    ]

    for scenario_name, drop_duplicates in scenarios:
        cleaned = clean_dataset(df, drop_duplicates=drop_duplicates)
        X, y = split_features_target(cleaned)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        for variant_name in LOGISTIC_VARIANTS:
            pipeline = build_logistic_pipeline(X_train, variant_name)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            cv_pipeline = build_logistic_pipeline(X, variant_name)
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            cv_scores = cross_validate(
                cv_pipeline,
                X,
                y,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "f1"],
                n_jobs=1,
            )

            rows.append(
                {
                    "Dataset Scenario": scenario_name,
                    "Model Version": variant_name,
                    "Test Accuracy": accuracy_score(y_test, predictions),
                    "Test Precision": precision_score(
                        y_test, predictions, zero_division=0
                    ),
                    "Test Recall": recall_score(y_test, predictions, zero_division=0),
                    "Test F1-score": f1_score(y_test, predictions, zero_division=0),
                    "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
                    "CV Accuracy Std": cv_scores["test_accuracy"].std(),
                }
            )

    return pd.DataFrame(rows)


def tune_knn_hyperparameters(
    df: pd.DataFrame,
    n_neighbors_range: list[int] | None = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Search over KNN hyperparameters using StratifiedKFold cross-validation.

    Tuning is performed exclusively on the training split — the held-out test
    set is never seen during this step, preserving honest final evaluation.

    Parameters
    ----------
    df:
        Raw dataset as returned by ``load_dataset``.
    n_neighbors_range:
        Odd values of k to evaluate.  Defaults to ``range(3, 26, 2)``.
    drop_duplicates:
        Whether to remove duplicate rows before splitting, consistent with
        the rest of the training pipeline.

    Returns
    -------
    pd.DataFrame
        One row per (k, weights, p) combination with CV accuracy mean/std,
        sorted by mean CV accuracy descending.
    """
    if n_neighbors_range is None:
        n_neighbors_range = list(range(3, 26, 2))  # odd numbers 3..25

    cleaned = clean_dataset(df, drop_duplicates=drop_duplicates)
    X, y = split_features_target(cleaned)

    X_train, _X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = [
        {"weights": w, "p": p}
        for w in ("uniform", "distance")
        for p in (1, 2)
    ]

    rows = []
    for k in n_neighbors_range:
        for params in param_grid:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_train)),
                    (
                        "classifier",
                        KNeighborsClassifier(
                            n_neighbors=k,
                            weights=params["weights"],
                            metric="minkowski",
                            p=params["p"],
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "f1"],
                n_jobs=1,
            )
            rows.append(
                {
                    "n_neighbors": k,
                    "weights": params["weights"],
                    "metric": f"minkowski (p={params['p']})",
                    "p": params["p"],
                    "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
                    "CV Accuracy Std": cv_scores["test_accuracy"].std(),
                    "CV F1 Mean": cv_scores["test_f1"].mean(),
                    "CV F1 Std": cv_scores["test_f1"].std(),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["CV Accuracy Mean", "CV F1 Mean"], ascending=False
    ).reset_index(drop=True)
    return results


def metrics_to_table(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(model_metrics["accuracy"], 4),
                "Precision": round(model_metrics["precision"], 4),
                "Recall": round(model_metrics["recall"], 4),
                "F1-score": round(model_metrics["f1_score"], 4),
            }
        )
    return pd.DataFrame(rows)


def logistic_regression_coefficients(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Selected model does not expose coefficients.")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    if "interactions" in model.named_steps:
        feature_names = model.named_steps["interactions"].get_feature_names_out(
            feature_names
        )
    coefficients = classifier.coef_[0]

    table = pd.DataFrame(
        {
            "Raw Feature": feature_names,
            "Feature": [readable_feature_name(name) for name in feature_names],
            "Direction": [
                "Toward heart disease" if value > 0 else "Toward no heart disease"
                for value in coefficients
            ],
            "Coefficient": coefficients,
            "AbsCoefficient": np.abs(coefficients),
        }
    )
    return table.sort_values("AbsCoefficient", ascending=False).drop(
        columns=["AbsCoefficient"]
    )


def patient_prediction_contributions(
    model: Pipeline,
    patient_input: pd.DataFrame,
    top_n: int | None = 10,
) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Selected model does not expose coefficients.")

    transformed = model.named_steps["preprocessor"].transform(patient_input)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    if "interactions" in model.named_steps:
        transformed = model.named_steps["interactions"].transform(transformed)
        feature_names = model.named_steps["interactions"].get_feature_names_out(
            feature_names
        )

    values = np.asarray(transformed).reshape(-1)
    coefficients = classifier.coef_[0]
    contributions = values * coefficients

    table = pd.DataFrame(
        {
            "Feature": [readable_feature_name(name) for name in feature_names],
            "Input Effect": [
                "Pushes toward heart disease" if value > 0 else "Pushes toward no heart disease"
                for value in contributions
            ],
            "Contribution": contributions,
            "AbsContribution": np.abs(contributions),
        }
    )
    table = table[table["AbsContribution"] > 0]
    sorted_table = table.sort_values("AbsContribution", ascending=False)
    if top_n is not None:
        sorted_table = sorted_table.head(top_n)

    return sorted_table.drop(columns=["AbsContribution"]).reset_index(drop=True)


def readable_feature_name(raw_feature_name: str) -> str:
    if " " in raw_feature_name:
        readable_parts = [
            readable_feature_name(part)
            for part in raw_feature_name.split(" ")
        ]
        return " x ".join(readable_parts)

    feature_name = raw_feature_name
    for prefix in ("numeric__", "categorical__"):
        if feature_name.startswith(prefix):
            feature_name = feature_name.removeprefix(prefix)
            break

    if raw_feature_name.startswith("categorical__") and "_" in feature_name:
        column_name, category_value = feature_name.rsplit("_", 1)
        label = FEATURE_LABELS.get(column_name, column_name)
        category_label = CATEGORY_LABELS.get(column_name, {}).get(category_value)
        if category_label:
            return f"{label} = {category_value} ({category_label})"
        return f"{label} = {category_value}"

    return FEATURE_LABELS.get(feature_name, feature_name)


def random_forest_feature_importances(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("Selected model does not expose feature importances.")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    if "interactions" in model.named_steps:
        feature_names = model.named_steps["interactions"].get_feature_names_out(
            feature_names
        )
    importances = classifier.feature_importances_

    table = pd.DataFrame(
        {
            "Raw Feature": feature_names,
            "Feature": [readable_feature_name(name) for name in feature_names],
            "Importance": importances,
        }
    )
    return table.sort_values("Importance", ascending=False)


def save_artifacts(
    models: dict[str, Pipeline],
    metrics: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        filename = model_name.lower().replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        joblib.dump(model, out / f"{filename}.joblib")

    if "Logistic Regression" in models:
        logistic_regression_coefficients(models["Logistic Regression"]).to_csv(
            out / "logistic_regression_coefficients.csv",
            index=False,
        )

    if "Random Forest" in models:
        random_forest_feature_importances(models["Random Forest"]).to_csv(
            out / "random_forest_importances.csv",
            index=False,
        )

    table = metrics_to_table(metrics)
    table.to_csv(out / "model_comparison.csv", index=False)

    with (out / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

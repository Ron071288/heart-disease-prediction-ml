from __future__ import annotations

import pandas as pd
import streamlit as st

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
    train_models,
)
from heart_visuals import (
    create_confusion_matrix_chart,
    create_feature_coefficient_chart,
    create_feature_importance_chart,
    create_knn_tuning_chart,
    create_model_comparison_chart,
)


DEVELOPER_USERNAME = "ron123"
DEVELOPER_PASSWORD = "admin123"


DISPLAY_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Result",
    "thalachh": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Oldpeak",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia Result",
}


CATEGORY_DESCRIPTIONS = {
    "sex": {0: "0 - Female", 1: "1 - Male"},
    "cp": {
        0: "0 - Typical angina",
        1: "1 - Atypical angina",
        2: "2 - Non-anginal pain",
        3: "3 - Asymptomatic",
    },
    "fbs": {
        0: "0 - Fasting blood sugar <= 120 mg/dl",
        1: "1 - Fasting blood sugar > 120 mg/dl",
    },
    "restecg": {
        0: "0 - Normal",
        1: "1 - ST-T wave abnormality",
        2: "2 - Left ventricular hypertrophy",
    },
    "exang": {0: "0 - No", 1: "1 - Yes"},
    "slope": {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"},
    "thal": {
        0: "0 - Unknown",
        1: "1 - Fixed defect",
        2: "2 - Normal",
        3: "3 - Reversible defect",
    },
}


NUMERIC_INPUT_SETTINGS = {
    "age": {"min": 0, "max": 120, "step": 1},
    "trestbps": {"min": 40, "max": 260, "step": 1},
    "chol": {"min": 80, "max": 700, "step": 1},
    "thalachh": {"min": 40, "max": 250, "step": 1},
    "oldpeak": {"min": 0.0, "max": 10.0, "step": 0.1, "format": "%.1f"},
}


st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset(csv_path: str | None) -> pd.DataFrame:
    return load_dataset(csv_path)


@st.cache_resource(show_spinner=False)
def get_trained_models(csv_path: str | None):
    df = get_dataset(csv_path)
    return train_models(df)


def display_label(column_name: str) -> str:
    return DISPLAY_LABELS.get(column_name, column_name)


def format_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    display_table = table.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")
    return display_table


def select_best_prediction_model(model_metrics: dict) -> str | None:
    if not model_metrics:
        return None

    return max(
        model_metrics,
        key=lambda model_name: (
            model_metrics[model_name]["accuracy"],
            model_metrics[model_name]["precision"],
            model_metrics[model_name]["recall"],
            model_metrics[model_name]["f1_score"],
        ),
    )


def model_display_name(model_name: str) -> str:
    if model_name == LOGISTIC_MODEL_NAME:
        return FINAL_LOGISTIC_VARIANT
    return model_name


def numeric_input(column_name: str, series: pd.Series):
    settings = NUMERIC_INPUT_SETTINGS.get(column_name, {})
    median_value = float(series.median())

    if pd.api.types.is_integer_dtype(series):
        return st.number_input(
            display_label(column_name),
            min_value=settings.get("min"),
            max_value=settings.get("max"),
            value=int(round(median_value)),
            step=settings.get("step", 1),
        )

    return st.number_input(
        display_label(column_name),
        min_value=settings.get("min"),
        max_value=settings.get("max"),
        value=round(median_value, 1),
        step=settings.get("step", 0.1),
        format=settings.get("format", "%.1f"),
    )


def user_input_form(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=["target"])
    user_values = {}

    st.subheader("Patient Details")
    columns = st.columns(2)

    for index, column_name in enumerate(feature_df.columns):
        series = feature_df[column_name]
        current_column = columns[index % 2]

        with current_column:
            if pd.api.types.is_numeric_dtype(series):
                median_value = float(series.median())

                if series.nunique() <= 10:
                    options = sorted(series.dropna().unique().tolist())
                    description_map = CATEGORY_DESCRIPTIONS.get(column_name, {})
                    formatted_options = [
                        description_map.get(option, str(option)) for option in options
                    ]
                    selected_text = st.selectbox(
                        display_label(column_name),
                        options=formatted_options,
                        index=options.index(median_value) if median_value in options else 0,
                    )
                    user_values[column_name] = options[formatted_options.index(selected_text)]
                else:
                    user_values[column_name] = numeric_input(column_name, series)
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                user_values[column_name] = st.selectbox(display_label(column_name), options=options)

    return pd.DataFrame([user_values])


if "developer_authenticated" not in st.session_state:
    st.session_state.developer_authenticated = False
if "show_developer_login" not in st.session_state:
    st.session_state.show_developer_login = False

title_column, action_column = st.columns([4, 1])
with title_column:
    st.title("Heart Disease Prediction")
    st.caption("Interactive Machine Learning Model Prototype")
with action_column:
    st.write("")
    st.write("")
    if st.session_state.developer_authenticated:
        if st.button("Log out", width="stretch"):
            st.session_state.developer_authenticated = False
            st.session_state.show_developer_login = False
            st.rerun()
    else:
        login_button_label = (
            "Patient View" if st.session_state.show_developer_login else "Developer Login"
        )
        if st.button(login_button_label, width="stretch"):
            st.session_state.show_developer_login = not st.session_state.show_developer_login
            st.rerun()

st.info("This prototype is for academic demonstration only and is not a medical diagnosis tool.")

if (
    not st.session_state.developer_authenticated
    and st.session_state.show_developer_login
):
    with st.container(border=True):
        st.subheader("Developer Login")
        st.caption("Sign in to view dataset analysis, model comparison, and backend outputs.")
        login_left, _login_right = st.columns([1, 2])
        with login_left:
            username = st.text_input("Username", key="developer_login_username")
            password = st.text_input(
                "Password",
                type="password",
                key="developer_login_password",
            )
            if st.button("Login", type="primary"):
                if username == DEVELOPER_USERNAME and password == DEVELOPER_PASSWORD:
                    st.session_state.developer_authenticated = True
                    st.session_state.show_developer_login = False
                    st.rerun()
                else:
                    st.error("Invalid developer credentials")
            st.caption("Demo credentials: ron123 / admin123")
    st.stop()

csv_path = None

try:
    df = get_dataset(csv_path)
    models, metrics, _, _ = get_trained_models(csv_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

analysis = dataset_analysis(df)
metric_table = metrics_to_table(metrics)
best_model_name = select_best_prediction_model(metrics)

if not st.session_state.developer_authenticated:
    st.subheader("Patient Prediction")
    patient_input = user_input_form(df)

    if st.button("Predict Heart Disease", type="primary"):
        if best_model_name is None or best_model_name not in models:
            st.warning("No trained prediction model is available.")
        else:
            model = models[best_model_name]
            prediction = int(model.predict(patient_input)[0])

            probability_text = ""
            if hasattr(model.named_steps["classifier"], "predict_proba"):
                probability = model.predict_proba(patient_input)[0][1]
                probability_text = f"Predicted probability of heart disease: {probability:.2%}"

            if prediction == 1:
                st.error("Prediction result: Heart disease detected")
            else:
                st.success("Prediction result: No heart disease detected")

            if probability_text:
                st.info(probability_text)

else:
    st.subheader("Developer Backend View")
    st.caption(
        "Backend view contains dataset analysis, model comparison, evaluation charts, "
        "and model explanation outputs."
    )

    st.subheader("Dataset Summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Raw samples", analysis["raw_rows"])
    metric_columns[1].metric("Duplicates removed", analysis["duplicates"])
    metric_columns[2].metric("Training samples", analysis["deduplicated_rows"])
    metric_columns[3].metric(
        "Training class balance",
        f"{analysis['deduplicated_no_heart_disease']} / {analysis['deduplicated_heart_disease']}",
        help="No heart disease / Heart disease",
    )
    st.caption(
        "Exact duplicate rows are removed before training. Logistic Regression also uses "
        "interaction features to learn combined feature patterns."
    )

    st.subheader("Model Comparison")
    st.dataframe(format_metric_table(metric_table), width="stretch", hide_index=True)
    st.pyplot(create_model_comparison_chart(metric_table), width="stretch")

    st.subheader("All Model Details")
    for model_name in [LOGISTIC_MODEL_NAME, RANDOM_FOREST_MODEL_NAME, KNN_MODEL_NAME]:
        if model_name not in models or model_name not in metrics:
            continue

        selected_metrics = metrics[model_name]
        if model_name == LOGISTIC_MODEL_NAME:
            st.markdown("### Logistic Regression Model")
            st.caption(f"Setup: {FINAL_LOGISTIC_VARIANT}")
        else:
            st.markdown(f"### {model_display_name(model_name)}")
        score_columns = st.columns(4)
        score_columns[0].metric("Accuracy", f"{selected_metrics['accuracy']:.2%}")
        score_columns[1].metric("Precision", f"{selected_metrics['precision']:.2%}")
        score_columns[2].metric("Recall", f"{selected_metrics['recall']:.2%}")
        score_columns[3].metric("F1-score", f"{selected_metrics['f1_score']:.2%}")

        if model_name == LOGISTIC_MODEL_NAME:
            coefficient_table = logistic_regression_coefficients(models[LOGISTIC_MODEL_NAME])
            st.subheader("Global Model Explanation")
            st.caption(
                "These charts explain how the trained model behaves overall, not only one patient."
            )
            left_chart, right_chart = st.columns([0.9, 1.1])
            with left_chart:
                st.pyplot(
                    create_confusion_matrix_chart(selected_metrics["confusion_matrix"]),
                    width="stretch",
                )
            with right_chart:
                st.pyplot(create_feature_coefficient_chart(coefficient_table), width="stretch")

            with st.expander("View all model feature coefficients", expanded=False):
                st.caption(
                    "The chart above shows the strongest effects only. This table includes every "
                    "non-zero single feature and interaction feature used by Logistic Regression "
                    "after preprocessing."
                )
                non_zero_coefficients = coefficient_table[
                    coefficient_table["Coefficient"].abs() > 0
                ].reset_index(drop=True)
                st.dataframe(
                    non_zero_coefficients[["Feature", "Direction", "Coefficient"]],
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Download all model coefficients CSV",
                    data=coefficient_table.to_csv(index=False).encode("utf-8"),
                    file_name="logistic_regression_all_coefficients.csv",
                    mime="text/csv",
                )
        elif model_name == RANDOM_FOREST_MODEL_NAME:
            importance_table = random_forest_feature_importances(
                models[RANDOM_FOREST_MODEL_NAME]
            )
            left_chart, right_chart = st.columns(2)
            with left_chart:
                st.pyplot(
                    create_confusion_matrix_chart(selected_metrics["confusion_matrix"]),
                    width="stretch",
                )
            with right_chart:
                st.pyplot(create_feature_importance_chart(importance_table), width="stretch")
        elif model_name == KNN_MODEL_NAME:
            import os as _os

            _artifacts_dir = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "artifacts"
            )
            _tuning_csv = _os.path.join(_artifacts_dir, "knn_tuning_results.csv")

            left_chart, right_chart = st.columns(2)
            with left_chart:
                st.pyplot(
                    create_confusion_matrix_chart(selected_metrics["confusion_matrix"]),
                    width="stretch",
                )
            with right_chart:
                if _os.path.exists(_tuning_csv):
                    tuning_results = pd.read_csv(_tuning_csv)
                    st.pyplot(create_knn_tuning_chart(tuning_results), width="stretch")
                else:
                    st.info(
                        "KNN tuning chart not available yet. "
                        "Run `python train_knn.py` to generate "
                        "`artifacts/knn_tuning_results.csv`, then refresh."
                    )

            st.subheader("KNN Hyperparameter Tuning Results")
            if _os.path.exists(_tuning_csv):
                tuning_results = pd.read_csv(_tuning_csv)
                display_cols = [
                    "n_neighbors", "weights", "metric",
                    "CV Accuracy Mean", "CV Accuracy Std",
                    "CV F1 Mean", "CV F1 Std",
                ]
                fmt_cols = ["CV Accuracy Mean", "CV Accuracy Std", "CV F1 Mean", "CV F1 Std"]
                tuning_display = tuning_results[display_cols].copy()
                for col in fmt_cols:
                    tuning_display[col] = tuning_display[col].map(lambda v: f"{v * 100:.2f}%")
                st.caption(
                    "All 52 combinations evaluated (k, weights, metric) sorted by "
                    "5-fold CV accuracy on the training set. The best combination "
                    "is highlighted at the top."
                )
                st.dataframe(tuning_display, width="stretch", hide_index=True)
                st.download_button(
                    "Download full tuning results CSV",
                    data=tuning_results.to_csv(index=False).encode("utf-8"),
                    file_name="knn_tuning_results.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "Tuning results not available. "
                    "Run `python train_knn.py` first."
                )

        st.divider()

st.caption("Academic prototype only. Prediction results should not be used as medical advice.")

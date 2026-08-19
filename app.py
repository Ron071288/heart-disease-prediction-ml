from __future__ import annotations

import pandas as pd
import streamlit as st

from heart_model import (
    BEST_MODEL_NAME,
    FINAL_LOGISTIC_VARIANT,
    LOGISTIC_MODEL_NAME,
    dataset_analysis,
    load_dataset,
    logistic_regression_coefficients,
    metrics_to_table,
    train_models,
)
from heart_visuals import (
    create_confusion_matrix_chart,
    create_feature_coefficient_chart,
    create_metrics_chart,
)


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
    "thal": {0: "0 - Unknown", 1: "1 - Fixed defect", 2: "2 - Normal", 3: "3 - Reversible defect"},
}


FRONTEND_MODEL_OPTIONS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Random Forest",
]


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
                    if pd.api.types.is_integer_dtype(series):
                        user_values[column_name] = st.number_input(
                            display_label(column_name),
                            value=int(round(series.median())),
                            step=1,
                        )
                    else:
                        user_values[column_name] = st.number_input(
                            display_label(column_name),
                            value=median_value,
                            step=0.1,
                            format="%.1f",
                        )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                user_values[column_name] = st.selectbox(display_label(column_name), options=options)

    return pd.DataFrame([user_values])


st.title("Heart Disease Prediction")
st.caption("Logistic Regression Backend Prototype with Interaction Features")

with st.sidebar:
    st.header("Dataset")
    local_csv = st.text_input(
        "Optional local CSV path",
        value="",
        help="Leave empty to load the Kaggle dataset using KaggleHub.",
    )
    csv_path = local_csv.strip() or None

try:
    df = get_dataset(csv_path)
    models, metrics, _, _ = get_trained_models(csv_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

analysis = dataset_analysis(df)
metric_table = metrics_to_table(metrics)
logistic_metrics = metrics[LOGISTIC_MODEL_NAME]
coefficient_table = logistic_regression_coefficients(models[LOGISTIC_MODEL_NAME])

st.subheader("Prediction")
model_column, backend_column = st.columns(2)
with model_column:
    selected_frontend_model = st.selectbox(
        "Select model",
        options=FRONTEND_MODEL_OPTIONS,
        help="The frontend is prepared for all group models. The current backend uses Logistic Regression with interaction features.",
    )
with backend_column:
    st.text_input("Backend model currently connected", value=FINAL_LOGISTIC_VARIANT, disabled=True)

patient_input = user_input_form(df)

if st.button("Predict Heart Disease", type="primary"):
    model = models[BEST_MODEL_NAME]
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

    st.write("Frontend selected model:", selected_frontend_model)
    st.write("Backend prediction model:", FINAL_LOGISTIC_VARIANT)
    st.write("Patient input")
    st.dataframe(patient_input, width="stretch")

st.divider()

with st.expander("Model Analysis and Charts", expanded=False):
    st.write("Dataset analysis")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Raw samples", analysis["raw_rows"])
    metric_columns[1].metric("Duplicate rows found", analysis["duplicates"])
    metric_columns[2].metric("Unique rows", analysis["deduplicated_rows"])
    metric_columns[3].metric(
        "Raw class balance",
        f"{analysis['raw_no_heart_disease']} / {analysis['raw_heart_disease']}",
        help="No heart disease / Heart disease",
    )
    st.caption(
        "The interaction-feature backend uses the Kaggle-provided dataset after validation, "
        "so the model can learn combined feature patterns."
    )

    st.write("Backend model performance")
    score_columns = st.columns(4)
    score_columns[0].metric("Accuracy", f"{logistic_metrics['accuracy']:.2%}")
    score_columns[1].metric("Precision", f"{logistic_metrics['precision']:.2%}")
    score_columns[2].metric("Recall", f"{logistic_metrics['recall']:.2%}")
    score_columns[3].metric("F1-score", f"{logistic_metrics['f1_score']:.2%}")

    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.pyplot(create_metrics_chart(metric_table), width="stretch")
    with right_chart:
        st.pyplot(
            create_confusion_matrix_chart(logistic_metrics["confusion_matrix"]),
            width="stretch",
        )
    st.pyplot(
        create_feature_coefficient_chart(coefficient_table),
        width="stretch",
    )

st.caption(
    "This prototype is for academic demonstration only and is not a medical diagnosis tool."
)

# Heart Disease Prediction

This folder contains the Logistic Regression model implementation and a simple Streamlit prototype for the group demo.

## Dataset

Kaggle dataset:
https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset/data

The code can load the dataset in two ways:

1. Automatically using KaggleHub.
2. From a local CSV file downloaded from Kaggle.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train and Evaluate Models

Automatically download from KaggleHub:

```bash
python train_logistic_regression.py
```

Or use a local CSV:

```bash
python train_logistic_regression.py --csv /path/to/heart.csv
```

The script uses an 80% training and 20% testing split. The current backend implementation trains and evaluates Logistic Regression. KNN and Random Forest can be added later by other group members using the same dataset, preprocessing flow, and evaluation metrics.

Saved outputs:

- `artifacts/model_comparison.csv`
- `artifacts/metrics.json`
- trained `.joblib` model file

## Run Prototype

```bash
streamlit run app.py
```

The prototype lets the user choose a model option, enter patient details, and view the prediction result. At this stage, the backend prediction uses the trained Logistic Regression model. After integration, the selected frontend model can be connected to each group member's backend model.

## Logistic Regression Explanation

Logistic Regression can be explained as:

- A supervised binary classification model.
- Suitable because the target has two classes: heart disease and no heart disease.
- Interpretable because it estimates the probability of heart disease.
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

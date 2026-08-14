# Heart Disease Prediction - Logistic Regression

This folder contains the code for Ron's Logistic Regression model and a simple Streamlit prototype for the group demo.

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

The script uses an 80% training and 20% testing split. It trains Logistic Regression, KNN, and Random Forest so the models can be compared fairly, but Ron's main model is Logistic Regression.

Saved outputs:

- `artifacts/model_comparison.csv`
- `artifacts/metrics.json`
- trained `.joblib` model files

## Run Prototype

```bash
streamlit run app.py
```

The prototype lets the user choose a model, enter patient details, and view the prediction result.

## Ron's Focus

Ron should explain Logistic Regression as:

- A supervised binary classification model.
- Suitable because the target has two classes: heart disease and no heart disease.
- Interpretable because it estimates the probability of heart disease.
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.


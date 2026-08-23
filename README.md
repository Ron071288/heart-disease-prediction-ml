# Heart Disease Prediction

This folder contains a heart disease prediction prototype using supervised machine learning. The integrated system trains Logistic Regression, Random Forest, and K-Nearest Neighbors using the same dataset, preprocessing steps, train-test split, and evaluation metrics.

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

## Train and Evaluate All Models

Automatically download from KaggleHub:

```bash
python train_all_models.py
```

Or use a local CSV:

```bash
python train_all_models.py --csv /path/to/heart.csv
```

The script removes exact duplicate rows, then uses an 80% training and 20% testing split. All models use the same preprocessing flow: missing-value handling, scaling for numeric features, and one-hot encoding for categorical features. Logistic Regression also uses interaction features so it can learn combined effects between features.

Individual model scripts are also provided for member-specific demonstration:

- `train_logistic_regression.py`
- `train_random_forest.py`
- `train_knn.py`

Saved outputs:

- `artifacts/model_comparison.csv`
- `artifacts/metrics.json`
- trained `.joblib` model files
- chart images for model comparison, confusion matrix, coefficients, and feature importances

## Run Prototype

```bash
python -m streamlit run app.py
```

The prototype lets the user choose a model, enter patient details, and view the prediction result. It also includes dataset analysis, model comparison, confusion matrix charts, and model-specific feature explanation where available.

## Logistic Regression Explanation

Logistic Regression can be explained as:

- A supervised binary classification model.
- Suitable because the target has two classes: heart disease and no heart disease.
- Interpretable because it estimates the probability of heart disease.
- Improved with interaction features, which combine two existing inputs so the model can learn their joint effect.
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

## Submission Checklist

- Documentation follows the provided documentation template.
- AI topic and problem are clearly stated: supervised machine learning for heart disease prediction.
- Dataset source is disclosed: Kaggle Heart Dataset.
- Dataset analysis is included: raw samples, duplicate rows, cleaned samples, and class balance.
- Prototype source code is included.
- Evaluation metrics are included for model comparison.
- AI Disclosure Statement and Plagiarism Statement are completed in the report template.

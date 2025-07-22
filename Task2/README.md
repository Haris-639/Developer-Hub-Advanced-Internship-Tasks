# Telco Customer Churn Prediction

This project builds a complete machine learning pipeline to predict whether a customer will churn using the **Telco Customer Churn** dataset. The task is a **binary classification** problem built using **Scikit-learn**, with models including **Logistic Regression** and **Random Forest**.

## Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target Variable**: `Churn` (Yes/No)
- **Features**: Customer demographic info, account details, service subscriptions, etc.

## Objective

To:
- Preprocess the dataset (cleaning, encoding, scaling)
- Train classification models using Scikit-learn
- Perform hyperparameter tuning using GridSearchCV
- Evaluate model performance using classification metrics
- Save the final model pipeline for reuse

## Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Joblib

## Pipeline Steps

- **Data Loading** – Read the Telco dataset from CSV.
- **Data Cleaning** – Handle missing values and convert data types.
- **Feature Engineering** – Split features and target, separate categorical and numerical columns.
- **Preprocessing** – Use `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
- **Model Building** – Train Logistic Regression and Random Forest using `Pipeline`.
- **Model Tuning** – Use `GridSearchCV` for hyperparameter optimization.
- **Evaluation** – Use `classification_report` to assess accuracy, precision, recall, F1-score.
- **Model Export** – Save best-performing models using `joblib`.

## Evaluation Metric

Each model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**


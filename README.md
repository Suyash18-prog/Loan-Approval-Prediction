# Loan Approval Prediction System

An end-to-end, leakage-safe machine-learning project for predicting a Kaggle-style loan application outcome (`Y`/`N`). This is decision-support software, not a substitute for regulated credit underwriting.

## Architecture

`CSV -> train/test split -> feature engineering -> IQR clipping -> imputation -> robust scaling/one-hot encoding -> model -> metrics/model artifact`

The same serialized `Pipeline` contains feature engineering and preprocessing, so inference cannot accidentally use a different transformation path. FastAPI loads the artifact once at startup; Streamlit calls the API.

## Dataset requirements

Download the public Kaggle Loan Prediction dataset (commonly named `train.csv`) and save it as `data/raw/loan_prediction.csv`. It must contain:

`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and `Loan_Status`.

`Loan_ID` is accepted and dropped. `Loan_Status` is the target (`Y` approved, `N` rejected).

| Feature | Meaning and expected relationship |
|---|---|
| Gender | Applicant-reported gender; historical association is dataset-specific, not causal. |
| Married | Household status; may proxy household finances in historical records. |
| Dependents | Number of dependents; higher obligations can affect affordability. |
| Education | Graduate/non-graduate indicator; can correlate with income, not guarantee approval. |
| Self_Employed | Employment type; income stability may differ by cohort. |
| Applicant/CoapplicantIncome | Monthly income fields used to derive total income and affordability ratios. |
| LoanAmount | Requested loan amount (dataset units are typically thousands). |
| Loan_Amount_Term | Repayment term in months; used to estimate EMI. |
| Credit_History | Whether historical credit guidelines were met; often highly predictive. |
| Property_Area | Urban, semiurban, or rural location category. |
| Loan_Status | Approval target: `Y` or `N`. |

## Run locally

```bash
cd load-approval-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.eda --data data/raw/loan_prediction.csv --output-dir data/processed/eda
python -m src.train --data data/raw/loan_prediction.csv
pytest -q
uvicorn backend.main:app --reload
# in another terminal:
streamlit run frontend/app.py
```

Swagger UI is at `http://localhost:8000/docs`. The API exposes `GET /`, `GET /health`, `GET /model-info`, and validated `POST /predict`.

## EDA

`src/eda.py` produces dataset dimensions, descriptive statistics, missing-value analysis, duplicate counts, target distribution, income and loan amount distributions, credit-history analysis, and a numeric correlation heatmap. The output is written to `data/processed/eda/`. Outliers are inspected in EDA and clipped only within training folds when justified; no target or test-fold information is used by preprocessing.

## Feature engineering and modeling

The pipeline derives `TotalIncome`, `LoanToIncomeRatio`, estimated monthly EMI, `IncomePerDependent`, and `LoanAmountRelativeToIncome`. Numeric values use training-fold IQR clipping, median imputation, and robust scaling; categorical values use most-frequent imputation and unknown-safe one-hot encoding.

Training compares Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost, and a tuned Random Forest. The selected model maximizes held-out F1 and breaks ties with ROC-AUC. Metrics include accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report. Accuracy alone hides asymmetric false-approval and false-rejection costs.

Run `python -m src.train ...` to generate data-derived `models/loan_model.joblib` and `models/loan_model.metrics.json`; no evaluation values are hardcoded. Feature importance can be extracted from the fitted estimator, and SHAP should be run as a separate review step for tree models. Importance and SHAP describe associations, not causation.

## Testing and future improvements

Tests cover feature derivation, missing values, validation, and API health behavior. Future work includes probability calibration, fairness analysis across protected groups, threshold optimization with business costs, monitoring/drift detection, authentication, and a reviewed SHAP report.

## Screenshots

Add deployment screenshots here after running the Streamlit app.

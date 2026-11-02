"""Leakage-safe preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .feature_engineering import LoanFeatureEngineer

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "LoanToIncomeRatio",
    "IncomePerDependent",
    "EstimatedEMI",
    "LoanAmountRelativeToIncome",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using training-fold IQR thresholds."""

    def __init__(self, columns: list[str] | None = None, whisker: float = 1.5) -> None:
        self.columns = columns
        self.whisker = whisker

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IQRClipper":
        self.columns_ = self.columns or list(X.select_dtypes(include=np.number).columns)
        q1 = X[self.columns_].quantile(0.25)
        q3 = X[self.columns_].quantile(0.75)
        spread = q3 - q1
        self.lower_ = q1 - self.whisker * spread
        self.upper_ = q3 + self.whisker * spread
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.columns_:
            if column in frame:
                frame[column] = frame[column].clip(self.lower_[column], self.upper_[column])
        return frame


def build_preprocessor() -> Pipeline:
    """Return the complete feature and preprocessing pipeline."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("features", LoanFeatureEngineer()),
            ("outlier_clipper", IQRClipper(columns=NUMERIC_FEATURES)),
            ("preprocessor", transformer),
        ]
    )

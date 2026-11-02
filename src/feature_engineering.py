"""Feature engineering for the loan approval dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add financially meaningful features without using the target."""

    def __init__(self) -> None:
        self.required_columns = (
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Dependents",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LoanFeatureEngineer":
        missing = [column for column in self.required_columns if column not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        applicant = pd.to_numeric(frame["ApplicantIncome"], errors="coerce")
        coapplicant = pd.to_numeric(frame["CoapplicantIncome"], errors="coerce")
        loan = pd.to_numeric(frame["LoanAmount"], errors="coerce")
        term = pd.to_numeric(frame["Loan_Amount_Term"], errors="coerce")
        dependents = (
            frame["Dependents"].astype("string").replace({"3+": "3"}).pipe(
                pd.to_numeric, errors="coerce"
            )
        )

        total_income = (applicant + coapplicant).clip(lower=0)
        frame["TotalIncome"] = total_income
        frame["LoanToIncomeRatio"] = loan / total_income.replace(0, np.nan)
        frame["IncomePerDependent"] = total_income / (dependents.fillna(0) + 1)

        monthly_rate = 0.08 / 12
        months = term.clip(lower=1)
        frame["EstimatedEMI"] = (
            loan * 1000 * monthly_rate * (1 + monthly_rate) ** months
        ) / (((1 + monthly_rate) ** months) - 1)
        frame["LoanAmountRelativeToIncome"] = loan / (total_income + 1)
        return frame.replace([np.inf, -np.inf], np.nan)

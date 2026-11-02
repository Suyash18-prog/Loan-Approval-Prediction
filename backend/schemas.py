"""Pydantic request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: float = Field(gt=0, le=10_000_000)
    CoapplicantIncome: float = Field(ge=0, le=10_000_000)
    LoanAmount: float = Field(gt=0, le=10_000)
    Loan_Amount_Term: float = Field(gt=0, le=600)
    Credit_History: float = Field(ge=0, le=1)
    Property_Area: Literal["Urban", "Semiurban", "Rural"]


class PredictionResponse(BaseModel):
    loan_status: Literal["Approved", "Rejected"]
    approval_probability: float
    rejection_probability: float

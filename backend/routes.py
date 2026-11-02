"""FastAPI routes."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from .schemas import LoanApplication, PredictionResponse

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"service": "loan-approval-prediction", "status": "ok"}


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    artifact = request.app.state.artifact
    return {
        "status": "ok" if artifact is not None else "degraded",
        "model_available": artifact is not None,
    }


@router.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return artifact["metadata"] | {"metrics": artifact["metrics"]}


@router.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, request: Request) -> dict[str, Any]:
    if request.app.state.artifact is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    try:
        pipeline = request.app.state.artifact["pipeline"]
        row = application.model_dump()
        frame = pd.DataFrame([row])
        prediction = str(pipeline.predict(frame)[0])
        probabilities = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        approval_probability = float(probabilities[classes.index("Y")])
        return {
            "loan_status": "Approved" if prediction == "Y" else "Rejected",
            "approval_probability": approval_probability,
            "rejection_probability": 1 - approval_probability,
        }
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {error}") from error

"""Reusable model loading and prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "loan_model.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run "
            "`python -m src.train --data data/raw/loan_prediction.csv` first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Model artifact has an invalid format.")
    return artifact


def predict_application(
    application: dict[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, float | str]:
    artifact = load_model(model_path)
    row = pd.DataFrame([application])
    pipeline = artifact["pipeline"]
    prediction = str(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    classes = list(pipeline.classes_)
    approval_index = classes.index("Y") if "Y" in classes else 1
    approval_probability = float(probabilities[approval_index])
    return {
        "loan_status": "Approved" if prediction == "Y" else "Rejected",
        "approval_probability": approval_probability,
        "rejection_probability": 1.0 - approval_probability,
    }

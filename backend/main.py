"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.predict import DEFAULT_MODEL_PATH, load_model

from .routes import router


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> FastAPI:
    app = FastAPI(
        title="Loan Approval Prediction API",
        version="1.0.0",
        description="Predict loan approval from applicant and financial details.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    try:
        artifact = load_model(model_path)
    except (FileNotFoundError, ValueError):
        artifact = None
    app.state.artifact = artifact
    app.include_router(router)
    return app


app = create_app()

"""Train, compare, tune, and serialize loan approval models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor

TARGET = "Loan_Status"
DROP_COLUMNS = ["Loan_ID"]


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Keep the dataset's Y/N labels while satisfying XGBoost's numeric-label API."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = 2,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "EncodedXGBClassifier":
        encoded = pd.Series(y).map({"N": 0, "Y": 1})
        if encoded.isna().any():
            raise ValueError("XGBoost labels must be N or Y.")
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, encoded.astype(int), **kwargs)
        return self

    @property
    def classes_(self) -> np.ndarray:
        return np.array(["N", "Y"])

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        encoded = self.estimator_.predict(X, **kwargs)
        return np.where(np.asarray(encoded).astype(int) == 1, "Y", "N")

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Kaggle-style CSV and validate its target."""
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    frame = frame.drop(columns=[column for column in DROP_COLUMNS if column in frame])
    target = frame.pop(TARGET).astype("string").str.upper().str.strip()
    valid = target.isin(["Y", "N"])
    if not valid.all():
        raise ValueError("Loan_Status must contain only Y and N values.")
    return frame, target


def make_models() -> dict[str, Any]:
    """Create the requested model families with reproducible defaults."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": EncodedXGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("Y")]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "Y").astype(int), probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["N", "Y"]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=["N", "Y"], zero_division=0, output_dict=True
        ),
    }


def train_models(data_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in make_models().items():
        pipeline = Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        fitted[name] = pipeline

    tuning_pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor()),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
    )
    search = GridSearchCV(
        tuning_pipeline,
        {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
        },
        scoring=make_scorer(f1_score, pos_label="Y"),
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_results = evaluate_model(search.best_estimator_, X_test, y_test)
    results["Tuned Random Forest"] = tuned_results
    fitted["Tuned Random Forest"] = search.best_estimator_
    best_name = max(results, key=lambda key: (results[key]["f1"], results[key]["roc_auc"]))
    artifact = {
        "pipeline": fitted[best_name],
        "metadata": {
            "model_name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": list(X.columns),
            "target": TARGET,
            "selection_metric": "f1, then roc_auc",
            "best_params": search.best_params_,
        },
        "metrics": results,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    destination.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to Kaggle-style CSV.")
    parser.add_argument(
        "--model-out",
        default=str(Path(__file__).resolve().parents[1] / "models" / "loan_model.joblib"),
    )
    args = parser.parse_args()
    artifact = train_models(args.data, args.model_out)
    print(f"Saved {artifact['metadata']['model_name']} to {args.model_out}")
    print(json.dumps(artifact["metrics"][artifact["metadata"]["model_name"]], indent=2))


if __name__ == "__main__":
    main()

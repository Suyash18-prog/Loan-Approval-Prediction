"""Evaluation helpers and optional visualization exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save a labeled confusion matrix from a training metrics record."""
    matrix = metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", xticklabels=["Rejected", "Approved"],
             yticklabels=["Rejected", "Approved"])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

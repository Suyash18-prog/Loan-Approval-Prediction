"""Generate reproducible EDA summaries and plots for a Kaggle-style dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET = "Loan_Status"


def run_eda(data_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Write summaries and diagnostic plots; return key dataset facts."""
    data = pd.read_csv(data_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = data[TARGET].astype("string").str.upper().str.strip()
    numeric = data.select_dtypes(include="number")

    summary = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": data.isna().sum().to_dict(),
        "target_distribution": target.value_counts(dropna=False).to_dict(),
        "describe": data.describe(include="all").transpose().fillna("").to_dict(),
    }
    pd.DataFrame(summary["missing_values"].items(), columns=["feature", "missing"]).to_csv(
        destination / "missing_values.csv", index=False
    )
    data.describe(include="all").transpose().to_csv(destination / "statistical_summary.csv")

    sns.set_theme(style="whitegrid")
    _save_countplot(data, TARGET, destination / "target_distribution.png", "Loan status")
    _save_countplot(data, "Credit_History", destination / "credit_history_by_status.png",
                    "Credit history distribution")
    _save_hist(data, "ApplicantIncome", destination / "applicant_income_distribution.png")
    _save_hist(data, "LoanAmount", destination / "loan_amount_distribution.png")
    if not numeric.empty:
        figure, axis = plt.subplots(figsize=(10, 7))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis)
        axis.set_title("Numeric feature correlation (association, not causation)")
        figure.tight_layout()
        figure.savefig(destination / "correlation_heatmap.png", dpi=150)
        plt.close(figure)
    return summary


def _save_countplot(
    data: pd.DataFrame, column: str, path: Path, title: str
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(data=data, x=column, hue=TARGET if column != TARGET else None, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_hist(data: pd.DataFrame, column: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(data=data, x=column, hue=TARGET, kde=True, element="step", ax=axis)
    axis.set_title(f"{column} distribution by loan status")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="data/processed/eda")
    arguments = parser.parse_args()
    facts = run_eda(arguments.data, arguments.output_dir)
    print(f"Analysed {facts['rows']} rows and {facts['columns']} columns.")

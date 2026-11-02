
from pathlib import Path
import json

import pandas as pd


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

TARGET_CANDIDATES = [
    "implantation_success",
    "implantation",
    "implantation outcome",
    "outcome",
    "target",
    "label"
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a supported research dataset."""

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    if file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file: {file_path}")


def inspect_dataset(file_path: Path) -> dict:
    """Generate a reproducible dataset quality report."""

    df = load_dataset(file_path)

    # Normalize column names for inspection only.
    normalized_columns = {
        col: str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    }

    target_columns = [
        col for col, normalized in normalized_columns.items()
        if normalized in TARGET_CANDIDATES
    ]

    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ).round(2),
        "unique_values": df.nunique(dropna=True).values
    })

    profile_path = REPORT_DIR / (
        f"{file_path.stem}_column_profile.csv"
    )

    profile.to_csv(profile_path, index=False)

    report = {
        "filename": file_path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_candidates": [
            str(col) for col in target_columns
        ],
        "column_names": [
            str(col) for col in df.columns
        ],
        "profile_file": str(profile_path)
    }

    return report


def main():

    files = [
        file for file in RAW_DIR.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"No CSV or XLSX files found in {RAW_DIR}"
        )

    reports = []

    for file in files:

        print("\n" + "=" * 60)
        print(f"INSPECTING: {file.name}")
        print("=" * 60)

        report = inspect_dataset(file)
        reports.append(report)

        print(f"Rows: {report['rows']}")
        print(f"Columns: {report['columns']}")
        print(f"Duplicate rows: {report['duplicate_rows']}")
        print(f"Missing cells: {report['missing_cells']}")

        print("\nColumn names:")
        for column in report["column_names"]:
            print(f" - {column}")

        print("\nPossible target columns:")
        print(report["target_candidates"])

        print("\nProfile saved:")
        print(report["profile_file"])

    report_path = REPORT_DIR / "dataset_audit.json"

    report_path.write_text(
        json.dumps(reports, indent=4),
        encoding="utf-8"
    )

    print("\nDataset inspection completed.")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()

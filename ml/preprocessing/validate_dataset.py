
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "ml" / "data" / "raw"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FILE = RAW_DIR / "Dataset Clean.xlsx"

TARGET = "Output(failed implantation or successful implantation)"


def load_data():

    df = pd.read_excel(FILE)

    df.columns = df.columns.astype(str).str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column not found. Available: {df.columns.tolist()}"
        )

    return df


def analyze_target(df):

    target = df[TARGET]

    print("\nTARGET ANALYSIS")
    print("=" * 60)

    print("\nData type:")
    print(target.dtype)

    print("\nUnique values:")
    print(target.dropna().unique().tolist())

    print("\nValue counts:")
    print(target.value_counts(dropna=False).to_string())

    print("\nTarget distribution (%):")
    print(
        (target.value_counts(normalize=True, dropna=False) * 100)
        .round(2)
        .to_string()
    )


def analyze_missing_values(df):

    print("\nMISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2)
    })

    print(missing[missing["missing_count"] > 0])

    missing.to_csv(
        REPORT_DIR / "missing_value_analysis.csv"
    )


def analyze_numeric_features(df):

    print("\nNUMERIC FEATURE ANALYSIS")
    print("=" * 60)

    numeric = df.select_dtypes(include="number")

    summary = numeric.describe().T

    print(summary.to_string())

    summary.to_csv(
        REPORT_DIR / "numeric_summary.csv"
    )


def investigate_beta_hcg(df):

    print("\nBETA-HCG INVESTIGATION")
    print("=" * 60)

    column = "β-HCG"

    if column not in df.columns:
        print("Column not found.")
        return

    print(df[column].describe())

    print("\nUnique values:")
    print(df[column].dropna().unique()[:20])


def compare_datasets():

    excel = pd.read_excel(
        RAW_DIR / "Dataset Clean.xlsx"
    )

    csv = pd.read_csv(
        RAW_DIR / "Normalized CSV.csv"
    )

    print("\nDATASET COMPARISON")
    print("=" * 60)

    print("Excel shape:", excel.shape)
    print("CSV shape:", csv.shape)

    print("Excel duplicates:", excel.duplicated().sum())
    print("CSV duplicates:", csv.duplicated().sum())

    print(
        "\nExcel column names:",
        excel.columns.tolist()
    )

    print(
        "\nCSV column names:",
        csv.columns.tolist()
    )


def main():

    df = load_data()

    analyze_target(df)

    analyze_missing_values(df)

    analyze_numeric_features(df)

    investigate_beta_hcg(df)

    compare_datasets()

    print("\nResearch data validation completed.")

    print(
        "Reports generated at:",
        REPORT_DIR
    )


if __name__ == "__main__":
    main()

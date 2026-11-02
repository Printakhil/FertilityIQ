
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "ml" / "data" / "raw" / "Dataset Clean.xlsx"
)

TARGET = (
    "Output(failed implantation or successful implantation)"
)


def load_data():

    df = pd.read_excel(DATA_PATH)

    df.columns = (
        df.columns.astype(str).str.strip()
    )

    return df


def validate_schema(df):

    required_columns = [
        "Female Age",
        "AMH",
        "Male Age",
        "Sperm count",
        "Sperm motility",
        "Sperm morphology",
        "Oocyte Maturity MII",
        "Oocyte Maturity MI",
        "Oocyte Maturity others",
        "Number of fertilized eggs",
        "Embryo Grade 1",
        "Embryo Grade 2",
        "Embryo Grade 3",
        "Embryo Grade 4",
        "Number of embryos",
        TARGET
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Schema validation passed.")

    return True


def validate_target(df):

    invalid = (
        df[TARGET].notna()
        & ~df[TARGET].isin([0, 1])
    )

    if invalid.any():

        raise ValueError(
            "Unexpected target values detected."
        )

    print("Target validation passed.")

    return True


def validate_numeric_features(df):

    numeric_columns = [
        "Female Age",
        "AMH",
        "Male Age",
        "Sperm count",
        "Sperm motility",
        "Sperm morphology",
        "Oocyte Maturity MII",
        "Oocyte Maturity MI",
        "Oocyte Maturity others",
        "Number of fertilized eggs",
        "Embryo Grade 1",
        "Embryo Grade 2",
        "Embryo Grade 3",
        "Embryo Grade 4",
        "Number of embryos"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print("Numeric feature validation completed.")

    return df


def main():

    df = load_data()

    validate_schema(df)

    validate_target(df)

    df = validate_numeric_features(df)

    print("\nDATA VALIDATION SUMMARY")
    print("-----------------------")

    print("Dataset shape:", df.shape)

    print(
        "Missing target records:",
        df[TARGET].isna().sum()
    )

    print(
        "Duplicate records:",
        df.duplicated().sum()
    )

    print("\nData validation completed successfully.")


if __name__ == "__main__":
    main()


from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import joblib


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    ROOT / "ml" / "data" / "raw" / "Dataset Clean.xlsx"
)

PROCESSED_DIR = ROOT / "ml" / "data" / "processed"

ARTIFACT_DIR = ROOT / "ml" / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = (
    "Output(failed implantation or successful implantation)"
)


# =====================================================
# CANDIDATE FEATURES
# =====================================================

FEATURES = [
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


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    df = pd.read_excel(RAW_PATH)

    df.columns = df.columns.astype(str).str.strip()

    return df


# =====================================================
# DATA PREPARATION
# =====================================================

def prepare_data(df):

    required = FEATURES + [TARGET]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = df[required].copy()

    for column in FEATURES + [TARGET]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Exclude records with unknown outcomes.
    data = data.dropna(subset=[TARGET])

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Unexpected target encoding.")

    # Keep original observations; don't automatically
    # delete AMH outliers.

    data[TARGET] = data[TARGET].astype(int)

    return data


# =====================================================
# SPLIT DATA
# =====================================================

def split_data(data):

    X = data[FEATURES]
    y = data[TARGET]

    # 70% training, 15% validation, 15% testing.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )


# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURES
            )
        ]
    )

    return preprocessor


# =====================================================
# SAVE PROCESSED DATA
# =====================================================

def save_split(
    X,
    y,
    filename,
    preprocessor
):

    transformed = preprocessor.transform(X)

    processed = pd.DataFrame(
        transformed,
        columns=FEATURES,
        index=X.index
    )

    processed[TARGET] = y.values

    processed.to_csv(
        PROCESSED_DIR / filename,
        index=False
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("Loading IVF dataset...")

    df = load_data()

    print("Original dataset:", df.shape)

    data = prepare_data(df)

    print("After target validation:", data.shape)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_data(data)

    print("\nSPLIT SUMMARY")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Fit preprocessing ONLY on training data.
    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    joblib.dump(
        preprocessor,
        ARTIFACT_DIR / "preprocessor.joblib"
    )

    save_split(
        X_train,
        y_train,
        "train.csv",
        preprocessor
    )

    save_split(
        X_val,
        y_val,
        "validation.csv",
        preprocessor
    )

    save_split(
        X_test,
        y_test,
        "test.csv",
        preprocessor
    )

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "original_records": int(len(df)),
        "usable_records": int(len(data)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "split_method": "stratified random split",
        "random_state": 42,
        "patient_grouping": "Not possible: patient ID not verified",
        "clinical_prediction_timing": "Not yet established",
        "amh_outlier_policy": "Retained pending investigation",
        "excluded_variables": [
            "β-HCG",
            "Egg retrieval date",
            "Medicines and treatments",
            "frezzing"
        ]
    }

    with open(
        PROCESSED_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("\nPreprocessing completed successfully.")

    print("Artifacts saved:", ARTIFACT_DIR)
    print("Processed datasets:", PROCESSED_DIR)


if __name__ == "__main__":
    main()

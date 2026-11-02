
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "ml" / "models" / "best_model.joblib"

FEATURE_COLUMNS = [
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
]

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_outcome(features: dict):

    model = get_model()

    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Preserve the exact training feature order.
    input_df = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    probabilities = model.predict_proba(input_df)[0]

    predicted_class = int(probabilities.argmax())

    return {
        "predicted_class": predicted_class,
        "class_0_probability": round(float(probabilities[0]), 4),
        "class_1_probability": round(float(probabilities[1]), 4),
        "model": "XGBoost",
        "research_only": True,
    }

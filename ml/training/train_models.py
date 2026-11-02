
"""
FertilityIQ - Model Training Pipeline

Models:
1. Logistic Regression (baseline)
2. XGBoost
3. LightGBM

Model selection is performed on validation ROC-AUC.
The test set is deliberately not used here.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "reports" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load preprocessed training and validation datasets."""

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")

    target_column = (
        "Output(failed implantation or successful implantation)"
    )

    if target_column not in train.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data. "
            f"Available columns: {list(train.columns)}"
        )

    if target_column not in validation.columns:
        raise ValueError("Target column missing from validation data.")

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column].astype(int)

    X_val = validation.drop(columns=[target_column])
    y_val = validation[target_column].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if X_train.isna().all().any():
        raise ValueError("At least one training feature contains only missing values.")

    return X_train, y_train, X_val, y_val



def calculate_metrics(model, X, y):
    """Calculate classification and probability metrics."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def build_models():
    """Define candidate models."""

    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.04,
                num_leaves=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )),
        ]),
    }


def main():
    print("=" * 65)
    print("FERTILITYIQ - MODEL TRAINING PIPELINE")
    print("=" * 65)

    X_train, y_train, X_val, y_val = load_data()

    print(f"\nTraining records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'-' * 55}")
        print(f"Training: {name}")
        print("-" * 55)

        model.fit(X_train, y_train)

        metrics = calculate_metrics(model, X_val, y_val)
        results[name] = metrics

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"Brier:     {metrics['brier_score']:.4f}")

        model_path = MODEL_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")

    # Select the model using validation ROC-AUC.
    best_model_name = max(
        results,
        key=lambda name: results[name]["roc_auc"],
    )

    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values(
        by="roc_auc",
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv"
    )

    with open(
        REPORT_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    best_model_path = MODEL_DIR / f"{best_model_name.lower()}_model.joblib"

    # Copy the selected validation model to a stable path.
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    with open(
        MODEL_DIR / "model_selection.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({
            "selected_model": best_model_name,
            "selection_metric": "validation ROC-AUC",
            "validation_roc_auc": results[best_model_name]["roc_auc"],
            "threshold": 0.5,
            "test_set_used_for_selection": False,
        }, file, indent=4)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(comparison.round(4).to_string())

    print(f"\nSelected model: {best_model_name}")
    print("Selection metric: Validation ROC-AUC")
    print("Test set was not used.")

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()

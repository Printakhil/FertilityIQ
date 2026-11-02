
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "reports" / "evaluation"
PLOT_DIR = REPORT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 65)
    print("FERTILITYIQ - FINAL TEST EVALUATION")
    print("=" * 65)

    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    target_column = (
        "Output(failed implantation or successful implantation)"
    )

    if target_column not in test.columns:
        raise ValueError(
            f"Target column not found. Available columns: {list(test.columns)}"
        )

    X_test = test.drop(columns=[target_column])
    y_test = test[target_column].astype(int)

    model = joblib.load(MODEL_DIR / "best_model.joblib")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(
            brier_score_loss(y_test, probabilities)
        ),
    }

    print("\nTEST METRICS")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name:15s}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions))

    # Save numerical results.
    with open(
        REPORT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Confusion matrix.
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.title("FertilityIQ - Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost (AUC = {metrics['roc_auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # Precision-recall curve.
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {metrics['pr_auc']:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Test Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    print("\nEvaluation artifacts saved:")
    print(REPORT_DIR)
    print(PLOT_DIR)

    print("\nFinal test evaluation completed.")


if __name__ == "__main__":
    main()

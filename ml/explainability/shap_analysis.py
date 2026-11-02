
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "reports" / "explainability"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():

    print("=" * 60)
    print("FERTILITYIQ - SHAP EXPLAINABILITY")
    print("=" * 60)

    # Load the trained model and test data.
    model = joblib.load(MODEL_DIR / "best_model.joblib")

    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    target_column = (
        "Output(failed implantation or successful implantation)"
    )

    X_test = test.drop(columns=[target_column])

    # Access the XGBoost estimator inside the sklearn pipeline.
    imputer = model.named_steps["imputer"]
    xgb_model = model.named_steps["model"]

    X_transformed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print("\nCalculating SHAP values...")

    # TreeExplainer is suitable for XGBoost.
    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(X_transformed)

    # Global feature importance bar plot.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        show=False,
    )
    plt.title("FertilityIQ - Global Feature Importance")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SHAP beeswarm summary plot.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        show=False,
    )
    plt.title("FertilityIQ - SHAP Summary")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Calculate mean absolute SHAP values.
    importance = pd.DataFrame({
        "feature": X_transformed.columns,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nSHAP analysis completed.")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()

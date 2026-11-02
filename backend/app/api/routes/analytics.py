
from pathlib import Path
import json
import pandas as pd

from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Research Analytics"]
)

# analytics.py is inside backend/app/api/routes/
# parents[4] correctly resolves to the FertilityIQ project root.
ROOT = Path(__file__).resolve().parents[4]

REPORTS = ROOT / "reports"


@router.get("/model-comparison")
def get_model_comparison():
    path = REPORTS / "training" / "validation_metrics.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Validation metrics file not found",
                "expected_path": str(path),
                "project_root": str(ROOT)
            }
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        return {
            "success": True,
            "evaluation_split": "validation",
            "models": metrics
        }

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid validation metrics JSON: {error}"
        )


@router.get("/shap")
def get_shap_importance():
    path = (
        REPORTS
        / "explainability"
        / "shap_feature_importance.csv"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "SHAP results file not found",
                "expected_path": str(path),
                "project_root": str(ROOT)
            }
        )

    try:
        df = pd.read_csv(path)

        return {
            "success": True,
            "features": df.to_dict(orient="records")
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read SHAP results: {error}"
        )

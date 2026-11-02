
from fastapi import APIRouter, HTTPException

from backend.app.schemas.prediction import IVFPredictionInput
from backend.app.services.prediction_service import predict_outcome


router = APIRouter(
    prefix="/api/v1",
    tags=["IVF Prediction"]
)


@router.post("/predict")
def predict_ivf_outcome(payload: IVFPredictionInput):

    try:
        features = payload.model_dump(by_alias=True)

        result = predict_outcome(features)

        return {
            "success": True,
            "prediction": result,
            "notice": (
                "Research prototype only. This output is not "
                "a clinical diagnosis or treatment recommendation."
            )
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

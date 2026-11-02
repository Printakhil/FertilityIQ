from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():

    return {
        "status": "healthy",
        "service": "FertilityIQ API",
        "version": "1.0.0",
        "message": "Backend is running successfully"
    }
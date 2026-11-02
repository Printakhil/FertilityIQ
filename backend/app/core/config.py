from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str = "FertilityIQ"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Explainable AI research prototype "
        "for IVF outcome analysis."
    )

    API_V1_PREFIX: str = "/api/v1"

    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
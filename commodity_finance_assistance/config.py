from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Commodity Finance Assistance"
    app_env: str = "development"
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
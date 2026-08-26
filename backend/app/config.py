import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "CodeMind AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Security
    SECRET_KEY: str = Field(default="codemind-super-secret-key-change-in-production-32bytes!")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./codemind.db"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # AI API Keys & Endpoints
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    LOCAL_MODEL_ENDPOINT: str = "http://localhost:11434/v1"
    LOCAL_MODEL_NAME: str = "local-llama3"

    DEFAULT_MODEL: str = "local-llama3"

    # Sandbox & Security
    SANDBOX_TIMEOUT_SECONDS: int = 15
    SANDBOX_MAX_MEMORY_MB: int = 512

    # Storage
    PROJECTS_DIR: str = "./projects"
    UPLOADS_DIR: str = "./uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

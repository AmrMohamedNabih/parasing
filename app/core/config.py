from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"

    # PDF Storage — local filesystem
    PDF_STORAGE_PATH: str = "./storage/pdfs"

    # Pipeline defaults
    DEFAULT_PIPELINE_MODE: str = "balanced"
    DEFAULT_OCR_ENGINE: str = "easyocr"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Kafka Integration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9094"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

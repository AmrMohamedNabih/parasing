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

    # --- Phase 2: RAG Query Service ---

    # Qdrant Vector Database
    QDRANT_URL: str = "http://127.0.0.1:6333"

    # Gemini LLM
    GEMINI_API_KEY: str = "AIzaSyD0HaIMQ1kG7vSwOzLFI21B2aY7p-adwhg"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Kafka Topics
    KAFKA_QUESTION_TOPIC: str = "question-requests"
    KAFKA_RESPONSE_TOPIC: str = "question-responses"
    KAFKA_ERROR_TOPIC: str = "question-errors"
    KAFKA_DELETE_TOPIC: str = "document-deletions"

    # Search
    TOP_K_RESULTS: int = 8
    SIMILARITY_THRESHOLD: float = 0.55

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Allow extra fields without crashing
    }


settings = Settings()

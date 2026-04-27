"""
RAG Service — Configuration
-----------------------------
All settings are loaded from the same .env file used by Phase 1.
Phase 2 variables are additive — Phase 1 settings are re-declared
here so this service is fully self-contained.
"""

from pydantic_settings import BaseSettings


class RagSettings(BaseSettings):
    # ── Phase 1 (shared) ──────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9094"
    DEBUG: bool = False

    # ── Qdrant ───────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"

    # ── Gemini ───────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── Kafka topics ─────────────────────────────────────────────────────
    KAFKA_QUESTION_TOPIC: str = "question-requests"
    KAFKA_RESPONSE_TOPIC: str = "question-responses"
    KAFKA_ERROR_TOPIC: str = "question-errors"

    # ── Search ───────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = 8
    SIMILARITY_THRESHOLD: float = 0.55   # tuned for e5-small cosine scores

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",   # silently ignore Phase 1 vars (HOST, PORT, PDF_STORAGE_PATH, …)
    }


settings = RagSettings()

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
    KAFKA_BOOTSTRAP_SERVERS: str = "127.0.0.1:9094"
    DEBUG: bool = False

    # ── Qdrant ───────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://127.0.0.1:6333"
 
    # ── LLM Settings ─────────────────────────────────────────────────────
    LLM_PROVIDER: str = "mistral"  # "gemini" | "openai" | "mistral"
    
    # Gemini
    GEMINI_API_KEY: str = "AIzaSyD0HaIMQ1kG7vSwOzLFI21B2aY7p-adwhg"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # Mistral
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"

    # OpenAI
    OPENAI_API_KEY: str = "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"
    OPENAI_MODEL: str = "gpt-4.1-mini"

    # ── Kafka topics ─────────────────────────────────────────────────────
    KAFKA_QUESTION_TOPIC: str = "question-requests"
    KAFKA_RESPONSE_TOPIC: str = "question-responses"
    KAFKA_ERROR_TOPIC: str = "question-errors"
    KAFKA_DELETE_TOPIC: str = "document-deletions"
    KAFKA_EDAG_BUILD_TOPIC: str = "edag.build.requested"
    KAFKA_EDAG_COMPLETED_TOPIC: str = "edag.build.completed"

    # ── Search ───────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = 30
    SIMILARITY_THRESHOLD: float = 0.55   # tuned for e5-small cosine scores

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",   # silently ignore Phase 1 vars (HOST, PORT, PDF_STORAGE_PATH, …)
    }


settings = RagSettings()

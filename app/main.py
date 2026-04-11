"""
RAG Parsing Server — FastAPI Application
=========================================
Phase 1: Intelligent PDF extraction with structured DB persistence.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# Ensure the project root is on the path so intelligent_extractor.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy debug logs from internal libraries
logging.getLogger("aiokafka").setLevel(logging.INFO)
logging.getLogger("watchfiles").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from app.workers.kafka_consumer import consume_document_events


# ============================================================
# Lifespan — startup / shutdown hooks
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup and once on shutdown.
    Guarantees storage directory exists before first request.
    """
    logger.info("=" * 60)
    logger.info("RAG Parsing Server starting up")
    logger.info(f"  DB     : {settings.DATABASE_URL[:50]}...")
    logger.info(f"  Storage: {settings.PDF_STORAGE_PATH}")
    logger.info(f"  Debug  : {settings.DEBUG}")
    logger.info(f"  Kafka  : {settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info("=" * 60)

    # Ensure storage directory exists
    Path(settings.PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

    # Start Kafka consumer background task
    consumer_task = asyncio.create_task(consume_document_events())

    yield  # ← server is live here

    logger.info("RAG Parsing Server shutting down")
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


# ============================================================
# App instance
# ============================================================

app = FastAPI(
    title="RAG Parsing Server",
    description=(
        "Production-grade PDF extraction API. Receives PDF documents, "
        "runs the 7-stage Intelligent Pipeline, and stores structured "
        "text blocks per user and subject. Phase 1 of the RAG pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ============================================================
# Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Routers
# ============================================================

from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.documents import router as documents_router

app.include_router(health_router)
app.include_router(documents_router, prefix="/api/v1")


# ============================================================
# Dev server entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

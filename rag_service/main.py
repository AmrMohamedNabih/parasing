"""
RAG Service — FastAPI Application (Phase 2)
============================================
Runs on port 8001. Phase 1 (parsing server, port 8000) is untouched.

Startup sequence:
  1. Load & warm up the embedding model (SentenceTransformer)
  2. Connect Qdrant client
  3. Configure Gemini SDK
  4. Start shared Kafka producer
  5. Launch Kafka consumer background task
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_service.config import settings
from rag_service.services.embedding_service import embedding_service
from rag_service.services.search_service    import search_service
from rag_service.services.generation_service import generation_service
from rag_service.workers.kafka_consumer      import (
    consume_question_events,
    start_producer,
    stop_producer,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_service.log", encoding="utf-8")
    ]
)
logging.getLogger("aiokafka").setLevel(logging.INFO)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("RAG Service (Phase 2) starting up")
    logger.info("  Qdrant : %s", settings.QDRANT_URL)
    logger.info("  Kafka  : %s", settings.KAFKA_BOOTSTRAP_SERVERS)
    logger.info("  Model  : gemini / %s", settings.GEMINI_MODEL)
    logger.info("  Threshold: %.2f  Top-K: %d", settings.SIMILARITY_THRESHOLD, settings.TOP_K_RESULTS)
    logger.info("=" * 60)

    # 1. Embedding model (CPU — runs in thread executor inside load())
    await embedding_service.load()

    # 2. Qdrant
    search_service.connect()

    # 3. Gemini
    generation_service.configure()

    # 4. Kafka producer
    await start_producer()

    # 5. Kafka consumer background task
    consumer_task = asyncio.create_task(consume_question_events())

    yield  # ← server is live

    logger.info("RAG Service shutting down…")
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    await stop_producer()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Query Service",
    description=(
        "Phase 2 of the RAG pipeline. Accepts questions via HTTP or Kafka, "
        "performs semantic search over embedded text blocks in Qdrant, "
        "and streams answers from Gemini via SSE and/or Kafka."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error("Validation error for %s %s", request.method, request.url)
    logger.error("Error detail: %s", exc.errors())
    logger.error("Raw body: %s", body.decode() if body else "EMPTY")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode() if body else None},
    )

from rag_service.api.routes import router
app.include_router(router)


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_service.main:app", host="0.0.0.0", port=8001, reload=settings.DEBUG)

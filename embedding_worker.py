"""
Embedding Worker
----------------
Standalone async worker that polls PostgreSQL every 30 seconds for text_blocks
rows where embedded_at IS NULL (and parent document status = 'done'), embeds
them with intfloat/multilingual-e5-small, upserts to Qdrant, then sets
embedded_at = now() on the processed rows.

Design decisions:
- Uses asyncpg directly (no SQLAlchemy) — the worker is fully standalone.
- Qdrant upsert uses deterministic UUID5 point IDs for idempotent re-runs.
- Ordering guarantee: Qdrant first → DB second. If DB update fails, blocks
  are simply re-embedded next cycle (safe because upsert is idempotent).
- CPU-bound encode() runs in a ThreadPoolExecutor to keep the event loop free.
"""

import asyncio
import logging
import os
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_db",
)
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = "rag_text_blocks"
VECTOR_DIM = 1024
POLL_INTERVAL = 30          # seconds between polls
FETCH_LIMIT = 64            # rows per DB fetch
ENCODE_BATCH_SIZE = 32      # SentenceTransformer internal batch size
MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embedding_worker")

# ── Thread pool for CPU-bound encoding ───────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed_worker")
_shutdown = False


def _db_url_to_asyncpg(url: str) -> str:
    """Strip SQLAlchemy driver prefix so asyncpg can use the URL directly."""
    return url.replace("postgresql+asyncpg://", "postgresql://")


# ── Worker class ─────────────────────────────────────────────────────────────

class EmbeddingWorker:
    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.qdrant: QdrantClient | None = None
        self.pool: asyncpg.Pool | None = None

    # ── Startup / shutdown ────────────────────────────────────────────────

    async def startup(self) -> None:
        logger.info("Loading model: %s", MODEL_NAME)
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(
            _executor, lambda: SentenceTransformer(MODEL_NAME)
        )
        logger.info("Model loaded.")

        logger.info("Connecting to Qdrant at %s", QDRANT_URL)
        self.qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
        await self._ensure_collection()

        logger.info("Connecting to PostgreSQL...")
        self.pool = await asyncpg.create_pool(
            dsn=_db_url_to_asyncpg(DATABASE_URL),
            min_size=2,
            max_size=5,
        )
        logger.info("PostgreSQL pool ready.")

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.pool:
            await self.pool.close()
        if self.qdrant:
            self.qdrant.close()

    async def _ensure_collection(self) -> None:
        loop = asyncio.get_running_loop()

        def _create() -> None:
            names = [c.name for c in self.qdrant.get_collections().collections]
            if COLLECTION_NAME not in names:
                logger.info("Creating Qdrant collection '%s'...", COLLECTION_NAME)
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_DIM, distance=Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
            else:
                logger.info("Qdrant collection '%s' already exists.", COLLECTION_NAME)

        await loop.run_in_executor(_executor, _create)

    # ── DB helpers ────────────────────────────────────────────────────────

    async def _fetch_batch(self) -> list[dict]:
        """
        Fetch up to FETCH_LIMIT unembedded blocks.
        Uses a correlated subquery for page_number to avoid row-multiplication
        from a JOIN on document_pages.
        """
        query = """
            SELECT
                tb.id,
                tb.text,
                tb.document_id,
                tb.user_id,
                tb.direction,
                tb.source_stage,
                tb.confidence,
                tb.word_count,
                tb.bbox,
                (
                    SELECT dp.page_number
                    FROM   document_pages dp
                    WHERE  dp.id = tb.page_id
                    LIMIT  1
                ) AS page_number,
                d.subject_id
            FROM  text_blocks  tb
            JOIN  documents    d  ON d.id = tb.document_id
            WHERE tb.embedded_at IS NULL
              AND d.status = 'done'
            LIMIT $1
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, FETCH_LIMIT)
        return [dict(r) for r in rows]

    async def _mark_embedded(self, ids: list[str]) -> None:
        uuid_ids = [uuid.UUID(i) for i in ids]
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE text_blocks SET embedded_at = now() WHERE id = ANY($1::uuid[])",
                uuid_ids,
            )

    # ── Encoding ──────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]):
        prefixed = texts # BGE-M3 does not need 'passage: ' prefix
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            partial(
                self.model.encode,
                prefixed,
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )

    # ── Qdrant helpers ───────────────────────────────────────────────────

    @staticmethod
    def _point_id(text_block_id: str) -> str:
        """Deterministic UUID5 so re-runs are idempotent upserts."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text_block_id))

    @staticmethod
    def _bbox_list(bbox) -> list:
        if isinstance(bbox, list):
            return bbox
        if isinstance(bbox, dict):
            return [bbox.get("x0", 0), bbox.get("y0", 0),
                    bbox.get("x1", 0), bbox.get("y1", 0)]
        return []

    async def _upsert(self, points: list[PointStruct]) -> None:
        """3-attempt exponential back-off on Qdrant failures."""
        loop = asyncio.get_running_loop()
        for attempt in range(3):
            try:
                await loop.run_in_executor(
                    _executor,
                    lambda: self.qdrant.upsert(
                        collection_name=COLLECTION_NAME, points=points
                    ),
                )
                return
            except (UnexpectedResponse, Exception) as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Qdrant upsert attempt %d failed: %s — retrying in %ds",
                    attempt + 1, exc, wait,
                )
                await asyncio.sleep(wait)

    # ── Batch processing ─────────────────────────────────────────────────

    async def process_batch(self, rows: list[dict]) -> int:
        if not rows:
            return 0

        embeddings = await self._embed([r["text"] for r in rows])

        points = [
            PointStruct(
                id=self._point_id(str(r["id"])),
                vector=vec.tolist(),
                payload={
                    "text_block_id": str(r["id"]),
                    "document_id":   str(r["document_id"]),
                    "user_id":       str(r["user_id"]),
                    "subject_id":    str(r["subject_id"]),
                    "page_number":   r["page_number"] or 0,
                    "direction":     r["direction"] or "ltr",
                    "source_stage":  r["source_stage"] or "unknown",
                    "confidence":    float(r["confidence"] or 0.0),
                    "word_count":    int(r["word_count"] or 0),
                    "bbox":          self._bbox_list(r["bbox"]),
                    "text":          r["text"],
                },
            )
            for r, vec in zip(rows, embeddings)
        ]

        # ── Qdrant first — DB second (atomic ordering) ───────────────────
        try:
            await self._upsert(points)
        except Exception as exc:
            logger.error(
                "Qdrant upsert failed for batch of %d blocks — will retry next cycle: %s",
                len(rows), exc,
            )
            return 0

        try:
            await self._mark_embedded([str(r["id"]) for r in rows])
        except Exception as exc:
            logger.error(
                "DB update failed after Qdrant upsert (%d blocks). "
                "Blocks will be safely re-upserted next cycle (idempotent). Error: %s",
                len(rows), exc,
            )

        # Log per-document progress
        doc_counts: dict[str, int] = {}
        for r in rows:
            key = str(r["document_id"])
            doc_counts[key] = doc_counts.get(key, 0) + 1
        for doc_id, n in doc_counts.items():
            logger.info("Embedded %d blocks for document %s", n, doc_id)

        return len(rows)

    # ── Main poll loop ────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("Polling every %ds for unembedded blocks…", POLL_INTERVAL)
        while not _shutdown:
            try:
                rows = await self._fetch_batch()
                if rows:
                    await self.process_batch(rows)
                else:
                    logger.debug("No unembedded blocks. Sleeping %ds.", POLL_INTERVAL)
            except Exception as exc:
                logger.exception("Unexpected error in polling loop: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    global _shutdown
    worker = EmbeddingWorker()

    # Graceful shutdown on SIGTERM / SIGINT (works on Linux/macOS/Windows)
    def _handle_signal(signum, frame):  # noqa: ARG001
        global _shutdown
        logger.info("Signal %s received — shutting down.", signum)
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    await worker.startup()
    try:
        await worker.run()
    finally:
        await worker.shutdown()
        logger.info("Embedding worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())

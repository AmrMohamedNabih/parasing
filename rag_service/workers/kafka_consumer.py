"""
Kafka Consumer — Question Processing Worker
--------------------------------------------
Consumes question-requests, runs semantic search + Gemini streaming,
and publishes chunks to question-responses (and/or in-memory SSE queues).

Key design decisions:
- group_id="rag-service", auto_offset_reset="latest" — skips historical
  messages on restart; appropriate for interactive Q&A.
- asyncio.Semaphore(5) caps concurrent generation tasks.
- Per-question asyncio.Queue is created BEFORE Kafka publish so the SSE
  endpoint never races against the consumer.
- Queues/state are auto-evicted after 5 minutes via call_later.
- Dead-letter: failed questions are published to KAFKA_ERROR_TOPIC.
- 120-second timeout on the full generation pipeline per question.
"""

import asyncio
import json
import logging
from collections import namedtuple
from typing import Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from rag_service.config import settings
from rag_service.schemas import (
    EnqueueResponse,
    QuestionChunk,
    QuestionRequest,
    QuestionStatusResponse,
    SourceBlock,
)

logger = logging.getLogger(__name__)

# ── Semaphore ─────────────────────────────────────────────────────────────────
_semaphore = asyncio.Semaphore(5)

# ── Shared state (in-memory, evicted after 5 min) ────────────────────────────
QueueMeta = namedtuple("QueueMeta", ["user_id", "stream_via"])

STREAM_QUEUES: dict[str, asyncio.Queue] = {}
STREAM_META:   dict[str, QueueMeta]     = {}


class _QuestionState:
    __slots__ = ("user_id", "status", "chunk_count", "is_final")

    def __init__(self, user_id: str) -> None:
        self.user_id    = user_id
        self.status     = "queued"
        self.chunk_count = 0
        self.is_final   = False


QUESTION_STATE: dict[str, _QuestionState] = {}

# ── Shared Kafka producer (singleton) ────────────────────────────────────────
_producer: Optional[AIOKafkaProducer] = None


async def start_producer() -> None:
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await _producer.start()
    logger.info("Kafka producer started.")


async def stop_producer() -> None:
    global _producer
    if _producer:
        await _producer.stop()
        logger.info("Kafka producer stopped.")


def get_producer() -> AIOKafkaProducer:
    return _producer


# ── State helpers ─────────────────────────────────────────────────────────────

def register_question(req: QuestionRequest) -> None:
    """
    Called BEFORE publishing to Kafka so the SSE endpoint never races.
    Queue + meta + state are created atomically here.
    """
    qid = req.question_id
    STREAM_QUEUES[qid] = asyncio.Queue()
    STREAM_META[qid]   = QueueMeta(user_id=req.user_id, stream_via=req.stream_via)
    QUESTION_STATE[qid] = _QuestionState(user_id=req.user_id)

    loop = asyncio.get_event_loop()
    loop.call_later(300, _evict, qid)


def _evict(question_id: str) -> None:
    STREAM_QUEUES.pop(question_id, None)
    STREAM_META.pop(question_id, None)
    QUESTION_STATE.pop(question_id, None)
    logger.debug("Evicted question state: %s", question_id)


# ── Publishing helpers ────────────────────────────────────────────────────────

async def _publish(question_id: str, chunk_index: int, chunk: str,
                   is_final: bool, sources: list[SourceBlock] | None = None) -> None:
    payload = {
        "question_id": question_id,
        "chunk_index": chunk_index,
        "chunk":       chunk,
        "is_final":    is_final,
        "sources":     [s.model_dump() for s in sources] if sources else [],
    }
    meta = STREAM_META.get(question_id)

    # Kafka publish
    if meta and meta.stream_via in ("kafka", "both"):
        try:
            await _producer.send(settings.KAFKA_RESPONSE_TOPIC, payload)
        except Exception as exc:
            logger.error("Kafka publish failed for %s: %s", question_id, exc)

    # SSE queue
    if meta and meta.stream_via in ("http", "both"):
        q = STREAM_QUEUES.get(question_id)
        if q:
            await q.put(payload)

    # State tracking
    state = QUESTION_STATE.get(question_id)
    if state:
        state.chunk_count += 1
        if is_final:
            state.is_final = True
            state.status   = "done"


async def _publish_error(question_id: str, message: str) -> None:
    await _publish(question_id, 0, message, is_final=True)
    state = QUESTION_STATE.get(question_id)
    if state:
        state.status = "error"


async def _publish_dlq(req: QuestionRequest, error: str) -> None:
    """Dead-letter: publish full payload to question-errors topic."""
    try:
        await _producer.send(
            settings.KAFKA_ERROR_TOPIC,
            {"question_id": req.question_id, "error": error, "original": req.model_dump()},
        )
        logger.warning("Published to DLQ: %s — %s", req.question_id, error)
    except Exception as exc:
        logger.error("DLQ publish also failed: %s", exc)


# ── Core question processor ───────────────────────────────────────────────────

async def _process_question(req: QuestionRequest) -> None:
    from rag_service.services.embedding_service import embedding_service
    from rag_service.services.search_service    import search_service
    from rag_service.services.generation_service import generation_service
    from rag_service.services.document_service import document_service
    from app.db.session import AsyncSessionFactory
    from app.db.models.question_history import QuestionHistory
    import uuid

    state = QUESTION_STATE.get(req.question_id)
    if state:
        state.status = "processing"

    async with _semaphore:
        try:
            await _run_pipeline(req, embedding_service, search_service, generation_service)
        except asyncio.TimeoutError:
            logger.error("Pipeline timed out for %s", req.question_id)
            await _publish_error(req.question_id, "Generation timed out after 120 s.")
            await _publish_dlq(req, "TimeoutError")
        except Exception as exc:
            logger.exception("Unhandled error for %s: %s", req.question_id, exc)
            await _publish_error(req.question_id, "An internal error occurred.")
            await _publish_dlq(req, str(exc))


async def _run_pipeline(req, embedding_service, search_service, generation_service) -> None:
    from rag_service.services.document_service import document_service
    # 1. Embed query
    query_vector = await embedding_service.encode_query(req.question)

    # 2. Context Retrieval
    is_rag_required = (req.document_ids and len(req.document_ids) > 0) or req.global_search
    scored_points = []
    majority_rtl = False

    try:
        if is_rag_required:
            if req.deep_analysis or req.task_plan or req.mindmap_mode or req.notebook_mode:
                logger.info("Holistic mode requested (deep_analysis=%s, task_plan=%s, mindmap=%s, notebook=%s) for %s. Fetching full text from DB.", 
                            req.deep_analysis, req.task_plan, req.mindmap_mode, req.notebook_mode, req.question_id)
                scored_points = await document_service.fetch_full_text(
                    document_ids=req.document_ids,
                    subject_id=req.subject_id
                )
                majority_rtl = sum(1 for p in scored_points if p.payload.get("direction") == "rtl") > len(scored_points) / 2 if scored_points else False
            else:
                scored_points, majority_rtl = await search_service.semantic_search(
                    query_vector=query_vector,
                    user_id=req.user_id,
                    subject_id=req.subject_id,
                    document_ids=req.document_ids,
                    top_k=req.top_k or settings.TOP_K_RESULTS,
                    use_edag=req.global_search,
                )
        else:
            logger.info("No documents or global search requested for %s. Skipping retrieval.", req.question_id)
    except Exception as exc:
        logger.error("Context retrieval failed for %s: %s", req.question_id, exc)
        await _publish_error(req.question_id, "Context retrieval service unavailable.")
        return

    # 3. No results above threshold
    if not scored_points:
        await _publish(
            req.question_id, 0,
            "I could not find relevant information in the documents.",
            is_final=True, sources=[],
        )
        return

    # 4. Stream generation (120 s hard timeout)
    chunk_index = 0
    sources = generation_service.build_sources(scored_points)
    full_answer_parts = []

    async def _generate() -> None:
        nonlocal chunk_index
        async for text_piece in generation_service.stream_answer(
            question=req.question,
            scored_points=scored_points,
            language=req.language,
            majority_rtl=majority_rtl,
            deep_analysis=req.deep_analysis,
            task_plan=req.task_plan
        ):
            full_answer_parts.append(text_piece)
            is_last = False   # interim chunks
            await _publish(req.question_id, chunk_index, text_piece, is_last)
            chunk_index += 1

        # Final sentinel chunk with sources
        await _publish(req.question_id, chunk_index, "", is_final=True, sources=sources)

    await asyncio.wait_for(_generate(), timeout=120.0)

    # 5. Save to database history
    try:
        from app.db.session import AsyncSessionFactory
        from app.db.models.question_history import QuestionHistory
        import uuid

        full_answer = "".join(full_answer_parts).strip()
        async with AsyncSessionFactory() as db:
            history = QuestionHistory(
                id=uuid.UUID(req.question_id),
                user_id=uuid.UUID(req.user_id),
                subject_id=uuid.UUID(req.subject_id) if req.subject_id else None,
                question_text=req.question,
                answer_text=full_answer,
                retrieved_chunk_ids=[uuid.UUID(s.text_block_id) for s in sources]
            )
            db.add(history)
            await db.commit()
            logger.info("Saved async question history for %s", req.question_id)
    except Exception as exc:
        logger.error("Failed to save async question history for %s: %s", req.question_id, exc)


# ── Consumer loop ─────────────────────────────────────────────────────────────

async def consume_question_events() -> None:
    logger.info("Starting Kafka consumer on '%s' and '%s'", settings.KAFKA_QUESTION_TOPIC, settings.KAFKA_EDAG_COMPLETED_TOPIC)

    consumer = AIOKafkaConsumer(
        settings.KAFKA_QUESTION_TOPIC,
        settings.KAFKA_EDAG_COMPLETED_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="rag-service",
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode()),
    )

    for attempt in range(5):
        try:
            await consumer.start()
            logger.info("Kafka consumer connected.")
            break
        except Exception as exc:
            logger.warning("Kafka connect attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(5)
    else:
        logger.error("Could not connect to Kafka after 5 attempts.")
        return

    try:
        async for msg in consumer:
            try:
                # Handle Question Requests
                if msg.topic == settings.KAFKA_QUESTION_TOPIC:
                    req = QuestionRequest(**msg.value)
                    # State is pre-registered so SSE never races
                    if req.question_id not in QUESTION_STATE:
                        register_question(req)
                    asyncio.create_task(_process_question(req))
                
                # Handle EDAG Build Completed
                elif msg.topic == settings.KAFKA_EDAG_COMPLETED_TOPIC:
                    subject_id = msg.value.get("subject_id")
                    status = msg.value.get("status")
                    if subject_id and status == "success":
                        from edag.edag_retriever import invalidate_edag_cache
                        invalidate_edag_cache(subject_id)
            except Exception as exc:
                logger.error("Could not parse Kafka message from topic %s: %s — %s", msg.topic, msg.value, exc)
    finally:
        await consumer.stop()

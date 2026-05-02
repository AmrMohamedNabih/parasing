"""
RAG Service — HTTP Routes
--------------------------
POST /questions          — enqueue a question (publishes to Kafka)
GET  /questions/{id}     — poll question status (SSE reconnect recovery)
GET  /stream/answer      — Server-Sent Events stream for a question
POST /questions/ask      — synchronous ask: returns full answer + sources as JSON
GET  /health             — liveness probe
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from typing import Optional

from rag_service.schemas import (
    EnqueueResponse,
    QuestionRequest,
    QuestionStatusResponse,
    SourceBlock,
    AskRequest,
    AskResponse,
)
from rag_service.workers.kafka_consumer import (
    QUESTION_STATE,
    STREAM_META,
    STREAM_QUEUES,
    get_producer,
    register_question,
)
from rag_service.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /questions ───────────────────────────────────────────────────────────

@router.post("/questions", response_model=EnqueueResponse, status_code=202)
async def enqueue_question(req: QuestionRequest) -> EnqueueResponse:
    """
    Accept a question, create the SSE queue (before Kafka publish to avoid
    race conditions), then publish to the question-requests topic.
    """
    if not req.question_id:
        req.question_id = str(uuid.uuid4())

    # Create queue FIRST — consumer may process before we return
    register_question(req)

    producer = get_producer()
    try:
        await producer.send(
            settings.KAFKA_QUESTION_TOPIC,
            req.model_dump(),
        )
    except Exception as exc:
        logger.error("Failed to publish question %s: %s", req.question_id, exc)
        raise HTTPException(status_code=503, detail="Message broker unavailable.")

    return EnqueueResponse(question_id=req.question_id, status="queued")


# ── GET /questions/{question_id} ─────────────────────────────────────────────

@router.get("/questions/{question_id}", response_model=QuestionStatusResponse)
async def get_question_status(
    question_id: str,
    user_id: str = Query(..., description="Must match the original requester"),
) -> QuestionStatusResponse:
    """
    Poll question state for SSE reconnect recovery.
    Returns chunk_count so clients know how many chunks to skip on reconnect.
    """
    state = QUESTION_STATE.get(question_id)
    if not state:
        raise HTTPException(status_code=404, detail="Question not found (may have expired).")
    if state.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden.")

    return QuestionStatusResponse(
        question_id=question_id,
        status=state.status,
        chunk_count=state.chunk_count,
        is_final=state.is_final,
    )


# ── GET /stream/answer ────────────────────────────────────────────────────────

@router.get("/stream/answer")
async def stream_answer(
    question_id: str = Query(...),
    user_id: str     = Query(..., description="Must match the original requester"),
):
    """
    SSE endpoint. Yields chunks until is_final=True or 5-minute timeout.

    Security: user_id is validated against the queue owner so one user
    cannot read another user's stream.

    Race condition: if the consumer hasn't started yet, waits up to 30 s
    for the first chunk before returning a timeout error.
    """
    meta = STREAM_META.get(question_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Question not found.")
    if meta.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden.")

    queue = STREAM_QUEUES.get(question_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Stream queue not available.")

    async def _event_generator():
        # Wait up to 30 s for the first chunk (handles slow consumer startup)
        try:
            first_payload = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield {
                "event": "chunk",
                "data": json.dumps({"error": "No response started within 30 s."}),
            }
            return

        yield {"event": "chunk", "data": json.dumps(first_payload)}
        if first_payload.get("is_final"):
            return

        # Stream remaining chunks with 5-minute overall timeout
        deadline = 300.0
        elapsed  = 0.0
        while elapsed < deadline:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                yield {"event": "chunk", "data": json.dumps(payload)}
                if payload.get("is_final"):
                    return
            except asyncio.TimeoutError:
                elapsed += 10.0

        yield {
            "event": "chunk",
            "data": json.dumps({"error": "Stream timed out after 5 minutes."}),
        }

    return EventSourceResponse(_event_generator())


# ── POST /questions/ask (synchronous — no Kafka, returns full JSON) ──────────



@router.post("/questions/ask", response_model=AskResponse)
async def ask_question(req: AskRequest) -> AskResponse:
    """
    Synchronous RAG endpoint. Embeds the question, searches Qdrant,
    generates a full answer via Gemini (collecting all streamed chunks),
    and returns the complete answer + source blocks as plain JSON.
    No Kafka, no SSE queues — works even when Kafka is down.
    """
    from rag_service.services.embedding_service import embedding_service
    from rag_service.services.search_service import search_service
    from rag_service.services.generation_service import generation_service
    from app.db.session import AsyncSessionFactory
    from app.db.models.question_history import QuestionHistory

    question_id = str(uuid.uuid4())

    # 1. Embed the question
    try:
        query_vector = await embedding_service.encode_query(req.question)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding service unavailable.")

    # 2. Semantic search in Qdrant
    try:
        scored_points, majority_rtl = await search_service.semantic_search(
            query_vector=query_vector,
            user_id=req.user_id,
            subject_id=req.subject_id,
            document_id=req.document_id,
            top_k=req.top_k or settings.TOP_K_RESULTS,
        )
    except Exception as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable.")

    # 3. No results above threshold
    if not scored_points:
        return AskResponse(
            question_id=question_id,
            answer="I could not find relevant information in the documents for your question.",
            sources=[],
        )

    # 4. Generate answer — collect all streamed chunks into one string
    language = req.language
    if majority_rtl:
        language = "ar"

    answer_parts: list[str] = []
    try:
        async def _collect():
            async for text_piece in generation_service.stream_answer(
                question=req.question,
                scored_points=scored_points,
                language=language,
                majority_rtl=majority_rtl,
            ):
                answer_parts.append(text_piece)

        await asyncio.wait_for(_collect(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error("Generation timed out for question %s", question_id)
        raise HTTPException(status_code=504, detail="Answer generation timed out. Please try again.")
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    full_answer = "".join(answer_parts).strip()

    # 5. Build source blocks
    sources = generation_service.build_sources(scored_points)

    logger.info("Synchronous ask complete for question %s (%d chars, %d sources)",
                question_id, len(full_answer), len(sources))

    # 6. Save to database history
    try:
        async with AsyncSessionFactory() as db:
            history = QuestionHistory(
                id=uuid.UUID(question_id),
                user_id=uuid.UUID(req.user_id),
                subject_id=uuid.UUID(req.subject_id) if req.subject_id else None,
                question_text=req.question,
                answer_text=full_answer,
                retrieved_chunk_ids=[uuid.UUID(s.text_block_id) for s in sources]
            )
            db.add(history)
            await db.commit()
            logger.info("Saved question history for %s", question_id)
    except Exception as exc:
        logger.error("Failed to save question history for %s: %s", question_id, exc)
        # We don't fail the request if history saving fails, but we log it.

    return AskResponse(
        question_id=question_id,
        answer=full_answer,
        sources=sources,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "rag-service"}

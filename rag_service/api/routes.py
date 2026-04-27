"""
RAG Service — HTTP Routes
--------------------------
POST /questions          — enqueue a question (publishes to Kafka)
GET  /questions/{id}     — poll question status (SSE reconnect recovery)
GET  /stream/answer      — Server-Sent Events stream for a question
GET  /health             — liveness probe
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from rag_service.schemas import (
    EnqueueResponse,
    QuestionRequest,
    QuestionStatusResponse,
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


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "rag-service"}

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
from pathlib import Path

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
from rag_service.services.document_service import document_service
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

    # 1. Log request
    logger.info("Incoming ask request: %s", req.model_dump())
    logger.info("Incoming ask request (summary present): %s", bool(req.summary))

    # 1. Embed the question
    try:
        query_vector = await embedding_service.encode_query(req.question)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding service unavailable.")

    # 2. Context Retrieval
    is_rag_required = (req.document_ids and len(req.document_ids) > 0) or req.global_search
    scored_points = []
    majority_rtl = False

    try:
        if is_rag_required:
            if req.deep_analysis or req.mindmap_mode or req.notebook_mode:
                # Use full text for specific deep analysis, mindmap mode, or notebook creation (holistic tasks)
                logger.info("Fetching full text from DB. deep_analysis=%s, mindmap_mode=%s, notebook_mode=%s", 
                            req.deep_analysis, req.mindmap_mode, req.notebook_mode)
                scored_points = await document_service.fetch_full_text(
                    document_ids=req.document_ids, 
                    subject_id=req.subject_id
                )
                majority_rtl = sum(1 for p in scored_points if p.payload.get("direction") == "rtl") > len(scored_points) / 2 if scored_points else False
            else:
                # Both normal search AND global search (Entire Subject) use semantic retrieval
                logger.info("Performing semantic search. global_search=%s", req.global_search)
                
                scored_points, majority_rtl = await search_service.semantic_search(
                    query_vector=query_vector,
                    user_id=req.user_id,
                    subject_id=req.subject_id,
                    # If global search, ignore specific document_ids to search the whole subject
                    document_ids=req.document_ids if not req.global_search else None,
                    top_k=req.top_k, # Now handled dynamically in search_service
                )
        else:
            logger.info("No documents or global search requested. Skipping retrieval.")
            # In this case, we proceed with an empty context (only history/summary)
    except Exception as exc:
        logger.error("Context retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable.")

    # 3. Handle No Results (Only if RAG was explicitly requested)
    if is_rag_required and not scored_points:
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
            async for chunk in generation_service.stream_answer(
                req.question,
                scored_points,
                req.language,
                majority_rtl,
                deep_analysis=req.deep_analysis,
                task_plan=req.task_plan,
                summary=req.summary,
                mindmap_mode=req.mindmap_mode,
                notebook_mode=req.notebook_mode
            ):
                answer_parts.append(chunk)

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


class SummarizeRequest(BaseModel):
    history_text: str


@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    logger.info("Summarize request received")
    from rag_service.services.generation_service import generation_service
    new_summary = await generation_service.generate_updated_summary(req.history_text)
    logger.info("Summarize result: %s", new_summary[:50] + "..." if new_summary else "NONE")
    return {"updatedSummary": new_summary}


class TestQueryRequest(BaseModel):
    subject_id: str
    question: str


@router.post("/api/edag/test-query")
async def test_edag_query(req: TestQueryRequest):
    """
    DEBUG ENDPOINT: Runs EDAG search for a question and returns
    only the retrieved leaf indices and their scores.
    """
    from rag_service.services.embedding_service import embedding_service
    from edag.edag_retriever import get_or_load_edag_retriever
    import numpy as np

    try:
        # 1. Embed
        q_vec = await embedding_service.encode_query(req.question)
        q_vec_np = np.array(q_vec, dtype=np.float32)

        # 2. Search
        retriever = get_or_load_edag_retriever(req.subject_id)
        results, trace = await retriever.search_async(q_vec_np, top_k=settings.TOP_K_RESULTS)

        # 3. Map results back to indices for the visualizer
        # We need to find the index of each leaf in the original list
        indices = []
        for res in results:
            # Match by chunk_id
            chunk_id = res.payload.get("text_block_id")
            for idx, meta in enumerate(retriever._leaf_meta):
                if meta.get("chunk_id") == chunk_id:
                    indices.append({"index": idx, "score": res.score})
                    break
        
        return {
            "retrieved_indices": indices,
            "trace": trace
        }
    except Exception as e:
        logger.error("EDAG test query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



# ── EDAG Visualization Endpoints ──────────────────────────────────────────────

@router.get("/api/edag/subjects")
async def list_edag_subjects():
    """Returns list of subjects that have a 'ready' EDAG graph."""
    from app.db.session import AsyncSessionFactory
    from app.db.models.subject import Subject
    from sqlalchemy import select

    async with AsyncSessionFactory() as db:
        result = await db.execute(
            select(Subject.id, Subject.name, Subject.edag_leaf_count, Subject.edag_built_at)
            .where(Subject.edag_status == "ready")
        )
        subjects = [
            {
                "id": str(row.id),
                "name": row.name,
                "leaf_count": row.edag_leaf_count,
                "built_at": row.edag_built_at.isoformat() if row.edag_built_at else None,
            }
            for row in result
        ]
        return subjects


@router.get("/api/edag/graph/{subject_id}")
async def get_edag_graph(subject_id: str):
    """Returns the graph.json for a specific subject."""
    # edag_graphs is in the parent directory of rag_service (the 'parsing' root)
    graph_path = Path(__file__).parent.parent.parent / "edag_graphs" / subject_id / "graph.json"
    
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="EDAG graph not found for this subject.")
    
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.error("Failed to read EDAG graph for %s: %s", subject_id, e)
        raise HTTPException(status_code=500, detail="Failed to load graph data.")


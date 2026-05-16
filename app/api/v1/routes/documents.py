"""
Documents Router — /api/v1/documents
--------------------------------------
Endpoints for uploading PDFs and querying extraction results.

All endpoints are user-scoped: user_id and subject_id are required.
Users and subjects are auto-created (upserted) on first upload.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.dependencies import get_db
from app.db.models.document import Document, DocumentPage, DocumentStatus, TextBlock
from app.db.models.subject import Subject
from app.db.models.user import User
from app.schemas.document import (
    DocumentListItem,
    DocumentPageResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    DocumentProcessingStatusItem,
    BatchProcessingStatusResponse,
)
from app.services.storage_service import storage_service
from app.workers.parsing_worker import run_parsing_job

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


# ============================================================
# GET /documents/batch-status  — Batch embedding/parsing status
# ============================================================

@router.get(
    "/batch-status",
    response_model=BatchProcessingStatusResponse,
    summary="Get detailed processing status for multiple documents",
    description=(
        "Returns the parsing + embedding phase for each requested document. "
        "Use this to know when a document is truly ready for RAG queries."
    ),
)
async def get_batch_processing_status(
    document_ids: str,   # comma-separated UUIDs
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BatchProcessingStatusResponse:
    """
    Accepts ?document_ids=uuid1,uuid2,... and returns a status item for each.

    Embedding progress is derived from counting text_blocks WHERE embedded_at IS NOT NULL
    vs total text_blocks for the document. This is the ground truth since the embedding
    worker only sets embedded_at after successfully pushing to Qdrant.
    """
    from sqlalchemy import func as sa_func

    # Parse and validate UUIDs — skip malformed ones gracefully
    raw_ids = [s.strip() for s in document_ids.split(",") if s.strip()]
    parsed_ids: list[uuid.UUID] = []
    for raw in raw_ids:
        try:
            parsed_ids.append(uuid.UUID(raw))
        except ValueError:
            logger.warning("batch-status: skipping invalid UUID '%s'", raw)

    if not parsed_ids:
        return BatchProcessingStatusResponse(statuses=[])

    # 1. Fetch all requested documents in one query
    docs_result = await db.execute(
        select(Document).where(Document.id.in_(parsed_ids))
    )
    docs = {doc.id: doc for doc in docs_result.scalars().all()}

    # 2. Count embedded blocks per document in one query
    embedded_counts_result = await db.execute(
        select(
            TextBlock.document_id,
            sa_func.count(TextBlock.id).label("embedded_count"),
        )
        .where(
            TextBlock.document_id.in_(parsed_ids),
            TextBlock.embedded_at.isnot(None),
        )
        .group_by(TextBlock.document_id)
    )
    embedded_counts: dict[uuid.UUID, int] = {
        row.document_id: row.embedded_count for row in embedded_counts_result
    }

    # 3. Build response items
    statuses: list[DocumentProcessingStatusItem] = []
    for doc_id in parsed_ids:
        doc = docs.get(doc_id)
        if doc is None:
            # Document not found in parsing DB — still pending upload
            statuses.append(DocumentProcessingStatusItem(
                document_id=doc_id,
                document_status="pending",
                embedding_phase="not_started",
                total_blocks=0,
                embedded_blocks=0,
                error_message=None,
                is_ready=False,
            ))
            continue

        total_blocks = doc.total_blocks or 0
        embedded = embedded_counts.get(doc_id, 0)
        doc_status = doc.status  # pending | processing | done | failed

        # Derive embedding phase
        if doc_status == DocumentStatus.FAILED.value:
            embedding_phase = "not_started"
            is_ready = False
        elif doc_status != DocumentStatus.DONE.value:
            # Still parsing — embedding hasn't started
            embedding_phase = "not_started"
            is_ready = False
        elif total_blocks == 0:
            # Parsed but no blocks yet (very unlikely edge case)
            embedding_phase = "in_progress"
            is_ready = False
        elif embedded >= total_blocks:
            embedding_phase = "complete"
            is_ready = True
        else:
            embedding_phase = "in_progress"
            is_ready = False

        statuses.append(DocumentProcessingStatusItem(
            document_id=doc_id,
            document_status=doc_status,
            embedding_phase=embedding_phase,
            total_blocks=total_blocks,
            embedded_blocks=embedded,
            error_message=doc.error_message,
            is_ready=is_ready,
        ))

    return BatchProcessingStatusResponse(statuses=statuses)

# The maximum PDF size accepted (100 MB)
MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024


# ============================================================
# Helper: Auto-upsert User + Subject
# ============================================================

async def _get_or_create_user(db: AsyncSession, user_id: uuid.UUID) -> User:
    """Return existing user or create and persist a new one."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(id=user_id)
        db.add(user)
        await db.flush()
        logger.info(f"Auto-created user {user_id}")
    return user


async def _get_or_create_subject(
    db: AsyncSession,
    subject_id: uuid.UUID,
    user_id: uuid.UUID,
    subject_name: str,
) -> Subject:
    """Return existing subject or create a new one under the user."""
    result = await db.execute(select(Subject).where(Subject.id == subject_id))
    subject = result.scalar_one_or_none()
    if subject is None:
        subject = Subject(id=subject_id, user_id=user_id, name=subject_name)
        db.add(subject)
        await db.flush()
        logger.info(f"Auto-created subject {subject_id} ('{subject_name}') for user {user_id}")
    return subject


# ============================================================
# POST /documents  — Upload and queue a PDF
# ============================================================

@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload a PDF for extraction",
    description=(
        "Accepts a PDF file and queues it for the 7-stage intelligent extraction pipeline. "
        "Returns immediately with a document_id. Poll GET /documents/{document_id} for status."
    ),
)
async def upload_document(
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="PDF file to parse"),
    user_id: uuid.UUID = Form(..., description="UUID of the owning user"),
    subject_id: uuid.UUID = Form(..., description="UUID of the subject"),
    subject_name: str = Form(default="default", description="Subject name (used on first creation)"),
    mode: str = Form(default="balanced", description="Pipeline mode: fast | balanced | thorough"),
    ocr_engine: str = Form(default="easyocr", description="OCR engine: easyocr | tesseract"),
) -> DocumentUploadResponse:

    # --- Validate file type ---
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # --- Validate mode and engine ---
    if mode not in ("fast", "balanced", "thorough"):
        raise HTTPException(status_code=400, detail="mode must be: fast | balanced | thorough")
    if ocr_engine not in ("easyocr", "tesseract"):
        raise HTTPException(status_code=400, detail="ocr_engine must be: easyocr | tesseract")

    # --- Read file bytes ---
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 100 MB limit")

    # --- Auto-upsert User and Subject ---
    await _get_or_create_user(db, user_id)
    await _get_or_create_subject(db, subject_id, user_id, subject_name)

    # --- Create Document record ---
    document_id = uuid.uuid4()
    document = Document(
        id=document_id,
        subject_id=subject_id,
        user_id=user_id,
        filename=file.filename,
        status=DocumentStatus.PENDING.value,
        pipeline_mode=mode,
        ocr_engine=ocr_engine,
        pdf_path="",  # Filled in after file is saved
    )
    db.add(document)
    await db.flush()  # Get the document.id before saving file

    # --- Persist PDF to storage ---
    pdf_path = await storage_service.save(
        file_bytes=file_bytes,
        user_id=user_id,
        subject_id=subject_id,
        document_id=document_id,
    )
    document.pdf_path = pdf_path
    await db.commit()

    logger.info(
        f"Document {document_id} accepted | user={user_id} subject={subject_id} "
        f"file='{file.filename}' mode={mode} engine={ocr_engine}"
    )

    # --- Enqueue background parsing job ---
    background_tasks.add_task(run_parsing_job, document_id)

    return DocumentUploadResponse(
        document_id=document_id,
        status=DocumentStatus.PENDING.value,
        user_id=user_id,
        subject_id=subject_id,
        filename=file.filename,
        message=f"PDF accepted. Processing started in '{mode}' mode with {ocr_engine.upper()}.",
    )


# ============================================================
# GET /documents/{document_id}  — Status & summary
# ============================================================

@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
)
async def get_document_status(
    document_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentStatusResponse:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse.model_validate(doc)


# ============================================================
# GET /documents/{document_id}/pages  — Full extraction result
# ============================================================

@router.get(
    "/{document_id}/pages",
    response_model=list[DocumentPageResponse],
    summary="Get all pages with extracted text blocks",
)
async def get_document_pages(
    document_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentPageResponse]:
    # Verify document exists and is done
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != DocumentStatus.DONE.value:
        raise HTTPException(
            status_code=409,
            detail=f"Document is not ready yet. Current status: '{doc.status}'",
        )

    # Load pages with blocks eagerly
    pages_result = await db.execute(
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .options(selectinload(DocumentPage.blocks))
        .order_by(DocumentPage.page_number)
    )
    pages = pages_result.scalars().all()
    return [DocumentPageResponse.model_validate(p) for p in pages]


# ============================================================
# GET /documents?user_id=&subject_id=  — List with query params
# ============================================================

@router.get(
    "",
    response_model=list[DocumentListItem],
    summary="List all documents for a user+subject",
    description="Returns documents for the given user_id and subject_id ordered by newest first.",
)
async def list_documents(
    user_id: uuid.UUID,
    subject_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentListItem]:
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id, Document.subject_id == subject_id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [DocumentListItem.model_validate(d) for d in docs]

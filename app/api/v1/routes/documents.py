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
)
from app.services.storage_service import storage_service
from app.workers.parsing_worker import run_parsing_job

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

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

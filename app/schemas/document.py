import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# Request Schemas
# ============================================================

class DocumentUploadRequest(BaseModel):
    """Query/form parameters accompanying the PDF file upload."""

    user_id: uuid.UUID = Field(..., description="Unique ID of the user who owns this document")
    subject_id: uuid.UUID = Field(
        ..., description="Unique ID of the subject this document belongs to"
    )
    subject_name: str = Field(
        default="default",
        max_length=255,
        description="Subject name — used to create the subject if it doesn't exist",
    )
    mode: Literal["fast", "balanced", "thorough"] = Field(
        default="balanced",
        description="Pipeline processing mode. 'fast' skips full-page OCR; 'thorough' enables grid OCR",
    )
    ocr_engine: Literal["easyocr", "tesseract"] = Field(
        default="easyocr",
        description="OCR engine. EasyOCR for better Arabic/English accuracy; Tesseract for speed",
    )


# ============================================================
# Response Schemas
# ============================================================

class DocumentUploadResponse(BaseModel):
    """Returned immediately after a PDF is accepted for processing."""

    document_id: uuid.UUID
    status: str
    user_id: uuid.UUID
    subject_id: uuid.UUID
    filename: str
    message: str


class TextBlockResponse(BaseModel):
    """A single extracted text block."""

    id: uuid.UUID
    block_id: str | None
    text: str
    direction: str
    rtl_ratio: float
    confidence: float
    source_stage: str | None
    bbox: list[float] | None
    column_num: int | None
    font: str | None
    font_size: float | None
    word_count: int

    model_config = {"from_attributes": True}


class DocumentPageResponse(BaseModel):
    """A single document page with its text blocks."""

    id: uuid.UUID
    page_number: int
    width: float | None
    height: float | None
    columns: int
    text_coverage: float | None
    confidence: float | None
    execution_time: float | None
    blocks: list[TextBlockResponse] = []

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """Document summary — status, stats, and pipeline config."""

    id: uuid.UUID
    user_id: uuid.UUID
    subject_id: uuid.UUID
    filename: str
    status: str
    total_pages: int | None
    total_blocks: int | None
    avg_confidence: float | None
    pipeline_mode: str
    ocr_engine: str
    error_message: str | None
    created_at: datetime
    processed_at: datetime | None

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    """Compact document entry for list views."""

    id: uuid.UUID
    filename: str
    status: str
    total_pages: int | None
    total_blocks: int | None
    avg_confidence: float | None
    created_at: datetime
    processed_at: datetime | None

    model_config = {"from_attributes": True}

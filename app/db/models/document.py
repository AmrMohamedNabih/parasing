import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models.subject import Subject
    from app.db.models.user import User

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


# ============================================================
# Enums
# ============================================================

class DocumentStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# ============================================================
# Document
# ============================================================

class Document(Base):
    """
    Represents a single uploaded PDF document belonging to a user and subject.
    The status field tracks the asynchronous parsing pipeline state.
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DocumentStatus.PENDING.value, index=True
    )
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_blocks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    pipeline_mode: Mapped[str] = mapped_column(String(50), nullable=False, default="balanced")
    ocr_engine: Mapped[str] = mapped_column(String(50), nullable=False, default="easyocr")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    subject: Mapped["Subject"] = relationship("Subject", back_populates="documents")  # noqa: F821
    user: Mapped["User"] = relationship("User", back_populates="documents")  # noqa: F821
    pages: Mapped[list["DocumentPage"]] = relationship(
        "DocumentPage", back_populates="document", cascade="all, delete-orphan",
        order_by="DocumentPage.page_number"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} file='{self.filename}' status='{self.status}'>"


# ============================================================
# DocumentPage
# ============================================================

class DocumentPage(Base):
    """
    Represents a single page extracted from a Document.
    Stores layout metadata: dimensions, column count, coverage, confidence.
    """
    __tablename__ = "document_pages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    columns: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    text_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="pages")
    blocks: Mapped[list["TextBlock"]] = relationship(
        "TextBlock", back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentPage doc_id={self.document_id} page={self.page_number}>"


# ============================================================
# TextBlock
# ============================================================

class TextBlock(Base):
    """
    A single extracted text block from a document page.

    Denormalized fields (document_id, user_id) allow fast RAG queries
    like "get all Arabic blocks for user X, subject Y" without multi-level joins.

    The `source_stage` field records which pipeline stage produced this block:
    direct | block_ocr | full_page_ocr | image_ocr | post_processed
    """
    __tablename__ = "text_blocks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalized for fast direct queries
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Pipeline metadata
    block_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_block_ids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Language / direction
    direction: Mapped[str] = mapped_column(String(10), nullable=False, default="ltr")
    rtl_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Quality
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Layout — stored as JSONB so it's queryable later for spatial RAG
    bbox: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    column_num: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Typography (from digital PDF layers)
    font: Mapped[str | None] = mapped_column(String(255), nullable=True)
    font_size: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Embedding tracking
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Relationships
    page: Mapped["DocumentPage"] = relationship("DocumentPage", back_populates="blocks")

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ") if self.text else ""
        return f"<TextBlock id={self.id} stage='{self.source_stage}' text='{preview}...'>"

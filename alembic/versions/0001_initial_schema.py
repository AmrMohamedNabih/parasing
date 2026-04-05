"""Initial schema — users, subjects, documents, document_pages, text_blocks

Revision ID: 0001
Revises:
Create Date: 2026-04-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # --------------------------------------------------------------- subjects
    op.create_table(
        "subjects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_subjects_user_id_name"),
    )
    op.create_index("ix_subjects_user_id", "subjects", ["user_id"])

    # --------------------------------------------------------------- documents
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("total_blocks", sa.Integer(), nullable=True),
        sa.Column("avg_confidence", sa.Float(), nullable=True),
        sa.Column(
            "pipeline_mode",
            sa.String(50),
            nullable=False,
            server_default=sa.text("'balanced'"),
        ),
        sa.Column(
            "ocr_engine",
            sa.String(50),
            nullable=False,
            server_default=sa.text("'easyocr'"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("pdf_path", sa.String(1000), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["subject_id"], ["subjects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_subject_id", "documents", ["subject_id"])
    op.create_index("ix_documents_status", "documents", ["status"])

    # --------------------------------------------------------- document_pages
    op.create_table(
        "document_pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
        sa.Column("columns", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("text_coverage", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("execution_time", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_document_pages_document_id", "document_pages", ["document_id"])

    # ------------------------------------------------------------ text_blocks
    op.create_table(
        "text_blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("block_id", sa.String(100), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "direction",
            sa.String(10),
            nullable=False,
            server_default=sa.text("'ltr'"),
        ),
        sa.Column(
            "rtl_ratio",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.0"),
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.0"),
        ),
        sa.Column("source_stage", sa.String(50), nullable=True),
        sa.Column("bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("column_num", sa.Integer(), nullable=True),
        sa.Column("font", sa.String(255), nullable=True),
        sa.Column("font_size", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["page_id"], ["document_pages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_text_blocks_page_id", "text_blocks", ["page_id"])
    op.create_index("ix_text_blocks_document_id", "text_blocks", ["document_id"])
    op.create_index("ix_text_blocks_user_id", "text_blocks", ["user_id"])


def downgrade() -> None:
    op.drop_table("text_blocks")
    op.drop_table("document_pages")
    op.drop_table("documents")
    op.drop_table("subjects")
    op.drop_table("users")

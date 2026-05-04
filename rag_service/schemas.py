"""
RAG Service — Pydantic Schemas
--------------------------------
Request / response models for the Kafka message bus and HTTP endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field
import uuid as _uuid


# ── Inbound ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Published to Kafka topic: question-requests"""
    question_id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    user_id: str
    subject_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    deep_analysis: bool = False
    question: str
    language: str = "auto"          # "ar" | "en" | "auto"
    top_k: int = 15
    task_plan: bool = False
    stream_via: str = "http"        # "kafka" | "http" | "both"

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


# ── Outbound ─────────────────────────────────────────────────────────────────

class SourceBlock(BaseModel):
    """One retrieved passage cited in the final answer."""
    text_block_id: str
    document_id: str
    page_number: int
    text_snippet: str               # first 120 chars
    score: float

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


class QuestionChunk(BaseModel):
    """One streaming chunk published to Kafka topic: question-responses"""
    question_id: str
    chunk_index: int
    chunk: str
    is_final: bool
    sources: list[SourceBlock] = []

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


# ── HTTP responses ────────────────────────────────────────────────────────────

class EnqueueResponse(BaseModel):
    question_id: str
    status: str = "queued"

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


class QuestionStatusResponse(BaseModel):
    question_id: str
    status: str             # "queued" | "processing" | "done" | "error"
    chunk_count: int
    is_final: bool

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


class AskRequest(BaseModel):
    user_id: str
    subject_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    deep_analysis: bool = False
    question: str
    language: str = "auto"
    top_k: int = 15
    task_plan: bool = False

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }


class AskResponse(BaseModel):
    question_id: str
    answer: str
    sources: list[SourceBlock]

    model_config = {
        "populate_by_name": True,
        "alias_generator": lambda s: "".join(
            word.capitalize() if i > 0 else word 
            for i, word in enumerate(s.split("_"))
        )
    }

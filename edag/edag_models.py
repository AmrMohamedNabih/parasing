"""
EDAG — Pydantic Schemas for Kafka Messages
-------------------------------------------
Used by both the builder (producer) and the RAG service (consumer).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EDAGBuildRequest(BaseModel):
    """Published to Kafka topic: edag.build.requested"""
    subject_id: str
    user_id: str
    triggered_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EDAGBuildCompleted(BaseModel):
    """Published to Kafka topic: edag.build.completed"""
    subject_id: str
    leaf_count: int = 0
    built_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str  # "success" | "skipped_insufficient_chunks" | "failed"
    error: Optional[str] = None

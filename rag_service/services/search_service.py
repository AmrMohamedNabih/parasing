"""
Search Service
--------------
Wraps the Qdrant client. semantic_search() builds a multi-tenant filter,
performs cosine similarity search, and detects Arabic-majority result sets
based purely on the retrieved passage payloads (not the question text).
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    ScoredPoint,
)

from rag_service.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    def connect(self) -> None:
        self._client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
        logger.info("Qdrant client connected: %s", settings.QDRANT_URL)

    async def semantic_search(
        self,
        query_vector: list[float],
        user_id: str,
        subject_id: Optional[str] = None,
        document_id: Optional[str] = None,
        top_k: int = 8,
    ) -> tuple[list[ScoredPoint], bool]:
        """
        Returns (scored_points, majority_rtl).

        majority_rtl — True when strictly >50 % of retrieved passages have
        direction='rtl' in their Qdrant payload. Used by generation_service
        to auto-switch the system prompt to Arabic.
        """
        must: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if subject_id:
            must.append(FieldCondition(key="subject_id", match=MatchValue(value=subject_id)))
        if document_id:
            must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))

        if not hasattr(self._client, "search"):
            logger.error("QdrantClient missing 'search' method. Available: %s", dir(self._client))
            # Try fallback if possible
            if hasattr(self._client, "query_points"):
                 results = self._client.query_points(
                    collection_name="rag_text_blocks",
                    query=query_vector,
                    query_filter=Filter(must=must),
                    limit=top_k,
                    score_threshold=settings.SIMILARITY_THRESHOLD,
                    with_payload=True,
                ).points
            else:
                raise AttributeError("QdrantClient has neither 'search' nor 'query_points'")
        else:
            results: list[ScoredPoint] = self._client.search(
                collection_name="rag_text_blocks",
                query_vector=query_vector,
                query_filter=Filter(must=must),
                limit=top_k,
                score_threshold=settings.SIMILARITY_THRESHOLD,
                with_payload=True,
            )

        # ── Arabic majority detection — from payload, never from question ──
        rtl_count = sum(
            1 for p in results if p.payload.get("direction") == "rtl"
        )
        majority_rtl = rtl_count > len(results) / 2 if results else False

        return results, majority_rtl


search_service = SearchService()

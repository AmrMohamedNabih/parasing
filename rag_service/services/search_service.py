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
        document_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> tuple[list[ScoredPoint], bool]:
        """
        Returns (scored_points, majority_rtl).

        majority_rtl — True when strictly >50 % of retrieved passages have
        direction='rtl' in their Qdrant payload. Used by generation_service
        to auto-switch the system prompt to Arabic.
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        logger.info("SEARCH PARAMS: top_k=%d, threshold=%f", top_k, settings.SIMILARITY_THRESHOLD)

        must: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if subject_id:
            must.append(FieldCondition(key="subject_id", match=MatchValue(value=subject_id)))
        if document_ids:
            from qdrant_client.models import MatchAny
            must.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids)))

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
                limit=40,  # Increase limit to allow dynamic filtering a wider range
                score_threshold=settings.SIMILARITY_THRESHOLD,
                with_payload=True,
            )

        logger.info("QDRANT RETURNED %d points", len(results))

        # ── Dynamic Similarity Filtering ──────────────────────────────────────
        if results:
            top_score = results[0].score
            # Only keep chunks within 0.07 of the best match, and above absolute threshold
            dynamic_threshold = max(settings.SIMILARITY_THRESHOLD, top_score - 0.07)
            
            original_count = len(results)
            results = [p for p in results if p.score >= dynamic_threshold]
            
            logger.info(
                "Dynamic Filtering: Kept %d/%d points (Top: %.4f, Dyn Threshold: %.4f)",
                len(results), original_count, top_score, dynamic_threshold
            )

        # ── Arabic majority detection — from payload, never from question ──
        rtl_count = sum(
            1 for p in results if p.payload.get("direction") == "rtl"
        )
        majority_rtl = rtl_count > len(results) / 2 if results else False

        # ── Resolve filenames from PostgreSQL ──
        await self._resolve_filenames(results)

        return results, majority_rtl

    async def _resolve_filenames(self, results: list[ScoredPoint]) -> None:
        doc_ids = list(set(p.payload.get("document_id") for p in results if p.payload.get("document_id")))
        if not doc_ids:
            return
            
        from sqlalchemy import select
        from app.db.session import AsyncSessionFactory
        from app.db.models.document import Document
        import uuid
        
        try:
            async with AsyncSessionFactory() as db:
                stmt = select(Document.id, Document.filename).where(Document.id.in_([uuid.UUID(did) for did in doc_ids]))
                res = await db.execute(stmt)
                mapping = {str(row[0]): row[1] for row in res.all()}
                
                for p in results:
                    did = p.payload.get("document_id")
                    if did in mapping:
                        p.payload["filename"] = mapping[did]
        except Exception as e:
            logger.error("Failed to resolve filenames: %s", e)

    async def delete_document(self, document_id: str) -> None:
        """
        Deletes all vectors associated with a specific document_id from Qdrant.
        """
        logger.info("Deleting vectors for document_id: %s", document_id)
        try:
            self._client.delete(
                collection_name="rag_text_blocks",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info("Successfully deleted vectors for document %s", document_id)
        except Exception as e:
            logger.error("Failed to delete vectors for document %s: %s", document_id, e)


search_service = SearchService()

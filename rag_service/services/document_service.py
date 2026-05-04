import logging
from typing import List
import uuid
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.session import AsyncSessionFactory
from app.db.models.document import TextBlock, DocumentPage

logger = logging.getLogger(__name__)

class DocumentService:
    async def fetch_full_text(self, document_ids: List[str]) -> List[any]:
        """
        Fetches all text blocks from PostgreSQL for the given document IDs.
        Returns a list of objects mimicking Qdrant's ScoredPoint structure.
        """
        if not document_ids:
            return []
            
        results = []
        try:
            # Convert string IDs to UUIDs
            doc_uuids = [uuid.UUID(did) for did in document_ids]
            
            async with AsyncSessionFactory() as db:
                # Query text blocks joining with pages and documents
                from app.db.models.document import Document
                stmt = (
                    select(TextBlock, Document.filename)
                    .join(DocumentPage, TextBlock.page_id == DocumentPage.id)
                    .join(Document, TextBlock.document_id == Document.id)
                    .where(TextBlock.document_id.in_(doc_uuids))
                    .options(selectinload(TextBlock.page))
                    .order_by(DocumentPage.page_number, TextBlock.id)
                )
                
                res = await db.execute(stmt)
                rows = res.all() # Each row is (TextBlock, filename)
                
                logger.info("Fetched %d blocks from PostgreSQL for documents %s", len(rows), document_ids)
                
                for b, fname in rows:
                    # Mock ScoredPoint structure for compatibility with generation_service
                    # We use a simple class or namedtuple that supports .payload and .score
                    class MockScoredPoint:
                        def __init__(self, payload, score):
                            self.payload = payload
                            self.score = score
                            
                    results.append(MockScoredPoint(
                        payload={
                            "text": b.text,
                            "page_number": b.page.page_number if b.page else 0,
                            "document_id": str(b.document_id),
                            "filename": fname,
                            "direction": b.direction,
                            "text_block_id": str(b.id)
                        },
                        score=1.0 # Max score for direct extraction
                    ))
            return results
        except Exception as e:
            logger.error("Failed to fetch full text from DB: %s", e)
            return []

document_service = DocumentService()

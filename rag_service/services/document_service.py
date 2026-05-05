import logging
from typing import List
import uuid
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.session import AsyncSessionFactory
from app.db.models.document import TextBlock, DocumentPage

logger = logging.getLogger(__name__)

class DocumentService:
    async def fetch_full_text(self, document_ids: List[str] = None, subject_id: str = None) -> List[any]:
        """
        Fetches all text blocks from PostgreSQL for given document IDs or an entire subject.
        Returns a list of objects mimicking Qdrant's ScoredPoint structure.
        """
        if not document_ids and not subject_id:
            return []
            
        results = []
        try:
            async with AsyncSessionFactory() as db:
                from app.db.models.document import Document
                stmt = (
                    select(TextBlock, Document.filename)
                    .join(DocumentPage, TextBlock.page_id == DocumentPage.id)
                    .join(Document, TextBlock.document_id == Document.id)
                )

                if document_ids:
                    doc_uuids = [uuid.UUID(did) for did in document_ids]
                    stmt = stmt.where(TextBlock.document_id.in_(doc_uuids))
                elif subject_id:
                    stmt = stmt.where(Document.subject_id == uuid.UUID(subject_id))

                stmt = stmt.options(selectinload(TextBlock.page)).order_by(Document.id, DocumentPage.page_number, TextBlock.id)
                
                res = await db.execute(stmt)
                rows = res.all()
                
                logger.info("Fetched %d blocks from PostgreSQL for %s", len(rows), f"documents {document_ids}" if document_ids else f"subject {subject_id}")
                
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

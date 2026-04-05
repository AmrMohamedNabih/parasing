"""
Parsing Worker
--------------
Background task runner. Each task creates its own DB session so it is
completely decoupled from the HTTP request lifecycle.

When you add Kafka in the future, replace this with a consumer that calls
`run_parsing_job(document_id)` for each message.
"""

import logging
import uuid

from app.db.session import AsyncSessionFactory
from app.services.parsing_service import ParsingService

logger = logging.getLogger(__name__)


async def run_parsing_job(document_id: uuid.UUID) -> None:
    """
    Entry point for background document processing.

    Creates a dedicated DB session (independent of the HTTP request),
    delegates to ParsingService, and guarantees the session is closed
    even if an exception occurs.

    Kafka-ready: a consumer can call this function for each received message.
    """
    logger.info(f"Worker started for document_id={document_id}")
    service = ParsingService()

    async with AsyncSessionFactory() as db:
        try:
            await service.process_document(document_id=document_id, db=db)
        except Exception as e:
            # ParsingService already handles status=FAILED, but we log here too
            logger.exception(f"Unhandled error in parsing worker for {document_id}: {e}")
        finally:
            await db.close()

    logger.info(f"Worker finished for document_id={document_id}")

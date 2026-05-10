"""
Parsing Service
---------------
Orchestrates the 7-stage IntelligentPDFExtractor pipeline and persists
all results to the database.

Design decisions:
- The extractor runs synchronously in a ThreadPoolExecutor to avoid
  blocking the async event loop (OCR is CPU-bound).
- This service is the single chokepoint between the extraction engine
  and the DB — it can be called from an HTTP handler, a BackgroundTask,
  or a Kafka consumer without any changes.
"""

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document, DocumentPage, DocumentStatus, TextBlock
from app.db.models.subject import Subject

# Import the core 7-stage pipeline from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from intelligent_extractor import IntelligentPDFExtractor
from pipeline_models import DocumentResult

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound OCR work
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr_worker")


class ParsingService:
    """
    Orchestrates PDF extraction and DB persistence.

    Usage:
        service = ParsingService()
        await service.process_document(document_id, db_session)
    """

    async def process_document(
        self,
        document_id: uuid.UUID,
        db: AsyncSession,
    ) -> None:
        """
        Full pipeline: load document → extract → persist all pages & blocks.
        Updates document.status throughout.
        """
        # 1. Load document record
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()

        if doc is None:
            logger.error(f"Document {document_id} not found in DB — aborting")
            return

        # 2. Mark as PROCESSING
        doc.status = DocumentStatus.PROCESSING.value
        await db.commit()
        logger.info(f"[{document_id}] Status → PROCESSING | file={doc.filename}")

        try:
            # 3. Run the 7-stage pipeline in a thread pool (non-blocking)
            extraction_result: DocumentResult = await self._run_pipeline(
                pdf_path=doc.pdf_path,
                mode=doc.pipeline_mode,
                ocr_engine=doc.ocr_engine,
            )

            # 3.5 Semantic chunking and merging
            final_chunks_by_page = await self._chunk_and_merge(extraction_result)

            # 4. Persist all pages and blocks
            await self._persist_results(
                db=db,
                doc=doc,
                result=extraction_result,
                final_chunks_by_page=final_chunks_by_page
            )

            # 5. Mark as DONE
            doc.status = DocumentStatus.DONE.value
            doc.processed_at = datetime.now(timezone.utc)
            doc.total_pages = extraction_result.total_pages
            doc.total_blocks = extraction_result.total_blocks
            doc.avg_confidence = round(extraction_result.avg_confidence, 4)
            await db.commit()
            logger.info(
                f"[{document_id}] Status → DONE | "
                f"pages={doc.total_pages} blocks={doc.total_blocks} "
                f"confidence={doc.avg_confidence:.2f}"
            )

            # 6. Fire EDAG build trigger (fire-and-forget, never raises)
            await maybe_trigger_edag_build(
                subject_id=str(doc.subject_id),
                user_id=str(doc.user_id),
                db=db,
            )

        except Exception as e:
            logger.exception(f"[{document_id}] Pipeline failed: {e}")
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = str(e)[:2000]  # Truncate for DB column
            await db.commit()

    async def _run_pipeline(
        self,
        pdf_path: str,
        mode: str,
        ocr_engine: str,
    ) -> DocumentResult:
        """
        Run IntelligentPDFExtractor in a thread pool to keep the event loop free.
        """
        loop = asyncio.get_running_loop()

        def _extract() -> DocumentResult:
            extractor = IntelligentPDFExtractor(
                lang="ara+eng",
                mode=mode,
                dpi=300,
                ocr_engine=ocr_engine,
            )
            return extractor.extract_from_pdf(
                pdf_path=pdf_path,
                output_json_path=None,  # DB is the output — no file dumps
                extract_images=False,   # Phase 1 — image OCR text is captured, images not saved to disk
                images_dir=None,
            )

        return await loop.run_in_executor(_executor, _extract)

    async def _chunk_and_merge(self, result: DocumentResult) -> dict:
        """
        Run semantic chunking and neural merging in a thread pool.
        """
        loop = asyncio.get_running_loop()

        def _do_chunk() -> dict:
            from semantic_chunker import SemanticChunker
            from chunk_merger import neural_chunk_merger
            from pipeline_models import ChunkingConfig
            
            config = ChunkingConfig()
            chunker = SemanticChunker(config)
            
            final_chunks = {}
            for page in result.pages:
                semantic_chunks = chunker.chunk_page(page.page_number, page.blocks)
                merged_chunks = neural_chunk_merger.merge_chunks(semantic_chunks, config)
                final_chunks[page.page_number] = merged_chunks
            return final_chunks

        return await loop.run_in_executor(_executor, _do_chunk)

    async def _persist_results(
        self,
        db: AsyncSession,
        doc: Document,
        result: DocumentResult,
        final_chunks_by_page: dict,
    ) -> None:
        """
        Bulk-insert all pages and their text blocks in a single transaction.
        """
        for page_data in result.pages:
            # Create page record
            page = DocumentPage(
                id=uuid.uuid4(),
                document_id=doc.id,
                page_number=page_data.page_number,
                width=page_data.width,
                height=page_data.height,
                columns=page_data.columns,
                text_coverage=round(page_data.text_coverage, 4),
                confidence=round(page_data.total_confidence, 4),
                execution_time=round(page_data.total_execution_time, 4),
            )
            db.add(page)
            # Flush to get page.id for FK references
            await db.flush()

            # Create all text blocks for this page from final merged chunks
            block_records = []
            final_chunks = final_chunks_by_page.get(page_data.page_number, [])
            for chunk in final_chunks:
                tb = TextBlock(
                    id=uuid.uuid4(),
                    page_id=page.id,
                    document_id=doc.id,
                    user_id=doc.user_id,
                    block_id=chunk.chunk_id,
                    text=chunk.text.replace("\x00", ""), # Fix: Strip null bytes for PG compatibility
                    direction=chunk.direction.value,
                    rtl_ratio=0.0,
                    confidence=round(chunk.confidence, 4),
                    source_stage="semantic_merged",
                    bbox=chunk.bbox,
                    column_num=1,
                    font=None,
                    font_size=None,
                    word_count=chunk.token_count, # Storing token count instead of word count
                    source_block_ids=chunk.source_block_ids,
                )
                block_records.append(tb)

            # Bulk add for efficiency
            db.add_all(block_records)

        # Single commit for the whole document
        await db.commit()
        logger.debug(f"Persisted {result.total_pages} pages, {result.total_blocks} blocks")


# ── EDAG Build Trigger ────────────────────────────────────────────────────────

MIN_CHUNKS_FOR_EDAG = 100


async def maybe_trigger_edag_build(
    subject_id: str,
    user_id: str,
    db: AsyncSession,
) -> None:
    """
    After a document finishes processing, check if the subject has crossed
    MIN_CHUNKS_FOR_EDAG. If so and no build is in progress, publish a
    'edag.build.requested' Kafka message and mark edag_status='building'.

    Uses a one-shot aiokafka producer (start → send → stop).
    Wrapped in try/except so failures never affect the main pipeline.
    """
    import os
    try:
        from sqlalchemy import func, update
        from app.db.models.document import TextBlock

        # Count total embedded (or pending) chunks for this subject
        chunk_count_result = await db.execute(
            select(func.count(TextBlock.id)).where(
                TextBlock.page_id.in_(
                    select(TextBlock.page_id).join(
                        Document, Document.id == TextBlock.document_id
                    ).where(Document.subject_id == uuid.UUID(subject_id))
                )
            )
        )
        # Simpler: count text_blocks where document belongs to subject
        chunk_count_result = await db.execute(
            select(func.count(TextBlock.id))
            .join(Document, Document.id == TextBlock.document_id)
            .where(Document.subject_id == uuid.UUID(subject_id))
        )
        chunk_count = chunk_count_result.scalar() or 0

        logger.info(
            "[EDAG] Subject %s has %d total chunks (threshold=%d)",
            subject_id, chunk_count, MIN_CHUNKS_FOR_EDAG
        )

        if chunk_count < MIN_CHUNKS_FOR_EDAG:
            return

        # Check current edag_status — skip if already building or ready
        subject_result = await db.execute(
            select(Subject).where(Subject.id == uuid.UUID(subject_id))
        )
        subject = subject_result.scalar_one_or_none()
        if subject is None:
            return

        if subject.edag_status in ("building", "ready"):
            logger.info(
                "[EDAG] Subject %s edag_status='%s' — skipping trigger",
                subject_id, subject.edag_status
            )
            return

        # Mark as building
        subject.edag_status = "building"
        await db.commit()
        logger.info("[EDAG] Subject %s edag_status → 'building'", subject_id)

        # Fire Kafka message via one-shot producer
        kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9094")
        try:
            from aiokafka import AIOKafkaProducer
            producer = AIOKafkaProducer(
                bootstrap_servers=kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            await producer.start()
            try:
                await producer.send(
                    "edag.build.requested",
                    {
                        "subject_id": subject_id,
                        "user_id": user_id,
                        "triggered_at": datetime.utcnow().isoformat(),
                    },
                )
                logger.info(
                    "[EDAG] Published edag.build.requested for subject=%s", subject_id
                )
            finally:
                await producer.stop()
        except Exception as kafka_exc:
            logger.warning(
                "[EDAG] Kafka publish failed for subject=%s: %s — "
                "status will remain 'building' and retry on next document.",
                subject_id, kafka_exc
            )

    except Exception as exc:
        logger.warning(
            "[EDAG] maybe_trigger_edag_build failed for subject=%s: %s",
            subject_id, exc
        )


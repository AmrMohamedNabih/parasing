import asyncio
import json
import logging
import uuid

from aiokafka import AIOKafkaConsumer
from sqlalchemy import select

from app.core.config import settings
from app.db.models.document import Document, DocumentStatus
from app.db.models.subject import Subject
from app.db.models.user import User
from app.db.session import AsyncSessionFactory
from app.services.storage_service import storage_service
from app.workers.parsing_worker import run_parsing_job

logger = logging.getLogger(__name__)


async def _get_or_create_user(db, user_id: uuid.UUID) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(id=user_id)
        db.add(user)
        await db.flush()
        logger.info(f"Kafka consumer auto-created user {user_id}")
    return user


async def _get_or_create_subject(
    db, subject_id: uuid.UUID, user_id: uuid.UUID, subject_name: str
) -> Subject:
    result = await db.execute(select(Subject).where(Subject.id == subject_id))
    subject = result.scalar_one_or_none()
    if subject is None:
        subject = Subject(id=subject_id, user_id=user_id, name=subject_name)
        db.add(subject)
        await db.flush()
        logger.info(f"Kafka consumer auto-created subject {subject_id}")
    return subject


async def consume_document_events():
    """Background task to consume Kafka events."""
    logger.info(f"Starting Kafka consumer for topic 'document-uploads' on {settings.KAFKA_BOOTSTRAP_SERVERS}")

    consumer = AIOKafkaConsumer(
        'document-uploads',
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    for _ in range(5):
        try:
            await consumer.start()
            logger.info("Kafka consumer connected successfully.")
            break
        except Exception as e:
            logger.warning(f"Kafka consumer start failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
    else:
        logger.error("Failed to start Kafka consumer after 5 retries. Exiting consumer loop.")
        return

    try:
        async for msg in consumer:
            event = msg.value
            logger.info(f"Consumed event: {event}")

            try:
                document_id = uuid.UUID(event['documentId'])
                subject_id = uuid.UUID(event['subjectId'])
                user_id = uuid.UUID(event['userId'])
                filename = event.get('fileName', 'unknown.pdf')
                mode = event.get('pipelineMode', settings.DEFAULT_PIPELINE_MODE)
                engine = event.get('ocrEngine', settings.DEFAULT_OCR_ENGINE)

                async with AsyncSessionFactory() as db:
                    await _get_or_create_user(db, user_id)
                    await _get_or_create_subject(db, subject_id, user_id, "default")

                    pdf_path = storage_service.resolve_path(user_id, subject_id, document_id)

                    doc_result = await db.execute(select(Document).where(Document.id == document_id))
                    doc = doc_result.scalar_one_or_none()
                    if not doc:
                        doc = Document(
                            id=document_id,
                            subject_id=subject_id,
                            user_id=user_id,
                            filename=filename,
                            status=DocumentStatus.PENDING.value,
                            pipeline_mode=mode,
                            ocr_engine=engine,
                            pdf_path=pdf_path,
                        )
                        db.add(doc)
                        await db.commit()

                # Dispatch background task for the PDF extraction
                asyncio.create_task(run_parsing_job(document_id))
            except Exception as e:
                logger.error(f"Error processing message {event}: {e}", exc_info=True)

    finally:
        await consumer.stop()

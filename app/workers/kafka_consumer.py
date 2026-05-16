import asyncio
import json
import logging
import uuid

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from sqlalchemy import select, delete

from app.core.config import settings
from app.db.models.document import Document, DocumentStatus
from app.db.models.subject import Subject
from app.db.models.user import User
from app.db.session import AsyncSessionFactory
from app.services.storage_service import storage_service
from app.workers.parsing_worker import run_parsing_job
from rag_service.services.search_service import search_service

logger = logging.getLogger(__name__)


async def _get_or_create_user(db, user_id: uuid.UUID) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(id=user_id)
        db.add(user)
        try:
            await db.flush()
            logger.info(f"Kafka consumer auto-created user {user_id}")
        except Exception:
            await db.rollback()
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                raise
    return user


async def _get_or_create_subject(
    db, subject_id: uuid.UUID, user_id: uuid.UUID, subject_name: str
) -> Subject:
    """
    Ensures a subject exists with the given ID.
    If the ID is missing but the name is taken, we rename the existing subject
    to free up the name for the new ID (which is the source of truth from Java).
    """
    # 1. Try finding by ID first — if found, we are done
    result = await db.execute(select(Subject).where(Subject.id == subject_id))
    subject = result.scalar_one_or_none()
    if subject:
        return subject

    # 2. ID not found. Check if the name is taken by another ID for this user.
    result = await db.execute(
        select(Subject).where(
            Subject.user_id == user_id,
            Subject.name == subject_name
        )
    )
    conflicting_subject = result.scalar_one_or_none()

    if conflicting_subject:
        # Resolve conflict: Rename the old one to free up the name
        logger.warning(
            f"Subject name conflict for '{subject_name}' (user {user_id}). "
            f"Existing ID {conflicting_subject.id} != New ID {subject_id}. "
            f"Renaming old subject to free name."
        )
        conflicting_subject.name = f"{subject_name}_old_{str(conflicting_subject.id)[:8]}"
        await db.flush()

    # 3. Create the new subject with the correct ID
    subject = Subject(id=subject_id, user_id=user_id, name=subject_name)
    db.add(subject)
    try:
        await db.flush()
        logger.info(f"Kafka consumer created subject {subject_id} ('{subject_name}')")
    except Exception as e:
        await db.rollback()
        # Race condition check
        result = await db.execute(select(Subject).where(Subject.id == subject_id))
        subject = result.scalar_one_or_none()
        if not subject:
            raise e
    return subject


async def _handle_document_deletion(event: dict, producer: AIOKafkaProducer):
    """
    Cleans up all traces of a document from PostgreSQL, Qdrant, and local storage.
    """
    try:
        document_id = uuid.UUID(event['documentId'])
        subject_id = uuid.UUID(event['subjectId'])
        user_id = uuid.UUID(event['userId'])

        logger.info(f"Starting global cleanup for document {document_id}")

        # 1. Delete from PostgreSQL (Cascades to text_blocks and pages)
        async with AsyncSessionFactory() as db:
            await db.execute(delete(Document).where(Document.id == document_id))
            await db.commit()
            logger.info(f"Deleted document {document_id} from PostgreSQL")

        # 2. Delete from Local Storage (PDF file)
        await storage_service.delete(user_id, subject_id, document_id)
        logger.info(f"Deleted document {document_id} from local storage")

        # 3. Delete from Qdrant Vector DB
        # Ensure search_service is connected (it's a singleton)
        search_service.connect()
        await search_service.delete_document(str(document_id))

        logger.info(f"Global cleanup complete for document {document_id}")

        # 4. Trigger EDAG rebuild/cleanup
        try:
            await producer.send(
                settings.KAFKA_EDAG_BUILD_TOPIC,
                {"subject_id": str(subject_id), "user_id": str(user_id)}
            )
            logger.info(f"Triggered EDAG rebuild for subject {subject_id} after deletion")
        except Exception as pe:
            logger.error(f"Failed to trigger EDAG rebuild for subject {subject_id}: {pe}")

    except Exception as e:
        logger.error(f"Failed to handle document deletion for {event}: {e}", exc_info=True)


async def consume_document_events():
    """Background task to consume Kafka events."""
    logger.info(f"Starting Kafka consumer for topic 'document-uploads' on {settings.KAFKA_BOOTSTRAP_SERVERS}")

    consumer = AIOKafkaConsumer(
        settings.KAFKA_DELETE_TOPIC,
        'document-uploads',
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id='parsing-worker',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    for _ in range(5):
        try:
            await consumer.start()
            await producer.start()
            logger.info("Kafka consumer and producer connected successfully.")
            break
        except Exception as e:
            logger.warning(f"Kafka start failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
    else:
        logger.error("Failed to start Kafka components after 5 retries. Exiting.")
        return

    try:
        async for msg in consumer:
            event = msg.value
            logger.info(f"Consumed event: {event}")

            try:
                if msg.topic == settings.KAFKA_DELETE_TOPIC:
                    await _handle_document_deletion(event, producer)
                    continue

                document_id = uuid.UUID(event['documentId'])
                subject_id = uuid.UUID(event['subjectId'])
                user_id = uuid.UUID(event['userId'])
                filename = event.get('fileName', 'unknown.pdf')
                subject_name = event.get('subjectName', 'default')
                mode = event.get('pipelineMode', settings.DEFAULT_PIPELINE_MODE)
                engine = event.get('ocrEngine', settings.DEFAULT_OCR_ENGINE)

                async with AsyncSessionFactory() as db:
                    await _get_or_create_user(db, user_id)
                    subject = await _get_or_create_subject(db, subject_id, user_id, subject_name)

                    pdf_path = storage_service.resolve_path(user_id, subject_id, document_id)

                    doc_result = await db.execute(select(Document).where(Document.id == document_id))
                    doc = doc_result.scalar_one_or_none()
                    if not doc:
                        doc = Document(
                            id=document_id,
                            subject_id=subject.id,
                            user_id=user_id,
                            filename=filename,
                            status=DocumentStatus.PENDING.value,
                            pipeline_mode=mode,
                            ocr_engine=engine,
                            pdf_path=pdf_path,
                        )
                        db.add(doc)
                    else:
                        # Update existing document in case metadata or path changed
                        doc.subject_id = subject.id
                        doc.filename = filename
                        doc.pipeline_mode = mode
                        doc.ocr_engine = engine
                        doc.pdf_path = pdf_path
                        doc.status = DocumentStatus.PENDING.value
                        doc.error_message = None

                    await db.commit()

                # Dispatch background task for the PDF extraction
                asyncio.create_task(run_parsing_job(document_id))
            except Exception as e:
                logger.error(f"Error processing message {event}: {e}", exc_info=True)

    finally:
        await consumer.stop()
        await producer.stop()

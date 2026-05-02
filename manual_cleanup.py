import asyncio
import uuid
from app.db.session import AsyncSessionFactory
from sqlalchemy import delete
from app.db.models.document import Document
from app.services.storage_service import storage_service
from rag_service.services.search_service import search_service
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

async def manual_cleanup(doc_id_str):
    doc_id = uuid.UUID(doc_id_str)
    print(f"--- Manual Cleanup for Document: {doc_id} ---")
    
    async with AsyncSessionFactory() as db:
        # 1. Delete from PostgreSQL
        await db.execute(delete(Document).where(Document.id == doc_id))
        await db.commit()
        print("CLEANUP: Deleted from PostgreSQL (rag_db)")
    
    # 2. Delete from Qdrant
    search_service.connect()
    await search_service.delete_document(str(doc_id))
    print("CLEANUP: Deleted from Qdrant")
    
    print("--- Cleanup Finished ---")

if __name__ == "__main__":
    doc_id = "04519e43-3819-49f2-a985-789cfc846a9d"
    asyncio.run(manual_cleanup(doc_id))

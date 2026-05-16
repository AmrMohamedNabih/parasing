import asyncio
from sqlalchemy import select, func, update
from app.db.session import AsyncSessionFactory
from app.db.models.document import Document, TextBlock

async def main():
    async with AsyncSessionFactory() as db:
        # Get count of text blocks per document
        result = await db.execute(
            select(
                TextBlock.document_id,
                func.count(TextBlock.id).label("block_count")
            ).group_by(TextBlock.document_id)
        )
        counts = result.all()
        
        for row in counts:
            doc_id = row.document_id
            block_count = row.block_count
            
            # Get the document
            doc_result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = doc_result.scalar_one_or_none()
            
            if doc and doc.total_blocks != block_count:
                print(f"Updating doc {doc_id} from {doc.total_blocks} to {block_count}")
                doc.total_blocks = block_count
                await db.commit()

asyncio.run(main())

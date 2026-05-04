import asyncio
import uuid
from sqlalchemy import select
from app.db.session import AsyncSessionFactory
from app.db.models.document import TextBlock, DocumentPage

# Data from the log
question_id = "ad58fc07-d782-47c3-9f98-8a3905404f40"
question_text = "which one ibram or amr was work in kamel ?"
retrieved_ids = [
    "8e9f4eed-5597-4a36-9bbd-bafd9d91d9f8",
    "35bda7a8-b0b8-4091-8327-e3b52d1f65bf",
    "6c521079-f541-472c-9dd1-e16a77d4810e",
    "001c6f4b-0131-4191-840a-f0fbbf034c49",
    "875b09a1-0fde-4e75-bf01-662286176e02",
    "5f86ae80-baf6-460f-a591-6a3fcb353069",
    "44ad07a8-8894-42c5-b609-0db2e8f01d0f"
]

async def reconstruct():
    async with AsyncSessionFactory() as db:
        stmt = (
            select(TextBlock)
            .join(DocumentPage, TextBlock.page_id == DocumentPage.id)
            .where(TextBlock.id.in_([uuid.UUID(rid) for rid in retrieved_ids]))
            .order_by(DocumentPage.page_number, TextBlock.id)
        )
        res = await db.execute(stmt)
        blocks = res.scalars().all()
        
        print(f"\n--- RECONSTRUCTED PROMPT FOR {question_id} ---\n")
        print("SYSTEM PROMPT (Deep Analysis=True):")
        print("You are a highly analytical research assistant. Your task is to provide a deep, comprehensive analysis and summary of the provided context passages. Focus on key findings, methodologies, conclusions, and any significant takeaways. Structure your response logically with headers and bullet points where appropriate. Base your answer ONLY on the provided context. Respond in English.")
        
        print("\nUSER PROMPT:")
        print("Context passages:\n")
        for i, b in enumerate(blocks, 1):
            print(f"[PASSAGE {i}] (doc {b.document_id}, page {b.page_id}, score 1.00)")
            print(b.text)
            print()
        
        print(f"---\nQuestion: {question_text}")
        print("\n(Note: This is a deep analysis request. Please provide a thorough breakdown.)\n\nAnswer:")

if __name__ == "__main__":
    asyncio.run(reconstruct())

import asyncio
import uuid
from sqlalchemy import select
from app.db.session import AsyncSessionFactory
from app.db.models.subject import Subject

async def check_db():
    async with AsyncSessionFactory() as db:
        user_id = uuid.UUID('3609e2a2-e4b5-4629-8e0b-d5745c806bca')
        result = await db.execute(select(Subject).where(Subject.user_id == user_id))
        subjects = result.scalars().all()
        print(f"Subjects for user {user_id}:")
        for s in subjects:
            print(f"  ID: {s.id}, Name: {s.name}")

if __name__ == "__main__":
    asyncio.run(check_db())

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionFactory


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that provides an async DB session per request.
    The session is automatically closed after the request completes.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

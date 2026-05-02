import asyncio
import asyncpg
import os

async def check_tables():
    dsn = "postgresql://rag_user:rag_password@localhost:5433/rag_db"
    conn = await asyncpg.connect(dsn)
    try:
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        print("Tables in rag_db:")
        for t in tables:
            print(f"- {t['table_name']}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_tables())

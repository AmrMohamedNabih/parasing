import asyncio
import asyncpg

async def check():
    try:
        conn = await asyncpg.connect('postgresql://rag_user:rag_password@localhost:5433/rag_db')
        rows = await conn.fetch('SELECT id, left(text, 50) as snippet FROM text_blocks LIMIT 10')
        for r in rows:
            print(f"ID: {r['id']} | Text: {r['snippet']}")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check())

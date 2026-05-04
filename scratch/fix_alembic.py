import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://rag_user:rag_password@localhost:5433/rag_db')
    await conn.execute("UPDATE alembic_version SET version_num = '427a05b3b3c9'")
    await conn.close()
    print("Fixed alembic version")

if __name__ == '__main__':
    asyncio.run(main())

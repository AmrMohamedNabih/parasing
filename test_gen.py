import asyncio
import logging
from rag_service.config import settings
from rag_service.services.generation_service import generation_service

logging.basicConfig(level=logging.INFO)

async def main():
    generation_service.configure()
    res = await generation_service.generate_updated_summary("User: Hello\nAI: Hi")
    print(f"Result: {res}")

if __name__ == '__main__':
    asyncio.run(main())

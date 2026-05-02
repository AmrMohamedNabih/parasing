import asyncio
from aiokafka import AIOKafkaConsumer

async def list_topics():
    print("--- Listing Kafka Topics ---")
    consumer = AIOKafkaConsumer(
        bootstrap_servers='localhost:9094',
        request_timeout_ms=5000
    )
    await consumer.start()
    try:
        topics = await consumer.topics()
        print(f"Topics found: {topics}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(list_topics())

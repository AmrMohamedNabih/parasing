import asyncio
import json
import os
from aiokafka import AIOKafkaProducer

async def trigger_build(subject_id: str):
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9094")
    producer = AIOKafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await producer.start()
    try:
        payload = {
            "subject_id": subject_id,
            "user_id": "system",
            "triggered_at": "manual"
        }
        await producer.send("edag.build.requested", payload)
        print(f"Sent build request for {subject_id}")
    finally:
        await producer.stop()

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "935b0097-c8ce-4f30-ba11-c7db6ace1504"
    asyncio.run(trigger_build(sid))

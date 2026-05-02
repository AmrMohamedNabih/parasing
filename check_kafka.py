import asyncio
import json
from aiokafka import AIOKafkaConsumer

async def check_kafka_topic(topic):
    print(f"--- Checking Kafka Topic: {topic} ---")
    try:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers='localhost:9094',
            auto_offset_reset='earliest',
            consumer_timeout_ms=5000,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        await consumer.start()
        try:
            print("Waiting for messages...")
            async for msg in consumer:
                print(f"Message at offset {msg.offset}: {msg.value}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await consumer.stop()
    except Exception as e:
        print(f"Error: {e}")
    print("--- Done ---")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else 'document-uploads'
    asyncio.run(check_kafka_topic(topic))

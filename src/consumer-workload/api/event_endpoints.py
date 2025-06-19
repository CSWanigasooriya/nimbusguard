import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from kafka import KafkaProducer, KafkaConsumer, errors
from kafka.admin import KafkaAdminClient, NewTopic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/events", tags=["events"])

executor = ThreadPoolExecutor(max_workers=2)

# Background consumer management
background_consumers = {}  # topic -> {"thread": thread, "consumer": consumer_instance, "running": bool}


class BackgroundEventConsumer:
    def __init__(self, topic: str, group_id: str, bootstrap_servers: str = 'kafka:9092'):
        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.running = False
        self.consumer = None
        self.thread = None

    def start(self):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.thread.start()
        print(f"[BACKGROUND CONSUMER] Started background consumer for topic {self.topic} with group {self.group_id}")
        logger.info(f"Started background consumer for topic {self.topic} with group {self.group_id}")
        return True

    def stop(self):
        if not self.running:
            return False
        print(f"[BACKGROUND CONSUMER] Stopping consumer for topic {self.topic}")
        self.running = False
        if self.consumer:
            self.consumer.close()
        if self.thread:
            self.thread.join(timeout=5)
        print(f"[BACKGROUND CONSUMER] Stopped consumer for topic {self.topic}")
        logger.info(f"Stopped background consumer for topic {self.topic}")
        return True

    def _consume_loop(self):
        print(f"[BACKGROUND CONSUMER] Starting consume loop for topic {self.topic}")
        while self.running:
            try:
                print(f"[BACKGROUND CONSUMER] Creating KafkaConsumer for topic {self.topic}")
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=5000  # 5 second timeout for polling
                )

                print(f"[BACKGROUND CONSUMER] Connected to topic {self.topic}, waiting for messages...")
                logger.info(f"Background consumer connected to topic {self.topic}")

                for message in self.consumer:
                    if not self.running:
                        print(f"[BACKGROUND CONSUMER] Stopping - running flag is False")
                        break

                    try:
                        event_data = json.loads(message.value.decode('utf-8'))
                        print(f"[BACKGROUND CONSUMER] Processing message: {event_data}")
                        self._process_event(event_data, message)
                    except Exception as e:
                        print(f"[BACKGROUND CONSUMER] Error processing message: {e}")
                        logger.error(f"Error processing message: {e}")

                print(f"[BACKGROUND CONSUMER] Closing consumer for topic {self.topic}")
                if self.consumer:
                    self.consumer.close()

            except Exception as e:
                print(f"[BACKGROUND CONSUMER] Error in consume loop: {e}")
                logger.error(f"Background consumer error: {e}")
                if self.running:
                    print(f"[BACKGROUND CONSUMER] Retrying in 5 seconds...")
                    logger.info("Retrying background consumer in 5 seconds...")
                    time.sleep(5)

    def _process_event(self, event_data: dict, message):
        """Process consumed event - override this for custom processing"""
        print(
            f"[BACKGROUND CONSUMER] ✅ PROCESSED EVENT: {event_data} from partition {message.partition}, offset {message.offset}")
        logger.info(f"Background consumer processed event: {event_data}", extra={
            "topic": self.topic,
            "partition": message.partition,
            "offset": message.offset,
            "event_type": event_data.get("event_type"),
            "service": event_data.get("service"),
            "value": event_data.get("value")
        })


class EventTriggerRequest(BaseModel):
    event_type: str
    service: str
    value: float


class ClearTopicRequest(BaseModel):
    topic: str = "scaling-events"  # Default to scaling-events for backward compatibility


class ConsumerTriggerRequest(BaseModel):
    topic: str = "scaling-events"
    group_id: str = "background-consumer"
    auto_restart: bool = True


@router.post("/produce")
async def produce_event(request: EventTriggerRequest):
    try:
        producer = KafkaProducer(
            bootstrap_servers='kafka:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        producer.send('scaling-events', request.dict())
        producer.flush()
        logger.info("Produced event to Kafka", extra=request.dict())
        return {"status": "event produced", **request.dict()}
    except Exception as e:
        logger.error(f"Failed to produce event: {e}")
        raise HTTPException(status_code=500, detail="Failed to produce event")


def fetch_kafka_messages(max_messages, timeout, group_id='background-consumer'):
    messages = []
    try:
        # Use the same group_id as background consumer to see only unconsumed messages
        consumer = KafkaConsumer(
            'scaling-events',
            bootstrap_servers='kafka:9092',
            group_id=group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=False  # Don't commit to avoid affecting background consumer
        )
        end_time = time.time() + timeout
        while len(messages) < max_messages and time.time() < end_time:
            records = consumer.poll(timeout_ms=500)
            for tp, msgs in records.items():
                for message in msgs:
                    messages.append(json.loads(message.value.decode('utf-8')))
                    if len(messages) >= max_messages:
                        break
        consumer.close()
    except Exception as e:
        logger.error(f"Failed to consume events: {e}")
    return messages


@router.get("/")
async def get_events(max_messages: int = 10, timeout: int = 5, group_id: str = 'background-consumer'):
    """Get events that haven't been consumed by the specified consumer group"""
    loop = asyncio.get_event_loop()
    messages = await loop.run_in_executor(
        executor, fetch_kafka_messages, max_messages, timeout, group_id
    )
    logger.info(f"Consumed {len(messages)} unconsumed events from Kafka for group {group_id}")
    return {"messages": messages, "group_id": group_id}


@router.post("/clear")
async def clear_topic(request: ClearTopicRequest):
    try:
        admin = KafkaAdminClient(bootstrap_servers='kafka:9092')
        topic = request.topic

        # Check if topic exists first
        existing_topics = list(admin.list_topics())

        if topic in existing_topics:
            # Delete the topic
            delete_result = admin.delete_topics([topic])

            # Wait for deletion to complete with retries
            max_retries = 30
            for i in range(max_retries):
                import time;
                time.sleep(2)
                try:
                    current_topics = list(admin.list_topics())
                    if topic not in current_topics:
                        logger.info(f"Topic {topic} successfully deleted after {i + 1} retries")
                        break
                except Exception as e:
                    logger.warning(f"Error checking topics (retry {i + 1}): {e}")
                if i == max_retries - 1:
                    logger.warning(f"Topic {topic} still exists after {max_retries} retries, proceeding anyway")
                    # Don't raise exception, just log warning and continue

        # Recreate the topic (only if it doesn't exist)
        try:
            current_topics = list(admin.list_topics())
            if topic not in current_topics:
                admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
                logger.info(f"Topic {topic} successfully recreated")
            else:
                logger.info(f"Topic {topic} already exists, skipping recreation")
        except Exception as create_error:
            logger.warning(f"Error during topic recreation: {create_error}")

        admin.close()
        logger.info(f"Cleared all messages from topic {topic}")
        return {"status": "cleared", "topic": topic}
    except Exception as e:
        logger.error(f"Failed to clear topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear topic: {str(e)}")


@router.post("/trigger-consumer")
async def trigger_background_consumer(request: ConsumerTriggerRequest):
    """Start a background consumer for continuous event processing"""
    consumer_key = f"{request.topic}:{request.group_id}"

    if consumer_key in background_consumers and background_consumers[consumer_key]["running"]:
        return {
            "status": "already_running",
            "topic": request.topic,
            "group_id": request.group_id,
            "message": "Background consumer is already running for this topic and group"
        }

    try:
        consumer = BackgroundEventConsumer(
            topic=request.topic,
            group_id=request.group_id
        )

        if consumer.start():
            background_consumers[consumer_key] = {
                "consumer": consumer,
                "running": True,
                "topic": request.topic,
                "group_id": request.group_id
            }

            logger.info(f"Triggered background consumer for topic {request.topic}")
            return {
                "status": "started",
                "topic": request.topic,
                "group_id": request.group_id,
                "consumer_key": consumer_key
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start background consumer")

    except Exception as e:
        logger.error(f"Failed to trigger background consumer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger background consumer: {str(e)}")


@router.post("/stop-consumer")
async def stop_background_consumer(request: ConsumerTriggerRequest):
    """Stop a background consumer"""
    consumer_key = f"{request.topic}:{request.group_id}"

    if consumer_key not in background_consumers:
        raise HTTPException(status_code=404,
                            detail=f"No background consumer found for topic {request.topic} and group {request.group_id}")

    try:
        consumer_info = background_consumers[consumer_key]
        consumer = consumer_info["consumer"]

        if consumer.stop():
            background_consumers[consumer_key]["running"] = False
            del background_consumers[consumer_key]

            logger.info(f"Stopped background consumer for topic {request.topic}")
            return {
                "status": "stopped",
                "topic": request.topic,
                "group_id": request.group_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to stop background consumer")

    except Exception as e:
        logger.error(f"Failed to stop background consumer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop background consumer: {str(e)}")


@router.get("/consumers/status")
async def get_consumers_status():
    """Get status of all background consumers"""
    consumers_status = []

    for consumer_key, consumer_info in background_consumers.items():
        consumers_status.append({
            "consumer_key": consumer_key,
            "topic": consumer_info["topic"],
            "group_id": consumer_info["group_id"],
            "running": consumer_info["running"]
        })

    return {
        "total_consumers": len(background_consumers),
        "consumers": consumers_status
    }

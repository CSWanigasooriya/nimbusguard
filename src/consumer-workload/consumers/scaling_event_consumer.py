import threading
import logging
import time
from kafka import KafkaConsumer, errors

logger = logging.getLogger(__name__)

class ScalingEventConsumer:
    def __init__(self, topic='scaling-events', bootstrap_servers='kafka:9092', group_id='nimbusguard-consumer'):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._thread = None
        self._running = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        logger.info('Started Kafka consumer thread', extra={'topic': self.topic})

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None

    def _consume(self):
        logger.info('Kafka consumer thread started and polling for messages', extra={'topic': self.topic})
        while self._running:
            try:
                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset='earliest',
                    enable_auto_commit=True
                )
                logger.info('Kafka consumer connected', extra={'topic': self.topic})
                for message in consumer:
                    logger.info('Received Kafka event', extra={'event': message.value.decode('utf-8')})
                    if not self._running:
                        break
                consumer.close()
            except errors.NoBrokersAvailable:
                logger.warning('No Kafka brokers available, retrying in 5 seconds...')
                time.sleep(5) 
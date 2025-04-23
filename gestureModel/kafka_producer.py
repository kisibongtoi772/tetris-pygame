from kafka import KafkaProducer
import json
import time
import logging

class TetrisKafkaProducer:
    """A reusable Kafka producer for Tetris game commands"""
    
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='tetris-commands'):
        """Initialize the Kafka producer with specified servers and topic"""
        self.topic = topic
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.connected = True
            self.logger.info(f"Connected to Kafka at {bootstrap_servers}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Kafka: {e}")
            print(f"Failed to connect to Kafka: {e}")
    
    def send_command(self, command, additional_data=None):
        """Send a command to the specified Kafka topic"""
        if not self.connected:
            self.logger.warning("Not connected to Kafka. Command not sent.")
            return False
            
        message = {
            'command': command,
            'timestamp': time.time()
        }
        
        # Add any additional data if provided
        if additional_data and isinstance(additional_data, dict):
            message.update(additional_data)
            
        try:
            self.producer.send(self.topic, message)
            self.producer.flush()
            self.logger.debug(f"Sent command: {command}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            return False
    
    def close(self):
        """Close the Kafka producer connection"""
        if self.connected:
            self.producer.close()
            self.connected = False
            self.logger.info("Kafka producer connection closed")
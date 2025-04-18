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

# Example usage as a standalone script
def main():
    """Interactive command-line interface"""
    producer = TetrisKafkaProducer()
    if not producer.connected:
        print("Failed to initialize Kafka producer. Exiting.")
        return
        
    print("Tetris Kafka Command Sender")
    print("Available commands: left, right, down, up (rotate), quit")
    print("Type 'exit' to quit this program")
    
    try:
        while True:
            command = input("> ").strip().lower()
            if command == "exit":
                break
                
            if command in ["left", "right", "down", "up", "quit"]:
                success = producer.send_command(command)
                if success:
                    print(f"Sent command: {command}")
                else:
                    print("Failed to send command")
            else:
                print("Unknown command")
    finally:
        producer.close()

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
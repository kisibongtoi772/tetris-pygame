import voice_model
import torch
import time
from queue import Queue
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.kafka_producer import TetrisKafkaProducer
import signal
class VoiceRecognizer:
    def __init__(self, kafka_producer=None):
        self.kakfa_producer = kafka_producer or TetrisKafkaProducer.get_instance()
        self.command_labels = ["rotation_left", "rotation_right", "move_left", "move_right", "down", "yes", "no", "pause", "speed"]
        self.model = None
        self.running = True
        self.model_loaded = False
        
    def load_model(self):
        print("Loading voice recognition model...")
        self.model = voice_model.VoiceCommandRecognizer()
        self.model.load_state_dict(torch.load("voice_model.pth", map_location="cpu"))
        self.model_loaded = True
        print("Voice model loaded successfully!")
        
    def predict_command(self):
        if not self.model_loaded:
            self.load_model()
        try:
            # Record audio
            voice_model.record_audio("test.wav", duration=2)

            # Predict command
            predicted_index = voice_model.predict_command(self.model, "test.wav")
            predicted_command = self.command_labels[predicted_index]

            print(f"\n🧠 Predicted Command: **{predicted_command}**")
            return predicted_command
        except Exception as e:
            print(f"Error predicting voice command: {e}")
            return None
    
    def voice_listener_loop(self):
        # Load the model only once at startup
        self.load_model()
        
        print("Voice recognition started - listening for commands...")
        
        while self.running:
            try:
                command = self.predict_command()
                if command:

                    self.kakfa_producer.send_command(command)
                    print(f"Voice command added to queue: {command}")
                # Add a small delay to prevent CPU overuse
                time.sleep(0.5)
            except Exception as e:
                print(f"[Voice Thread] Error: {e}")
                # Add some delay after an error to avoid rapid error loops
                time.sleep(1)

def main():
    """
    Main function that initializes the voice recognizer and starts processing voice commands,
    sending them to the Kafka queue.
    """
    print("Starting Voice Command Producer for Tetris")
    
    # Create voice recognizer
    kafka_producer = TetrisKafkaProducer.get_instance()
    voice_recognizer = VoiceRecognizer(kafka_producer)
    
    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down voice command producer...")
        voice_recognizer.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start the voice recognition loop
        print(f"Voice recognition active - listening for commands...")
        print(f"Press Ctrl+C to exit")
        voice_recognizer.voice_listener_loop()
    except Exception as e:
        print(f"Error in voice command producer: {e}")
        return 1
    finally:
        print("Voice command producer shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
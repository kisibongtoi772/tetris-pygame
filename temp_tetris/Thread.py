import threading
from HelloWorld import TetrisGame
import pyaudio
from VoiceCommandRec import VoiceCommandRecognizer
import torch
import wave
from LoadSpect import load_spectrogram
from KafkaProd import send_voice_command


def listen_for_voice_commands():
    # Setup audio recording
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    frames_per_buffer=1024,
                    input=True)

    # Load the trained model
    model = VoiceCommandRecognizer()
    model.load_state_dict(torch.load('voice_command_recognizer.pth'))
    model.eval()

    # Command mapping
    commands = {0: "move_left", 1: "move_right", 2: "rotate", 3: "drop"}

    print("Voice recognition activated! Speak commands...")

    # Continuously listen for commands
    while True:
        try:
            # Record a short audio snippet
            frames = []
            for i in range(0, int(44100 / 1024 * 1)):  # 1 second of audio
                data = stream.read(1024)
                frames.append(data)

            # Save temporary file
            temp_file = "temp_command.wav"
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Process through model
            spectrogram = load_spectrogram(temp_file)
            input_tensor = torch.from_numpy(spectrogram)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = output.argmax(dim=1).item()
                command = commands[predicted_class]

            # Send to Kafka
            send_voice_command(command)
            print(f"Recognized and sent command: {command}")

        except Exception as e:
            print(f"Error in voice processing: {e}")

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()



tetris_game = TetrisGame()
game_thread = threading.Thread(target=tetris_game.run)
game_thread.start()

voice_thread = threading.Thread(target=listen_for_voice_commands)
voice_thread.start()


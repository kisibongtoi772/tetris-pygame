import pyaudio
import wave
import time

def record_audio(output_file):
    chunk = 1024  # Number of frames per buffer
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44.1 kHz
    seconds = 2  # Duration of recording

    p = pyaudio.PyAudio()  # Create an instance of PyAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize empty list to store frames


    # Record audio in chunks and append to frames list
    print(f"Recording: {output_file}... Speak now!")
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print(f"Finished recording: {output_file}")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PyAudio instance
    p.terminate()

    # Save the recorded audio to a file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Loop to record multiple files
commands = ["rotation_left","rotation_right"]
num_files_per_command = 4 # Number of recordings per command

for command in commands:
    for i in range (1,num_files_per_command+1):
        filename = f"{command}_{i}.wav"
        record_audio(filename)
        time.sleep(2)



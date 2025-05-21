import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pyaudio
import wave
import numpy as np
import os
import noisereduce as nr
from scipy.io import wavfile

# -----------------------------
# 1. Define your model class
# -----------------------------
class VoiceCommandRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically calculate fc input size
        sample_input = torch.randn(1, 1, 128, 87)
        with torch.no_grad():
            sample_output = self.forward_feature_extractor(sample_input)
        self.fc_input_size = sample_output.numel()

        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward_feature_extractor(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x

    def forward(self, x):
        x = self.forward_feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# -----------------------------
# 2. Function to record audio
# -----------------------------
def record_audio_without(output_file="test.wav", duration=2):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording... Speak now!")
    frames = []
    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def record_audio(output_file="test.wav", duration=2):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100  # Sampling rate

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording... Speak now!")
    frames = []

    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save raw audio to temp file first
    temp_raw_file = "temp_raw.wav"
    wf = wave.open(temp_raw_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Apply noise reduction
    rate, data = wavfile.read(temp_raw_file)
    data = data.astype(np.float32)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # Save the cleaned audio
    reduced_noise = reduced_noise.astype(np.int16)
    wavfile.write(output_file, rate, reduced_noise)

def record_audio2(output_file="test.wav", duration=2):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100  # Sampling rate

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording... Speak now!")
    frames = []

    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save raw audio to temp file
    temp_raw_file = "temp_raw.wav"
    wf = wave.open(temp_raw_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Read raw WAV and reduce noise
    rate, data = wavfile.read(temp_raw_file)

    # Convert to float32 for processing
    data = data.astype(np.float32)

    # Use first 0.25 seconds as noise profile (adjust as needed)
    noise_clip = data[:int(0.25 * rate)]

    # Apply aggressive noise reduction
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        y_noise=noise_clip,   # explicitly provide noise profile
        prop_decrease=1.0,    # maximum noise reduction
        stationary=True       # assume constant background noise
    )

    # Save cleaned audio
    reduced_noise = np.clip(reduced_noise, -32768, 32767).astype(np.int16)
    wavfile.write(output_file, rate, reduced_noise)

    print(f"Cleaned audio saved to: {output_file}")

    # Clean up temp file
    os.remove(temp_raw_file)

# -----------------------------
# 3. Load spectrogram from file
# -----------------------------
def load_spectrogram(filename):
    y, sr = librosa.load(filename, sr=44100, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram.reshape(1, 1, *spectrogram.shape)

# -----------------------------
# 4. Prediction function
# -----------------------------
def predict_command(model, audio_path, threshold=0.7):
    model.eval()
    x = load_spectrogram(audio_path)
    x_tensor = torch.tensor(x)
    with torch.no_grad():
        output = model(x_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    return predicted_idx
import librosa
import numpy as np

def load_spectrogram(filename):
    y, sr = librosa.load(filename, sr=44100, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram.reshape(1, 1, *spectrogram.shape)
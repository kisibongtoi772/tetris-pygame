import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


# Define the neural network model
class VoiceCommandRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 🔹 Define fc_input_size dynamically
        sample_input = torch.randn(1, 1, 128, 87)  # Simulate a spectrogram input
        with torch.no_grad():
            sample_output = self.forward_feature_extractor(sample_input)
        self.fc_input_size = sample_output.numel()  # Get the number of elements

        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward_feature_extractor(self, x):
        """ Extract features without flattening for size calculation. """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x

    def forward(self, x):
        x = self.forward_feature_extractor(x)
        x = x.view(x.size(0), -1)  # ✅ Dynamically reshape
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Function to load a spectrogram from a file
def load_spectrogram(filename):
    y, sr = librosa.load(filename, sr=44100, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram.reshape(1, *spectrogram.shape)


# Dataset class
class VoiceCommandDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = load_spectrogram(self.files[index])
        y = self.labels[index]
        return x, y


# Define the commands
commands = ["rotation_left", "rotation_right", "move_left", "move_right", "down", "yes", "no", "pause", "speed"]

# Generate the files
files = []
labels = []

for label, command in enumerate(commands):
    for i in range(1, 37):  # 1 to 36 inclusive
        files.append(f"{command}_{i}.wav")
        labels.append(label)  # 0 for first command, 1 for second, etc.


dataset = VoiceCommandDataset(files, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model setup
model = VoiceCommandRecognizer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

MODEL_PATH = "voice_model.pth"

# Check if model already exists
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("Training new model...")

    # Train the model
    for epoch in range(10):  # 10 epochs
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved successfully!")

# Model is now ready to be used without retraining
print("Model loaded and ready to use!")

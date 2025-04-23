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
        self.fc2 = nn.Linear(128, 5)

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


# Define dataset
files = ['rotation_left_1.wav', 'rotation_left_2.wav', 'rotation_left_3.wav', 'rotation_left_4.wav',
         "rotation_left_5.wav", "rotation_left_6.wav", "rotation_left_7.wav", "rotation_left_8.wav",
         "rotation_left_9.wav", "rotation_left_10.wav", "rotation_left_11.wav", "rotation_left_12.wav",
         "rotation_left_13.wav", "rotation_left_14.wav", "rotation_left_15.wav", "rotation_left_16.wav",
         "rotation_left_17.wav", "rotation_left_18.wav", "rotation_left_19.wav", "rotation_left_20.wav",
         "rotation_left_21.wav", "rotation_left_22.wav", "rotation_left_23.wav","rotation_left_24.wav",
         "rotation_left_25.wav", "rotation_left_26.wav", "rotation_left_27.wav","rotation_left_28.wav",
         "rotation_left_29.wav", "rotation_left_30.wav", "rotation_left_31.wav", "rotation_left_32.wav",
         "rotation_left_33.wav", "rotation_left_34.wav", "rotation_left_35.wav","rotation_left_36.wav",

         'rotation_right_1.wav', 'rotation_right_2.wav', 'rotation_right_3.wav', 'rotation_right_4.wav',
         "rotation_right_5.wav", "rotation_right_6.wav", "rotation_right_7.wav","rotation_right_8.wav",
         "rotation_right_9.wav", "rotation_right_10.wav", "rotation_right_11.wav","rotation_right_12.wav",
         "rotation_right_13.wav", "rotation_right_14.wav","rotation_right_15.wav", "rotation_right_16.wav",
         "rotation_right_17.wav", "rotation_right_18.wav", "rotation_right_19.wav", "rotation_right_20.wav",
         "rotation_right_21.wav","rotation_right_22.wav", "rotation_right_23.wav", "rotation_right_24.wav",
         "rotation_right_25.wav","rotation_right_26.wav","rotation_right_27.wav","rotation_right_28.wav",
         "rotation_right_29.wav", "rotation_right_30.wav", "rotation_right_31.wav", "rotation_right_32.wav",
         "rotation_right_33.wav", "rotation_right_34.wav", "rotation_right_35.wav", "rotation_right_36.wav",

          "move_left_1.wav", "move_left_2.wav", "move_left_3.wav", "move_left_4.wav",
         "move_left_5.wav", "move_left_6.wav", "move_left_7.wav", "move_left_8.wav",
         "move_left_9.wav", "move_left_10.wav", "move_left_11.wav", "move_left_12.wav",
         "move_left_13.wav", "move_left_14.wav", "move_left_15.wav", "move_left_16.wav",
         "move_left_17.wav", "move_left_18.wav", "move_left_19.wav", "move_left_20.wav",
         "move_left_21.wav","move_left_22.wav", "move_left_23.wav", "move_left_24.wav",
         "move_left_25.wav", "move_left_26.wav", "move_left_27.wav", "move_left_28.wav",
         "move_left_29.wav", "move_left_30.wav", "move_left_31.wav", "move_left_32.wav",
         "move_left_33.wav", "move_left_34.wav", "move_left_35.wav", "move_left_36.wav",

          "move_right_1.wav", "move_right_2.wav", "move_right_3.wav","move_right_4.wav",
         "move_right_5.wav", "move_right_6.wav", "move_right_7.wav", "move_right_8.wav",
         "move_right_9.wav", "move_right_10.wav", "move_right_11.wav", "move_right_12.wav",
         "move_right_13.wav", "move_right_14.wav", "move_right_15.wav", "move_right_16.wav",
         "move_right_17.wav", "move_right_18.wav", "move_right_19.wav", "move_right_20.wav",
         "move_right_21.wav", "move_right_22.wav", "move_right_23.wav", "move_right_24.wav",
         "move_right_25.wav", "move_right_26.wav", "move_right_27.wav", "move_right_28.wav",
         "move_right_29.wav","move_right_30.wav", "move_right_31.wav", "move_right_32.wav",
         "move_right_33.wav","move_right_34.wav", "move_right_35.wav", "move_right_36.wav",

          "down_1.wav", "down_2.wav", "down_3.wav", "down_4.wav","down_5.wav", "down_6.wav", "down_7.wav", "down_8.wav",
         "down_9.wav", "down_10.wav", "down_11.wav", "down_12.wav","down_13.wav", "down_14.wav", "down_15.wav", "down_16.wav",
         "down_17.wav", "down_18.wav", "down_19.wav", "down_20.wav", "down_21.wav", "down_22.wav", "down_23.wav", "down_24.wav",
         "down_25.wav", "down_26.wav", "down_27.wav","down_28.wav", "down_29.wav", "down_30.wav", "down_31.wav", "down_32.wav",
         "down_33.wav", "down_34.wav", "down_35.wav", "down_36.wav"]
labels = [0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,
          1,1,1,1,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,
          2,2,2,2,2,2,2,2,2,2,2,2 ,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,
          3,3,3,3,3,3,3,3,3,3,3,3 ,3,3,3,3,3,3,3,3,3,3,3,3, 3,3,3,3,3,3,3,3,3,3,3,3,
          4,4,4,4,4,4,4,4,4,4,4,4 ,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4]  # 0 = rotation left, 1 = rotation right, 2 = move left, 3 = move_right, 4 = down

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

from torch.utils.data import Dataset, DataLoader
from VoiceCommandRec import VoiceCommandRecognizer
from LoadSpect import load_spectrogram
import torch
import torch.nn as nn

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

files = ['left','right','yes','no','pause','speed']
labels = [0, 1, 2, 3]
dataset = VoiceCommandDataset(files, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = VoiceCommandRecognizer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
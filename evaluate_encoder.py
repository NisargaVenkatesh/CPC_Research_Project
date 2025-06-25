import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from cpc.encoder import Encoder
from dataset import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder
encoder = Encoder()
encoder.load_state_dict(torch.load("encoder_best.pth", map_location=device))
encoder.eval().to(device)

# Load dataset
dataset = AudioDataset(segment_len=16000, max_files=500)
dataloader = DataLoader(dataset, batch_size=1)

X = []
y = []

with torch.no_grad():
    for waveform, label in dataloader:
        waveform = waveform.to(device)
        z = encoder(waveform)  # [1, T, D]
        z_pooled = z.mean(dim=1).squeeze().cpu().numpy()  # [D]
        X.append(z_pooled)
        y.append(label.item())

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Predict & evaluate
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)

print(f"Classification accuracy (speaker ID): {acc:.4f}")
